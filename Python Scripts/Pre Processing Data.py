#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared Preprocessing and Data Engineering Pipeline
for AI-Driven Tactical Preparation in Professional Tennis

This module consolidates the shared preprocessing used across:
- Serve Strategy Optimisation
- Predictability Index
- Break-Point Forecasting
- Rally Advantage
- Momentum Modelling
"""

#%% Importing Packages 

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


#%% Parameter Settings

@dataclass
class Config:
    start_year: int = 2005
    end_year: int = 2025

    rolling_window: int = 20
    min_history_matches: int = 8

    n_style_clusters: int = 4
    drop_missing_bp_stats: bool = True


CFG = Config()

ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"
SLAM_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"

SERVE_BASICS_URL = MCP_BASE + "charting-m-stats-ServeBasics.csv"
KEYPOINTS_SERVE_URL = MCP_BASE + "charting-m-stats-KeyPointsServe.csv"
KEYPOINTS_RETURN_URL = MCP_BASE + "charting-m-stats-KeyPointsReturn.csv"
MCP_MATCHES_URL = MCP_BASE + "charting-m-matches.csv"

MCP_POINTS_FILES = [
    MCP_BASE + "charting-m-points-to-2009.csv",
    MCP_BASE + "charting-m-points-2010s.csv",
    MCP_BASE + "charting-m-points-2020s.csv",
]


#%% 1. General Funtions

def parse_1st_serve_direction(code: str) -> str:
    """
    MCP first-serve code mapping:
      4 -> Wide
      5 -> Body
      6 -> T
    """
    if not isinstance(code, str) or code == "" or pd.isna(code):
        return "Unknown"

    c0 = code[0]
    if c0 == "4":
        return "Wide"
    if c0 == "5":
        return "Body"
    if c0 == "6":
        return "T"
    return "Unknown"


def speed_to_bucket(speed_kmh: Optional[float]) -> str:
    """
    Serve speed bucket mapping:
    - Fast   : >= 195
    - Medium : 175 to 194
    - Slow   : < 175
    - Unknown: missing or zero
    """
    if pd.isna(speed_kmh) or speed_kmh == 0:
        return "Unknown"

    speed_kmh = float(speed_kmh)
    if speed_kmh >= 195:
        return "Fast"
    elif speed_kmh >= 175:
        return "Medium"
    return "Slow"


def score_to_bucket(score_str: str) -> str:
    """
    Tactical score-state mapping from server perspective.
    """
    if pd.isna(score_str):
        return "Unknown"

    s = str(score_str).strip().upper()

    if s in ["DEUCE", "40-40"]:
        return "Deuce"

    server_gp = {"40-0", "40-15", "40-30", "AD-40"}
    returner_bp = {"0-40", "15-40", "30-40", "40-AD"}

    if s in server_gp:
        return "GamePointServer"
    if s in returner_bp:
        return "BreakPoint"
    if s in {"0-0", "15-15", "30-30"}:
        return "NeutralEven"
    if s in {"15-0", "0-15", "30-15", "15-30"}:
        return "NeutralOther"

    return "Other"


def build_server_score_string(p1_score: str, p2_score: str, point_server: int) -> str:
    """
    Convert raw score columns into server-perspective score string.
    """
    s1 = str(p1_score).strip()
    s2 = str(p2_score).strip()
    if point_server == 1:
        return f"{s1}-{s2}"
    return f"{s2}-{s1}"


#%% Player style clustering (MCP)

def build_player_style_clusters(cfg: Config = CFG) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Build player style clusters from MCP serve and pressure statistics.
    Returns:
        player_features : DataFrame containing style labels
        player_style_map: dict {player_name -> style_label}
    """
    serve_basics = pd.read_csv(SERVE_BASICS_URL)
    keypoints_serve = pd.read_csv(KEYPOINTS_SERVE_URL)
    keypoints_return = pd.read_csv(KEYPOINTS_RETURN_URL)

    # Serve-based features
    sb = serve_basics.copy()
    sb["pts"] = sb["pts"].replace(0, np.nan)

    sb["unreturned_pct"] = sb["unret"] / sb["pts"]
    sb["ace_rate"] = sb["aces"] / sb["pts"]
    sb["short_point_pct"] = sb["pts_won_lte_3_shots"] / sb["pts"]

    serve_summary = (
        sb.groupby("player")[["unreturned_pct", "ace_rate", "short_point_pct"]]
        .mean()
        .reset_index()
    )

    # Pressure features
    ks = keypoints_serve.copy()
    kr = keypoints_return.copy()

    ks["pts"] = ks["pts"].replace(0, np.nan)
    kr["pts"] = kr["pts"].replace(0, np.nan)

    ks["pressure_serve_win_pct"] = ks["pts_won"] / ks["pts"]
    kr["pressure_return_win_pct"] = kr["pts_won"] / kr["pts"]

    pressure_summary = (
        ks.groupby("player")[["pressure_serve_win_pct"]]
        .mean()
        .merge(
            kr.groupby("player")[["pressure_return_win_pct"]].mean(),
            on="player",
            how="outer",
        )
    )

    player_features = (
        serve_summary.merge(pressure_summary, on="player", how="outer")
        .dropna()
        .reset_index(drop=True)
    )

    # Clustering
    X = player_features.drop(columns=["player"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=cfg.n_style_clusters, random_state=42, n_init=10)
    player_features["style_cluster"] = kmeans.fit_predict(X_scaled)

    cluster_labels = {
        0: "Big-Serve Aggressor",
        1: "Baseline Grinder",
        2: "All-Court Balanced",
        3: "Counter-Puncher",
    }
    player_features["style_label"] = player_features["style_cluster"].map(cluster_labels)

    player_style_map = dict(zip(player_features["player"], player_features["style_label"]))
    return player_features, player_style_map


#%% Loading MCP data

def load_mcp_matches() -> pd.DataFrame:
    """
    Load and standardise MCP match metadata.
    """
    mcp_matches = pd.read_csv(MCP_MATCHES_URL)
    return mcp_matches.rename(columns={
        "match_id": "match_id",
        "Player 1": "p1_name",
        "Player 2": "p2_name",
        "Surface": "surface",
    })[["match_id", "p1_name", "p2_name", "surface"]]


def load_mcp_points() -> pd.DataFrame:
    """
    Load and concatenate MCP point files.
    """
    dfs = []
    for url in MCP_POINTS_FILES:
        df = pd.read_csv(url)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_mcp_points_dir(player_style_map: Dict[str, str]) -> pd.DataFrame:
    """
    Build MCP point dataset with:
    - server/returner labels
    - server-centric point outcome
    - first-serve direction
    - server/opponent style labels
    """
    mcp_points_all = load_mcp_points()
    mcp_matches_std = load_mcp_matches()

    df = mcp_points_all.merge(mcp_matches_std, on="match_id", how="left")
    df = df.rename(columns={"PtWinner": "point_winner_raw"})

    # Server / returner mapping
    df["server_name"] = np.where(df["Svr"] == 1, df["p1_name"], df["p2_name"])
    df["returner_name"] = np.where(df["Svr"] == 1, df["p2_name"], df["p1_name"])

    # Server-centric point outcome
    def server_point_won(row) -> float:
        if row["point_winner_raw"] == 1 and row["server_name"] == row["p1_name"]:
            return 1
        if row["point_winner_raw"] == 2 and row["server_name"] == row["p2_name"]:
            return 1
        if row["point_winner_raw"] in [1, 2]:
            return 0
        return np.nan

    df["point_won"] = df.apply(server_point_won, axis=1)
    df = df.dropna(subset=["point_won"]).copy()
    df["point_won"] = df["point_won"].astype(int)

    # First-serve direction
    df["serve_direction"] = df["1st"].apply(parse_1st_serve_direction)
    df = df[df["serve_direction"].isin(["Wide", "Body", "T"])].copy()

    # Style labels
    df["server_type"] = df["server_name"].map(player_style_map).fillna("Unknown")
    df["opponent_type"] = df["returner_name"].map(player_style_map).fillna("Unknown")

    df["surface"] = df["surface"].astype(str)
    return df


#%% 4. Grand Slam point-by-point data

def load_slam_data(years: range = range(2011, 2018),
                   slams: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Grand Slam matches and points.
    """
    if slams is None:
        slams = ["ausopen", "frenchopen", "wimbledon", "usopen"]

    matches_list = []
    points_list = []

    for year in years:
        for slam in slams:
            matches_file = f"{SLAM_BASE}{year}-{slam}-matches.csv"
            points_file = f"{SLAM_BASE}{year}-{slam}-points.csv"

            try:
                m = pd.read_csv(matches_file)
                m["year"] = year
                m["slam"] = slam
                matches_list.append(m)
            except Exception:
                pass

            try:
                p = pd.read_csv(points_file)
                p["year"] = year
                p["slam"] = slam
                points_list.append(p)
            except Exception:
                pass

    if not matches_list or not points_list:
        raise RuntimeError("Unable to load Slam match or point data.")

    slam_matches = pd.concat(matches_list, ignore_index=True)
    slam_points = pd.concat(points_list, ignore_index=True)

    slam_matches["surface"] = slam_matches["slam"].map({
        "ausopen": "Hard",
        "frenchopen": "Clay",
        "wimbledon": "Grass",
        "usopen": "Hard",
    })

    return slam_matches, slam_points


def preprocess_slam_points(player_style_map: Dict[str, str],
                           years: range = range(2011, 2018)) -> pd.DataFrame:
    """
    Build Slam point-level modelling dataset with:
    - server/returner labels
    - server-centric point outcome
    - speed buckets
    - serve number
    - score buckets
    - style labels
    """
    slam_matches, slam_points = load_slam_data(years=years)

    required_point_cols = ["match_id", "PointServer", "PointWinner", "P1Score", "P2Score"]
    df_points = slam_points.copy()

    for col in required_point_cols:
        if col in df_points.columns:
            df_points = df_points[df_points[col].notna()]
        else:
            raise ValueError(f"Missing required Slam point column: {col}")

    match_cols = ["match_id", "player1", "player2", "surface", "year", "slam"]
    df_matches = slam_matches[match_cols].copy()

    df = df_points.merge(df_matches, on="match_id", how="left")
    df = df.rename(columns={"player1": "p1_name", "player2": "p2_name"})

    df["PointServer"] = pd.to_numeric(df["PointServer"], errors="coerce")
    df["PointWinner"] = pd.to_numeric(df["PointWinner"], errors="coerce")

    # Server / returner mapping
    df["server_name"] = np.where(df["PointServer"] == 1, df["p1_name"], df["p2_name"])
    df["returner_name"] = np.where(df["PointServer"] == 1, df["p2_name"], df["p1_name"])

    # Server-centric point outcome
    def slam_server_point_won(row) -> float:
        if row["PointWinner"] == row["PointServer"]:
            return 1
        if row["PointWinner"] in [1, 2]:
            return 0
        return np.nan

    df["point_won"] = df.apply(slam_server_point_won, axis=1)
    df = df.dropna(subset=["point_won"]).copy()
    df["point_won"] = df["point_won"].astype(int)

    # Serve speed
    if "Speed_KMH" in df.columns:
        df["serve_speed"] = pd.to_numeric(df["Speed_KMH"], errors="coerce")
    else:
        df["serve_speed"] = np.nan
    df["speed_bucket"] = df["serve_speed"].apply(speed_to_bucket)

    # Serve number
    if "ServeNumber" in df.columns:
        df["serve_number"] = pd.to_numeric(df["ServeNumber"], errors="coerce").fillna(1).astype(int)
    else:
        df["serve_number"] = 1

    # Score string + score bucket
    df["score_str"] = df.apply(
        lambda row: build_server_score_string(row["P1Score"], row["P2Score"], int(row["PointServer"])),
        axis=1
    )
    df["score_bucket"] = df["score_str"].apply(score_to_bucket)

    # Style labels
    df["server_type"] = df["server_name"].map(player_style_map).fillna("Unknown")
    df["opponent_type"] = df["returner_name"].map(player_style_map).fillna("Unknown")

    df["surface"] = df["surface"].astype(str)
    return df


#%% ATP match data for BP models

def load_atp_matches(cfg: Config = CFG) -> pd.DataFrame:
    """
    Load ATP match data across year range and standardise.
    """
    dfs = []
    for year in range(cfg.start_year, cfg.end_year + 1):
        url = f"{ATP_BASE}atp_matches_{year}.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            df["source_url"] = url
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("No ATP match files could be loaded.")

    matches = pd.concat(dfs, ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    matches = matches.dropna(subset=["tourney_date"]).copy()

    if "tourney_id" in matches.columns and "match_num" in matches.columns:
        matches["match_uid"] = matches["tourney_id"].astype(str) + "_" + matches["match_num"].astype(str)
    else:
        matches["match_uid"] = "synthetic_" + matches.index.astype(str)

    bp_cols = ["w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
    for col in bp_cols:
        if col not in matches.columns:
            raise ValueError(f"Missing ATP column: {col}")
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

    if cfg.drop_missing_bp_stats:
        matches = matches.dropna(subset=bp_cols).copy()
    else:
        matches[bp_cols] = matches[bp_cols].fillna(0)

    for col in bp_cols:
        matches = matches[matches[col] >= 0]

    matches = matches.sort_values(["tourney_date", "match_uid"]).reset_index(drop=True)
    return matches


def derive_player_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ATP match data into player-centric rows:
    one row per player per match.
    """
    winners = pd.DataFrame({
        "match_uid": matches["match_uid"],
        "date": matches["tourney_date"],
        "surface": matches.get("surface", np.nan),
        "tourney_level": matches.get("tourney_level", np.nan),
        "round": matches.get("round", np.nan),
        "best_of": matches.get("best_of", np.nan),

        "player_name": matches["winner_name"],
        "opponent_name": matches["loser_name"],

        "player_rank": matches.get("winner_rank", np.nan),
        "opponent_rank": matches.get("loser_rank", np.nan),
        "player_rank_points": matches.get("winner_rank_points", np.nan),
        "opponent_rank_points": matches.get("loser_rank_points", np.nan),

        "player_hand": matches.get("winner_hand", np.nan),
        "opponent_hand": matches.get("loser_hand", np.nan),

        "bp_created": matches["l_bpFaced"],
        "bp_converted": (matches["l_bpFaced"] - matches["l_bpSaved"]).clip(lower=0),
        "bp_faced": matches["w_bpFaced"],
        "bp_saved": matches["w_bpSaved"],

        "won_match": 1,
    })

    losers = pd.DataFrame({
        "match_uid": matches["match_uid"],
        "date": matches["tourney_date"],
        "surface": matches.get("surface", np.nan),
        "tourney_level": matches.get("tourney_level", np.nan),
        "round": matches.get("round", np.nan),
        "best_of": matches.get("best_of", np.nan),

        "player_name": matches["loser_name"],
        "opponent_name": matches["winner_name"],

        "player_rank": matches.get("loser_rank", np.nan),
        "opponent_rank": matches.get("winner_rank", np.nan),
        "player_rank_points": matches.get("loser_rank_points", np.nan),
        "opponent_rank_points": matches.get("winner_rank_points", np.nan),

        "player_hand": matches.get("loser_hand", np.nan),
        "opponent_hand": matches.get("winner_hand", np.nan),

        "bp_created": matches["w_bpFaced"],
        "bp_converted": (matches["w_bpFaced"] - matches["w_bpSaved"]).clip(lower=0),
        "bp_faced": matches["l_bpFaced"],
        "bp_saved": matches["l_bpSaved"],

        "won_match": 0,
    })

    pm = pd.concat([winners, losers], ignore_index=True)
    pm = pm.sort_values(["date", "match_uid", "player_name"]).reset_index(drop=True)

    pm["rank_diff"] = pm["player_rank"] - pm["opponent_rank"]
    pm["rp_diff"] = pm["player_rank_points"] - pm["opponent_rank_points"]

    pm["bp_conv_rate"] = np.where(pm["bp_created"] > 0, pm["bp_converted"] / pm["bp_created"], np.nan)
    pm["bp_save_rate"] = np.where(pm["bp_faced"] > 0, pm["bp_saved"] / pm["bp_faced"], np.nan)

    return pm


#%% 6. Rolling features

def add_rolling_features(pm: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """
    Create rolling historical features with shift(1) to preserve temporal validity.
    Then merge opponent rolling features onto each row.
    """
    pm = pm.copy()
    w = cfg.rolling_window
    min_p = max(3, cfg.min_history_matches)

    def _roll(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["date", "match_uid"]).copy()

        for col in ["bp_created", "bp_converted", "bp_faced", "bp_saved"]:
            group[f"r_{col}_mean"] = group[col].shift(1).rolling(w, min_periods=min_p).mean()
            group[f"r_{col}_sum"] = group[col].shift(1).rolling(w, min_periods=min_p).sum()

        group["r_bp_conv_rate"] = np.where(
            group["r_bp_created_sum"] > 0,
            group["r_bp_converted_sum"] / group["r_bp_created_sum"],
            np.nan
        )

        group["r_bp_save_rate"] = np.where(
            group["r_bp_faced_sum"] > 0,
            group["r_bp_saved_sum"] / group["r_bp_faced_sum"],
            np.nan
        )

        group["r_win_rate"] = group["won_match"].shift(1).rolling(w, min_periods=min_p).mean()
        group["r_matches_played"] = group["won_match"].shift(1).rolling(w, min_periods=min_p).count()

        return group

    pm = pm.groupby("player_name", group_keys=False).apply(_roll)

    # Require sufficient prior history
    pm = pm[pm["r_matches_played"].notna()].copy()
    pm = pm[pm["r_matches_played"] >= cfg.min_history_matches].copy()

    # Opponent rolling feature merge
    opp_cols = [
        "match_uid", "player_name",
        "r_bp_created_mean", "r_bp_converted_mean",
        "r_bp_faced_mean", "r_bp_saved_mean",
        "r_bp_conv_rate", "r_bp_save_rate",
        "r_win_rate", "r_matches_played",
        "player_rank", "player_rank_points"
    ]

    opp = pm[opp_cols].copy()
    opp = opp.rename(columns={"player_name": "opponent_name"})
    opp = opp.rename(columns={
        col: f"opp_{col}" for col in opp.columns
        if col not in ["match_uid", "opponent_name"]
    })

    pm = pm.merge(
        opp,
        on=["match_uid", "opponent_name"],
        how="left",
        validate="m:1"
    )

    return pm


def add_style_features(df: pd.DataFrame, player_style_map: Dict[str, str]) -> pd.DataFrame:
    """
    Attach player/opponent style labels.
    """
    df = df.copy()

    if "player_name" in df.columns:
        df["player_type"] = df["player_name"].map(player_style_map).fillna("Unknown")
    if "opponent_name" in df.columns:
        df["opponent_type"] = df["opponent_name"].map(player_style_map).fillna("Unknown")

    return df


#% 7. Pipeline

def build_shared_datasets(cfg: Config = CFG) -> Dict[str, pd.DataFrame]:
    """
    Run the full shared preprocessing pipeline and return reusable datasets.
    """
    print("Building player style clusters...")
    player_features, player_style_map = build_player_style_clusters(cfg)

    print("Building MCP direction dataset...")
    mcp_points_dir = build_mcp_points_dir(player_style_map)

    print("Building Slam point dataset...")
    slam_points_model = preprocess_slam_points(player_style_map)

    print("Building ATP player-match dataset...")
    atp_matches = load_atp_matches(cfg)
    player_match = derive_player_match_rows(atp_matches)
    player_match = add_rolling_features(player_match, cfg)
    player_match = add_style_features(player_match, player_style_map)

    return {
        "player_features": player_features,
        "mcp_points_dir": mcp_points_dir,
        "slam_points_model": slam_points_model,
        "player_match": player_match,
    }


#%% 8. main run

if __name__ == "__main__":
    datasets = build_shared_datasets(CFG)

    print("\nDatasets created:")
    for name, df in datasets.items():
        print(f"- {name}: {df.shape}")

    # saving files
    datasets["player_features"].to_csv("player_features.csv", index=False)
    datasets["mcp_points_dir"].to_csv("mcp_points_dir.csv", index=False)
    datasets["slam_points_model"].to_csv("slam_points_model.csv", index=False)
    datasets["player_match"].to_csv("player_match.csv", index=False)

    print("\nSaved shared preprocessing outputs to CSV.")