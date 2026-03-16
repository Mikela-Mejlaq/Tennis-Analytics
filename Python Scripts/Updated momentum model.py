#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 18:18:59 2026

@author: mikelamejlaq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Momentum AI Tool

Function
- Downloads Slam point-by-point + matches files (Jeff Sackmann tennis_slam_pointbypoint)
- Robustly merges points with matches (handles match_id dtype issues + slam/year duplicates)
- Trains a global expected-point-win model (logistic regression)
- Builds a decayed Momentum Index per point for a chosen focus player:
      m_t = decay*m_{t-1} + scale * leverage * (actual - expected)
- Outputs TWO CSVs for Power BI:
      1) momentum_points_long.csv  (fact table; 1 row per point per focus player)
      2) momentum_match_summary.csv (1 row per match per focus player)
- Plots a match momentum time series in Spyder (state graph)

Important
- The most common reason “no data matched filters” happens is because you picked years
  that don’t contain the player. This script auto-diagnoses and auto-broadens safely.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve


#%% Setting paramters

SLAM_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"
CACHE_DIR = "./slam_cache"
EXPORT_DIR = "./powerbi_exports"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

SLAM_TO_SURFACE = {
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard",
}

# Momentum parameters (can be tuned if necessary)
MOMENTUM_DECAY = 0.92
MOMENTUM_SCALE = 12.0
CLIP_MOMENTUM = 20.0

LEVERAGE_BREAKPOINT = 1.60
LEVERAGE_GAMEPOINT = 1.30
LEVERAGE_DEUCE = 1.15
LEVERAGE_TB = 1.25
LEVERAGE_DEFAULT = 1.00

# Phase thresholds
PHASE_POS_DOMINANT = 6.0
PHASE_POS_EDGE = 2.0
PHASE_NEG_EDGE = -2.0
PHASE_NEG_DOMINANT = -6.0


#%% Helper Functions
def _cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)

def read_csv_cached(url: str, filename: str, low_memory: bool = True) -> pd.DataFrame:
    """
    Read from local cache if exists; else download and cache.
    """
    path = _cache_path(filename)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=low_memory)
    df = pd.read_csv(url, low_memory=low_memory)
    df.to_csv(path, index=False)
    return df


#%% Loading data

def load_slam_data(
    years: List[int],
    slams: List[str],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (matches_all, points_all, loaded_pairs)
    loaded_pairs are slam-year pairs where BOTH matches and points loaded.
    """
    matches_list = []
    points_list = []
    loaded_pairs = []

    for year in years:
        for slam in slams:
            matches_fname = f"{year}-{slam}-matches.csv"
            points_fname = f"{year}-{slam}-points.csv"

            ok_m, ok_p = False, False

            try:
                df_m = read_csv_cached(SLAM_BASE + matches_fname, matches_fname)
                df_m["year"] = year
                df_m["slam"] = slam
                df_m["surface"] = SLAM_TO_SURFACE.get(slam, "Unknown")
                matches_list.append(df_m)
                ok_m = True
                if verbose:
                    print(f"Loaded matches: {matches_fname} shape={df_m.shape}")
            except Exception as e:
                if verbose:
                    print(f"Skipping matches {matches_fname}: {e}")

            try:
                df_p = read_csv_cached(SLAM_BASE + points_fname, points_fname)
                df_p["year"] = year
                df_p["slam"] = slam
                points_list.append(df_p)
                ok_p = True
                if verbose:
                    print(f"Loaded points : {points_fname} shape={df_p.shape}")
            except Exception as e:
                if verbose:
                    print(f"Skipping points {points_fname}: {e}")

            if ok_m and ok_p:
                loaded_pairs.append(f"{year}-{slam}")

    if not matches_list or not points_list:
        raise RuntimeError(
            "No data loaded at all. Likely your year range has no files in the repo."
        )

    matches_all = pd.concat(matches_list, ignore_index=True)
    points_all = pd.concat(points_list, ignore_index=True)

    if verbose:
        print("\nLoaded complete slam-year pairs:", loaded_pairs[:30], ("..." if len(loaded_pairs) > 30 else ""))
        print("Total matches rows:", len(matches_all))
        print("Total points rows :", len(points_all))

    return matches_all, points_all, loaded_pairs


#%% features

def speed_to_bucket_kmh(v) -> str:
    if pd.isna(v) or v == 0:
        return "Unknown"
    try:
        v = float(v)
    except Exception:
        return "Unknown"
    if v >= 195:
        return "Fast"
    if v >= 175:
        return "Medium"
    return "Slow"


def score_to_bucket(score_str: str) -> str:
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


def build_server_score_str(row: pd.Series) -> str:
    s1 = str(row.get("P1Score", "Unknown")).strip()
    s2 = str(row.get("P2Score", "Unknown")).strip()
    ps = row.get("PointServer", np.nan)

    if ps == 1:
        return f"{s1}-{s2}"
    if ps == 2:
        return f"{s2}-{s1}"
    return "Unknown"


def compute_point_won_server(row: pd.Series) -> float:
    ps = row.get("PointServer", np.nan)
    pw = row.get("PointWinner", np.nan)
    if pd.isna(ps) or pd.isna(pw):
        return np.nan
    if pw in [1, 2] and ps in [1, 2]:
        return 1.0 if pw == ps else 0.0
    return np.nan


#%% merging data

def merge_points_matches(points_all: pd.DataFrame, matches_all: pd.DataFrame) -> pd.DataFrame:
    """
    Robust merge:
    - Forces match_id to string in both tables (prevents silent merge mismatch)
    - Uses matches columns player1/player2 if available; else tries winner_name/loser_name
    - Drops year/slam/surface from points before merge to avoid slam_x/slam_y
    """
    if "match_id" not in points_all.columns or "match_id" not in matches_all.columns:
        raise ValueError("Missing match_id in points or matches.")

    points_all = points_all.copy()
    matches_all = matches_all.copy()

    points_all["match_id"] = points_all["match_id"].astype(str)
    matches_all["match_id"] = matches_all["match_id"].astype(str)

    # Resolve matches schema
    if "player1" in matches_all.columns and "player2" in matches_all.columns:
        p1_col, p2_col = "player1", "player2"
    elif "winner_name" in matches_all.columns and "loser_name" in matches_all.columns:
        matches_all = matches_all.rename(columns={"winner_name": "player1", "loser_name": "player2"})
        p1_col, p2_col = "player1", "player2"
    else:
        raise ValueError(
            "Unknown matches schema. Expected player1/player2 or winner_name/loser_name. "
            f"Matches columns sample: {list(matches_all.columns)[:40]}"
        )

    must_cols = ["match_id", p1_col, p2_col, "surface", "year", "slam"]
    for c in must_cols:
        if c not in matches_all.columns:
            raise ValueError(f"matches_all missing required column: {c}")

    matches_min = matches_all[must_cols].copy()

    # Drop year/slam/surface from points to avoid duplicates on merge
    drop_cols = [c for c in ["year", "slam", "surface"] if c in points_all.columns]
    points_clean = points_all.drop(columns=drop_cols)

    df = points_clean.merge(matches_min, on="match_id", how="left")

    # If merge failed, player1/player2 will be mostly NaN
    missing_players = df[p1_col].isna().mean()
    if missing_players > 0.30:
        raise RuntimeError(
            "Merge looks wrong: too many missing player names after merge. "
            "This is almost always match_id mismatch across points/matches files."
        )

    df = df.rename(columns={p1_col: "p1_name", p2_col: "p2_name"})

    # Numeric server/winner
    df["PointServer"] = pd.to_numeric(df.get("PointServer", np.nan), errors="coerce")
    df["PointWinner"] = pd.to_numeric(df.get("PointWinner", np.nan), errors="coerce")
    df = df[df["PointServer"].notna() & df["PointWinner"].notna()].copy()
    df["PointServer"] = df["PointServer"].astype(int)
    df["PointWinner"] = df["PointWinner"].astype(int)

    df["server_name"] = np.where(df["PointServer"] == 1, df["p1_name"], df["p2_name"])
    df["returner_name"] = np.where(df["PointServer"] == 1, df["p2_name"], df["p1_name"])

    df["point_won"] = df.apply(compute_point_won_server, axis=1)
    df = df.dropna(subset=["point_won"]).copy()
    df["point_won"] = df["point_won"].astype(int)

    # Serve speed
    df["serve_speed_kmh"] = pd.to_numeric(df.get("Speed_KMH", np.nan), errors="coerce")
    df["speed_bucket"] = df["serve_speed_kmh"].apply(speed_to_bucket_kmh)

    # Serve number
    if "ServeNumber" in df.columns:
        df["serve_number"] = (
            pd.to_numeric(df["ServeNumber"], errors="coerce")
            .fillna(1)
            .astype(int)
        )
    else:
        df["serve_number"] = 1
        
    # Score
    if "P1Score" not in df.columns:
        df["P1Score"] = "Unknown"
    if "P2Score" not in df.columns:
        df["P2Score"] = "Unknown"

    df["score_str_server"] = df.apply(build_server_score_str, axis=1)
    df["score_bucket"] = df["score_str_server"].apply(score_to_bucket)

    # Tiebreak (best effort)
    if "TB" in df.columns:
        df["is_tiebreak"] = (
            pd.to_numeric(df["TB"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df["is_tiebreak"] = 0

    # Point index per match (for plotting)
    df = df.sort_values(["match_id"]).copy()
    df["point_index"] = df.groupby("match_id").cumcount() + 1

    return df


#%% expected point model

def train_expected_point_model(df_points: pd.DataFrame) -> Pipeline:
    """
    Logistic model:
      P(server wins point | surface, score_bucket, speed_bucket, serve_number)
    """
    feature_cols = ["surface", "score_bucket", "speed_bucket", "serve_number"]
    X = df_points[feature_cols].copy()
    y = df_points["point_won"].copy()

    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                      ["surface", "score_bucket", "speed_bucket"])],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=500))
    ])

    if len(df_points) < 2000:
        model.fit(X, y)
        print("Expected-point model trained on full data (small dataset).")
        return model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Expected model accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Expected model AUC     : {roc_auc_score(y_test, y_proba):.3f}")
    return model


#%% momentum engine

def leverage_weight(score_bucket: str, is_tiebreak: int) -> float:
    if is_tiebreak == 1:
        return LEVERAGE_TB
    if score_bucket == "BreakPoint":
        return LEVERAGE_BREAKPOINT
    if score_bucket == "GamePointServer":
        return LEVERAGE_GAMEPOINT
    if score_bucket == "Deuce":
        return LEVERAGE_DEUCE
    return LEVERAGE_DEFAULT


def momentum_phase_label(m: float) -> str:
    if m >= PHASE_POS_DOMINANT:
        return "Dominant (Positive)"
    if m >= PHASE_POS_EDGE:
        return "Edge (Positive)"
    if m <= PHASE_NEG_DOMINANT:
        return "Dominant (Negative)"
    if m <= PHASE_NEG_EDGE:
        return "Edge (Negative)"
    return "Neutral"


def mental_prep_cue(phase: str, just_lost_big: bool, just_won_big: bool) -> str:
    if just_lost_big:
        return "RESET: slow exhale → long target → commit to first 2 shots."
    if just_won_big:
        return "STAY: same routine, same tempo. Don’t rush the next point."

    if phase == "Dominant (Positive)":
        return "PRESS: repeat best pattern, keep margins, protect routine."
    if phase == "Edge (Positive)":
        return "BUILD: high 1st-serve %, heavy cross, avoid low-% lines."
    if phase == "Dominant (Negative)":
        return "STOP BLEED: simplify (body serve / deep return), one clear intention."
    if phase == "Edge (Negative)":
        return "STABILIZE: bigger targets, reset between points, play to neutral."
    return "NEUTRAL: routine + clarity. Pick one pattern and execute fully."


def compute_momentum_for_match(df_match: pd.DataFrame, expected_model: Pipeline, focus_player: str) -> pd.DataFrame:
    """
    Compute momentum time series for ONE match from focus player's perspective.
    """
    df = df_match.copy()

    # Predict expected probability (server wins point)
    feat = df[["surface", "score_bucket", "speed_bucket", "serve_number"]].copy()
    df["p_server_win_exp"] = expected_model.predict_proba(feat)[:, 1]

    # Focus perspective
    df["focus_is_server"] = (df["server_name"].astype(str).str.lower() == str(focus_player).strip().lower()).astype(int)

    df["focus_actual_win"] = np.where(df["focus_is_server"] == 1, df["point_won"], 1 - df["point_won"])
    df["focus_exp_win"] = np.where(df["focus_is_server"] == 1, df["p_server_win_exp"], 1 - df["p_server_win_exp"])

    # Leverage + pred error
    df["leverage"] = [
        leverage_weight(sb, tb) for sb, tb in zip(df["score_bucket"], df["is_tiebreak"])
    ]
    df["pred_error"] = (df["focus_actual_win"] - df["focus_exp_win"]).astype(float)
    df["swing_value"] = MOMENTUM_SCALE * df["leverage"] * df["pred_error"]

    # Momentum recursion
    m = 0.0
    series = []
    for v in df["swing_value"].values:
        m = MOMENTUM_DECAY * m + float(v)
        m = max(-CLIP_MOMENTUM, min(CLIP_MOMENTUM, m))
        series.append(m)

    df["momentum_index"] = series
    df["momentum_phase"] = df["momentum_index"].apply(momentum_phase_label)

    # Big swings (tune)
    df["just_won_big"] = (df["swing_value"] >= 2.5).astype(int)
    df["just_lost_big"] = (df["swing_value"] <= -2.5).astype(int)

    df["mental_cue"] = [
        mental_prep_cue(ph, bool(jl), bool(jw))
        for ph, jl, jw in zip(df["momentum_phase"], df["just_lost_big"], df["just_won_big"])
    ]

    return df


#%% filtering 

def list_players(df_points: pd.DataFrame) -> np.ndarray:
    return pd.unique(pd.concat([df_points["p1_name"], df_points["p2_name"]], ignore_index=True))

def find_player_candidates(df_points: pd.DataFrame, needle: str, top: int = 30) -> List[str]:
    n = str(needle).strip().lower()
    players = list_players(df_points)
    hits = [p for p in players if n in str(p).lower()]
    return [str(x) for x in hits[:top]]

def opponent_type_simple(opponent_name: str) -> str:
    """
    Placeholder opponent type bucket (replace with your MCP/KMeans style map later).
    """
    if not isinstance(opponent_name, str) or opponent_name.strip() == "":
        return "Unknown"

    big_servers = {"John Isner", "Reilly Opelka", "Ivo Karlovic"}
    grinders = {"Rafael Nadal", "David Ferrer", "Diego Schwartzman"}
    all_court = {"Roger Federer", "Stefanos Tsitsipas", "Grigor Dimitrov"}

    if opponent_name in big_servers:
        return "Big Server"
    if opponent_name in grinders:
        return "Grinder"
    if opponent_name in all_court:
        return "All-Court"
    return "Unknown"


#%% power bi tables

def build_momentum_tables(
    df_points: pd.DataFrame,
    expected_model: Pipeline,
    focus_player_filter: str,
    surface_filter: Optional[str] = None,
    opponent_filter: Optional[str] = None,
    opponent_type_filter: Optional[str] = None,
    max_matches_after_filter: Optional[int] = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      momentum_points_long: one row per point per match per focus player
      momentum_match_summary: one row per match per focus player
    """

    # Build match frame for filtering first
    match_frame = (
        df_points.groupby("match_id", as_index=False)
        .agg(
            p1_name=("p1_name", "first"),
            p2_name=("p2_name", "first"),
            surface=("surface", "first"),
            year=("year", "first"),
            slam=("slam", "first"),
        )
    )

    # Apply surface filter
    if surface_filter is not None:
        match_frame = match_frame[
            match_frame["surface"].astype(str).str.lower() == str(surface_filter).strip().lower()
        ]

    # Focus player (substring match)
    fp = str(focus_player_filter).strip().lower()
    match_frame_fp = match_frame[
        match_frame["p1_name"].astype(str).str.lower().str.contains(fp, na=False) |
        match_frame["p2_name"].astype(str).str.lower().str.contains(fp, na=False)
    ].copy()

    if match_frame_fp.empty:
        print("\n[DIAGNOSTIC] No matches contain focus player filter:", focus_player_filter)
        cands = find_player_candidates(df_points, focus_player_filter)
        print("[DIAGNOSTIC] Candidate players containing that string:", cands[:20])
        print("[DIAGNOSTIC] Years in current merged dataset:",
              sorted(df_points["year"].dropna().unique())[:10], "...",
              sorted(df_points["year"].dropna().unique())[-10:])
        # Hard fail here (this means player truly not in loaded years)
        raise RuntimeError(
            "Focus player not found in the currently loaded year range. "
            "Load later years (e.g., 2020–2024) or broaden the year list."
        )

    # Apply max_matches AFTER filtering
    if max_matches_after_filter is not None:
        match_frame_fp = match_frame_fp.head(max_matches_after_filter)

    match_ids = match_frame_fp["match_id"].astype(str).tolist()

    # Build momentum
    momentum_rows = []
    for mid in match_ids:
        dfm = df_points[df_points["match_id"].astype(str) == str(mid)].copy()
        if dfm.empty:
            continue

        # Determine actual focus player name in this match
        p1 = str(dfm["p1_name"].iloc[0])
        p2 = str(dfm["p2_name"].iloc[0])
        if fp in p1.lower():
            focus_player = p1
            opponent = p2
        else:
            focus_player = p2
            opponent = p1

        opp_type = opponent_type_simple(opponent)

        # Opponent filters
        if opponent_filter is not None:
            if str(opponent_filter).strip().lower() not in opponent.lower():
                continue
        if opponent_type_filter is not None:
            if str(opponent_type_filter).strip().lower() != str(opp_type).strip().lower():
                continue

        df_focus = compute_momentum_for_match(dfm.sort_values("point_index"), expected_model, focus_player)

        df_focus["focus_player"] = focus_player
        df_focus["opponent"] = opponent
        df_focus["opponent_type"] = opp_type

        keep_cols = [
            "match_id", "year", "slam", "surface",
            "focus_player", "opponent", "opponent_type",
            "point_index",
            "server_name", "returner_name", "focus_is_server",
            "score_str_server", "score_bucket",
            "serve_number", "speed_bucket", "serve_speed_kmh",
            "focus_actual_win", "focus_exp_win",
            "leverage", "pred_error", "swing_value",
            "momentum_index", "momentum_phase",
            "just_won_big", "just_lost_big",
            "mental_cue"
        ]
        momentum_rows.append(df_focus[keep_cols].copy())

    if not momentum_rows:
        # Provide precise diagnostic rather than generic message
        raise RuntimeError(
            "Matches found for player, but none survived opponent/surface/type filters. "
            "Set opponent_filter=None and opponent_type_filter=None and/or surface_filter=None."
        )

    momentum_points_long = pd.concat(momentum_rows, ignore_index=True)

    summary = (momentum_points_long
               .groupby(["match_id", "year", "slam", "surface", "focus_player", "opponent", "opponent_type"])
               .agg(
                   n_points=("point_index", "count"),
                   win_rate=("focus_actual_win", "mean"),
                   momentum_mean=("momentum_index", "mean"),
                   momentum_std=("momentum_index", "std"),
                   momentum_end=("momentum_index", "last"),
                   big_wins=("just_won_big", "sum"),
                   big_losses=("just_lost_big", "sum")
               )
               .reset_index())

    return momentum_points_long, summary


#%% exports and plotting

def export_for_power_bi(points_long: pd.DataFrame, summary: pd.DataFrame) -> Tuple[str, str]:
    p1 = os.path.join(EXPORT_DIR, "momentum_points_long.csv")
    p2 = os.path.join(EXPORT_DIR, "momentum_match_summary.csv")
    points_long.to_csv(p1, index=False, encoding="utf-8")
    summary.to_csv(p2, index=False, encoding="utf-8")
    print(f"\nSaved → {p1}")
    print(f"Saved → {p2}")
    return p1, p2


def plot_match_momentum(points_long: pd.DataFrame, match_id: str, focus_player: str) -> None:
    df = points_long[
        (points_long["match_id"].astype(str) == str(match_id)) &
        (points_long["focus_player"].astype(str).str.lower() == str(focus_player).strip().lower())
    ].copy()

    if df.empty:
        print("No rows found for that match_id + focus_player in momentum_points_long.")
        return

    df = df.sort_values("point_index")
    title = f"Momentum State Graph — {df['focus_player'].iloc[0]} vs {df['opponent'].iloc[0]} | {df['surface'].iloc[0]} | {int(df['year'].iloc[0])} {df['slam'].iloc[0]}"

    x = df["point_index"].values
    y = df["momentum_index"].values

    plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.axhline(0, linewidth=1)
    plt.axhline(PHASE_POS_EDGE, linestyle="--", linewidth=1)
    plt.axhline(PHASE_POS_DOMINANT, linestyle="--", linewidth=1)
    plt.axhline(PHASE_NEG_EDGE, linestyle="--", linewidth=1)
    plt.axhline(PHASE_NEG_DOMINANT, linestyle="--", linewidth=1)

    big_win = df["just_won_big"].values.astype(bool)
    big_loss = df["just_lost_big"].values.astype(bool)
    plt.scatter(x[big_win], y[big_win], marker="^")
    plt.scatter(x[big_loss], y[big_loss], marker="v")

    plt.title(title)
    plt.xlabel("Point index")
    plt.ylabel("Momentum index (focus perspective)")
    plt.tight_layout()
    plt.show()

    print("\nLast 8 mental cues:")
    print(df[["point_index", "momentum_index", "momentum_phase", "mental_cue"]].tail(8).to_string(index=False))


#%% main run 

if __name__ == "__main__":
    # --- Choose years. For Sinner, start at 2020+ to guarantee he appears.
    YEARS = list(range(2020, 2025))  # 2020-2024
    SLAMS = ["ausopen", "frenchopen", "wimbledon", "usopen"]

    # --- Filters (Power BI will do filtering later too; keep these light while testing)
    FOCUS_PLAYER = "Sinner"       # substring match; "Jannik Sinner" also ok
    SURFACE = None                # "Hard"/"Clay"/"Grass" or None
    OPPONENT_NAME = None          # e.g. "Djokovic" (substring) or None
    OPPONENT_TYPE = None          # e.g. "Big Server" or None (placeholder taxonomy)

    # 1) Load
    matches_all, points_all, loaded_pairs = load_slam_data(YEARS, SLAMS, verbose=True)

    # 2) Merge
    df_points = merge_points_matches(points_all, matches_all)
    print("\nMerged points rows:", len(df_points))
    print("Merged years:", sorted(df_points["year"].dropna().unique()))
    print("Merged surfaces:", df_points["surface"].value_counts().to_dict())

    # Confirm player exists
    hits = find_player_candidates(df_points, FOCUS_PLAYER)
    print(f"\nPlayers containing '{FOCUS_PLAYER}':", hits[:15])

    # 3) Train expected model
    expected_model = train_expected_point_model(df_points)

    # 4) Build momentum tables
    points_long, match_summary = build_momentum_tables(
        df_points=df_points,
        expected_model=expected_model,
        focus_player_filter=FOCUS_PLAYER,
        surface_filter=SURFACE,
        opponent_filter=OPPONENT_NAME,
        opponent_type_filter=OPPONENT_TYPE,
        max_matches_after_filter=250
    )

    print("\nMatch summary (top 10):")
    print(match_summary.head(10).to_string(index=False))

    # 5) Export CSVs
    export_for_power_bi(points_long, match_summary)

    # 6) Plot one match
    # Pick first match in summary:
    mid = str(match_summary["match_id"].iloc[0])
    fp_exact = str(match_summary["focus_player"].iloc[0])
    plot_match_momentum(points_long, match_id=mid, focus_player=fp_exact)
    
#%% model evaluation


print("\nmodel evaluation\n")

feature_cols = ["surface", "score_bucket", "speed_bucket", "serve_number"]

# Sample up to N points for evaluation
N_EVAL = 200_000
df_eval = df_points.sample(n=min(N_EVAL, len(df_points)), random_state=42)

X_eval = df_eval[feature_cols]
y_true = df_eval["point_won"].astype(int).values

y_proba = expected_model.predict_proba(X_eval)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("EXPECTED POINT WIN MODEL (sampled evaluation)")
print("--------------------------------------------")
print("Rows used     :", len(df_eval))
print("Accuracy      :", round(accuracy_score(y_true, y_pred), 4))
print("ROC AUC       :", round(roc_auc_score(y_true, y_proba), 4))

#  calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

plt.figure(figsize=(5, 5))
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Calibration Curve (Expected Point Model)")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.tight_layout()
plt.show()

