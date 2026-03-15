#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: mikelamejlaq

Break-Point Forecasts

Predicts for a given player vs opponent (or opponent-type):
  - Expected BP Created
  - Expected BP Converted + conversion probability
  - Expected BP Faced
  - Expected BP Saved + save probability
"""

# %% Importing packages

import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# %% Setting parameters

@dataclass
class Config:
    START_YEAR: int = 2005
    END_YEAR: int = 2025

    ROLLING_WINDOW: int = 20
    MIN_HISTORY_MATCHES: int = 8

    N_STYLE_CLUSTERS: int = 4
    BINOMIAL_EXPAND_CAP: int = 30

    DROP_MISSING_BP_STATS: bool = True


CFG = Config()

ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"

serve_basics_url     = MCP_BASE + "charting-m-stats-ServeBasics.csv"
keypoints_serve_url  = MCP_BASE + "charting-m-stats-KeyPointsServe.csv"
keypoints_return_url = MCP_BASE + "charting-m-stats-KeyPointsReturn.csv"

# Power BI outputs
OUT_DIR_NAME = "powerbi_exports"
OUT_PREDICTIONS = "bp_predictions.csv"
OUT_FEATURE_SCHEMA = "bp_feature_schema.csv"
OUT_PLAYER_STYLES = "bp_player_styles.csv"
OUT_LATEST_SNAPSHOT = "bp_latest_snapshot.csv"

# %% 1) MCP style clusters

def build_player_style_clusters(cfg: Config) -> Dict[str, str]:
    serve_basics     = pd.read_csv(serve_basics_url)
    keypoints_serve  = pd.read_csv(keypoints_serve_url)
    keypoints_return = pd.read_csv(keypoints_return_url)

    sb = serve_basics.copy()
    sb["pts"] = sb["pts"].replace(0, np.nan)
    sb["unreturned_pct"]  = sb["unret"] / sb["pts"]
    sb["ace_rate"]        = sb["aces"] / sb["pts"]
    sb["short_point_pct"] = sb["pts_won_lte_3_shots"] / sb["pts"]

    serve_summary = sb.groupby("player").agg({
        "unreturned_pct": "mean",
        "ace_rate": "mean",
        "short_point_pct": "mean"
    }).reset_index()

    ks = keypoints_serve.copy()
    kr = keypoints_return.copy()
    ks["pts"] = ks["pts"].replace(0, np.nan)
    kr["pts"] = kr["pts"].replace(0, np.nan)

    ks["pressure_serve_win_pct"]  = ks["pts_won"] / ks["pts"]
    kr["pressure_return_win_pct"] = kr["pts_won"] / kr["pts"]

    pressure_summary = ks.groupby("player").agg({
        "pressure_serve_win_pct": "mean"
    }).merge(
        kr.groupby("player").agg({"pressure_return_win_pct": "mean"}),
        on="player", how="outer"
    )

    player_features = serve_summary.merge(pressure_summary, on="player", how="outer")
    player_features = player_features.dropna().reset_index(drop=True)

    X_pf = player_features.drop(columns=["player"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pf)

    kmeans = KMeans(n_clusters=cfg.N_STYLE_CLUSTERS, random_state=42, n_init=10)
    player_features["style_cluster"] = kmeans.fit_predict(X_scaled)

    cluster_labels = {
        0: "Big-Serve Aggressor",
        1: "Baseline Grinder",
        2: "All-Court Balanced",
        3: "Counter-Puncher"
    }
    player_features["style_label"] = player_features["style_cluster"].map(cluster_labels).fillna("Unknown")

    return dict(zip(player_features["player"], player_features["style_label"]))

# %% 2) Load ATP matches from GitHub

def load_atp_matches_github(cfg: Config) -> pd.DataFrame:
    urls = [f"{ATP_BASE}atp_matches_{y}.csv" for y in range(cfg.START_YEAR, cfg.END_YEAR + 1)]

    dfs = []
    for url in urls:
        try:
            df = pd.read_csv(url, low_memory=False)
            df["source_url"] = url
            dfs.append(df)
            print(f"Loaded: {url}  shape={df.shape}")
        except Exception as e:
            print(f"Skipping {url}: {e}")

    if not dfs:
        raise FileNotFoundError(
            "No ATP match files could be loaded from GitHub.\n"
            "Check internet access / proxy / SSL, or reduce year range."
        )

    matches = pd.concat(dfs, ignore_index=True)

    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    matches = matches.dropna(subset=["tourney_date"]).copy()

    if "tourney_id" not in matches.columns or "match_num" not in matches.columns:
        matches["match_uid"] = "synthetic_" + matches.index.astype(str)
    else:
        matches["match_uid"] = matches["tourney_id"].astype(str) + "_" + matches["match_num"].astype(str)

    needed = ["w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
    missing = [c for c in needed if c not in matches.columns]
    if missing:
        raise ValueError(f"Missing required BP columns in loaded data: {missing}")

    for c in needed:
        matches[c] = pd.to_numeric(matches[c], errors="coerce")

    if cfg.DROP_MISSING_BP_STATS:
        matches = matches.dropna(subset=needed).copy()
    else:
        matches[needed] = matches[needed].fillna(0)

    for c in needed:
        matches = matches[matches[c] >= 0]

    matches = matches.sort_values(["tourney_date", "match_uid"]).reset_index(drop=True)
    return matches

# %% 3) Player match table

def derive_player_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame({
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

        "won_match": 1
    })

    l = pd.DataFrame({
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

        "won_match": 0
    })

    pm = pd.concat([w, l], ignore_index=True)
    pm = pm.sort_values(["date", "match_uid", "player_name"]).reset_index(drop=True)

    pm["rank_diff"] = pm["player_rank"] - pm["opponent_rank"]
    pm["rp_diff"] = pm["player_rank_points"] - pm["opponent_rank_points"]
    pm["bp_conv_rate"] = np.where(pm["bp_created"] > 0, pm["bp_converted"] / pm["bp_created"], np.nan)
    pm["bp_save_rate"] = np.where(pm["bp_faced"] > 0, pm["bp_saved"] / pm["bp_faced"], np.nan)

    return pm

# %% 4) Rolling features

def add_rolling_features(pm: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pm = pm.copy()
    w = cfg.ROLLING_WINDOW
    m = max(3, cfg.MIN_HISTORY_MATCHES)

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["date", "match_uid"]).copy()

        for col in ["bp_created", "bp_converted", "bp_faced", "bp_saved"]:
            g[f"r_{col}_mean"] = g[col].shift(1).rolling(w, min_periods=m).mean()
            g[f"r_{col}_sum"]  = g[col].shift(1).rolling(w, min_periods=m).sum()

        g["r_bp_conv_rate"] = np.where(
            g["r_bp_created_sum"] > 0,
            g["r_bp_converted_sum"] / g["r_bp_created_sum"],
            np.nan
        )
        g["r_bp_save_rate"] = np.where(
            g["r_bp_faced_sum"] > 0,
            g["r_bp_saved_sum"] / g["r_bp_faced_sum"],
            np.nan
        )

        g["r_win_rate"] = g["won_match"].shift(1).rolling(w, min_periods=m).mean()
        g["r_matches_played"] = g["won_match"].shift(1).rolling(w, min_periods=m).count()
        return g

    pm = pm.groupby("player_name", group_keys=False).apply(_roll)

    pm = pm[pm["r_matches_played"].notna()].copy()
    pm = pm[pm["r_matches_played"] >= cfg.MIN_HISTORY_MATCHES].copy()

    opp_cols = [
        "match_uid", "player_name",
        "r_bp_created_mean", "r_bp_converted_mean", "r_bp_faced_mean", "r_bp_saved_mean",
        "r_bp_conv_rate", "r_bp_save_rate", "r_win_rate", "r_matches_played",
        "player_rank", "player_rank_points"
    ]

    opp = pm[opp_cols].copy()
    opp = opp.rename(columns={"player_name": "opponent_name"})
    opp = opp.rename(columns={c: f"opp_{c}" for c in opp.columns if c not in ["match_uid", "opponent_name"]})

    pm = pm.merge(opp, on=["match_uid", "opponent_name"], how="left", validate="m:1")
    return pm

# %% 5) Style features

def add_style_features(pm: pd.DataFrame, style_map: Dict[str, str]) -> pd.DataFrame:
    pm = pm.copy()
    pm["player_type"] = pm["player_name"].map(style_map).fillna("Unknown")
    pm["opponent_type"] = pm["opponent_name"].map(style_map).fillna("Unknown")
    return pm

# %% 6) Model training

def build_feature_lists(pm: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [
        "player_rank", "opponent_rank", "rank_diff",
        "player_rank_points", "opponent_rank_points", "rp_diff",
        "best_of",

        "r_bp_created_mean", "r_bp_converted_mean", "r_bp_faced_mean", "r_bp_saved_mean",
        "r_bp_conv_rate", "r_bp_save_rate", "r_win_rate", "r_matches_played",

        "opp_r_bp_created_mean", "opp_r_bp_converted_mean", "opp_r_bp_faced_mean", "opp_r_bp_saved_mean",
        "opp_r_bp_conv_rate", "opp_r_bp_save_rate", "opp_r_win_rate", "opp_r_matches_played",
        "opp_player_rank", "opp_player_rank_points"
    ]
    numeric_cols = [c for c in numeric_cols if c in pm.columns]

    categorical_cols = [
        "surface", "tourney_level", "round",
        "player_hand", "opponent_hand",
        "player_type", "opponent_type"
    ]
    categorical_cols = [c for c in categorical_cols if c in pm.columns]

    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[("num", num_tf, numeric_cols), ("cat", cat_tf, categorical_cols)],
        remainder="drop"
    )


def expand_binomial_rows(
    df: pd.DataFrame,
    feature_cols: List[str],
    trials_col: str,
    succ_col: str,
    cap: int,
    seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_rows, y_rows = [], []

    for i in range(len(df)):
        t = df.iloc[i][trials_col]
        s = df.iloc[i][succ_col]
        if pd.isna(t) or t <= 0:
            continue

        t = int(t)
        s = int(s)
        p = (s / t) if t > 0 else 0.0

        if t > cap:
            yi = rng.binomial(1, p, size=cap)
            reps = cap
        else:
            yi = np.array([1] * s + [0] * (t - s), dtype=int)
            rng.shuffle(yi)
            reps = t

        Xi = pd.concat([df.iloc[[i]][feature_cols]] * reps, ignore_index=True)
        X_rows.append(Xi)
        y_rows.append(yi)

    if not X_rows:
        raise RuntimeError("Binomial expansion produced no rows (no trials).")

    return pd.concat(X_rows, ignore_index=True), np.concatenate(y_rows)


@dataclass
class TrainedBPModels:
    numeric_cols: List[str]
    categorical_cols: List[str]
    model_created: Pipeline
    model_faced: Pipeline
    model_conv_prob: Pipeline
    model_save_prob: Pipeline


def train_bp_models(pm: pd.DataFrame, cfg: Config) -> TrainedBPModels:
    num_cols, cat_cols = build_feature_lists(pm)
    feat_cols = num_cols + cat_cols
    pre = make_preprocessor(num_cols, cat_cols)

    X = pm[feat_cols].copy()
    y_created = pm["bp_created"].astype(float).values
    y_faced   = pm["bp_faced"].astype(float).values

    model_created = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1.0, max_iter=2000))])
    model_faced   = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1.0, max_iter=2000))])

    base = pm[feat_cols + ["bp_created", "bp_converted", "bp_faced", "bp_saved"]].copy()

    X_conv_exp, y_conv_exp = expand_binomial_rows(
        base, feat_cols, "bp_created", "bp_converted", cfg.BINOMIAL_EXPAND_CAP, seed=7
    )
    X_save_exp, y_save_exp = expand_binomial_rows(
        base, feat_cols, "bp_faced", "bp_saved", cfg.BINOMIAL_EXPAND_CAP, seed=11
    )

    model_conv_prob = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    model_save_prob = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))])

    model_created.fit(X, y_created)
    model_faced.fit(X, y_faced)
    model_conv_prob.fit(X_conv_exp, y_conv_exp)
    model_save_prob.fit(X_save_exp, y_save_exp)

    return TrainedBPModels(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        model_created=model_created,
        model_faced=model_faced,
        model_conv_prob=model_conv_prob,
        model_save_prob=model_save_prob
    )

# %% 7) Model evaluation

def quick_eval_counts(pm: pd.DataFrame, models: TrainedBPModels, n_splits: int = 5) -> Dict[str, float]:
    X = pm[models.numeric_cols + models.categorical_cols].copy()
    y_created = pm["bp_created"].astype(float).values
    y_faced = pm["bp_faced"].astype(float).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes_c, maes_f = [], []

    for tr, te in tscv.split(X):
        pre = make_preprocessor(models.numeric_cols, models.categorical_cols)
        mc = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1.0, max_iter=2000))])
        mf = Pipeline([("pre", pre), ("reg", PoissonRegressor(alpha=1.0, max_iter=2000))])

        mc.fit(X.iloc[tr], y_created[tr])
        mf.fit(X.iloc[tr], y_faced[tr])

        maes_c.append(mean_absolute_error(y_created[te], mc.predict(X.iloc[te])))
        maes_f.append(mean_absolute_error(y_faced[te], mf.predict(X.iloc[te])))

    return {"MAE_bp_created": float(np.mean(maes_c)), "MAE_bp_faced": float(np.mean(maes_f))}


# %% 8) forecasting layer

def get_latest_context_row(pm: pd.DataFrame, player: str) -> pd.Series:
    dfp = pm[pm["player_name"] == player].copy()
    if dfp.empty:
        raise ValueError(f"No usable history for player '{player}'. Lower MIN_HISTORY_MATCHES or expand years.")
    return dfp.sort_values(["date", "match_uid"]).iloc[-1]


def build_synthetic_opponent_profile(pm: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    df = pm.copy()

    if "surface" in filters and "surface" in df.columns:
        df = df[df["surface"] == filters["surface"]]
    if "opponent_type" in filters and "opponent_type" in df.columns:
        df = df[df["opponent_type"] == filters["opponent_type"]]
    if "opponent_rank_max" in filters:
        df = df[pd.to_numeric(df["opponent_rank"], errors="coerce") <= float(filters["opponent_rank_max"])]
    if "opponent_rank_min" in filters:
        df = df[pd.to_numeric(df["opponent_rank"], errors="coerce") >= float(filters["opponent_rank_min"])]

    if df.empty:
        raise ValueError("Opponent-type filters returned no rows. Loosen filters or expand years.")

    prof: Dict[str, Any] = {}
    num_fields = [
        "opponent_rank", "opponent_rank_points",
        "opp_r_bp_created_mean", "opp_r_bp_converted_mean", "opp_r_bp_faced_mean", "opp_r_bp_saved_mean",
        "opp_r_bp_conv_rate", "opp_r_bp_save_rate", "opp_r_win_rate", "opp_r_matches_played",
        "opp_player_rank", "opp_player_rank_points"
    ]
    for f in num_fields:
        if f in df.columns:
            prof[f] = float(pd.to_numeric(df[f], errors="coerce").median())

    for f in ["opponent_hand", "opponent_type"]:
        if f in df.columns:
            mode = df[f].mode(dropna=True)
            prof[f] = mode.iloc[0] if not mode.empty else "Unknown"

    return prof


def build_forecast_row(
    pm: pd.DataFrame,
    player: str,
    opponent: Optional[str] = None,
    surface: Optional[str] = None,
    tourney_level: Optional[str] = None,
    best_of: int = 3,
    round_name: Optional[str] = None,
    opponent_type_filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    p = get_latest_context_row(pm, player)

    row: Dict[str, Any] = {
        "surface": surface if surface is not None else p.get("surface", "Unknown"),
        "tourney_level": tourney_level if tourney_level is not None else p.get("tourney_level", "Unknown"),
        "round": round_name if round_name is not None else p.get("round", "Unknown"),
        "best_of": best_of,

        "player_rank": p.get("player_rank", np.nan),
        "player_rank_points": p.get("player_rank_points", np.nan),
        "player_hand": p.get("player_hand", "Unknown"),
        "player_type": p.get("player_type", "Unknown"),

        "r_bp_created_mean": p.get("r_bp_created_mean", np.nan),
        "r_bp_converted_mean": p.get("r_bp_converted_mean", np.nan),
        "r_bp_faced_mean": p.get("r_bp_faced_mean", np.nan),
        "r_bp_saved_mean": p.get("r_bp_saved_mean", np.nan),
        "r_bp_conv_rate": p.get("r_bp_conv_rate", np.nan),
        "r_bp_save_rate": p.get("r_bp_save_rate", np.nan),
        "r_win_rate": p.get("r_win_rate", np.nan),
        "r_matches_played": p.get("r_matches_played", np.nan),
    }

    if opponent is not None:
        o = get_latest_context_row(pm, opponent)

        row["opponent_rank"] = o.get("player_rank", np.nan)
        row["opponent_rank_points"] = o.get("player_rank_points", np.nan)
        row["opponent_hand"] = o.get("player_hand", "Unknown")
        row["opponent_type"] = o.get("player_type", "Unknown")

        row["opp_r_bp_created_mean"] = o.get("r_bp_created_mean", np.nan)
        row["opp_r_bp_converted_mean"] = o.get("r_bp_converted_mean", np.nan)
        row["opp_r_bp_faced_mean"] = o.get("r_bp_faced_mean", np.nan)
        row["opp_r_bp_saved_mean"] = o.get("r_bp_saved_mean", np.nan)
        row["opp_r_bp_conv_rate"] = o.get("r_bp_conv_rate", np.nan)
        row["opp_r_bp_save_rate"] = o.get("r_bp_save_rate", np.nan)
        row["opp_r_win_rate"] = o.get("r_win_rate", np.nan)
        row["opp_r_matches_played"] = o.get("r_matches_played", np.nan)

        row["opp_player_rank"] = o.get("player_rank", np.nan)
        row["opp_player_rank_points"] = o.get("player_rank_points", np.nan)

    else:
        if opponent_type_filters is None:
            opponent_type_filters = {"surface": row["surface"], "opponent_rank_min": 1, "opponent_rank_max": 200}

        prof = build_synthetic_opponent_profile(pm, opponent_type_filters)

        row["opponent_rank"] = prof.get("opponent_rank", np.nan)
        row["opponent_rank_points"] = prof.get("opponent_rank_points", np.nan)
        row["opponent_hand"] = prof.get("opponent_hand", "Unknown")
        row["opponent_type"] = prof.get("opponent_type", "Unknown")

        row["opp_r_bp_created_mean"] = prof.get("opp_r_bp_created_mean", np.nan)
        row["opp_r_bp_converted_mean"] = prof.get("opp_r_bp_converted_mean", np.nan)
        row["opp_r_bp_faced_mean"] = prof.get("opp_r_bp_faced_mean", np.nan)
        row["opp_r_bp_saved_mean"] = prof.get("opp_r_bp_saved_mean", np.nan)
        row["opp_r_bp_conv_rate"] = prof.get("opp_r_bp_conv_rate", np.nan)
        row["opp_r_bp_save_rate"] = prof.get("opp_r_bp_save_rate", np.nan)
        row["opp_r_win_rate"] = prof.get("opp_r_win_rate", np.nan)
        row["opp_r_matches_played"] = prof.get("opp_r_matches_played", np.nan)

        row["opp_player_rank"] = prof.get("opp_player_rank", np.nan)
        row["opp_player_rank_points"] = prof.get("opp_player_rank_points", np.nan)

    row["rank_diff"] = row["player_rank"] - row["opponent_rank"] if pd.notna(row["player_rank"]) and pd.notna(row["opponent_rank"]) else np.nan
    row["rp_diff"] = row["player_rank_points"] - row["opponent_rank_points"] if pd.notna(row["player_rank_points"]) and pd.notna(row["opponent_rank_points"]) else np.nan

    return pd.DataFrame([row])


def align_features_for_prediction(X_one: pd.DataFrame, models: TrainedBPModels) -> pd.DataFrame:
    expected = models.numeric_cols + models.categorical_cols
    X_one = X_one.copy()
    for c in expected:
        if c not in X_one.columns:
            X_one[c] = np.nan
    X_one = X_one[expected].copy()
    return X_one


def forecast_break_points(models: TrainedBPModels, X_one: pd.DataFrame) -> Dict[str, float]:
    created = float(models.model_created.predict(X_one)[0])
    faced = float(models.model_faced.predict(X_one)[0])

    p_conv = float(models.model_conv_prob.predict_proba(X_one)[0, 1])
    p_save = float(models.model_save_prob.predict_proba(X_one)[0, 1])

    created = max(0.0, created)
    faced = max(0.0, faced)
    p_conv = float(np.clip(p_conv, 0.0, 1.0))
    p_save = float(np.clip(p_save, 0.0, 1.0))

    return {
        "bp_created_exp": created,
        "bp_converted_exp": created * p_conv,
        "bp_faced_exp": faced,
        "bp_saved_exp": faced * p_save,
        "bp_conv_prob": p_conv,
        "bp_save_prob": p_save
    }


def print_bp_forecast(player: str, opponent_label: str, surface: str, out: Dict[str, float]):
    print("\n" + "=" * 72)
    print(f"BREAK-POINT FORECAST — {player} vs {opponent_label}  (Surface={surface})")
    print("=" * 72)
    print(f"BP Created (exp)   : {out['bp_created_exp']:.2f}")
    print(f"BP Converted (exp) : {out['bp_converted_exp']:.2f}   | Conversion% ≈ {out['bp_conv_prob']:.1%}")
    print("-" * 72)
    print(f"BP Faced (exp)     : {out['bp_faced_exp']:.2f}")
    print(f"BP Saved (exp)     : {out['bp_saved_exp']:.2f}   | Save% ≈ {out['bp_save_prob']:.1%}")
    print("=" * 72 + "\n")


# %% 9) Exports for PowerBI

def get_output_dir() -> Path:
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    out_dir = base / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_predictions_table(models: TrainedBPModels, pm: pd.DataFrame) -> pd.DataFrame:
    """
    Power BI fact table. Add more scenarios as necessary.
    """
    scenarios = [
        {
            "scenario_id": "SINNER_vs_DJOKOVIC_HARD_SF",
            "player": "Jannik Sinner",
            "opponent": "Novak Djokovic",
            "surface": "Hard",
            "tourney_level": "M",
            "best_of": 3,
            "round_name": "SF",
            "opp_type_filters": None
        },
        {
            "scenario_id": "SINNER_vs_TOP30_BASELINEGRINDER_HARD_QF",
            "player": "Jannik Sinner",
            "opponent": None,
            "surface": "Hard",
            "tourney_level": "A",
            "best_of": 3,
            "round_name": "QF",
            "opp_type_filters": {
                "surface": "Hard",
                "opponent_type": "Baseline Grinder",
                "opponent_rank_min": 1,
                "opponent_rank_max": 30
            }
        }
    ]

    rows = []
    for sc in scenarios:
        X = build_forecast_row(
            pm=pm,
            player=sc["player"],
            opponent=sc["opponent"],
            surface=sc["surface"],
            tourney_level=sc["tourney_level"],
            best_of=sc["best_of"],
            round_name=sc["round_name"],
            opponent_type_filters=sc["opp_type_filters"]
        )
        X = align_features_for_prediction(X, models)
        out = forecast_break_points(models, X)

        rows.append({
            "scenario_id": sc["scenario_id"],
            "player_name": sc["player"],
            "opponent_name": sc["opponent"] if sc["opponent"] is not None else "OPPONENT_TYPE",
            "surface": sc["surface"],
            "tourney_level": sc["tourney_level"],
            "round": sc["round_name"],
            "best_of": sc["best_of"],
            "bp_created_exp": out["bp_created_exp"],
            "bp_converted_exp": out["bp_converted_exp"],
            "bp_faced_exp": out["bp_faced_exp"],
            "bp_saved_exp": out["bp_saved_exp"],
            "bp_conv_prob": out["bp_conv_prob"],
            "bp_save_prob": out["bp_save_prob"],
        })

    return pd.DataFrame(rows)


def build_feature_schema_table(models: TrainedBPModels) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_name": models.numeric_cols + models.categorical_cols,
        "feature_type": (["numeric"] * len(models.numeric_cols)) + (["categorical"] * len(models.categorical_cols))
    })


def build_player_styles_table(style_map: Dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "player_name": list(style_map.keys()),
        "player_type": list(style_map.values())
    }).drop_duplicates()


def build_latest_snapshot(pm: pd.DataFrame) -> pd.DataFrame:
    """
    Latest rolling features per player (for Power BI player cards).
    """
    snap = (pm.sort_values(["date", "match_uid"])
              .groupby("player_name", as_index=False)
              .tail(1)
              .copy())
    keep_cols = [
        "player_name", "date", "surface",
        "player_type",
        "player_rank", "player_rank_points",
        "r_bp_created_mean", "r_bp_converted_mean", "r_bp_faced_mean", "r_bp_saved_mean",
        "r_bp_conv_rate", "r_bp_save_rate", "r_win_rate", "r_matches_played"
    ]
    keep_cols = [c for c in keep_cols if c in snap.columns]
    return snap[keep_cols].reset_index(drop=True)


def export_powerbi_tables(style_map: Dict[str, str], pm: pd.DataFrame, models: TrainedBPModels) -> None:
    out_dir = get_output_dir()

    df_pred = build_predictions_table(models, pm)
    df_schema = build_feature_schema_table(models)
    df_styles = build_player_styles_table(style_map)
    df_snap = build_latest_snapshot(pm)

    p1 = out_dir / OUT_PREDICTIONS
    p2 = out_dir / OUT_FEATURE_SCHEMA
    p3 = out_dir / OUT_PLAYER_STYLES
    p4 = out_dir / OUT_LATEST_SNAPSHOT

    df_pred.to_csv(p1, index=False)
    df_schema.to_csv(p2, index=False)
    df_styles.to_csv(p3, index=False)
    df_snap.to_csv(p4, index=False)

    print("\n================ POWER BI EXPORT COMPLETE ================")
    print("Export folder:", out_dir)
    print("Saved files:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)
    print(" -", p4)
    print("==========================================================\n")


# %% 10) Main run

def main():
    print("Loading MCP style features + building clusters...")
    style_map = build_player_style_clusters(CFG)
    print(f"Style map size: {len(style_map):,}")

    print("Loading ATP matches from GitHub...")
    matches = load_atp_matches_github(CFG)
    print(f"Loaded matches: {len(matches):,}")

    print("Building player-match table...")
    pm = derive_player_match_rows(matches)

    print("Adding rolling features...")
    pm = add_rolling_features(pm, CFG)

    print("Adding style features...")
    pm = add_style_features(pm, style_map)

    print(f"Training rows after rolling filter: {len(pm):,}")
    print(f"Unique players: {pm['player_name'].nunique():,}")

    print("Training models...")
    models = train_bp_models(pm, CFG)
    print("Training complete.")

    try:
        eval_res = quick_eval_counts(pm, models, n_splits=5)
        print("Quick eval (counts only):", eval_res)
    except Exception as e:
        print("Quick eval skipped:", e)

    print("\nMODEL TRAIN FEATURE LIST (fit schema):")
    print(models.numeric_cols + models.categorical_cols)

    # Example 1
    player = "Jannik Sinner"
    opponent = "Novak Djokovic"
    surface = "Hard"

    X1 = build_forecast_row(
        pm=pm,
        player=player,
        opponent=opponent,
        surface=surface,
        tourney_level="M",
        best_of=3,
        round_name="SF",
        opponent_type_filters=None
    )
    X1 = align_features_for_prediction(X1, models)
    out1 = forecast_break_points(models, X1)
    print_bp_forecast(player, opponent, surface, out1)

    # Example 2
    player2 = "Jannik Sinner"
    surface2 = "Hard"

    opp_type_filters = {
        "surface": surface2,
        "opponent_type": "Baseline Grinder",
        "opponent_rank_min": 1,
        "opponent_rank_max": 30
    }

    X2 = build_forecast_row(
        pm=pm,
        player=player2,
        opponent=None,
        surface=surface2,
        tourney_level="A",
        best_of=3,
        round_name="QF",
        opponent_type_filters=opp_type_filters
    )
    X2 = align_features_for_prediction(X2, models)
    out2 = forecast_break_points(models, X2)
    print_bp_forecast(player2, "OpponentType(Top30 Baseline Grinder)", surface2, out2)

    #Power BI export tables
    export_powerbi_tables(style_map, pm, models)


if __name__ == "__main__":
    main()
