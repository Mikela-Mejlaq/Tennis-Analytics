#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mikelamejlaq
"""

#%% Rally Advantage Model
# ------------------------------------------------------------
# - Loads MCP data from GitHub
# - Infers rally length from MCP shot-sequence strings in '1st'/'2nd'
# - Builds player style clusters
# - Trains global context model (logistic regression)
# - Computes Rally Advantage = mean(actual - expected) by player/surface/rally_bucket
# - Exports Power BI-ready CSV
# - Plots a bar chart in Spyder - giving an idea of results

# Importing packages

import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


#%% Parameter settings

MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"

POINT_FILES = [
    "charting-m-points-to-2009.csv",
    "charting-m-points-2010s.csv",
    "charting-m-points-2020s.csv",
]

MATCHES_FILE = "charting-m-matches.csv"

# Optional style feature files
SERVE_BASICS_FILE     = "charting-m-stats-ServeBasics.csv"
KEYPOINTS_SERVE_FILE  = "charting-m-stats-KeyPointsServe.csv"
KEYPOINTS_RETURN_FILE = "charting-m-stats-KeyPointsReturn.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROWS_POINTS = None  # set to e.g. 1_500_000 if you run out of RAM

POWERBI_OUTFILE = "rally_advantage_powerbi.csv"

# Plot defaults
PLOT_SURFACE = "Hard"
PLOT_ROLE = "Server"         # "Server" or "Returner"
PLOT_RALLY_BUCKET = "10+"
PLOT_MIN_POINTS = 250
PLOT_TOP_N = 20


#%% Creating the feature buckets

def safe_read_csv(url: str, **kwargs) -> pd.DataFrame:
    print(f"Loading: {url}")
    return pd.read_csv(url, **kwargs)

def standardize_surface(s: str) -> str:
    if pd.isna(s):
        return "Unknown"
    s = str(s).strip()
    return s if s else "Unknown"

def score_to_bucket(score_str: str) -> str:
    # MCP points generally don't have a clean score; so unknowns needs to be set here.
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

def rally_to_bucket(v) -> str:
    if pd.isna(v):
        return "Unknown"
    try:
        v = float(v)
    except Exception:
        return "Unknown"
    if v <= 0:
        return "Unknown"
    if v <= 3:
        return "0-3"
    if v <= 6:
        return "4-6"
    if v <= 9:
        return "7-9"
    return "10+"


#%% Outcome, serve number and rally length from the MCP Data

def infer_serve_number(df: pd.DataFrame) -> pd.Series:
    """
    MCP points: if '2nd' exists and is non-empty -> serve_number = 2 else 1.
    Robust to dtype issues.
    """
    if "2nd" in df.columns:
        c = df["2nd"].astype(str).fillna("").str.strip()
        is_second = (c != "") & (c.str.lower() != "nan")
        return np.where(is_second, 2, 1).astype(int)
    return pd.Series(1, index=df.index, dtype=int)

def compute_server_point_won(df: pd.DataFrame) -> pd.Series:
    """
    MCP: PtWinner = 1/2 means player1/player2 won the point.
    Server mapping: Svr==1 -> p1 serves else p2 serves.
    Returns point_won from server perspective.
    """
    if "PtWinner" in df.columns:
        pw = df["PtWinner"]
    elif "point_winner_raw" in df.columns:
        pw = df["point_winner_raw"]
    else:
        raise ValueError("Missing PtWinner column in MCP points.")

    if not all(c in df.columns for c in ["Svr", "p1_name", "p2_name"]):
        raise ValueError("Missing Svr/p1_name/p2_name in merged points df.")

    server_is_p1 = (df["Svr"] == 1)

    out = np.where(
        ((pw == 1) & server_is_p1) | ((pw == 2) & (~server_is_p1)),
        1,
        np.where(pw.isin([1, 2]), 0, np.nan)
    )
    return pd.to_numeric(out, errors="coerce")

def find_rally_length(df: pd.DataFrame) -> pd.Series:
    """
    MCP points often do NOT have a rally length column.
    Infer rally length from the shot-sequence notation in '1st'/'2nd'.

    Definition used:
      rally_len = number of shots AFTER the serve (return + rally shots)
    Example from data: '4ffbbf*' -> 5

    Strategy:
      1) If a numeric rally column exists, use it
      2) Else parse shot letters from the 2nd-serve sequence (if present) else 1st
    """
    # 1) True numeric rally columns (not commonly found in the data)
    numeric_candidates = [
        "rally_len", "rally_length", "rallyLength", "RallyLength",
        "shot_count", "shots", "Shots", "rallyCount", "RallyCount"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.2:
                print(f"Using rally length column: {c}")
                return s

    # 2) Infer from MCP shot strings
    if "1st" not in df.columns:
        raise ValueError("MCP points missing '1st' column; cannot infer rally length.")

    # Choose the serve string used: if 2nd serve exists -> use it, else 1st
    if "2nd" in df.columns:
        second = df["2nd"].astype(str).fillna("").str.strip()
        use_second = (second != "") & (second.str.lower() != "nan")
        seq = pd.Series(np.where(use_second, df["2nd"].astype(str), df["1st"].astype(str)), index=df.index)
    else:
        seq = df["1st"].astype(str)

    seq = seq.fillna("").astype(str).str.strip()

    # Shot-letter regex (covers common MCP encodings)
    # f b s r v l o m h i j k u p d
    shot_re = re.compile(r"[fbsrvloomhijkupd]")

    def count_shots_after_serve(code: str) -> float:
        if code is None:
            return np.nan
        t = str(code).strip()
        if t == "" or t.lower() == "nan":
            return np.nan

        # Remove leading serve direction digit 4/5/6 and +/=/-
        t = re.sub(r"^[456][\+\=\-]?", "", t)

        shots = shot_re.findall(t)
        return float(len(shots)) if shots else np.nan

    rally_len = seq.apply(count_shots_after_serve)

    if rally_len.notna().mean() < 0.05:
        raise ValueError(
            "Could not infer rally length from '1st'/'2nd' strings. "
            "Your MCP files may not include shot sequences in these columns "
            "Print points.columns and inspect."
        )

    print("Inferred rally length from MCP '1st'/'2nd' shot strings.")
    return rally_len


#%% PLayer style clustering

def build_player_style_map() -> dict:
    """
    Build player-style_label from MCP stats.
    If download/columns fail, returns {} and styles become 'Unknown'.
    """
    try:
        serve_basics     = safe_read_csv(MCP_BASE + SERVE_BASICS_FILE)
        keypoints_serve  = safe_read_csv(MCP_BASE + KEYPOINTS_SERVE_FILE)
        keypoints_return = safe_read_csv(MCP_BASE + KEYPOINTS_RETURN_FILE)

        sb = serve_basics.copy()
        sb["pts"] = sb["pts"].replace(0, np.nan)
        sb["unreturned_pct"] = sb["unret"] / sb["pts"]
        sb["ace_rate"] = sb["aces"] / sb["pts"]

        if "pts_won_lte_3_shots" in sb.columns:
            sb["short_point_pct"] = sb["pts_won_lte_3_shots"] / sb["pts"]
        else:
            sb["short_point_pct"] = np.nan

        serve_summary = sb.groupby("player").agg({
            "unreturned_pct": "mean",
            "ace_rate": "mean",
            "short_point_pct": "mean"
        }).reset_index()

        ks = keypoints_serve.copy()
        kr = keypoints_return.copy()
        ks["pts"] = ks["pts"].replace(0, np.nan)
        kr["pts"] = kr["pts"].replace(0, np.nan)

        ks["pressure_serve_win_pct"] = ks["pts_won"] / ks["pts"]
        kr["pressure_return_win_pct"] = kr["pts_won"] / kr["pts"]

        pressure_summary = (
            ks.groupby("player").agg({"pressure_serve_win_pct": "mean"})
              .merge(kr.groupby("player").agg({"pressure_return_win_pct": "mean"}),
                     on="player", how="outer")
        )

        player_features = serve_summary.merge(pressure_summary, on="player", how="outer")
        player_features = player_features.dropna().reset_index(drop=True)

        if len(player_features) < 50:
            return {}

        X_pf = player_features.drop(columns=["player"])
        X_scaled = StandardScaler().fit_transform(X_pf)

        kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
        player_features["cluster"] = kmeans.fit_predict(X_scaled)

        labels = {
            0: "Big-Serve Aggressor",
            1: "Baseline Grinder",
            2: "All-Court Balanced",
            3: "Counter-Puncher"
        }
        player_features["style_label"] = player_features["cluster"].map(labels).fillna("Unknown")

        style_map = dict(zip(player_features["player"], player_features["style_label"]))
        print(f"Built style labels for {len(style_map)} players.")
        return style_map

    except Exception as e:
        print(f"[Style clustering skipped] {e}")
        return {}


#%% Load MCP points and matches

matches = safe_read_csv(MCP_BASE + MATCHES_FILE)

points_list = []
for fn in POINT_FILES:
    try:
        dfp = safe_read_csv(MCP_BASE + fn)
        if MAX_ROWS_POINTS is not None and len(dfp) > MAX_ROWS_POINTS:
            dfp = dfp.sample(MAX_ROWS_POINTS, random_state=RANDOM_STATE)
        points_list.append(dfp)
        print(f"Loaded {fn}: shape={dfp.shape}")
    except Exception as e:
        print(f"Skipping {fn}: {e}")

if not points_list:
    raise RuntimeError("No MCP points files could be loaded.")

points = pd.concat(points_list, ignore_index=True)

print("\nTotal points loaded:", points.shape)
print("Total matches loaded:", matches.shape)

# PLayer style map
player_style_map = build_player_style_map()


#%% Building modelling dataframe

m = matches.rename(columns={
    "match_id": "match_id",
    "Player 1": "p1_name",
    "Player 2": "p2_name",
    "Surface": "surface"
})

need_cols = ["match_id", "p1_name", "p2_name", "surface"]
for c in need_cols:
    if c not in m.columns:
        raise ValueError(f"Matches missing required column: {c}")

m = m[need_cols].copy()
m["surface"] = m["surface"].apply(standardize_surface)

if "match_id" not in points.columns:
    raise ValueError("Points missing match_id")

df = points.merge(m, on="match_id", how="left")

if "Svr" not in df.columns:
    raise ValueError("Points missing Svr column")

df["server_name"] = np.where(df["Svr"] == 1, df["p1_name"], df["p2_name"])
df["returner_name"] = np.where(df["Svr"] == 1, df["p2_name"], df["p1_name"])

df["point_won"] = compute_server_point_won(df)
df = df.dropna(subset=["point_won"]).copy()
df["point_won"] = df["point_won"].astype(int)

df["serve_number"] = infer_serve_number(df)

df["rally_len"] = find_rally_length(df)
df["rally_bucket"] = df["rally_len"].apply(rally_to_bucket)
df = df[df["rally_bucket"] != "Unknown"].copy()

# Score bucket (usually Unknown for MCP points)
score_candidates = ["Score", "score", "PointScore", "PScore", "GameScore"]
score_col = next((c for c in score_candidates if c in df.columns), None)
if score_col is None:
    df["score_bucket"] = "Unknown"
else:
    df["score_bucket"] = df[score_col].apply(score_to_bucket)

df["server_type"] = df["server_name"].apply(lambda x: player_style_map.get(x, "Unknown"))
df["opponent_type"] = df["returner_name"].apply(lambda x: player_style_map.get(x, "Unknown"))

df_model = df[[
    "surface",
    "server_name", "returner_name",
    "server_type", "opponent_type",
    "serve_number",
    "score_bucket",
    "rally_bucket",
    "point_won"
]].copy()

print("Model DF shape:", df_model.shape)
print("Surface counts:\n", df_model["surface"].value_counts().head(10))
print("Rally bucket counts:\n", df_model["rally_bucket"].value_counts())


#%% Train global context model

feature_cols = ["surface", "rally_bucket", "score_bucket", "server_type", "opponent_type", "serve_number"]
X = df_model[feature_cols]
y = df_model["point_won"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["surface", "rally_bucket", "score_bucket", "server_type", "opponent_type"]),
    ],
    remainder="passthrough"
)

global_rally_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=800))
])

if len(df_model) >= 5000:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    global_rally_model.fit(X_tr, y_tr)
    proba = global_rally_model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
else:
    global_rally_model.fit(X, y)
    auc = np.nan

print("Global rally model trained. ROC AUC (if split):", auc)


#%% Compute rally advantage tables (PowerBI inputs)

expected_server = global_rally_model.predict_proba(df_model[feature_cols])[:, 1]

tmp = df_model.copy()
tmp["expected_server_win"] = expected_server
tmp["actual_server_win"] = tmp["point_won"].astype(float)

# Server role
tmp_server = tmp.copy()
tmp_server["entity_name"] = tmp_server["server_name"]
tmp_server["role"] = "Server"
tmp_server["actual"] = tmp_server["actual_server_win"]
tmp_server["expected"] = tmp_server["expected_server_win"]
tmp_server["residual"] = tmp_server["actual"] - tmp_server["expected"]

adv_server = (
    tmp_server.groupby(["entity_name", "role", "surface", "rally_bucket"])
      .agg(
          n_points=("residual", "size"),
          rally_advantage=("residual", "mean"),
          avg_actual=("actual", "mean"),
          avg_expected=("expected", "mean")
      )
      .reset_index()
)

# Returner role
tmp_return = tmp.copy()
tmp_return["entity_name"] = tmp_return["returner_name"]
tmp_return["role"] = "Returner"
tmp_return["actual"] = 1.0 - tmp_return["actual_server_win"]
tmp_return["expected"] = 1.0 - tmp_return["expected_server_win"]
tmp_return["residual"] = tmp_return["actual"] - tmp_return["expected"]

adv_return = (
    tmp_return.groupby(["entity_name", "role", "surface", "rally_bucket"])
      .agg(
          n_points=("residual", "size"),
          rally_advantage=("residual", "mean"),
          avg_actual=("actual", "mean"),
          avg_expected=("expected", "mean")
      )
      .reset_index()
)

adv_all = pd.concat([adv_server, adv_return], ignore_index=True)

print("Advantage table rows:", len(adv_all))
print(adv_all.head())

#%% exporting the tables

adv_all.to_csv(POWERBI_OUTFILE, index=False)
print(f"Saved Power BI CSV: {POWERBI_OUTFILE}")

#%% Spyder bar plot

sub = adv_all[
    (adv_all["surface"] == PLOT_SURFACE) &
    (adv_all["rally_bucket"] == PLOT_RALLY_BUCKET) &
    (adv_all["role"] == PLOT_ROLE) &
    (adv_all["n_points"] >= PLOT_MIN_POINTS)
].copy()

if sub.empty:
    print("No rows match plot filters. Try lowering PLOT_MIN_POINTS or changing surface/bucket/role.")
else:
    sub = sub.sort_values("rally_advantage", ascending=False).head(PLOT_TOP_N)

    plt.figure(figsize=(10, 5))
    plt.bar(sub["entity_name"], sub["rally_advantage"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Rally Advantage = mean(actual − expected)")
    plt.title(f"Rally Advantage ({PLOT_ROLE}) – {PLOT_SURFACE} – bucket {PLOT_RALLY_BUCKET} (min {PLOT_MIN_POINTS} pts)")
    plt.tight_layout()
    plt.show()

    print("\nTop rows for plot:")
    print(sub.to_string(index=False))
    
