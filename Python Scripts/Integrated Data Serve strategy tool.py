"""
Created on Sat Dec  6 13:10:21 2025
@author: mikelamejlaq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated MCP + Slam serve strategy model

- MCP:
    - Player style clustering
    - 1st-serve direction (Wide / Body / T) from charting-m-points-*.csv
    - Opponent-specific + style-specific direction success

- Slam PBP:
    - Speed_KMH → speed_bucket (Fast/Medium/Slow)
    - Score → score_bucket (BreakPoint, Deuce, etc.)
    - Surface
    - Global logistic model P(point_won | surface, score_bucket, speed_bucket, server_type, opponent_type, serve_number)

- Combined model:
    For a given (player, opponent, surface, score, serve_number, direction, speed_bucket):
        P_combined = alpha * P_dir_MCP + (1 - alpha) * P_slam_global

- Main uses (shown here, but to be presented in dashboard):
    recommend_serve_options(...) → list of (direction, speed, probability, source info)
    plot_serve_heatmap(...) → direction × speed heatmap
"""

#%% Importing Necessary Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

#%% 1. Player Style Clusters from MCP

MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"

serve_basics_url     = MCP_BASE + "charting-m-stats-ServeBasics.csv"
keypoints_serve_url  = MCP_BASE + "charting-m-stats-KeyPointsServe.csv"
keypoints_return_url = MCP_BASE + "charting-m-stats-KeyPointsReturn.csv"
mcp_matches_url      = MCP_BASE + "charting-m-matches.csv"

serve_basics     = pd.read_csv(serve_basics_url)
keypoints_serve  = pd.read_csv(keypoints_serve_url)
keypoints_return = pd.read_csv(keypoints_return_url)
mcp_matches      = pd.read_csv(mcp_matches_url)

# Serve-based features for style clustering
sb = serve_basics.copy()
sb["pts"] = sb["pts"].replace(0, np.nan)

sb["unreturned_pct"]   = sb["unret"] / sb["pts"]
sb["ace_rate"]         = sb["aces"] / sb["pts"]
sb["short_point_pct"]  = sb["pts_won_lte_3_shots"] / sb["pts"]

serve_summary = sb.groupby("player").agg({
    "unreturned_pct": "mean",
    "ace_rate": "mean",
    "short_point_pct": "mean"
}).reset_index()

# Pressure features
ks = keypoints_serve.copy()
kr = keypoints_return.copy()

ks["pts"] = ks["pts"].replace(0, np.nan)
kr["pts"] = kr["pts"].replace(0, np.nan)

ks["pressure_serve_win_pct"]   = ks["pts_won"] / ks["pts"]
kr["pressure_return_win_pct"]  = kr["pts_won"] / kr["pts"]

pressure_summary = ks.groupby("player").agg({
    "pressure_serve_win_pct": "mean"
}).merge(
    kr.groupby("player").agg({"pressure_return_win_pct": "mean"}),
    on="player", how="outer"
)

player_features = (
    serve_summary
    .merge(pressure_summary, on="player", how="outer")
)

player_features = player_features.dropna().reset_index(drop=True)

# KMeans clustering into 4 style groups
X_pf = player_features.drop(columns=["player"])
scaler_styles = StandardScaler()
X_scaled = scaler_styles.fit_transform(X_pf)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
player_features["style_cluster"] = kmeans.fit_predict(X_scaled)

cluster_labels = {
    0: "Big-Serve Aggressor",
    1: "Baseline Grinder",
    2: "All-Court Balanced",
    3: "Counter-Puncher"
}
player_features["style_label"] = player_features["style_cluster"].map(cluster_labels)

print("Sample player style labels:")
print(player_features[["player", "style_label"]].head())
print("\nCounts by style label:")
print(player_features["style_label"].value_counts())

player_style_map = dict(zip(player_features["player"], player_features["style_label"]))

#%% 2. MCP Points: 1st-Serce Direction Success (WIDE / BODY / T)

# Load MCP men's points (three decades)
mcp_points_files = [
    MCP_BASE + "charting-m-points-to-2009.csv",
    MCP_BASE + "charting-m-points-2010s.csv",
    MCP_BASE + "charting-m-points-2020s.csv",
]

mcp_points_list = []
for url in mcp_points_files:
    try:
        df_tmp = pd.read_csv(url)
        mcp_points_list.append(df_tmp)
        print(f"Loaded MCP points: {url}, shape={df_tmp.shape}")
    except Exception as e:
        print(f"Skipping MCP points file {url}: {e}")

mcp_points_all = pd.concat(mcp_points_list, ignore_index=True)
print("\nTotal MCP points:", mcp_points_all.shape)

print("MCP points columns sample:", mcp_points_all.columns.tolist()[:20])

# Standardize match columns for MCP matches
mcp_matches_std = mcp_matches.rename(columns={
    "match_id": "match_id",
    "Player 1": "p1_name",
    "Player 2": "p2_name",
    "Surface": "surface"
})[["match_id", "p1_name", "p2_name", "surface"]]

# Merge basic match info into MCP points
mcp_points = mcp_points_all.merge(
    mcp_matches_std,
    on="match_id",
    how="left"
)

# Rename PtWinner
mcp_points = mcp_points.rename(columns={"PtWinner": "point_winner_raw"})

# Server / returner mapping
mcp_points["server_name"] = np.where(mcp_points["Svr"] == 1,
                                     mcp_points["p1_name"],
                                     mcp_points["p2_name"])
mcp_points["returner_name"] = np.where(mcp_points["Svr"] == 1,
                                       mcp_points["p2_name"],
                                       mcp_points["p1_name"])

# Point outcome from server perspective
def mcp_server_point_won(row):
    if row["point_winner_raw"] == 1 and row["server_name"] == row["p1_name"]:
        return 1
    elif row["point_winner_raw"] == 2 and row["server_name"] == row["p2_name"]:
        return 1
    elif row["point_winner_raw"] in [1, 2]:
        return 0
    else:
        return np.nan

mcp_points["point_won"] = mcp_points.apply(mcp_server_point_won, axis=1)
mcp_points = mcp_points.dropna(subset=["point_won"])
mcp_points["point_won"] = mcp_points["point_won"].astype(int)

# Parse 1st-serve direction from "1st" column
# According to MCP coding: first digit 4 = wide, 5 = body, 6 = T.
def parse_1st_serve_direction(code: str) -> str:
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

mcp_points["serve_direction"] = mcp_points["1st"].apply(parse_1st_serve_direction)

# Restrict to points where we have a meaningful first-serve direction
mcp_points_dir = mcp_points[(mcp_points["serve_direction"].isin(["Wide", "Body", "T"]))].copy()

print("\nMCP serve_direction distribution (1st serve only):")
print(mcp_points_dir["serve_direction"].value_counts())

# Attach styles (server_type, opponent_type)
server_styles = player_features.rename(columns={
    "player": "server_name",
    "style_label": "server_type"
})[["server_name", "server_type"]]

opp_styles = player_features.rename(columns={
    "player": "opponent_name",
    "style_label": "opponent_type"
})[["opponent_name", "opponent_type"]]

mcp_points_dir = mcp_points_dir.merge(server_styles, on="server_name", how="left")

mcp_points_dir = mcp_points_dir.merge(
    opp_styles,
    left_on="returner_name",
    right_on="opponent_name",
    how="left"
).drop(columns=["opponent_name"])

mcp_points_dir["server_type"]   = mcp_points_dir["server_type"].fillna("Unknown")
mcp_points_dir["opponent_type"] = mcp_points_dir["opponent_type"].fillna("Unknown")
mcp_points_dir["surface"]       = mcp_points_dir["surface"].astype(str)

print("\nMCP direction modeling df shape:", mcp_points_dir.shape)

# Build empirical direction stats (H2H, player, type)

# H2H: server vs specific opponent
mcp_h2h_dir_stats = (
    mcp_points_dir
    .groupby(
        ["server_name", "returner_name", "surface",
         "server_type", "opponent_type",
         "serve_direction"]
    )["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count": "n_points", "mean": "win_prob"})
)

# Player vs opponent type
mcp_player_dir_stats = (
    mcp_points_dir
    .groupby(
        ["server_name", "surface", "server_type", "opponent_type",
         "serve_direction"]
    )["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count": "n_points", "mean": "win_prob"})
)

# Style vs style
mcp_type_dir_stats = (
    mcp_points_dir
    .groupby(
        ["server_type", "surface", "opponent_type",
         "serve_direction"]
    )["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count": "n_points", "mean": "win_prob"})
)

print("\nMCP H2H dir stats head:")
print(mcp_h2h_dir_stats.head())

print("\nMCP player dir stats head:")
print(mcp_player_dir_stats.head())

print("\nMCP type dir stats head:")
print(mcp_type_dir_stats.head())

#%% 3. Slam PBP: Speed and Score Model (NO DIRECTION)

SLAM_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"

# Can extend later if possible
slam_years = range(2011, 2018)
slam_names = ["ausopen", "frenchopen", "wimbledon", "usopen"]

slam_points_list = []
slam_matches_list = []

for year in slam_years:
    for slam in slam_names:
        points_fname  = f"{year}-{slam}-points.csv"
        matches_fname = f"{year}-{slam}-matches.csv"

        try:
            df_m = pd.read_csv(SLAM_BASE + matches_fname)
            df_m["year"] = year
            df_m["slam"] = slam
            slam_matches_list.append(df_m)
            print(f"Loaded Slam matches {matches_fname}: shape={df_m.shape}")
        except Exception as e:
            print(f"Skipping Slam matches {matches_fname}: {e}")

        try:
            df_p = pd.read_csv(SLAM_BASE + points_fname)
            df_p["year"] = year
            df_p["slam"] = slam
            slam_points_list.append(df_p)
            print(f"Loaded Slam points {points_fname}: shape={df_p.shape}")
        except Exception as e:
            print(f"Skipping Slam points {points_fname}: {e}")

slam_matches_all = pd.concat(slam_matches_list, ignore_index=True)
slam_points_all  = pd.concat(slam_points_list,  ignore_index=True)

# Assigning surface by tournament
slam_matches_all["surface"] = slam_matches_all["slam"].map({
    "ausopen": "Hard",
    "frenchopen": "Clay",
    "wimbledon": "Grass",
    "usopen": "Hard"
})

print("\nTotal Slam matches:", slam_matches_all.shape)
print("Total Slam points:", slam_points_all.shape)
print("Slam matches surface distribution:", slam_matches_all["surface"].value_counts())

# Cleaning Slam points minimal columns
dfp_slam = slam_points_all.copy()

for col in ["match_id", "PointServer", "PointWinner", "P1Score", "P2Score"]:
    if col not in dfp_slam.columns:
        print(f"WARNING: Slam points missing column {col}")
    else:
        dfp_slam = dfp_slam[dfp_slam[col].notna()]

# Minimal matches
match_id_col = "match_id"
p1_col       = "player1"
p2_col       = "player2"
surface_col  = "surface"

for c in [match_id_col, p1_col, p2_col]:
    if c not in slam_matches_all.columns:
        print(f"WARNING: Slam matches missing column {c}")

slam_matches_min = slam_matches_all[[match_id_col, p1_col, p2_col, surface_col, "year", "slam"]].copy()

# Merge
df_slam = dfp_slam.merge(slam_matches_min, on="match_id", how="left")

df_slam = df_slam.rename(columns={
    p1_col: "p1_name",
    p2_col: "p2_name",
    surface_col: "surface"
})

df_slam["PointServer"] = pd.to_numeric(df_slam["PointServer"], errors="coerce")
df_slam["PointWinner"] = pd.to_numeric(df_slam["PointWinner"], errors="coerce")

df_slam["server_name"] = np.where(df_slam["PointServer"] == 1, df_slam["p1_name"], df_slam["p2_name"])
df_slam["returner_name"] = np.where(df_slam["PointServer"] == 1, df_slam["p2_name"], df_slam["p1_name"])

def slam_server_point_won(row):
    if row["PointWinner"] == row["PointServer"]:
        return 1
    elif row["PointWinner"] in [1,2]:
        return 0
    else:
        return np.nan

df_slam["point_won"] = df_slam.apply(slam_server_point_won, axis=1)
df_slam = df_slam.dropna(subset=["point_won"])
df_slam["point_won"] = df_slam["point_won"].astype(int)

# Speed and speed_bucket
if "Speed_KMH" in df_slam.columns:
    df_slam["serve_speed"] = pd.to_numeric(df_slam["Speed_KMH"], errors="coerce")
else:
    df_slam["serve_speed"] = np.nan

def speed_to_bucket(v):
    if pd.isna(v) or v == 0:
        return "Unknown"
    v = float(v)
    if v >= 195:
        return "Fast"
    elif v >= 175:
        return "Medium"
    else:
        return "Slow"

df_slam["speed_bucket"] = df_slam["serve_speed"].apply(speed_to_bucket)

# Serve number
if "ServeNumber" in df_slam.columns:
    df_slam["serve_number"] = pd.to_numeric(df_slam["ServeNumber"], errors="coerce").fillna(1).astype(int)
else:
    df_slam["serve_number"] = 1

# Score string + bucket (server perspective)
def slam_build_score_str(row):
    s1 = str(row["P1Score"]).strip()
    s2 = str(row["P2Score"]).strip()
    if row["PointServer"] == 1:
        return f"{s1}-{s2}"
    else:
        return f"{s2}-{s1}"

df_slam["score_str"] = df_slam.apply(slam_build_score_str, axis=1)

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

df_slam["score_bucket"] = df_slam["score_str"].apply(score_to_bucket)

# Attach styles for Slam as well
slam_server_styles = player_features.rename(columns={
    "player": "server_name",
    "style_label": "server_type"
})[["server_name", "server_type"]]

slam_opp_styles = player_features.rename(columns={
    "player": "opponent_name",
    "style_label": "opponent_type"
})[["opponent_name", "opponent_type"]]

df_slam = df_slam.merge(slam_server_styles, on="server_name", how="left")
df_slam = df_slam.merge(
    slam_opp_styles,
    left_on="returner_name",
    right_on="opponent_name",
    how="left"
).drop(columns=["opponent_name"])

df_slam["server_type"]   = df_slam["server_type"].fillna("Unknown")
df_slam["opponent_type"] = df_slam["opponent_type"].fillna("Unknown")
df_slam["surface"]       = df_slam["surface"].astype(str)

# No reliable serve_side, so set constant
df_slam["serve_side"] = "Unknown"

# Modeling df for Slam speed/score model
df_slam_model = df_slam.dropna(subset=["point_won"]).copy()

df_slam_model["score_bucket"] = df_slam_model["score_bucket"].astype(str)
df_slam_model["speed_bucket"] = df_slam_model["speed_bucket"].astype(str)

print("\nSlam modeling df shape:", df_slam_model.shape)

#%% 4. Train Slam Global Model (P(point_won | surface, score, speed, type))

slam_feature_cols = [
    "surface",
    "score_bucket",
    "speed_bucket",
    "server_type",
    "opponent_type",
    "serve_number",
]

X_slam = df_slam_model[slam_feature_cols]
y_slam = df_slam_model["point_won"]

preprocess_slam = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["surface", "score_bucket", "speed_bucket",
          "server_type", "opponent_type"]),
    ],
    remainder="passthrough"  # keep serve_number numeric
)

global_slam_model = Pipeline(steps=[
    ("preprocess", preprocess_slam),
    ("clf", LogisticRegression(max_iter=500)),
])

if len(df_slam_model) < 50:
    print("\nWARNING: Slam data too small, training on full without split.")
    X_train_slam = X_slam
    y_train_slam = y_slam
    X_test_slam  = X_slam
    y_test_slam  = y_slam
else:
    X_train_slam, X_test_slam, y_train_slam, y_test_slam = train_test_split(
        X_slam, y_slam, test_size=0.2, random_state=42, stratify=y_slam
    )

print("\nFitting global_slam_model...")
global_slam_model.fit(X_train_slam, y_train_slam)

y_pred_slam  = global_slam_model.predict(X_test_slam)
y_proba_slam = global_slam_model.predict_proba(X_test_slam)[:, 1]

print("\nSlam global model Accuracy:", accuracy_score(y_test_slam, y_pred_slam))
print("Slam global model ROC AUC:", roc_auc_score(y_test_slam, y_proba_slam))
print("\nSlam global model classification report:")
print(classification_report(y_test_slam, y_pred_slam))

#%% 5. Estimator: MCP Direction + SLAM SPEED/SCORE

def get_style_label(player_name: str, default_type: str = "Unknown") -> str:
    return player_style_map.get(player_name, default_type)

def estimate_direction_prob_MCP(
    player_name: str,
    opponent_name: Optional[str],
    surface: str,
    server_type: str,
    opponent_type: str,
    serve_direction: str,
    min_h2h_points: int = 20,
    min_player_points: int = 40,
    min_type_points: int = 80,
):

    # 1) H2H layer
    if opponent_name is not None:
        mask_h2h = (
            (mcp_h2h_dir_stats["server_name"]   == player_name) &
            (mcp_h2h_dir_stats["returner_name"] == opponent_name) &
            (mcp_h2h_dir_stats["surface"]       == surface) &
            (mcp_h2h_dir_stats["serve_direction"] == serve_direction)
        )
        s_h2h = mcp_h2h_dir_stats[mask_h2h]
        if not s_h2h.empty and s_h2h["n_points"].iloc[0] >= min_h2h_points:
            return float(s_h2h["win_prob"].iloc[0]), "MCP_H2H", int(s_h2h["n_points"].iloc[0])

    # 2) Player vs opponent_type
    mask_player = (
        (mcp_player_dir_stats["server_name"]    == player_name) &
        (mcp_player_dir_stats["surface"]        == surface) &
        (mcp_player_dir_stats["opponent_type"]  == opponent_type) &
        (mcp_player_dir_stats["serve_direction"] == serve_direction)
    )
    s_player = mcp_player_dir_stats[mask_player]
    if not s_player.empty and s_player["n_points"].iloc[0] >= min_player_points:
        return float(s_player["win_prob"].iloc[0]), "MCP_player_vs_style", int(s_player["n_points"].iloc[0])

    # 3) Style vs style
    mask_type = (
        (mcp_type_dir_stats["server_type"]      == server_type) &
        (mcp_type_dir_stats["surface"]          == surface) &
        (mcp_type_dir_stats["opponent_type"]    == opponent_type) &
        (mcp_type_dir_stats["serve_direction"]  == serve_direction)
    )
    s_type = mcp_type_dir_stats[mask_type]
    if not s_type.empty and s_type["n_points"].iloc[0] >= min_type_points:
        return float(s_type["win_prob"].iloc[0]), "MCP_style_vs_style", int(s_type["n_points"].iloc[0])

    # No directional info → let caller know
    return None, "MCP_no_data", 0

def estimate_slam_speed_score_prob(
    surface: str,
    score_str: str,
    speed_bucket: str,
    server_type: str,
    opponent_type: str,
    serve_number: int,
):
    """
    Use Slam global model to estimate P(point won) given surface, score, speed,
    server_type, opponent_type, serve_number. Directionless.
    """
    score_bucket = score_to_bucket(score_str)
    row = pd.DataFrame([{
        "surface": surface,
        "score_bucket": score_bucket,
        "speed_bucket": speed_bucket,
        "server_type": server_type,
        "opponent_type": opponent_type,
        "serve_number": serve_number,
    }])
    prob = float(global_slam_model.predict_proba(row)[0][1])
    return prob, score_bucket

def estimate_win_prob_for_option(
    player_name: str,
    player_type: str,
    surface: str,
    score_str: str,
    serve_side: str,   #currently informational - possibly to include
    serve_number: int,
    serve_direction: str,
    speed_bucket: str,
    opponent_name: Optional[str] = None,
    opponent_type: Optional[str] = None,
    alpha_direction_weight: float = 0.6,
):
    """
    Combined estimator:

      P_combined = alpha * P_dir_MCP + (1 - alpha) * P_slam_global

    if MCP has no directional data, falls back to P_slam_global only.
    """
    # Styles
    server_type = get_style_label(player_name, default_type=player_type)
    if opponent_name is not None and opponent_name in player_style_map:
        opp_type_label = player_style_map[opponent_name]
    elif opponent_type is not None:
        opp_type_label = opponent_type
    else:
        opp_type_label = "Unknown"

    # Slam speed + score estimate
    p_slam, score_bucket = estimate_slam_speed_score_prob(
        surface=surface,
        score_str=score_str,
        speed_bucket=speed_bucket,
        server_type=server_type,
        opponent_type=opp_type_label,
        serve_number=serve_number,
    )

    # MCP direction-specific estimate
    p_dir, mcp_source, n_dir = estimate_direction_prob_MCP(
        player_name=player_name,
        opponent_name=opponent_name,
        surface=surface,
        server_type=server_type,
        opponent_type=opp_type_label,
        serve_direction=serve_direction,
    )

    if p_dir is None:
        # No directional info; pure Slam-based
        return p_slam, f"Slam_only({mcp_source})", 0, server_type, opp_type_label, score_bucket

    # Blend MCP directional signal with Slam speed+score baseline
    p_combined = alpha_direction_weight * p_dir + (1 - alpha_direction_weight) * p_slam

    # Clip to [0.01, 0.99] for interpretability
    p_combined = max(0.01, min(0.99, p_combined))

    combined_source = f"Integrated(MCP:{mcp_source}, Slam_speed_score)"
    return p_combined, combined_source, n_dir, server_type, opp_type_label, score_bucket

#%% 6. Recommending serve options (opponent-specific)

def recommend_serve_options(
    player_name: str,
    player_type: str,
    surface: str,
    score_str: str,
    serve_side: str,
    serve_number: int,
    opponent_name: Optional[str] = None,
    opponent_type: Optional[str] = None,
    directions: List[str] = ("Wide", "T", "Body"),
    speeds: List[str] = ("Fast", "Medium", "Slow"),
) -> List[Dict[str, Any]]:
    """
    Evaluate all (direction, speed) options and return them sorted
    by predicted P(point won).

    Using integrated MCP+Slam estimator.
    """
    results = []
    for d in directions:
        for s in speeds:
            prob, source, n_dir, server_type, opp_type_label, score_bucket = estimate_win_prob_for_option(
                player_name=player_name,
                player_type=player_type,
                surface=surface,
                score_str=score_str,
                serve_side=serve_side,
                serve_number=serve_number,
                serve_direction=d,
                speed_bucket=s,
                opponent_name=opponent_name,
                opponent_type=opponent_type,
            )
            results.append({
                "player_name": player_name,
                "server_type": server_type,
                "surface": surface,
                "opponent_name": opponent_name,
                "opponent_type": opp_type_label,
                "user_serve_side": serve_side,
                "serve_number": serve_number,
                "score_str": score_str,
                "score_bucket": score_bucket,
                "serve_direction": d,
                "speed_bucket": s,
                "prob_win_point": prob,
                "data_source": source,
                "n_points_direction_MCP": n_dir,
            })

    results_sorted = sorted(results, key=lambda x: x["prob_win_point"], reverse=True)
    return results_sorted

#%% 7. Heatmap plot

def plot_serve_heatmap(options_list: List[Dict[str, Any]], title: Optional[str] = None):
    """
    Plot heatmap: rows = serve_direction, columns = speed_bucket, values = P(point won).
    """
    df_opts = pd.DataFrame(options_list)
    pivot = df_opts.pivot(index="serve_direction", columns="speed_bucket", values="prob_win_point")

    # Title for the heatmap, based on the features
    if title is None and len(df_opts) > 0:
        player       = df_opts["player_name"].iloc[0]
        opponent     = df_opts["opponent_name"].iloc[0]
        surface      = df_opts["surface"].iloc[0]
        serve_side   = df_opts["user_serve_side"].iloc[0]
        score_str    = df_opts["score_str"].iloc[0]
        serve_number = df_opts["serve_number"].iloc[0]

        suffix = "1st serve" if serve_number == 1 else f"{serve_number}nd serve"
        title = f"{player} vs {opponent} – {surface}, {serve_side} side, score {score_str}, {suffix}"

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot, aspect="auto")

    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.colorbar(im, label="P(point won)")

    plt.title(title)
    plt.xlabel("Serve Speed Bucket")
    plt.ylabel("Serve Direction")

    # annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.show()

#%% 8. Example Use Case – Sinner vs Djokovic

example_options = recommend_serve_options(
    player_name="Jannik Sinner",
    player_type="Baseline Grinder",   # fallback style if not in clusters
    surface="Hard",
    score_str="30-40",
    serve_side="Ad",
    serve_number=1,
    opponent_name="Novak Djokovic",   # specific opponent
    opponent_type=None,               # fill this in if no data for opponent
)

print("\nServe options ranked (best to worst):")
for opt in example_options:
    print(
        f"{opt['serve_direction']:>4} / {opt['speed_bucket']:>6} → "
        f"P(win)={opt['prob_win_point']:.3f}  "
        f"[{opt['data_source']}, MCP_n={opt['n_points_direction_MCP']}]"
    )

best = example_options[0]
print("\nRecommended pattern dict:")
print(best)

plot_serve_heatmap(example_options)

#%% 9. Exporting data for PowerBI Dashboard (Selected 5 Players)

print("\nBuilding Power BI export table (Selected 5 players)...")


# 1. Fixed player list (to reduce run time)

players_to_export = [
    "Jannik Sinner",
    "Carlos Alcaraz",
    "Alexander Zverev",
    "Novak Djokovic",
    "Lorenzo Musetti",
]

# keep only those actually present in MCP direction dataset
players_to_export = [
    p for p in players_to_export
    if p in mcp_points_dir["server_name"].unique()
]

print("Players included:")
for p in players_to_export:
    print(" -", p)


# 2. Export grid definition

surfaces_to_export = ["Hard", "Clay", "Grass"]
serve_numbers_to_export = [1, 2]

score_bucket_to_example_score = {
    "NeutralEven": "0-0",
    "NeutralOther": "15-0",
    "Deuce": "40-40",
    "BreakPoint": "30-40",
    "GamePointServer": "40-30",
}

directions = ["Wide", "T", "Body"]
speeds = ["Fast", "Medium", "Slow"]

max_opponents_per_player = 40   # adjust if more computational power available

rows = []

# 3. Build predictions

for player in players_to_export:

    player_type = get_style_label(player, default_type="Unknown")

    # choose most common opponents actually faced in MCP charted data
    opps = (
        mcp_points_dir[mcp_points_dir["server_name"] == player]["returner_name"]
        .value_counts()
        .head(max_opponents_per_player)
        .index
        .tolist()
    )

    print(f"\nProcessing {player} vs {len(opps)} opponents...")

    for opp in opps:
        for surface in surfaces_to_export:
            for serve_number in serve_numbers_to_export:
                for score_bucket, score_str in score_bucket_to_example_score.items():

                    opts = recommend_serve_options(
                        player_name=player,
                        player_type=player_type,
                        surface=surface,
                        score_str=score_str,
                        serve_side="Unknown",
                        serve_number=serve_number,
                        opponent_name=opp,
                        opponent_type=None,
                        directions=directions,
                        speeds=speeds
                    )

                    for r in opts:
                        rows.append({
                            "player_name": r["player_name"],
                            "player_type": r["server_type"],
                            "opponent_name": r["opponent_name"],
                            "opponent_type": r["opponent_type"],
                            "surface": r["surface"],
                            "serve_number": r["serve_number"],
                            "score_bucket": score_bucket,
                            "serve_direction": r["serve_direction"],
                            "speed_bucket": r["speed_bucket"],
                            "prob_win_point": round(r["prob_win_point"], 4),
                            "data_source": r["data_source"],
                            "n_points_direction_MCP": r["n_points_direction_MCP"],
                        })


# 4. Save CSV

export_df = pd.DataFrame(rows)

out_file = "serve_predictions_selected5_powerbi.csv"
export_df.to_csv(out_file, index=False)

print("Export complete")
print("File:", out_file)
print("Rows:", len(export_df))
print("Columns:", export_df.columns.tolist())
print(export_df.head())

#%% 10. Evaluation Metrics

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

print("==============================")
print("EVALUATION: MCP DIRECTION MODEL")
print("==============================")

# 1) Train/test split on MCP direction points (these are the points used to build mcp_*_dir_stats)
mcp_eval = mcp_points_dir.dropna(subset=["server_name", "returner_name", "surface", "serve_direction", "point_won"]).copy()

if len(mcp_eval) < 5000:
    print(f"Warning: MCP eval dataset is small (n={len(mcp_eval)}). Metrics may be noisy.")

train_df, test_df = train_test_split(
    mcp_eval,
    test_size=0.2,
    random_state=42,
    stratify=mcp_eval["point_won"]
)

# 2) Rebuild stats tables on TRAIN only (prevents leakage)
mcp_h2h_dir_stats_train = (
    train_df
    .groupby(["server_name", "returner_name", "surface", "server_type", "opponent_type", "serve_direction"])["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count":"n_points", "mean":"win_prob"})
)

mcp_player_dir_stats_train = (
    train_df
    .groupby(["server_name", "surface", "server_type", "opponent_type", "serve_direction"])["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count":"n_points", "mean":"win_prob"})
)

mcp_type_dir_stats_train = (
    train_df
    .groupby(["server_type", "surface", "opponent_type", "serve_direction"])["point_won"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count":"n_points", "mean":"win_prob"})
)

# 3) Predictor using the SAME hierarchy logic (but using TRAIN tables)
def predict_mcp_direction_prob(row, min_h2h=20, min_player=40, min_type=80):
    server = row["server_name"]
    ret    = row["returner_name"]
    surf   = row["surface"]
    d      = row["serve_direction"]

    # styles (already in mcp_points_dir as server_type/opponent_type)
    stype  = row.get("server_type", "Unknown")
    otype  = row.get("opponent_type", "Unknown")

    # 1) H2H
    m = (
        (mcp_h2h_dir_stats_train["server_name"] == server) &
        (mcp_h2h_dir_stats_train["returner_name"] == ret) &
        (mcp_h2h_dir_stats_train["surface"] == surf) &
        (mcp_h2h_dir_stats_train["serve_direction"] == d)
    )
    s = mcp_h2h_dir_stats_train[m]
    if not s.empty and int(s["n_points"].iloc[0]) >= min_h2h:
        return float(s["win_prob"].iloc[0]), "MCP_H2H"

    # 2) player vs opponent style
    m = (
        (mcp_player_dir_stats_train["server_name"] == server) &
        (mcp_player_dir_stats_train["surface"] == surf) &
        (mcp_player_dir_stats_train["opponent_type"] == otype) &
        (mcp_player_dir_stats_train["serve_direction"] == d)
    )
    s = mcp_player_dir_stats_train[m]
    if not s.empty and int(s["n_points"].iloc[0]) >= min_player:
        return float(s["win_prob"].iloc[0]), "MCP_player_vs_style"

    # 3) style vs style
    m = (
        (mcp_type_dir_stats_train["server_type"] == stype) &
        (mcp_type_dir_stats_train["surface"] == surf) &
        (mcp_type_dir_stats_train["opponent_type"] == otype) &
        (mcp_type_dir_stats_train["serve_direction"] == d)
    )
    s = mcp_type_dir_stats_train[m]
    if not s.empty and int(s["n_points"].iloc[0]) >= min_type:
        return float(s["win_prob"].iloc[0]), "MCP_style_vs_style"

    # 4) fallback: global mean from TRAIN (uninformative baseline)
    return float(train_df["point_won"].mean()), "MCP_global_mean_fallback"

# 4) Predict on TEST
pred_probs = []
pred_sources = []

for _, r in test_df.iterrows():
    p, src = predict_mcp_direction_prob(r)
    # keep probabilities away from 0/1 for log loss stability
    p = max(1e-4, min(1 - 1e-4, p))
    pred_probs.append(p)
    pred_sources.append(src)

y_true = test_df["point_won"].values
y_prob = np.array(pred_probs)
y_hat  = (y_prob >= 0.5).astype(int)

# 5) Metrics
try:
    auc = roc_auc_score(y_true, y_prob)
except Exception:
    auc = np.nan

acc = accuracy_score(y_true, y_hat)
brier = brier_score_loss(y_true, y_prob)
ll = log_loss(y_true, y_prob)

print(f"TEST size: {len(test_df):,}")
print(f"Accuracy (threshold 0.50): {acc:.3f}")
print(f"ROC AUC: {auc:.3f}")
print(f"Brier score (lower is better): {brier:.4f}")
print(f"Log loss (lower is better): {ll:.4f}")

# Source breakdown (how often each hierarchy level was used)
src_counts = pd.Series(pred_sources).value_counts(normalize=True).round(3)
print("Prediction source usage (% of test points):")
print(src_counts)

print("==============================")
print("NOTE: Your Slam model accuracy/AUC printed earlier evaluates")
print("      speed+score+surface+style prediction on Slam points.")
print("      This cell evaluates the MCP direction hierarchy on MCP points.")
print("==============================")


