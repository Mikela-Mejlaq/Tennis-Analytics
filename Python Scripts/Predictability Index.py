#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mikelamejlaq
"""

"""
Predictability Index (Serve Direction) from Match Charting Project (MCP)

Interpretation:
  Predictability Index (0-100) where higher = more predictable direction choice.
"""

# importing packages
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import chisquare

#%% 0) Settings
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"

MCP_POINTS_FILES = [
    MCP_BASE + "charting-m-points-to-2009.csv",
    MCP_BASE + "charting-m-points-2010s.csv",
    MCP_BASE + "charting-m-points-2020s.csv",
]

MCP_MATCHES_URL = MCP_BASE + "charting-m-matches.csv"

# Minimum sample size for a context row to be considered "actionable"
MIN_POINTS = 40

# Output paths
OUT_PI_CONTEXT_LONG = "pi_context_long.csv"
OUT_PI_RADAR = "pi_radar.csv"
OUT_PI_DIRECTION_COUNTS = "pi_direction_counts.csv"

#%% 1) Load data
def load_mcp_matches() -> pd.DataFrame:
    m = pd.read_csv(MCP_MATCHES_URL)

    # Standardise expected columns
    m_std = m.rename(columns={
        "match_id": "match_id",
        "Player 1": "p1_name",
        "Player 2": "p2_name",
        "Surface": "surface"
    })[["match_id", "p1_name", "p2_name", "surface"]]

    return m_std


def load_mcp_points() -> pd.DataFrame:
    dfs = []
    for url in MCP_POINTS_FILES:
        try:
            df = pd.read_csv(url)
            dfs.append(df)
            print(f"Loaded: {url}  shape={df.shape}")
        except Exception as e:
            print(f"WARNING: failed to load {url}: {e}")

    if not dfs:
        raise RuntimeError("No MCP points files could be loaded.")

    points_all = pd.concat(dfs, ignore_index=True)
    print(f"Total MCP points: {points_all.shape}")
    return points_all

#%% 2) Feature engineering
def parse_1st_serve_direction(code: str) -> str:
    """
    MCP '1st' serve coding:
      first digit 4 = wide
      first digit 5 = body
      first digit 6 = T
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


def build_mcp_points_dir(mcp_points_all: pd.DataFrame, mcp_matches_std: pd.DataFrame) -> pd.DataFrame:
    # Merge match info
    df = mcp_points_all.merge(mcp_matches_std, on="match_id", how="left")

    # Standardise point winner column
    df = df.rename(columns={"PtWinner": "point_winner_raw"})

    # Map server/returner using Svr flag
    df["server_name"] = np.where(df["Svr"] == 1, df["p1_name"], df["p2_name"])
    df["returner_name"] = np.where(df["Svr"] == 1, df["p2_name"], df["p1_name"])

    # Outcome from server perspective (1 if server won the point)
    def server_point_won(row):
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

    # Serve direction (1st serve only coding)
    df["serve_direction"] = df["1st"].apply(parse_1st_serve_direction)

    # Filter to meaningful directions
    df = df[df["serve_direction"].isin(["Wide", "Body", "T"])].copy()

    # Ensure surface is string
    df["surface"] = df["surface"].astype(str)

    print(f"MCP points with direction: {df.shape}")
    print("Direction distribution:")
    print(df["serve_direction"].value_counts())

    return df

#%% 3) Predictability calculation

def shannon_entropy_from_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return float(-(probs * np.log2(probs)).sum())


def compute_pi_from_counts(counts: pd.Series) -> dict:
    """
    counts index: ["Wide","Body","T"]
    Returns PI metrics + top tendency.
    """
    total = int(counts.sum())
    if total == 0:
        return {
            "n_points": 0,
            "entropy_bits": np.nan,
            "entropy_norm": np.nan,
            "predictability_index": np.nan,
            "top_choice": None,
            "top_prob": np.nan,
            "second_choice": None,
            "second_prob": np.nan,
            "exploit_gap": np.nan,
        }

    probs = (counts / total).sort_values(ascending=False)

    K = 3
    H = shannon_entropy_from_probs(probs.values)
    Hmax = np.log2(K)
    Hnorm = H / Hmax if Hmax > 0 else np.nan
    PI = (1 - Hnorm) * 100

    top_choice = probs.index[0]
    top_prob = float(probs.iloc[0])

    if len(probs) > 1:
        second_choice = probs.index[1]
        second_prob = float(probs.iloc[1])
        gap = top_prob - second_prob
    else:
        second_choice, second_prob, gap = None, np.nan, np.nan

    return {
        "n_points": total,
        "entropy_bits": float(H),
        "entropy_norm": float(Hnorm),
        "predictability_index": float(PI),
        "top_choice": str(top_choice),
        "top_prob": float(top_prob),
        "second_choice": None if second_choice is None else str(second_choice),
        "second_prob": float(second_prob) if not np.isnan(second_prob) else np.nan,
        "exploit_gap": float(gap) if not np.isnan(gap) else np.nan,
    }

#%% 4) Build PI tables

def context_pi_table(df: pd.DataFrame, group_cols: list, context_label: str, min_points: int) -> pd.DataFrame:
    allowed = ["Wide", "Body", "T"]
    temp = df.copy()
    temp["serve_direction"] = pd.Categorical(temp["serve_direction"], categories=allowed)

    # Count directions per context
    counts = (
        temp.groupby(group_cols + ["serve_direction"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
    )

    # Ensure all columns exist
    for c in allowed:
        if c not in counts.columns:
            counts[c] = 0

    rows = []
    for _, r in counts.iterrows():
        cts = pd.Series({k: int(r[k]) for k in allowed})
        metrics = compute_pi_from_counts(cts)

        if metrics["n_points"] < min_points:
            continue

        out = {col: r[col] for col in group_cols}
        out.update({
            "context": context_label,
            "p_wide": cts["Wide"] / metrics["n_points"],
            "p_body": cts["Body"] / metrics["n_points"],
            "p_t": cts["T"] / metrics["n_points"],
        })
        out.update(metrics)
        rows.append(out)

    return pd.DataFrame(rows)


def build_predictability_outputs(mcp_points_dir: pd.DataFrame, min_points: int = 40):
    # Long-form contexts
    pi_overall = context_pi_table(
        mcp_points_dir, ["server_name"], "Overall", min_points
    )

    pi_surface = context_pi_table(
        mcp_points_dir, ["server_name", "surface"], "On Surface", min_points
    )

    pi_vs_opp = context_pi_table(
        mcp_points_dir, ["server_name", "returner_name"], "Vs Opponent", min_points
    )

    pi_vs_opp_surface = context_pi_table(
        mcp_points_dir, ["server_name", "returner_name", "surface"], "Vs Opponent On Surface", min_points
    )

    pi_context_long = pd.concat(
        [pi_overall, pi_surface, pi_vs_opp, pi_vs_opp_surface],
        ignore_index=True
    )

    pi_radar = pi_context_long.copy()
    # Make sure these columns exist so Power BI slicers don’t break
    # summarises serve-direction behaviour for a given context
    if "returner_name" not in pi_radar.columns:
        pi_radar["returner_name"] = None
    if "surface" not in pi_radar.columns:
        pi_radar["surface"] = None

    pi_radar = pi_radar.rename(columns={
        "context": "axis",
        "predictability_index": "value"
    })[[
        "server_name", "returner_name", "surface",
        "axis", "value", "n_points",
        "top_choice", "top_prob", "second_choice", "second_prob", "exploit_gap"
    ]]

    # Direction probabilities table (this will be used for a stacked bar chart in the Power BI dashboard)
    pi_direction_counts = pi_context_long[[
        "server_name",
        "surface" if "surface" in pi_context_long.columns else None,
        "returner_name" if "returner_name" in pi_context_long.columns else None,
        "context",
        "n_points",
        "p_wide", "p_body", "p_t"
    ]].copy()

    # Drop the None columns if they occurred from the conditional selection above
    pi_direction_counts = pi_direction_counts.loc[:, [c for c in pi_direction_counts.columns if c is not None]]

    return pi_context_long, pi_radar, pi_direction_counts

#%% 5) Main run

def main():
    print("Loading MCP matches...")
    mcp_matches_std = load_mcp_matches()

    print("Loading MCP points...")
    mcp_points_all = load_mcp_points()

    print("Building MCP points with serve direction...")
    mcp_points_dir = build_mcp_points_dir(mcp_points_all, mcp_matches_std)

    print("Computing predictability outputs...")
    pi_context_long, pi_radar, pi_direction_counts = build_predictability_outputs(
        mcp_points_dir, min_points=MIN_POINTS
    )
    

    print("Saving outputs to CSV...")
    pi_context_long.to_csv(OUT_PI_CONTEXT_LONG, index=False)
    pi_radar.to_csv(OUT_PI_RADAR, index=False)
    pi_direction_counts.to_csv(OUT_PI_DIRECTION_COUNTS, index=False)

    print("Done.")
    print("\nFiles created:")
    print(f" - {OUT_PI_CONTEXT_LONG}  (Power BI main table)")
    print(f" - {OUT_PI_RADAR}         (Radar chart ready)")
    print(f" - {OUT_PI_DIRECTION_COUNTS} (Direction distribution)")


if __name__ == "__main__":
    main()
    

#%% Helper Function

def print_predictability_scenario(
    pi_context_long: pd.DataFrame,
    server_name: str,
    returner_name: str = None,
    surface: str = None,
    context: str = None
):
    """
    Print a single Predictability Index scenario
    """

    df = pi_context_long.copy()

    # Apply filters step by step
    df = df[df["server_name"] == server_name]

    if returner_name is not None and "returner_name" in df.columns:
        df = df[df["returner_name"] == returner_name]

    if surface is not None and "surface" in df.columns:
        df = df[df["surface"] == surface]

    if context is not None:
        df = df[df["context"] == context]

    if df.empty:
        print("\nNo data available for this scenario.")
        return

    # If multiple rows remain, take the one with most data
    row = df.sort_values("n_points", ascending=False).iloc[0]

    print("================ PREDICTABILITY SCENARIO ================")
    print(f"Server      : {row['server_name']}")
    if 'returner_name' in row and pd.notna(row['returner_name']):
        print(f"Returner    : {row['returner_name']}")
    if 'surface' in row and pd.notna(row['surface']):
        print(f"Surface     : {row['surface']}")
    print(f"Context     : {row['context']}")
    print("---------------------------------------------------------")
    print(f"Points used : {int(row['n_points'])}")
    print(f"Predictability Index : {row['predictability_index']:.1f} / 100")
    print(f"Entropy (norm)       : {row['entropy_norm']:.2f}")
    print("")
    print("Serve direction probabilities:")
    print(f"  Wide : {row['p_wide']:.1%}")
    print(f"  Body : {row['p_body']:.1%}")
    print(f"  T    : {row['p_t']:.1%}")
    print("")
    print("Exploit insight:")
    print(f"  Top choice   : {row['top_choice']} ({row['top_prob']:.1%})")
    if pd.notna(row['second_choice']):
        print(f"  Second       : {row['second_choice']} ({row['second_prob']:.1%})")
        print(f"  Gap          : {row['exploit_gap']:.1%}")
    print("=========================================================\n")
    
#%% Example for Sinner against Djokovic

if __name__ == "__main__":
    pi_context_long = pd.read_csv("pi_context_long.csv")

    print_predictability_scenario(
        pi_context_long=pi_context_long,
        server_name="Jannik Sinner",
        returner_name="Novak Djokovic",
        surface="Hard",
        context="Vs Opponent On Surface"
    )

#%% 6) Statistical Validation: Chi-Square Test

def test_direction_nonrandomness(
    mcp_points_dir: pd.DataFrame,
    server_name: str,
    returner_name: Optional[str] = None,
    surface: Optional[str] = None
):
    df = mcp_points_dir.copy()
    df = df[df["server_name"] == server_name]

    if returner_name is not None:
        df = df[df["returner_name"] == returner_name]

    if surface is not None:
        df = df[df["surface"] == surface]

    if df.empty:
        print("No data for this scenario.")
        return

    counts = (
        df["serve_direction"]
        .value_counts()
        .reindex(["Wide", "Body", "T"])
        .fillna(0)
        .astype(int)
    )

    observed = counts.values.astype(float) 
    total = observed.sum()


    expected = np.ones_like(observed, dtype=float) * (total / 3.0)

    chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    print("========== CHI-SQUARE VALIDATION ==========")
    print(f"Server: {server_name}")
    if returner_name:
        print(f"Returner: {returner_name}")
    if surface:
        print(f"Surface: {surface}")
    print("-------------------------------------------")
    print(f"Observed counts: {counts.to_dict()}")
    print(f"Total points: {int(total)}")
    print(f"Expected counts (uniform): {{'Wide': {expected[0]:.3f}, 'Body': {expected[1]:.3f}, 'T': {expected[2]:.3f}}}")
    print(f"Chi-square statistic: {chi_stat:.3f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Direction choice is statistically non-random (reject uniformity).")
    else:
        print("Result: Cannot reject uniform randomness.")
    print("===========================================")

    
#%% Running the validation
if __name__ == "__main__":

    print("Building data for validation...")

    mcp_matches_std = load_mcp_matches()
    mcp_points_all = load_mcp_points()
    mcp_points_dir = build_mcp_points_dir(mcp_points_all, mcp_matches_std)

    test_direction_nonrandomness(
        mcp_points_dir=mcp_points_dir,
        server_name="Jannik Sinner",
        returner_name="Novak Djokovic",
        surface="Hard"
    )