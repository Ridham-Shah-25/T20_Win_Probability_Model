"""Streamlit app: match selector + win-probability timeline.

Pick any model match and see the ball-by-ball win-probability arc from the
calibrated production model, on a fixed team perspective, with wicket and
boundary markers — the same trajectory the evaluation harness plots.

Run from the repo root:

    streamlit run app/streamlit_app.py

Requires the pipeline artifacts to exist (``data/processed/*.parquet`` and
``models/xgb_calibrated.joblib``); build them with ``scripts/run_ingest.py`` →
``run_features.py`` → ``run_eval.py`` if missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.models import load_model  # noqa: E402
from t20wp.wpa import build_match_trajectory  # noqa: E402

DATA = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "xgb_calibrated.joblib"
DEFAULT_MATCH_ID = "951373"  # 2016 Men's T20 World Cup final

st.set_page_config(page_title="T20 Win Probability", layout="wide")


@st.cache_data(show_spinner=False)
def load_frames():
    """Load the match table, feature rows and ball events (cached)."""
    matches = pd.read_parquet(DATA / "matches.parquet")
    matches = matches[matches["is_model_match"]].copy()
    features = pd.read_parquet(DATA / "features.parquet")
    balls = pd.read_parquet(
        DATA / "balls.parquet",
        columns=["match_id", "innings", "is_super_over",
                 "is_wicket", "runs_batter"],
    )
    return matches, features, balls


@st.cache_resource(show_spinner=False)
def load_calibrated_model():
    """Load the calibrated production model (cached across reruns)."""
    return load_model(MODEL_PATH)


def match_label(row) -> str:
    """Human-readable dropdown label for a match."""
    date = pd.to_datetime(row["date"]).date()
    comp = "IPL" if row["competition"] == "ipl" else "T20I"
    return (f"{date} · {comp} · {row['team1']} vs {row['team2']} "
            f"@ {row['venue']} (#{row['match_id']})")


def main() -> None:
    st.title("T20 Win Probability")
    st.caption(
        "Ball-by-ball P(win) from the calibrated XGBoost model, with the "
        "Win-Probability-Added (WPA) clutch metric behind it."
    )

    if not MODEL_PATH.exists() or not (DATA / "features.parquet").exists():
        st.error(
            "Pipeline artifacts missing. Run scripts/run_ingest.py, "
            "run_features.py and run_eval.py first."
        )
        return

    matches, features, balls = load_frames()
    model = load_calibrated_model()

    # --- sidebar filters ---
    st.sidebar.header("Pick a match")
    comps = {"All": None, "IPL": "ipl", "T20I": "t20i"}
    comp_choice = st.sidebar.selectbox("Competition", list(comps))
    pool = matches
    if comps[comp_choice] is not None:
        pool = pool[pool["competition"] == comps[comp_choice]]

    seasons = ["All"] + sorted(pool["season"].dropna().unique(), reverse=True)
    season_choice = st.sidebar.selectbox("Season", seasons)
    if season_choice != "All":
        pool = pool[pool["season"] == season_choice]

    pool = pool.sort_values("date", ascending=False).reset_index(drop=True)
    if pool.empty:
        st.warning("No matches for that filter.")
        return

    labels = {match_label(r): r["match_id"] for _, r in pool.iterrows()}
    # Default to the 2016 WT20 final if it is in the current pool.
    default_idx = next(
        (i for i, mid in enumerate(labels.values()) if mid == DEFAULT_MATCH_ID), 0
    )
    label = st.sidebar.selectbox("Match", list(labels), index=default_idx)
    match_id = labels[label]
    row = pool[pool["match_id"] == match_id].iloc[0]

    # --- perspective ---
    teams = [row["team1"], row["team2"]]
    team = st.sidebar.radio("Show win probability for", teams, index=0)

    # --- build + plot trajectory ---
    traj = build_match_trajectory(features, balls, match_id, model)
    if traj.empty:
        st.warning("No ball data for this match.")
        return

    # build_match_trajectory already returns rows sorted by (innings, ball_seq)
    # with a fresh 0..n-1 index, so no re-sort is needed here.
    df = traj
    df["wp_team"] = np.where(df["batting_team"] == team, df["wp"], 1.0 - df["wp"])
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, df["wp_team"], color="#1f77b4", lw=1.6, label=f"P({team} win)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    if (df["innings"] == 2).any():
        boundary = int((df["innings"] == 1).sum())
        ax.axvline(boundary - 0.5, color="black", lw=0.8, ls="--", alpha=0.6)
    wk = df[df["is_wicket"].fillna(False).astype(bool)]
    ax.scatter(wk.index, df.loc[wk.index, "wp_team"], color="red",
               marker="v", s=45, zorder=5, label="wicket")
    bd = df[df["runs_batter"].isin([4, 6])]
    ax.scatter(bd.index, df.loc[bd.index, "wp_team"], color="green",
               marker="^", s=35, zorder=4, label="4/6")
    ax.set_xlabel("Delivery (global index across both innings)")
    ax.set_ylabel(f"P({team} wins)")
    ax.set_ylim(0, 1)
    ax.set_title(match_label(row))
    ax.legend(loc="best")
    fig.tight_layout()

    winner = (str(row["winner"]) if pd.notna(row["winner"])
              and str(row["winner"]).strip() else "no result / tie")
    final_wp = float(df["wp_team"].iloc[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Final P({team} win)", f"{final_wp:.0%}")
    c2.metric("Actual winner", winner)
    c3.metric("Deliveries", len(df))

    st.pyplot(fig)

    with st.expander("Ball-by-ball table"):
        show = df[["innings", "ball_seq", "batting_team", "batter", "bowler",
                   "wp_team", "is_wicket", "runs_batter"]].rename(
            columns={"wp_team": f"P({team} win)"})
        st.dataframe(show, width="stretch", height=360)


if __name__ == "__main__":
    main()
