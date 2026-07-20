"""Streamlit app: win-probability explorer + what-if calculator.

Two modes:

* **Explore a match** — pick any model match and see the ball-by-ball
  win-probability arc from the calibrated production model, on a fixed team
  perspective, with wicket and boundary markers (the same trajectory the
  evaluation harness plots).
* **Win probability calculator** — enter a live match state (innings, score,
  wickets, balls, target) and read the calibrated model's P(batting team win)
  for that exact situation.

Run from the repo root:

    streamlit run app/streamlit_app.py

Requires the pipeline artifacts (``data/processed/*.parquet`` and
``models/xgb_calibrated.joblib``). These are gitignored and reproducible, so on
a fresh checkout / deploy they are downloaded on first launch from the GitHub
release named by ``T20WP_ARTIFACT_RELEASE`` (see ``ensure_artifacts``). To build
them locally instead, run ``scripts/run_ingest.py`` → ``run_features.py`` →
``run_eval.py``.
"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.features import (  # noqa: E402
    BALLS_PER_INNINGS, default_runs_last_24, scenario_features,
)
from t20wp.models import load_model, predict_win_prob  # noqa: E402
from t20wp.wpa import build_match_trajectory  # noqa: E402

DATA = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "xgb_calibrated.joblib"
DEFAULT_MATCH_ID = "951373"  # 2016 Men's T20 World Cup final
DEFAULT_VENUE_PAR = 155.0  # league-typical 1st-innings total; overridable

st.set_page_config(page_title="T20 Win Probability", layout="wide")

# On a fresh deploy the parquet/model files are absent (gitignored, reproducible).
# Missing files are fetched from this GitHub release; override via env var to
# point at your own fork's release.
ARTIFACT_RELEASE = os.environ.get(
    "T20WP_ARTIFACT_RELEASE",
    "https://github.com/Ridham-Shah-25/T20_Win_Probability_Model"
    "/releases/download/artifacts-v1",
)

# Runtime artifact filename -> local destination path.
ARTIFACTS = {
    "matches.parquet": DATA / "matches.parquet",
    "features.parquet": DATA / "features.parquet",
    "balls.parquet": DATA / "balls.parquet",
    "xgb_calibrated.joblib": MODEL_PATH,
}


def _ssl_context() -> ssl.SSLContext:
    """SSL context backed by certifi's CA bundle when available.

    Some Python builds (e.g. the python.org macOS framework) ship without a
    configured system trust store, so an explicit bundle avoids
    CERTIFICATE_VERIFY_FAILED. certifi ships as a Streamlit dependency.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


@st.cache_resource(show_spinner=False)
def ensure_artifacts() -> bool:
    """Download any missing runtime artifacts from the GitHub release.

    Returns True once every file is present locally. On download failure the
    resource cache is cleared so a later rerun retries rather than caching the
    failure, and False is returned so the caller can show guidance.
    """
    ctx = _ssl_context()
    for name, dest in ARTIFACTS.items():
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ARTIFACT_RELEASE}/{name}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with st.spinner(f"Downloading {name} (first launch only)…"):
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, context=ctx) as resp, \
                        open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            st.error(f"Could not download {name} from {url}\n\n{exc}")
            ensure_artifacts.clear()
            return False
    return True


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


def explore_match(matches, features, balls, model) -> None:
    """Match-explorer mode: ball-by-ball WP arc for a chosen match."""
    st.sidebar.header("Pick a match")
    comps = {"All": None, "IPL": "ipl", "T20I": "t20i"}
    comp_choice = st.sidebar.selectbox("Competition", list(comps))
    pool = matches
    if comps[comp_choice] is not None:
        pool = pool[pool["competition"] == comps[comp_choice]]

    # Calendar year (from match date) rather than the raw "season" label,
    # which for cross-new-year tours reads as e.g. "2025/26" — confusing
    # compared to a plain year.
    years = ["All"] + sorted(
        {int(y) for y in pool["date"].dt.year.dropna().unique()}, reverse=True
    )
    year_choice = st.sidebar.selectbox("Year", years)
    if year_choice != "All":
        pool = pool[pool["date"].dt.year == year_choice]

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

    # --- headline result ---
    winner = (str(row["winner"]) if pd.notna(row["winner"])
              and str(row["winner"]).strip() else "no result / tie")
    final_wp = float(df["wp_team"].iloc[-1])
    c1, c2 = st.columns(2)
    c1.metric(f"Final P({team} win)", f"{final_wp:.0%}")
    c2.metric("Actual winner", winner)

    # The chart marks the selected delivery, but the picker reads better
    # below the chart — reserve the slot and fill it once the pick is known.
    chart_slot = st.empty()

    # --- pick a delivery ---
    # Sliders rather than dropdowns: every value stays on screen instead of
    # hiding behind a popup, which got clipped in the narrow sidebar.
    st.divider()
    st.subheader("Jump to any point in the match")
    st.caption(
        "Choose an innings, over and ball to see the match situation and the "
        "model's win probability at that moment. The orange dot marks it on "
        "the chart above."
    )

    innings_present = sorted(int(i) for i in df["innings"].unique())
    inn_names = {
        i: (f"{'1st' if i == 1 else '2nd'} innings · "
            f"{df.loc[df['innings'] == i, 'batting_team'].iloc[0]} batting")
        for i in innings_present
    }
    inn_choice = st.radio("Innings", innings_present,
                          format_func=lambda i: inn_names[i], horizontal=True)
    df_inn = df[df["innings"] == inn_choice]

    pick_over, pick_ball = st.columns(2)
    overs_present = sorted(int(o) for o in df_inn["over"].unique())
    with pick_over:
        # `over` is 0-indexed in the data; show it as the 1-based over number.
        over_choice = st.select_slider(
            "Over", options=overs_present, format_func=lambda o: str(o + 1))
    balls_present = sorted(
        int(b) for b in
        df_inn.loc[df_inn["over"] == over_choice, "ball_in_over"].unique()
    )
    with pick_ball:
        # An over runs past 6 balls when it contains a wide or a no-ball.
        ball_choice = st.select_slider(
            "Ball in the over", options=balls_present, format_func=str)

    sel_row = df_inn[
        (df_inn["over"] == over_choice) & (df_inn["ball_in_over"] == ball_choice)
    ].iloc[0]
    # df's index is the global 0..n-1 delivery position, which is also the
    # chart's x-axis, so this locates the selection on the plot.
    sel_idx = sel_row.name

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
    ax.axvline(sel_idx, color="#ff7f0e", lw=0.8, ls=":", alpha=0.6)
    ax.scatter([sel_idx], [df.loc[sel_idx, "wp_team"]], color="#ff7f0e",
               marker="o", s=110, zorder=6, edgecolor="black", linewidth=0.8,
               label="selected ball")
    ax.set_xlabel("Delivery (global index across both innings)")
    ax.set_ylabel(f"P({team} wins)")
    ax.set_ylim(0, 1)
    ax.set_title(match_label(row))
    ax.legend(loc="best")
    fig.tight_layout()
    chart_slot.pyplot(fig)

    # --- situation at the selected delivery ---
    # score/wickets in the feature rows are the state *after* the delivery,
    # so this reads as the situation once the ball has been bowled.
    bat_team = sel_row["batting_team"]
    wickets_fallen = 10 - int(sel_row["wickets_in_hand"])
    # Overs come from legal balls bowled, not the picker's ball number: an
    # over containing a wide reaches ball 7 but still only 6 legal balls.
    legal_bowled = BALLS_PER_INNINGS - int(sel_row["balls_remaining"])
    overs_txt = f"{legal_bowled // 6}.{legal_bowled % 6}"

    st.markdown(f"#### Current situation — {bat_team} batting")
    s1, s2, s3 = st.columns(3)
    s1.metric("Score", f"{int(sel_row['score'])}/{wickets_fallen}")
    s2.metric("Overs bowled", overs_txt)
    s3.metric(f"P({team} win) here", f"{sel_row['wp_team']:.0%}")

    if int(sel_row["is_second_innings"]) == 1:
        st.markdown(
            f"**Chasing:** needs **{int(sel_row['runs_required'])}** runs from "
            f"**{int(sel_row['balls_remaining'])}** balls"
        )

    if pd.notna(sel_row["is_wicket"]) and bool(sel_row["is_wicket"]):
        outcome = "wicket falls"
    else:
        runs_off_bat = (int(sel_row["runs_batter"])
                        if pd.notna(sel_row["runs_batter"]) else 0)
        outcome = f"{runs_off_bat} off the bat"
    st.caption(
        f"On strike: {sel_row['batter']}  ·  Non-striker: "
        f"{sel_row['non_striker']}  ·  Bowler: {sel_row['bowler']}  ·  "
        f"This ball: {outcome}"
    )


def _default_venue_par(features) -> float:
    """League-median 1st-innings par, used as the calculator's default."""
    if features is not None and "venue_par" in features:
        med = features["venue_par"].median()
        if pd.notna(med):
            return round(float(med), 1)
    return DEFAULT_VENUE_PAR


def win_probability_calculator(features, model) -> None:
    """What-if mode: enter a live match state, read the model's P(win)."""
    st.subheader("Win probability calculator")
    st.caption(
        "Enter a live match state and read the calibrated model's "
        "P(batting team wins) for that exact situation."
    )

    par_default = _default_venue_par(features)

    st.sidebar.header("Match state")
    innings = st.sidebar.radio("Innings", [1, 2], index=1,
                               help="1 = setting a total, 2 = chasing.")
    comp = st.sidebar.selectbox("Competition", ["T20I", "IPL"])
    is_ipl = comp == "IPL"

    overs_completed = st.sidebar.slider("Completed overs", 0, 20, 10)
    balls_this_over = st.sidebar.slider(
        "Balls into current over", 0, 5, 0,
        help="Legal balls bowled = completed overs × 6 + this.")
    if overs_completed == 20:
        balls_this_over = 0  # innings is over after 20 completed overs
    legal_balls = min(overs_completed * 6 + balls_this_over, BALLS_PER_INNINGS)
    overs_done = f"{legal_balls // 6}.{legal_balls % 6}"  # cricket over.ball
    score = st.sidebar.number_input("Score (runs)", 0, 400, 90, step=1)
    wickets_fallen = st.sidebar.slider("Wickets fallen", 0, 10, 3)

    target = None
    if innings == 2:
        target = st.sidebar.number_input(
            "Target (runs to win)", 1, 400, 160, step=1,
            help="1st-innings total + 1.",
        )

    with st.sidebar.expander("Advanced context (defaults are league-typical)"):
        st.caption(
            "These are historical/contextual inputs the ball-by-ball pipeline "
            "computes from prior matches; they cannot be derived from the live "
            "state, so the defaults are sensible league values you can adjust."
        )
        venue_par = st.number_input(
            "Venue par (expected 1st-innings total)", 80, 260,
            int(round(par_default)), step=1,
        )
        batting_strength = st.slider(
            "Batting team strength (prior win rate)", 0.0, 1.0, 0.5, step=0.01)
        bowling_strength = st.slider(
            "Bowling team strength (prior win rate)", 0.0, 1.0, 0.5, step=0.01)
        # Momentum over the last ~4 overs. Defaults track the current run rate
        # (runs) and assume no recent wickets; override to model a collapse or
        # a flurry.
        default_runs_24 = int(round(default_runs_last_24(score, legal_balls)))
        runs_last_24 = st.number_input(
            "Runs in last 24 balls", 0, 200, default_runs_24, step=1)
        wickets_last_24 = st.slider(
            "Wickets in last 24 balls", 0, 10, 0,
            help="Recent collapses hurt WP; a flurry of boundaries helps it.")
        # is_ipl and is_associate_match are mutually exclusive tiers; only
        # offer the associate toggle for T20Is.
        is_associate = (
            st.checkbox("Associate-nation T20I", value=False)
            if not is_ipl else False
        )

    # A chase can be terminally decided by the game state, in which case the
    # model should not be consulted at all (those states are win/loss facts,
    # and out-of-balls/out-of-wickets losing states never appear in training).
    if innings == 2 and target is not None:
        chase_over = legal_balls >= BALLS_PER_INNINGS or wickets_fallen >= 10
        if score >= target:
            st.success(
                f"Chase complete — {score} reaches the {target} target. "
                "P(batting team win) = 100%."
            )
            return
        if chase_over:  # innings ended short of the target
            if score == target - 1:
                st.info(
                    f"Scores level ({score} vs target {target}) with the "
                    "innings over — match tied."
                )
            else:
                st.error(
                    f"Chase failed — {score}, {target - score} short with no "
                    "balls or wickets left. P(batting team win) = 0%."
                )
            return

    feats = scenario_features(
        innings=innings, score=int(score), wickets_fallen=int(wickets_fallen),
        legal_balls=legal_balls, target=int(target) if target else None,
        venue_par=float(venue_par), batting_strength=float(batting_strength),
        bowling_strength=float(bowling_strength),
        runs_last_24=float(runs_last_24), wickets_last_24=float(wickets_last_24),
        is_ipl=is_ipl, is_associate_match=is_associate,
    )
    wp = float(predict_win_prob(model, feats)[0])

    c1, c2 = st.columns(2)
    c1.metric("P(batting team wins)", f"{wp:.1%}")
    c2.metric("P(bowling team wins)", f"{1 - wp:.1%}")

    balls_left = int(feats["balls_remaining"].iloc[0])
    if innings == 2:
        st.write(
            f"**Chasing:** {score}/{wickets_fallen} after "
            f"{overs_done} overs, needing {int(target) - score} from "
            f"{balls_left} balls (RRR "
            f"{feats['required_run_rate'].iloc[0]:.2f}, CRR "
            f"{feats['current_run_rate'].iloc[0]:.2f})."
        )
    else:
        # projected_score is CRR x 20 by construction, so showing it alongside
        # CRR restates the same number and reads as an under-forecast.
        st.write(
            f"**Batting first:** {score}/{wickets_fallen} after "
            f"{overs_done} overs (CRR "
            f"{feats['current_run_rate'].iloc[0]:.2f})."
        )

    with st.expander("Model input features"):
        st.dataframe(feats.T.rename(columns={0: "value"}), width="stretch")


def main() -> None:
    st.title("T20 Win Probability")
    st.caption(
        "Ball-by-ball P(win) from the calibrated XGBoost model, with the "
        "Win-Probability-Added (WPA) clutch metric behind it."
    )

    if not ensure_artifacts():
        st.info(
            "Point `T20WP_ARTIFACT_RELEASE` at a GitHub release hosting the "
            "parquet/model files, or build them locally with "
            "scripts/run_ingest.py → run_features.py → run_eval.py."
        )
        return

    matches, features, balls = load_frames()
    model = load_calibrated_model()

    mode = st.sidebar.radio(
        "Mode", ["Explore a match", "Win probability calculator"]
    )
    st.sidebar.divider()
    if mode == "Explore a match":
        explore_match(matches, features, balls, model)
    else:
        win_probability_calculator(features, model)


if __name__ == "__main__":
    main()
