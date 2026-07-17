"""Build leakage-safe per-ball features from the ingested tables.

Every row is the game state AFTER a delivery, labelled with the match
outcome for the batting team (1 win, 0 loss, 0.5 tie). All historical
features (venue par, team strength) are computed only from matches
strictly earlier in the chronological ordering — never from the match
being featurized or later ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BALLS_PER_INNINGS = 120
MOMENTUM_WINDOW = 24  # deliveries (~4 overs)
VENUE_SHRINKAGE_K = 10  # pseudo-matches of tier average blended into venue par
TEAM_WINDOW = 30  # past matches used for team strength
TEAM_SHRINKAGE_M = 5  # pseudo-matches of 0.5 blended into team win rate
RRR_CAP = 36.0

ID_COLS = [
    "match_id", "date", "season", "competition", "tier", "venue",
    "batting_team", "bowling_team", "batter", "non_striker", "bowler",
    "innings", "over", "ball_in_over", "ball_seq",
]

FEATURE_COLS = [
    "is_second_innings", "balls_remaining", "wickets_in_hand", "score",
    "current_run_rate", "runs_required", "required_run_rate",
    "rrr_minus_crr", "projected_score", "projected_vs_par",
    "target_vs_par", "venue_par", "runs_last_24", "wickets_last_24",
    "batting_strength", "bowling_strength", "strength_diff",
    "is_ipl", "is_associate_match",
]

LABEL_COL = "won"


def _normalize_venue(venue: pd.Series) -> pd.Series:
    return (
        venue.str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _match_order(matches: pd.DataFrame) -> pd.DataFrame:
    """Chronological match ordering used by every historical feature."""
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)


def _venue_par(matches: pd.DataFrame, inn1_totals: pd.Series) -> pd.Series:
    """Expected 1st-innings score per match from strictly prior matches.

    Blend of the venue's prior mean and the competition tier's prior mean,
    weighted by how many prior matches the venue has (shrinkage). Falls
    back to the global prior mean, then a constant, for early matches.
    """
    m = _match_order(matches).copy()
    m["inn1"] = m["match_id"].map(inn1_totals)
    m["venue_key"] = _normalize_venue(m["venue"])

    grp = m.groupby("venue_key")["inn1"]
    venue_mean = grp.transform(lambda s: s.expanding().mean().shift(1))
    venue_n = grp.cumcount()

    tier_mean = m.groupby("tier")["inn1"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    global_mean = m["inn1"].expanding().mean().shift(1)
    fallback = tier_mean.fillna(global_mean).fillna(160.0)

    par = (venue_n * venue_mean.fillna(0) + VENUE_SHRINKAGE_K * fallback) / (
        venue_n + VENUE_SHRINKAGE_K
    )
    return pd.Series(par.values, index=m["match_id"]).rename("venue_par")


def _team_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """Rolling win rate per team entering each match (prior matches only)."""
    m = _match_order(matches)
    long = pd.concat(
        [
            m.assign(team=m["team1"]),
            m.assign(team=m["team2"]),
        ]
    ).sort_values(["date", "match_id"], kind="stable")
    long["result"] = np.where(
        long["outcome_type"] == "tie", 0.5,
        (long["winner"] == long["team"]).astype(float),
    )

    grp = long.groupby("team")["result"]
    prior_sum = grp.transform(
        lambda s: s.rolling(TEAM_WINDOW, min_periods=1).sum().shift(1)
    ).fillna(0)
    prior_n = grp.transform(
        lambda s: s.rolling(TEAM_WINDOW, min_periods=1).count().shift(1)
    ).fillna(0)
    long["strength"] = (prior_sum + TEAM_SHRINKAGE_M * 0.5) / (
        prior_n + TEAM_SHRINKAGE_M
    )
    return long[["match_id", "team", "strength"]]


def build_features(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches[matches["is_model_match"]].copy()
    matches["tier"] = np.select(
        [
            matches["competition"] == "ipl",
            matches["is_full_member_match"],
        ],
        ["ipl", "t20i_full_member"],
        default="t20i_associate",
    )

    b = balls[
        balls["match_id"].isin(matches["match_id"])
        & ~balls["is_super_over"]
        & (balls["innings"] <= 2)
    ].copy()

    meta_cols = ["match_id", "date", "season", "competition", "tier", "venue",
                 "winner", "outcome_type"]
    b = b.merge(matches[meta_cols], on="match_id", how="left")
    b = b.sort_values(["date", "match_id", "innings"], kind="stable").reset_index(drop=True)

    g = b.groupby(["match_id", "innings"], sort=False)
    b["ball_seq"] = g.cumcount() + 1
    b["score"] = g["runs_total"].cumsum()
    b["wickets_fallen"] = g["n_wickets_on_ball"].cumsum()
    b["legal_balls"] = g["is_legal"].cumsum()

    b["balls_remaining"] = (BALLS_PER_INNINGS - b["legal_balls"]).clip(lower=0)
    b["wickets_in_hand"] = 10 - b["wickets_fallen"]
    b["current_run_rate"] = 6 * b["score"] / b["legal_balls"].clip(lower=1)

    # Chase target: Cricsheet omits it for a handful of matches; for
    # non-DLS matches it is always 1st-innings total + 1.
    inn1_totals = b[b["innings"] == 1].groupby("match_id")["runs_total"].sum()
    derived_target = b["match_id"].map(inn1_totals) + 1
    b["target"] = np.where(b["target_runs"] > 0, b["target_runs"], derived_target)

    is2 = b["innings"] == 2
    b["is_second_innings"] = is2.astype(int)
    b["runs_required"] = np.where(is2, b["target"] - b["score"], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rrr = 6 * b["runs_required"] / b["balls_remaining"]
    rrr = np.where(
        b["balls_remaining"] == 0,
        np.where(b["runs_required"] > 0, RRR_CAP, 0.0),
        rrr,
    )
    b["required_run_rate"] = pd.Series(rrr).clip(0, RRR_CAP).where(is2)
    b["rrr_minus_crr"] = b["required_run_rate"] - b["current_run_rate"]

    b["projected_score"] = np.where(
        ~is2, b["score"] * BALLS_PER_INNINGS / b["legal_balls"].clip(lower=1), np.nan
    )

    b["runs_last_24"] = g["runs_total"].transform(
        lambda s: s.rolling(MOMENTUM_WINDOW, min_periods=1).sum()
    )
    b["wickets_last_24"] = g["n_wickets_on_ball"].transform(
        lambda s: s.rolling(MOMENTUM_WINDOW, min_periods=1).sum()
    )

    b["venue_par"] = b["match_id"].map(_venue_par(matches, inn1_totals))
    b["projected_vs_par"] = b["projected_score"] - b["venue_par"]
    b["target_vs_par"] = np.where(is2, b["target"] - b["venue_par"], np.nan)

    strength = _team_strength(matches)
    for side in ("batting", "bowling"):
        b = b.merge(
            strength.rename(
                columns={"team": f"{side}_team", "strength": f"{side}_strength"}
            ),
            on=["match_id", f"{side}_team"],
            how="left",
        )
    b["strength_diff"] = b["batting_strength"] - b["bowling_strength"]

    b["is_ipl"] = (b["competition"] == "ipl").astype(int)
    b["is_associate_match"] = (b["tier"] == "t20i_associate").astype(int)

    b[LABEL_COL] = np.where(
        b["outcome_type"] == "tie", 0.5,
        (b["winner"] == b["batting_team"]).astype(float),
    )

    return b[ID_COLS + FEATURE_COLS + [LABEL_COL]]


def default_runs_last_24(score: int, legal_balls: int) -> float:
    """Momentum default: current pace over a 24-ball window, capped at ``score``.

    Mirrors the pipeline's ``runs_last_24`` rolling sum (``min_periods=1``),
    which before 24 balls equals the cumulative innings runs. Shared by
    :func:`scenario_features` and the what-if app so the default can't drift.
    """
    return min(float(score), score * MOMENTUM_WINDOW / max(legal_balls, 1))


def scenario_features(
    *,
    innings: int,
    score: int,
    wickets_fallen: int,
    legal_balls: int,
    target: int | None = None,
    venue_par: float,
    batting_strength: float = 0.5,
    bowling_strength: float = 0.5,
    runs_last_24: float | None = None,
    wickets_last_24: float = 0.0,
    is_ipl: bool = False,
    is_associate_match: bool = False,
) -> pd.DataFrame:
    """Build a single ``FEATURE_COLS`` row from raw match state.

    Reconstructs exactly the feature derivations of :func:`build_features` for
    one delivery-state, so the "what-if" calculator scores identically to the
    ball-by-ball pipeline. Returns a one-row DataFrame with the columns in
    ``FEATURE_COLS`` order (and the same per-innings NaN pattern the model was
    trained on: chase features are NaN in the 1st innings, ``projected_*`` are
    NaN in the 2nd).

    Raw state (derivable from the game): ``innings`` (1 or 2), ``score``,
    ``wickets_fallen`` (0-10), ``legal_balls`` bowled so far in this innings
    (0-120), and ``target`` (1st-innings total + 1; required for a chase).

    Contextual inputs that the pipeline computes from history and cannot be
    recovered from raw state -- caller must supply them (the app exposes them
    as advanced inputs with data-driven defaults): ``venue_par`` (expected
    1st-innings total at the venue), ``batting_strength``/``bowling_strength``
    (rolling prior win rates, 0-1), and the momentum window
    ``runs_last_24``/``wickets_last_24`` (defaults to the current run rate over
    24 balls / 0 when not supplied).
    """
    if innings not in (1, 2):
        raise ValueError(f"innings must be 1 or 2, got {innings}")
    if not 0 <= wickets_fallen <= 10:
        raise ValueError(f"wickets_fallen must be 0-10, got {wickets_fallen}")
    if not 0 <= legal_balls <= BALLS_PER_INNINGS:
        raise ValueError(
            f"legal_balls must be 0-{BALLS_PER_INNINGS}, got {legal_balls}"
        )

    is2 = innings == 2
    balls_remaining = max(BALLS_PER_INNINGS - legal_balls, 0)
    wickets_in_hand = 10 - wickets_fallen
    current_run_rate = 6 * score / max(legal_balls, 1)

    if runs_last_24 is None:
        runs_last_24 = default_runs_last_24(score, legal_balls)

    if is2:
        if target is None:
            raise ValueError("target is required for a 2nd-innings chase")
        runs_required = float(target - score)
        if balls_remaining == 0:
            rrr = RRR_CAP if runs_required > 0 else 0.0
        else:
            rrr = 6 * runs_required / balls_remaining
        required_run_rate = float(min(max(rrr, 0.0), RRR_CAP))
        rrr_minus_crr = required_run_rate - current_run_rate
        projected_score = np.nan
        projected_vs_par = np.nan
        target_vs_par = float(target - venue_par)
    else:
        runs_required = np.nan
        required_run_rate = np.nan
        rrr_minus_crr = np.nan
        projected_score = score * BALLS_PER_INNINGS / max(legal_balls, 1)
        projected_vs_par = projected_score - venue_par
        target_vs_par = np.nan

    row = {
        "is_second_innings": int(is2),
        "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand,
        "score": score,
        "current_run_rate": current_run_rate,
        "runs_required": runs_required,
        "required_run_rate": required_run_rate,
        "rrr_minus_crr": rrr_minus_crr,
        "projected_score": projected_score,
        "projected_vs_par": projected_vs_par,
        "target_vs_par": target_vs_par,
        "venue_par": float(venue_par),
        "runs_last_24": float(runs_last_24),
        "wickets_last_24": float(wickets_last_24),
        "batting_strength": float(batting_strength),
        "bowling_strength": float(bowling_strength),
        "strength_diff": float(batting_strength - bowling_strength),
        "is_ipl": int(is_ipl),
        "is_associate_match": int(is_associate_match),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def make_splits(
    matches: pd.DataFrame, train_frac: float = 0.75, val_frac: float = 0.10
) -> dict:
    """Freeze a match-level, time-based split. Returns cutoff dates + IDs."""
    m = _match_order(matches[matches["is_model_match"]])
    n = len(m)
    train_end = m["date"].iloc[int(n * train_frac) - 1]
    val_end = m["date"].iloc[int(n * (train_frac + val_frac)) - 1]

    train = m[m["date"] <= train_end]
    val = m[(m["date"] > train_end) & (m["date"] <= val_end)]
    test = m[m["date"] > val_end]
    return {
        "train_end_date": str(train_end.date()),
        "val_end_date": str(val_end.date()),
        "train_match_ids": train["match_id"].tolist(),
        "val_match_ids": val["match_id"].tolist(),
        "test_match_ids": test["match_id"].tolist(),
    }
