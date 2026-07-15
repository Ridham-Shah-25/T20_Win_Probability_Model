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
