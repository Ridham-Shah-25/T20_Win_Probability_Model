"""Win Probability Added (WPA): per-ball ΔWP credit and clutch leaderboards.

Scores every ball with the calibrated model to get ``wp`` = P(batting team
wins), then converts consecutive win probabilities into a per-ball change
(ΔWP) and credits it to the striker and bowler.

Perspective invariant (the error-prone part). Stored ``wp`` is
batting-team-relative and the two innings have OPPOSITE perspectives:
innings-1 ``wp`` = P(team batting first wins); innings-2 ``wp`` = P(chasing
team wins). ΔWP is therefore computed PER INNINGS as the change in the CURRENT
batting team's win prob, with each innings' prior converted to that team's
perspective:

- innings 1 prior (before ball 1) = ``0.5``; ``delta_wp = wp - wp.shift(1)``,
  first ball diffs against ``0.5``.
- innings 2 prior (before ball 1) = ``1 - last_wp_inn1`` (convert the
  batting-first team's final WP to the chasing team's perspective);
  ``delta_wp = wp - wp.shift(1)``, first ball diffs against that prior.

This guarantees per-innings telescoping:
``sum(delta_inn1) == last_wp_inn1 - 0.5`` and
``sum(delta_inn2) == last_wp_inn2 - (1 - last_wp_inn1)``.

Credit sign convention. ``batter_credit = +delta_wp`` is credited to the
striker (``batter``); ``bowler_credit = -delta_wp`` is credited to the
``bowler``. Because ``wp`` is already current-batting-team-relative, a ball
that raises the batting team's WP rewards the batter and penalizes the bowler.
For the bowler "clutch" score we sum ``bowler_credit = sum(-delta_wp)``, so a
bowler who most reduced the batting team's WP earns a high POSITIVE score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from t20wp.features import FEATURE_COLS
from t20wp.models import predict_win_prob

# Default minimum sample size (balls faced/bowled per group) for leaderboards.
DEFAULT_MIN_BALLS = 300

# Context columns carried alongside the per-ball WP for aggregation/reporting.
CONTEXT_COLS = [
    "match_id", "innings", "ball_seq", "season", "competition", "is_ipl",
    "batter", "bowler", "batting_team", "bowling_team",
]

# Default minimum balls for a player/season group to appear on a leaderboard;
# filters out tiny, high-variance samples.
DEFAULT_MIN_BALLS = 300


def compute_ball_wp(features: pd.DataFrame, model) -> pd.DataFrame:
    """Score every ball: add ``wp`` = P(batting team wins) per row.

    Keeps the id/context columns needed downstream and drops the rest of the
    feature matrix (recomputable from ``features``).
    """
    wp = predict_win_prob(model, features[FEATURE_COLS])
    out = features[CONTEXT_COLS].copy()
    out["wp"] = wp
    return out


def compute_delta_wp(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_wp`` and per-player credits using the per-innings priors.

    Rows are ordered within each ``(match_id, innings)`` by ``ball_seq`` before
    diffing. The first ball of innings 1 diffs against ``0.5``; the first ball
    of innings 2 diffs against ``1 - last_wp_inn1`` (chasing-team perspective).

    Fails fast (``ValueError``) on input that would silently produce NaN
    credits: null/non-finite ``wp``, ``innings`` outside ``{1, 2}``, or any
    match with an innings-2 group but no innings-1 group (which would leave the
    innings-2 prior ``1 - last_wp_inn1`` undefined).
    """
    if not np.isfinite(df["wp"].to_numpy(dtype=float)).all():
        raise ValueError("compute_delta_wp: 'wp' contains null or non-finite values")
    bad_innings = set(pd.unique(df["innings"])) - {1, 2}
    if bad_innings:
        raise ValueError(
            f"compute_delta_wp: 'innings' has values outside {{1, 2}}: {sorted(bad_innings)}"
        )
    inn1_matches = set(df.loc[df["innings"] == 1, "match_id"].unique())
    inn2_matches = set(df.loc[df["innings"] == 2, "match_id"].unique())
    missing_inn1 = inn2_matches - inn1_matches
    if missing_inn1:
        raise ValueError(
            "compute_delta_wp: matches have an innings-2 group but no innings-1 "
            f"group (innings-2 prior undefined): {sorted(missing_inn1)[:5]}"
        )

    df = df.sort_values(["match_id", "innings", "ball_seq"]).reset_index(drop=True)

    # Previous-ball wp within each innings; NaN on the first ball of an innings.
    wp_prev = df.groupby(["match_id", "innings"], sort=False)["wp"].shift(1)

    # Final wp of innings 1 per match (df is already ball_seq-ordered).
    inn1 = df[df["innings"] == 1]
    last_wp_inn1 = inn1.groupby("match_id")["wp"].last()
    prior_inn2 = 1.0 - df["match_id"].map(last_wp_inn1)

    # Prior for the first ball of each innings, on the batting team's perspective.
    first_prior = pd.Series(
        np.where(df["innings"].values == 1, 0.5, prior_inn2.values),
        index=df.index,
    )
    wp_prev = wp_prev.where(wp_prev.notna(), first_prior)

    df["delta_wp"] = df["wp"] - wp_prev
    df["batter_credit"] = df["delta_wp"]
    df["bowler_credit"] = -df["delta_wp"]
    return df


def leaderboard(
    df: pd.DataFrame,
    role: str,
    by: tuple[str, ...] = ("season",),
    min_balls: int = DEFAULT_MIN_BALLS,
) -> pd.DataFrame:
    """Aggregate per-ball credit into a player (+ ``by``) clutch leaderboard.

    ``role`` is ``"batter"`` or ``"bowler"``. Groups by the role column plus
    ``by``, sums that role's credit (``batter_credit`` / ``bowler_credit``),
    counts balls, filters groups with ``balls >= min_balls``, and sorts by
    summed credit descending. Bowler ``clutch = sum(bowler_credit) =
    sum(-delta_wp)``, so a good clutch bowler scores high and positive.
    Returns the ``balls`` sample size so low-threshold reads are possible.
    """
    if role not in ("batter", "bowler"):
        raise ValueError(f"role must be 'batter' or 'bowler', got {role!r}")
    credit_col = f"{role}_credit"
    group_cols = [role] + list(by)
    agg = (
        df.groupby(group_cols, sort=False)
        .agg(clutch=(credit_col, "sum"), balls=(role, "size"))
        .reset_index()
    )
    agg = agg[agg["balls"] >= min_balls]
    return agg.sort_values("clutch", ascending=False).reset_index(drop=True)
