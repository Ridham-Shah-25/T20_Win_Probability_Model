"""Run Phase 2: build features, freeze the time-based split, verify no leakage.

Writes data/processed/features.parquet, data/processed/splits.json and
reports/phase2_gate.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.features import (  # noqa: E402
    FEATURE_COLS,
    VENUE_SHRINKAGE_K,
    _normalize_venue,
    build_features,
    make_splits,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}: {name} {detail}")


def verify_venue_par_no_leakage(features, matches, inn1, n_samples=8):
    """Brute-force recompute venue par for sampled matches, independently of
    the expanding/shift implementation, using only strictly-prior matches."""
    m = matches[matches["is_model_match"]].copy()
    m["tier"] = np.select(
        [m["competition"] == "ipl", m["is_full_member_match"]],
        ["ipl", "t20i_full_member"], default="t20i_associate",
    )
    m["venue_key"] = _normalize_venue(m["venue"])
    m["inn1"] = m["match_id"].map(inn1)
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    sample = m.iloc[rng.choice(np.arange(50, len(m)), n_samples, replace=False)]
    for _, row in sample.iterrows():
        prior = m[
            (m["date"] < row["date"])
            | ((m["date"] == row["date"]) & (m["match_id"] < row["match_id"]))
        ]
        venue_prior = prior[prior["venue_key"] == row["venue_key"]]["inn1"]
        tier_prior = prior[prior["tier"] == row["tier"]]["inn1"]
        fallback = tier_prior.mean() if len(tier_prior) else (
            prior["inn1"].mean() if len(prior) else 160.0
        )
        n = len(venue_prior)
        expected = (n * (venue_prior.mean() if n else 0) + VENUE_SHRINKAGE_K * fallback) / (
            n + VENUE_SHRINKAGE_K
        )
        actual = features.loc[features["match_id"] == row["match_id"], "venue_par"].iloc[0]
        check(
            f"venue_par leakage-safe ({row['match_id']})",
            bool(np.isclose(actual, expected)),
            f"expected {expected:.2f}, got {actual:.2f}",
        )


def main() -> None:
    matches = pd.read_parquet(ROOT / "data/processed/matches.parquet")
    balls = pd.read_parquet(ROOT / "data/processed/balls.parquet")

    features = build_features(balls, matches)
    features.to_parquet(ROOT / "data/processed/features.parquet", index=False)

    splits = make_splits(matches)
    (ROOT / "data/processed/splits.json").write_text(json.dumps(splits, indent=1))

    # --- Assertions ---------------------------------------------------
    model_ids = set(matches.loc[matches["is_model_match"], "match_id"])
    split_ids = [set(splits[f"{s}_match_ids"]) for s in ("train", "val", "test")]
    check(
        "splits partition all model matches",
        set.union(*split_ids) == model_ids
        and sum(len(s) for s in split_ids) == len(model_ids),
    )
    md = matches.set_index("match_id")["date"]
    check(
        "split date separation (train < val < test)",
        md[list(split_ids[0])].max() < md[list(split_ids[1])].min()
        and md[list(split_ids[1])].max() < md[list(split_ids[2])].min(),
    )

    check("no negative balls_remaining", (features["balls_remaining"] >= 0).all())
    check("no negative wickets_in_hand", (features["wickets_in_hand"] >= 0).all())

    always_present = [
        c for c in FEATURE_COLS
        if c not in ("runs_required", "required_run_rate", "rrr_minus_crr",
                     "projected_score", "projected_vs_par", "target_vs_par")
    ]
    check(
        "no NaNs in unconditional features",
        not features[always_present].isna().any().any(),
    )
    inn2 = features[features["is_second_innings"] == 1]
    check(
        "chase features present on all 2nd-innings rows",
        not inn2[["runs_required", "required_run_rate"]].isna().any().any(),
    )

    # Label is relative to the batting team: constant within an innings,
    # complementary across the two innings (both 0.5 for ties).
    check(
        "labels constant within match-innings",
        (features.groupby(["match_id", "innings"])["won"].nunique() == 1).all(),
    )
    per_inn = features.groupby(["match_id", "innings"])["won"].first().unstack()
    both = per_inn.dropna()
    check(
        "labels complementary across innings",
        bool(np.isclose(both[1] + both[2], 1.0).all()),
        f"({len(both)} matches with both innings)",
    )
    # Winning chase must end with runs_required <= 0
    won_chases = matches[
        matches["is_model_match"] & (matches["win_by_wickets"] > 0)
    ]["match_id"]
    last_ball = inn2[inn2["match_id"].isin(won_chases)].groupby("match_id").tail(1)
    check(
        "won chases end with runs_required <= 0",
        (last_ball["runs_required"] <= 0).all(),
        f"({len(last_ball)} chases)",
    )

    inn1 = features[features["innings"] == 1].groupby("match_id")["score"].max()
    verify_venue_par_no_leakage(features, matches, inn1)

    # --- Walkthrough: 2016 World T20 final (Brathwaite over) ----------
    m16 = matches[
        (matches["date"] == "2016-04-03")
        & (matches["team1"].isin(["England", "West Indies"]))
    ]
    walkthrough = pd.DataFrame()
    if len(m16):
        mid = m16.iloc[0]["match_id"]
        cols = ["over", "ball_in_over", "batter", "bowler", "score",
                "wickets_in_hand", "balls_remaining", "runs_required",
                "required_run_rate", "rrr_minus_crr", "runs_last_24",
                "wickets_last_24", "won"]
        walkthrough = (
            features[(features["match_id"] == mid) & (features["innings"] == 2)]
            .tail(10)[cols]
            .round(2)
        )

    # --- Report --------------------------------------------------------
    lines = ["# Phase 2 gate report", ""]
    lines.append(f"Feature rows: **{len(features):,}** across "
                 f"**{features['match_id'].nunique():,}** matches | "
                 f"features: {len(FEATURE_COLS)}")
    lines.append("")

    lines.extend(["## Frozen split (by match, by time)", ""])
    split_tbl = pd.DataFrame(
        {
            "matches": [len(s) for s in split_ids],
            "first date": [md[list(s)].min().date() for s in split_ids],
            "last date": [md[list(s)].max().date() for s in split_ids],
        },
        index=["train", "val", "test"],
    )
    lines.extend([split_tbl.to_markdown(), ""])

    lines.extend(["## Checks", ""])
    lines.extend(
        f"- {'✅' if p else '❌'} {name} {detail}" for name, p, detail in CHECKS
    )
    lines.append("")

    lines.extend(["## Feature summary", "",
                  features[FEATURE_COLS].describe().T.round(2).to_markdown(), ""])

    if len(walkthrough):
        lines.extend([
            "## Walkthrough: WT20 2016 final, WI chasing 156 (last 10 balls)",
            "", walkthrough.to_markdown(index=False), "",
        ])

    report = "\n".join(lines)
    (ROOT / "reports/phase2_gate.md").write_text(report)
    print(f"\nSaved: reports/phase2_gate.md")
    if not all(p for _, p, _ in CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
