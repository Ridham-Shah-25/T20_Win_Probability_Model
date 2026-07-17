"""Unit tests for the ``scenario_features`` what-if feature builder.

Locks the calculator's feature reconstruction to the ball-by-ball pipeline:
the same derived columns, the same per-innings NaN pattern (chase features
NaN in the 1st innings, ``projected_*`` NaN in the 2nd), and the same edge
behavior (``RRR_CAP`` clamp, ``balls_remaining == 0`` branch).

The strongest guardrail also runs when the pipeline artifacts exist: it
rebuilds a real ``features.parquet`` row from its raw state and asserts every
feature matches. That check is skipped when the parquet is absent (e.g. CI
without regenerated data), leaving the self-contained checks below.

Run with ``pytest`` from the repo root, or directly:
``python tests/test_features.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.features import FEATURE_COLS, RRR_CAP, scenario_features  # noqa: E402


def test_columns_order_and_second_innings_nan_pattern():
    row = scenario_features(
        innings=2, score=53, wickets_fallen=3, legal_balls=59,
        target=156, venue_par=153.7,
    )
    assert list(row.columns) == FEATURE_COLS
    assert len(row) == 1
    r = row.iloc[0]
    # 2nd innings: projected_* are NaN, chase features populated.
    assert pd.isna(r["projected_score"])
    assert pd.isna(r["projected_vs_par"])
    for col in ("runs_required", "required_run_rate", "rrr_minus_crr",
                "target_vs_par"):
        assert not pd.isna(r[col]), col
    assert r["is_second_innings"] == 1
    assert r["balls_remaining"] == 61
    assert r["wickets_in_hand"] == 7
    assert r["runs_required"] == 156 - 53


def test_first_innings_nan_pattern():
    row = scenario_features(
        innings=1, score=88, wickets_fallen=4, legal_balls=71, venue_par=150.0,
    )
    r = row.iloc[0]
    # 1st innings: chase features NaN, projected_* populated.
    for col in ("runs_required", "required_run_rate", "rrr_minus_crr",
                "target_vs_par"):
        assert pd.isna(r[col]), col
    assert not pd.isna(r["projected_score"])
    assert not pd.isna(r["projected_vs_par"])
    # projected = score * 120 / legal_balls.
    assert r["projected_score"] == 88 * 120 / 71


def test_current_run_rate_and_strength_diff():
    r = scenario_features(
        innings=1, score=60, wickets_fallen=2, legal_balls=36, venue_par=160.0,
        batting_strength=0.7, bowling_strength=0.4,
    ).iloc[0]
    assert r["current_run_rate"] == 6 * 60 / 36  # 10.0
    assert abs(r["strength_diff"] - (0.7 - 0.4)) < 1e-12


def test_rrr_cap_and_balls_remaining_zero():
    # Impossible ask on the last ball -> RRR clamped to the training cap.
    r = scenario_features(
        innings=2, score=100, wickets_fallen=5, legal_balls=120,
        target=250, venue_par=160.0,
    ).iloc[0]
    assert r["balls_remaining"] == 0
    assert r["required_run_rate"] == RRR_CAP
    # Chase already won on the last ball -> RRR is 0, not the cap.
    r2 = scenario_features(
        innings=2, score=250, wickets_fallen=5, legal_balls=120,
        target=250, venue_par=160.0,
    ).iloc[0]
    assert r2["balls_remaining"] == 0
    assert r2["required_run_rate"] == 0.0


def test_default_momentum_tracks_run_rate():
    # runs_last_24 defaults to CRR over a 24-ball window, capped at the runs
    # scored so far (mirrors the pipeline's min_periods=1 rolling sum).
    r = scenario_features(
        innings=1, score=60, wickets_fallen=1, legal_balls=36, venue_par=160.0,
    ).iloc[0]
    assert abs(r["runs_last_24"] - r["current_run_rate"] * 24 / 6) < 1e-9
    assert r["wickets_last_24"] == 0.0
    # Early in the innings (< 24 balls) the cap binds: the window can't exceed
    # the total runs scored.
    early = scenario_features(
        innings=1, score=10, wickets_fallen=0, legal_balls=6, venue_par=160.0,
    ).iloc[0]
    assert early["runs_last_24"] == 10


def test_input_validation():
    for bad in ({"innings": 3}, {"wickets_fallen": 11}, {"legal_balls": 121}):
        kwargs = {"innings": 1, "score": 50, "wickets_fallen": 1,
                  "legal_balls": 30, "venue_par": 160.0}
        kwargs.update(bad)
        with pytest.raises(ValueError):
            scenario_features(**kwargs)
    # 2nd innings without a target must fail.
    with pytest.raises(ValueError):
        scenario_features(innings=2, score=50, wickets_fallen=1,
                          legal_balls=30, venue_par=160.0)


def test_reproduces_real_feature_rows_when_available():
    """If artifacts exist, rebuild real rows from raw state and compare."""
    fp = ROOT / "data" / "processed" / "features.parquet"
    if not fp.exists():
        pytest.skip("features.parquet not regenerated in this environment")
    f = pd.read_parquet(fp)
    f = f[f["match_id"] == "951373"]
    if f.empty:
        pytest.skip("match 951373 not present in features.parquet")
    for innings in (1, 2):
        sub = f[f["innings"] == innings]
        if len(sub) < 61:
            continue
        real = sub.iloc[60]
        legal_balls = int(120 - real["balls_remaining"])
        wickets_fallen = int(10 - real["wickets_in_hand"])
        target = (int(real["score"] + real["runs_required"])
                  if innings == 2 else None)
        got = scenario_features(
            innings=innings, score=int(real["score"]),
            wickets_fallen=wickets_fallen, legal_balls=legal_balls,
            target=target, venue_par=float(real["venue_par"]),
            batting_strength=float(real["batting_strength"]),
            bowling_strength=float(real["bowling_strength"]),
            runs_last_24=float(real["runs_last_24"]),
            wickets_last_24=float(real["wickets_last_24"]),
        ).iloc[0]
        for col in FEATURE_COLS:
            a, b = got[col], real[col]
            if pd.isna(a) or pd.isna(b):
                assert pd.isna(a) and pd.isna(b), f"{col} NaN mismatch"
            else:
                assert abs(float(a) - float(b)) < 1e-9, f"{col}: {a} vs {b}"


if __name__ == "__main__":
    test_columns_order_and_second_innings_nan_pattern()
    test_first_innings_nan_pattern()
    test_current_run_rate_and_strength_diff()
    test_rrr_cap_and_balls_remaining_zero()
    test_default_momentum_tracks_run_rate()
    test_input_validation()
    test_reproduces_real_feature_rows_when_available()
    print("All feature tests passed.")
