"""Unit tests for Phase 4 WPA ΔWP orientation and credit balance.

Constructs a tiny synthetic two-innings match with KNOWN ``wp`` values and
asserts that ``compute_delta_wp`` produces the correct per-innings telescoping
(with the innings-2 prior converted to the chasing team's perspective) and
that per-ball credit sums to zero. This locks the batting-team-relative
orientation down before running on real data.

Run with ``pytest`` from the repo root, or directly:
``python tests/test_wpa.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.wpa import compute_delta_wp, leaderboard  # noqa: E402


def _fake_match():
    """One match, two innings, three balls each, with hand-picked wp values.

    innings 1 wp (P team batting first wins): 0.55, 0.60, 0.58 -> last = 0.58
    innings 2 wp (P chasing team wins):       0.45, 0.50, 0.70 -> last = 0.70
    Rows are intentionally shuffled to prove the sort-before-diff logic.
    """
    rows = [
        # match_id, innings, ball_seq, wp, batter, bowler
        ("m1", 1, 1, 0.55, "A", "X"),
        ("m1", 1, 2, 0.60, "B", "X"),
        ("m1", 1, 3, 0.58, "A", "Y"),
        ("m1", 2, 1, 0.45, "C", "P"),
        ("m1", 2, 2, 0.50, "D", "P"),
        ("m1", 2, 3, 0.70, "C", "Q"),
    ]
    df = pd.DataFrame(
        rows, columns=["match_id", "innings", "ball_seq", "wp", "batter", "bowler"]
    )
    # Shuffle to ensure compute_delta_wp re-sorts by (match_id, innings, ball_seq).
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_per_innings_telescoping():
    out = compute_delta_wp(_fake_match())
    inn1 = out[out["innings"] == 1]
    inn2 = out[out["innings"] == 2]

    last_wp_inn1 = 0.58
    last_wp_inn2 = 0.70

    res1 = inn1["delta_wp"].sum() - (last_wp_inn1 - 0.5)
    res2 = inn2["delta_wp"].sum() - (last_wp_inn2 - (1.0 - last_wp_inn1))
    assert abs(res1) < 1e-9, f"innings-1 telescoping residual {res1}"
    assert abs(res2) < 1e-9, f"innings-2 telescoping residual {res2}"


def test_first_ball_priors():
    out = compute_delta_wp(_fake_match()).sort_values(
        ["innings", "ball_seq"]
    ).reset_index(drop=True)
    # innings-1 first ball diffs against 0.5.
    d_i1_b1 = out[(out.innings == 1) & (out.ball_seq == 1)]["delta_wp"].iloc[0]
    assert abs(d_i1_b1 - (0.55 - 0.5)) < 1e-9
    # innings-2 first ball diffs against 1 - last_wp_inn1 = 1 - 0.58 = 0.42.
    d_i2_b1 = out[(out.innings == 2) & (out.ball_seq == 1)]["delta_wp"].iloc[0]
    assert abs(d_i2_b1 - (0.45 - 0.42)) < 1e-9


def test_credit_balance():
    out = compute_delta_wp(_fake_match())
    bal = (out["batter_credit"] + out["bowler_credit"]).abs().max()
    assert bal < 1e-9, f"credit imbalance {bal}"
    # bowler_credit is the negative of delta_wp.
    assert np.allclose(out["bowler_credit"], -out["delta_wp"])


def test_leaderboard_min_balls_and_sort():
    out = compute_delta_wp(_fake_match())
    out["season"] = "S1"
    lb = leaderboard(out, "batter", by=("season",), min_balls=1)
    assert list(lb.columns) == ["batter", "season", "clutch", "balls"]
    assert lb["clutch"].is_monotonic_decreasing
    # min_balls filter removes small groups.
    lb_hi = leaderboard(out, "batter", by=("season",), min_balls=3)
    assert (lb_hi["balls"] >= 3).all()


if __name__ == "__main__":
    test_per_innings_telescoping()
    test_first_ball_priors()
    test_credit_balance()
    test_leaderboard_min_balls_and_sort()
    print("All WPA tests passed.")
