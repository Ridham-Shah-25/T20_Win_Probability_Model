"""Unit tests for the Phase 3 evaluation harness and model layer.

Locks down the riskiest logic: the sklearn calibration return-order pitfall,
calibration-table column order / monotonicity / count conservation, the
ECE <= MCE ordering, and that XGB early stopping actually triggers.

Run with ``pytest`` from the repo root, or directly:
``python tests/test_evaluate.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.evaluate import (  # noqa: E402
    calibration_table,
    evaluate_probs,
    reliability_stats,
)


def test_evaluate_probs_finite_and_base_rate():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    out = evaluate_probs(y, p)
    assert np.isfinite(out["log_loss"])
    assert np.isfinite(out["brier"])
    assert out["n"] == 4
    assert out["base_rate"] == 0.5


def test_calibration_table_columns_and_conservation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    tbl = calibration_table(y, p, n_bins=10)
    assert list(tbl.columns) == ["prob_pred", "prob_true", "count"]
    assert tbl["prob_pred"].is_monotonic_increasing
    assert int(tbl["count"].sum()) == len(y)


def test_reliability_stats_ordering():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = rng.random(2000)
    stats = reliability_stats(y, p)
    assert stats["mce"] >= stats["ece"] >= 0.0


def test_tune_xgb_early_stopping():
    # Self-contained synthetic data so the test needs no generated parquet.
    from t20wp.features import FEATURE_COLS
    from t20wp.models import tune_xgb

    rng = np.random.default_rng(7)
    n_tr, n_val = 2000, 500
    n_feat = len(FEATURE_COLS)
    X_all = rng.standard_normal((n_tr + n_val, n_feat))
    # A learnable signal so the model fits and then early-stops well short of
    # n_estimators rather than running to the cap.
    logits = X_all[:, 0] * 1.5 - X_all[:, 1]
    y_all = (rng.random(n_tr + n_val) < 1 / (1 + np.exp(-logits))).astype(int)
    X_tr = pd.DataFrame(X_all[:n_tr], columns=FEATURE_COLS)
    X_val = pd.DataFrame(X_all[n_tr:], columns=FEATURE_COLS)
    y_tr, y_val = y_all[:n_tr], y_all[n_tr:]

    grid = [{"max_depth": 4, "learning_rate": 0.1}]
    best_model, best_params, trials = tune_xgb(X_tr, y_tr, X_val, y_val, grid=grid)
    n_estimators = 2000
    assert int(best_model.best_iteration) < n_estimators - 1
    assert int(trials["best_iteration"].iloc[0]) < n_estimators - 1


if __name__ == "__main__":
    test_evaluate_probs_finite_and_base_rate()
    test_calibration_table_columns_and_conservation()
    test_reliability_stats_ordering()
    test_tune_xgb_early_stopping()
    print("All tests passed.")
