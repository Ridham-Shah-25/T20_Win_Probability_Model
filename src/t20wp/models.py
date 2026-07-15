"""Model layer: split loading, three model builders, and persistence.

Provides the logistic-regression baseline, an early-stopped XGBoost model
tuned over a small manual grid, and an isotonic-calibrated wrapper. Also
handles split loading (filter by frozen match ids, drop ties) and joblib
persistence.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from t20wp.features import FEATURE_COLS, ID_COLS, LABEL_COL


def load_split(features, splits, split_name, drop_ties=True):
    """Return ``(X, y, meta)`` for a frozen split.

    Filters ``features`` to the split's match ids; optionally drops tie rows
    (``won == 0.5``). ``X`` = ``features[FEATURE_COLS]``, ``meta`` =
    ``features[ID_COLS]``.

    ``y`` is cast to int only when ties are dropped (labels are then a clean
    0/1); with ``drop_ties=False`` the labels are returned as float so a tie
    (``0.5``) is not silently truncated to a loss. The binary models require
    ``drop_ties=True``.
    """
    ids = set(splits[f"{split_name}_match_ids"])
    sub = features[features["match_id"].isin(ids)]
    if drop_ties:
        sub = sub[sub[LABEL_COL] != 0.5]
    X = sub[FEATURE_COLS].reset_index(drop=True)
    y = sub[LABEL_COL].values
    if drop_ties:
        y = y.astype(int)
    meta = sub[ID_COLS].reset_index(drop=True)
    return X, y, meta


def build_logreg() -> Pipeline:
    """Logistic-regression baseline: median impute + scale + LR."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )


# Hyperparameter columns recorded for every trial so trials_df has a stable
# schema across both search stages (missing keys fall back to these defaults).
_XGB_PARAM_DEFAULTS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
}

_N_ESTIMATORS = 2000
_EARLY_STOPPING_ROUNDS = 50


def _fit_xgb(params: dict, X_tr, y_tr, X_val, y_val):
    """Fit one early-stopped XGB and return ``(model, val_log_loss)``.

    ``params`` may set any subset of the tuned hyperparameters; the rest fall
    back to ``_XGB_PARAM_DEFAULTS``.
    """
    tuned = {**_XGB_PARAM_DEFAULTS, **params}
    model = XGBClassifier(
        n_estimators=_N_ESTIMATORS,
        early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        **tuned,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_prob = model.predict_proba(X_val)[:, 1]
    return model, float(log_loss(y_val, val_prob, labels=[0, 1]))


def _trial_row(params: dict, model, val_loss: float, stage: str) -> dict:
    """One ``trials_df`` record: stage, full hyperparameters, fit result."""
    tuned = {**_XGB_PARAM_DEFAULTS, **params}
    return {
        "stage": stage,
        **tuned,
        "best_iteration": int(model.best_iteration),
        "val_log_loss": val_loss,
    }


def tune_xgb(X_tr, y_tr, X_val, y_val, grid=None):
    """Two-stage manual XGBoost search with early stopping on the val set.

    All trials use ``n_estimators=2000``, ``early_stopping_rounds=50``,
    ``eval_metric="logloss"``, ``tree_method="hist"``, scored by validation
    log loss. Returns ``(best_model, best_params, trials_df)`` where
    ``trials_df`` records every trial's hyperparameters, ``best_iteration``,
    and val log loss (with a ``stage`` column).

    Stage 1 searches ``max_depth in {4,6} x learning_rate in {0.03,0.1} x
    min_child_weight in {1,5}``. The best stage-1 config is then fixed and
    stage 2 searches ``subsample in {0.8,1.0} x colsample_bytree in {0.8,1.0}
    x reg_lambda in {1.0,5.0}`` on top of it. The overall best across both
    stages is returned.

    If an explicit ``grid`` (list of param dicts) is passed, a single-stage
    search over exactly those dicts is run instead (used by the fast unit
    test).
    """
    trials = []
    best_model = None
    best_params = None
    best_loss = np.inf

    def run(params: dict, stage: str) -> float:
        """Fit ``params``, record the trial, update the global best; return val loss."""
        nonlocal best_model, best_params, best_loss
        model, val_loss = _fit_xgb(params, X_tr, y_tr, X_val, y_val)
        trials.append(_trial_row(params, model, val_loss, stage))
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = {**_XGB_PARAM_DEFAULTS, **params}
        return val_loss

    if grid is not None:
        for params in grid:
            run(params, "custom")
    else:
        # Stage 1: structure — max_depth x learning_rate x min_child_weight.
        stage1_best = None
        stage1_best_loss = np.inf
        for max_depth in (4, 6):
            for learning_rate in (0.03, 0.1):
                for min_child_weight in (1, 5):
                    params = {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "min_child_weight": min_child_weight,
                    }
                    val_loss = run(params, "stage1_structure")
                    if val_loss < stage1_best_loss:
                        stage1_best_loss = val_loss
                        stage1_best = params

        # Stage 2: regularization/sampling on top of the best stage-1 config.
        for subsample in (0.8, 1.0):
            for colsample_bytree in (0.8, 1.0):
                for reg_lambda in (1.0, 5.0):
                    params = {
                        **stage1_best,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "reg_lambda": reg_lambda,
                    }
                    run(params, "stage2_regularization")

    trials_df = pd.DataFrame(trials).sort_values("val_log_loss").reset_index(drop=True)
    return best_model, best_params, trials_df


def calibrate_xgb(base_xgb, X_val, y_val) -> CalibratedClassifierCV:
    """Isotonic-calibrate an already-fitted XGB using a frozen estimator.

    ``CalibratedClassifierCV(cv="prefit")`` was removed in sklearn 1.9; use
    ``FrozenEstimator`` to wrap the fitted base model instead.
    """
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base_xgb), method="isotonic"
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def save_models(models: dict, out_dir) -> None:
    """Persist each ``name -> estimator`` to ``out_dir/<name>.joblib``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out_dir / f"{name}.joblib")


def load_model(path):
    """Load a joblib-persisted model."""
    return joblib.load(path)


def predict_win_prob(model, X) -> np.ndarray:
    """P(win) = ``predict_proba(X)[:, 1]``."""
    return model.predict_proba(X)[:, 1]
