"""Evaluation harness: metrics, calibration data/plots, WP trajectories.

Pure metrics + plotting functions on ``(y_true, y_prob)`` arrays plus a
DataFrame-based WP-trajectory plotter. This module intentionally does NOT
import any model libraries (xgboost / sklearn estimators) so the showcase
notebook can import it cheaply for plotting only. Calibration *metrics* use
``sklearn.metrics`` (log loss / Brier), which is a light dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def evaluate_probs(y_true, y_prob) -> dict:
    """Core probabilistic metrics for a set of predictions.

    Returns ``{"log_loss", "brier", "n", "base_rate"}``. ``log_loss`` is
    computed with explicit ``labels=[0, 1]`` so it is well-defined even when
    a slice happens to contain a single class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
    }


def _bin_ids(y_prob: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Assign each prediction to a quantile bin.

    Quantile edges via ``np.quantile`` + ``np.unique`` (drops duplicate edges
    that arise with tied probabilities, e.g. isotonic output), then
    ``np.digitize``. Returns ``(bin_ids, n_effective_bins)``.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    # Interior edges only; digitize maps to [0, len(interior)] bins.
    interior = edges[1:-1]
    bin_ids = np.digitize(y_prob, interior, right=False)
    return bin_ids, len(interior) + 1


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-curve data with columns ``prob_pred, prob_true, count``.

    Self-contained reimplementation: bin IDs are computed ONCE (quantile edges
    deduped via ``np.unique``, then ``np.digitize``) and ``prob_pred`` (mean
    predicted per bin), ``prob_true`` (mean outcome per bin) and ``count`` are
    all derived from those same bin IDs in a single pass. This keeps the three
    columns aligned even with tied probabilities and avoids mixing
    ``sklearn.calibration_curve`` output (which returns ``(prob_true,
    prob_pred)`` and no counts) with a separate ``qcut`` counting pass.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_ids, _ = _bin_ids(y_prob, n_bins)

    df = pd.DataFrame({"bin": bin_ids, "y": y_true, "p": y_prob})
    grp = df.groupby("bin", sort=True)
    out = pd.DataFrame(
        {
            "prob_pred": grp["p"].mean(),
            "prob_true": grp["y"].mean(),
            "count": grp["y"].size(),
        }
    ).reset_index(drop=True)
    return out[["prob_pred", "prob_true", "count"]]


def reliability_stats(y_true, y_prob, n_bins: int = 10) -> dict:
    """Expected / maximum calibration error from the calibration table.

    ``MCE = max(|prob_true - prob_pred|)``;
    ``ECE = sum(count / N * |prob_true - prob_pred|)`` over bins.
    """
    tbl = calibration_table(y_true, y_prob, n_bins=n_bins)
    gap = (tbl["prob_true"] - tbl["prob_pred"]).abs()
    total = tbl["count"].sum()
    ece = float((tbl["count"] / total * gap).sum()) if total else 0.0
    mce = float(gap.max()) if len(gap) else 0.0
    return {"ece": ece, "mce": mce}


def plot_calibration(curves: dict, out_path) -> None:
    """One reliability-diagram PNG, one line per model plus a diagonal.

    ``curves`` maps model name -> calibration_table DataFrame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, tbl in curves.items():
        ax.plot(tbl["prob_pred"], tbl["prob_true"], marker="o", label=name)
    ax.set_xlabel("Mean predicted P(win)")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_wp_trajectory(traj_df: pd.DataFrame, out_path, title: str, team: str) -> None:
    """Plot P(``team`` wins) across a match on a FIXED team perspective.

    ``traj_df`` must contain: ``ball_seq``, ``innings``, ``batting_team``,
    ``wp`` (batting-team-relative), and event columns ``is_wicket``,
    ``runs_batter`` (from ``balls.parquet``). The stored ``wp`` is transformed
    to the fixed ``team`` perspective: rows where ``batting_team == team`` use
    ``wp``, others use ``1 - wp``. The x axis is a global delivery index across
    both innings (``ball_seq`` counts all deliveries incl. wides/no-balls).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = traj_df.sort_values(["innings", "ball_seq"]).reset_index(drop=True)
    df["wp_team"] = np.where(df["batting_team"] == team, df["wp"], 1.0 - df["wp"])
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, df["wp_team"], color="#1f77b4", lw=1.5, label=f"P({team} win)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")

    # innings boundary
    if (df["innings"] == 2).any():
        boundary = int((df["innings"] == 1).sum())
        ax.axvline(boundary - 0.5, color="black", lw=0.8, ls="--", alpha=0.6)

    wk = df[df["is_wicket"].fillna(False).astype(bool)]
    ax.scatter(
        x[wk.index], df.loc[wk.index, "wp_team"], color="red", marker="v",
        s=45, zorder=5, label="wicket",
    )
    boundary_mask = df["runs_batter"].isin([4, 6])
    bd = df[boundary_mask]
    ax.scatter(
        x[bd.index], df.loc[bd.index, "wp_team"], color="green", marker="^",
        s=35, zorder=4, label="4/6",
    )

    ax.set_xlabel("Delivery (global index across both innings)")
    ax.set_ylabel(f"P({team} wins)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def metrics_table(results: dict) -> pd.DataFrame:
    """Assemble a per-model metrics table for markdown reporting.

    ``results`` maps model name -> dict from ``evaluate_probs`` (optionally
    augmented with ``ece``/``mce``). Row order follows insertion order.
    """
    rows = []
    for name, m in results.items():
        row = {"model": name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)
