"""Run Phase 4: per-ball WPA credit + player/season clutch leaderboards.

Scores every ball of every match with the calibrated model (ties INCLUDED —
WP trajectories are descriptive), computes per-innings ΔWP on the correct
batting-team perspective, credits striker/bowler, persists per-ball WPA, and
builds overall + IPL-only batter/bowler clutch leaderboards per season and an
all-seasons aggregate. Writes reports/phase4_gate.md.

Hard gates (exit 1 on fail): per-innings ΔWP telescoping (|residual| < 1e-9)
on a sample of matches for BOTH innings formulas; per-ball credit balance
(batter_credit + bowler_credit == 0); all artifacts written.
Soft gates (warn, don't exit): top-10 IPL clutch batters look like
high-impact finishers/anchors (median balls-faced >= min_balls, no obvious
bowler-only names).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.models import load_model  # noqa: E402
from t20wp.wpa import (  # noqa: E402
    DEFAULT_MIN_BALLS,
    compute_ball_wp,
    compute_delta_wp,
    leaderboard,
)

# Leaderboard variants built for every (scope, role): a per-season board and an
# all-seasons aggregate. Maps variant name -> (`by` grouping, CSV filename
# suffix). Each variant is materialized per role as ``f"{role}_{variant}"``.
LEADERBOARD_VARIANTS = {
    "season": (("season",), ""),
    "all": ((), "_allseasons"),
}
ROLES = ("batter", "bowler")

# Recognizable high-impact IPL finishers/anchors used only for the soft
# eyeball gate (top-10 batter sanity). Not exhaustive — it warns, never fails.
KNOWN_IMPACT_BATTERS = {
    "V Kohli", "MS Dhoni", "AD Russell", "F du Plessis", "RR Pant",
    "AB de Villiers", "RG Sharma", "DA Warner", "KL Rahul", "SV Samson",
    "SA Yadav", "HH Pandya", "SS Iyer", "Shubman Gill", "Q de Kock",
    "JC Buttler", "GJ Maxwell", "N Pooran", "TM Head", "B Sai Sudharsan",
    "RD Gaikwad", "Ishan Kishan", "AT Rayudu", "KA Pollard", "MP Stoinis",
    "DA Miller", "SP Narine", "Tilak Varma", "PP Shaw", "YBK Jaiswal",
    "Abhishek Sharma", "RA Tripathi", "SR Watson", "CH Gayle", "RV Uthappa",
}

CHECKS: list[tuple[str, bool, str]] = []
WARNINGS: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}: {name} {detail}")


def warn(name: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "WARN"
    print(f"{status}: {name} {detail}")
    if not ok:
        WARNINGS.append(f"{name} {detail}")


def _telescope_residuals(match_df: pd.DataFrame) -> tuple[float, float]:
    """Per-innings ΔWP telescoping residuals for one match (both innings)."""
    m = match_df.sort_values(["innings", "ball_seq"])
    inn1 = m[m["innings"] == 1]
    inn2 = m[m["innings"] == 2]
    res1 = res2 = 0.0
    if len(inn1):
        last1 = inn1["wp"].iloc[-1]
        res1 = inn1["delta_wp"].sum() - (last1 - 0.5)
        if len(inn2):
            last2 = inn2["wp"].iloc[-1]
            res2 = inn2["delta_wp"].sum() - (last2 - (1.0 - last1))
    return abs(res1), abs(res2)


def main() -> None:
    models_dir = ROOT / "models"
    wpa_dir = ROOT / "reports/wpa"
    wpa_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    model = load_model(models_dir / "xgb_calibrated.joblib")

    print(f"Scoring {len(features):,} balls with calibrated model...")
    ball_wp = compute_ball_wp(features, model)
    print("Computing per-innings ΔWP + credits...")
    wpa = compute_delta_wp(ball_wp)

    # --- Persist per-ball WPA ----------------------------------------
    keep = [
        "match_id", "innings", "ball_seq", "season", "is_ipl",
        "batter", "bowler", "wp", "delta_wp", "batter_credit", "bowler_credit",
    ]
    balls_path = models_dir / "wpa_balls.parquet"
    wpa[keep].to_parquet(balls_path, index=False)
    print(f"Saved {balls_path}")

    # --- HARD: per-innings telescoping on a sample of matches --------
    sample_ids = (
        wpa[["match_id"]].drop_duplicates()["match_id"].head(200).tolist()
    )
    max_res1 = max_res2 = 0.0
    for mid in sample_ids:
        r1, r2 = _telescope_residuals(wpa[wpa["match_id"] == mid])
        max_res1 = max(max_res1, r1)
        max_res2 = max(max_res2, r2)
    check(
        "innings-1 ΔWP telescopes (sum == last_wp - 0.5)",
        max_res1 < 1e-9,
        f"(max |residual| = {max_res1:.2e} over {len(sample_ids)} matches)",
    )
    check(
        "innings-2 ΔWP telescopes (sum == last_wp - (1 - last_wp_inn1))",
        max_res2 < 1e-9,
        f"(max |residual| = {max_res2:.2e} over {len(sample_ids)} matches)",
    )

    # --- HARD: per-ball credit balance -------------------------------
    imbalance = float((wpa["batter_credit"] + wpa["bowler_credit"]).abs().max())
    check(
        "per-ball credit balance (batter_credit + bowler_credit == 0)",
        imbalance < 1e-9,
        f"(max |sum| = {imbalance:.2e})",
    )

    # --- Leaderboards -------------------------------------------------
    # Build every (scope, role, variant) board once, keyed by
    # `f"{role}_{variant}"` per scope, and derive the CSV filenames from the
    # same spec so the board set is declared in exactly one place.
    ipl = wpa[wpa["is_ipl"] == 1]
    scopes = {"all": wpa, "ipl": ipl}

    def build_set(df):
        return {
            f"{role}_{variant}": leaderboard(df, role, by=by, min_balls=DEFAULT_MIN_BALLS)
            for role in ROLES
            for variant, (by, _suffix) in LEADERBOARD_VARIANTS.items()
        }

    boards = {scope: build_set(df) for scope, df in scopes.items()}
    all_lb = boards["all"]
    ipl_lb = boards["ipl"]

    csvs = {
        f"{scope}_{role}_clutch{suffix}.csv": boards[scope][f"{role}_{variant}"]
        for scope in scopes
        for role in ROLES
        for variant, (_by, suffix) in LEADERBOARD_VARIANTS.items()
    }
    for name, tbl in csvs.items():
        tbl.to_csv(wpa_dir / name, index=False)
        print(f"Saved reports/wpa/{name} ({len(tbl)} rows)")

    # --- HARD: data artifacts written --------------------------------
    # Checked BEFORE the gate report is built so these appear in the report's
    # "Checks (hard gates)" audit trail. The gate report's own existence is
    # verified after it is written (it cannot assert its own existence here).
    data_artifacts = [balls_path] + [wpa_dir / name for name in csvs]
    for path in data_artifacts:
        check(f"artifact written: {path.name}", path.exists())

    # --- SOFT: IPL top-10 batter name sanity -------------------------
    top10 = ipl_lb["batter_all"].head(10)
    median_balls = float(top10["balls"].median()) if len(top10) else 0.0
    warn(
        "top-10 IPL clutch batters median balls-faced >= min_balls",
        median_balls >= DEFAULT_MIN_BALLS,
        f"(median = {median_balls:.0f}, min_balls = {DEFAULT_MIN_BALLS})",
    )
    top10_names = top10["batter"].tolist()
    n_known = sum(n in KNOWN_IMPACT_BATTERS for n in top10_names)
    warn(
        "top-10 IPL clutch batters include recognizable impact players",
        n_known >= 5,
        f"({n_known}/10 recognized: {top10_names})",
    )

    # A recent season present in IPL data for the report (prefer 2024/2025).
    ipl_seasons = sorted(ipl["season"].unique())
    recent = next((s for s in ("2025", "2024") if s in ipl_seasons),
                  ipl_seasons[-1] if ipl_seasons else None)

    # --- Report -------------------------------------------------------
    def md_table(tbl, n=20):
        t = tbl.head(n).copy()
        if "clutch" in t.columns:
            t["clutch"] = t["clutch"].round(3)
        return t.to_markdown(index=False)

    lines = ["# Phase 4 gate report — Win Probability Added (WPA)", ""]
    lines.append(
        "Per-ball ΔWP is computed on the CURRENT batting team's perspective, "
        "per innings. Innings-1 prior = 0.5; innings-2 prior = "
        "`1 - last_wp_inn1` (batting-first team's final WP converted to the "
        "chasing team's perspective). `batter_credit = +ΔWP` (striker), "
        "`bowler_credit = -ΔWP`. Bowler clutch = `sum(bowler_credit) = "
        "sum(-ΔWP)`, so a high positive score means the bowler most reduced "
        f"the batting team's WP. Minimum sample = {DEFAULT_MIN_BALLS} balls per group."
    )
    lines.append("")

    lines.append("## Top-20 IPL clutch batters (all seasons)")
    lines.extend(["", md_table(ipl_lb["batter_all"]), ""])
    lines.append("## Top-20 IPL clutch bowlers (all seasons)")
    lines.extend(["", md_table(ipl_lb["bowler_all"]), ""])

    if recent is not None:
        rb = ipl_lb["batter_season"]
        rbw = ipl_lb["bowler_season"]
        lines.append(f"## Top-20 IPL clutch batters ({recent})")
        lines.extend(["", md_table(rb[rb["season"] == recent]), ""])
        lines.append(f"## Top-20 IPL clutch bowlers ({recent})")
        lines.extend(["", md_table(rbw[rbw["season"] == recent]), ""])

    lines.append("## Checks (hard gates)")
    lines.append("")
    lines.extend(
        f"- {'PASS' if p else 'FAIL'} {name} {detail}" for name, p, detail in CHECKS
    )
    lines.append("")
    lines.append("## Soft checks (reported, non-blocking)")
    lines.append("")
    if WARNINGS:
        lines.extend(f"- WARN: {w}" for w in WARNINGS)
    else:
        lines.append("- all soft checks passed")
    lines.append("")

    gate_path = ROOT / "reports/phase4_gate.md"
    gate_path.write_text("\n".join(lines))
    print("Saved: reports/phase4_gate.md")

    # --- HARD: gate report written -----------------------------------
    # Verified after writing; the data artifacts were already checked above so
    # they are recorded in the report's Checks section.
    check(f"artifact written: {gate_path.name}", gate_path.exists())

    print("\n=== Top-20 IPL clutch batters (all seasons) ===")
    print(ipl_lb["batter_all"].head(20).to_string(index=False))
    print("\n=== Top-20 IPL clutch bowlers (all seasons) ===")
    print(ipl_lb["bowler_all"].head(20).to_string(index=False))

    if not all(p for _, p, _ in CHECKS):
        print("HARD GATE FAILURE")
        sys.exit(1)
    print("\nAll hard gates passed.")


if __name__ == "__main__":
    main()
