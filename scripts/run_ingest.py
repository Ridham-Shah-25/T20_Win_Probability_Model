"""Run Phase 1 ingestion and write the gate report to reports/phase1_gate.md."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t20wp.ingest import ingest  # noqa: E402


def main() -> None:
    matches, balls = ingest(
        raw_dirs={"ipl": ROOT / "data/raw/ipl", "t20i": ROOT / "data/raw/t20s"},
        out_dir=ROOT / "data/processed",
    )

    lines = ["# Phase 1 gate report", ""]

    def section(title: str, obj) -> None:
        lines.extend([f"## {title}", "", obj.to_markdown(), ""])

    lines.append(f"Matches parsed: **{len(matches)}** | ball rows: **{len(balls):,}**")
    lines.append("")

    section("Matches by competition", matches["competition"].value_counts().to_frame("matches"))

    section(
        "Exclusions",
        matches.groupby(["competition", "exclusion_reason"])
        .size()
        .to_frame("matches")
        .reset_index()
        .replace({"exclusion_reason": {"": "KEPT (model matches)"}}),
    )

    model = matches[matches["is_model_match"]]
    section(
        "Model matches by competition",
        model["competition"].value_counts().to_frame("matches"),
    )

    section(
        "Outcome type (model matches)",
        model.groupby(["competition", "outcome_type"]).size().to_frame("matches").reset_index(),
    )

    # Chasing-team win rate among decided model matches
    decided = model[model["outcome_type"] == "win"].copy()
    decided["chasing_won"] = decided["win_by_wickets"] > 0
    section(
        "Chasing-team win rate (decided model matches)",
        decided.groupby("competition")["chasing_won"].agg(["mean", "count"]).round(3),
    )

    # T20I tier split
    t20i = model[model["competition"] == "t20i"]
    section(
        "T20I model matches: full-member vs associate",
        t20i["is_full_member_match"]
        .map({True: "full-member only", False: "involves associate"})
        .value_counts()
        .to_frame("matches"),
    )

    section(
        "Model matches by season",
        model.groupby(["season", "competition"]).size().unstack(fill_value=0),
    )

    # Sanity: 1st-innings runs and balls for full 20-over first innings
    b = balls[~balls["is_super_over"] & balls["match_id"].isin(model["match_id"])]
    inn1 = b[b["innings"] == 1].groupby("match_id").agg(
        runs=("runs_total", "sum"), legal_balls=("is_legal", "sum")
    )
    inn1 = inn1.join(model.set_index("match_id")[["competition", "is_full_member_match"]])
    inn1["tier"] = inn1["competition"]
    inn1.loc[(inn1["competition"] == "t20i") & ~inn1["is_full_member_match"], "tier"] = (
        "t20i_associate"
    )
    inn1.loc[(inn1["competition"] == "t20i") & inn1["is_full_member_match"], "tier"] = (
        "t20i_full_member"
    )
    section(
        "1st-innings sanity (model matches)",
        inn1.groupby("tier").agg(
            avg_runs=("runs", "mean"),
            med_runs=("runs", "median"),
            avg_legal_balls=("legal_balls", "mean"),
            matches=("runs", "count"),
        ).round(1),
    )

    # Structural checks
    checks = {
        "innings values": sorted(balls["innings"].unique().tolist()),
        "max legal balls in an innings": int(inn1["legal_balls"].max()),
        "super-over ball rows": int(balls["is_super_over"].sum()),
        "2nd-innings rows missing target": int(
            (b[(b["innings"] == 2)]["target_runs"] == 0).sum()
        ),
        "distinct venues (model matches)": int(model["venue"].nunique()),
        "distinct cities (model matches)": int(model["city"].nunique()),
        "date range": f"{matches['date'].min().date()} → {matches['date'].max().date()}",
    }
    lines.extend(["## Structural checks", ""])
    lines.extend(f"- {k}: {v}" for k, v in checks.items())
    lines.append("")

    report = "\n".join(lines)
    out = ROOT / "reports/phase1_gate.md"
    out.write_text(report)
    print(report)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
