"""Parse Cricsheet T20 JSON into ball-by-ball and match-level parquet tables.

Every match is kept in the output with exclusion flags (DLS, no result, tie,
non-standard format) rather than being dropped here, so downstream stages and
the gate report can see exactly what was excluded and why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FULL_MEMBERS = {
    "Afghanistan",
    "Australia",
    "Bangladesh",
    "England",
    "India",
    "Ireland",
    "New Zealand",
    "Pakistan",
    "South Africa",
    "Sri Lanka",
    "West Indies",
    "Zimbabwe",
}


def parse_match(path: Path, competition: str) -> tuple[dict, list[dict]]:
    """Parse one Cricsheet JSON file into a match-metadata row and ball rows."""
    with open(path) as f:
        data = json.load(f)

    info = data["info"]
    match_id = path.stem
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    teams = info["teams"]

    outcome_type = "win"
    if "winner" not in outcome:
        result = outcome.get("result", "no result")
        outcome_type = "tie" if result == "tie" else "no_result"

    match_row = {
        "match_id": match_id,
        "competition": competition,
        "season": str(info.get("season", "")),
        "date": info["dates"][0],
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "team1": teams[0],
        "team2": teams[1],
        "gender": info.get("gender", ""),
        "match_type": info.get("match_type", ""),
        "overs": info.get("overs", 0),
        "balls_per_over": info.get("balls_per_over", 6),
        "toss_winner": info.get("toss", {}).get("winner", ""),
        "toss_decision": info.get("toss", {}).get("decision", ""),
        "outcome_type": outcome_type,
        "winner": outcome.get("winner", ""),
        "win_by_runs": by.get("runs", 0),
        "win_by_wickets": by.get("wickets", 0),
        "tie_decided_by": outcome.get("eliminator", "")
        or outcome.get("bowl_out", ""),
        "is_dls": "method" in outcome,
        "dls_method": outcome.get("method", ""),
        "event_name": info.get("event", {}).get("name", ""),
        "is_full_member_match": all(t in FULL_MEMBERS for t in teams),
    }

    ball_rows = []
    for inn_idx, innings in enumerate(data.get("innings", []), start=1):
        is_super_over = bool(innings.get("super_over", False))
        batting_team = innings["team"]
        bowling_team = teams[1] if batting_team == teams[0] else teams[0]
        target = innings.get("target", {})

        for over in innings.get("overs", []):
            for ball_idx, d in enumerate(over["deliveries"], start=1):
                extras = d.get("extras", {})
                wickets = d.get("wickets", [])
                ball_rows.append(
                    {
                        "match_id": match_id,
                        "innings": inn_idx,
                        "is_super_over": is_super_over,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "over": over["over"],
                        "ball_in_over": ball_idx,
                        "batter": d["batter"],
                        "non_striker": d["non_striker"],
                        "bowler": d["bowler"],
                        "runs_batter": d["runs"]["batter"],
                        "runs_extras": d["runs"]["extras"],
                        "runs_total": d["runs"]["total"],
                        "wides": extras.get("wides", 0),
                        "noballs": extras.get("noballs", 0),
                        "byes": extras.get("byes", 0),
                        "legbyes": extras.get("legbyes", 0),
                        "penalty": extras.get("penalty", 0),
                        "is_legal": "wides" not in extras
                        and "noballs" not in extras,
                        "is_wicket": len(wickets) > 0,
                        "wicket_kind": wickets[0]["kind"] if wickets else "",
                        "player_out": wickets[0]["player_out"] if wickets else "",
                        "n_wickets_on_ball": len(wickets),
                        "target_runs": target.get("runs", 0),
                        "target_overs": target.get("overs", 0),
                    }
                )

    return match_row, ball_rows


def ingest(raw_dirs: dict[str, Path], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all matches from {competition: dir} into parquet tables in out_dir."""
    match_rows, ball_rows = [], []
    for competition, raw_dir in raw_dirs.items():
        for path in sorted(raw_dir.glob("*.json")):
            m, balls = parse_match(path, competition)
            match_rows.append(m)
            ball_rows.extend(balls)

    matches = pd.DataFrame(match_rows)
    balls = pd.DataFrame(ball_rows)

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # A match is usable for the WP model only with a clean binary outcome
    # under standard 20-over rules. Ties are kept (labelled 0.5 or excluded
    # at the feature stage); DLS and no-result matches are not.
    matches["exclusion_reason"] = ""
    matches.loc[matches["outcome_type"] == "no_result", "exclusion_reason"] = "no_result"
    matches.loc[matches["is_dls"], "exclusion_reason"] = "dls"
    matches.loc[matches["overs"] != 20, "exclusion_reason"] = "not_20_overs"
    matches.loc[matches["gender"] != "male", "exclusion_reason"] = "not_male"
    matches["is_model_match"] = matches["exclusion_reason"] == ""

    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_dir / "matches.parquet", index=False)
    balls.to_parquet(out_dir / "balls.parquet", index=False)
    return matches, balls
