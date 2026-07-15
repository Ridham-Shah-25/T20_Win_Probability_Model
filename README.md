# T20 Win Probability Model

Ball-by-ball win probability for T20 cricket, trained on Cricsheet data
(IPL + men's T20 internationals, ~4,500 matches). Predicts P(batting team
wins) after every delivery, with a Win Probability Added (WPA) metric
derived per player.

> Work in progress — Phase 1 (data ingestion) complete.

## Data

Ball-by-ball JSON from [Cricsheet](https://cricsheet.org/):

- `ipl_json.zip` — all IPL matches (2008–)
- `t20s_male_json.zip` — all men's T20 internationals (2005–)

Place both zips in `data/raw/` and extract to `data/raw/ipl/` and
`data/raw/t20s/`, then run:

```bash
pip install -r requirements.txt
python scripts/run_ingest.py
```

This writes two tables to `data/processed/`:

- `matches.parquet` — one row per match: teams, venue, date, outcome,
  and exclusion flags (DLS-affected, no-result, and non-standard matches
  are flagged rather than silently dropped)
- `balls.parquet` — one row per delivery (~1.07M rows): game state, runs,
  extras, wickets, super-over flags, chase target

and a data-quality gate report to `reports/phase1_gate.md`.

## Methodology notes

- **DLS / no-result matches are excluded** — the model targets a clean
  binary outcome under standard 20-over rules.
- **Super overs are flagged and excluded** from training rows; tied
  matches are handled at the labelling stage.
- **All train/test splitting is by match and by time** (train on older
  seasons, test on recent) — never random by row, since deliveries within
  a match share an outcome label.
- Venue and team features are computed only from matches strictly before
  the ball being predicted (no leakage).

## Repo structure

```
src/t20wp/       package code (ingestion, features, models, evaluation)
scripts/         runnable entry points, one per phase
reports/         generated gate reports and evaluation outputs
notebooks/       showcase notebooks (thin — logic lives in src/)
data/            raw + processed data (gitignored, reproducible)
```
