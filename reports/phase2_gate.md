# Phase 2 gate report

Feature rows: **1,040,499** across **4,469** matches | features: 19

## Frozen split (by match, by time)

|       |   matches | first date   | last date   |
|:------|----------:|:-------------|:------------|
| train |      3355 | 2005-02-17   | 2024-10-22  |
| val   |       445 | 2024-10-23   | 2025-07-21  |
| test  |       669 | 2025-07-22   | 2026-07-13  |

## Checks

- ✅ splits partition all model matches 
- ✅ split date separation (train < val < test) 
- ✅ no negative balls_remaining 
- ✅ no negative wickets_in_hand 
- ✅ no NaNs in unconditional features 
- ✅ chase features present on all 2nd-innings rows 
- ✅ labels constant within match-innings 
- ✅ labels complementary across innings (4469 matches with both innings)
- ✅ won chases end with runs_required <= 0 (2282 chases)
- ✅ venue_par leakage-safe (546410) expected 138.35, got 138.35
- ✅ venue_par leakage-safe (1436483) expected 138.29, got 138.29
- ✅ venue_par leakage-safe (1472526) expected 164.95, got 164.95
- ✅ venue_par leakage-safe (1422121) expected 178.26, got 178.26
- ✅ venue_par leakage-safe (1256722) expected 155.17, got 155.17
- ✅ venue_par leakage-safe (1299575) expected 146.29, got 146.29
- ✅ venue_par leakage-safe (548312) expected 154.79, got 154.79
- ✅ venue_par leakage-safe (1462913) expected 134.90, got 134.90

## Feature summary

|                    |           count |   mean |   std |     min |    25% |    50% |    75% |     max |
|:-------------------|----------------:|-------:|------:|--------:|-------:|-------:|-------:|--------:|
| is_second_innings  |      1.0405e+06 |   0.47 |  0.5  |    0    |   0    |   0    |   1    |    1    |
| balls_remaining    |      1.0405e+06 |  62.49 | 33.9  |    0    |  34    |  63    |  92    |  120    |
| wickets_in_hand    |      1.0405e+06 |   7.27 |  2.27 |    0    |   6    |   8    |   9    |   10    |
| score              |      1.0405e+06 |  71.25 | 48.15 |    0    |  32    |  66    | 104    |  344    |
| current_run_rate   |      1.0405e+06 |   7.29 |  2.65 |    0    |   5.79 |   7.22 |   8.7  |   78    |
| runs_required      | 492343          |  92.9  | 51.86 |   -5    |  52    |  91    | 130    |  345    |
| required_run_rate  | 492343          |   9.86 |  6.33 |    0    |   6.48 |   8.67 |  11.19 |   36    |
| rrr_minus_crr      | 492343          |   2.55 |  7.17 |  -59.55 |  -1.43 |   1.64 |   4.95 |   35.2  |
| projected_score    | 548156          | 145.37 | 51.71 |    0    | 116    | 144.62 | 173.79 | 1560    |
| projected_vs_par   | 548156          |  -6.38 | 50.38 | -231    | -35.05 |  -7.48 |  20.35 | 1402.76 |
| target_vs_par      | 492343          |   8.56 | 36    | -125.65 | -14.95 |   8.39 |  30.69 |  201.58 |
| venue_par          |      1.0405e+06 | 151.95 | 13.56 |   98.45 | 141.68 | 152.76 | 161.56 |  231    |
| runs_last_24       |      1.0405e+06 |  26.37 | 12.42 |    0    |  18    |  26    |  34    |   97    |
| wickets_last_24    |      1.0405e+06 |   1.07 |  1.04 |    0    |   0    |   1    |   2    |    8    |
| batting_strength   |      1.0405e+06 |   0.5  |  0.12 |    0.15 |   0.43 |   0.5  |   0.58 |    0.84 |
| bowling_strength   |      1.0405e+06 |   0.5  |  0.12 |    0.15 |   0.43 |   0.5  |   0.59 |    0.84 |
| strength_diff      |      1.0405e+06 |  -0    |  0.16 |   -0.61 |  -0.1  |   0    |   0.1  |    0.61 |
| is_ipl             |      1.0405e+06 |   0.28 |  0.45 |    0    |   0    |   0    |   1    |    1    |
| is_associate_match |      1.0405e+06 |   0.49 |  0.5  |    0    |   0    |   0    |   1    |    1    |

## Walkthrough: WT20 2016 final, WI chasing 156 (last 10 balls)

|   over |   ball_in_over | batter        | bowler    |   score |   wickets_in_hand |   balls_remaining |   runs_required |   required_run_rate |   rrr_minus_crr |   runs_last_24 |   wickets_last_24 |   won |
|-------:|---------------:|:--------------|:----------|--------:|------------------:|------------------:|----------------:|--------------------:|----------------:|---------------:|------------------:|------:|
|     18 |              1 | MN Samuels    | CJ Jordan |     133 |                 4 |                11 |              23 |               12.55 |            5.22 |             42 |                 2 |     1 |
|     18 |              2 | MN Samuels    | CJ Jordan |     134 |                 4 |                10 |              22 |               13.2  |            5.89 |             42 |                 2 |     1 |
|     18 |              3 | CR Brathwaite | CJ Jordan |     135 |                 4 |                 9 |              21 |               14    |            6.7  |             37 |                 2 |     1 |
|     18 |              4 | MN Samuels    | CJ Jordan |     136 |                 4 |                 8 |              20 |               15    |            7.71 |             32 |                 2 |     1 |
|     18 |              5 | CR Brathwaite | CJ Jordan |     137 |                 4 |                 7 |              19 |               16.29 |            9.01 |             33 |                 1 |     1 |
|     18 |              6 | MN Samuels    | CJ Jordan |     137 |                 4 |                 6 |              19 |               19    |           11.79 |             32 |                 1 |     1 |
|     19 |              1 | CR Brathwaite | BA Stokes |     143 |                 4 |                 5 |              13 |               15.6  |            8.14 |             36 |                 1 |     1 |
|     19 |              2 | CR Brathwaite | BA Stokes |     149 |                 4 |                 4 |               7 |               10.5  |            2.79 |             42 |                 0 |     1 |
|     19 |              3 | CR Brathwaite | BA Stokes |     155 |                 4 |                 3 |               1 |                2    |           -5.95 |             48 |                 0 |     1 |
|     19 |              4 | CR Brathwaite | BA Stokes |     161 |                 4 |                 2 |              -5 |                0    |           -8.19 |             52 |                 0 |     1 |
