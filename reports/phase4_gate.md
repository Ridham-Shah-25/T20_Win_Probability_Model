# Phase 4 gate report — Win Probability Added (WPA)

Per-ball ΔWP is computed on the CURRENT batting team's perspective, per innings. Innings-1 prior = 0.5; innings-2 prior = `1 - last_wp_inn1` (batting-first team's final WP converted to the chasing team's perspective). `batter_credit = +ΔWP` (striker), `bowler_credit = -ΔWP`. Bowler clutch = `sum(bowler_credit) = sum(-ΔWP)`, so a high positive score means the bowler most reduced the batting team's WP. Minimum sample = 300 balls per group.

## Top-20 IPL clutch batters (all seasons)

| batter         |   clutch |   balls |
|:---------------|---------:|--------:|
| AB de Villiers |   10.061 |    3411 |
| CH Gayle       |    7.353 |    3367 |
| V Sehwag       |    6.841 |    1787 |
| JC Buttler     |    6.414 |    3131 |
| KA Pollard     |    6.067 |    2433 |
| AD Russell     |    5.773 |    1586 |
| DA Warner      |    5.647 |    4697 |
| SK Raina       |    5.403 |    4102 |
| V Kohli        |    5.296 |    7022 |
| RR Pant        |    5.237 |    2658 |
| SA Yadav       |    4.855 |    3127 |
| GJ Maxwell     |    4.79  |    1870 |
| RG Sharma      |    4.467 |    5648 |
| N Pooran       |    4.436 |    1571 |
| YK Pathan      |    4.429 |    2278 |
| Shubman Gill   |    3.97  |    3260 |
| S Dhawan       |    3.752 |    5316 |
| V Suryavanshi  |    3.692 |     473 |
| RV Uthappa     |    3.663 |    3874 |
| SS Iyer        |    3.594 |    3129 |

## Top-20 IPL clutch bowlers (all seasons)

| bowler          |   clutch |   balls |
|:----------------|---------:|--------:|
| SP Narine       |   11.667 |    4583 |
| B Kumar         |   10.163 |    4666 |
| JJ Bumrah       |    6.951 |    3743 |
| Rashid Khan     |    6.913 |    3516 |
| SL Malinga      |    6.646 |    2974 |
| R Ashwin        |    6.576 |    4737 |
| DW Steyn        |    4.607 |    2228 |
| YS Chahal       |    4.563 |    4117 |
| Harbhajan Singh |    3.944 |    3496 |
| M Muralitharan  |    3.376 |    1528 |
| A Kumble        |    3.141 |     983 |
| AR Patel        |    3.108 |    3600 |
| Kuldeep Yadav   |    2.959 |    2317 |
| PP Chawla       |    2.538 |    3793 |
| Z Khan          |    2.491 |    2237 |
| RA Jadeja       |    2.207 |    4281 |
| DP Nannes       |    1.906 |     668 |
| TA Boult        |    1.834 |    2800 |
| CV Varun        |    1.826 |    2138 |
| MJ McClenaghan  |    1.806 |    1346 |

## Top-20 IPL clutch batters (2025)

| batter          |   season |   clutch |   balls |
|:----------------|---------:|---------:|--------:|
| SS Iyer         |     2025 |    2.132 |     344 |
| B Sai Sudharsan |     2025 |    0.913 |     504 |
| SA Yadav        |     2025 |    0.904 |     415 |
| KL Rahul        |     2025 |    0.801 |     352 |
| MR Marsh        |     2025 |    0.744 |     401 |
| AK Markram      |     2025 |    0.641 |     306 |
| YBK Jaiswal     |     2025 |    0.616 |     359 |
| JC Buttler      |     2025 |    0.498 |     308 |
| V Kohli         |     2025 |    0.474 |     466 |
| Shubman Gill    |     2025 |    0.332 |     386 |

## Top-20 IPL clutch bowlers (2025)

| bowler            |   season |   clutch |   balls |
|:------------------|---------:|---------:|--------:|
| Kuldeep Yadav     |     2025 |    0.885 |     311 |
| DS Rathi          |     2025 |    0.291 |     321 |
| Yash Dayal        |     2025 |    0.278 |     305 |
| Mohammed Siraj    |     2025 |    0.223 |     349 |
| M Prasidh Krishna |     2025 |    0.172 |     340 |
| Arshdeep Singh    |     2025 |    0.127 |     367 |
| Suyash Sharma     |     2025 |    0.078 |     305 |
| TA Boult          |     2025 |   -0.047 |     333 |
| Noor Ahmad        |     2025 |   -0.092 |     314 |
| B Kumar           |     2025 |   -0.204 |     318 |
| Rashid Khan       |     2025 |   -0.301 |     312 |

## Checks (hard gates)

- PASS innings-1 ΔWP telescopes (sum == last_wp - 0.5) (max |residual| = 0.00e+00 over 200 matches)
- PASS innings-2 ΔWP telescopes (sum == last_wp - (1 - last_wp_inn1)) (max |residual| = 0.00e+00 over 200 matches)
- PASS per-ball credit balance (batter_credit + bowler_credit == 0) (max |sum| = 0.00e+00)
- PASS artifact written: wpa_balls.parquet 
- PASS artifact written: all_batter_clutch.csv 
- PASS artifact written: all_batter_clutch_allseasons.csv 
- PASS artifact written: all_bowler_clutch.csv 
- PASS artifact written: all_bowler_clutch_allseasons.csv 
- PASS artifact written: ipl_batter_clutch.csv 
- PASS artifact written: ipl_batter_clutch_allseasons.csv 
- PASS artifact written: ipl_bowler_clutch.csv 
- PASS artifact written: ipl_bowler_clutch_allseasons.csv 

## Soft checks (reported, non-blocking)

- all soft checks passed
