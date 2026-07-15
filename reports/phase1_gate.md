# Phase 1 gate report

Matches parsed: **4707** | ball rows: **1,076,776**

## Matches by competition

| competition   |   matches |
|:--------------|----------:|
| t20i          |      3464 |
| ipl           |      1243 |

## Exclusions

|    | competition   | exclusion_reason     |   matches |
|---:|:--------------|:---------------------|----------:|
|  0 | ipl           | KEPT (model matches) |      1211 |
|  1 | ipl           | dls                  |        23 |
|  2 | ipl           | no_result            |         9 |
|  3 | t20i          | KEPT (model matches) |      3258 |
|  4 | t20i          | dls                  |       121 |
|  5 | t20i          | no_result            |        77 |
|  6 | t20i          | not_20_overs         |         8 |

## Model matches by competition

| competition   |   matches |
|:--------------|----------:|
| t20i          |      3258 |
| ipl           |      1211 |

## Outcome type (model matches)

|    | competition   | outcome_type   |   matches |
|---:|:--------------|:---------------|----------:|
|  0 | ipl           | tie            |        16 |
|  1 | ipl           | win            |      1195 |
|  2 | t20i          | tie            |        36 |
|  3 | t20i          | win            |      3222 |

## Chasing-team win rate (decided model matches)

| competition   |   mean |   count |
|:--------------|-------:|--------:|
| ipl           |  0.544 |    1195 |
| t20i          |  0.507 |    3222 |

## T20I model matches: full-member vs associate

| is_full_member_match   |   matches |
|:-----------------------|----------:|
| involves associate     |      2232 |
| full-member only       |      1026 |

## Model matches by season

| season   |   ipl |   t20i |
|:---------|------:|-------:|
| 2004/05  |     0 |      1 |
| 2005     |     0 |      1 |
| 2005/06  |     0 |      4 |
| 2006     |     0 |      2 |
| 2006/07  |     0 |      3 |
| 2007     |     0 |      2 |
| 2007/08  |    56 |     38 |
| 2008     |     0 |      5 |
| 2008/09  |     0 |     13 |
| 2009     |    54 |     31 |
| 2009/10  |    60 |     18 |
| 2010     |     0 |     31 |
| 2010/11  |     0 |     11 |
| 2011     |    69 |     10 |
| 2011/12  |     0 |     28 |
| 2012     |    74 |     15 |
| 2012/13  |     0 |     41 |
| 2013     |    76 |     14 |
| 2013/14  |     0 |     57 |
| 2014     |    59 |      3 |
| 2014/15  |     0 |     10 |
| 2015     |    55 |     26 |
| 2015/16  |     0 |     80 |
| 2016     |    56 |      9 |
| 2016/17  |     0 |     26 |
| 2017     |    58 |     13 |
| 2017/18  |     0 |     37 |
| 2018     |    57 |     27 |
| 2018/19  |     0 |     47 |
| 2019     |    59 |     67 |
| 2019/20  |     0 |    140 |
| 2020     |     0 |      9 |
| 2020/21  |    60 |     34 |
| 2021     |    60 |    109 |
| 2021/22  |     0 |    194 |
| 2022     |    74 |    231 |
| 2022/23  |     0 |    173 |
| 2023     |    71 |    167 |
| 2023/24  |     0 |    214 |
| 2024     |    71 |    318 |
| 2024/25  |     0 |    207 |
| 2025     |    70 |    297 |
| 2025/26  |     0 |    279 |
| 2026     |    72 |    216 |

## 1st-innings sanity (model matches)

| tier             |   avg_runs |   med_runs |   avg_legal_balls |   matches |
|:-----------------|-----------:|-----------:|------------------:|----------:|
| ipl              |      169.2 |        168 |             119.1 |      1211 |
| t20i_associate   |      143   |        144 |             116.4 |      2232 |
| t20i_full_member |      163.7 |        165 |             118.3 |      1026 |

## Structural checks

- innings values: [1, 2, 3, 4, 5, 6, 7, 8]
- max legal balls in an innings: 120
- super-over ball rows: 576
- 2nd-innings rows missing target: 2271
- distinct venues (model matches): 341
- distinct cities (model matches): 203
- date range: 2005-02-17 → 2026-07-13
