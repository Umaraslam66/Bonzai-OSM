# Overture vocab analysis (release 2026-04-15.0)

> `distinct_nonnull` excludes NULL. Thresholds are counts (floor).

## Field summary

| field                              |      total |   null_pct |   distinct_nonnull |   >=_1000 |   >=_10000 |   >=_100000 |   >=_1000000 |   singletons | action                                                                                                                      |
|:-----------------------------------|-----------:|-----------:|-------------------:|----------:|-----------:|------------:|-------------:|-------------:|:----------------------------------------------------------------------------------------------------------------------------|
| base_infrastructure_class          |  149530606 |       0    |                163 |       154 |        115 |          73 |           24 |            0 | KEEP all 163 values; 115/163 already >10k floor. Candidate for primary vocab layer.                                         |
| base_infrastructure_subtype        |  149530606 |       0    |                 18 |        18 |         17 |          17 |           10 |            0 | KEEP ALL 18 values (tight vocab, all well-populated).                                                                       |
| base_land_class                    |   73017393 |       0    |                 42 |        38 |         31 |          22 |            9 |            0 | KEEP all 42 values; 31/42 already >10k floor. Candidate for primary vocab layer.                                            |
| base_land_cover_subtype            |  123302114 |       0    |                 10 |        10 |         10 |          10 |            8 |            0 | KEEP ALL 10 values (tight vocab, all well-populated).                                                                       |
| base_land_subtype                  |   73017393 |       0    |                 13 |        12 |         11 |          11 |            8 |            0 | KEEP ALL 13 values (tight vocab, all well-populated).                                                                       |
| base_land_surface                  |   73017393 |      99.74 |                 23 |         6 |          2 |           1 |            0 |            0 | SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit. |
| base_land_use_class                |   54159685 |       0    |                109 |        98 |         67 |          34 |           10 |            0 | KEEP all 109 values; 67/109 already >10k floor. Candidate for primary vocab layer.                                          |
| base_land_use_subtype              |   54159685 |       0    |                 23 |        23 |         23 |          17 |            9 |            0 | KEEP ALL 23 values (tight vocab, all well-populated).                                                                       |
| base_water_class                   |   64677485 |       0    |                 35 |        32 |         23 |          13 |            8 |            0 | KEEP all 35 values; 23/35 already >10k floor. Candidate for primary vocab layer.                                            |
| base_water_subtype                 |   64677485 |       0    |                 12 |        12 |         11 |          10 |            7 |            0 | KEEP ALL 12 values (tight vocab, all well-populated).                                                                       |
| buildings_building_class           | 2542597608 |      94.48 |                 87 |        87 |         73 |          40 |           17 |            0 | SKIP-OR-BINARY: 94% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit.  |
| buildings_building_facade_material | 2542597608 |      99.92 |                 11 |        11 |         11 |           5 |            0 |            0 | SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit. |
| buildings_building_roof_material   | 2542597608 |      99.93 |                 14 |        13 |          9 |           5 |            0 |            0 | SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit. |
| buildings_building_roof_shape      | 2542597608 |      99.67 |                 14 |        12 |         10 |           6 |            2 |            1 | SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit. |
| buildings_building_subtype         | 2542597608 |      94.41 |                 13 |        13 |         13 |          12 |            8 |            0 | SKIP-OR-BINARY: 94% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit.  |
| divisions_division_class           |    4597236 |      27.61 |                  4 |         4 |          4 |           3 |            2 |            0 | KEEP all 4 values; 4/4 already >10k floor. Candidate for primary vocab layer.                                               |
| divisions_division_country         |    4597236 |       0.02 |                271 |       151 |         78 |          11 |            0 |            7 | REVIEW manually.                                                                                                            |
| divisions_division_subtype         |    4597236 |       0    |                  9 |         7 |          6 |           4 |            1 |            0 | KEEP ALL 9 values (tight vocab, all well-populated).                                                                        |
| places_alternate                   |   86164058 |       0    |               2070 |      1289 |        698 |         180 |           13 |           23 | TAIL-TRIM: keep 698 values >= 10,000, bucket the remaining 1372 into <other>. Or use hierarchical fallback.                 |
| places_place_basic_category        |   75495994 |       7.78 |                267 |       241 |        204 |         114 |           15 |            1 | KEEP all 267 values; 204/267 already >10k floor. Candidate for primary vocab layer.                                         |
| places_place_brand_wikidata        |   75495994 |      98.07 |               3039 |       258 |         19 |           0 |            0 |           74 | SKIP-OR-BINARY: 98% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit.  |
| places_place_operating_status      |   75495994 |       0    |                  3 |         2 |          1 |           1 |            1 |            0 | KEEP ALL 3 values (tight vocab, all well-populated).                                                                        |
| places_place_taxonomy_primary      |   75495994 |       3.98 |               1981 |      1246 |        668 |         173 |            7 |            9 | TAIL-TRIM: keep 668 values >= 10,000, bucket the remaining 1313 into <other>. Or use hierarchical fallback.                 |
| places_primary                     |   75495994 |       3.98 |               2080 |      1292 |        690 |         178 |            5 |           15 | TAIL-TRIM: keep 690 values >= 10,000, bucket the remaining 1390 into <other>. Or use hierarchical fallback.                 |
| transportation_segment_class       |  344140478 |       0.01 |                 24 |        24 |         22 |          18 |           16 |            0 | KEEP ALL 24 values (tight vocab, all well-populated).                                                                       |
| transportation_segment_subclass    |  344140478 |      89.77 |                  7 |         7 |          7 |           6 |            6 |            0 | SKIP-OR-BINARY: 90% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit.  |
| transportation_segment_subtype     |  344140478 |       0    |                  3 |         3 |          3 |           2 |            2 |            0 | KEEP ALL 3 values (tight vocab, all well-populated).                                                                        |

## Per-field top-5

### `base_infrastructure_class`
- total: **149,530,606**
- null: **0.0%** (0)
- distinct non-null: **163**
- above 10,000 floor: **115**
- above 100,000 aspirational: **73**
- singletons: 0
- recommended: _KEEP all 163 values; 115/163 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 19,147,789 | 12.81% | `power_pole` |
| 18,198,292 | 12.17% | `power_tower` |
| 12,673,961 | 8.48% | `crossing` |
| 9,090,867 | 6.08% | `fence` |
| 7,242,261 | 4.84% | `bridge` |

### `base_infrastructure_subtype`
- total: **149,530,606**
- null: **0.0%** (0)
- distinct non-null: **18**
- above 10,000 floor: **17**
- above 100,000 aspirational: **17**
- singletons: 0
- recommended: _KEEP ALL 18 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 48,418,057 | 32.38% | `power` |
| 32,840,870 | 21.96% | `barrier` |
| 25,171,805 | 16.83% | `transportation` |
| 18,621,239 | 12.45% | `transit` |
| 7,477,986 | 5.0% | `bridge` |

### `base_land_class`
- total: **73,017,393**
- null: **0.0%** (0)
- distinct non-null: **42**
- above 10,000 floor: **31**
- above 100,000 aspirational: **22**
- singletons: 0
- recommended: _KEEP all 42 values; 31/42 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 32,660,527 | 44.73% | `tree` |
| 12,179,534 | 16.68% | `wood` |
| 5,941,385 | 8.14% | `forest` |
| 5,425,824 | 7.43% | `scrub` |
| 4,465,696 | 6.12% | `wetland` |

### `base_land_cover_subtype`
- total: **123,302,114**
- null: **0.0%** (0)
- distinct non-null: **10**
- above 10,000 floor: **10**
- above 100,000 aspirational: **10**
- singletons: 0
- recommended: _KEEP ALL 10 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 34,679,621 | 28.13% | `shrub` |
| 20,034,385 | 16.25% | `forest` |
| 18,648,040 | 15.12% | `barren` |
| 17,795,413 | 14.43% | `grass` |
| 10,443,789 | 8.47% | `crop` |

### `base_land_subtype`
- total: **73,017,393**
- null: **0.0%** (0)
- distinct non-null: **13**
- above 10,000 floor: **11**
- above 100,000 aspirational: **11**
- singletons: 0
- recommended: _KEEP ALL 13 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 34,655,640 | 47.46% | `tree` |
| 18,120,919 | 24.82% | `forest` |
| 6,437,882 | 8.82% | `shrub` |
| 4,465,696 | 6.12% | `wetland` |
| 2,599,665 | 3.56% | `physical` |

### `base_land_surface`
- total: **73,017,393**
- null: **99.74%** (72,827,933)
- distinct non-null: **23**
- above 10,000 floor: **2**
- above 100,000 aspirational: **1**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 125,210 | 0.17% | `grass` |
| 47,116 | 0.06% | `sand` |
| 5,532 | 0.01% | `gravel` |
| 4,970 | 0.01% | `pebblestone` |
| 1,551 | 0.0% | `recreation_sand` |

### `base_land_use_class`
- total: **54,159,685**
- null: **0.0%** (0)
- distinct non-null: **109**
- above 10,000 floor: **67**
- above 100,000 aspirational: **34**
- singletons: 0
- recommended: _KEEP all 109 values; 67/109 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 11,301,237 | 20.87% | `farmland` |
| 10,381,885 | 19.17% | `residential` |
| 6,750,567 | 12.46% | `grass` |
| 5,345,984 | 9.87% | `meadow` |
| 2,605,896 | 4.81% | `pitch` |

### `base_land_use_subtype`
- total: **54,159,685**
- null: **0.0%** (0)
- distinct non-null: **23**
- above 10,000 floor: **23**
- above 100,000 aspirational: **17**
- singletons: 0
- recommended: _KEEP ALL 23 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 18,215,800 | 33.63% | `agriculture` |
| 10,505,114 | 19.4% | `residential` |
| 6,750,567 | 12.46% | `managed` |
| 4,895,706 | 9.04% | `horticulture` |
| 3,777,061 | 6.97% | `recreation` |

### `base_water_class`
- total: **64,677,485**
- null: **0.0%** (0)
- distinct non-null: **35**
- above 10,000 floor: **23**
- above 100,000 aspirational: **13**
- singletons: 0
- recommended: _KEEP all 35 values; 23/35 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 28,869,175 | 44.64% | `stream` |
| 17,344,552 | 26.82% | `water` |
| 4,630,804 | 7.16% | `ditch` |
| 3,127,280 | 4.84% | `pond` |
| 2,817,394 | 4.36% | `swimming_pool` |

### `base_water_subtype`
- total: **64,677,485**
- null: **0.0%** (0)
- distinct non-null: **12**
- above 10,000 floor: **11**
- above 100,000 aspirational: **10**
- singletons: 0
- recommended: _KEEP ALL 12 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 28,869,175 | 44.64% | `stream` |
| 17,499,776 | 27.06% | `water` |
| 7,533,007 | 11.65% | `canal` |
| 3,137,795 | 4.85% | `pond` |
| 2,841,906 | 4.39% | `human_made` |

### `buildings_building_class`
- total: **2,542,597,608**
- null: **94.48%** (2,402,315,214)
- distinct non-null: **87**
- above 10,000 floor: **73**
- above 100,000 aspirational: **40**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 94% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 64,790,454 | 2.55% | `house` |
| 15,899,993 | 0.63% | `residential` |
| 9,741,811 | 0.38% | `detached` |
| 8,006,431 | 0.31% | `garage` |
| 7,625,096 | 0.3% | `apartments` |

### `buildings_building_facade_material`
- total: **2,542,597,608**
- null: **99.92%** (2,540,498,303)
- distinct non-null: **11**
- above 10,000 floor: **11**
- above 100,000 aspirational: **5**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 574,174 | 0.02% | `brick` |
| 482,413 | 0.02% | `plaster` |
| 449,355 | 0.02% | `cement_block` |
| 229,756 | 0.01% | `wood` |
| 167,394 | 0.01% | `concrete` |

### `buildings_building_roof_material`
- total: **2,542,597,608**
- null: **99.93%** (2,540,747,463)
- distinct non-null: **14**
- above 10,000 floor: **9**
- above 100,000 aspirational: **5**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 974,148 | 0.04% | `roof_tiles` |
| 343,126 | 0.01% | `metal` |
| 141,907 | 0.01% | `concrete` |
| 132,764 | 0.01% | `eternit` |
| 127,352 | 0.01% | `tar_paper` |

### `buildings_building_roof_shape`
- total: **2,542,597,608**
- null: **99.67%** (2,534,247,900)
- distinct non-null: **14**
- above 10,000 floor: **10**
- above 100,000 aspirational: **6**
- singletons: 1
- recommended: _SKIP-OR-BINARY: 100% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 5,053,259 | 0.2% | `gabled` |
| 1,615,756 | 0.06% | `flat` |
| 945,274 | 0.04% | `hipped` |
| 255,421 | 0.01% | `pyramidal` |
| 167,600 | 0.01% | `skillion` |

### `buildings_building_subtype`
- total: **2,542,597,608**
- null: **94.41%** (2,400,422,156)
- distinct non-null: **13**
- above 10,000 floor: **13**
- above 100,000 aspirational: **12**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 94% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 114,756,469 | 4.51% | `residential` |
| 8,522,265 | 0.34% | `outbuilding` |
| 4,752,431 | 0.19% | `commercial` |
| 4,537,474 | 0.18% | `agricultural` |
| 3,640,749 | 0.14% | `industrial` |

### `divisions_division_class`
- total: **4,597,236**
- null: **27.61%** (1,269,380)
- distinct non-null: **4**
- above 10,000 floor: **4**
- above 100,000 aspirational: **3**
- singletons: 0
- recommended: _KEEP all 4 values; 4/4 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 1,678,614 | 36.51% | `hamlet` |
| 1,530,193 | 33.29% | `village` |
| 104,845 | 2.28% | `town` |
| 14,204 | 0.31% | `city` |

### `divisions_division_country`
- total: **4,597,236**
- null: **0.02%** (1,111)
- distinct non-null: **271**
- above 10,000 floor: **78**
- above 100,000 aspirational: **11**
- singletons: 7
- recommended: _REVIEW manually._

| count | pct | value |
|---:|---:|:---|
| 521,730 | 11.35% | `CN` |
| 356,459 | 7.75% | `IN` |
| 322,611 | 7.02% | `FR` |
| 220,058 | 4.79% | `JP` |
| 205,140 | 4.46% | `US` |

### `divisions_division_subtype`
- total: **4,597,236**
- null: **0.0%** (0)
- distinct non-null: **9**
- above 10,000 floor: **6**
- above 100,000 aspirational: **4**
- singletons: 0
- recommended: _KEEP ALL 9 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 3,446,286 | 74.96% | `locality` |
| 704,793 | 15.33% | `neighborhood` |
| 228,402 | 4.97% | `microhood` |
| 153,394 | 3.34% | `macrohood` |
| 38,840 | 0.84% | `county` |

### `places_alternate`
- total: **86,164,058**
- null: **0.0%** (0)
- distinct non-null: **2070**
- above 10,000 floor: **698**
- above 100,000 aspirational: **180**
- singletons: 23
- recommended: _TAIL-TRIM: keep 698 values >= 10,000, bucket the remaining 1372 into <other>. Or use hierarchical fallback._

| count | pct | value |
|---:|---:|:---|
| 3,386,033 | 3.93% | `restaurant` |
| 2,072,302 | 2.41% | `shopping` |
| 2,066,542 | 2.4% | `health_and_medical` |
| 1,749,215 | 2.03% | `professional_services` |
| 1,564,282 | 1.82% | `beauty_and_spa` |

### `places_place_basic_category`
- total: **75,495,994**
- null: **7.78%** (5,874,489)
- distinct non-null: **267**
- above 10,000 floor: **204**
- above 100,000 aspirational: **114**
- singletons: 1
- recommended: _KEEP all 267 values; 204/267 already >10k floor. Candidate for primary vocab layer._

| count | pct | value |
|---:|---:|:---|
| 5,557,446 | 7.36% | `restaurant` |
| 3,403,139 | 4.51% | `personal_or_beauty_service` |
| 2,731,132 | 3.62% | `fashion_and_apparel_store` |
| 2,419,026 | 3.2% | `home_service` |
| 1,983,670 | 2.63% | `professional_service` |

### `places_place_brand_wikidata`
- total: **75,495,994**
- null: **98.07%** (74,038,592)
- distinct non-null: **3039**
- above 10,000 floor: **19**
- above 100,000 aspirational: **0**
- singletons: 74
- recommended: _SKIP-OR-BINARY: 98% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 56,966 | 0.08% | `Q24933714` |
| 41,338 | 0.05% | `Q861042` |
| 34,633 | 0.05% | `Q1345267` |
| 33,241 | 0.04% | `Q16974764` |
| 32,563 | 0.04% | `Q857063` |

### `places_place_operating_status`
- total: **75,495,994**
- null: **0.0%** (0)
- distinct non-null: **3**
- above 10,000 floor: **1**
- above 100,000 aspirational: **1**
- singletons: 0
- recommended: _KEEP ALL 3 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 75,489,415 | 99.99% | `open` |
| 6,549 | 0.01% | `closed` |
| 30 | 0.0% | `temporarily closed` |

### `places_place_taxonomy_primary`
- total: **75,495,994**
- null: **3.98%** (3,008,015)
- distinct non-null: **1981**
- above 10,000 floor: **668**
- above 100,000 aspirational: **173**
- singletons: 9
- recommended: _TAIL-TRIM: keep 668 values >= 10,000, bucket the remaining 1313 into <other>. Or use hierarchical fallback._

| count | pct | value |
|---:|---:|:---|
| 1,797,257 | 2.38% | `restaurant` |
| 1,333,278 | 1.77% | `beauty_salon` |
| 1,295,132 | 1.72% | `professional_service` |
| 1,263,268 | 1.67% | `hotel` |
| 1,148,700 | 1.52% | `shopping` |

### `places_primary`
- total: **75,495,994**
- null: **3.98%** (3,006,406)
- distinct non-null: **2080**
- above 10,000 floor: **690**
- above 100,000 aspirational: **178**
- singletons: 15
- recommended: _TAIL-TRIM: keep 690 values >= 10,000, bucket the remaining 1390 into <other>. Or use hierarchical fallback._

| count | pct | value |
|---:|---:|:---|
| 1,797,248 | 2.38% | `restaurant` |
| 1,333,278 | 1.77% | `beauty_salon` |
| 1,295,132 | 1.72% | `professional_services` |
| 1,263,268 | 1.67% | `hotel` |
| 1,011,648 | 1.34% | `landmark_and_historical_building` |

### `transportation_segment_class`
- total: **344,140,478**
- null: **0.01%** (28,741)
- distinct non-null: **24**
- above 10,000 floor: **22**
- above 100,000 aspirational: **18**
- singletons: 0
- recommended: _KEEP ALL 24 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 127,190,332 | 36.96% | `residential` |
| 60,449,515 | 17.57% | `service` |
| 30,009,932 | 8.72% | `unclassified` |
| 26,056,895 | 7.57% | `track` |
| 23,542,848 | 6.84% | `footway` |

### `transportation_segment_subclass`
- total: **344,140,478**
- null: **89.77%** (308,936,491)
- distinct non-null: **7**
- above 10,000 floor: **7**
- above 100,000 aspirational: **6**
- singletons: 0
- recommended: _SKIP-OR-BINARY: 90% null — label mostly absent. Either drop this field from vocab, or compress to a single is_present bit._

| count | pct | value |
|---:|---:|:---|
| 17,792,302 | 5.17% | `driveway` |
| 6,588,285 | 1.91% | `parking_aisle` |
| 4,004,529 | 1.16% | `sidewalk` |
| 2,463,298 | 0.72% | `crosswalk` |
| 2,430,088 | 0.71% | `link` |

### `transportation_segment_subtype`
- total: **344,140,478**
- null: **0.0%** (0)
- distinct non-null: **3**
- above 10,000 floor: **3**
- above 100,000 aspirational: **2**
- singletons: 0
- recommended: _KEEP ALL 3 values (tight vocab, all well-populated). _

| count | pct | value |
|---:|---:|:---|
| 342,089,020 | 99.4% | `road` |
| 2,022,717 | 0.59% | `rail` |
| 28,741 | 0.01% | `water` |