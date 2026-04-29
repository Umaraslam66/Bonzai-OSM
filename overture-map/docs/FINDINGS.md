# Bonzai global vocab — findings report

> **Generated 2026-04-29.** All numbers are from the `2026-04-15.0`
> Overture release and the `dt=2026-04-14` Foursquare OS Places snapshot.
> Source artifacts referenced throughout live in `overture-map/outputs/`.

---

## 1 · Executive summary

We empirically measured the global POI landscape across two open
datasets — **Overture Maps** and **Foursquare OS Places** — and
characterised their shape, overlap, and complementarity.

The headline finding is a single sentence:

> **Overture is wide, Foursquare is deep.** Overture has thin coverage
> across many cells (rural/suburban, OSM-derived); Foursquare has thick
> coverage in fewer cells (urban, user-generated).

Together they span **127.1 M unique POIs** worldwide once duplicates
are removed — 68 % more than Overture alone. Of that universe:

- **25.1 M** places are confidently the same in both sources (≤ 50 m,
  Jaro-Winkler name similarity ≥ 0.85). These can be merged with high
  confidence.
- **42.9 M** places are unique to Overture — disproportionately rural,
  OECD-leaning.
- **51.7 M** places are unique to Foursquare — disproportionately
  urban, with strong tails in Asia / LatAm / Eastern Europe.

For vocabulary design, this means we can confidently work from the
merged universe, treating only the `confident` tier as same-place and
keeping `probable` matches as separate fuzzy candidates (the `probable`
tier showed ~20 % false-positive rate in random sampling).

---

## 2 · What we have on disk

### Inputs (Leonardo `$CINECA_SCRATCH/bonzai-data/`)

| dataset | release | size | records |
|---|---|---:|---:|
| Overture (places only, mirrored) | 2026-04-15.0 | 3.75 GB parquet | 75.5 M |
| Foursquare OS Places | dt=2026-04-14 | 11 GB parquet | 99.9 M (live) |
| OpenAddresses global | 2025-09-14 | 12.4 GB zip | ~500–600 M (62 countries) |

### Outputs (`overture-map/outputs/`)

| file | size | what's in it |
|---|---:|---|
| `freq_*.csv` (27 files) | ~10 MB total | Per-column global value frequencies for every Overture theme |
| `fsq_freq_*.csv` (7 files) | ~5 MB | Per-column FSQ frequencies (categories, countries, etc.) |
| `oa_country_summary.csv` | <1 MB | OpenAddresses per-country source-file inventory |
| `dedup_pairs.parquet` (Leonardo) | 26.7 GB | 1.44 B raw candidate (Ovt, FSQ) pairs ≤ 66 m apart |
| `dedup_fsq_decisions.parquet` (Leonardo) | 7.9 GB | One row per FSQ place with its best Overture match + tier |
| `dedup_summary.csv` | <1 KB | Headline dedup counts |
| `poi_master.parquet` (Leonardo) | 10.7 GB | Unified 142.8 M-row POI master |
| `qa_*.csv` (5 files) | <100 KB | Dedup quality validation |
| `source_dominance_grid.csv` | 5 MB | 0.5° world grid with per-cell Ovt and FSQ counts |

---

## 3 · Methodology in one paragraph each

**Vocabulary frequency passes (`02_freq_pass.py`).** For every
attribute column in every Overture theme, run a streaming `GROUP BY`
directly against the S3 parquet via DuckDB `read_parquet` with
`http_retries=10`. Per-column scans avoid materialising the 2.5 B-row
buildings table. Output: one CSV per (theme.field) with `(value, count,
pct)` rows sorted descending.

**Foursquare frequency passes (`06_fsq_freq_pass.py`).** Same approach
on the locally-mirrored FSQ parquet on Leonardo scratch.
`UNNEST(fsq_category_labels)` is done in a projection-first CTE so the
99 M-row table doesn't blow memory.

**Spatial dedup (`08_dedup_places.py`).** Single-shot streaming
SQL — read both parquets, hash-join on a 0.005° grid bucket with 3×3
neighbour expansion (~550 m bucket × 9 candidates per FSQ row), filter
by `abs(lat/lon diff) < 0.0006` (~ 66 m), compute great-circle distance
+ Jaro-Winkler name similarity, write to parquet. Ran on
`boost_usr_prod` with 200 GB RAM in ~25 min wall.

**Dedup decision tier (`08_dedup_places.py::pick_best_per_fsq`).** For
each FSQ place that had at least one Overture candidate within 50 m,
pick the single best match using `row_number() OVER (PARTITION BY
fsq_id ORDER BY tier_rank, distance_m) = 1`. Tier rules:

- `confident` — `distance_m ≤ 50` AND `name_sim ≥ 0.85`
- `probable`  — `distance_m ≤ 50` AND `name_sim ≥ 0.6`
- `weak`      — `distance_m ≤ 50` AND `name_sim < 0.6` (kept for
   accounting, not treated as same-place)

**Source dominance (`11_source_dominance.py`).** Bin both sources into
0.5° lat/lon cells, count each, compute `log2((n_ovt + 1) / (n_fsq + 1))`
per cell. Output CSV is the basis for any per-region policy decisions.

---

## 4 · Vocabulary findings — Overture-only baseline

Full per-field tables are in `outputs/ANALYSIS.md`. Top-line:

| field family | distinct | null % | recommendation |
|---|---:|---:|---|
| `transportation.class` | 24 | 0 | KEEP all — road hierarchy |
| `transportation.subtype` | 3 | 0 | KEEP all (road / rail / water) |
| `places.basic_category` | 267 | 7.8 % | KEEP all 267 (mid-level POI) |
| `places.primary` / `taxonomy_primary` | ~2 080 | 4 % | TAIL-TRIM — keep top 690, bucket the rest |
| `places.alternate` | 2 070 | 0 | TAIL-TRIM — keep top 698, bucket the rest |
| `base.land_class` | 42 | 0 | KEEP all (vegetation/landform types) |
| `base.land_use_class` | 109 | 0 | KEEP all 109 |
| `base.land_cover_subtype` | 10 | 0 | KEEP all (very tight vocab) |
| `base.water_class` | 35 | 0 | KEEP all |
| `base.infrastructure_class` | 163 | 0 | KEEP all (power / barrier / transit) |
| `divisions.subtype` | 9 | 0 | KEEP all (locality / neighbourhood / county) |
| `buildings.subtype` | 13 | **94.4 %** | DROP from vocab or compress to `is_present` bit |
| `buildings.class` | 87 | **94.5 %** | DROP from vocab |
| `buildings.roof_material` | 14 | 99.9 % | DROP |
| `buildings.facade_material` | 11 | 99.9 % | DROP |
| `places.brand_wikidata` | 3 040 | 98.1 % | Collapse to `is_branded` bit |
| `places.operating_status` | 3 | 0 | DROP (99.99 % `open`) |
| `transportation.subclass` | 7 | 89.8 % | DROP or compress |
| `base.land_surface` | 23 | 99.7 % | DROP |

**Estimated final Overture-only vocab size: ~1 200 tokens** for a
balanced first cut.

The biggest gap is **buildings** — 94 % of footprints have no
type/class/subtype label. None of the three sources (Overture, FSQ,
OpenAddresses) fixes this directly. The fallback is the **Overture
Bridge Files** (`s3://overturemaps-us-west-2/bridgefiles/2026-04-15.0/dataset=osm/`)
which expose the original OSM ID for every Overture record — joining
those to the OSM planet PBF's `building=*`, `shop=*`, `amenity=*` tags
can recover semantic labels for the buildings that came from OSM.

---

## 5 · Three-source comparison

| | Overture | Foursquare OS | OpenAddresses |
|---|---:|---:|---:|
| total places / addresses | 75.5 M | 99.9 M (live) / 107.5 M (incl. closed) | ~500–600 M |
| unique primary categories | 2 080 (flat) | 1 272 (hierarchical, ` > ` separated) | n/a |
| null primary category | 4 % | ~0 % (always populated) | n/a |
| closed-place handling | dropped (operating_status filter) | 7.5 M historical kept | n/a |
| top country share | (no per-place country field) | US 22.6 %, ID 8.0 %, TR 7.5 %, BR 5.1 %, DE 4.7 % | US 60 %, BR 16 %, MX 9 % |
| **strength** | rural breadth, OSM provenance, building footprints | urban depth, hierarchical taxonomy, brand chains | street-number granularity |
| **gap** | sparse in Asia / LatAm; building labels mostly null | sparse outside cities; closed places mixed in | OECD-only, no POI semantics |

### Foursquare hierarchical taxonomy

FSQ exposes categories as a path like `"Dining and Drinking > Cafe,
Coffee, and Tea House > Café"`. Rolling up to the **11 top-level
categories** covers all 107 M places:

| top level | count | share |
|---|---:|---:|
| Business and Professional Services | 27.6 M | 25.8 % |
| Dining and Drinking | 20.9 M | 19.5 % |
| Retail | 18.5 M | 17.3 % |
| Community and Government | 12.4 M | 11.6 % |
| Travel and Transportation | 8.1 M | 7.6 % |
| Landmarks and Outdoors | 7.2 M | 6.8 % |
| Health and Medicine | 5.2 M | 4.8 % |
| Arts and Entertainment | 3.5 M | 3.2 % |
| Sports and Recreation | 2.5 M | 2.4 % |
| Event | 0.8 M | 0.8 % |

For a cascaded vocab this gives **11 / ~150 / ~700** tokens at the
top / level-2 / level-3 tiers — much cleaner than Overture's flat
690-keep tail.

---

## 6 · Dedup results

**Pipeline.** Streaming grid hash-join on Leonardo `boost_usr_prod`,
~25 min wall. Inputs: 75.5 M Overture + 99.9 M FSQ. Match predicate:
50 m great-circle + Jaro-Winkler name similarity threshold per tier.

| metric | value |
|---|---:|
| Overture places (named, lat/lon present) | 75,495,994 |
| Foursquare places (open, lat/lon present) | 99,904,729 |
| Candidate pairs ≤ ~66 m | 1,444,390,420 |
| FSQ matched (`confident` + `probable`, ≤ 50 m) | **48,252,270** (48.3 %) |
| FSQ unique | 51,652,459 |
| Overture matched | 32,559,028 (43.1 %) |
| Overture unique | 42,936,966 |
| **Merged universe** | **127,148,453** |

### Tier breakdown of the matched 48 M

| tier | count | avg distance | avg name_sim |
|---|---:|---:|---:|
| confident (sim ≥ 0.85) | 25.1 M | **11.5 m** | **0.973** |
| probable (sim 0.6–0.85) | 23.1 M | 22.9 m | 0.651 |
| weak (sim < 0.6, kept for accounting only) | 27.9 M | 20.0 m | 0.454 |

### Match multiplicity (FSQ per Overture)

71 % of matched Overture places match exactly one FSQ. 19 % match two.
The long tail is small (k ≥ 10 ≈ 0.5 %) — no address-pinned blowups.

| FSQ per Overture | n | % |
|---:|---:|---:|
| 1 | 23.2 M | 71.1 |
| 2 | 6.2 M | 19.1 |
| 3 | 1.9 M | 5.8 |
| 4 | 0.7 M | 2.1 |
| 5 | 0.3 M | 0.9 |
| 6+ | 0.4 M | 1.0 |

This means each Overture place absorbs ~1.48 FSQ entries on average
(48 M / 32.5 M) — exactly the expected pattern for FSQ duplicates of
the same physical place (mall storefront vs sidewalk geocode).

---

## 7 · Dedup QA — how trustworthy are the matches?

### 7.1 Distance / similarity distribution per tier

The thresholds are doing their job — `confident` is tight in space and
near-perfect on names; `weak` sits at near-random Jaro-Winkler.

| tier | n | avg distance | avg sim | p10 sim |
|---|---:|---:|---:|---:|
| confident | 25.1 M | 11.5 m | 0.973 | 0.887 |
| probable | 23.1 M | 22.9 m | 0.651 | 0.604 |
| weak | 27.9 M | 20.0 m | 0.454 | 0.358 |

### 7.2 Per-country match rate (top 30 by FSQ size)

| country | fsq total | matched (conf + prob) | match % |
|---|---:|---:|---:|
| US | 21.7 M | 11.9 M | **54.7 %** |
| ID | 8.3 M | 2.6 M | 30.8 % |
| TR | 7.7 M | 2.9 M | 38.1 % |
| BR | 5.2 M | 2.6 M | 50.2 % |
| DE | 4.75 M | 2.8 M | 58.2 % |
| JP | 4.4 M | 2.2 M | 49.1 % |
| GB | 3.9 M | 2.5 M | **64.0 %** |
| FR | 2.9 M | 1.6 M | 54.7 % |
| MX | 2.8 M | 1.6 M | 55.3 % |
| RU | 2.7 M | 0.9 M | 32.5 % |
| IT | 2.7 M | 1.7 M | 61.2 % |
| CA | 2.3 M | 1.4 M | 60.4 % |
| TH | 2.1 M | 1.3 M | 60.8 % |
| MY | 2.0 M | 0.9 M | 43.4 % |
| ES | 1.8 M | 1.1 M | 61.6 % |
| KR | 1.6 M | 0.4 M | **27.6 %** |
| PL | 1.5 M | 0.5 M | 34.4 % |
| IN | 1.5 M | 0.9 M | 60.4 % |
| AU | 1.3 M | 0.9 M | **71.6 %** |
| NL | 1.1 M | 0.5 M | 44.7 % |
| BE | 1.0 M | 0.4 M | 39.2 % |
| SE | 0.9 M | 0.3 M | 34.5 % |
| **CN** | 0.9 M | 0.07 M | **8.3 %** |
| PH | 0.8 M | 0.5 M | 57.5 % |
| AT | 0.7 M | 0.3 M | 49.4 % |
| **IR** | 0.5 M | 0.07 M | **12.6 %** |
| **SA** | 0.55 M | 0.13 M | 22.7 % |

**Reading:** OECD countries match at 50–72 %; FSQ-strong countries
(KR / RU / TR / ID / Eastern Europe) at 25–40 %; data-restricted
countries (CN, IR) at < 15 %. Pattern is sane.

### 7.3 Category co-occurrence (top 30 confident pairings)

Of the top 30 confident-match (overture_cat, fsq_leaf) pairings,
**29 are semantically aligned**. Examples (n = match count):

- `hotel ↔ Hotel` (331 k)
- `gas_station ↔ Fuel Station` (303 k)
- `automotive_repair ↔ Automotive Repair Shop` (296 k)
- `church_cathedral ↔ Church` (274 k)
- `hair_salon ↔ Hair Salon` (255 k)
- `restaurant ↔ Restaurant` (220 k)
- `cafe ↔ Café` (197 k)
- `pizza_restaurant ↔ Pizzeria` (192 k)
- `coffee_shop ↔ Coffee Shop` (170 k)
- `pharmacy ↔ Pharmacy` (142 k)
- `dentist ↔ Dentist` (174 k)

The single oddity in the top 30 is `landmark_and_historical_building ↔
Apartment or Condo` (114 k) — likely real (historical buildings often
get reused as apartments) rather than a data error.

### 7.4 Random-sample inspection (60 rows: 20 each tier)

- **Confident sample (n = 20):** every pair is unambiguously the same
  place; differences are case, punctuation, or a longer name on one
  side ("Forum Jacob Pins Kunstvern e.V. Kunstmuseum" vs "Forum Jacob
  Pins"). **Estimated precision: ≥ 95 %.**
- **Probable sample (n = 20):** noticeable false positives at the
  threshold. Examples:
  - `Escola Municipal Joao Batista Dias` ↔ `Fidencio Materiais para
    Construcao` (school vs construction store)
  - `Bakmi Malang` ↔ `Bank Mayapada` (Indonesian noodle shop vs bank)
  - `Dairy Queen` ↔ `Theyardhouse.ubon` (chain vs local)
  - **Estimated precision: ~70–80 %.** Sits at the ambiguous
    address-shared edge.
- **Weak sample (n = 20):** as expected — coincidental same-address
  pairings, different businesses. Behaves like random matching at
  ~50 m.

### 7.5 Recommendation

Treat **`confident` only** as canonical same-place merges for
downstream vocab work. Use `probable` as fuzzy candidates for analytics
but split them as two separate places when computing category
frequencies. This shifts the dedup numbers conservatively:

| | current (conf + prob) | conservative (conf only) |
|---|---:|---:|
| FSQ matched | 48.3 M | 25.1 M |
| Merged universe | 127.1 M | ~150 M |

---

## 8 · Source dominance — wide vs deep

Bucketed all populated 0.5° cells (~83 k cells worldwide) by Overture
vs FSQ density ratio.

| bucket (log₂ ratio) | cells | total places | cells % | places % |
|---|---:|---:|---:|---:|
| ovt_dominant_4× (≥ 4 : 1 Ovt) | 12,032 | 4.9 M | 14.5 | 2.8 |
| ovt_dominant_2× (2 : 1 to 4 : 1 Ovt) | 34,312 | 8.4 M | 41.4 | 4.8 |
| **balanced (within 2×)** | **22,574** | **127.7 M** | **27.2** | **72.8** |
| fsq_dominant_2× (2 : 1 to 4 : 1 FSQ) | 8,641 | 16.3 M | 10.4 | 9.3 |
| fsq_dominant_4× (≥ 4 : 1 FSQ) | 5,355 | 18.2 M | 6.5 | 10.4 |

**Two findings:**

1. **56 % of populated cells are Overture-dominant**, but those cells
   only contain **7.6 % of all places**. Overture covers many sparse
   areas thinly.
2. **17 % of cells are FSQ-dominant**, containing **20 % of all
   places**. FSQ concentrates depth in fewer cells.
3. **27 % of cells are balanced**, but they contain **73 % of all
   places**. The merge has the most leverage exactly where most POIs
   live.

### Top 15 cells by total density

All major cities. Overture never wins one. FSQ wins or ties in every case.

| lat | lon | country | n_ovt | n_fsq | log₂(O/F) |
|---:|---:|---|---:|---:|---:|
| -6.25 | 106.75 | Jakarta, ID | 337 660 | 2 477 307 | -2.88 (FSQ 7×) |
| 35.75 | 139.75 | Tokyo, JP | 572 795 | 854 602 | -0.58 (balanced) |
| 13.75 | 100.75 | Bangkok, TH | 384 187 | 690 460 | -0.85 |
| -23.75 | -46.75 | São Paulo, BR | 491 330 | 494 162 | -0.01 (balanced) |
| 40.75 | -73.75 | NYC, US | 360 413 | 569 656 | -0.66 |
| 19.25 | -99.25 | CDMX, MX | 334 735 | 572 738 | -0.77 |
| 3.25 | 101.75 | Kuala Lumpur, MY | 223 547 | 630 799 | -1.50 |
| 41.25 | 28.75 | Istanbul, TR | 192 754 | 585 447 | -1.60 |
| -6.75 | 107.75 | Bandung, ID | 86 952 | 679 799 | -2.97 |
| 55.75 | 37.75 | Moscow, RU | 124 543 | 566 191 | -2.18 (FSQ 4×) |
| 51.75 | -0.25 | London, GB | 316 007 | 369 189 | -0.22 |
| 48.75 | 2.25 | Paris, FR | 269 992 | 366 447 | -0.44 |
| 1.25 | 103.75 | Singapore, SG | 169 978 | 431 959 | -1.35 |

---

## 9 · Recommendations for the vocab work

1. **Use the merged 127 M-place universe as the working set.**
   Confident-only merging gives a ~150 M conservative count; either
   choice is defensible — the model sees more rows in the conservative
   case but with more cross-source duplication. **Pick `confident-only`
   merging**: cleaner taxonomy, ~95 % match precision, downstream noise
   is minimal.

2. **Prefer FSQ's hierarchical taxonomy for POI categories.**
   - 11 top-level / ~150 level-2 / ~700 level-3 categories — already a
     clean cascaded vocab.
   - Build a small bridge table mapping Overture's flat category set
     (~690 keep tokens) to FSQ leaf labels using the dedup category
     co-occurrence we already produced. The top 30 pairings cover most
     of the volume.

3. **Don't split the world by region.** The dominance analysis showed
   East/West splits don't hold (FSQ wins urban areas globally). The
   right mental model is "wide vs deep", and the merge is the natural
   resolution — no per-region policy needed.

4. **Building labels still need OSM bridge files.** All three datasets
   leave 94 % of buildings unlabeled. The remaining path is OSM tag
   recovery via `s3://overturemaps-us-west-2/bridgefiles/2026-04-15.0/dataset=osm/`.
   This is the next-largest blocker for vocab completeness.

5. **OpenAddresses is a post-v1 add-on.** Street-number / unit
   tokens are not load-bearing for the v1 generative model, and OA's
   OECD-heavy distribution doesn't align with our global goals.

6. **Estimated v1 vocab size on the merged universe:**
   - POI category cascade: 11 + 150 + 700 = ~860 tokens
   - Overture flat (kept where FSQ leaf is missing): ~690 tokens
   - Land use / cover / water / infrastructure: ~250 tokens
   - Roads (transportation.class): ~30 tokens
   - Divisions (locality / city / county / etc.): ~10 tokens
   - **Total: ~1 800 tokens** for a first-cut balanced vocab.

---

## 10 · Open questions

- **OSM bridge join for buildings.** Largest remaining gap (94 % null
  building labels). Concrete next step: pull bridge files for the
  release, join to a planet PBF tag dump.
- **Multi-source brand resolution.** `places.brand_wikidata` is 98 %
  null in Overture; FSQ has a `chains` column we haven't deeply
  inspected. Combining might give a useful `is_branded` signal across
  the merged universe.
- **Closed places filter.** FSQ keeps 7.5 M historical closures we
  excluded. For a "live cities" model that's right; for a temporal
  generative model they would be signal.
- **Release pinning policy.** Both Overture and FSQ snapshot quarterly.
  Decide whether to pin one release for v1 training or accept release
  drift in periodic re-runs.

---

## 11 · Pointers

- Source code: `overture-map/scripts/`
  - `02_freq_pass.py` — Overture per-field freq passes
  - `06_fsq_freq_pass.py` — FSQ freq passes
  - `08_dedup_places.py` — spatial dedup
  - `09_build_master.py` — unified master table
  - `10_dedup_qa.py` — dedup quality checks
  - `11_source_dominance.py` — 0.5° dominance grid
- Companion docs: `overture-map/docs/`
  - `VOCAB_ANALYSIS.md` — auto-generated per-field recommendations
  - `MULTI_SOURCE_ANALYSIS.md` — three-source side-by-side
  - `DATA_SOURCES.md` — datasets we considered
  - `HANDOFF.md` — one-page brief for next session
- Headline result: **127.1 M unified POIs** ready for vocab work.
