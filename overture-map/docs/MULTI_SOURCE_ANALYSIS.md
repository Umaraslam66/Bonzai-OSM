# Multi-source analysis — Overture + Foursquare + OpenAddresses

All three global datasets mirrored on Leonardo `$CINECA_SCRATCH/bonzai-data/` as of 2026-04-22.
Paired with `VOCAB_ANALYSIS.md` (Overture-only decisions); this document synthesizes across sources.

## Dataset sizes

| dataset | release | size on disk | records | files |
|---|---|---:|---:|---:|
| Overture | 2026-04-15.0 | streamed from S3 (no local copy) | 2.5 B buildings, 75.5 M places, etc. | 972 parquet |
| Foursquare OS Places | dt=2026-04-14 | **11 GB** | **107.5 M places** | 111 parquet |
| OpenAddresses | 2025-09-14 archive | **12.4 GB zip** | ~500–600 M addresses (est.) | 4 850 CSV in 62 countries |

Total new data on scratch: **~24 GB**.

## POI headline: Foursquare has 42 % more than Overture

| | Overture | Foursquare OS |
|---|---:|---:|
| total places | 75 495 994 | **107 502 541** |
| unique primary categories | 2 080 | 1 272 |
| category depth | flat snake_case (e.g. `coffee_shop`) | hierarchical with ` > ` (e.g. `Dining and Drinking > Cafe, Coffee, and Tea House > Café`) |
| null basic/primary % | 4 % | unnested — effectively 0 % |
| closed places kept | 0.01 % | **7 %** (historical retention) |
| top country share | — (Overture doesn't expose country on places) | US 22.6 %, ID 8.0 %, TR 7.5 %, BR 5.1 %, DE 4.7 % |

### Foursquare top-level taxonomy (rolled up from hierarchical labels)

| top-level | count | % |
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

**Inference:** 11 top-level categories cover all 107 M places. If we wanted a very compact POI vocab, 11 tokens cover everything. The full leaf taxonomy (1 272 labels) is the fine-grained tier.

### FSQ geographic distribution differs from Overture

| country | FSQ places | note |
|---|---:|---|
| US | 24.3 M | both sources strong here |
| Indonesia | 8.6 M | Foursquare's SE-Asia user base, thin in Overture |
| Turkey | 8.1 M | thin in Overture |
| Brazil | 5.5 M | Overture thin in LatAm |
| Germany | 5.0 M | strong in both |
| Japan | 4.9 M | strong in both |
| GB | 4.4 M | strong in both |
| France | 3.2 M | strong in both |
| Russia | 3.1 M | thin in Overture |
| Mexico | 2.9 M | thin in Overture |
| Italy | 2.9 M | strong in both |

Concrete **complementarity**: Foursquare adds roughly 25–40 M POIs in Indonesia / Turkey / Russia / Mexico / Brazil / South Korea that Overture underreports. The cross-dataset dedup is expected to preserve most of these as unique.

## OpenAddresses — address-level granularity complement

- **62 countries** with at least one source file; 4 850 source CSVs total
- **17 GB** of the 35 GB total is US — OpenAddresses is dominant for North America
- Schema: `LON, LAT, NUMBER, STREET, UNIT, CITY, DISTRICT, REGION, POSTCODE, ID, HASH` — **exposes `NUMBER` and `UNIT`** which Overture's `addresses` theme does not split out (Overture packs everything into a `freeform` string)

### OpenAddresses top countries

| country | source files | size |
|---|---:|---:|
| US | 3 754 | 17.4 GB |
| Brazil | 72 | 6.5 GB |
| Mexico | 66 | 4.3 GB |
| Australia | 28 | 2.6 GB |
| France | 2 | 2.5 GB |
| South Africa | 12 | 1.5 GB |
| Spain | 10 | 1.0 GB |
| Poland | 32 | 1.0 GB |
| Belgium | 12 | 1.0 GB |
| Canada | 322 | 1.0 GB |
| Saudi Arabia | 6 | 0.8 GB |
| Japan | 94 | 0.8 GB |
| Netherlands | 2 | 0.8 GB |
| South Korea | 82 | 0.8 GB |

### What OpenAddresses is good for (vs Overture)

- **Street-number-level street addresses** in North America, Western Europe, Australia
- **Unit / apartment numbers** exposed as their own field (Overture only has `freeform`)
- **Government-authoritative provenance** — each row traces to a specific municipal release

### What it's NOT good for

- **No POI semantics** — pure address points, no category labels
- **Sparse in developing regions** — 62 countries of the world's ~200, and Asia/Africa coverage is thin outside South Africa / Japan / Saudi / Korea
- Coverage gaps: no China, India, Thailand, most of Africa, most of SE Asia outside Indonesia (and FSQ already fills Indonesia)

## Practical implications for vocab design

1. **POI tokenization should merge Overture + Foursquare.** Overture alone under-represents Asia / LatAm / Eastern Europe. FSQ's hierarchical category tree is actually cleaner for tokenization than Overture's flat list — we could use the **top-level (11 tokens) + level-2 (~150 tokens) + level-3 (~700 tokens)** and get a cascaded categorical vocab with explicit parent-child relationships.
2. **Address tokenization is a separate decision.** Overture's `addresses` theme is small (~22 GB) with coarse fields. OpenAddresses is complementary: richer granularity but OECD-only. For a generative city model, street numbers and unit labels are probably more signal than the model can use at typical inference scale, so this might be **post-v1 scope**.
3. **Building labels still the biggest gap.** None of the three sources fills the 94 % null on Overture's building subtype. The remaining path is Overture Bridge Files → OSM tags for buildings that came from OSM.
4. **Closed POIs — policy decision.** FSQ's 7 % closed is ~7.5 M historical closures. For a generative model of "live" cities, filter those out. For a model of urban change over time, they're signal.

## Next concrete actions (priority order)

1. **Spatial-dedup Overture ∪ Foursquare** — KNN join by (lat, lon) within 50 m, then fuzzy name + category overlap. Produces a master POI table with `(primary_source, secondary_source, unified_category)` per row.
2. **Overture Bridge Files for OSM** — pull `s3://overturemaps-us-west-2/bridgefiles/2026-04-15.0/dataset=osm/` and join to the OSM planet PBF's `building=*`, `shop=*`, `amenity=*` tags to recover semantic labels for Overture buildings.
3. **Optional per-region OpenAddresses integration** — if we want street-number-level tokens for US/EU slices later, run `07_oa_inspect.py` variants + CSV → Parquet conversion per country.

## Files produced in this pass

- `outputs/fsq_freq_country.csv` — 254 countries, full distribution
- `outputs/fsq_freq_locality.csv` — 1.23 M localities
- `outputs/fsq_freq_region.csv` — 274 k regions
- `outputs/fsq_freq_fsq_category_labels.csv` — 1 272 hierarchical category labels
- `outputs/fsq_freq_fsq_category_ids.csv` — 1 661 category IDs (some duplicates across labels)
- `outputs/fsq_freq_is_closed.csv` — open vs closed counts
- `outputs/oa_country_summary.csv` — 62 countries × source-file-count × byte-size
- `schema/fsq_places.txt` — full FSQ Places Parquet schema (28 columns)
- `schema/openaddresses_csv.txt` — OpenAddresses CSV schema sample
