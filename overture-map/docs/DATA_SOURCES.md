# Data sources — what we're using, what we considered

Research completed 2026-04-22. Goal: understand whether Overture alone is sufficient, and what supplementary data would help fill the gaps we surfaced in `VOCAB_ANALYSIS.md` (especially the 94 % null on building subtypes).

## Currently used

### Overture Maps — release `2026-04-15.0`
- **S3:** `s3://overturemaps-us-west-2/release/2026-04-15.0/` (public, anonymous, us-west-2)
- **Format:** GeoParquet + zstd compression, hive-partitioned by `theme=/type=`
- **License:** [CDLA Permissive 2.0](https://cdla.dev/permissive-2-0/)
- **Total:** 972 files, 608 GB compressed
- **What's in it:**
  | theme | records |
  |---|---:|
  | places | 75.5 M |
  | buildings | 2.54 B |
  | transportation segments | 344 M |
  | base/land_cover | 123 M |
  | base/water | 64.7 M |
  | base/infrastructure | 149 M |
  | base/land_use | 54 M |
  | divisions (admin) | 4.6 M |
- **Under the hood** Overture is already a conflated merge of: OSM, Google Open Buildings, Microsoft Building Footprints, Meta Places, Microsoft Places, Esri Community Maps, PinMeTo, IGN Spain, geoBoundaries, and since Sept 2025 a curated slice of Foursquare Places. So a lot of sources are already fused in.

## Supplementary sources we considered

### Foursquare OS Places

- **License:** Apache 2.0
- **Latest release:** 2026-04-14
- **Size:** 193 GB Parquet (up from ~11 GB early-2025; schema now richer)
- **Records:** 100 M+ globally (~104 M last confirmed)
- **Access (changed 2025):**
  - The old public S3 bucket `s3://fsq-os-places-us-east-1/` is **deprecated**.
  - Current: HuggingFace Datasets — `foursquare/fsq-os-places`, gated (free sign-up + accept ToS, generate token).
    Path: `hf://datasets/foursquare/fsq-os-places/release/dt=2026-04-14/places/parquet/*.parquet`
  - Also via the Foursquare Places Portal (Iceberg catalog, access tokens).
- **Schema key fields:** `fsq_place_id`, `name`, `fsq_category_ids[]`, `date_closed`, `date_refreshed`, `latitude`, `longitude`, `address`, `locality`, `region`, `postcode`, `country`.
- **Overlap with Overture:**
  - Overture pulled ~6 M Foursquare POIs in the 2025-09-24.0 release.
  - **Overture does NOT publish a GERS ↔ Foursquare ID bridge** (Overture bridge files only cover Esri, geoBoundaries, IGN, Meta Places, Microsoft Places, OSM, PinMeTo).
  - Dedup between our Overture 75.5 M and a fresh Foursquare pull needs spatial + name + category matching, not ID join.
- **Expected marginal gain after dedup:** ~40–60 M new POIs, mostly in Asia / LatAm / Africa where Overture (and OSM) coverage is thin.

### Microsoft Global Building Footprints
- **License:** CDLA Permissive 2.0
- **Records:** 1.4 B buildings + ~1.4 B Microsoft-derived height estimates
- **Format:** GZIP-compressed CSV, 113 GB across 30 344 files, per-country indexed in `dataset-links.csv`
- **S3 + GitHub:** [microsoft/GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
- **Relevance:** already fused into Overture. Height estimates are the one thing Overture's `buildings.height` is not always filled with — could supplement.

### Google Open Buildings v3
- **License:** CC BY 4.0
- **Records:** 1.8 B buildings (Africa, Asia, LatAm focus)
- **Relevance:** already fused into Overture.

### Google-Microsoft Open Buildings combined (VIDA)
- **Source:** [source.coop/vida/google-microsoft-open-buildings](https://source.coop/vida/google-microsoft-open-buildings)
- **Records:** 2.58 B footprints (185 GeoParquet partitions)
- **Relevance:** basically equivalent to Overture's buildings theme. No marginal count gain.

### OpenAddresses
- **License:** mixed (per-country, mostly permissive government releases)
- **Records:** 600 M+ global addresses
- **Format:** CSV (`LON,LAT,NUMBER,STREET,UNIT,CITY,DISTRICT,REGION,POSTCODE,ID,HASH`)
- **Access:** batch downloads at [results.openaddresses.io](https://results.openaddresses.io/) or [batch.openaddresses.io](https://batch.openaddresses.io/)
- **Coverage:** 34 countries complete (US, CA, AU, most of EU), 7 substantial, 11 minimal. Asia/Africa/LatAm sparse.
- **vs. Overture addresses:** Overture addresses theme is ~22 GB (we didn't scan record count). OpenAddresses has granular unit / number / postcode with government authority. Useful for augmenting in OECD regions where we want fine street-level detail.

### OSM raw (via TagInfo / planet.osm.pbf)
- **License:** ODbL
- **Records:** ~3 B geometry nodes, ~100 M tagged features, ~580 M building polygons with `building=*`, ~55 M POI-ish features (amenity/shop/tourism/leisure)
- **Relevance:** two uses:
  1. **Semantic label recovery** — Overture buildings are 94 % null on subtype. The corresponding OSM building polygon often has `building=house`, `building=school`, `office=*`, `shop=*` etc. Overture publishes an OSM bridge file we can use to join Overture buildings ← → OSM tags and recover labels.
  2. **Tier-3 tag frequency** — your existing `scripts/taginfo_to_csv.py` (in the repo root, not this subproject) uses TagInfo JSON dumps to see every OSM key/value. That's the empirical vocab your v3 tokenizer (`TAG_KEY_VALUE`) was built on.

### Overture Bridge Files
- **Path:** `s3://overturemaps-us-west-2/bridgefiles/<RELEASE>/`
- **Format:** Parquet partitioned by `dataset`, `theme`, `type`
- **Maps:** GERS ID ↔ source record ID
- **Sources covered:** Esri Community Maps, geoBoundaries, IGN Spain, Meta Places, Microsoft Places, OSM, PinMeTo
- **Sources NOT covered:** Foursquare, Google Open Buildings
- **Primary use for us:** recover OSM `building=*` tags for the 94 % of Overture buildings that have null subtype, via join on (Overture id → OSM bridge → OSM tag).

## Commercial sources — considered, rejected

| source | places | cost | why rejected |
|---|---:|---:|---|
| Google Places API | ~200 M | $3–10 M one-time + ToS violation + account termination | ToS forbids bulk download |
| SafeGraph Places | ~43 M US / partial global | 6-figure commercial license | not training-licensable |
| Apple / Here | proprietary | not for resale | no bulk access |

## Recommended next additions (order of value)

1. **Overture Bridge Files → OSM tag join** to recover building subtypes on the 94 % null. Cheapest, highest value. No new download budget. All we need is to pull the bridge files for our release and join against OSM planet tags (which you already have on Leonardo at `$CINECA_SCRATCH/osm/raw/planet-latest.osm.pbf`).
2. **Foursquare OS Places merge** for non-OECD POI density. Requires HuggingFace sign-up. ~40 GB download (or direct stream once authenticated). Dedup via spatial+name matching.
3. **OpenAddresses** for US/CA/AU/EU granular street-level addresses if the generative model later needs street-number fidelity.
4. **Microsoft height estimates** if building 3-D generation becomes a priority and Overture's height nulls hurt.

None of these are urgent for the v1 vocab — Overture is sufficient for a global tokenizer baseline. They become relevant when (a) a downstream model hits the 94 % null wall on buildings, or (b) regional imbalance becomes the dominant training artifact.
