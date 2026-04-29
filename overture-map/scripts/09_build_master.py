"""Build the unified POI master table.

Inputs:
  - outputs/dedup_fsq_decisions.parquet  (best Overture match per FSQ place)
  - $OVT_MIRROR/places_2026-04-15.parquet (Overture places, named only)
  - $FSQ_GLOB                              (Foursquare OS Places, live)

Output:
  - outputs/poi_master.parquet             (~127 M rows)

Schema (one row per merged place):
  merged_id      VARCHAR  -- coalesce(ovt_id, fsq_id) — stable per source
  ovt_id         VARCHAR  -- null for fsq_only
  fsq_id         VARCHAR  -- null for ovt_only
  ovt_name       VARCHAR  -- null when no Overture
  fsq_name       VARCHAR  -- null when no Foursquare
  ovt_primary_cat VARCHAR -- Overture flat category (e.g. coffee_shop)
  ovt_basic_cat  VARCHAR  -- Overture mid-level category
  fsq_primary_cat VARCHAR -- FSQ hierarchical label
                          --   ("Dining and Drinking > Cafe, Coffee, ... > Café")
  fsq_country    VARCHAR  -- null when no Foursquare
  lat, lon       DOUBLE   -- Overture coords if matched, else FSQ coords
  distance_m     DOUBLE   -- match distance, null if not matched
  name_sim       DOUBLE   -- Jaro-Winkler similarity, null if not matched
  tier           VARCHAR  -- 'confident' | 'probable' | null
  source_set     VARCHAR  -- 'both_confident' | 'both_probable'
                          -- | 'ovt_only' | 'fsq_only'

Run on Leonardo boost_usr_prod (200 GB RAM, 32 cores).  Login-node WILL OOM.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SCRATCH = os.environ.get(
    "BONZAI_SCRATCH",
    "/leonardo_scratch/large/userexternal/uaslam00/bonzai-data",
)
OVT_MIRROR = os.environ.get(
    "OVT_MIRROR",
    f"{SCRATCH}/overture-mirror/places_2026-04-15.parquet",
)
FSQ_GLOB = os.environ.get(
    "FSQ_GLOB",
    f"{SCRATCH}/fsq/dt=2026-04-14/release/dt=2026-04-14/places/parquet/*.parquet",
)
DECISIONS = os.environ.get(
    "DEDUP_DECISIONS",
    str(OUTPUTS / "dedup_fsq_decisions.parquet"),
)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs ; LOAD httpfs")
    con.execute("SET threads TO 32")
    con.execute("SET memory_limit='180GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = f"{SCRATCH}/overture-mirror/duckdb_tmp"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def build_master(con: duckdb.DuckDBPyConnection) -> None:
    out = OUTPUTS / "poi_master.parquet"
    if out.exists():
        print(f"[master] already present: {out}")
        return

    print("[master] building unified POI master table...", flush=True)
    t0 = time.time()

    # Single streaming UNION ALL written directly to parquet.  DuckDB pipelines
    # all three legs and writes incrementally; only the antijoin hash tables
    # (matched ids) live in memory simultaneously.
    sql = f"""
        COPY (
            WITH matched_ids AS (
                SELECT overture_id, fsq_id
                FROM read_parquet('{DECISIONS}')
                WHERE tier IN ('confident', 'probable')
            ),
            matched AS (
                SELECT
                    overture_id                       AS merged_id,
                    overture_id                       AS ovt_id,
                    fsq_id                            AS fsq_id,
                    overture_name                     AS ovt_name,
                    fsq_name                          AS fsq_name,
                    overture_cat                      AS ovt_primary_cat,
                    overture_basic                    AS ovt_basic_cat,
                    fsq_cat                           AS fsq_primary_cat,
                    fsq_country                       AS fsq_country,
                    ovt_lat                           AS lat,
                    ovt_lon                           AS lon,
                    distance_m                        AS distance_m,
                    name_sim                          AS name_sim,
                    tier                              AS tier,
                    'both_' || tier                   AS source_set
                FROM read_parquet('{DECISIONS}')
                WHERE tier IN ('confident', 'probable')
            ),
            ovt_only AS (
                SELECT
                    o.id                              AS merged_id,
                    o.id                              AS ovt_id,
                    NULL                              AS fsq_id,
                    o.name                            AS ovt_name,
                    NULL                              AS fsq_name,
                    o.primary_cat                     AS ovt_primary_cat,
                    o.basic_cat                       AS ovt_basic_cat,
                    NULL                              AS fsq_primary_cat,
                    NULL                              AS fsq_country,
                    o.lat                             AS lat,
                    o.lon                             AS lon,
                    NULL::DOUBLE                      AS distance_m,
                    NULL::DOUBLE                      AS name_sim,
                    NULL                              AS tier,
                    'ovt_only'                        AS source_set
                FROM read_parquet('{OVT_MIRROR}') o
                LEFT JOIN (SELECT DISTINCT overture_id FROM matched_ids) m
                  ON o.id = m.overture_id
                WHERE o.name IS NOT NULL
                  AND m.overture_id IS NULL
            ),
            fsq_only AS (
                SELECT
                    f.fsq_place_id                    AS merged_id,
                    NULL                              AS ovt_id,
                    f.fsq_place_id                    AS fsq_id,
                    NULL                              AS ovt_name,
                    f.name                            AS fsq_name,
                    NULL                              AS ovt_primary_cat,
                    NULL                              AS ovt_basic_cat,
                    f.fsq_category_labels[1]          AS fsq_primary_cat,
                    f.country                         AS fsq_country,
                    f.latitude                        AS lat,
                    f.longitude                       AS lon,
                    NULL::DOUBLE                      AS distance_m,
                    NULL::DOUBLE                      AS name_sim,
                    NULL                              AS tier,
                    'fsq_only'                        AS source_set
                FROM read_parquet('{FSQ_GLOB}') f
                LEFT JOIN (SELECT DISTINCT fsq_id FROM matched_ids) m
                  ON f.fsq_place_id = m.fsq_id
                WHERE f.name IS NOT NULL
                  AND f.longitude IS NOT NULL
                  AND f.latitude IS NOT NULL
                  AND f.date_closed IS NULL
                  AND m.fsq_id IS NULL
            )
            SELECT * FROM matched
            UNION ALL
            SELECT * FROM ovt_only
            UNION ALL
            SELECT * FROM fsq_only
        )
        TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(
        f"[master] {n:,} rows in {time.time()-t0:.1f}s "
        f"({out.stat().st_size/1e9:.2f} GB)",
        flush=True,
    )


def summarize(con: duckdb.DuckDBPyConnection) -> None:
    out = OUTPUTS / "poi_master.parquet"
    print("\n== source_set breakdown ==", flush=True)
    df = con.execute(
        f"""
        SELECT source_set, count(*) AS n,
               ROUND(100.0 * count(*) / SUM(count(*)) OVER (), 2) AS pct
        FROM read_parquet('{out}')
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchdf()
    print(df.to_string(index=False), flush=True)

    print("\n== tier breakdown (matched only) ==", flush=True)
    df = con.execute(
        f"""
        SELECT tier, count(*) AS n
        FROM read_parquet('{out}')
        WHERE tier IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchdf()
    print(df.to_string(index=False), flush=True)

    print("\n== top 20 fsq_country (master) ==", flush=True)
    df = con.execute(
        f"""
        SELECT fsq_country, count(*) AS n
        FROM read_parquet('{out}')
        WHERE fsq_country IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT 20
        """
    ).fetchdf()
    print(df.to_string(index=False), flush=True)

    # Categorical recoverability — how many master rows have at least one
    # category label (Overture or FSQ).  Empty strings count as null here.
    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{out}')"
    ).fetchone()[0]
    n_with_cat = con.execute(
        f"""
        SELECT count(*) FROM read_parquet('{out}')
        WHERE COALESCE(ovt_primary_cat, '') <> ''
           OR COALESCE(fsq_primary_cat, '') <> ''
        """
    ).fetchone()[0]
    print(
        f"\nmaster rows with at least one category label: "
        f"{n_with_cat:,} / {n_total:,} ({100.0 * n_with_cat / n_total:.1f}%)",
        flush=True,
    )


def main() -> None:
    con = connect()
    build_master(con)
    summarize(con)


if __name__ == "__main__":
    main()
