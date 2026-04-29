"""Spatial dedup: Overture places ∪ Foursquare OS places.

Two phases:

  1. Mirror Overture places S3 → local parquet with the columns we need
     (id, name, primary_cat, basic_cat, lon, lat, confidence). Skipped if the
     output parquet already exists.

  2. Hash-join on a ~550 m grid (0.005° bucket), expand to the 3×3 neighbor
     grid cells, then filter by great-circle distance ≤ 50 m. For each
     surviving candidate pair we compute a jaro-winkler name similarity and
     classify the match as:
         - confident_match:  dist ≤ 50 m AND name_sim ≥ 0.85
         - probable_match:   dist ≤ 50 m AND name_sim ≥ 0.6
         - weak_match:       dist ≤ 50 m AND name_sim < 0.6
     For each FSQ place we keep at most one row (its nearest confident match
     if any; else nearest probable; else nearest weak; else unmatched).

Outputs:
  outputs/dedup_pairs.parquet    — all candidate (overture_id, fsq_id) pairs
  outputs/dedup_fsq_decisions.parquet — one row per FSQ place + best match
  outputs/dedup_summary.csv       — headline counts per decision category

Run:
    FSQ_DIR=... OVT_MIRROR=... python scripts/08_dedup_places.py
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

S3_PLACES = (
    "s3://overturemaps-us-west-2/release/2026-04-15.0/"
    "theme=places/type=place/*"
)


def connect() -> duckdb.DuckDBPyConnection:
    # In-memory connection — no persistent DB file.  Login-node cgroup is
    # tight (~33 GB), so we let DuckDB spill to disk via temp_directory and
    # cap memory_limit well below the cgroup ceiling.
    con = duckdb.connect()  # :memory:
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("SET s3_region='us-west-2'")
    con.execute("SET http_retries=10")
    con.execute("SET http_retry_wait_ms=500")
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit='10GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = f"{SCRATCH}/overture-mirror/duckdb_tmp"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def mirror_overture(con: duckdb.DuckDBPyConnection) -> None:
    dest = Path(OVT_MIRROR)
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        print(f"[mirror] already present: {dest} ({size_gb:.2f} GB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mirror] writing {dest}  (may take 5-15 min from S3)")
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT id,
                   names.primary        AS name,
                   categories.primary   AS primary_cat,
                   basic_category       AS basic_cat,
                   ST_X(geometry)       AS lon,
                   ST_Y(geometry)       AS lat,
                   confidence
            FROM read_parquet('{S3_PLACES}', hive_partitioning=true)
            WHERE geometry IS NOT NULL
        )
        TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    print(f"[mirror] done in {time.time()-t0:.1f}s  ({dest.stat().st_size/1e9:.2f} GB)")


def build_tables(con: duckdb.DuckDBPyConnection) -> None:
    # No-op on the streaming path: we read parquet directly inside the
    # join query.  Materializing ovt+fsq into DuckDB tables on a login node
    # blew the cgroup; this skip is the whole point of the rewrite.
    return None


def find_pairs(con: duckdb.DuckDBPyConnection) -> None:
    out = OUTPUTS / "dedup_pairs.parquet"
    if out.exists():
        print(f"[pairs] already present: {out}")
        return

    # Streaming single-shot grid hash-join, written directly to parquet.
    # No persistent tables, no intermediate materialization — DuckDB pipelines:
    #   parquet readers → hash-build on Overture side
    #                  → probe with FSQ × 3×3 neighbour expansion
    #                  → distance + name-sim filter
    #                  → parquet writer
    # Hash table is the only big thing in memory: ~75 M ovt rows × the columns
    # we keep.  We keep the hash side narrow (id + grid keys + name) and only
    # rejoin the wider columns through the small probe side.
    print("[pairs] streaming grid hash-join → dedup_pairs.parquet...", flush=True)
    t0 = time.time()

    sql = f"""
        COPY (
            WITH ovt_g AS (
                SELECT id,
                       name,
                       primary_cat,
                       basic_cat,
                       lon, lat,
                       CAST(FLOOR(lon * 200) AS INTEGER) AS gx,
                       CAST(FLOOR(lat * 200) AS INTEGER) AS gy
                FROM read_parquet('{OVT_MIRROR}')
                WHERE name IS NOT NULL
            ),
            fsq_e AS (
                SELECT fsq_place_id AS id,
                       name,
                       fsq_category_labels[1] AS primary_cat,
                       country,
                       longitude AS lon,
                       latitude  AS lat,
                       CAST(FLOOR(longitude * 200) AS INTEGER) + dx AS kx,
                       CAST(FLOOR(latitude  * 200) AS INTEGER) + dy AS ky
                FROM read_parquet('{FSQ_GLOB}'),
                     (VALUES (-1,-1),(-1,0),(-1,1),
                             ( 0,-1),( 0,0),( 0,1),
                             ( 1,-1),( 1,0),( 1,1)) AS t(dx, dy)
                WHERE name IS NOT NULL
                  AND longitude IS NOT NULL
                  AND latitude  IS NOT NULL
                  AND date_closed IS NULL
            )
            SELECT o.id            AS overture_id,
                   f.id            AS fsq_id,
                   o.name          AS overture_name,
                   f.name          AS fsq_name,
                   o.primary_cat   AS overture_cat,
                   o.basic_cat     AS overture_basic,
                   f.primary_cat   AS fsq_cat,
                   f.country       AS fsq_country,
                   o.lon AS ovt_lon, o.lat AS ovt_lat,
                   f.lon AS fsq_lon, f.lat AS fsq_lat,
                   2 * 6371000 * asin(sqrt(
                       pow(sin(radians(f.lat - o.lat) / 2), 2)
                     + cos(radians(o.lat)) * cos(radians(f.lat))
                     * pow(sin(radians(f.lon - o.lon) / 2), 2)
                   )) AS distance_m,
                   jaro_winkler_similarity(lower(o.name), lower(f.name)) AS name_sim
            FROM ovt_g AS o
            JOIN fsq_e AS f
              ON o.gx = f.kx AND o.gy = f.ky
            WHERE abs(o.lat - f.lat) < 0.0006
              AND abs(o.lon - f.lon) < 0.0006
        )
        TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(
        f"[pairs] {n:,} candidate pairs in {time.time()-t0:.1f}s "
        f"({out.stat().st_size/1e6:.1f} MB)",
        flush=True,
    )


def pick_best_per_fsq(con: duckdb.DuckDBPyConnection) -> None:
    out = OUTPUTS / "dedup_fsq_decisions.parquet"
    if out.exists():
        print(f"[pick ] already present: {out}")
        return
    pairs = OUTPUTS / "dedup_pairs.parquet"
    print("[pick ] scoring + selecting best Overture match per FSQ place...", flush=True)

    # Pre-filter to <= 50 m BEFORE the window function — the 1.44 B candidate
    # rows shrink to ~100-200 M, so the partition-by-fsq_id sort is tractable.
    # We drop the "too_far" tier entirely; 50–100 m matches were never useful.
    con.execute(
        f"""
        COPY (
            SELECT * FROM (
                SELECT *,
                       CASE
                         WHEN name_sim >= 0.85 THEN 'confident'
                         WHEN name_sim >= 0.60 THEN 'probable'
                         ELSE 'weak'
                       END AS tier
                FROM read_parquet('{pairs}')
                WHERE distance_m <= 50
            )
            QUALIFY row_number() OVER (
                PARTITION BY fsq_id
                ORDER BY CASE tier
                           WHEN 'confident' THEN 1
                           WHEN 'probable'  THEN 2
                           ELSE 3
                         END,
                         distance_m
            ) = 1
        )
        TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"[pick ] {n:,} FSQ places with at least one Overture match within 50 m")


def summarize(con: duckdb.DuckDBPyConnection) -> None:
    dec = OUTPUTS / "dedup_fsq_decisions.parquet"
    n_fsq = con.execute(
        f"SELECT count(*) FROM read_parquet('{FSQ_GLOB}') "
        "WHERE name IS NOT NULL AND longitude IS NOT NULL "
        "AND latitude IS NOT NULL AND date_closed IS NULL"
    ).fetchone()[0]
    n_ovt = con.execute(
        f"SELECT count(*) FROM read_parquet('{OVT_MIRROR}') WHERE name IS NOT NULL"
    ).fetchone()[0]

    tier_counts = con.execute(
        f"""
        SELECT tier, count(*) AS n
        FROM read_parquet('{dec}')
        GROUP BY tier ORDER BY 2 DESC
        """
    ).fetchdf()
    print("\n== tier counts ==")
    print(tier_counts.to_string(index=False))

    matched_fsq = int(
        con.execute(
            f"""
            SELECT count(DISTINCT fsq_id)
            FROM read_parquet('{dec}')
            WHERE tier IN ('confident', 'probable')
            """
        ).fetchone()[0]
    )
    unmatched_fsq = n_fsq - matched_fsq

    matched_ovt = int(
        con.execute(
            f"""
            SELECT count(DISTINCT overture_id)
            FROM read_parquet('{dec}')
            WHERE tier IN ('confident', 'probable')
            """
        ).fetchone()[0]
    )
    unmatched_ovt = n_ovt - matched_ovt

    summary = [
        ("total_overture_places",   n_ovt),
        ("total_fsq_places",        n_fsq),
        ("matched_fsq_to_ovt",      matched_fsq),
        ("unmatched_fsq_only",      unmatched_fsq),
        ("overture_with_match",     matched_ovt),
        ("overture_only",           unmatched_ovt),
        ("merged_universe_size",    n_ovt + unmatched_fsq),
    ]

    import csv
    out = OUTPUTS / "dedup_summary.csv"
    with open(out, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerows(summary)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\n== summary ==")
    for k, v in summary:
        print(f"  {k:<28} {v:>14,}")


def main() -> None:
    con = connect()
    mirror_overture(con)
    build_tables(con)
    find_pairs(con)
    pick_best_per_fsq(con)
    summarize(con)


if __name__ == "__main__":
    main()
