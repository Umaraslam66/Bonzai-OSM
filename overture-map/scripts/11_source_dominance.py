"""Source dominance analysis — where is Overture vs Foursquare denser?

Bins both POI sources into a 0.5° lat/lon grid (~55 km × 55 km at equator),
then for each populated cell computes:
  - n_ovt        Overture place count
  - n_fsq        Foursquare place count
  - log2_ratio   log2((n_ovt + 1) / (n_fsq + 1))
                 positive → Overture dominant, negative → Foursquare dominant
                 |x| ≥ 1  → ≥ 2:1 dominance
                 |x| ≥ 2  → ≥ 4:1 dominance

Output: outputs/source_dominance_grid.csv  (~10 k populated cells globally)

Run on Leonardo lrd_all_serial — pure aggregation, no joins, ~5 min.
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

# 0.5° grid: lat * 2, lon * 2 — coarse enough to render globally, fine
# enough to distinguish urban areas from rural surroundings.
GRID_FACTOR = 2  # 1 / 0.5


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit='10GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = f"{SCRATCH}/overture-mirror/duckdb_tmp"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def compute(con: duckdb.DuckDBPyConnection) -> None:
    out = OUTPUTS / "source_dominance_grid.csv"
    if out.exists():
        print(f"[grid] already present: {out}")
    else:
        print("[grid] computing 0.5° dominance grid...", flush=True)
        t0 = time.time()
        con.execute(
            f"""
            COPY (
                WITH ovt AS (
                    SELECT CAST(FLOOR(lat * {GRID_FACTOR}) AS INTEGER) AS gy,
                           CAST(FLOOR(lon * {GRID_FACTOR}) AS INTEGER) AS gx,
                           count(*) AS n_ovt
                    FROM read_parquet('{OVT_MIRROR}')
                    WHERE name IS NOT NULL
                    GROUP BY 1, 2
                ),
                fsq AS (
                    SELECT CAST(FLOOR(latitude  * {GRID_FACTOR}) AS INTEGER) AS gy,
                           CAST(FLOOR(longitude * {GRID_FACTOR}) AS INTEGER) AS gx,
                           count(*) AS n_fsq,
                           mode(country)                                  AS country
                    FROM read_parquet('{FSQ_GLOB}')
                    WHERE name IS NOT NULL
                      AND longitude IS NOT NULL
                      AND latitude  IS NOT NULL
                      AND date_closed IS NULL
                    GROUP BY 1, 2
                )
                SELECT
                    COALESCE(o.gy, f.gy)                                  AS gy,
                    COALESCE(o.gx, f.gx)                                  AS gx,
                    (COALESCE(o.gy, f.gy) + 0.5) / {GRID_FACTOR}.0        AS lat_center,
                    (COALESCE(o.gx, f.gx) + 0.5) / {GRID_FACTOR}.0        AS lon_center,
                    COALESCE(o.n_ovt, 0)                                  AS n_ovt,
                    COALESCE(f.n_fsq, 0)                                  AS n_fsq,
                    COALESCE(o.n_ovt, 0) + COALESCE(f.n_fsq, 0)           AS n_total,
                    log2(
                        (COALESCE(o.n_ovt, 0) + 1.0)::DOUBLE /
                        (COALESCE(f.n_fsq, 0) + 1.0)::DOUBLE
                    )                                                     AS log2_ratio,
                    f.country                                             AS country
                FROM ovt o FULL OUTER JOIN fsq f USING (gy, gx)
            )
            TO '{out}' (FORMAT CSV, HEADER)
            """
        )
        print(f"[grid] wrote {out} in {time.time()-t0:.1f}s", flush=True)

    # Headline summary
    print()
    print("== summary ==")
    df = con.execute(
        f"""
        SELECT
            CASE
                WHEN log2_ratio >=  2 THEN 'ovt_dominant_4x'
                WHEN log2_ratio >=  1 THEN 'ovt_dominant_2x'
                WHEN log2_ratio >  -1 THEN 'balanced'
                WHEN log2_ratio > -2  THEN 'fsq_dominant_2x'
                ELSE 'fsq_dominant_4x'
            END AS bucket,
            count(*)                              AS cells,
            sum(n_total)                          AS total_places,
            ROUND(100.0 * count(*) /
                  SUM(count(*)) OVER (), 1)       AS cells_pct,
            ROUND(100.0 * sum(n_total) /
                  SUM(sum(n_total)) OVER (), 1)   AS places_pct
        FROM read_csv_auto('{out}')
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    print(df.to_string(index=False))

    print("\n== top 15 cells by total places ==")
    df = con.execute(
        f"""
        SELECT lat_center, lon_center, country, n_ovt, n_fsq,
               ROUND(log2_ratio, 2) AS log2_ratio
        FROM read_csv_auto('{out}')
        ORDER BY n_total DESC LIMIT 15
        """
    ).fetchdf()
    print(df.to_string(index=False))


def main() -> None:
    con = connect()
    compute(con)


if __name__ == "__main__":
    main()
