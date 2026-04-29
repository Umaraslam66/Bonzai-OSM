"""Dedup QA — sanity checks on Overture ∪ Foursquare match decisions.

Reads:
  - outputs/dedup_fsq_decisions.parquet   (one row per FSQ with ≤50 m candidate)
  - $OVT_MIRROR/places_2026-04-15.parquet (Overture places)
  - $FSQ_GLOB                              (Foursquare OS Places, live)

Writes:
  - outputs/qa_distance_sim_per_tier.csv   — distance/name_sim distribution
  - outputs/qa_per_country_match_rate.csv  — match rate per country
  - outputs/qa_category_cooccurrence.csv   — top (ovt_cat, fsq_cat) pairs
  - outputs/qa_overture_multiplicity.csv   — how many FSQs per Overture
  - outputs/qa_random_sample.csv           — 60 random matches for eyeball QA

Headline summary printed to stdout — no heavy aggregates, runs in <10 min on
boost_usr_prod or ~15 min on a beefy login.
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
    con.execute("SET threads TO 32")
    con.execute("SET memory_limit='180GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp = f"{SCRATCH}/overture-mirror/duckdb_tmp"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def header(s: str) -> None:
    print(f"\n{'=' * 60}\n  {s}\n{'=' * 60}", flush=True)


def qa_distance_sim_per_tier(con: duckdb.DuckDBPyConnection) -> None:
    header("QA 1 — distance + name_sim distribution per tier")
    df = con.execute(
        f"""
        SELECT tier,
               count(*)                                             AS n,
               ROUND(avg(distance_m), 1)                            AS avg_dist,
               ROUND(quantile_cont(distance_m, 0.5), 1)             AS p50_dist,
               ROUND(quantile_cont(distance_m, 0.9), 1)             AS p90_dist,
               ROUND(avg(name_sim), 3)                              AS avg_sim,
               ROUND(quantile_cont(name_sim, 0.5), 3)               AS p50_sim,
               ROUND(quantile_cont(name_sim, 0.1), 3)               AS p10_sim
        FROM read_parquet('{DECISIONS}')
        GROUP BY tier ORDER BY n DESC
        """
    ).fetchdf()
    print(df.to_string(index=False))
    df.to_csv(OUTPUTS / "qa_distance_sim_per_tier.csv", index=False)
    print("\n>>> sanity: confident should have low avg_dist (<15m) and high "
          "avg_sim (>0.93); weak should have name_sim near random (~0.4-0.5).")


def qa_per_country_match_rate(con: duckdb.DuckDBPyConnection) -> None:
    header("QA 2 — match rate per country (top 30 by FSQ size)")
    # Total FSQ per country (live places only)
    fsq_per_country = f"""
        SELECT country,
               count(*) AS fsq_total
        FROM read_parquet('{FSQ_GLOB}')
        WHERE name IS NOT NULL
          AND longitude IS NOT NULL
          AND latitude IS NOT NULL
          AND date_closed IS NULL
        GROUP BY country
    """
    matched_per_country = f"""
        SELECT fsq_country AS country,
               count(*) AS matched
        FROM read_parquet('{DECISIONS}')
        WHERE tier IN ('confident', 'probable')
        GROUP BY 1
    """
    df = con.execute(
        f"""
        WITH t AS ({fsq_per_country}),
             m AS ({matched_per_country})
        SELECT t.country,
               t.fsq_total,
               COALESCE(m.matched, 0)                                    AS fsq_matched,
               ROUND(100.0 * COALESCE(m.matched, 0) / t.fsq_total, 1)    AS match_pct
        FROM t LEFT JOIN m USING (country)
        WHERE t.country IS NOT NULL
        ORDER BY t.fsq_total DESC
        LIMIT 30
        """
    ).fetchdf()
    print(df.to_string(index=False))
    df.to_csv(OUTPUTS / "qa_per_country_match_rate.csv", index=False)
    print("\n>>> sanity: US/DE/JP/GB/FR/IT should have high match rate (>40%); "
          "ID/TR/RU/BR/MX should have low (<25%).")


def qa_category_cooccurrence(con: duckdb.DuckDBPyConnection) -> None:
    header("QA 3 — top 30 (ovt_cat, fsq_top_label) co-occurrence (confident only)")
    df = con.execute(
        f"""
        WITH conf AS (
            SELECT overture_cat,
                   -- Strip the FSQ hierarchy down to just the leaf label
                   trim(split_part(fsq_cat, ' > ', -1)) AS fsq_leaf
            FROM read_parquet('{DECISIONS}')
            WHERE tier = 'confident'
              AND overture_cat IS NOT NULL
              AND fsq_cat IS NOT NULL
        )
        SELECT overture_cat, fsq_leaf, count(*) AS n
        FROM conf
        GROUP BY 1, 2
        ORDER BY n DESC
        LIMIT 30
        """
    ).fetchdf()
    print(df.to_string(index=False))
    df.to_csv(OUTPUTS / "qa_category_cooccurrence.csv", index=False)
    print("\n>>> sanity: pairings should be semantically aligned, e.g. "
          "'restaurant' ↔ 'Restaurant', 'coffee_shop' ↔ 'Café', "
          "'gas_station' ↔ 'Gas Station'.")


def qa_overture_multiplicity(con: duckdb.DuckDBPyConnection) -> None:
    header("QA 4 — multiplicity: how many FSQs per Overture id (confident+probable)")
    df = con.execute(
        f"""
        WITH per_ovt AS (
            SELECT overture_id, count(*) AS k
            FROM read_parquet('{DECISIONS}')
            WHERE tier IN ('confident', 'probable')
            GROUP BY 1
        )
        SELECT k AS fsq_per_overture,
               count(*) AS n,
               ROUND(100.0 * count(*) / SUM(count(*)) OVER (), 2) AS pct
        FROM per_ovt
        GROUP BY 1
        ORDER BY 1
        LIMIT 12
        """
    ).fetchdf()
    print(df.to_string(index=False))
    df.to_csv(OUTPUTS / "qa_overture_multiplicity.csv", index=False)
    print("\n>>> sanity: most Overtures should match 1 FSQ (k=1).  "
          "k=2-5 expected (FSQ duplicates).  k>20 suggests address-level "
          "ambiguity or category-pinned errors.")


def qa_random_sample(con: duckdb.DuckDBPyConnection) -> None:
    header("QA 5 — random sample of 20 confident + 20 probable + 20 weak matches")
    df = con.execute(
        f"""
        SELECT *
        FROM (
            SELECT 'confident' AS bucket, fsq_name, overture_name,
                   overture_cat, fsq_cat, distance_m, name_sim, fsq_country
            FROM read_parquet('{DECISIONS}')
            WHERE tier = 'confident'
            USING SAMPLE 20 ROWS
        )
        UNION ALL
        SELECT *
        FROM (
            SELECT 'probable' AS bucket, fsq_name, overture_name,
                   overture_cat, fsq_cat, distance_m, name_sim, fsq_country
            FROM read_parquet('{DECISIONS}')
            WHERE tier = 'probable'
            USING SAMPLE 20 ROWS
        )
        UNION ALL
        SELECT *
        FROM (
            SELECT 'weak' AS bucket, fsq_name, overture_name,
                   overture_cat, fsq_cat, distance_m, name_sim, fsq_country
            FROM read_parquet('{DECISIONS}')
            WHERE tier = 'weak'
            USING SAMPLE 20 ROWS
        )
        ORDER BY bucket, fsq_country
        """
    ).fetchdf()
    print(df.to_string(index=False))
    df.to_csv(OUTPUTS / "qa_random_sample.csv", index=False)
    print("\n>>> manual eyeball: confident pairs should obviously be the same "
          "place (case/punctuation differences only); weak pairs should look "
          "like coincidences (different businesses sharing an address).")


def main() -> None:
    t0 = time.time()
    con = connect()
    qa_distance_sim_per_tier(con)
    qa_per_country_match_rate(con)
    qa_category_cooccurrence(con)
    qa_overture_multiplicity(con)
    qa_random_sample(con)
    print(f"\n== QA complete in {time.time()-t0:.1f}s ==")


if __name__ == "__main__":
    main()
