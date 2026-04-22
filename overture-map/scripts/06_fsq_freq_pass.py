"""Frequency pass on locally-mirrored Foursquare OS Places Parquet.

No S3 — we read the local Parquet files on $CINECA_SCRATCH (or wherever
FSQ_DIR points). Same streaming GROUP BY pattern as 02_freq_pass.py so
memory stays bounded.

Fields scanned (derived from the FSQ OS Places schema):
  - fsq_category_labels  (top-level category string, unnested)
  - fsq_category_ids     (category ID, unnested)
  - country
  - locality
  - region
  - date_closed          (null = still open)
  - chains (brand)

Output: outputs/fsq_freq_<field>.csv — same format as Overture CSVs.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

FSQ_DIR = os.environ.get(
    "FSQ_DIR",
    "/leonardo_scratch/large/userexternal/uaslam00/bonzai-data/fsq/dt=2026-04-14",
)

# Only the places table. The categories/ subdir is a tiny lookup we don't
# scan here — if we need it, we join later.
PARQUET_GLOB = f"{FSQ_DIR}/release/dt=2026-04-14/places/parquet/*.parquet"

# (col_alias, sql_expr)
# Field names come from the FSQ OS Places schema documented at
# https://docs.foursquare.com/data-products/docs/places-os-data-schema.
# UNNEST(...) in the expr tells build_sql to project-then-unnest a list col.
PASSES = [
    ("country",                "country"),
    ("locality",               "locality"),
    ("region",                 "region"),
    ("is_closed",              "CASE WHEN date_closed IS NULL THEN 'open' ELSE 'closed' END"),
    ("fsq_category_labels",    "UNNEST(fsq_category_labels)"),
    ("fsq_category_ids",       "UNNEST(fsq_category_ids)"),
    ("chains_any",             "CASE WHEN len(chains) > 0 THEN 'has_chain' ELSE 'none' END"),
]


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 4;")
    return con


def build_sql(expr: str) -> str:
    if expr.upper().startswith("UNNEST"):
        inner = expr[len("UNNEST("):-1]
        return f"""
            WITH proj AS (
                SELECT {inner} AS arr
                FROM read_parquet('{PARQUET_GLOB}')
                WHERE {inner} IS NOT NULL
            )
            SELECT v AS value, COUNT(*) AS n
            FROM proj, UNNEST(arr) AS u(v)
            GROUP BY 1 ORDER BY n DESC
        """
    return f"""
        SELECT {expr} AS value, COUNT(*) AS n
        FROM read_parquet('{PARQUET_GLOB}')
        GROUP BY 1 ORDER BY n DESC
    """


def main() -> None:
    con = connect()

    # First, probe schema so we know which columns actually exist
    print("[fsq] probing schema...", flush=True)
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{PARQUET_GLOB}') LIMIT 0"
    ).fetchdf()
    (OUTPUTS.parent / "schema" / "fsq_places.txt").write_text(
        schema.to_string(index=False)
    )
    available = set(schema["column_name"])
    print(f"       {len(available)} columns found:", sorted(available))

    for alias, expr in PASSES:
        out = OUTPUTS / f"fsq_freq_{alias}.csv"
        if out.exists():
            print(f"[skip] {alias}")
            continue
        # Skip if the column doesn't exist (fsq schema shifts release-to-release)
        base_col = expr.split("(")[-1].rstrip(")")
        base_col = base_col.split(".")[0].split()[0]
        root = base_col if base_col in available else None
        if root is None and not any(c in available for c in expr.split()):
            print(f"[skip] {alias} — expected column missing (saw {expr})")
            continue

        print(f"[run ] {alias}: {expr}", flush=True)
        t0 = time.time()
        try:
            df = con.execute(build_sql(expr)).fetchdf()
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        total = int(df["n"].sum())
        df["pct"] = df["n"] / total * 100 if total else 0
        df.to_csv(out, index=False)
        dt = time.time() - t0
        top_v = df.iloc[0]["value"] if len(df) else None
        top_n = int(df.iloc[0]["n"]) if len(df) else 0
        print(f"       rows={len(df):,} total={total:,} top={top_v!r}:{top_n:,}  ({dt:.1f}s)")

    print("-- fsq freq pass complete --")


if __name__ == "__main__":
    main()
