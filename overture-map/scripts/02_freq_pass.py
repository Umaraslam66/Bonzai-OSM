"""Multi-column frequency pass — streaming, per column.

For each (theme, type, [columns]) tuple we run ONE query per column directly
against read_parquet — DuckDB projects just that column from Parquet and
streams aggregation. Memory is bounded by the # of distinct values, not row
count. This is critical for buildings (2.5B rows): the materialize-then-group
approach OOM-kills at ~33 GB resident.

Trade-off: N columns = N S3 scans per theme. But since each scan only pulls
one column's data, total network is the same as the materialize approach.

Writes freq_<theme>_<type>_<col>.csv per column. Full distribution — every
distinct value, including singletons. Restartable via existing-file skip.
"""
from __future__ import annotations
import time
import sys
import pandas as pd
from common import connect, theme_path, OUTPUTS

# (theme, type, [(col_alias, sql_expr), ...])
# sql_expr is the expression used in SELECT, col_alias names the output file
PASSES = [
    ("base", "land_cover",     [("subtype", "subtype")]),
    ("base", "land",           [("subtype", "subtype"), ("class", "class"), ("surface", "surface")]),
    ("base", "water",          [("subtype", "subtype"), ("class", "class")]),
    ("base", "land_use",       [("subtype", "subtype"), ("class", "class")]),
    ("base", "infrastructure", [("subtype", "subtype"), ("class", "class")]),
    ("places", "place",        [("basic_category", "basic_category"),
                                ("taxonomy_primary", "taxonomy.primary"),
                                ("operating_status", "operating_status"),
                                ("brand_wikidata", "brand.wikidata")]),
    ("divisions", "division",  [("subtype", "subtype"), ("country", "country"),
                                ("class", "class")]),
    ("transportation", "segment", [("subtype", "subtype"),
                                    ("class", "class"),
                                    ("subclass", "subclass")]),
    ("buildings", "building",  [("subtype", "subtype"), ("class", "class"),
                                ("roof_shape", "roof_shape"),
                                ("roof_material", "roof_material"),
                                ("facade_material", "facade_material")]),
]

# allow filter via CLI: .venv/bin/python 08_subtype_pass.py base
want = sys.argv[1] if len(sys.argv) > 1 else None

con = connect()  # default threads = # cores; Leonardo nodes have many

for theme, type_, cols in PASSES:
    if want and theme != want and type_ != want:
        continue

    # skip if all outputs already exist. The loop below already skips
    # per-column but this short-circuits the theme print noise.
    all_out = [OUTPUTS / f"freq_{theme}_{type_}_{alias}.csv" for alias, _ in cols]
    if all(p.exists() for p in all_out):
        print(f"[skip] {theme}/{type_} — all {len(cols)} columns done")
        continue

    path = theme_path(theme, type_)
    print(f"[scan] {theme}/{type_}: columns = {[a for a,_ in cols]}", flush=True)
    theme_t0 = time.time()

    for alias, expr in cols:
        out = OUTPUTS / f"freq_{theme}_{type_}_{alias}.csv"
        if out.exists():
            print(f"  [skip] {alias}")
            continue
        t0 = time.time()
        # Streaming GROUP BY directly on the Parquet scan. DuckDB projects
        # only `expr` from Parquet. Two aggregates so we emit pct and n.
        sql = f"""
            SELECT {expr} AS value, COUNT(*) AS n
            FROM read_parquet('{path}', hive_partitioning=true)
            GROUP BY 1 ORDER BY n DESC
        """
        try:
            df = con.execute(sql).fetchdf()
        except Exception as e:
            print(f"  [FAIL] {alias}: {e}")
            continue
        total = int(df["n"].sum())
        df["pct"] = df["n"] / total * 100 if total else 0
        df.to_csv(out, index=False)
        top_v = df.iloc[0]["value"] if len(df) else None
        top_n = int(df.iloc[0]["n"]) if len(df) else 0
        dt = time.time() - t0
        print(
            f"  [ok ] {alias}: {len(df):,} distinct, total={total:,}, "
            f"top={top_v!r}:{top_n:,}  ({dt:.1f}s)",
            flush=True,
        )

    print(f"       total {theme}/{type_} wall time: {time.time()-theme_t0:.1f}s\n", flush=True)

print("-- subtype pass complete --")
