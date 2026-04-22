"""DESCRIBE each theme/type and dump the full schema to schema/*.txt.

Projection-only — does not scan row data, only Parquet footers.
"""
from __future__ import annotations
from common import connect, theme_path, SCHEMA_DIR

TARGETS = [
    ("places", "place"),
    ("buildings", "building"),
    ("transportation", "segment"),
    ("transportation", "connector"),
    ("base", "land"),
    ("base", "land_use"),
    ("base", "land_cover"),
    ("base", "water"),
    ("base", "infrastructure"),
    ("base", "bathymetry"),
    ("addresses", "address"),
    ("divisions", "division"),
    ("divisions", "division_area"),
    ("divisions", "division_boundary"),
]

con = connect()
for theme, type_ in TARGETS:
    path = theme_path(theme, type_)
    print(f"-- {theme}/{type_}")
    try:
        df = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}', hive_partitioning=true) LIMIT 0"
        ).fetchdf()
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    out = SCHEMA_DIR / f"{theme}__{type_}.txt"
    out.write_text(df.to_string(index=False))
    print(f"  wrote {out.relative_to(SCHEMA_DIR.parent)}  ({len(df)} columns)")
