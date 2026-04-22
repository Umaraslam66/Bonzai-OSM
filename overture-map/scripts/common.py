"""Shared helpers for Overture freq-analysis scripts.

Pin the release here. Every script imports RELEASE and S3_BASE from this module
so re-running against a future release is a single-line change.
"""
from __future__ import annotations
import duckdb
from pathlib import Path

RELEASE = "2026-04-15.0"
S3_BUCKET = "overturemaps-us-west-2"
S3_BASE = f"s3://{S3_BUCKET}/release/{RELEASE}"

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SCHEMA_DIR = ROOT / "schema"
PLOTS = OUTPUTS / "plots"
for d in (OUTPUTS, SCHEMA_DIR, PLOTS):
    d.mkdir(parents=True, exist_ok=True)


def connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """Get a configured DuckDB connection.

    threads=None uses DuckDB's default (one per core). Set lower on laptops
    to avoid exhausting FDs / sockets when scanning many Parquet files.
    """
    con = duckdb.connect()
    if threads is not None:
        con.execute(f"SET threads TO {threads};")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET http_retries=10;")
    con.execute("SET http_retry_wait_ms=500;")
    con.execute("SET http_timeout=120000;")
    return con


def theme_path(theme: str, type_: str) -> str:
    return f"{S3_BASE}/theme={theme}/type={type_}/*"


STOCKHOLM_BBOX = dict(minx=17.8, miny=59.2, maxx=18.3, maxy=59.5)


def bbox_filter_sql(bbox: dict | None) -> str:
    if not bbox:
        return ""
    return (
        f" WHERE bbox.xmin >= {bbox['minx']} AND bbox.xmax <= {bbox['maxx']}"
        f" AND bbox.ymin >= {bbox['miny']} AND bbox.ymax <= {bbox['maxy']}"
    )
