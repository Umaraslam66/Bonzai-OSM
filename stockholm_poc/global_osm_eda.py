"""
global_osm_eda.py
=================

Global OSM Exploratory Data Analysis — scan the planet .osm.pbf once,
count every unique value of the six keys that dominate our downstream
tokenizer vocabulary, and write per-key CSVs sorted by frequency.

Target keys
-----------
amenity, shop, building, highway, landuse, natural

Pipeline
--------
1. **Stage 1 — pyosmium scan** (runs on the driver, single-threaded by
   design): one streaming pass over the PBF, accumulating
   `Counter(value -> count)` for each target key. Memory is bounded by
   the number of unique values across the planet (~10^5), not file
   size. Intermediate results are written as per-key Parquet.

2. **Stage 2 — PySpark aggregation** (distributed, local-mode by
   default on the Leonardo node): read the intermediate Parquet,
   compute percentages of each key's total, sort descending, and write
   one CSV per key in the requested format.

Why this shape (honest notes)
-----------------------------
- `.osm.pbf` is not splittable by Spark's default readers without
  external help (`osm-parquetizer`, Sedona's OSM datasource, etc.).
  The only dependable way to ingest raw tags from a planet PBF in
  reasonable time is pyosmium's streaming handler.
- Tag counting is O(tags) in time and O(unique-values) in memory, so
  Spark's parallelism gives us nothing at the parse stage. We use
  Spark for the post-parse aggregation + sort + CSV write so the
  scaffolding is there for heavier future queries (per-region counts,
  co-occurrence matrices, bbox filters) without rewriting the script.
- **Magellan** is abandoned (last commit 2018); do not use.
- **Sedona** would also work as the stage-2 engine but needs exact
  Sedona-JAR / Spark version alignment, which is fragile on Leonardo
  without a tested baseline. Plain PySpark here.

Usage
-----
    python global_osm_eda.py \\
        --input  /leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf \\
        --output-dir /leonardo_work/AIFAC_P02_222/eda/outputs \\
        --intermediate-dir /leonardo_work/AIFAC_P02_222/eda/intermediate

Flags:
    --skip-scan    skip stage 1 (re-use existing intermediate parquets)
    --skip-spark   skip stage 2 (parquet-only output)
    --keys ...     override the default target-key list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import osmium
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyosmium is required (pip install 'osmium>=3.7,<4')") from exc


TARGET_KEYS = ("amenity", "shop", "building", "highway", "landuse", "natural")

logger = logging.getLogger("global_osm_eda")


# ---------------------------------------------------------------------------
# Stage 1 — pyosmium scan
# ---------------------------------------------------------------------------


class TagCounter(osmium.SimpleHandler):
    """One streaming pass: for each target key, accumulate a
    `Counter(value -> count)` across nodes, ways, and relations.
    """

    def __init__(self, keys: Iterable[str]) -> None:
        super().__init__()
        self.keys = tuple(keys)
        self.counts: Dict[str, Counter[str]] = {k: Counter() for k in self.keys}
        self._n_nodes = 0
        self._n_ways = 0
        self._n_relations = 0

    def _ingest(self, tags) -> None:
        for key in self.keys:
            v = tags.get(key)
            if v is not None and v != "":
                self.counts[key][v] += 1

    def node(self, n) -> None:  # pyosmium callback
        self._n_nodes += 1
        if n.tags:
            self._ingest(n.tags)

    def way(self, w) -> None:  # pyosmium callback
        self._n_ways += 1
        if w.tags:
            self._ingest(w.tags)

    def relation(self, r) -> None:  # pyosmium callback
        self._n_relations += 1
        if r.tags:
            self._ingest(r.tags)


def scan_pbf(pbf_path: str, keys: Iterable[str]) -> Dict[str, Counter]:
    """Drive the TagCounter over the PBF. Uses `locations=False` because
    tag counting does not need node location reconstruction — this is
    what makes a planet-scale scan feasible on a modest node.
    """
    logger.info("stage 1: streaming PBF with pyosmium -> %s", pbf_path)
    handler = TagCounter(keys)
    handler.apply_file(pbf_path, locations=False)
    logger.info(
        "stage 1 done: %d nodes, %d ways, %d relations visited",
        handler._n_nodes, handler._n_ways, handler._n_relations,
    )
    for key, counter in handler.counts.items():
        logger.info(
            "  key=%s unique_values=%d total_occurrences=%d",
            key, len(counter), sum(counter.values()),
        )
    return handler.counts


def write_intermediate_parquet(
    counts: Dict[str, Counter], out_dir: str
) -> Dict[str, str]:
    """Persist per-key (value, count) pairs as Parquet so stage 2 can
    reread them without rescanning the PBF.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for key, counter in counts.items():
        if not counter:
            logger.warning("no tags found for key=%s; skipping", key)
            continue
        values = sorted(counter.keys())
        vals_count = [counter[v] for v in values]
        table = pa.table({"tag_value": values, "count": vals_count})
        p = out / f"{key}.parquet"
        pq.write_table(table, p, compression="zstd")
        paths[key] = str(p)
        logger.info("wrote %s (%d rows)", p, len(values))
    # Manifest for reproducibility.
    manifest = {
        "keys": list(paths.keys()),
        "parquet_paths": paths,
        "n_unique_per_key": {k: len(counts[k]) for k in paths},
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return paths


# ---------------------------------------------------------------------------
# Stage 2 — PySpark aggregation
# ---------------------------------------------------------------------------


def spark_aggregate_and_export(
    parquet_paths: Dict[str, str],
    output_dir: str,
    driver_memory: str,
    num_shuffle_partitions: int,
) -> None:
    """Read per-key intermediate parquet, compute percentage-of-total
    per value, sort by count descending, write per-key CSV.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyspark is required for stage 2 "
            "(pip install 'pyspark>=3.5')"
        ) from exc

    # Spark local mode — we are running on one Leonardo node. Workers
    # live in the same JVM as the driver so no cluster manager is needed.
    master = os.environ.get("SPARK_MASTER", "local[*]")
    logger.info("stage 2: starting Spark (master=%s, driver_mem=%s)", master, driver_memory)

    spark = (
        SparkSession.builder
        .appName("global_osm_eda")
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(num_shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    try:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        summary: List[dict] = []
        for key, path in parquet_paths.items():
            logger.info("aggregating key=%s from %s", key, path)
            df = spark.read.parquet(path)

            total = df.agg(F.sum("count").alias("t")).collect()[0]["t"] or 0
            if total == 0:
                logger.warning("key=%s has total=0; skipping CSV write", key)
                continue

            enriched = (
                df.withColumn(
                    "percentage",
                    F.round(F.col("count") / F.lit(total) * F.lit(100.0), 4),
                )
                .select("tag_value", "count", "percentage")
                .orderBy(F.col("count").desc())
            )

            # coalesce(1) -> single CSV file inside the output dir so
            # downstream tools can open it directly in Excel.
            csv_dir = out_root / f"eda_{key}_counts"
            (
                enriched.coalesce(1)
                .write.option("header", "true")
                .mode("overwrite")
                .csv(str(csv_dir))
            )
            # Promote the spark-written part-*.csv to a stable filename
            # at the sibling level so humans don't have to dig.
            _promote_spark_csv(csv_dir, out_root / f"eda_{key}_counts.csv")

            top5 = enriched.limit(5).collect()
            logger.info(
                "  key=%s total=%d unique=%d top5=%s",
                key, total, df.count(),
                [(r["tag_value"], r["count"], float(r["percentage"])) for r in top5],
            )
            summary.append({
                "key": key,
                "total_occurrences": int(total),
                "unique_values": int(df.count()),
                "top5": [
                    {
                        "value": r["tag_value"],
                        "count": int(r["count"]),
                        "percentage": float(r["percentage"]),
                    }
                    for r in top5
                ],
            })

        with (out_root / "eda_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    finally:
        spark.stop()


def _promote_spark_csv(spark_dir: Path, target_file: Path) -> None:
    """Spark's CSV writer emits a directory with part-*.csv files. With
    coalesce(1) we get exactly one part file; copy it to a stable
    sibling path so the human-facing output is a single .csv.
    """
    parts = sorted(spark_dir.glob("part-*.csv"))
    if not parts:
        return
    target_file.write_bytes(parts[0].read_bytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the planet (or extract) .osm.pbf")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directory to write per-key CSVs + summary JSON")
    parser.add_argument("--intermediate-dir", default=None,
                        help="Directory to write per-key intermediate parquets "
                             "(default: <output-dir>/intermediate)")
    parser.add_argument("--keys", nargs="+", default=list(TARGET_KEYS),
                        help=f"Tag keys to aggregate (default: {' '.join(TARGET_KEYS)})")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip stage 1 and reuse existing intermediate parquets")
    parser.add_argument("--skip-spark", action="store_true",
                        help="Skip stage 2 (leave intermediate parquets only)")
    parser.add_argument("--spark-driver-memory", default="32g",
                        help="Memory for the Spark driver (default: 32g)")
    parser.add_argument("--spark-shuffle-partitions", type=int, default=8,
                        help="spark.sql.shuffle.partitions (default: 8)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    inter_dir = args.intermediate_dir or os.path.join(args.output_dir, "intermediate")

    if not args.skip_scan:
        if not os.path.isfile(args.input):
            logger.error("PBF not found: %s", args.input)
            return 2
        counts = scan_pbf(args.input, args.keys)
        parquet_paths = write_intermediate_parquet(counts, inter_dir)
    else:
        logger.info("--skip-scan: loading manifest from %s", inter_dir)
        manifest_path = Path(inter_dir) / "manifest.json"
        if not manifest_path.exists():
            logger.error("no manifest.json in %s", inter_dir)
            return 2
        with manifest_path.open("r", encoding="utf-8") as fh:
            parquet_paths = json.load(fh)["parquet_paths"]

    if args.skip_spark:
        logger.info("--skip-spark: done after stage 1")
        return 0

    spark_aggregate_and_export(
        parquet_paths=parquet_paths,
        output_dir=args.output_dir,
        driver_memory=args.spark_driver_memory,
        num_shuffle_partitions=args.spark_shuffle_partitions,
    )

    logger.info("done. per-key CSVs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
