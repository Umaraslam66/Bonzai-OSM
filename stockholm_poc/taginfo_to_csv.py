"""
taginfo_to_csv.py
=================

Convert TagInfo per-key JSON responses into the same CSV format
`global_osm_eda.py` produces: `tag_value, count, percentage`.

Reads every `<key>.json` in --input-dir (files produced by e.g.
    curl 'https://taginfo.openstreetmap.org/api/4/key/values?key=amenity&rp=500'
written as `amenity.json`) and writes matching `eda_<key>_counts.csv`
into --output-dir.

Also prints a summary with the top-10 values per key so you can
eyeball the tokenizer vocabulary without opening each CSV.
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/tmp/taginfo",
                        help="Directory with <key>.json files from TagInfo")
    parser.add_argument("--output-dir", default="/tmp/taginfo/csv",
                        help="Where to write eda_<key>_counts.csv files")
    parser.add_argument("--top", type=int, default=10,
                        help="How many top-N entries to print per key")
    args = parser.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    summary = {}
    for path in sorted(src.glob("*.json")):
        key = path.stem
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "data" not in payload:
            print(f"{key}: no 'data' field (error? payload={payload!r})")
            continue

        rows = payload["data"]
        total = payload.get("total", sum(r["count"] for r in rows))

        out_csv = dst / f"eda_{key}_counts.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tag_value", "count", "percentage"])
            for r in rows:
                pct = round(100.0 * r["count"] / total, 4) if total else 0.0
                writer.writerow([r["value"], r["count"], pct])

        summary[key] = {
            "total": total,
            "unique_returned": len(rows),
            "top": [
                {"value": r["value"], "count": r["count"],
                 "pct": round(100.0 * r["count"] / total, 3) if total else 0.0}
                for r in rows[: args.top]
            ],
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
