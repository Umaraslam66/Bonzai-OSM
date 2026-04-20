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
        # TagInfo's `total` field on this endpoint is the count of
        # *unique values*, not the sum of occurrences. Each row already
        # has a `fraction` (share of occurrences, 0..1) which is the
        # correct thing to convert to a percentage. We also derive the
        # true total-occurrences count from the highest-count row's
        # fraction so the CSV remains self-contained.
        def pct(row):
            frac = row.get("fraction")
            if frac is None and rows:
                # Fallback: approximate from the top row when fraction
                # is absent (older TagInfo schema).
                return round(100.0 * row["count"] / max(sum(r["count"] for r in rows), 1), 4)
            return round(100.0 * float(frac), 4)

        top_row = rows[0]
        if top_row.get("fraction"):
            total_occurrences = int(round(top_row["count"] / float(top_row["fraction"])))
        else:
            total_occurrences = sum(r["count"] for r in rows)

        out_csv = dst / f"eda_{key}_counts.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tag_value", "count", "percentage"])
            for r in rows:
                writer.writerow([r["value"], r["count"], pct(r)])

        summary[key] = {
            "total_occurrences": total_occurrences,
            "unique_values_reported": payload.get("total"),
            "rows_returned": len(rows),
            "top": [
                {"value": r["value"], "count": r["count"], "pct": pct(r)}
                for r in rows[: args.top]
            ],
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
