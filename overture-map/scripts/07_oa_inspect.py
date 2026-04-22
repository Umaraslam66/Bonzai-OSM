"""Inspect the OpenAddresses global archive without fully extracting it.

The zip contains one subfolder per source (country/state/city), each with its
own small CSV plus a VRT/JSON manifest. We list the archive contents, count
records per source by peeking inside CSVs, and write summary CSVs to outputs/.

Run:
    OA_ZIP=/path/to/openaddr-collected-global.zip python 07_oa_inspect.py

Does NOT unzip the whole thing (would be hundreds of GB expanded).
Uses zipfile.open() on individual CSV entries.
"""
from __future__ import annotations
import csv
import io
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
SCHEMA_DIR = ROOT / "schema"
SCHEMA_DIR.mkdir(exist_ok=True)

DEFAULT_ZIP = "/leonardo_scratch/large/userexternal/uaslam00/bonzai-data/openaddresses/openaddr-collected-global.zip"
OA_ZIP = os.environ.get("OA_ZIP", DEFAULT_ZIP)

if not os.path.isfile(OA_ZIP):
    print(f"ERROR: OA zip not found at {OA_ZIP}", file=sys.stderr)
    sys.exit(1)

print(f"[oa] opening {OA_ZIP}  ({os.path.getsize(OA_ZIP)/1e9:.2f} GB)")
zf = zipfile.ZipFile(OA_ZIP, "r")
members = zf.namelist()
print(f"[oa] {len(members):,} entries inside")

# Group by top-level country (first path segment), count CSV rows
per_country = defaultdict(lambda: {"sources": 0, "csv_bytes": 0})
csv_members = []
for name in members:
    parts = name.split("/")
    if len(parts) < 2:
        continue
    country = parts[0]
    per_country[country]["sources"] += 1
    if name.endswith(".csv"):
        csv_members.append(name)
        info = zf.getinfo(name)
        per_country[country]["csv_bytes"] += info.file_size

print(f"[oa] {len(csv_members):,} CSV members across {len(per_country)} countries")

# Sample a small chunk of the first CSV to learn the column schema
if csv_members:
    first = csv_members[0]
    with zf.open(first) as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
    schema_out = SCHEMA_DIR / "openaddresses_csv.txt"
    schema_out.write_text(f"Sample member: {first}\n\n{head[:2000]}")
    print(f"[oa] wrote sample schema to {schema_out.relative_to(ROOT)}")

# Summary CSV
rows = []
for country, d in sorted(per_country.items(), key=lambda x: -x[1]["csv_bytes"]):
    rows.append({
        "country": country,
        "source_files": d["sources"],
        "csv_bytes": d["csv_bytes"],
        "csv_mb": round(d["csv_bytes"] / 1e6, 1),
    })

out = OUTPUTS / "oa_country_summary.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=["country", "source_files", "csv_bytes", "csv_mb"])
    w.writeheader()
    w.writerows(rows)
print(f"[oa] wrote {out.relative_to(ROOT)}")
print()
print("Top-10 countries by CSV bytes:")
for r in rows[:10]:
    print(f"  {r['country']:<20} {r['source_files']:>5} sources  {r['csv_mb']:>8.1f} MB")
