"""Print a concise human-readable inventory of:
  - fields (columns) per theme — from schema/*.txt
  - value distributions we already have on disk — from outputs/freq_*.csv

Pure local — does NOT hit S3.
"""
from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
from common import SCHEMA_DIR, OUTPUTS

print("=" * 70)
print("  ATTRIBUTE FIELDS PER THEME (from Parquet DESCRIBE)")
print("=" * 70)
for p in sorted(SCHEMA_DIR.glob("*.txt")):
    df = pd.read_fwf(p)
    # drop hive-partition meta and rebuild from first two cols
    # simpler: read raw and pretty-print
    lines = p.read_text().splitlines()
    print(f"\n--- {p.stem} ---")
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # grab the column name (first token) and the column_type (everything before YES/NO null flag)
        m = re.match(r"^\s*(\S+)\s+(.+?)\s+(YES|NO)\s+.*$", ln)
        if m:
            name, typ, _ = m.groups()
            # truncate very long struct types for readability
            if len(typ) > 90:
                typ = typ[:87] + "..."
            print(f"  {name:<25} {typ}")

print("\n" + "=" * 70)
print("  VALUE DISTRIBUTIONS ON DISK")
print("=" * 70)

for csv in sorted(OUTPUTS.glob("freq_*.csv")):
    df = pd.read_csv(csv)
    df = df[df["category"].notna()]
    total = int(df["n"].sum())
    buckets = {
        ">= 1,000,000": int((df["n"] >= 1_000_000).sum()),
        ">= 100,000":   int((df["n"] >= 100_000).sum()),
        ">= 10,000":    int((df["n"] >= 10_000).sum()),
        ">= 1,000":     int((df["n"] >= 1_000).sum()),
        ">= 100":       int((df["n"] >= 100).sum()),
        ">= 10":        int((df["n"] >= 10).sum()),
        ">= 1":         int((df["n"] >= 1).sum()),
    }
    rare_n = int((df["n"] == 1).sum())
    print(f"\n--- {csv.stem} ---")
    print(f"  total rows with a value: {total:,}")
    print(f"  distinct values:         {len(df):,}")
    print(f"  singleton (appears once): {rare_n:,}")
    for k, v in buckets.items():
        print(f"    count {k:<14} {v:>6,} distinct values")
