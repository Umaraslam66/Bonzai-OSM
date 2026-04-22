"""Synthesize freq_*.csv into a vocab-decision report.

Pure local. Reads outputs/freq_*.csv and prints, per field:
  - total labelled rows
  - distinct values
  - null count & pct (value is NaN)
  - distinct values surviving floor thresholds
  - top-5 values with pct
  - recommended action

Writes outputs/ANALYSIS.md.
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

FLOORS = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
VOCAB_FLOOR = 10_000
ASPIRATIONAL = 100_000


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # older CSVs used 'category' as the value-column name
    if "value" not in df.columns and "category" in df.columns:
        df = df.rename(columns={"category": "value"})
    df["is_null"] = df["value"].isna()
    return df


def analyze_one(csv: Path) -> dict:
    df = load(csv)
    total = int(df["n"].sum())
    null_n = int(df.loc[df["is_null"], "n"].sum())
    nonnull = df[~df["is_null"]].copy()
    distinct = len(nonnull)
    top5 = nonnull.head(5)
    cuts = {f">=_{f}": int((nonnull["n"] >= f).sum()) for f in FLOORS}
    return {
        "field": csv.stem.replace("freq_", ""),
        "total": total,
        "null_n": null_n,
        "null_pct": round(100 * null_n / total, 2) if total else 0,
        "distinct_nonnull": distinct,
        "singletons": int((nonnull["n"] == 1).sum()),
        **cuts,
        "top5": [(str(r["value"]), int(r["n"]), round(float(r["pct"]), 2)) for _, r in top5.iterrows()],
    }


def recommend(row: dict) -> str:
    d = row["distinct_nonnull"]
    null = row["null_pct"]
    above_floor = row[f">=_{VOCAB_FLOOR}"]
    total = row["total"]

    # high null → structural problem, not vocab size
    if null >= 60:
        return (f"SKIP-OR-BINARY: {null:.0f}% null — label mostly absent. "
                f"Either drop this field from vocab, or compress to a single "
                f"is_present bit.")
    # tiny vocab
    if d <= 30 and null < 10:
        return f"KEEP ALL {d} values (tight vocab, all well-populated). "
    # moderate
    if d <= 300 and above_floor / max(d, 1) > 0.5:
        return (f"KEEP all {d} values; {above_floor}/{d} already >10k floor. "
                f"Candidate for primary vocab layer.")
    # wide tail
    if d > 300:
        return (f"TAIL-TRIM: keep {above_floor} values >= {VOCAB_FLOOR:,}, "
                f"bucket the remaining {d - above_floor} into <other>. "
                f"Or use hierarchical fallback.")
    return "REVIEW manually."


def main() -> None:
    csvs = sorted(OUTPUTS.glob("freq_*.csv"))
    rows = [analyze_one(p) for p in csvs]
    for r in rows:
        r["action"] = recommend(r)

    # summary table
    cols = ["field", "total", "null_pct", "distinct_nonnull",
            ">=_1000", ">=_10000", ">=_100000", ">=_1000000",
            "singletons", "action"]
    df = pd.DataFrame(rows)[cols]

    print("=" * 110)
    print(" VOCAB-DECISION SUMMARY (global counts, release 2026-04-15.0)")
    print("=" * 110)
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(df.to_string(index=False))

    print("\n" + "=" * 110)
    print(" TOP-5 PER FIELD")
    print("=" * 110)
    for r in rows:
        print(f"\n-- {r['field']}")
        print(f"   total={r['total']:,}  null={r['null_pct']}%  distinct={r['distinct_nonnull']}")
        for v, n, pct in r["top5"]:
            print(f"     {n:>12,}  {pct:>6.2f}%  {v}")

    # write a markdown version
    md = ["# Overture vocab analysis (release 2026-04-15.0)", ""]
    md += ["> `distinct_nonnull` excludes NULL. Thresholds are counts (floor)."]
    md += ["", "## Field summary", "", df.to_markdown(index=False)]
    md += ["", "## Per-field top-5"]
    for r in rows:
        md += [f"\n### `{r['field']}`",
               f"- total: **{r['total']:,}**",
               f"- null: **{r['null_pct']}%** ({r['null_n']:,})",
               f"- distinct non-null: **{r['distinct_nonnull']}**",
               f"- above {VOCAB_FLOOR:,} floor: **{r[f'>=_{VOCAB_FLOOR}']}**",
               f"- above {ASPIRATIONAL:,} aspirational: **{r[f'>=_{ASPIRATIONAL}']}**",
               f"- singletons: {r['singletons']}",
               f"- recommended: _{r['action']}_", "",
               "| count | pct | value |",
               "|---:|---:|:---|"]
        for v, n, pct in r["top5"]:
            md.append(f"| {n:,} | {pct}% | `{v}` |")
    docs_dir = OUTPUTS.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    out = docs_dir / "VOCAB_ANALYSIS.md"
    out.write_text("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
