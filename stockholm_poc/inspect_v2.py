"""Post-run inspector for the v2 Stockholm tokenizer output.

Prints the vocabulary audit (unique tokens per family), the Parquet
row/row-group counts, per-kind object counts, and one sample token
sequence for each feature class.
"""
import os
import json
from collections import Counter

import pyarrow.parquet as pq


def main() -> None:
    work = os.environ["WORK"]
    vocab_path = os.path.join(work, "stockholm_poc/outputs/stockholm_vocab.json")
    parquet_path = os.path.join(work, "stockholm_poc/outputs/stockholm_tokens.parquet")

    v = json.load(open(vocab_path))
    fam = Counter(
        t.lstrip("<").rstrip(">").split("_", 1)[0] for t in v["tokens"]
    )
    print("total vocab:", v["size"])
    for k, n in fam.most_common():
        print(f"  {k:<14} {n}")

    pf = pq.ParquetFile(parquet_path)
    tbl = pf.read().to_pandas()
    print("rows:", len(tbl), "row groups:", pf.num_row_groups)
    print("by kind:", tbl["kind"].value_counts().to_dict())

    for kind in ["BUILDING", "ROAD", "POI", "LANDUSE", "WATERWAY", "RAILWAY"]:
        sel = tbl[tbl["kind"] == kind]
        if len(sel) == 0:
            print(f"--- {kind}: (none)")
            continue
        print(f"--- {kind} ---")
        print(list(sel.iloc[0]["tokens"])[:18])


if __name__ == "__main__":
    main()
