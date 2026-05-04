"""Plan 3 orchestrator -- submits Painter + Writer training in parallel.

Computes country weights from a one-pass scan over BONZAI_TRAIN_URL,
forwards the resulting JSON to two sbatch jobs (Painter + Writer), and
finally prints the BONZAI_SAMPLE_FROM_CKPT command to run after both
finish.

Env-vars consumed:
    BONZAI_TRAIN_URL                   WebDataset shard glob (all 3 countries)
    BONZAI_VAL_URL                     held-out glob
    BONZAI_OUT                         output root on $WORK (subdirs created)
    BONZAI_VAE_CKPT                    frozen VAE ckpt path for Stage A

    BONZAI_COUNTRY_COUNTS_JSON         (test-only) skip the shard scan;
                                       use this dict directly.
    BONZAI_FAKE_SBATCH                 (test-only) path to fake sbatch script
                                       used in place of real `sbatch`.
    BONZAI_FAKE_SBATCH_LOG             (test-only) where the fake submit
                                       should append "<stage>\\t<weights>\\n".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def count_countries(train_url: str) -> dict[str, int]:
    """One-pass scan of the train shards counting metadata.country.

    Lightweight: only loads the metadata.json sidecar of each record,
    skipping raster/tokens. ~10 s for 1,888 tiles.
    """
    import webdataset as wds  # noqa: PLC0415
    counts: dict[str, int] = {}
    ds = wds.WebDataset(train_url, shardshuffle=False, empty_check=False)
    for sample in ds:
        if "metadata.json" not in sample:
            continue
        meta = json.loads(sample["metadata.json"].decode("utf-8"))
        c = meta.get("country", "unknown")
        counts[c] = counts.get(c, 0) + 1
    return counts


def weights_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Inverse-proportional weights: rarest country gets weight 1.0."""
    if not counts:
        return {}
    n = min(counts.values())
    return {c: n / k for c, k in counts.items()}


def submit(stage: str, weights_json: str) -> None:
    """Submit one sbatch job for ``stage``.

    In production this calls ``sbatch scripts/leonardo_plan3.sbatch``;
    in tests it calls the fake sbatch in $BONZAI_FAKE_SBATCH and appends
    "<stage>\\t<weights_json>\\n" to $BONZAI_FAKE_SBATCH_LOG.
    """
    fake = os.environ.get("BONZAI_FAKE_SBATCH")
    if fake:
        log = os.environ["BONZAI_FAKE_SBATCH_LOG"]
        with open(log, "a") as f:
            f.write(f"{stage}\t{weights_json}\n")
        return

    out = Path(os.environ["BONZAI_OUT"]) / stage
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        BONZAI_STAGE=stage,
        BONZAI_PRESET="plan3",
        BONZAI_OUT=str(out),
        BONZAI_BATCH_SIZE="64" if stage == "stage_a" else "16",
        BONZAI_MAX_EPOCHS="50",
        BONZAI_COUNTRY_WEIGHTS_JSON=weights_json,
    )
    if stage == "stage_a":
        env["BONZAI_VAE_CKPT"] = os.environ["BONZAI_VAE_CKPT"]
    subprocess.run(
        ["sbatch", "scripts/leonardo_plan3.sbatch"], env=env, check=True,
    )


def main() -> None:
    counts_json = os.environ.get("BONZAI_COUNTRY_COUNTS_JSON")
    if counts_json:
        counts = json.loads(counts_json)
    else:
        train_url = os.environ["BONZAI_TRAIN_URL"]
        print(f"Scanning {train_url} for country counts...", flush=True)
        counts = count_countries(train_url)
        print(f"Country counts: {counts}", flush=True)

    weights = weights_from_counts(counts)
    weights_json = json.dumps(weights)
    print(f"Country weights: {weights}", flush=True)
    print(
        "Submitting Painter (stage_a) and Writer (stage_b) in parallel...",
        flush=True,
    )
    submit("stage_a", weights_json)
    submit("stage_b", weights_json)
    print(
        "Both jobs queued. After they finish, run:\n"
        "  BONZAI_SAMPLE_FROM_CKPT=1 \\\n"
        "  BONZAI_CKPT_DIR=$BONZAI_OUT \\\n"
        "  BONZAI_PRESET=plan3 \\\n"
        "  BONZAI_SAMPLE_OUT=$BONZAI_OUT/samples \\\n"
        "  BONZAI_NUM_SAMPLES=64 \\\n"
        "  python scripts/run_eval.py",
        flush=True,
    )


if __name__ == "__main__":
    main()
