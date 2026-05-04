"""Experiment 0 orchestrator: VAE -> DiT -> Inker -> eval -> report.

All artefacts go under $BONZAI_EXP0_OUT (default $WORK/bonzai-exp0).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

WORK = Path(os.environ["WORK"])
OUT = Path(os.environ.get("BONZAI_EXP0_OUT", WORK / "bonzai-exp0"))
OUT.mkdir(parents=True, exist_ok=True)
SHARD_TRAIN = WORK / "bonzai-tiles" / "synth" / "train"
SHARD_VAL = WORK / "bonzai-tiles" / "synth" / "val"
TRAIN_URL = f"{SHARD_TRAIN}/shard-{{000000..000008}}.tar"
VAL_URL = f"{SHARD_VAL}/shard-{{000000..000000}}.tar"


def run(stage: str, max_epochs: int, batch_size: int, out_subdir: str, env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    env.update({
        "BONZAI_STAGE": stage,
        "BONZAI_PRESET": "tiny",
        "BONZAI_TRAIN_URL": TRAIN_URL,
        "BONZAI_VAL_URL": VAL_URL,
        "BONZAI_OUT": str(OUT / out_subdir),
        "BONZAI_BATCH_SIZE": str(batch_size),
        "BONZAI_MAX_EPOCHS": str(max_epochs),
    })
    if env_extra:
        env.update(env_extra)
    runner = Path(__file__).resolve().parent / "_train_runner.py"
    subprocess.run(["python", str(runner)], env=env, check=True)


def find_latest_ckpt(stage_dir: Path) -> str:
    candidates = list(stage_dir.rglob("*.ckpt"))
    if not candidates:
        raise SystemExit(f"no checkpoint under {stage_dir}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main() -> None:
    print("=== Experiment 0 ===", flush=True)
    print(f"Output dir: {OUT}", flush=True)

    print("\n[1/4] Training tiny VAE...", flush=True)
    run("vae", max_epochs=50, batch_size=8, out_subdir="vae")

    vae_ckpt = find_latest_ckpt(OUT / "vae")
    print(f"VAE checkpoint: {vae_ckpt}", flush=True)

    print("\n[2/4] Training tiny Stage A (DiT)...", flush=True)
    run(
        "stage_a", max_epochs=1, batch_size=8, out_subdir="stage_a",
        env_extra={"BONZAI_VAE_CKPT": vae_ckpt},
    )

    print("\n[3/4] Training tiny Stage B (Inker)...", flush=True)
    run("stage_b", max_epochs=1, batch_size=4, out_subdir="stage_b")

    print("\n[4/4] Running eval suite...", flush=True)
    eval_runner = Path(__file__).resolve().parent / "run_eval.py"
    env = os.environ.copy()
    env.update({
        "BONZAI_EXP0_OUT": str(OUT),
        "BONZAI_VAL_URL": VAL_URL,
    })
    subprocess.run(["python", str(eval_runner)], env=env, check=True)

    print("\nExperiment 0 complete.", flush=True)
    print(f"Report: {OUT / 'EXPERIMENT_0_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
