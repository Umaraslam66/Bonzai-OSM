"""Structural tests for Plan 3 sbatch + orchestrator."""
import json
import sys
from pathlib import Path


def test_leonardo_plan3_sbatch_uses_4_gpus():
    """The Plan 3 sbatch must request all 4 A100s on the boost_usr_prod
    node. boost_usr_prod bills per node regardless of GPU count, so a
    1-GPU job wastes 75% of the billed hour. See
    feedback_leonardo_full_node.md.
    """
    repo = Path(__file__).resolve().parents[1]
    sbatch = (repo / "scripts" / "leonardo_plan3.sbatch").read_text()
    assert "--gres=gpu:4" in sbatch
    assert "--ntasks-per-node=4" in sbatch
    assert "--partition=boost_usr_prod" in sbatch
    assert "--account=AIFAC_P02_222" in sbatch
    assert "BONZAI_DDP_DEVICES" in sbatch


def test_run_plan3_weights_inversely_proportional_to_counts():
    """weights_from_counts: rarest country -> weight 1.0;
    weight scales inversely with count.
    """
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    if "run_plan3" in sys.modules:
        del sys.modules["run_plan3"]
    import run_plan3
    weights = run_plan3.weights_from_counts({"sweden": 1301, "sri_lanka": 384, "singapore": 203})
    assert weights["singapore"] == 1.0  # rarest is keep-always
    assert weights["sweden"] < weights["sri_lanka"] < weights["singapore"]
    # Sweden weight ~= 203/1301 = 0.156
    assert abs(weights["sweden"] - 203 / 1301) < 1e-9


def test_run_plan3_submits_two_parallel_jobs(monkeypatch, tmp_path):
    """The orchestrator must submit one Painter and one Writer job, both
    seeing the same BONZAI_COUNTRY_WEIGHTS_JSON value. We use a fake
    sbatch that records argv to a log file.
    """
    fake_sbatch = tmp_path / "fake_sbatch.sh"
    fake_sbatch.write_text("#!/bin/bash\necho stub-sbatch\n")
    fake_sbatch.chmod(0o755)
    log = tmp_path / "submitted.log"
    log.write_text("")

    monkeypatch.setenv("BONZAI_FAKE_SBATCH", str(fake_sbatch))
    monkeypatch.setenv("BONZAI_FAKE_SBATCH_LOG", str(log))
    monkeypatch.setenv("BONZAI_TRAIN_URL", "stub://train")
    monkeypatch.setenv("BONZAI_VAL_URL", "stub://val")
    monkeypatch.setenv("BONZAI_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("BONZAI_VAE_CKPT", str(tmp_path / "vae.ckpt"))
    monkeypatch.setenv(
        "BONZAI_COUNTRY_COUNTS_JSON",
        json.dumps({"sweden": 1301, "sri_lanka": 384, "singapore": 203}),
    )

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    if "run_plan3" in sys.modules:
        del sys.modules["run_plan3"]
    import run_plan3
    run_plan3.main()

    log_lines = [line for line in log.read_text().strip().split("\n") if line]
    stages = sorted(line.split("\t")[0] for line in log_lines)
    assert stages == ["stage_a", "stage_b"]
    weights_seen = {line.split("\t")[1] for line in log_lines}
    assert len(weights_seen) == 1  # both jobs got the same weights JSON
    weights = json.loads(weights_seen.pop())
    assert set(weights) == {"sweden", "sri_lanka", "singapore"}
    assert weights["singapore"] > weights["sweden"]
