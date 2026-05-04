"""Structural tests for Plan 3 sbatch + orchestrator."""
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
