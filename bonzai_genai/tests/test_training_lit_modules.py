"""Lightning module smoke tests: each module should run forward + backward + step on tiny inputs."""
import pytest
import torch

from bonzai_genai.models.configs import TinyPreset, VAEConfig


@pytest.fixture
def synth_raster_batch():
    # Tiny batch shaped like the real one (B, 9, 512, 512) but small B=1
    x = torch.zeros(1, 9, 512, 512)
    x[:, :5] = (torch.rand_like(x[:, :5]) > 0.7).float()
    x[:, 5] = torch.rand_like(x[:, 5])
    x[:, 6:] = (torch.rand_like(x[:, 6:]) > 0.7).float()
    return x


def test_lit_vae_one_training_step(synth_raster_batch):
    from bonzai_genai.training.lit_vae import LitVAE
    lit = LitVAE(vae_config=VAEConfig.from_preset(TinyPreset), kl_weight=0.01, lr=1e-4)
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    loss = lit.training_step(synth_raster_batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_lit_stage_a_one_training_step(synth_raster_batch):
    from bonzai_genai.models.configs import DiTConfig
    from bonzai_genai.training.lit_stage_a import LitStageA
    lit = LitStageA(
        dit_config=DiTConfig.from_preset(TinyPreset),
        vae_config=VAEConfig.from_preset(TinyPreset),
        cfg_dropout_prob=0.1,
        lr=1e-4,
    )
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    loss = lit.training_step(synth_raster_batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_lit_stage_b_one_training_step():
    from bonzai_genai.models.configs import (
        InkerConfig,
        RasterEncoderConfig,
    )
    from bonzai_genai.training.lit_stage_b import LitStageB
    lit = LitStageB(
        inker_config=InkerConfig.from_preset(TinyPreset),
        raster_encoder_config=RasterEncoderConfig.from_preset(TinyPreset),
        lr=3e-4,
    )
    opt = lit.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    batch = {
        "raster": torch.randn(1, 9, 512, 512),
        "tokens": torch.randint(0, 1000, (1, 64)),
        "token_lens": torch.tensor([64]),
    }
    loss = lit.training_step(batch, batch_idx=0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_train_runner_builds_ddp_trainer_when_devices_gt_1(monkeypatch, tmp_path):
    """`BONZAI_DDP_DEVICES=4` should construct a Trainer with devices=4
    and strategy='ddp'. We patch lightning.Trainer to a sentinel that
    captures kwargs.
    """
    import importlib
    import sys
    from pathlib import Path

    monkeypatch.setenv("BONZAI_STAGE", "vae")
    monkeypatch.setenv("BONZAI_PRESET", "tiny")
    monkeypatch.setenv("BONZAI_TRAIN_URL", "stub://train")
    monkeypatch.setenv("BONZAI_VAL_URL", "stub://val")
    monkeypatch.setenv("BONZAI_OUT", str(tmp_path))
    monkeypatch.setenv("BONZAI_BATCH_SIZE", "1")
    monkeypatch.setenv("BONZAI_MAX_EPOCHS", "0")
    monkeypatch.setenv("BONZAI_DDP_DEVICES", "4")

    captured: dict = {}

    class _StubTrainer:
        def __init__(self, **kw):
            captured.update(kw)

        def fit(self, *a, **kw):
            return None

    import lightning as L  # noqa: N812
    monkeypatch.setattr(L, "Trainer", _StubTrainer)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    if "_train_runner" in sys.modules:
        del sys.modules["_train_runner"]
    runner = importlib.import_module("_train_runner")
    runner.main()

    assert captured.get("devices") == 4
    assert captured.get("strategy") == "ddp"
    assert captured.get("use_distributed_sampler") is False


def _assert_all_log_calls_use_sync_dist(cls) -> None:
    """AST walk: every self.log / self.log_dict call must pass sync_dist."""
    import ast
    import inspect
    src = inspect.getsource(cls)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("log", "log_dict"):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
            continue
        kwargs = {kw.arg for kw in node.keywords}
        assert "sync_dist" in kwargs, (
            f"missing sync_dist=True in: {ast.unparse(node)}"
        )


def test_stage_a_logging_uses_sync_dist():
    """Under DDP, self.log() without sync_dist=True logs only rank 0's
    value, silently desyncing metrics."""
    from bonzai_genai.training.lit_stage_a import LitStageA
    _assert_all_log_calls_use_sync_dist(LitStageA)


def test_stage_b_logging_uses_sync_dist():
    from bonzai_genai.training.lit_stage_b import LitStageB
    _assert_all_log_calls_use_sync_dist(LitStageB)


def test_vae_logging_uses_sync_dist():
    from bonzai_genai.training.lit_vae import LitVAE
    _assert_all_log_calls_use_sync_dist(LitVAE)


def test_train_runner_single_gpu_trainer_when_devices_unset(monkeypatch, tmp_path):
    """If BONZAI_DDP_DEVICES is unset (or 1), Trainer is constructed
    without strategy='ddp' (single-process mode), preserving the
    Phase 0b smoke behaviour.
    """
    import importlib
    import sys
    from pathlib import Path

    monkeypatch.setenv("BONZAI_STAGE", "vae")
    monkeypatch.setenv("BONZAI_PRESET", "tiny")
    monkeypatch.setenv("BONZAI_TRAIN_URL", "stub://train")
    monkeypatch.setenv("BONZAI_VAL_URL", "stub://val")
    monkeypatch.setenv("BONZAI_OUT", str(tmp_path))
    monkeypatch.setenv("BONZAI_BATCH_SIZE", "1")
    monkeypatch.setenv("BONZAI_MAX_EPOCHS", "0")
    monkeypatch.delenv("BONZAI_DDP_DEVICES", raising=False)

    captured: dict = {}

    class _StubTrainer:
        def __init__(self, **kw):
            captured.update(kw)

        def fit(self, *a, **kw):
            return None

    import lightning as L  # noqa: N812
    monkeypatch.setattr(L, "Trainer", _StubTrainer)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    if "_train_runner" in sys.modules:
        del sys.modules["_train_runner"]
    runner = importlib.import_module("_train_runner")
    runner.main()

    # Single-GPU path: strategy not set or default
    assert captured.get("strategy") != "ddp"
