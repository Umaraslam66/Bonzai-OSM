"""Tests for the LightningDataModule that wraps WebDataset shards."""
from pathlib import Path

import pytest
import torch


@pytest.fixture
def syn_corpus(tmp_path):
    """Build a tiny synth corpus in tmp_path/{train,val}/ for testing."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "corpus"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "prepare_tiles_local.py"),
            "synth-corpus",
            "-o", str(out),
            "--n-train", "20", "--n-val", "10", "--shard-size", "10",
        ],
        check=True,
    )
    return out


def test_data_module_yields_raster_only_batch(syn_corpus):
    from bonzai_genai.training.data_module import TileDataModule
    dm = TileDataModule(
        train_url=str(syn_corpus / "train" / "shard-{000000..000001}.tar"),
        val_url=str(syn_corpus / "val" / "shard-000000.tar"),
        batch_size=4,
        return_tokens=False,
        num_workers=0,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 9, 512, 512)
    assert batch.dtype == torch.float32


def test_data_module_yields_raster_and_tokens_batch(syn_corpus):
    from bonzai_genai.training.data_module import TileDataModule
    dm = TileDataModule(
        train_url=str(syn_corpus / "train" / "shard-{000000..000001}.tar"),
        val_url=str(syn_corpus / "val" / "shard-000000.tar"),
        batch_size=2,
        return_tokens=True,
        num_workers=0,
        max_token_len=4096,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert "raster" in batch and "tokens" in batch and "token_lens" in batch
    assert batch["raster"].shape == (2, 9, 512, 512)
    assert batch["tokens"].shape == (2, 4096)
    assert batch["tokens"].dtype == torch.long
    # token_lens are real lengths before pad
    assert (batch["token_lens"] <= 4096).all()
