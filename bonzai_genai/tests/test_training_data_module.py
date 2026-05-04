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


def _np_save_bytes(arr):
    import io

    import numpy as np
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_data_module_decoded_sample_has_country():
    """Each decoded sample must expose its TileMetadata.country string.

    Country lives in the .json sidecar of each WebDataset record. Without
    surfacing it the country-balanced sampler (T3) has nothing to weight by.
    """
    import json

    import numpy as np

    from bonzai_genai.training.data_module import _decode_bundle

    fake_sample = {
        "raster.npy": _np_save_bytes(np.zeros((9, 512, 512), dtype=np.float32)),
        "tokens.json": b"[1,2,3]",
        "metadata.json": json.dumps({
            "tile_id": "synth-0", "sw_lat": 0.0, "sw_lon": 0.0,
            "country": "synth", "koppen": "Af",
            "density_bucket": "rural", "primary_land_use": "green",
        }).encode(),
    }
    decoded = _decode_bundle(fake_sample)
    assert "country" in decoded
    assert decoded["country"] == "synth"


def test_data_module_decoded_sample_falls_back_to_unknown_country():
    """If a record has no metadata.json (older shards), country is 'unknown'."""
    import numpy as np

    from bonzai_genai.training.data_module import _decode_bundle
    fake_sample = {
        "raster.npy": _np_save_bytes(np.zeros((9, 512, 512), dtype=np.float32)),
        "tokens.json": b"[1,2,3]",
    }
    decoded = _decode_bundle(fake_sample)
    assert decoded["country"] == "unknown"
