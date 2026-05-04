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


def test_country_balance_filter_balances_imbalanced_stream():
    """Given a stream with country counts {SE:30, SR:10, SG:5}, the
    rejection filter should yield roughly equal counts per country
    (within tolerance) over enough samples.
    """
    import random

    from bonzai_genai.training.data_module import country_balance_filter

    def _stream():
        items = (["SE"] * 30) + (["SR"] * 10) + (["SG"] * 5)
        random.Random(42).shuffle(items)
        # Cycle the stream so we can pull many samples.
        for _ in range(200):
            for c in items:
                yield {"country": c, "payload": c}

    weights = {"SE": 1 / 30, "SR": 1 / 10, "SG": 1 / 5}
    seen = {"SE": 0, "SR": 0, "SG": 0}
    for i, item in enumerate(country_balance_filter(_stream(), weights, seed=0)):
        seen[item["country"]] += 1
        if i >= 3000:
            break
    mean = sum(seen.values()) / 3
    for c, n in seen.items():
        assert abs(n - mean) / mean < 0.25, f"{c}: {n} vs mean {mean}"


def test_country_balance_filter_passes_through_when_weights_empty():
    """No weights -> no filtering."""
    from bonzai_genai.training.data_module import country_balance_filter
    src = [{"country": "X", "id": i} for i in range(50)]
    out = list(country_balance_filter(iter(src), {}))
    assert len(out) == 50


def test_country_balance_filter_rank_seeds_produce_independent_streams():
    """Different seeds (different ranks) must produce independent samples."""
    import random

    from bonzai_genai.training.data_module import country_balance_filter

    def _stream(n):
        rng = random.Random(7)
        items = (["SE"] * 30) + (["SR"] * 10) + (["SG"] * 5)
        for _ in range(n):
            for c in items:
                yield {"country": c, "id": rng.random()}

    weights = {"SE": 1 / 30, "SR": 1 / 10, "SG": 1 / 5}
    s0 = [it["id"] for it in country_balance_filter(_stream(50), weights, seed=0)][:200]
    s1 = [it["id"] for it in country_balance_filter(_stream(50), weights, seed=1)][:200]
    # The two streams should overlap heavily in id values (same source) but
    # differ in keep/reject decisions, so the kept ID sequences should
    # differ in at least 5 of the first 50 positions.
    diffs = sum(1 for a, b in zip(s0[:50], s1[:50], strict=False) if a != b)
    assert diffs > 5, f"only {diffs} differences — sampler not rank-aware"
