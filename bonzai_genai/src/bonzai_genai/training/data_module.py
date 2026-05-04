"""LightningDataModule that wraps WebDataset shards into PyTorch DataLoaders.

Two modes:
    return_tokens=False — yields a (B, 9, 512, 512) float32 tensor (raster only).
    return_tokens=True  — yields {"raster": ..., "tokens": ..., "token_lens": ...}.

Token sequences are right-padded to ``max_token_len``; ``token_lens``
records the real length before padding.
"""
from __future__ import annotations

import io
import json

import lightning as L  # noqa: N812
import numpy as np
import torch
from torch.utils.data import DataLoader


def _decode_bundle(sample: dict) -> dict:
    """Decode a WebDataset sample into native Python objects.

    The shard format from ``ShardWriter`` puts ``raster.npy`` (np.save'd
    float32 array), ``tokens.json`` (JSON list of ints), and
    ``metadata.json`` (TileMetadata as JSON) into each record. We surface
    ``country`` so the country-balanced sampler has something to weight by.
    """
    raster = np.load(io.BytesIO(sample["raster.npy"]))
    tokens = json.loads(sample["tokens.json"].decode("utf-8"))
    if "metadata.json" in sample:
        meta = json.loads(sample["metadata.json"].decode("utf-8"))
        country = meta.get("country", "unknown")
    else:
        country = "unknown"
    return {
        "raster": raster.astype(np.float32),
        "tokens": np.asarray(tokens, dtype=np.int64),
        "country": country,
    }


def _collate_raster_only(items: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.stack([it["raster"] for it in items]))


def _collate_with_tokens(items: list[dict], max_len: int) -> dict[str, torch.Tensor]:
    rasters = torch.from_numpy(np.stack([it["raster"] for it in items]))
    bs = len(items)
    tokens_pad = torch.zeros(bs, max_len, dtype=torch.long)
    lens = torch.zeros(bs, dtype=torch.long)
    for i, it in enumerate(items):
        n = min(len(it["tokens"]), max_len)
        tokens_pad[i, :n] = torch.from_numpy(it["tokens"][:n])
        lens[i] = n
    return {"raster": rasters, "tokens": tokens_pad, "token_lens": lens}


class TileDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_url: str,
        val_url: str,
        batch_size: int = 8,
        return_tokens: bool = False,
        max_token_len: int = 4096,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _build(self, url: str):
        import webdataset as wds
        ds = (
            wds.WebDataset(url, shardshuffle=False, empty_check=False)
            .map(_decode_bundle)
        )
        return ds

    def setup(self, stage: str) -> None:
        self.train_ds = self._build(self.hparams.train_url)
        self.val_ds = self._build(self.hparams.val_url)

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        if self.hparams.return_tokens:
            max_len = self.hparams.max_token_len

            def collate(items):
                return _collate_with_tokens(items, max_len)
        else:
            collate = _collate_raster_only
        return DataLoader(
            ds.batched(self.hparams.batch_size, collation_fn=collate, partial=False),
            batch_size=None,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)
