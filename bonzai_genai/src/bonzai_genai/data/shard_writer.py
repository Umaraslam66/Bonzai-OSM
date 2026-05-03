"""WebDataset-format shard writer and reader.

Each shard is a tar archive of records like:
    000000.raster.npy
    000000.tokens.json
    000000.metadata.json
    000001.raster.npy
    ...

A `manifest.json` is written alongside the shards summarising counts.
"""
from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

from bonzai_genai.data.tile_bundle import TileBundle


class ShardWriter:
    """Streams TileBundles to size-bounded tar shards."""

    def __init__(self, output_dir: Path, shard_size: int = 1000):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._shard_index = 0
        self._record_index = 0
        self._current_tar: tarfile.TarFile | None = None
        self._counts: list[int] = []
        self._records_in_current = 0

    def _open_new_shard(self) -> None:
        path = self._dir / f"shard-{self._shard_index:06d}.tar"
        self._current_tar = tarfile.open(path, "w")
        self._records_in_current = 0

    def _maybe_roll(self) -> None:
        if self._current_tar is None:
            self._open_new_shard()
        elif self._records_in_current >= self._shard_size:
            self._current_tar.close()
            self._counts.append(self._records_in_current)
            self._shard_index += 1
            self._open_new_shard()

    def write(self, bundle: TileBundle) -> None:
        self._maybe_roll()
        assert self._current_tar is not None
        files = bundle.to_dict()
        prefix = f"{self._record_index:06d}"
        for fname, data in files.items():
            info = tarfile.TarInfo(name=f"{prefix}.{fname}")
            info.size = len(data)
            self._current_tar.addfile(info, io.BytesIO(data))
        self._record_index += 1
        self._records_in_current += 1

    def close(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
            self._counts.append(self._records_in_current)
            self._current_tar = None
        manifest = {
            "num_shards": len(self._counts),
            "num_records": sum(self._counts),
            "records_per_shard": self._counts,
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_shard_bundles(shard_dir: Path) -> Iterator[TileBundle]:
    """Yield every TileBundle in every shard under shard_dir."""
    shard_dir = Path(shard_dir)
    for shard in sorted(shard_dir.glob("shard-*.tar")):
        with tarfile.open(shard, "r") as tf:
            members = sorted(tf.getmembers(), key=lambda m: m.name)
            current: dict[str, bytes] = {}
            current_prefix: str | None = None
            for member in members:
                prefix, suffix = member.name.split(".", 1)
                if current_prefix is None:
                    current_prefix = prefix
                if prefix != current_prefix:
                    yield TileBundle.from_dict(current)
                    current = {}
                    current_prefix = prefix
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                current[suffix] = fobj.read()
            if current:
                yield TileBundle.from_dict(current)
