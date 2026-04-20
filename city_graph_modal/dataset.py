"""Dataset utilities for chunked heterogeneous city graphs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class CityGraphChunkDataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        manifest_path = self.root / f"{split}_manifest.jsonl"
        self.entries = [
            json.loads(line)["path"]
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rel_path = self.entries[index]
        with gzip.open(self.root / rel_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)

        item: Dict[str, torch.Tensor | str] = {
            "chunk_id": payload["chunk_id"],
            "node_types": torch.tensor(payload["node_types"], dtype=torch.long),
            "continuous": torch.tensor(payload["continuous"], dtype=torch.float32),
            "interior_mask": torch.tensor(payload["interior_mask"], dtype=torch.bool),
            "edge_index": torch.tensor(payload["edge_index"], dtype=torch.long),
            "edge_types": torch.tensor(payload["edge_types"], dtype=torch.long),
        }
        categorical: Dict[str, torch.Tensor] = {}
        for field, values in payload["categorical"].items():
            categorical[field] = torch.tensor(values, dtype=torch.long)
        item["categorical"] = categorical
        return item


def collate_city_graphs(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, object]:
    offsets = []
    total_nodes = 0
    for item in batch:
        offsets.append(total_nodes)
        total_nodes += int(item["node_types"].shape[0])

    node_types = torch.cat([item["node_types"] for item in batch], dim=0)
    continuous = torch.cat([item["continuous"] for item in batch], dim=0)
    interior_mask = torch.cat([item["interior_mask"] for item in batch], dim=0)

    cat_fields = batch[0]["categorical"].keys()
    categorical = {
        field: torch.cat([item["categorical"][field] for item in batch], dim=0)
        for field in cat_fields
    }

    edge_indices = []
    edge_types = []
    for item, offset in zip(batch, offsets):
        edge_indices.append(item["edge_index"] + offset)
        edge_types.append(item["edge_types"])
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
    edge_types = torch.cat(edge_types, dim=0) if edge_types else torch.zeros((0,), dtype=torch.long)

    batch_index = []
    graph_ptr = [0]
    for graph_idx, item in enumerate(batch):
        n = int(item["node_types"].shape[0])
        batch_index.append(torch.full((n,), graph_idx, dtype=torch.long))
        graph_ptr.append(graph_ptr[-1] + n)

    return {
        "chunk_ids": [item["chunk_id"] for item in batch],
        "node_types": node_types,
        "continuous": continuous,
        "categorical": categorical,
        "interior_mask": interior_mask,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "batch": torch.cat(batch_index, dim=0),
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
        "num_graphs": len(batch),
    }
