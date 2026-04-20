"""Evaluate a trained city-graph autoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import CityGraphChunkDataset, collate_city_graphs
    from .train import build_model, run_epoch
except ImportError:  # pragma: no cover
    from city_graph_modal.dataset import CityGraphChunkDataset, collate_city_graphs
    from city_graph_modal.train import build_model, run_epoch

logger = logging.getLogger("evaluate_city_graph")


def evaluate_checkpoint(
    dataset_root: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload["metadata"]
    config = payload["config"]

    dataset = CityGraphChunkDataset(dataset_root, split)
    if len(dataset) == 0:
        raise ValueError(f"split '{split}' is empty under {dataset_root}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_city_graphs,
    )

    model = build_model(
        metadata=metadata,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])

    metrics = run_epoch(
        model=model,
        loader=loader,
        device=device,
        optimizer=None,
        attribute_mask_prob=config["attribute_mask_prob"],
        edge_mask_prob=config["edge_mask_prob"],
        max_masked_edges=config["max_masked_edges"],
        edge_type_names=metadata["edge_types"],
        deterministic_eval=True,
    )
    result = {
        "checkpoint_path": checkpoint_path,
        "split": split,
        "device": str(device),
        "metrics": metrics,
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    logger.info("evaluation result: %s", result)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    evaluate_checkpoint(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
