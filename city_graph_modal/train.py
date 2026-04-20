"""Train a small city-graph autoencoder on chunked Luxembourg data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import CityGraphChunkDataset, collate_city_graphs
    from .model import CityGraphAutoencoder
except ImportError:  # pragma: no cover
    from city_graph_modal.dataset import CityGraphChunkDataset, collate_city_graphs
    from city_graph_modal.model import CityGraphAutoencoder

logger = logging.getLogger("train_city_graph")

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(metadata: dict, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> CityGraphAutoencoder:
    field_vocab_sizes = {
        field: len(values)
        for field, values in metadata["categorical_vocabs"].items()
    }
    return CityGraphAutoencoder(
        node_type_count=len(metadata["node_types"]),
        field_vocab_sizes=field_vocab_sizes,
        edge_type_count=len(metadata["edge_types"]),
        continuous_dim=len(metadata["continuous_fields"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


def _edge_metric_prefix(edge_type_name: str) -> str:
    return f"edge_{edge_type_name.lower()}"


def _stable_batch_seed(chunk_ids: Sequence[str], salt: str = "eval") -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(salt.encode("utf-8"))
    for chunk_id in chunk_ids:
        digest.update(b"\0")
        digest.update(chunk_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "big")


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def corrupt_categorical_features(
    categorical: Dict[str, torch.Tensor],
    interior_mask: torch.Tensor,
    mask_prob: float,
    torch_rng: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    corrupted = {}
    masks = {}
    for field, values in categorical.items():
        valid = (values >= 0) & interior_mask
        if valid.any():
            sample = torch.rand(values.shape, device=values.device, generator=torch_rng)
            mask = valid & (sample < mask_prob)
        else:
            mask = torch.zeros_like(values, dtype=torch.bool)
        masked_values = values.clone()
        masked_values[mask] = -1
        corrupted[field] = masked_values
        masks[field] = mask
    return corrupted, masks


def mask_edges(
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    interior_mask: torch.Tensor,
    mask_prob: float,
    max_masked_edges: int,
    torch_rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        return edge_index, edge_types, edge_index.new_zeros((2, 0)), edge_types.new_zeros((0,))

    src, dst = edge_index
    candidate = interior_mask[src] & interior_mask[dst]
    if not candidate.any():
        return edge_index, edge_types, edge_index.new_zeros((2, 0)), edge_types.new_zeros((0,))

    mask = candidate & (torch.rand(edge_types.size(0), device=edge_types.device, generator=torch_rng) < mask_prob)
    candidate_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if candidate_idx.numel() > max_masked_edges:
        perm = torch.randperm(candidate_idx.numel(), device=edge_types.device, generator=torch_rng)[:max_masked_edges]
        keep_mask = torch.ones(edge_types.size(0), dtype=torch.bool, device=edge_types.device)
        chosen = candidate_idx[perm]
        keep_mask[chosen] = False
    else:
        keep_mask = ~mask
        chosen = candidate_idx

    kept_edge_index = edge_index[:, keep_mask]
    kept_edge_types = edge_types[keep_mask]
    pos_pairs = edge_index[:, chosen]
    pos_labels = edge_types[chosen] + 1
    return kept_edge_index, kept_edge_types, pos_pairs, pos_labels


def sample_negative_pairs(
    batch_index: torch.Tensor,
    interior_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_samples: int,
    py_rng: Optional[random.Random] = None,
) -> torch.Tensor:
    if num_samples <= 0:
        return edge_index.new_zeros((2, 0))

    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    negatives = []
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if num_graphs == 0:
        return edge_index.new_zeros((2, 0))

    chooser = py_rng or random
    graph_nodes = [
        torch.nonzero((batch_index == graph_idx) & interior_mask, as_tuple=False).flatten().tolist()
        for graph_idx in range(num_graphs)
    ]
    attempts = 0
    while len(negatives) < num_samples and attempts < num_samples * 50:
        attempts += 1
        graph_idx = chooser.randrange(num_graphs)
        nodes = graph_nodes[graph_idx]
        if len(nodes) < 2:
            continue
        src = chooser.choice(nodes)
        dst = chooser.choice(nodes)
        if src == dst or (src, dst) in existing:
            continue
        negatives.append((src, dst))
        existing.add((src, dst))
    if not negatives:
        return edge_index.new_zeros((2, 0))
    return torch.tensor(negatives, dtype=torch.long, device=edge_index.device).T


def compute_losses(
    model: CityGraphAutoencoder,
    batch: Dict[str, object],
    attribute_mask_prob: float,
    edge_mask_prob: float,
    max_masked_edges: int,
    edge_type_names: Optional[Sequence[str]] = None,
    torch_rng: Optional[torch.Generator] = None,
    py_rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
    node_types = batch["node_types"]
    continuous = batch["continuous"]
    categorical = batch["categorical"]
    interior_mask = batch["interior_mask"]
    edge_index = batch["edge_index"]
    edge_types = batch["edge_types"]
    batch_index = batch["batch"]

    corrupted_categorical, field_masks = corrupt_categorical_features(
        categorical=categorical,
        interior_mask=interior_mask,
        mask_prob=attribute_mask_prob,
        torch_rng=torch_rng,
    )
    kept_edge_index, kept_edge_types, pos_pairs, pos_labels = mask_edges(
        edge_index=edge_index,
        edge_types=edge_types,
        interior_mask=interior_mask,
        mask_prob=edge_mask_prob,
        max_masked_edges=max_masked_edges,
        torch_rng=torch_rng,
    )

    hidden = model.encode(
        node_types=node_types,
        continuous=continuous,
        categorical=corrupted_categorical,
        edge_index=kept_edge_index,
        edge_types=kept_edge_types,
        batch=batch_index,
    )

    attr_logits = model.attribute_logits(hidden)
    attr_losses = []
    metrics: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for field, logits in attr_logits.items():
        mask = field_masks[field]
        target = categorical[field]
        if mask.any():
            loss = F.cross_entropy(logits[mask], target[mask])
            attr_losses.append(loss)
            pred = logits[mask].argmax(dim=-1)
            correct = int((pred == target[mask]).sum().item())
            total = int(mask.sum().item())
            counts[f"{field}_correct"] = correct
            counts[f"{field}_total"] = total

    if attr_losses:
        attr_loss = torch.stack(attr_losses).mean()
    else:
        attr_loss = hidden.new_tensor(0.0)

    neg_pairs = sample_negative_pairs(
        batch_index=batch_index,
        interior_mask=interior_mask,
        edge_index=edge_index,
        num_samples=int(pos_labels.numel()),
        py_rng=py_rng,
    )
    edge_pair_parts = []
    edge_labels_parts = []
    if pos_labels.numel() > 0:
        edge_pair_parts.append(pos_pairs)
        edge_labels_parts.append(pos_labels)
    if neg_pairs.numel() > 0:
        edge_pair_parts.append(neg_pairs)
        edge_labels_parts.append(torch.zeros(neg_pairs.size(1), dtype=torch.long, device=neg_pairs.device))

    if edge_pair_parts:
        pair_index = torch.cat(edge_pair_parts, dim=1)
        edge_labels = torch.cat(edge_labels_parts, dim=0)
        edge_logits = model.edge_logits(hidden, pair_index)
        edge_loss = F.cross_entropy(edge_logits, edge_labels)
        edge_pred = edge_logits.argmax(dim=-1)
        counts["edge_correct"] = int((edge_pred == edge_labels).sum().item())
        counts["edge_total"] = int(edge_labels.numel())
        if edge_type_names is not None:
            for edge_label, edge_name in enumerate(edge_type_names, start=1):
                prefix = _edge_metric_prefix(edge_name)
                pred_mask = edge_pred == edge_label
                label_mask = edge_labels == edge_label
                tp = int((pred_mask & label_mask).sum().item())
                pred_total = int(pred_mask.sum().item())
                support = int(label_mask.sum().item())
                counts[f"{prefix}_tp"] = tp
                counts[f"{prefix}_pred_total"] = pred_total
                counts[f"{prefix}_support"] = support
    else:
        edge_loss = hidden.new_tensor(0.0)
        counts["edge_correct"] = 0
        counts["edge_total"] = 0
        if edge_type_names is not None:
            for edge_name in edge_type_names:
                prefix = _edge_metric_prefix(edge_name)
                counts[f"{prefix}_tp"] = 0
                counts[f"{prefix}_pred_total"] = 0
                counts[f"{prefix}_support"] = 0

    total_loss = attr_loss + edge_loss
    metrics["attr_loss"] = float(attr_loss.item())
    metrics["edge_loss"] = float(edge_loss.item())
    metrics["loss"] = float(total_loss.item())
    return total_loss, metrics, counts


def run_epoch(
    model: CityGraphAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    attribute_mask_prob: float,
    edge_mask_prob: float,
    max_masked_edges: int,
    edge_type_names: Optional[Sequence[str]] = None,
    deterministic_eval: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sums: Dict[str, float] = defaultdict(float)
    loss_weights: Dict[str, float] = defaultdict(float)
    count_sums: Dict[str, int] = defaultdict(int)

    for batch in loader:
        torch_rng = None
        py_rng = None
        if deterministic_eval and not is_train:
            seed = _stable_batch_seed(batch["chunk_ids"])
            torch_rng = _make_torch_generator(device, seed)
            py_rng = random.Random(seed)
        batch = move_batch_to_device(batch, device)
        if is_train:
            loss, metrics, counts = compute_losses(
                model=model,
                batch=batch,
                attribute_mask_prob=attribute_mask_prob,
                edge_mask_prob=edge_mask_prob,
                max_masked_edges=max_masked_edges,
                edge_type_names=edge_type_names,
                torch_rng=torch_rng,
                py_rng=py_rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, metrics, counts = compute_losses(
                    model=model,
                    batch=batch,
                    attribute_mask_prob=attribute_mask_prob,
                    edge_mask_prob=edge_mask_prob,
                    max_masked_edges=max_masked_edges,
                    edge_type_names=edge_type_names,
                    torch_rng=torch_rng,
                    py_rng=py_rng,
                )

        for key, value in metrics.items():
            loss_sums[key] += value
            loss_weights[key] += 1.0
        for key, value in counts.items():
            count_sums[key] += value

    results = {
        key: (loss_sums[key] / max(loss_weights[key], 1.0))
        for key in loss_sums
    }

    for key, total in count_sums.items():
        if not key.endswith("_total") or total <= 0:
            continue
        prefix = key[: -len("_total")]
        correct = count_sums.get(f"{prefix}_correct")
        if correct is not None:
            results[f"{prefix}_acc"] = correct / total

    if edge_type_names is not None:
        macro_precision = []
        macro_recall = []
        macro_f1 = []
        for edge_name in edge_type_names:
            prefix = _edge_metric_prefix(edge_name)
            tp = count_sums.get(f"{prefix}_tp", 0)
            pred_total = count_sums.get(f"{prefix}_pred_total", 0)
            support = count_sums.get(f"{prefix}_support", 0)
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            results[f"{prefix}_precision"] = precision
            results[f"{prefix}_recall"] = recall
            results[f"{prefix}_f1"] = f1
            results[f"{prefix}_support"] = float(support)
            macro_precision.append(precision)
            macro_recall.append(recall)
            macro_f1.append(f1)
        if macro_precision:
            results["edge_macro_precision"] = sum(macro_precision) / len(macro_precision)
            results["edge_macro_recall"] = sum(macro_recall) / len(macro_recall)
            results["edge_macro_f1"] = sum(macro_f1) / len(macro_f1)

    return results


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out = dict(batch)
    for key in ("node_types", "continuous", "interior_mask", "edge_index", "edge_types", "batch", "graph_ptr"):
        out[key] = batch[key].to(device)
    out["categorical"] = {field: value.to(device) for field, value in batch["categorical"].items()}
    return out


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / "last.pt", ckpt_dir / "best.pt"


def save_checkpoint(
    path: Path,
    model: CityGraphAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    metadata: dict,
    config: dict,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "metadata": metadata,
            "config": config,
        },
        path,
    )


def maybe_init_wandb(config: dict):
    if wandb is None or not os.environ.get("WANDB_API_KEY"):
        return None
    return wandb.init(
        project=config["wandb_project"],
        name=config.get("wandb_run_name"),
        config=config,
    )


def train_model(
    dataset_root: str,
    output_dir: str,
    batch_size: int = 8,
    epochs: int = 12,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    hidden_dim: int = 256,
    num_layers: int = 5,
    num_heads: int = 4,
    dropout: float = 0.1,
    attribute_mask_prob: float = 0.30,
    edge_mask_prob: float = 0.15,
    max_masked_edges: int = 512,
    num_workers: int = 0,
    seed: int = 42,
    wandb_project: str = "bonzai-city-graph",
    wandb_run_name: Optional[str] = None,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_ds = CityGraphChunkDataset(dataset_root, "train")
    if len(train_ds) == 0:
        raise ValueError(f"train split is empty under {dataset_root}")

    val_split = "val"
    val_ds = CityGraphChunkDataset(dataset_root, val_split)
    if len(val_ds) == 0:
        fallback_val_ds = CityGraphChunkDataset(dataset_root, "test")
        if len(fallback_val_ds) == 0:
            raise ValueError(f"validation/test splits are empty under {dataset_root}")
        logger.warning("val split is empty; using test split for checkpoint selection")
        val_split = "test"
        val_ds = fallback_val_ds
    metadata = train_ds.metadata

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_city_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_city_graphs,
    )

    model = build_model(
        metadata=metadata,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    config = {
        "dataset_root": dataset_root,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "attribute_mask_prob": attribute_mask_prob,
        "edge_mask_prob": edge_mask_prob,
        "max_masked_edges": max_masked_edges,
        "seed": seed,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    run = maybe_init_wandb(config)

    last_ckpt, best_ckpt = checkpoint_paths(output_root)
    start_epoch = 0
    best_val_loss = float("inf")
    if last_ckpt.exists():
        payload = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        logger.info("resumed from %s at epoch %d", last_ckpt, start_epoch)

    history = []
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            attribute_mask_prob=attribute_mask_prob,
            edge_mask_prob=edge_mask_prob,
            max_masked_edges=max_masked_edges,
            edge_type_names=metadata["edge_types"],
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            attribute_mask_prob=attribute_mask_prob,
            edge_mask_prob=edge_mask_prob,
            max_masked_edges=max_masked_edges,
            edge_type_names=metadata["edge_types"],
            deterministic_eval=True,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "val_split": val_split}
        history.append(row)
        logger.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }
            )

        val_loss = float(val_metrics["loss"])
        save_checkpoint(
            path=last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=min(best_val_loss, val_loss),
            metadata=metadata,
            config=config,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metadata=metadata,
                config=config,
            )

    result = {
        "device": str(device),
        "best_val_loss": best_val_loss,
        "history": history,
        "last_checkpoint": str(last_ckpt),
        "best_checkpoint": str(best_ckpt),
    }
    with (output_root / "train_history.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    if run is not None:
        run.finish()
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attribute-mask-prob", type=float, default=0.30)
    parser.add_argument("--edge-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-masked-edges", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="bonzai-city-graph")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_model(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attribute_mask_prob=args.attribute_mask_prob,
        edge_mask_prob=args.edge_mask_prob,
        max_masked_edges=args.max_masked_edges,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
