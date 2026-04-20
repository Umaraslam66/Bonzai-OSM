"""Qualitative reconstruction review helpers for the city-graph smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .dataset import CityGraphChunkDataset, collate_city_graphs
from .train import (
    _make_torch_generator,
    _stable_batch_seed,
    build_model,
    corrupt_categorical_features,
    mask_edges,
    move_batch_to_device,
)


PRIMARY_DISPLAY_FIELD = {
    "ROAD_JUNCTION": "degree_bucket",
    "ROAD_SEGMENT": "road_class",
    "BUILDING": "building_class",
    "POI": "primary_tag_value",
    "LANDUSE": "landuse_class",
}


def _decode_vocab_value(metadata: dict, field: str, raw_value: int) -> Optional[str]:
    if raw_value < 0:
        return None
    values = metadata["categorical_vocabs"].get(field, [])
    if raw_value >= len(values):
        return None
    return values[raw_value]


def _decode_attr_map(metadata: dict, categorical: Dict[str, torch.Tensor], node_idx: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field, values in categorical.items():
        value = int(values[node_idx].item())
        decoded = _decode_vocab_value(metadata, field, value)
        if decoded is not None:
            out[field] = decoded
    return out


def _dedupe_edges(edge_index: torch.Tensor, edge_labels: torch.Tensor, edge_type_names: Sequence[str]) -> List[dict]:
    if edge_index.numel() == 0:
        return []
    seen = set()
    edges: List[dict] = []
    for idx in range(edge_index.size(1)):
        src = int(edge_index[0, idx].item())
        dst = int(edge_index[1, idx].item())
        label = int(edge_labels[idx].item())
        key = (min(src, dst), max(src, dst), label)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "src": src,
                "dst": dst,
                "relation": edge_type_names[label],
            }
        )
    return edges


def _dedupe_masked_predictions(
    pair_index: torch.Tensor,
    true_labels: torch.Tensor,
    pred_classes: torch.Tensor,
    edge_type_names: Sequence[str],
) -> List[dict]:
    if pair_index.numel() == 0:
        return []
    seen = set()
    edges: List[dict] = []
    for idx in range(pair_index.size(1)):
        src = int(pair_index[0, idx].item())
        dst = int(pair_index[1, idx].item())
        true_label = int(true_labels[idx].item())
        pred_class = int(pred_classes[idx].item())
        key = (min(src, dst), max(src, dst), true_label)
        if key in seen:
            continue
        seen.add(key)
        predicted_relation = edge_type_names[pred_class - 1] if pred_class > 0 else None
        edges.append(
            {
                "src": src,
                "dst": dst,
                "true_relation": edge_type_names[true_label - 1],
                "predicted_relation": predicted_relation,
                "recovered": pred_class == true_label,
            }
        )
    return edges


def _choose_indices(dataset: CityGraphChunkDataset, sample_count: int, min_nodes: int, max_nodes: int) -> List[int]:
    chosen: List[int] = []
    stride = max(len(dataset) // max(sample_count * 12, 1), 1)
    for idx in range(0, len(dataset), stride):
        item = dataset[idx]
        node_count = int(item["node_types"].shape[0])
        if node_count < min_nodes or node_count > max_nodes:
            continue
        distinct_types = int(item["node_types"].unique().numel())
        if distinct_types < 3:
            continue
        chosen.append(idx)
        if len(chosen) >= sample_count:
            return chosen

    if len(chosen) >= sample_count:
        return chosen[:sample_count]

    fallback = sorted({0, len(dataset) // 3, (2 * len(dataset)) // 3, max(len(dataset) - 1, 0)})
    for idx in fallback:
        if idx < len(dataset) and idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= sample_count:
            break
    return chosen[:sample_count]


def build_review_payload(
    dataset_root: str,
    checkpoint_path: str,
    split: str = "test",
    sample_count: int = 4,
    min_nodes: int = 48,
    max_nodes: int = 180,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload["metadata"]
    config = payload["config"]

    dataset = CityGraphChunkDataset(dataset_root, split)
    if len(dataset) == 0:
        raise ValueError(f"split '{split}' is empty under {dataset_root}")

    model = build_model(
        metadata=metadata,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    selected = _choose_indices(dataset, sample_count=sample_count, min_nodes=min_nodes, max_nodes=max_nodes)
    samples: List[dict] = []

    with torch.no_grad():
        for sample_idx, dataset_idx in enumerate(selected):
            item = dataset[dataset_idx]
            batch = collate_city_graphs([item])
            seed = _stable_batch_seed(batch["chunk_ids"], salt="review")
            torch_rng = _make_torch_generator(device, seed)

            batch = move_batch_to_device(batch, device)
            categorical = batch["categorical"]
            interior_mask = batch["interior_mask"]
            edge_index = batch["edge_index"]
            edge_types = batch["edge_types"]

            corrupted_categorical, field_masks = corrupt_categorical_features(
                categorical=categorical,
                interior_mask=interior_mask,
                mask_prob=config["attribute_mask_prob"],
                torch_rng=torch_rng,
            )
            kept_edge_index, kept_edge_types, pos_pairs, pos_labels = mask_edges(
                edge_index=edge_index,
                edge_types=edge_types,
                interior_mask=interior_mask,
                mask_prob=config["edge_mask_prob"],
                max_masked_edges=config["max_masked_edges"],
                torch_rng=torch_rng,
            )

            hidden = model.encode(
                node_types=batch["node_types"],
                continuous=batch["continuous"],
                categorical=corrupted_categorical,
                edge_index=kept_edge_index,
                edge_types=kept_edge_types,
                batch=batch["batch"],
            )
            attr_logits = model.attribute_logits(hidden)

            if pos_pairs.numel() > 0:
                edge_logits = model.edge_logits(hidden, pos_pairs)
                edge_pred = edge_logits.argmax(dim=-1)
            else:
                edge_pred = kept_edge_types.new_zeros((0,), dtype=torch.long)

            nodes = []
            node_types_cpu = batch["node_types"].detach().cpu()
            continuous_cpu = batch["continuous"].detach().cpu()
            categorical_cpu = {field: value.detach().cpu() for field, value in categorical.items()}
            field_masks_cpu = {field: value.detach().cpu() for field, value in field_masks.items()}

            for node_idx in range(node_types_cpu.size(0)):
                node_type_name = metadata["node_types"][int(node_types_cpu[node_idx].item())]
                display_field = PRIMARY_DISPLAY_FIELD.get(node_type_name)
                true_attrs = _decode_attr_map(metadata, categorical_cpu, node_idx)
                predicted_attrs: Dict[str, str] = {}
                masked_fields: List[str] = []
                for field, mask in field_masks_cpu.items():
                    if bool(mask[node_idx].item()):
                        masked_fields.append(field)
                        logits = attr_logits.get(field)
                        if logits is not None:
                            pred_idx = int(logits[node_idx].argmax(dim=-1).item())
                            decoded = _decode_vocab_value(metadata, field, pred_idx)
                            if decoded is not None:
                                predicted_attrs[field] = decoded

                nodes.append(
                    {
                        "id": node_idx,
                        "type": node_type_name,
                        "x": float(continuous_cpu[node_idx, 0].item()),
                        "y": float(continuous_cpu[node_idx, 1].item()),
                        "size_log1p": float(continuous_cpu[node_idx, 2].item()),
                        "degree_norm": float(continuous_cpu[node_idx, 3].item()),
                        "display_field": display_field,
                        "display_true": true_attrs.get(display_field) if display_field else None,
                        "display_observed": None if display_field in masked_fields else true_attrs.get(display_field),
                        "display_predicted": predicted_attrs.get(display_field) if display_field else None,
                        "masked_fields": masked_fields,
                        "true_attrs": true_attrs,
                        "predicted_attrs": predicted_attrs,
                    }
                )

            masked_edge_records = _dedupe_masked_predictions(
                pair_index=pos_pairs.detach().cpu(),
                true_labels=pos_labels.detach().cpu(),
                pred_classes=edge_pred.detach().cpu(),
                edge_type_names=metadata["edge_types"],
            )
            masked_node_count = sum(1 for node in nodes if node["masked_fields"])
            recovered_edge_count = sum(1 for edge in masked_edge_records if edge["recovered"])

            samples.append(
                {
                    "sample_index": sample_idx,
                    "dataset_index": dataset_idx,
                    "chunk_id": item["chunk_id"],
                    "node_count": len(nodes),
                    "original_edge_count": int(edge_types.size(0)),
                    "observed_edge_count": int(kept_edge_types.size(0)),
                    "masked_edge_count": int(pos_labels.size(0)),
                    "masked_node_count": masked_node_count,
                    "recovered_edge_count": recovered_edge_count,
                    "nodes": nodes,
                    "original_edges": _dedupe_edges(edge_index.detach().cpu(), edge_types.detach().cpu(), metadata["edge_types"]),
                    "observed_edges": _dedupe_edges(
                        kept_edge_index.detach().cpu(),
                        kept_edge_types.detach().cpu(),
                        metadata["edge_types"],
                    ),
                    "masked_edges": masked_edge_records,
                }
            )

    return {
        "checkpoint_path": checkpoint_path,
        "dataset_root": dataset_root,
        "split": split,
        "device": str(device),
        "sample_count": len(samples),
        "samples": samples,
    }
