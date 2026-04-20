"""Small hybrid graph encoder for city-graph reconstruction."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _to_dense_by_graph(h: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    counts = torch.bincount(batch, minlength=num_graphs)
    max_nodes = int(counts.max().item()) if counts.numel() > 0 else 0
    dense = h.new_zeros((num_graphs, max_nodes, h.size(-1)))
    mask = torch.zeros((num_graphs, max_nodes), dtype=torch.bool, device=h.device)
    for graph_idx in range(num_graphs):
        node_idx = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
        if node_idx.numel() == 0:
            continue
        dense[graph_idx, : node_idx.numel()] = h[node_idx]
        mask[graph_idx, : node_idx.numel()] = True
    return dense, mask


def _from_dense_by_graph(dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pieces = []
    for graph_idx in range(dense.size(0)):
        n = int(mask[graph_idx].sum().item())
        if n > 0:
            pieces.append(dense[graph_idx, :n])
    if not pieces:
        return dense.new_zeros((0, dense.size(-1)))
    return torch.cat(pieces, dim=0)


class HybridGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_type_count: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_embed = nn.Embedding(edge_type_count, hidden_dim)
        self.local_out = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() > 0:
            src, dst = edge_index
            messages = self.msg_proj(h[src]) + self.rel_embed(edge_types)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, messages)
            deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).unsqueeze(-1).to(h.dtype)
            agg = agg / deg
            h = self.norm1(h + self.dropout(self.local_out(agg)))

        dense, mask = _to_dense_by_graph(h, batch)
        if dense.numel() > 0:
            attn_out, _ = self.attn(dense, dense, dense, key_padding_mask=~mask)
            attn_flat = _from_dense_by_graph(attn_out, mask)
            h = self.norm2(h + self.dropout(attn_flat))

        h = self.norm3(h + self.dropout(self.ff(h)))
        return h


class CityGraphEncoder(nn.Module):
    def __init__(
        self,
        node_type_count: int,
        field_vocab_sizes: Dict[str, int],
        edge_type_count: int,
        continuous_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_type_embed = nn.Embedding(node_type_count, hidden_dim)
        self.cont_proj = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.field_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(vocab_size + 1, hidden_dim)
                for field, vocab_size in field_vocab_sizes.items()
            }
        )
        self.layers = nn.ModuleList(
            [
                HybridGraphLayer(
                    hidden_dim=hidden_dim,
                    edge_type_count=edge_type_count,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_types: torch.Tensor,
        continuous: torch.Tensor,
        categorical: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_type_embed(node_types) + self.cont_proj(continuous)
        for field, values in categorical.items():
            if field not in self.field_embeddings:
                continue
            h = h + self.field_embeddings[field](values + 1)
        for layer in self.layers:
            h = layer(h=h, edge_index=edge_index, edge_types=edge_types, batch=batch)
        return self.output_norm(h)


class CityGraphAutoencoder(nn.Module):
    def __init__(
        self,
        node_type_count: int,
        field_vocab_sizes: Dict[str, int],
        edge_type_count: int,
        continuous_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = CityGraphEncoder(
            node_type_count=node_type_count,
            field_vocab_sizes=field_vocab_sizes,
            edge_type_count=edge_type_count,
            continuous_dim=continuous_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attr_heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_dim, vocab_size)
                for field, vocab_size in field_vocab_sizes.items()
                if vocab_size > 0
            }
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_type_count + 1),
        )

    def encode(
        self,
        node_types: torch.Tensor,
        continuous: torch.Tensor,
        categorical: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            node_types=node_types,
            continuous=continuous,
            categorical=categorical,
            edge_index=edge_index,
            edge_types=edge_types,
            batch=batch,
        )

    def attribute_logits(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {field: head(hidden) for field, head in self.attr_heads.items()}

    def edge_logits(self, hidden: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        src, dst = pair_index
        pair_repr = torch.cat([hidden[src], hidden[dst]], dim=-1)
        return self.edge_head(pair_repr)
