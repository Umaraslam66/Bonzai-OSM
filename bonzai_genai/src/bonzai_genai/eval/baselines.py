"""§8.2 baselines: random crop / nearest neighbor / frequency-matched / perfect tile."""
from __future__ import annotations

import random as _random

import torch


def random_crop_baseline(pool_size: int, n_samples: int, seed: int = 0) -> list[int]:
    """Return ``n_samples`` random indices in [0, pool_size). Lower-bound baseline."""
    rng = _random.Random(seed)
    return [rng.randrange(pool_size) for _ in range(n_samples)]


def nearest_neighbor_baseline(pool: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """For each query, return the closest pool tile by L2 distance over flattened pixels."""
    pool_flat = pool.view(pool.shape[0], -1).float()
    q_flat = queries.view(queries.shape[0], -1).float()
    out = torch.zeros_like(queries)
    for i in range(q_flat.shape[0]):
        dists = ((pool_flat - q_flat[i:i + 1]) ** 2).sum(dim=-1)
        j = int(dists.argmin())
        out[i] = pool[j]
    return out


def frequency_matched_baseline(
    class_priors: dict[str, float], n_samples: int, seed: int = 0,
) -> list[str]:
    """Sample class labels from the empirical class prior."""
    rng = _random.Random(seed)
    classes = list(class_priors.keys())
    probs = list(class_priors.values())
    return rng.choices(classes, weights=probs, k=n_samples)


def perfect_baseline(pool: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Trivial upper-bound baseline: return the actual ground-truth tiles at given indices."""
    return torch.stack([pool[i] for i in indices])
