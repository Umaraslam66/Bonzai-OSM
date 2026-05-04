"""Tests for the §8.2 baselines."""
import torch


def test_random_crop_baseline_returns_random_indices():
    from bonzai_genai.eval.baselines import random_crop_baseline
    idx = random_crop_baseline(pool_size=50, n_samples=10, seed=0)
    assert len(idx) == 10
    assert all(0 <= i < 50 for i in idx)


def test_perfect_baseline_returns_input():
    from bonzai_genai.eval.baselines import perfect_baseline
    pool = torch.randn(20, 9, 64, 64)
    out = perfect_baseline(pool, indices=[0, 5, 10])
    assert torch.equal(out[0], pool[0])
    assert torch.equal(out[1], pool[5])


def test_nearest_neighbor_baseline_returns_closest():
    from bonzai_genai.eval.baselines import nearest_neighbor_baseline
    pool = torch.zeros(10, 9, 4, 4)
    pool[0] = 1.0
    pool[1] = 2.0
    query = torch.full((1, 9, 4, 4), 1.1)
    out = nearest_neighbor_baseline(pool, query)
    assert torch.equal(out[0], pool[0])
