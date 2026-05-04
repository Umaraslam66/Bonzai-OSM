"""Tests for Stage A eval metrics."""
import numpy as np
import pytest
import torch


def test_channel_iou_perfect_reconstruction_is_one():
    from bonzai_genai.eval.stage_a import channel_iou
    pred = torch.zeros(2, 9, 32, 32)
    pred[:, 0:4] = (torch.rand_like(pred[:, 0:4]) > 0.5).float()
    pred[:, 6:] = (torch.rand_like(pred[:, 6:]) > 0.5).float()
    gt = pred.clone()
    out = channel_iou(pred, gt)
    for ch, val in out.items():
        if ch == 5:  # density (continuous, MSE — perfect = 0)
            assert val == pytest.approx(0.0, abs=1e-6), f"channel {ch}"
        else:
            assert val == pytest.approx(1.0, abs=1e-6), f"channel {ch}"


def test_channel_iou_no_overlap_is_zero():
    from bonzai_genai.eval.stage_a import channel_iou
    pred = torch.zeros(1, 9, 32, 32)
    gt = torch.ones(1, 9, 32, 32)
    out = channel_iou(pred, gt, threshold=0.5)
    for ch, val in out.items():
        if ch == 5:  # density (continuous), skipped
            continue
        assert val == pytest.approx(0.0, abs=1e-6)


def test_fid_returns_nonnegative_finite_score():
    from bonzai_genai.eval.stage_a import fid_lite
    real = torch.randn(20, 9, 32, 32)
    fake = torch.randn(20, 9, 32, 32) + 0.1
    score = fid_lite(real, fake)
    assert np.isfinite(score)
    assert score >= 0
