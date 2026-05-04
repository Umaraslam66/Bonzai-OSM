"""Stage A (Sketcher) evaluation metrics.

Per global spec §8.1:
    - channel_iou: per-channel IoU on binary channels, MSE on density (continuous).
    - fid_lite: simplified FID computed in-channel (no Inception features).
    - conditioning_ablation: distance between conditional and unconditional
      sample distributions (live in Phase 1; no-op for Exp 0 unconditional).
"""
from __future__ import annotations

import numpy as np
import torch

# Channels per data/rasteriser.py: 0-4 binary roads, 5 density continuous, 6-8 binary masks
BINARY_CHANNELS = (0, 1, 2, 3, 4, 6, 7, 8)
CONTINUOUS_CHANNELS = (5,)


def channel_iou(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5,
) -> dict[int, float]:
    """Per-binary-channel IoU. Continuous channels return MSE."""
    out: dict[int, float] = {}
    pred = pred.detach()
    gt = gt.detach()
    for ch in BINARY_CHANNELS:
        p = pred[:, ch] > threshold
        g = gt[:, ch] > threshold
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        if union == 0:
            out[ch] = 1.0
        else:
            out[ch] = (inter / union).item()
    for ch in CONTINUOUS_CHANNELS:
        out[ch] = ((pred[:, ch] - gt[:, ch]) ** 2).mean().item()
    return out


def fid_lite(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Simplified FID: sum of per-channel mean+stddev distances.

    A full covariance over flattened (C*H*W=2.36M)-dim features blows memory,
    so we collapse to scalars per channel: ``(mu_r - mu_f)^2 + (sigma_r - sigma_f)^2``.
    Coarse divergence indicator only — Phase 1+ should use proper Inception-feature FID.
    """
    real_np = real.detach().float().cpu().numpy()
    fake_np = fake.detach().float().cpu().numpy()
    score = 0.0
    for c in range(real_np.shape[1]):
        rc = real_np[:, c].reshape(-1)
        fc = fake_np[:, c].reshape(-1)
        mu_diff = float(rc.mean() - fc.mean())
        sigma_diff = float(np.sqrt(rc.var()) - np.sqrt(fc.var()))
        score += mu_diff * mu_diff + sigma_diff * sigma_diff
    return float(max(score, 0.0))


def conditioning_ablation(
    cond_samples: torch.Tensor | None, uncond_samples: torch.Tensor | None,
) -> float:
    """Distance between conditional and unconditional sample distributions.

    Phase 0b: returns 0.0 since Experiment 0 is unconditional. Phase 1+
    computes a real KL via histogram approximation.
    """
    if cond_samples is None or uncond_samples is None:
        return 0.0
    return fid_lite(cond_samples, uncond_samples)
