"""Path-1 sanity check: load Experiment 0 checkpoints + actually sample tiles.

Runs locally on Mac (CPU or MPS) — no Leonardo time burned.

Produces ``bonzai_genai/results/samples/`` with three groups of PNGs:
  - vae_recon_*.png       : run a few val tiles through the trained VAE,
                             show input vs reconstruction (does the VAE compress correctly?)
  - dit_sampled_*.png     : sample latents from the trained DiT, decode through the
                             frozen VAE, render the 9-channel raster
                             (does the Sketcher produce anything coherent?)
  - inker_decoded_*.png   : feed a GT raster to the trained Inker via cross-attention,
                             greedy-sample tokens, decode the token stream back to
                             GeoJSON, and re-rasterise. Shows whether the Inker
                             produces structurally-valid output yet.

These are the visual "go" signal that the smoke harness skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.data.rasteriser import rasterise  # noqa: E402
from bonzai_genai.models.configs import (  # noqa: E402
    DiTConfig,
    InkerConfig,
    RasterEncoderConfig,
    TinyPreset,
    VAEConfig,
)
from bonzai_genai.synth.procedural import generate_synthetic_tile  # noqa: E402
from bonzai_genai.training.lit_stage_a import LitStageA  # noqa: E402
from bonzai_genai.training.lit_stage_b import LitStageB  # noqa: E402
from bonzai_genai.training.lit_vae import LitVAE  # noqa: E402
from bonzai_genai.training.samplers import dpmpp_sample, greedy_inker_sample  # noqa: E402
from bonzai_genai.vocab.attributes import load_default_vocab  # noqa: E402
from bonzai_genai.vocab.tokeniser import Tokeniser  # noqa: E402
from bonzai_genai.vocab.tokens import SpecialToken  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints" / "exp0"
OUT = REPO / "results" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = (
    "0: motorway", "1: trunk", "2: primary", "3: residential",
    "4: all-roads", "5: building density", "6: water",
    "7: green", "8: urban",
)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def _render_9ch(raster: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
    for ch in range(9):
        ax = axes[ch // 3, ch % 3]
        if ch == 5:
            ax.imshow(raster[ch], cmap="viridis", vmin=0, vmax=1)
        else:
            ax.imshow(raster[ch], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_compare(
    raster_a: np.ndarray, raster_b: np.ndarray, title: str, path: Path,
) -> None:
    """Side-by-side input vs output, summed roads + density per tile."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    for col, (rast, label) in enumerate(
        [(raster_a, "input"), (raster_b, "output")]
    ):
        all_roads = rast[4]
        density = rast[5]
        masks = rast[6:].sum(axis=0).clip(0, 1)
        axes[0, col].imshow(all_roads, cmap="gray_r", vmin=0, vmax=1)
        axes[0, col].set_title(f"{label} — all-roads", fontsize=10)
        axes[1, col].imshow(density, cmap="viridis", vmin=0, vmax=1)
        axes[1, col].set_title(f"{label} — building density", fontsize=10)
        for r in (0, 1):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
        # Third column: water/green/urban summed
        axes[0, 2].imshow(raster_b[6], cmap="Blues", vmin=0, vmax=1)
        axes[0, 2].set_title("output — water (ch 6)", fontsize=10)
        axes[1, 2].imshow(raster_b[8], cmap="Greys", vmin=0, vmax=1)
        axes[1, 2].set_title("output — urban (ch 8)", fontsize=10)
    for r in (0, 1):
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    device = _pick_device()
    print(f"Using device: {device}")

    # ---------- Load checkpoints ----------
    print("Loading VAE...", flush=True)
    vae_lit = LitVAE.load_from_checkpoint(
        str(CKPT / "vae.ckpt"),
        vae_config=VAEConfig.from_preset(TinyPreset),
        map_location=device,
    )
    vae = vae_lit.vae.eval().to(device)

    print("Loading Stage A (DiT)...", flush=True)
    stage_a_lit = LitStageA.load_from_checkpoint(
        str(CKPT / "stage_a.ckpt"),
        dit_config=DiTConfig.from_preset(TinyPreset),
        vae_config=VAEConfig.from_preset(TinyPreset),
        map_location=device,
    )
    dit = stage_a_lit.dit.eval().to(device)

    print("Loading Stage B (Inker + raster encoder)...", flush=True)
    stage_b_lit = LitStageB.load_from_checkpoint(
        str(CKPT / "stage_b.ckpt"),
        inker_config=InkerConfig.from_preset(TinyPreset),
        raster_encoder_config=RasterEncoderConfig.from_preset(TinyPreset),
        map_location=device,
    )
    inker = stage_b_lit.inker.eval().to(device)
    raster_encoder = stage_b_lit.encoder.eval().to(device)

    # ---------- VAE reconstruction sanity ----------
    print("\n[1/3] VAE reconstruction on 2 GT tiles...", flush=True)
    with torch.no_grad():
        for i, (seed, density) in enumerate([(0, "dense"), (7, "sparse")]):
            geom = generate_synthetic_tile(seed=seed, density=density)
            gt = torch.from_numpy(rasterise(geom)).unsqueeze(0).to(device)
            out = vae(gt)
            recon_logits = out["recon"]
            # Apply sigmoid for binary channels, clamp continuous channel
            recon = torch.sigmoid(recon_logits)
            # Continuous channel 5 stays as logit -> rough scale
            recon_np = recon.squeeze(0).cpu().numpy()
            gt_np = gt.squeeze(0).cpu().numpy()
            _render_compare(
                gt_np, recon_np,
                f"VAE recon — input vs output (seed={seed}, density='{density}')",
                OUT / f"vae_recon_{i}.png",
            )
            print(f"  saved vae_recon_{i}.png")

    # ---------- DiT sampling sanity ----------
    print("\n[2/3] DiT sampling — 4 random tiles from noise...", flush=True)
    with torch.no_grad():
        latents = dpmpp_sample(
            dit, batch_size=4, num_steps=25,
            latent_shape=(4, 64, 64), device=device,
        )
        # Decode through VAE decoder (logits -> sigmoid for binary)
        recon_logits = vae.decoder(latents)
        recon = torch.sigmoid(recon_logits)
        for i in range(4):
            sample_np = recon[i].cpu().numpy()
            _render_9ch(
                sample_np,
                f"DiT-sampled tile #{i} (decoded via VAE)",
                OUT / f"dit_sampled_{i}.png",
            )
            print(f"  saved dit_sampled_{i}.png")

    # ---------- Inker sampling sanity ----------
    print("\n[3/3] Inker greedy sampling — 2 GT-raster→tokens→decode...", flush=True)
    vocab = load_default_vocab()
    tok = Tokeniser(vocab)
    bos = int(SpecialToken.BOS)
    eos = int(SpecialToken.EOS)
    with torch.no_grad():
        for i, (seed, density) in enumerate([(0, "dense"), (3, "sparse")]):
            geom = generate_synthetic_tile(seed=seed, density=density)
            gt_raster = torch.from_numpy(rasterise(geom)).unsqueeze(0).to(device)
            tokens = greedy_inker_sample(
                inker, raster_encoder, gt_raster,
                max_tokens=256, bos_id=bos, eos_id=eos, constrained=False,
            )
            seq = tokens.squeeze(0).tolist()
            # Try to decode the sampled tokens back to GeoJSON
            try:
                decoded = tok.decode(seq)
                decoded_raster = rasterise(decoded)
                title = (
                    f"Inker-decoded tile (seed={seed}, density='{density}', "
                    f"{len(seq)} tokens, decode OK: "
                    f"{len(decoded.roads)} roads / {len(decoded.buildings)} bldgs / "
                    f"{len(decoded.land)} land / {len(decoded.pois)} POIs)"
                )
                _render_compare(
                    gt_raster.squeeze(0).cpu().numpy(), decoded_raster,
                    title, OUT / f"inker_decoded_{i}.png",
                )
                print(f"  saved inker_decoded_{i}.png — decode OK ({len(seq)} tokens)")
            except Exception as e:
                # Decode failed — render the GT and a black panel + error text
                print(f"  inker decode FAILED on sample {i}: {type(e).__name__}: {e}")
                fig, axes = plt.subplots(1, 2, figsize=(9, 5))
                axes[0].imshow(gt_raster.squeeze(0)[4].cpu().numpy(), cmap="gray_r")
                axes[0].set_title(f"input — all-roads (seed={seed})", fontsize=10)
                axes[1].text(
                    0.5, 0.5,
                    f"decode FAILED\n{type(e).__name__}\n{str(e)[:120]}",
                    ha="center", va="center", fontsize=9,
                    transform=axes[1].transAxes,
                )
                axes[1].set_xticks([])
                axes[1].set_yticks([])
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(
                    f"Inker decode failed (seed={seed}, {len(seq)} tokens)",
                    fontsize=11,
                )
                fig.tight_layout()
                fig.savefig(OUT / f"inker_decoded_{i}.png", dpi=110)
                plt.close(fig)

    print(f"\nAll samples saved to {OUT}")


if __name__ == "__main__":
    main()
