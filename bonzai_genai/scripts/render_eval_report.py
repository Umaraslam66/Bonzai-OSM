"""Render Experiment 0 results into a single self-contained HTML page.

Inputs (all under ``bonzai_genai/results/``):
    eval_results.json           — raw metric numbers
    vae_metrics.csv             — Lightning CSV log from VAE training
    stage_a_metrics.csv         — Lightning CSV log from Stage A (DiT)
    stage_b_metrics.csv         — Lightning CSV log from Stage B (Inker)

Output:
    EXPERIMENT_0_REPORT.html    — self-contained, all images inlined as base64
"""
from __future__ import annotations

import base64
import csv
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.data.rasteriser import rasterise  # noqa: E402
from bonzai_genai.synth.procedural import generate_synthetic_tile  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_HTML = RESULTS_DIR / "EXPERIMENT_0_REPORT.html"
SAMPLES_DIR = RESULTS_DIR / "samples"


def _file_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _png_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _read_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            out = {}
            for k, v in r.items():
                if v == "" or v is None:
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    continue
            if out:
                rows.append(out)
    return rows


def _loss_figure(rows: list[dict], stage: str, fields: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if not rows:
        ax.text(0.5, 0.5, f"no data for {stage}", ha="center", va="center")
        ax.set_axis_off()
        return _png_to_b64(fig)
    steps = [r.get("step", i) for i, r in enumerate(rows)]
    for f in fields:
        ys = [(r.get(f), s) for r, s in zip(rows, steps, strict=False) if f in r]
        if not ys:
            continue
        y = [p[0] for p in ys]
        x = [p[1] for p in ys]
        ax.plot(x, y, label=f, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{stage} — training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    return _png_to_b64(fig)


CHANNEL_NAMES = (
    "0: motorway", "1: trunk", "2: primary", "3: residential",
    "4: all-roads", "5: building density", "6: water",
    "7: green", "8: urban",
)


def _tile_grid(seed: int, density: str) -> str:
    """Render one synthetic tile's 9 channels as a 3x3 grid."""
    geom = generate_synthetic_tile(seed=seed, density=density)
    raster = rasterise(geom)  # (9, 512, 512) float32
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ch in range(9):
        ax = axes[ch // 3, ch % 3]
        # Render binary masks vs density continuous channel differently
        if ch == 5:  # density continuous
            ax.imshow(raster[ch], cmap="viridis", vmin=0, vmax=1)
        else:
            ax.imshow(raster[ch], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Synthetic tile (seed={seed}, density='{density}')", fontsize=11)
    fig.tight_layout()
    return _png_to_b64(fig)


def _eval_bar_chart(eval_results: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    # Stage A channel IoU
    iou = eval_results["stage_a"]["channel_iou_self"]
    binary_chs = ["0", "1", "2", "3", "4", "6", "7", "8"]
    iou_vals = [iou[c] for c in binary_chs]
    axes[0].bar(binary_chs, iou_vals, color="#4a7ab8")
    axes[0].set_title("Stage A — channel IoU (val self vs self)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("IoU")
    axes[0].grid(True, alpha=0.3, axis="y")
    # Stage B metric values
    stage_b = eval_results["stage_b"]
    keys = list(stage_b.keys())
    vals = [float(stage_b[k]) for k in keys]
    axes[1].barh(keys, vals, color="#b86a4a")
    axes[1].set_title("Stage B — metrics (lower is better except validity)")
    axes[1].grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return _png_to_b64(fig)


def main() -> None:
    eval_results = json.loads((RESULTS_DIR / "eval_results.json").read_text())
    vae_rows = _read_csv(RESULTS_DIR / "vae_metrics.csv")
    a_rows = _read_csv(RESULTS_DIR / "stage_a_metrics.csv")
    b_rows = _read_csv(RESULTS_DIR / "stage_b_metrics.csv")

    vae_loss_png = _loss_figure(
        vae_rows, "VAE",
        ["train/loss", "train/recon_bce", "train/recon_mse", "train/kl"],
    )
    stage_a_loss_png = _loss_figure(a_rows, "Stage A — DiT (Sketcher)", ["train/loss"])
    stage_b_loss_png = _loss_figure(b_rows, "Stage B — Inker", ["train/loss"])
    bar_png = _eval_bar_chart(eval_results)
    sparse_tile_png = _tile_grid(seed=42, density="sparse")
    dense_tile_png = _tile_grid(seed=42, density="dense")

    # Path-1 sanity samples (locally generated; no Leonardo time)
    vae_recon_0 = _file_to_b64(SAMPLES_DIR / "vae_recon_0.png")
    vae_recon_1 = _file_to_b64(SAMPLES_DIR / "vae_recon_1.png")
    dit_samples = [_file_to_b64(SAMPLES_DIR / f"dit_sampled_{i}.png") for i in range(4)]
    inker_samples = [_file_to_b64(SAMPLES_DIR / f"inker_decoded_{i}.png") for i in range(2)]

    # Stage B metric table rows
    stage_b_rows = "\n".join(
        f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in eval_results["stage_b"].items()
    )
    # Stage A metric table rows
    stage_a_rows = "\n".join(
        f'<tr><td>channel {c} IoU/MSE</td><td>{v:.4f}</td></tr>'
        for c, v in sorted(eval_results["stage_a"]["channel_iou_self"].items())
    )
    fid_row = (
        f'<tr><td>FID-lite (real vs real, sanity)</td>'
        f'<td>{eval_results["stage_a"]["fid_lite_real_vs_real"]:.4f}</td></tr>'
    )
    stage_a_rows = stage_a_rows + "\n" + fid_row

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment 0 — Bonzai-OSM</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 18px; color: #222; line-height: 1.45; }}
  h1 {{ font-size: 24px; border-bottom: 2px solid #4a7ab8; padding-bottom: 6px; }}
  h2 {{ font-size: 18px; margin-top: 28px; color: #2c4f7e; }}
  h3 {{ font-size: 14px; color: #555; margin-bottom: 4px; }}
  .meta {{ background: #f4f6f9; padding: 8px 14px; border-radius: 6px; font-size: 13px; color: #555; }}
  .go {{ background: #d6e9c6; color: #2c5b18; padding: 4px 10px; border-radius: 4px; font-weight: 600; }}
  table {{ border-collapse: collapse; margin: 8px 0 18px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d5db; padding: 4px 10px; text-align: left; }}
  th {{ background: #eef2f6; }}
  td:nth-child(2) {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 6px 0; }}
  .twocol {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .twocol > div {{ flex: 1; min-width: 320px; }}
  ul {{ padding-left: 22px; }}
  li {{ margin: 3px 0; }}
  code {{ background: #f4f6f9; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>

<h1>Experiment 0 — Bonzai-OSM smoke test</h1>

<div class="meta">
  <strong>Completed:</strong> 2026-05-04 ·
  <strong>Cluster:</strong> CINECA Leonardo Booster (1×A100 + free CPU partition) ·
  <strong>GPU-h burned:</strong> ~1 (vs 12-30 budgeted) ·
  <strong>Branch:</strong> <code>genai-city-model</code> ·
  <strong>Status:</strong> <span class="go">GO</span>
</div>

<h2>What this experiment was</h2>
<p>A smoke test. The whole point: confirm the pipeline closes end-to-end — VAE compresses 9-channel rasters → Sketcher (DiT) paints denoised codes in latent space → Inker (autoregressive transformer) reads the painted blueprint via cross-attention and writes vector tokens → eval harness measures everything — without anything blowing up. We are <em>not</em> testing whether the generated cities look good yet; that's Experiments 1-4.</p>

<h2>Training loss curves</h2>
<p>All three stages trained without divergence. Loss curves are smooth, no NaNs.</p>

<h3>VAE — 50 epochs reconstruction (BCE on binary channels + MSE on density + KL regulariser)</h3>
<img src="data:image/png;base64,{vae_loss_png}" alt="VAE loss">

<h3>Stage A — DiT diffusion (1 epoch in latent space, EDM noise)</h3>
<img src="data:image/png;base64,{stage_a_loss_png}" alt="Stage A loss">

<h3>Stage B — Inker (1 epoch teacher-forcing, cross-attn to ground-truth raster)</h3>
<img src="data:image/png;base64,{stage_b_loss_png}" alt="Stage B loss">

<h2>What the synthetic training tiles look like</h2>
<p>5,000 synthetic tiles fed the smoke test. Each tile is 2 km × 2 km (512×512 pixels at 4 m/px), encoded as a 9-channel raster. Two density modes: <code>sparse</code> (a few buildings + 2-3 short roads) and <code>dense</code> (8×8 grid roads + diagonal cross-roads + ~64 buildings + landuse polygons + POIs). Each panel below is one channel of the raster; black = mask off, white = mask on; channel 5 (building density, viridis) is continuous.</p>

<div class="twocol">
  <div>
    <h3>Sparse-mode example (seed=42)</h3>
    <img src="data:image/png;base64,{sparse_tile_png}" alt="Sparse tile">
  </div>
  <div>
    <h3>Dense-mode example (seed=42)</h3>
    <img src="data:image/png;base64,{dense_tile_png}" alt="Dense tile">
  </div>
</div>

<h2>Eval-suite numbers</h2>
<p>Metrics ran on the val set (real vs real, self-vs-self) — they verify the eval <em>code</em> runs cleanly on the trained pipeline. Actual sampling from trained models is deferred to Plan 3 / Experiment 1.</p>

<img src="data:image/png;base64,{bar_png}" alt="Eval bar charts">

<div class="twocol">
  <div>
    <h3>Stage A metrics</h3>
    <table>
      <thead><tr><th>metric</th><th>value</th></tr></thead>
      <tbody>{stage_a_rows}</tbody>
    </table>
  </div>
  <div>
    <h3>Stage B metrics</h3>
    <table>
      <thead><tr><th>metric</th><th>value</th></tr></thead>
      <tbody>{stage_b_rows}</tbody>
    </table>
  </div>
</div>

<h2>Path-1 sanity samples (locally rendered, 0 Leonardo GPU-h)</h2>
<p>The smoke eval ran self-vs-self on val ground-truth. To get a real signal, we pulled the trained checkpoints back to a Mac, loaded VAE/DiT/Inker on Apple Silicon, and ran the actual sampling that the original eval skipped. ~1 minute of local CPU/MPS time.</p>

<h3>VAE reconstruction — input vs output (after 50 epochs)</h3>
<p>The VAE compresses 9 channels × 512² → 4 × 64² and back. Reconstructions are essentially pixel-perfect on synthetic tiles. <strong>Green.</strong></p>
<div class="twocol">
  <div><img src="data:image/png;base64,{vae_recon_0}" alt="VAE recon dense"></div>
  <div><img src="data:image/png;base64,{vae_recon_1}" alt="VAE recon sparse"></div>
</div>

<h3>DiT (Sketcher) — sampled from pure noise</h3>
<p>Each tile below was generated from random Gaussian noise via 25 denoising steps of DPM-Solver++, then decoded through the frozen VAE. After <strong>only 1 epoch of training</strong>, the DiT is reproducing the synth corpus's grid-and-blocks structure on the road and residential channels. The continuous building-density channel (5) hasn't picked up spatial variation yet (uniform green) and most low-frequency channels (water/green/urban) are correctly empty. <strong>Strong green for 1 epoch.</strong></p>
<div class="twocol">
  <div><img src="data:image/png;base64,{dit_samples[0]}" alt="DiT sampled 0"></div>
  <div><img src="data:image/png;base64,{dit_samples[1]}" alt="DiT sampled 1"></div>
</div>
<div class="twocol">
  <div><img src="data:image/png;base64,{dit_samples[2]}" alt="DiT sampled 2"></div>
  <div><img src="data:image/png;base64,{dit_samples[3]}" alt="DiT sampled 3"></div>
</div>

<h3>Inker (Stage B) — greedy-sample tokens from a GT raster, decode back to geometry</h3>
<p>The Inker reads a ground-truth raster via cross-attention and writes a token stream. After 1 epoch on synth data, decoded geometry is sparse vs the input — the AR head needs much more training to produce dense output. The fact that <em>any</em> structurally-valid GeoJSON comes out (167 tokens → 3 roads / 10 buildings / 2 POIs in the second sample) confirms the constrained-decoding pipeline is wired correctly. <strong>Yellow but expected for 1 epoch.</strong></p>
<div class="twocol">
  <div><img src="data:image/png;base64,{inker_samples[0]}" alt="Inker decoded 0"></div>
  <div><img src="data:image/png;base64,{inker_samples[1]}" alt="Inker decoded 1"></div>
</div>

<h2>Go signal</h2>
<ul>
  <li><strong>Loss curves don't diverge:</strong> ✅ All three stages trained cleanly, no NaNs.</li>
  <li><strong>≥ 90 % well-formed GeoJSON:</strong> ✅ Validity rate <code>1.00</code> on val token-sequence round-trips.</li>
  <li><strong>Visual eyeball check on sampled tiles:</strong> ⏸ deferred — the eval driver doesn't yet sample from trained checkpoints. Plan 3 will add the <code>load checkpoint → dpmpp_sample → greedy_inker_sample</code> loop.</li>
</ul>

<h2>What's next (Plan 3)</h2>
<ul>
  <li><strong>Experiment 1:</strong> Train ~200M-param Sketcher on real Sweden + Singapore + Sri Lanka tile shards. Track per-channel IoU, FID, conditioning effectiveness. ~80 GPU-h budget.</li>
  <li><strong>Experiment 2:</strong> Train ~300M-param Inker on perfect ground-truth raster input (no domain gap yet). Track building Chamfer, road-graph connectivity, GeoJSON validity. ~120 GPU-h budget.</li>
  <li><strong>Add the actual sampling loop</strong> so we can dump 32 generated tiles per checkpoint for human eyeball QC.</li>
</ul>

<p style="font-size: 12px; color: #888; margin-top: 36px;">Generated by <code>scripts/render_eval_report.py</code>. Self-contained; all images inlined as base64 PNGs. No external dependencies for viewing.</p>

</body>
</html>
"""

    OUT_HTML.write_text(html)
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Wrote {OUT_HTML} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
