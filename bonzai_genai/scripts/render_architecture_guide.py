"""Generate an illustrated HTML architecture guide for Bonzai-OSM.

Output: ``bonzai_genai/results/ARCHITECTURE_GUIDE.html``

Walks through the two-stage Sketcher + Inker architecture using ONE running
example tile, with matplotlib diagrams generated inline (base64-encoded PNG).
Self-contained — the HTML opens in any browser with no external assets.
"""
from __future__ import annotations

import base64
import io
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.config import TILE_SIDE_M  # noqa: E402
from bonzai_genai.data.rasteriser import rasterise  # noqa: E402
from bonzai_genai.synth.procedural import generate_synthetic_tile  # noqa: E402
from bonzai_genai.vocab.attributes import load_default_vocab  # noqa: E402
from bonzai_genai.vocab.tokeniser import Tokeniser  # noqa: E402
from bonzai_genai.vocab.tokens import (  # noqa: E402
    NUM_NODE_REF_TOKENS,
    NUM_SPECIAL_TOKENS,
    SpecialToken,
)

REPO = Path(__file__).resolve().parents[1]
OUT_HTML = REPO / "results" / "ARCHITECTURE_GUIDE.html"


def _b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


CHANNEL_NAMES = (
    "0: motorway", "1: trunk", "2: primary", "3: residential",
    "4: all-roads", "5: building density", "6: water",
    "7: green", "8: urban",
)


# ---------------------------------------------------------------------------
# Visual #1 — the tile (just the all-roads channel + a simple coordinate grid)
# ---------------------------------------------------------------------------
def fig_tile_overview(geom_raster: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(geom_raster[4], cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.set_xticklabels(["0 m", "512 m", "1024 m", "1536 m", "2048 m"])
    ax.set_yticks([0, 128, 256, 384, 511])
    ax.set_yticklabels(["0 m", "512 m", "1024 m", "1536 m", "2048 m"])
    ax.set_xlabel("x (tile-local metres)")
    ax.set_ylabel("y (tile-local metres)")
    ax.set_title("Tile #42 — 2 km × 2 km square (all-roads layer)")
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #2 — the 9-channel layer cake
# ---------------------------------------------------------------------------
def fig_layer_cake(raster: np.ndarray) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ch in range(9):
        ax = axes[ch // 3, ch % 3]
        if ch == 5:
            ax.imshow(raster[ch], cmap="viridis", vmin=0, vmax=1)
        else:
            ax.imshow(raster[ch], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Tile #42 — 9-channel raster ('the layer cake')", fontsize=12)
    fig.tight_layout()
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #3 — token sequence as colored sequence (annotated)
# ---------------------------------------------------------------------------
def fig_token_strip(tokens: list[int], vocab) -> str:
    """Render the first ~80 tokens of a sequence as colored boxes."""
    n_show = min(80, len(tokens))
    seq = tokens[:n_show]

    def classify(t: int) -> tuple[str, str]:
        # Returns (category, label_text)
        if t < NUM_SPECIAL_TOKENS:
            try:
                name = SpecialToken(t).name
            except ValueError:
                name = f"S{t}"
            return ("special", name)
        x_lo = NUM_SPECIAL_TOKENS
        x_hi = x_lo + 512
        y_lo = x_hi
        y_hi = y_lo + 512
        nref_lo = y_hi
        nref_hi = nref_lo + NUM_NODE_REF_TOKENS
        if x_lo <= t < x_hi:
            return ("xcoord", f"x{t - x_lo}")
        if y_lo <= t < y_hi:
            return ("ycoord", f"y{t - y_lo}")
        if nref_lo <= t < nref_hi:
            return ("nref", f"n{t - nref_lo}")
        # Attribute
        try:
            name = vocab.name(t)
            short = name.split("=")[-1] if "=" in name else name
            return ("attr", short[:10])
        except KeyError:
            return ("unk", "?")

    colors = {
        "special": "#fbbf24",  # amber
        "xcoord":  "#60a5fa",  # blue
        "ycoord":  "#a78bfa",  # violet
        "nref":    "#34d399",  # green
        "attr":    "#f472b6",  # pink
        "unk":     "#aaaaaa",
    }
    legend_labels = {
        "special": "structural / control",
        "xcoord":  "x-coord bin (4 m)",
        "ycoord":  "y-coord bin (4 m)",
        "nref":    "road-node reference",
        "attr":    "attribute (e.g. road_class=residential)",
    }

    cols = 16
    rows = (n_show + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(11.5, 0.85 * rows + 1.4))
    cw, ch = 1.0, 0.9
    for i, t in enumerate(seq):
        cat, label = classify(t)
        x = (i % cols) * cw
        y = -(i // cols) * ch
        rect = mpatches.Rectangle(
            (x, y), cw - 0.05, -ch + 0.05,
            facecolor=colors[cat], edgecolor="#444", linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(
            x + cw / 2 - 0.025, y - ch / 2 + 0.025, label,
            ha="center", va="center", fontsize=7.5, color="#111",
        )
    ax.set_xlim(-0.2, cols * cw + 0.2)
    ax.set_ylim(-rows * ch - 0.5, ch + 0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    legend_handles = [
        mpatches.Patch(color=colors[k], label=v) for k, v in legend_labels.items()
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False,
    )
    ax.set_title(
        f"First {n_show} tokens of the encoded tile (full sequence: {len(tokens)} tokens)",
        fontsize=11,
    )
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #4 — vocabulary breakdown bar chart
# ---------------------------------------------------------------------------
def fig_vocab_breakdown(vocab) -> str:
    fams = {
        "special / control":  NUM_SPECIAL_TOKENS,
        "x-coord (4 m bins)": 512,
        "y-coord (4 m bins)": 512,
        "node-ref (road dots)": NUM_NODE_REF_TOKENS,
        "attribute":          len(vocab),
    }
    total = sum(fams.values())
    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0
    colors = ["#fbbf24", "#60a5fa", "#a78bfa", "#34d399", "#f472b6"]
    for (name, count), color in zip(fams.items(), colors, strict=False):
        ax.barh([0], [count], left=left, color=color, edgecolor="white")
        ax.text(
            left + count / 2, 0,
            f"{name}\n{count} tokens ({100 * count / total:.0f}%)",
            ha="center", va="center", fontsize=9, color="#111",
        )
        left += count
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"token id (0 → {total})")
    ax.set_title(f"Vocabulary breakdown — {total} tokens total")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #5 — forward diffusion (clean → noisy)
# ---------------------------------------------------------------------------
def fig_diffusion_forward(raster: np.ndarray) -> str:
    sigmas = [0.0, 0.5, 2.0, 10.0, 40.0, 80.0]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(13, 2.8))
    base = raster[4].copy()
    for ax, sigma in zip(axes, sigmas, strict=False):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(base.shape)
        noisy = base + sigma * noise
        # Normalise to [0,1] just for display
        lo, hi = noisy.min(), noisy.max()
        disp = (noisy - lo) / max(hi - lo, 1e-6)
        ax.imshow(disp, cmap="gray_r")
        ax.set_title(f"σ = {sigma:g}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Forward diffusion — adding noise to a tile (training-time data augmentation)",
        fontsize=11,
    )
    fig.tight_layout()
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #6 — pipeline diagram (boxes + arrows in matplotlib)
# ---------------------------------------------------------------------------
def fig_pipeline() -> str:
    fig, ax = plt.subplots(figsize=(13, 4.2))

    boxes = [
        # (x, y, w, h, text, color)
        (0.0, 1.0, 2.0, 1.6, "Text prompt\n+ region tags\n+ ControlNet hints", "#fef3c7"),
        (3.0, 1.0, 2.6, 1.6, "Sketcher (DiT)\n~400 M params\n50 denoising steps", "#dbeafe"),
        (6.6, 1.0, 2.0, 1.6, "9-ch raster\n(blueprint)\n512×512", "#e0e7ff"),
        (9.6, 1.0, 2.6, 1.6, "Inker (AR)\n~750 M params\nwrites tokens", "#fce7f3"),
        (13.2, 1.0, 2.0, 1.6, "GeoJSON\n(roads, buildings,\nPOIs, landuse)", "#dcfce7"),
    ]
    for x, y, w, h, text, color in boxes:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.2, edgecolor="#444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=10,
        )

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x_from = boxes[i][0] + boxes[i][2]
        x_to = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        arrow = FancyArrowPatch(
            (x_from + 0.05, y), (x_to - 0.05, y),
            arrowstyle="->", mutation_scale=18, color="#444", linewidth=1.4,
        )
        ax.add_patch(arrow)

    ax.set_xlim(-0.3, 15.5)
    ax.set_ylim(-0.2, 3.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Inference pipeline — prompt to GeoJSON", fontsize=12)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #7 — VAE compress / decompress diagram
# ---------------------------------------------------------------------------
def fig_vae_block() -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    items = [
        (0.0, 0.5, 2.4, 1.6, "9-channel raster\n512 × 512 × 9\n= 2.36M numbers", "#fef3c7"),
        (3.0, 0.5, 1.8, 1.6, "VAE encoder\n~5 conv\nlayers", "#dbeafe"),
        (5.4, 0.5, 2.4, 1.6, "Latent code\n64 × 64 × 4\n= 16K numbers\n(150× smaller)", "#e0e7ff"),
        (8.4, 0.5, 1.8, 1.6, "VAE decoder\n~5 conv\nlayers", "#fce7f3"),
        (10.8, 0.5, 2.4, 1.6, "9-channel raster\n512 × 512 × 9\n(reconstructed)", "#dcfce7"),
    ]
    for x, y, w, h, text, color in items:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.0, edgecolor="#444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrow_pairs = [(2.4, 3.0), (4.8, 5.4), (7.8, 8.4), (10.2, 10.8)]
    for x_from, x_to in arrow_pairs:
        ax.add_patch(FancyArrowPatch(
            (x_from, 1.3), (x_to, 1.3),
            arrowstyle="->", mutation_scale=16, color="#444", linewidth=1.2,
        ))
    ax.set_xlim(-0.3, 13.6)
    ax.set_ylim(0, 2.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("VAE — compresses the raster so the diffusion model can work fast", fontsize=11)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #8 — patch embedding (how DiT slices the latent into transformer tokens)
# ---------------------------------------------------------------------------
def fig_patch_embed() -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # 64x64 latent represented as a grid; each 2x2 block is a patch.
    grid = 64
    patch = 2
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (grid, grid))
    ax.imshow(img, cmap="RdBu", vmin=-2, vmax=2, alpha=0.35)
    # Draw patch boundaries (every 2 pixels)
    for k in range(0, grid + 1, patch):
        ax.axhline(k - 0.5, color="#222", linewidth=0.4, alpha=0.6)
        ax.axvline(k - 0.5, color="#222", linewidth=0.4, alpha=0.6)
    # Highlight one patch
    rect = mpatches.Rectangle((-0.5, -0.5), 2, 2, linewidth=2.4, edgecolor="#dc2626", facecolor="none")
    ax.add_patch(rect)
    ax.annotate(
        "one patch\n→ one transformer token\n(2×2×4 = 16 numbers → 512-dim embed)",
        xy=(1, 0), xytext=(8, -3),
        arrowprops=dict(arrowstyle="->", color="#dc2626", linewidth=1.2),
        fontsize=9, color="#dc2626",
    )
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(grid - 0.5, -0.5)
    ax.set_xticks(range(0, grid + 1, 8))
    ax.set_yticks(range(0, grid + 1, 8))
    ax.set_title(
        "DiT patch embedding — 64×64 latent cut into 32×32 = 1,024 transformer tokens",
        fontsize=11,
    )
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #9 — Inker autoregressive flow
# ---------------------------------------------------------------------------
def fig_inker_flow() -> str:
    fig, ax = plt.subplots(figsize=(13, 4.4))

    # Top row: token sequence being written one at a time
    seq_items = ["BOS", "LAYER_LAND", "LAYER_ROADS", "ROAD_NODE", "x_50", "y_80",
                 "ROAD_NODE", "x_95", "y_82", "ROAD_EDGE", "n_0", "n_1", "...", "EOS"]
    n = len(seq_items)
    cw = 0.95
    for i, t in enumerate(seq_items):
        x = i * cw
        # Special tokens vs coord tokens get different colors
        col = "#fbbf24" if t in {"BOS", "EOS", "LAYER_LAND", "LAYER_ROADS", "ROAD_NODE", "ROAD_EDGE"} else (
            "#60a5fa" if t.startswith("x_") else
            "#a78bfa" if t.startswith("y_") else
            "#34d399" if t.startswith("n_") else "#cccccc")
        rect = mpatches.Rectangle(
            (x, 3.2), cw - 0.08, 0.7,
            facecolor=col, edgecolor="#444", linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(x + cw / 2 - 0.04, 3.55, t, ha="center", va="center", fontsize=8.5)
    ax.text(-0.4, 3.55, "tokens →", ha="right", va="center", fontsize=10, fontstyle="italic", color="#666")

    # Mid row: arrow up from current token position to the "next-token" prediction point
    cursor_i = 11
    ax.add_patch(FancyArrowPatch(
        (cursor_i * cw + cw / 2, 3.18), (cursor_i * cw + cw / 2, 2.1),
        arrowstyle="->", mutation_scale=15, color="#dc2626", linewidth=1.5,
    ))
    ax.text(
        cursor_i * cw + cw / 2 + 0.4, 2.6,
        "predict next\ntoken given\neverything to the\nleft (causal)",
        fontsize=9, color="#dc2626",
    )

    # Lower-left: cross-attention to the raster blueprint
    raster_box = FancyBboxPatch(
        (0.3, 0.3), 4.5, 1.6, boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.2, edgecolor="#444", facecolor="#dbeafe",
    )
    ax.add_patch(raster_box)
    ax.text(2.55, 1.1, "Raster blueprint\n(from Sketcher or GT)\n→ CNN encoder → 32×32 features",
            ha="center", va="center", fontsize=9)

    # Cross-attention arrows from raster box up to current token
    ax.add_patch(FancyArrowPatch(
        (4.8, 1.4), (cursor_i * cw + cw / 2 - 0.15, 2.05),
        arrowstyle="->", mutation_scale=15, color="#2563eb", linewidth=1.3,
        connectionstyle="arc3,rad=-0.25",
    ))
    ax.text(
        7.5, 1.5,
        "cross-attention:\nthe Inker 'looks at'\nthe blueprint while\nwriting the next token",
        fontsize=9, color="#2563eb",
    )

    ax.set_xlim(-1.5, n * cw + 0.6)
    ax.set_ylim(-0.1, 4.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Inker — writes one token at a time, each step glances at the raster", fontsize=11)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #10 — constrained decoding mask in action
# ---------------------------------------------------------------------------
def fig_constrained_mask() -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    # Show the model's "raw" probability distribution + the mask + the result
    rng = np.random.default_rng(2)
    vocab_segments = [
        ("special",  NUM_SPECIAL_TOKENS, "#fbbf24"),
        ("x-coords", 512,                "#60a5fa"),
        ("y-coords", 512,                "#a78bfa"),
        ("node-refs", NUM_NODE_REF_TOKENS, "#34d399"),
        ("attrs",    300,                "#f472b6"),
    ]
    total_w = sum(w for _, w, _ in vocab_segments)
    # Three rows: raw model probs, mask (after we just emitted x_42), final probs
    rows = [
        ("model wants to emit", rng.dirichlet(np.ones(total_w) * 0.05) * 6, None),
        ("constrained mask\n(just emitted x_42)", None, "y_only"),
        ("final probabilities", None, "y_masked"),
    ]
    y0 = 2.2
    bar_h = 0.55
    for ridx, (label, raw, mask_kind) in enumerate(rows):
        y = y0 - ridx * 0.85
        x = 0
        for seg_name, seg_w, color in vocab_segments:
            if mask_kind == "y_only":
                allowed = seg_name == "y-coords"
                fill = color if allowed else "#e5e7eb"
                ax.add_patch(mpatches.Rectangle((x, y), seg_w / total_w * 10, bar_h,
                                                facecolor=fill, edgecolor="white", linewidth=0))
            elif mask_kind == "y_masked":
                # zero-out everything not y-coords; rescale
                fill = color if seg_name == "y-coords" else "#f3f4f6"
                ax.add_patch(mpatches.Rectangle((x, y), seg_w / total_w * 10, bar_h,
                                                facecolor=fill, edgecolor="white", linewidth=0))
            else:
                ax.add_patch(mpatches.Rectangle((x, y), seg_w / total_w * 10, bar_h,
                                                facecolor=color, edgecolor="white", linewidth=0))
            x += seg_w / total_w * 10
        ax.text(-0.2, y + bar_h / 2, label, ha="right", va="center", fontsize=9.5)
    # Segment names along the top
    x = 0
    for seg_name, seg_w, color in vocab_segments:
        w = seg_w / total_w * 10
        ax.text(x + w / 2, y0 + bar_h + 0.1, seg_name, ha="center", va="bottom", fontsize=9)
        x += w
    ax.set_xlim(-2.5, 10.5)
    ax.set_ylim(-0.4, 3.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "Constrained decoding — block illegal tokens at sample time (no retraining)",
        fontsize=11,
    )
    return _b64(fig)


# ---------------------------------------------------------------------------
# Visual #11 — training-time vs inference-time pipeline
# ---------------------------------------------------------------------------
def fig_train_vs_infer() -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.2))
    for row, mode in enumerate(["Training", "Inference"]):
        ax = axes[row]
        if mode == "Training":
            stages = [
                "Real tile\n(raster + tokens)",
                "Add noise\nto raster",
                "Sketcher predicts\nclean raster\n(loss = MSE)",
                "Inker reads GT\nraster → predicts\nnext token\n(loss = CE)",
            ]
            colors = ["#fef3c7", "#fee2e2", "#dbeafe", "#fce7f3"]
        else:
            stages = [
                "Prompt + tags",
                "Sketcher samples\nraster from noise\n(50 denoise steps)",
                "Inker writes tokens\nwith constrained\ndecoding",
                "Tokenizer →\nGeoJSON output",
            ]
            colors = ["#fef3c7", "#dbeafe", "#fce7f3", "#dcfce7"]
        x_left = 0
        ax.text(-0.3, 0.95, mode, fontsize=12, fontweight="bold", ha="right", va="center",
                color="#0f172a")
        for stage, color in zip(stages, colors, strict=False):
            box = FancyBboxPatch(
                (x_left, 0.3), 2.7, 1.3, boxstyle="round,pad=0.05,rounding_size=0.12",
                linewidth=1.0, edgecolor="#444", facecolor=color,
            )
            ax.add_patch(box)
            ax.text(x_left + 1.35, 0.95, stage, ha="center", va="center", fontsize=9)
            x_left += 3.1
        for i in range(len(stages) - 1):
            ax.add_patch(FancyArrowPatch(
                (i * 3.1 + 2.7 + 0.05, 0.95), (i * 3.1 + 3.1 - 0.05, 0.95),
                arrowstyle="->", mutation_scale=15, color="#444", linewidth=1.2,
            ))
        ax.set_xlim(-2.0, 13)
        ax.set_ylim(0.0, 1.8)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle("Same architecture, two modes: training vs inference", fontsize=12)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Use one running example tile across the whole guide.
    geom = generate_synthetic_tile(seed=42, density="dense")
    raster = rasterise(geom)
    vocab = load_default_vocab()
    tok = Tokeniser(vocab)
    tokens = tok.encode(geom)

    img_tile_overview = fig_tile_overview(raster)
    img_layer_cake = fig_layer_cake(raster)
    img_token_strip = fig_token_strip(tokens, vocab)
    img_vocab = fig_vocab_breakdown(vocab)
    img_diffusion = fig_diffusion_forward(raster)
    img_pipeline = fig_pipeline()
    img_vae = fig_vae_block()
    img_patch = fig_patch_embed()
    img_inker = fig_inker_flow()
    img_mask = fig_constrained_mask()
    img_train_vs_infer = fig_train_vs_infer()

    # Numeric facts derived from the example tile
    n_roads = len(geom.roads)
    n_buildings = len(geom.buildings)
    n_land = len(geom.land)
    n_pois = len(geom.pois)
    n_tokens = len(tokens)
    raster_numbers = 9 * 512 * 512
    latent_numbers = 4 * 64 * 64

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bonzai-OSM — Architecture Guide</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 1080px; margin: 24px auto; padding: 0 22px; color: #1f2937;
    line-height: 1.55; font-size: 15px;
  }}
  h1 {{ font-size: 26px; border-bottom: 2px solid #4a7ab8; padding-bottom: 8px; }}
  h2 {{ font-size: 21px; margin-top: 38px; color: #1e40af; border-left: 4px solid #4a7ab8; padding-left: 12px; }}
  h3 {{ font-size: 16px; color: #334155; margin-top: 22px; }}
  .lead {{ background: #eff6ff; padding: 12px 16px; border-radius: 6px; border-left: 4px solid #4a7ab8; margin: 14px 0; }}
  .analogy {{ background: #fef3c7; padding: 10px 14px; border-radius: 6px; font-size: 14px; margin: 12px 0; }}
  .analogy strong {{ color: #b45309; }}
  table {{ border-collapse: collapse; margin: 10px 0 16px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d5db; padding: 5px 12px; text-align: left; }}
  th {{ background: #eef2f6; }}
  td.num {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; text-align: right; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0 4px 0; background: white; }}
  figcaption {{ font-size: 12.5px; color: #64748b; margin-bottom: 14px; font-style: italic; }}
  code {{ background: #f1f5f9; padding: 1px 6px; border-radius: 3px; font-size: 13px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  ul {{ padding-left: 22px; }}
  li {{ margin: 4px 0; }}
  .keypt {{ background: #ecfdf5; padding: 10px 14px; border-radius: 6px; border-left: 4px solid #10b981; margin: 12px 0; font-size: 14px; }}
  .keypt strong {{ color: #047857; }}
  .twocol {{ display: flex; gap: 18px; flex-wrap: wrap; }}
  .twocol > div {{ flex: 1; min-width: 380px; }}
</style>
</head>
<body>

<h1>Bonzai-OSM — Architecture Guide</h1>
<p style="color:#64748b; font-size:14px;">A visual walkthrough using one running example tile. Generated 2026-05-04.</p>

<div class="lead">
<strong>What we're building, in one sentence:</strong> a model that takes a text description (<em>"dense Asian commercial, coastal, 2 km²"</em>) and produces a real, machine-readable city map of that area as GeoJSON — with roads, buildings, points of interest, and land use, all in their precise vector form (not just pixels).
</div>

<div class="analogy">
<strong>The big analogy:</strong> imagine two artists working on the same canvas. The first artist is a <strong>watercolour painter</strong> who lays down rough shapes: where the roads go, where the buildings cluster, which areas are water vs park vs urban. The second artist is a <strong>pen-and-ink illustrator</strong> who traces over the watercolour with precise lines: every road as a polyline, every building as a polygon with exact corners, every café or school as a point with a label. We call them the <strong>Sketcher</strong> and the <strong>Inker</strong>. They train separately. At generation time they run in sequence: Sketcher paints first, Inker traces second.
</div>

<h2>1. The unit of work — one tile</h2>

<p>The whole system operates on <strong>tiles</strong>. A tile is a 2 km × 2 km square of the world, picked from a tile grid that covers the whole planet. Each tile is treated independently — the model never sees more than one tile at a time. To make a city, we generate many tiles and stitch them together at the end.</p>

<p>Here's our running example, <strong>"Tile #42"</strong>. It's a synthetic tile from the smoke-test corpus — generated procedurally by a Python function, not from real OSM data, but representative of what real tiles look like in their raw form:</p>

<img src="data:image/png;base64,{img_tile_overview}" alt="Tile #42 overview">
<figcaption>Tile #42, all-roads layer. Coordinates start at 0,0 in the south-west corner and run to 2048 m × 2048 m. <strong>Coordinates are tile-local</strong> — we never store global lat/lon inside a tile. Stockholm and Singapore both look like "0 to 2048 m" inside their tiles.</figcaption>

<div class="keypt">
<strong>Key idea — tile-local coordinates:</strong> by always re-indexing each tile from its own SW corner, every tile in the world has the same coordinate range. The model doesn't have to learn 7 billion different latitude/longitude bands. It just learns "what does a 2 km × 2 km city look like?" Stitching is handled separately at sample time.
</div>

<h3>Tile #42 in numbers</h3>
<table>
<tr><th>property</th><th>value</th></tr>
<tr><td>roads (polylines)</td><td class="num">{n_roads}</td></tr>
<tr><td>buildings (polygons)</td><td class="num">{n_buildings}</td></tr>
<tr><td>landuse polygons</td><td class="num">{n_land}</td></tr>
<tr><td>POIs (points)</td><td class="num">{n_pois}</td></tr>
</table>

<h2>2. Two ways to look at the same tile</h2>

<p>Every tile lives in our system as <strong>two parallel representations</strong>:</p>

<ul>
<li><strong>The raster</strong> — a stack of 9 black-and-white pictures (a "layer cake"). Easy for a diffusion model to chew on, but you can't directly read a building polygon out of it.</li>
<li><strong>The vector tokens</strong> — a flat sequence of integers that, when decoded, gives back the exact roads/buildings/POIs. Easy for an autoregressive transformer (like an LLM) to write out, but harder for a diffusion model to learn directly.</li>
</ul>

<p>We keep <em>both</em> because each of our two artists prefers one. The Sketcher works on rasters; the Inker works on tokens.</p>

<h3>The raster — 9 transparent overlays</h3>

<img src="data:image/png;base64,{img_layer_cake}" alt="9-channel raster">
<figcaption>Tile #42 as a 9-channel raster (3×3 grid above). Each panel is a 512×512 black-and-white picture. Black = "this thing is here at this pixel"; white = "absent". The continuous channel 5 (building density) uses a heatmap colour scheme. The total raster is {raster_numbers:,} numbers per tile.</figcaption>

<table>
<tr><th>channel</th><th>type</th><th>what it is</th></tr>
<tr><td>0</td><td>binary mask</td><td>motorway-class roads</td></tr>
<tr><td>1</td><td>binary mask</td><td>trunk-class roads</td></tr>
<tr><td>2</td><td>binary mask</td><td>primary-class roads</td></tr>
<tr><td>3</td><td>binary mask</td><td>residential-class roads</td></tr>
<tr><td>4</td><td>binary mask</td><td><strong>all roads combined</strong> — the most informative single channel</td></tr>
<tr><td>5</td><td>continuous</td><td>building density (heatmap)</td></tr>
<tr><td>6</td><td>binary mask</td><td>water (lakes, rivers, sea)</td></tr>
<tr><td>7</td><td>binary mask</td><td>green (parks, forests)</td></tr>
<tr><td>8</td><td>binary mask</td><td>urban (built-up areas)</td></tr>
</table>

<h3>The vector tokens — a recipe</h3>

<p>The same tile encoded as a sequence of integers. Think of it like a cooking recipe with a fixed grammar: <em>"BOS (begin), now I'll describe the LAND layer, here's a polygon: open, class=park, vertex (x, y), vertex (x, y), …, close, now the ROADS layer, here are the road nodes, here are the edges connecting them, …, EOS (end)"</em>.</p>

<img src="data:image/png;base64,{img_token_strip}" alt="Token sequence first 80">
<figcaption>The first 80 tokens of Tile #42's full {n_tokens}-token encoding. Each box is one integer token. Colours are by family: amber = structural (BOS, layer markers, polygon delimiters), blue = x-coordinate bins (in 4 m steps), violet = y-coordinate bins, green = node references for the road graph, pink = attribute tokens (e.g. <code>building_class=residential</code>).</figcaption>

<h3>The vocabulary — only ~9,700 distinct tokens total</h3>

<img src="data:image/png;base64,{img_vocab}" alt="Vocab breakdown">
<figcaption>The whole token vocabulary. By comparison, GPT-style LLMs use ~50,000-100,000 BPE tokens. Our domain is constrained, so we get away with a tiny vocabulary — every token is meaningful (not arbitrary text fragments).</figcaption>

<div class="keypt">
<strong>Key idea — same tile, two formats:</strong> the raster is what the eye sees; the tokens are what a writer would dictate. Encoding a tile to either format is fully reversible (lossless up to coordinate quantisation). Most of our cleverness is about training models that move <em>between</em> the two formats.
</div>

<h2>3. The Sketcher (Stage A) — paint a blueprint via diffusion</h2>

<p>The Sketcher is a <strong>diffusion model</strong>. Here's how diffusion works in plain words:</p>

<div class="analogy">
<strong>Forward process:</strong> take a real tile; gradually add random noise until it becomes pure TV-static. The bigger the noise, the less recognisable the tile.<br><br>
<strong>Reverse process:</strong> train a network to undo one noise step at a time. At inference: start with pure noise, run the trained network 50 times, each time it removes a little noise. By the end you have a brand-new tile that looks like the training distribution but isn't a copy of any specific training example.
</div>

<img src="data:image/png;base64,{img_diffusion}" alt="Forward diffusion">
<figcaption>Forward diffusion on Tile #42. Left = the real tile (σ = 0). Right = pure noise (σ = 80). The model learns to <em>reverse</em> this — turn the right-most picture back into the left-most. At inference time, we don't know the left-most picture; we start with pure noise and let the model invent one.</figcaption>

<h3>Why we need a VAE in front</h3>

<p>The raw raster is 512 × 512 × 9 = {raster_numbers:,} numbers per tile. Doing diffusion at that resolution is brutally expensive — each denoising step has to process all 2.36 million numbers, and we need 50 such steps. That would take seconds per tile, and training would take weeks.</p>

<p>So we cheat: we train a small <strong>VAE</strong> (Variational Auto-Encoder, basically a learned compressor) that squeezes the raster down to a much smaller "latent" representation. The diffusion model works on the small representation; we only blow it back up to full size at the end.</p>

<img src="data:image/png;base64,{img_vae}" alt="VAE block diagram">
<figcaption>VAE pipeline: 9 × 512 × 512 = {raster_numbers:,} numbers in, 4 × 64 × 64 = {latent_numbers:,} numbers out (~150× smaller), then reverse. The VAE is trained <em>once</em> at the start and then frozen; the Sketcher learns on the small latent codes.</figcaption>

<h3>Inside the Sketcher — a transformer that paints</h3>

<p>The Sketcher itself is a <strong>DiT</strong> (Diffusion Transformer). The trick: even though the data is a 2D image (the latent), we slice it into small patches and treat each patch as a "token", then run a normal transformer over the resulting sequence. This is borrowed from how Vision Transformers work for classification, applied to diffusion.</p>

<img src="data:image/png;base64,{img_patch}" alt="Patch embedding">
<figcaption>The 64 × 64 latent gets cut into 32 × 32 = 1,024 small patches (each 2 × 2 × 4 numbers). Each patch becomes one "token" the transformer can attend to. So the DiT is effectively a transformer over a 1,024-token sequence — same scale as a short LLM context.</figcaption>

<div class="analogy">
<strong>Why patches?</strong> Transformers love sequences. A 64 × 64 image isn't a sequence — but if you cut it into patches and serialise them, you get a 1,024-token sequence the transformer can chew. Bonus: long-range attention now spans the whole tile (the transformer can compare any patch to any other patch in one step, vs a CNN where information has to flow layer-by-layer).
</div>

<h2>4. The Inker (Stage B) — write precise vectors</h2>

<p>The Inker is an <strong>autoregressive transformer</strong>, mechanically very similar to GPT or Llama. It writes the token sequence one token at a time, left to right. At each step, it:</p>

<ol>
<li>Looks at every token it has already written (causal self-attention).</li>
<li>Looks at the rasterised blueprint (cross-attention to a small CNN encoder of the raster).</li>
<li>Predicts a probability distribution over the ~9,700 vocabulary tokens.</li>
<li>Picks one (greedy at smoke time, beam search later) and appends it.</li>
</ol>

<img src="data:image/png;base64,{img_inker}" alt="Inker autoregressive flow">
<figcaption>The Inker writes left-to-right. The red arrow shows the "prediction cursor" — at this position the model has already written everything to the left and now decides the next token. The blue arrow shows cross-attention: it can also "look at" the blueprint raster while making that decision.</figcaption>

<h3>Constrained decoding — preventing nonsense output</h3>

<p>An autoregressive model could theoretically emit invalid token sequences: x without a y, BUILDING_OPEN with no matching BUILDING_CLOSE, layer markers in the wrong order, etc. That would parse to malformed GeoJSON.</p>

<p>We prevent this with <strong>logit masking</strong>: at sample time, we look at what was just emitted, figure out which tokens would be structurally valid next, and zero out everything else from the model's probability distribution. No retraining needed; pure decoder-side enforcement.</p>

<img src="data:image/png;base64,{img_mask}" alt="Constrained decoding mask">
<figcaption>Example: we just emitted an x-coordinate token (e.g. <code>x_42</code>). The grammar says the next token must be a y-coordinate. The mask zeros out everything except the y-coord segment. Even if the model's raw output gave non-zero probability to (say) BOS, the mask makes it impossible to pick BOS next.</figcaption>

<div class="analogy">
<strong>Why this matters:</strong> our buyers (sim engineers, researchers) need parseable GeoJSON. A 99.5%-valid output stream that occasionally produces a malformed polygon is much worse than a 95% creative-quality stream that's always 100% parseable. Logit masking trades some sample diversity for hard validity guarantees.
</div>

<h2>5. Putting it together</h2>

<p>At training time, the Sketcher and Inker are trained <strong>independently</strong>. The Sketcher learns to denoise rasters. The Inker learns to write tokens given a ground-truth raster. They never see each other.</p>

<p>At inference time, they run <strong>in sequence</strong>:</p>

<img src="data:image/png;base64,{img_pipeline}" alt="Inference pipeline">
<figcaption>Top-level flow: text prompt + region tags + control hints → Sketcher paints a 9-channel raster → that raster goes into the Inker → tokens out → tokens decoded into GeoJSON. The whole pipeline takes ~5-30 seconds per tile (50 diffusion steps + a few thousand AR token generations).</figcaption>

<img src="data:image/png;base64,{img_train_vs_infer}" alt="Training vs inference">
<figcaption>Same architecture, two modes. At training, the Inker reads <em>ground-truth</em> rasters (no Sketcher involved). At inference, the Inker reads <em>Sketcher-sampled</em> rasters. The mismatch between these two — called the "domain gap" — is one of our four de-risking experiments (Experiment 3).</figcaption>

<h2>6. Conditioning — how to steer the output</h2>

<p>So far we've described an unconditional generator: feed it noise, get a random tile. To make it useful we need <strong>conditioning</strong> — ways to tell it what kind of tile we want. The Sketcher accepts three kinds of conditioning, all merged into a single embedding vector that gets injected into every transformer block:</p>

<ul>
<li><strong>Text prompt</strong> — a CLIP-class text encoder converts <em>"dense Asian commercial, coastal"</em> into a 768-dim embedding.</li>
<li><strong>Region tags</strong> — country code, Köppen climate class, density bucket, primary land-use → small lookup tables → summed conditioning vector.</li>
<li><strong>Spatial controls (EasyControl LoRA)</strong> — terrain heightmap, coastline mask, style anchor (an existing tile to copy aesthetic from), constraint mask (where roads MUST go). Each control type gets its own small LoRA adapter that injects extra tokens into the DiT blocks.</li>
</ul>

<p>The Inker doesn't take its own conditioning — it's already conditioned by the Sketcher's output (which encodes everything the conditioning influenced).</p>

<h2>7. Why this design (vs alternatives)</h2>

<table>
<tr><th>alternative</th><th>why we didn't do it</th></tr>
<tr>
<td>Pure pixel diffusion (one big model that outputs GeoJSON-rendered pixels)</td>
<td>Pixels can't represent vector primitives precisely. A 1-pixel-wide road is rasterisation noise; a 4-pixel-wide road is a real road. Sim engineers need vector geometry, not blurry images.</td>
</tr>
<tr>
<td>Pure autoregressive (one big LLM that writes the whole GeoJSON token sequence from a text prompt, no raster)</td>
<td>LLMs are bad at long-range spatial coherence. Tiles have ~5,000+ tokens of geometric structure where layout matters. The Sketcher handles global structure; the Inker handles local precision.</td>
</tr>
<tr>
<td>Two-stage but with two LLMs (no diffusion)</td>
<td>Diffusion is dramatically better than AR at producing smooth 2D fields (think building density). AR models trying to fill a raster pixel-by-pixel scale poorly.</td>
</tr>
<tr>
<td>Direct vector diffusion (a diffusion model that outputs tokens directly)</td>
<td>Active research area but immature. Standard rasterised diffusion + AR-on-top is the safer 2026-era SOTA.</td>
</tr>
</table>

<h2>8. Where we stand right now</h2>

<p>This guide describes the <em>target</em> architecture. We've built it (~5 K LoC of custom-from-scratch PyTorch + PyTorch Lightning trainers + Slurm GPU job templates). We've run a smoke test on Leonardo with tiny smoke-size models on synthetic data — it produced coherent grid-and-blocks tiles after just one epoch of training (see the smoke samples in <code>EXPERIMENT_0_REPORT.html</code>).</p>

<p>Next up (<strong>Plan 3</strong>): bigger models trained on real Sweden / Singapore / Sri Lanka data, with actual sample dumps to confirm the architecture scales from synthetic grids to real urban data.</p>

<p style="font-size: 12.5px; color: #888; margin-top: 36px;">Generated by <code>scripts/render_architecture_guide.py</code>. Self-contained — no external dependencies, no internet required to view.</p>

</body>
</html>
"""

    OUT_HTML.write_text(html)
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Wrote {OUT_HTML} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
