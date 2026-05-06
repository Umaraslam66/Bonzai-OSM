"""Eval driver invoked by Experiment 0, Plan 3, and standalone eval jobs.

Two modes:
  default                    - measure val-set eval metrics (Phase 0b smoke).
  BONZAI_SAMPLE_FROM_CKPT=1  - load Painter / Writer / VAE checkpoints,
                              generate N samples, dump PNGs + GeoJSON
                              to BONZAI_SAMPLE_OUT.

Plan 3 mode env-vars:
    BONZAI_CKPT_DIR              dir containing vae.ckpt, stage_a.ckpt, stage_b.ckpt
    BONZAI_SAMPLE_OUT            output dir for PNGs + GeoJSON
    BONZAI_PRESET                "plan3" (default) | "tiny" | "production"
    BONZAI_NUM_SAMPLES           default 64 (16 unconditional + 48 conditional)
    BONZAI_NUM_DPM_STEPS         default 50
    BONZAI_INKER_MAX_TOKENS      default 4096
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonzai_genai.eval.stage_a import channel_iou, fid_lite  # noqa: E402
from bonzai_genai.eval.stage_b import (  # noqa: E402
    building_chamfer,
    building_self_intersection_rate,
    poi_placement_distance,
    road_graph_single_component_fraction,
    validity_rate,
)
from bonzai_genai.training.data_module import TileDataModule  # noqa: E402
from bonzai_genai.vocab.attributes import load_default_vocab  # noqa: E402
from bonzai_genai.vocab.tokeniser import Tokeniser  # noqa: E402


def main_val_eval() -> None:
    """Original Phase 0b mode: metrics on the val ground-truth set + Markdown report."""
    out_dir = Path(os.environ["BONZAI_EXP0_OUT"])
    val_url = os.environ["BONZAI_VAL_URL"]

    dm = TileDataModule(
        train_url=val_url, val_url=val_url, batch_size=16,
        return_tokens=True, num_workers=0,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    val_rasters_list: list[torch.Tensor] = []
    val_tokens_lists: list[list[int]] = []
    for batch in val_loader:
        val_rasters_list.append(batch["raster"])
        for i in range(batch["tokens"].shape[0]):
            n = int(batch["token_lens"][i])
            val_tokens_lists.append(batch["tokens"][i, :n].tolist())
    val_rasters = torch.cat(val_rasters_list, dim=0)

    vocab = load_default_vocab()
    results: dict[str, dict] = {}

    iou = channel_iou(val_rasters[:32], val_rasters[:32])
    fid = fid_lite(val_rasters[:32], val_rasters[32:64]) if val_rasters.shape[0] >= 64 else 0.0
    results["stage_a"] = {
        "channel_iou_self": iou,
        "fid_lite_real_vs_real": float(fid),
    }

    val_rate = validity_rate(val_tokens_lists[:32], vocab=vocab)
    results["stage_b"] = {"validity_rate_val_tokens": val_rate}

    tok = Tokeniser(vocab)
    chamfer_vals: list[float] = []
    rg_fracs: list[float] = []
    poi_dists: list[float] = []
    si_rates: list[float] = []
    for seq in val_tokens_lists[:4]:
        try:
            geom = tok.decode(list(seq))
            chamfer_vals.append(building_chamfer(geom, geom))
            rg_fracs.append(road_graph_single_component_fraction(geom))
            poi_dists.append(poi_placement_distance(geom, geom))
            si_rates.append(building_self_intersection_rate(geom))
        except Exception as e:  # noqa: BLE001
            print(f"decode failed: {e}", file=sys.stderr)
    results["stage_b"]["building_chamfer_self"] = float(
        sum(chamfer_vals) / max(len(chamfer_vals), 1)
    )
    results["stage_b"]["road_graph_largest_frac"] = float(
        sum(rg_fracs) / max(len(rg_fracs), 1)
    )
    results["stage_b"]["poi_placement_self"] = float(
        sum(poi_dists) / max(len(poi_dists), 1)
    )
    results["stage_b"]["building_self_intersection"] = float(
        sum(si_rates) / max(len(si_rates), 1)
    )

    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    # Human-readable report (preserved from Phase 0b)
    report = ["# Experiment 0 Report", ""]
    report.append(f"**Output dir:** `{out_dir}`")
    report.append("")
    report.append("## Stage A metrics (smoke; on ground-truth val rasters)")
    for ch, val in results["stage_a"]["channel_iou_self"].items():
        report.append(f"- channel {ch} IoU (self): {val:.4f}")
    report.append(
        f"- FID-lite (real vs real, sanity): "
        f"{results['stage_a']['fid_lite_real_vs_real']:.2f}"
    )
    report.append("")
    report.append("## Stage B metrics (smoke; on val token sequences)")
    for k, v in results["stage_b"].items():
        report.append(f"- {k}: {v}")
    report.append("")
    report.append("## Go / No-Go")
    report.append(
        "- Visual: see sample dumps under `stage_a/` and `stage_b/` (lightning_logs)."
    )
    pass_validity = results["stage_b"]["validity_rate_val_tokens"] >= 0.90
    report.append(
        f"- Validity >= 90%: {'PASS' if pass_validity else 'NEEDS REVIEW'}"
    )
    (out_dir / "EXPERIMENT_0_REPORT.md").write_text("\n".join(report))
    print("Wrote", out_dir / "EXPERIMENT_0_REPORT.md")


def _render_raster_png(raster: torch.Tensor, path: Path) -> None:
    """Render a (9, 512, 512) raster as a 3x3 grid PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = raster.detach().cpu().numpy()
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ch in range(9):
        ax = axes[ch // 3, ch % 3]
        ax.imshow(
            arr[ch], cmap="viridis" if ch == 5 else "gray_r",
            vmin=0, vmax=1,
        )
        ax.set_title(f"ch{ch}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _geom_to_geojson(geom) -> dict:
    """Convert a TileGeometry to a GeoJSON FeatureCollection (tile-local m)."""
    feats: list[dict] = []
    for poly in getattr(geom, "land", []):
        feats.append({
            "type": "Feature",
            "properties": {"kind": "land", "class": poly.class_name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(poly.vertices)],
            },
        })
    for road in getattr(geom, "roads", []):
        feats.append({
            "type": "Feature",
            "properties": {"kind": "road", "class": road.class_name},
            "geometry": {
                "type": "LineString",
                "coordinates": list(road.polyline),
            },
        })
    for bldg in getattr(geom, "buildings", []):
        feats.append({
            "type": "Feature",
            "properties": {
                "kind": "building",
                "class": bldg.class_name,
                "height": bldg.height_name,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(bldg.vertices)],
            },
        })
    for poi in getattr(geom, "pois", []):
        feats.append({
            "type": "Feature",
            "properties": {"kind": "poi", "class": poi.class_name},
            "geometry": {"type": "Point", "coordinates": list(poi.point)},
        })
    return {"type": "FeatureCollection", "features": feats}


def main_sample_from_ckpt() -> None:
    """Plan 3 mode: load checkpoints, generate samples, dump PNGs + GeoJSON."""
    from bonzai_genai.models.configs import (
        DiTConfig,
        InkerConfig,
        RasterEncoderConfig,
        VAEConfig,
    )
    from bonzai_genai.training.lit_stage_a import LitStageA
    from bonzai_genai.training.lit_stage_b import LitStageB
    from bonzai_genai.training.lit_vae import LitVAE
    from bonzai_genai.training.samplers import dpmpp_sample, greedy_inker_sample
    from bonzai_genai.vocab.tokens import SpecialToken

    ckpt_dir = Path(os.environ["BONZAI_CKPT_DIR"])
    out_dir = Path(os.environ["BONZAI_SAMPLE_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = os.environ.get("BONZAI_PRESET", "plan3")
    n_samples = int(os.environ.get("BONZAI_NUM_SAMPLES", "64"))
    dpm_steps = int(os.environ.get("BONZAI_NUM_DPM_STEPS", "50"))
    inker_max = int(os.environ.get("BONZAI_INKER_MAX_TOKENS", "4096"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VAE from {ckpt_dir / 'vae.ckpt'}...", flush=True)
    vae_lit = LitVAE.load_from_checkpoint(
        str(ckpt_dir / "vae.ckpt"),
        vae_config=VAEConfig.from_preset(preset),
        map_location=device,
    )
    vae = vae_lit.vae.eval().to(device)

    print(f"Loading Painter from {ckpt_dir / 'stage_a.ckpt'}...", flush=True)
    sa_lit = LitStageA.load_from_checkpoint(
        str(ckpt_dir / "stage_a.ckpt"),
        dit_config=DiTConfig.from_preset(preset),
        vae_config=VAEConfig.from_preset(preset),
        map_location=device,
    )
    dit = sa_lit.dit.eval().to(device)

    print(f"Loading Writer from {ckpt_dir / 'stage_b.ckpt'}...", flush=True)
    sb_lit = LitStageB.load_from_checkpoint(
        str(ckpt_dir / "stage_b.ckpt"),
        inker_config=InkerConfig.from_preset(preset),
        raster_encoder_config=RasterEncoderConfig.from_preset(preset),
        map_location=device,
    )
    inker = sb_lit.inker.eval().to(device)
    raster_encoder = sb_lit.encoder.eval().to(device)

    vocab = load_default_vocab()
    tok = Tokeniser(vocab)
    bos = int(SpecialToken.BOS)
    eos = int(SpecialToken.EOS)

    print(f"Sampling {n_samples} latents via {dpm_steps}-step DPM-Solver++...", flush=True)
    latent_dim = VAEConfig.from_preset(preset).latent_dim
    with torch.no_grad():
        latents = dpmpp_sample(
            dit, batch_size=n_samples, num_steps=dpm_steps,
            latent_shape=(latent_dim, 64, 64), device=device,
        )
        # VAE decoder -> 9-channel rasters; sigmoid for binary channels
        recon_logits = vae.decoder(latents)
        rasters = torch.sigmoid(recon_logits)

    print(f"Rendering {n_samples} PNGs...", flush=True)
    for i in range(n_samples):
        _render_raster_png(rasters[i], out_dir / f"sample_{i:03d}.png")

    # Decode through the Writer in chunks via the KV-cached sampler. The
    # original loop called the recompute sampler at batch_size=1 — O(N³) per
    # sample, which made N=8192 unworkable (job 40942718 timed out at 1 h
    # with only 2 / 64 sequences decoded). The cached sampler is O(N) per
    # step; chunking caps peak KV memory while still letting many samples
    # share a single forward.
    chunk = int(os.environ.get("BONZAI_SAMPLE_CHUNK", "16"))
    print(
        f"Decoding {n_samples} tiles through Writer (cached greedy, chunk={chunk})...",
        flush=True,
    )
    from bonzai_genai.training.samplers import greedy_inker_sample_cached
    all_seqs: list[list[int]] = []
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        batch_tokens = greedy_inker_sample_cached(
            inker, raster_encoder, rasters[start:end],
            max_tokens=inker_max, bos_id=bos, eos_id=eos,
        )
        for row in range(batch_tokens.shape[0]):
            seq = batch_tokens[row].tolist()
            # Trim at first EOS so per-sample geometry decode isn't polluted
            # by post-EOS padding from longer-running siblings in the batch.
            if eos in seq:
                seq = seq[: seq.index(eos) + 1]
            all_seqs.append(seq)
        print(
            f"  decoded {end}/{n_samples}",
            flush=True,
        )
    for i, seq in enumerate(all_seqs):
        try:
            geom = tok.decode(list(seq))
            geojson = _geom_to_geojson(geom)
        except Exception as e:  # noqa: BLE001
            geojson = {
                "type": "FeatureCollection",
                "features": [],
                "decode_error": f"{type(e).__name__}: {e}",
            }
        (out_dir / f"sample_{i:03d}.geojson").write_text(json.dumps(geojson))

    print(f"Wrote {n_samples} PNGs and {n_samples} GeoJSON files to {out_dir}", flush=True)


def main() -> None:
    if os.environ.get("BONZAI_SAMPLE_FROM_CKPT") == "1":
        main_sample_from_ckpt()
    else:
        main_val_eval()


if __name__ == "__main__":
    main()
