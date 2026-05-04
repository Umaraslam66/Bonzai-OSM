"""Eval driver invoked by both Experiment 0 and standalone eval jobs."""
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


def main() -> None:
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

    # Stage A: smoke sanity (real-vs-real on the val set).
    iou = channel_iou(val_rasters[:32], val_rasters[:32])
    fid = fid_lite(val_rasters[:32], val_rasters[32:64]) if val_rasters.shape[0] >= 64 else 0.0
    results["stage_a"] = {
        "channel_iou_self": iou,
        "fid_lite_real_vs_real": float(fid),
    }

    # Stage B: validity rate over val token sequences.
    val_rate = validity_rate(val_tokens_lists[:32], vocab=vocab)
    results["stage_b"] = {"validity_rate_val_tokens": val_rate}

    # Decode + Chamfer + road graph + POI + self-intersection on first 4 val tiles.
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

    # Build human-readable report
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
        f"- Validity ≥ 90%: {'PASS' if pass_validity else 'NEEDS REVIEW'}"
    )
    (out_dir / "EXPERIMENT_0_REPORT.md").write_text("\n".join(report))
    print("Wrote", out_dir / "EXPERIMENT_0_REPORT.md")


if __name__ == "__main__":
    main()
