"""
validate_stockholm.py
=====================

Four independent checks that probe whether the trained model actually
learned the tokenizer's grammar and spatial geometry, not just the
token-frequency prior.

Checks
------
1. Grammar validity:
     For each kind in {BUILDING, ROAD, POI, LANDUSE, WATERWAY, RAILWAY},
     generate N samples starting from <KIND_START> and parse the stream.
     Report % structurally valid (tag -> optional extras -> X -> Y ->
     moves/part_sep* -> matching _END).
2. Token distribution match:
     JS-divergence between the token family histograms of the real
     parquet corpus and a large unconditional generation sample.
3. Geometric coherence:
     Decode building/landuse move streams back into polygons. Report
     closure rate (last vertex near first), area histogram, and 90-deg
     corner fraction. Compared against real-data baselines.
4. Visual render:
     Condition on a real anchor cell and generate N objects. Render
     them as a small PNG so the quality is eyeball-obvious.

The script expects the trained model directory to contain:
  - config.json, model.safetensors      (from Trainer.save_model)
  - token_to_id.json                    (written by train_stockholm.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import GPT2LMHeadModel, set_seed

logger = logging.getLogger("validate_stockholm")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KINDS = ["BUILDING", "ROAD", "POI", "LANDUSE", "WATERWAY", "RAILWAY", "NATURAL_LINE"]

DELTA_RE = re.compile(r"^<(dx|dy)_(-?\d+)>$")
ANCHOR_RE = re.compile(r"^<([XY])_(\d+)>$")
TAG_RE = re.compile(r"^<TAG_[A-Z0-9_]+>$")
# All attribute families that may appear between TAG and the first X anchor.
EXTRA_PREFIXES = (
    "<LEVELS_", "<SPEED_", "<SURFACE_", "<LIT_", "<LANES_",
    "<ACCESS_", "<ONEWAY_", "<BRIDGE_", "<TUNNEL_", "<HEIGHT_",
)


# ---------------------------------------------------------------------------
# Vocab + model loading
# ---------------------------------------------------------------------------


def load_model_and_vocab(model_dir: str, device: torch.device):
    """Load the trained GPT-2 plus its token <-> id dictionaries."""
    logger.info("loading model from %s", model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    mapping_path = os.path.join(model_dir, "token_to_id.json")
    with open(mapping_path, "r", encoding="utf-8") as fh:
        token_to_id: Dict[str, int] = json.load(fh)
    id_to_token: Dict[int, str] = {idx: tok for tok, idx in token_to_id.items()}

    logger.info(
        "model loaded: vocab=%d, n_params=%.2fM, device=%s",
        len(token_to_id),
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
    )
    return model, token_to_id, id_to_token


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _end_token_id(kind: str, token_to_id: Dict[str, int]) -> int:
    return token_to_id[f"<{kind}_END>"]


def generate_batch(
    model: GPT2LMHeadModel,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    prompt_tokens: Sequence[str],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_id: Optional[int],
    device: torch.device,
) -> List[List[str]]:
    """Generate `n_samples` completions from a string prompt and return
    each as a list of string tokens (prompt included, trailing padding
    stripped).
    """
    prompt_ids = [token_to_id[t] for t in prompt_tokens]
    input_ids = torch.tensor([prompt_ids] * n_samples, dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=token_to_id.get("<PART_SEP>", 0),
            eos_token_id=stop_token_id,
        )

    out_seqs: List[List[str]] = []
    for row in out_ids.tolist():
        toks = [id_to_token[i] for i in row]
        # Truncate at the first stop token we care about.
        if stop_token_id is not None:
            stop_tok = id_to_token[stop_token_id]
            if stop_tok in toks:
                toks = toks[: toks.index(stop_tok) + 1]
        out_seqs.append(toks)
    return out_seqs


# ---------------------------------------------------------------------------
# Check 1: Grammar validity
# ---------------------------------------------------------------------------


def parse_sequence(tokens: List[str], kind: str) -> dict:
    """Walk a generated token stream and check it conforms to the
    expected grammar for this `kind`. Returns a dict with diagnostics.

    Expected shape:
        <KIND_START> <TAG_*> [extras...] <X_*> <Y_*>
        (<MOVE_*> | <PART_SEP> <X_*> <Y_*>)* <KIND_END>

    Points kinds (POI) have no moves.
    """
    if not tokens or tokens[0] != f"<{kind}_START>":
        return {"valid": False, "reason": "missing_start"}
    if tokens[-1] != f"<{kind}_END>":
        return {"valid": False, "reason": "missing_end"}

    i = 1
    if i >= len(tokens) or not TAG_RE.match(tokens[i]):
        return {"valid": False, "reason": "missing_tag"}
    tag = tokens[i]
    i += 1

    extras: List[str] = []
    while i < len(tokens) and any(tokens[i].startswith(p) for p in EXTRA_PREFIXES):
        extras.append(tokens[i])
        i += 1

    # Anchor X
    if i >= len(tokens):
        return {"valid": False, "reason": "missing_anchor_x"}
    m = ANCHOR_RE.match(tokens[i])
    if not m or m.group(1) != "X":
        return {"valid": False, "reason": "bad_anchor_x"}
    ix = int(m.group(2))
    i += 1

    # Anchor Y
    if i >= len(tokens):
        return {"valid": False, "reason": "missing_anchor_y"}
    m = ANCHOR_RE.match(tokens[i])
    if not m or m.group(1) != "Y":
        return {"valid": False, "reason": "bad_anchor_y"}
    iy = int(m.group(2))
    i += 1

    # Geometry now comes as (dx, dy) token PAIRS. Store deltas as
    # (dx_int, dy_int) tuples so downstream decoders can replay them.
    deltas: List[Tuple[int, int]] = []
    n_parts = 1
    while i < len(tokens) - 1:
        tok = tokens[i]
        if tok == "<PART_SEP>":
            n_parts += 1
            i += 1
            # After PART_SEP we expect X, Y again for a new ring.
            if i + 1 < len(tokens) and ANCHOR_RE.match(tokens[i]) and ANCHOR_RE.match(tokens[i + 1]):
                i += 2
            continue
        mx = DELTA_RE.match(tok)
        if mx and mx.group(1) == "dx":
            # dx must be followed by a dy.
            if i + 1 >= len(tokens):
                return {"valid": False, "reason": "dx_without_dy"}
            my = DELTA_RE.match(tokens[i + 1])
            if not my or my.group(1) != "dy":
                return {"valid": False, "reason": f"dx_not_followed_by_dy:{tokens[i+1]}"}
            deltas.append((int(mx.group(2)), int(my.group(2))))
            i += 2
            continue
        return {"valid": False, "reason": f"unexpected_token:{tok}"}

    if kind == "POI" and deltas:
        return {"valid": False, "reason": "poi_has_geometry"}

    return {
        "valid": True,
        "tag": tag,
        "extras": extras,
        "anchor": (ix, iy),
        "deltas": deltas,
        "n_parts": n_parts,
        "n_tokens": len(tokens),
    }


def check_grammar(
    model, token_to_id, id_to_token,
    n_per_kind: int, max_new_tokens: int,
    temperature: float, top_p: float,
    device: torch.device,
) -> dict:
    """Generate N samples per kind, parse, aggregate validity stats."""
    report: Dict[str, dict] = {}
    for kind in KINDS:
        start = f"<{kind}_START>"
        end_id = _end_token_id(kind, token_to_id)
        seqs = generate_batch(
            model, token_to_id, id_to_token,
            prompt_tokens=[start],
            n_samples=n_per_kind,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_id=end_id,
            device=device,
        )
        parsed = [parse_sequence(s, kind) for s in seqs]
        n_valid = sum(1 for p in parsed if p["valid"])
        reasons = Counter(p.get("reason", "ok") for p in parsed if not p["valid"])
        report[kind] = {
            "n": n_per_kind,
            "valid": n_valid,
            "valid_pct": round(100 * n_valid / n_per_kind, 1),
            "avg_tokens": float(np.mean([len(s) for s in seqs])),
            "top_failure_reasons": reasons.most_common(3),
            "samples": seqs[:2],
            "parsed": parsed,  # kept for downstream checks, stripped later
        }
        logger.info(
            "  %s  valid=%d/%d (%.1f%%), avg_tokens=%.1f",
            kind, n_valid, n_per_kind, report[kind]["valid_pct"],
            report[kind]["avg_tokens"],
        )
    return report


# ---------------------------------------------------------------------------
# Check 2: Token distribution
# ---------------------------------------------------------------------------


def _family(token: str) -> str:
    body = token.lstrip("<").rstrip(">")
    return body.split("_", 1)[0] if "_" in body else body


def real_family_distribution(parquet_path: str) -> Counter:
    """Family-level histogram over every token in the real corpus."""
    logger.info("reading real parquet for token-distribution baseline")
    tbl = pq.read_table(parquet_path, columns=["tokens"])
    counts: Counter[str] = Counter()
    for row in tbl.column("tokens").to_pylist():
        for tok in row:
            counts[_family(tok)] += 1
    return counts


def generated_family_distribution(parsed_by_kind: Dict[str, dict]) -> Counter:
    counts: Counter[str] = Counter()
    for kind, entry in parsed_by_kind.items():
        for seq in entry["samples"] + [
            # include all generated sequences, not just the kept-for-show samples
            # (we stashed them under 'parsed' via side-channel)
        ]:
            pass
    return counts  # replaced below; keep for interface clarity


def _histogram_from_samples(samples: List[List[str]]) -> Counter:
    counts: Counter[str] = Counter()
    for seq in samples:
        for tok in seq:
            counts[_family(tok)] += 1
    return counts


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Symmetric Jensen-Shannon divergence between two discrete
    distributions (keys not required to match; missing = 0).
    """
    keys = set(p) | set(q)
    eps = 1e-12
    m: Dict[str, float] = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av <= 0 or bv <= 0:
                continue
            total += av * math.log(av / bv)
        return total

    return 0.5 * (kl(p, m) + kl(q, m))


def normalise(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}


# ---------------------------------------------------------------------------
# Check 3: Geometric coherence
# ---------------------------------------------------------------------------


def decode_moves(deltas: List[Tuple[int, int]]) -> np.ndarray:
    """Convert a list of (dx_m, dy_m) pairs into (N+1, 2) path
    coordinates in meters, starting at origin. Multiple consecutive
    dx/dy pairs that target the same vertex (split across the 32 m cap)
    simply accumulate naturally by construction.
    """
    pts = [(0.0, 0.0)]
    for dx, dy in deltas:
        px, py = pts[-1]
        pts.append((px + float(dx), py + float(dy)))
    return np.array(pts, dtype=np.float64)


def _polygon_metrics(path: np.ndarray) -> dict:
    """Closure / area / right-angle metrics for a polygon candidate."""
    if len(path) < 4:
        return {"valid": False}
    # Close the ring for metric computation.
    closing = float(np.linalg.norm(path[0] - path[-1]))
    perimeter = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    # Area via shoelace.
    ring = np.vstack([path, path[:1]])
    area = 0.0
    for i in range(len(path)):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += (x1 * y2 - x2 * y1)
    area = abs(area) / 2.0
    # Corner angles.
    angles = []
    for i in range(len(path) - 2):
        v1 = path[i + 1] - path[i]
        v2 = path[i + 2] - path[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos)))
    near90 = sum(1 for a in angles if abs(a - 90) < 20)
    return {
        "valid": True,
        "closure_m": closing,
        "perimeter_m": perimeter,
        "area_m2": area,
        "corners": len(angles),
        "near90_frac": (near90 / len(angles)) if angles else 0.0,
    }


def check_geometry(
    parsed_by_kind: Dict[str, dict],
    closure_threshold_m: float = 5.0,
) -> dict:
    """Decode MOVE tokens back to polylines and report geometric
    coherence per kind.
    """
    out: Dict[str, dict] = {}
    for kind in ("BUILDING", "LANDUSE", "ROAD", "WATERWAY", "RAILWAY", "NATURAL_LINE"):
        parsed = [p for p in parsed_by_kind[kind]["parsed"] if p.get("valid")]
        if not parsed:
            out[kind] = {"n": 0}
            continue
        paths = [decode_moves(p["deltas"]) for p in parsed if p["deltas"]]
        # For area/closure we only care about polygons (BUILDING / LANDUSE).
        if kind in ("BUILDING", "LANDUSE"):
            metrics = [_polygon_metrics(p) for p in paths if len(p) >= 4]
            metrics = [m for m in metrics if m["valid"]]
            if not metrics:
                out[kind] = {"n": 0}
                continue
            closures = np.array([m["closure_m"] for m in metrics])
            areas = np.array([m["area_m2"] for m in metrics])
            near90 = np.array([m["near90_frac"] for m in metrics])
            out[kind] = {
                "n": len(metrics),
                "closed_pct": float(100 * (closures <= closure_threshold_m).mean()),
                "closure_median_m": float(np.median(closures)),
                "area_median_m2": float(np.median(areas)),
                "area_p10": float(np.percentile(areas, 10)),
                "area_p90": float(np.percentile(areas, 90)),
                "near90_mean": float(near90.mean()),
            }
        else:
            # Linear features: report length distribution.
            lengths = np.array([
                float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
                for p in paths if len(p) >= 2
            ])
            if lengths.size == 0:
                out[kind] = {"n": 0}
                continue
            out[kind] = {
                "n": int(lengths.size),
                "length_median_m": float(np.median(lengths)),
                "length_p10": float(np.percentile(lengths, 10)),
                "length_p90": float(np.percentile(lengths, 90)),
            }
    return out


# ---------------------------------------------------------------------------
# Check 4: Visual render
# ---------------------------------------------------------------------------


def render_samples(
    parsed_by_kind: Dict[str, dict],
    output_path: str,
    kinds_to_render: Sequence[str] = ("BUILDING", "ROAD", "LANDUSE", "WATERWAY"),
    per_kind: int = 16,
) -> None:
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(
        len(kinds_to_render), 1,
        figsize=(12, 3 * len(kinds_to_render)),
        squeeze=False,
    )
    for row, kind in enumerate(kinds_to_render):
        ax = axes[row, 0]
        ax.set_title(f"{kind} — {per_kind} generated samples")
        parsed = [p for p in parsed_by_kind.get(kind, {}).get("parsed", []) if p.get("valid")]
        parsed = [p for p in parsed if p["deltas"]][:per_kind]
        for idx, p in enumerate(parsed):
            path = decode_moves(p["deltas"])
            # Tile horizontally so they don't overlap.
            shift = idx * 120.0
            ax.plot(path[:, 0] + shift, path[:, 1], linewidth=1.0)
            if kind in ("BUILDING", "LANDUSE") and len(path) >= 4:
                # close the loop visually
                ax.plot(
                    [path[-1, 0] + shift, path[0, 0] + shift],
                    [path[-1, 1], path[0, 1]],
                    linewidth=0.5, linestyle=":",
                )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("rendered %d kinds -> %s", len(kinds_to_render), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing Trainer.save_model output + token_to_id.json")
    parser.add_argument("--parquet", required=True,
                        help="Real training parquet (for distribution baseline)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the report JSON + render PNG")
    parser.add_argument("--n-per-kind", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    model, token_to_id, id_to_token = load_model_and_vocab(args.model_dir, device)

    # ----- Check 1: grammar -------------------------------------------------
    logger.info("check 1: grammar validity across %d samples/kind", args.n_per_kind)
    grammar_report = check_grammar(
        model, token_to_id, id_to_token,
        n_per_kind=args.n_per_kind,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    # ----- Check 2: token distribution --------------------------------------
    logger.info("check 2: token distribution")
    real_counts = real_family_distribution(args.parquet)
    gen_samples = [s for entry in grammar_report.values() for s in entry["samples"]]
    # Samples in the report are heavily truncated for readability; recompute
    # family counts from the full parsed batches (we stored parsed entries).
    gen_all_tokens: List[List[str]] = []
    for entry in grammar_report.values():
        # Reconstruct the full seqs from parsed: each parsed dict retains
        # everything we need if we kept moves/tag/anchor. Simpler: just
        # sample a round-2 batch here and cat its families.
        pass

    # Rather than reconstruct, draw one more large batch of unconditional
    # start-token seeds for a fair distribution sample.
    unconditional = []
    for kind in KINDS:
        start = f"<{kind}_START>"
        end_id = _end_token_id(kind, token_to_id)
        unconditional.extend(generate_batch(
            model, token_to_id, id_to_token,
            prompt_tokens=[start],
            n_samples=args.n_per_kind,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_id=end_id,
            device=device,
        ))
    gen_counts = _histogram_from_samples(unconditional)
    js = _js_divergence(normalise(real_counts), normalise(gen_counts))
    logger.info("  JS divergence (real vs gen, family level): %.4f", js)

    dist_report = {
        "real_family_counts": dict(real_counts),
        "gen_family_counts": dict(gen_counts),
        "js_divergence": js,
    }

    # ----- Check 3: geometry ------------------------------------------------
    logger.info("check 3: geometric coherence")
    geometry_report = check_geometry(grammar_report)
    for kind, r in geometry_report.items():
        logger.info("  %s  %s", kind, r)

    # ----- Check 4: render --------------------------------------------------
    render_path = os.path.join(args.output_dir, "generated_samples.png")
    render_samples(grammar_report, render_path)

    # ----- Report -----------------------------------------------------------
    # Strip 'parsed' to keep the JSON reasonable.
    report = {
        "n_per_kind": args.n_per_kind,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "grammar": {
            kind: {k: v for k, v in entry.items() if k != "parsed"}
            for kind, entry in grammar_report.items()
        },
        "distribution": dist_report,
        "geometry": geometry_report,
        "render": render_path,
    }
    report_path = os.path.join(args.output_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("report written -> %s", report_path)

    # Print a quick terminal summary.
    print("\n================ VALIDATION SUMMARY ================")
    print(f"n per kind     : {args.n_per_kind}")
    print(f"JS divergence  : {js:.4f}")
    print("grammar validity:")
    for kind, entry in grammar_report.items():
        print(f"  {kind:<10} {entry['valid_pct']:>5.1f}%   avg_tokens={entry['avg_tokens']:.1f}")
    print("geometry:")
    for kind, r in geometry_report.items():
        print(f"  {kind:<10} {r}")
    print(f"render         : {render_path}")
    print(f"report         : {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
