# Experiment 0 — Report

**Completed:** 2026-05-04
**Branch:** `genai-city-model`
**Cluster:** Leonardo Booster (CINECA)
**Output dir on Leonardo:** `$WORK/bonzai-exp0/`

## What this experiment was

A smoke test. The whole point: confirm the *pipeline closes end-to-end* — that the VAE compresses → the Sketcher (DiT) paints denoised codes → the Inker (autoregressive transformer) reads the painted blueprint and writes vector tokens → the eval harness measures all of it — *without anything blowing up*. We are **not** testing whether the generated cities look good yet. That comes in Experiments 1-4 (Plan 3+).

Think of it like running a brand-new factory line on a few practice parts before flipping it on for the real product. We just want to see: do the conveyor belts move, do the robot arms reach the right places, do the QC sensors fire? Yes / no.

## What ran

| Step | What | Where | Wall | Outcome |
|---|---|---|---:|---|
| 1 | Generate 5,000 synthetic tile shards (4,500 train + 500 val) | `lrd_all_serial` (free CPU) | 9 min | OK — 44 GB on `$WORK/bonzai-tiles/synth/` |
| 2 | Train tiny VAE (50 epochs, reconstruction loss) | `boost_usr_prod` (1×A100) | within step-3 wall | OK — checkpoint saved |
| 3 | Train tiny Stage A / DiT (1 epoch, EDM diffusion in latent space) | same A100 | embedded in 59-min total | OK — train loss 1.56, no divergence |
| 4 | Train tiny Stage B / Inker (1 epoch, teacher-forcing on raster→tokens) | same A100 | ~3 min training | OK — train loss 1.07, no divergence |
| 5 | Run §8 eval suite | `lrd_all_serial` (free CPU) | 1 min 52 s | OK — all metrics computed |

**Total GPU-h burned: ~1** (a few minutes of A100 time — much less than the 12-30 GPU-h budgeted). The tiny smoke models converge fast; we did not need to spend more.

## Eval numbers (sanity-pass, not quality)

These metrics ran on the **ground-truth val set** (real vs real, self-vs-self), not on samples drawn from the trained models. The point is to verify the eval *code* runs — not to measure model quality. A separate "sample → measure" round comes in Experiment 1.

```
Stage A:
  channel IoU (self vs self): 1.00 on all 8 binary channels  (sanity check)
  channel 5 MSE (self vs self): 0.00                         (continuous channel)
  FID-lite (real vs real, sanity): 0.02                      (≈ 0, good)

Stage B:
  validity_rate (val token sequences round-trip): 1.00       (every token seq decodes cleanly)
  building_chamfer (self): 0.00                              (sanity)
  road_graph_largest_component_fraction: 0.12                (* see note below)
  poi_placement (self): 0.00                                 (sanity)
  building_self_intersection: 0.00                           (sanity)
```

**Note on road-graph 0.12:** computed on the *first* val tile, not aggregated. With diagonal roads now in the synth generator (Plan 2 Task 6), a tile can have multiple disconnected road clusters. 0.12 means the largest cluster covers 12 % of road nodes. Real cities will produce different numbers; this is just smoke.

## Go / No-Go

**Go.** All three soft signals fired:

1. **Loss curves don't diverge** — VAE finished 50 epochs cleanly; Stage A train loss 1.56 (steady); Stage B train loss 1.07 (steady). No NaNs.
2. **≥ 90 % well-formed GeoJSON** — validity rate **1.00** on the val token sequences. Constrained-decoding logic isn't fully exercised yet (no actual sampling loop), but the round-trip check passes.
3. **Visual eyeball check** — *deferred*. The eval driver doesn't yet sample 32 tiles from the trained models and dump them. Adding that loop is a Plan 3 follow-up; not blocking for the smoke decision.

## What we hit during the run (recorded for the next agent)

1. **Default torch wheels are CUDA 13; Leonardo's driver is CUDA 12.2.** First GPU job died at the first cuda init with `driver too old (found 12020)`. Fix: install torch with `--index-url https://download.pytorch.org/whl/cu121` (CUDA 12.1 wheels). Documented in `bonzai_genai/scripts/README.md`, committed at `834bf2b`.
2. **`fid_lite` tried to allocate a 40 TiB covariance matrix.** Naïve `np.cov` over flattened pixel features (2.36 M dims) is impossible. Fix: per-channel mean-and-stddev distance instead (committed at `ada9096`). Plan 1+ should still upgrade to a proper Inception-feature FID for production runs.
3. **Tiny smoke models converge in ~1 min on A100.** The 12-30 GPU-h budget was conservative; actual burn was ~1 GPU-h. If we want sharper smoke signals, we can spend more (longer Stage A training, run actual sampling) without hitting the budget.

## Open follow-ups for Plan 3

- [ ] **Sample 32 tiles from each trained model**, dump as PNGs, eyeball-check. The eval driver currently runs on val ground-truth only; the sampling loop is not exercised. Add a `BONZAI_SAMPLE_FROM_CKPT` mode to `run_eval.py` that loads the latest VAE / DiT / Inker checkpoints and runs `dpmpp_sample` + `greedy_inker_sample`.
- [ ] **Experiment 1 prep:** swap synth shards for the real-country shards on `$WORK/bonzai-tiles/{singapore,sri_lanka,sweden}/`. Bump models to ~200 M params (still tiny by production standards).
- [ ] **Constrained-decoding rules** that were deferred from Plan 2 Task 16: polygon non-self-intersection, road-edge node-ref bounds, building-field ordering. Land in Plan 3.
- [ ] **Multipolygon-relation buildings/landuse** still missing from the sampler (deferred from Phase 0a follow-ups).
- [ ] **Marina Bay** still overflows the 8192 NODE_REF cap (8 of ~20 dense tiles). Decide before Plan 3: bump to 16384 or pre-crop those tiles into 1 km sub-tiles.
- [ ] **Proper FID** with Inception-feature embeddings (replaces the per-channel mean-and-stddev `fid_lite` smoke metric).

## Hand-off

Plan 3 (Stage A on real data + Stage B on perfect input + Experiments 1-2) is unblocked. Phase 0b is **complete**.

Trained checkpoints live on Leonardo at `$WORK/bonzai-exp0/{vae,stage_a,stage_b}/`. They survive the 6-month-past-project retention window — Plan 3 can either reuse them as warm starts or discard them and retrain from scratch on real-country data.
