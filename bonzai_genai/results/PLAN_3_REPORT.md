# Plan 3a — Sweden-Only Report

**Status:** Complete. Mixed result — architecture verified, end-to-end pipeline NOT verified.

**Spec:** [`docs/superpowers/specs/2026-05-04-plan-3-experiments-1-and-2-design.md`](../../docs/superpowers/specs/2026-05-04-plan-3-experiments-1-and-2-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-04-plan-3-experiments-1-and-2.md`](../../docs/superpowers/plans/2026-05-04-plan-3-experiments-1-and-2.md)
**Branch:** `genai-city-model`
**Cluster:** Leonardo Booster (CINECA), `boost_usr_prod`, 4×A100-64GB nodes
**Output dir on Leonardo:** `$WORK/bonzai-plan3a-sweden/`
**Sample dir (local):** [`bonzai_genai/results/plan3a-sweden-samples/`](plan3a-sweden-samples/) — 64 PNGs + 64 GeoJSONs
**Period:** 2026-05-05 → 2026-05-06

## What this run was

Plan 3a is the de-risked variant of Plan 3 — Sweden tiles only (1,301 records, 14 shards), no country balancing, reusing the Experiment 0 VAE checkpoint (TinyPreset, trained on synth) frozen as the latent decoder. The goal was a single question:

> Does the architecture (Painter EDM-DiT + Writer Inker) extract gradient signal from real OSM data?

Plan 3b (deferred) was to extend to Singapore + Sri Lanka and add country-conditioned CFG.

## What ran

| # | What | Job | State | Wall | Outcome |
|---|---|---|---|---:|---|
| 1 | Painter smoke (4-GPU DDP, fast_dev_run=1) | 40867181 | COMPLETED | 2:19 | train/loss 33.8 → went, no NaN |
| 2 | Writer smoke (B=4, fast_dev_run=1) | 40867650 | COMPLETED | 2:00 | train/loss 9.24 |
| 3 | Painter full (B=64, 50 epochs, LIMIT=200) | 40867649 | COMPLETED | 4:58:14 | train/loss **52.4 → 0.26** |
| 4 | Writer full (B=16) — first attempt | 40867981 | FAILED (OOM) | 2:03 | 8k self-attn matrix overran 64 GiB |
| 5 | Writer smoke (B=8) — diagnostic | 40872840 | FAILED (OOM) | 2:15 | confirmed B=8 also OOM |
| 6 | Writer full (B=4, 50 epochs, LIMIT=200) | 40873784 | COMPLETED | 2:22:40 | train/loss **6.25 → 0.0006** |
| 7 | Eval — sample-from-ckpt (1 GPU) | 40942718 | TIMEOUT | 1:10:09 | 64/64 PNGs, only 2/64 GeoJSONs |
| 8 | Eval — sample-from-ckpt (KV-cached) | 40955363 | COMPLETED | 0:30:40 | 64 PNGs + 64 GeoJSONs |

**Total node-h burned:** ≈ **9.5 / 50 budgeted** (19 %).

## Key code changes shipped this Plan

| Change | Commit | Why |
|---|---|---|
| `data_module.py` `.repeat()` on train stream | `c0e674e` | DDP ranks with uneven shard counts hung at end of epoch; required for `LIMIT_TRAIN_BATCHES` to bound epoch length |
| KV-cached greedy Inker sampler + `Inker.forward_step` / `cache_cross_kv` | `97c691c` | Slow recompute sampler (O(N³) per sample) made eval untractable — first eval timed out at 1 h with 2/64 GeoJSONs. Cached sampler runs the same eval in 30:40 |

Both are bit-equivalent in behaviour to what they replace (cached sampler is verified token-equal against the slow one in `tests/test_training_samplers.py`); the slow path stays available for the constrained-decoding follow-up.

## Loss trajectories

Both stages logged train/loss only (val_loss never written — separate gap; see "Known instrumentation gaps" below).

**Painter (EDM diffusion in latent space):**
- step 9: 52.38
- step 199 (end of epoch 0): 1.18
- step 1899 (epoch 9): 0.43
- step 5709 (epoch 28): 0.29
- step 9999 (epoch 49, final): **0.26**

→ ~200× reduction; clear, rapid, healthy decline followed by a typical EDM plateau in the 0.2–0.3 band.

**Writer (cross-entropy, teacher-forced):**
- step 9: 6.25
- step 1899 (epoch 9): 0.058
- step 3799 (epoch 18): 0.0021
- step 9999 (epoch 49, final): **0.0006**

→ Loss collapses to ~10⁻³ very quickly. With 1,301 tiles cycled ~30× through `.repeat()` against a 274 M model, this is full memorisation of the training set, not generalisation.

## Sample inspection

### Painter PNGs (64 unconditional)

Visual character: **faint, sparse strokes resembling rural Sweden marginals.** Channels behave as:
- ch0 / ch3 (road classes): scattered low-contrast pixels — look like noise patterns rather than coherent road graphs
- ch4 / ch7 (likely POIs / something_with_dots): occasional small dark blobs
- ch5 (land-use density heatmap): a uniform teal background with mild local variation — recognisably "low density Sweden"
- ch1, ch2, ch6, ch8: nearly all blank

Sample contrast: `sample_055.png` is essentially empty; `sample_020.png` shows the densest output (still no recognisable road network).

The Painter has clearly learned the **marginal distribution** of Sweden tiles (mostly empty, sparse roads, low POI count) but **not the structural patterns** (road topology, building footprints, layer relationships).

### Writer GeoJSONs (64 unconditional)

**0 / 64 decoded successfully.** Every emitted token sequence violates the tokeniser grammar. Top failure modes:

| Count | Error |
|---:|---|
| 17 | `token_id 526 not in y-coord range` (x-coord followed by non-y) |
| 7 | `expected LAYER_ROADS` (layer-order violation) |
| 5 | `token_id 13 not in y-coord range` |
| 3 | `token_id 7 not in x-coord range` |
| 32 | other token-position mismatches |

Despite the near-zero training loss, the unconditional Writer outputs are **grammatically invalid**. Cause: when conditioned on Painter-generated rasters (which look unlike any Sweden tile the Writer ever saw at training time), the Writer falls off-distribution and emits tokens at positions where the grammar forbids them.

Eval ran with `constrained=False` (the spec called for unconditional sample; constrained decoding is gated for Plan 4). Even the existing mandatory-subset constraints would have caught the x→y and layer-order errors — see "Decision-tree call" below.

## Decision-tree call

The spec §7.3 enumerates six diagnoses. None of the named six match cleanly. The actual finding is closest to a hybrid of two unanticipated modes:

> **Compound failure: Painter-Writer raster distribution mismatch + reused synth VAE.**

What's actually broken:

1. **VAE was trained on synth data** (Experiment 0, TinyPreset). Its decoder cannot render Sweden-realistic 9-channel rasters because it never saw any. Painter latents go through this synth-trained decoder and come out as "synth-rendered statistical-soup of Sweden marginals." Smokes and the spec assumed the synth VAE would transfer; it does not.
2. **Writer memorised** at the cost of generalising (loss → 6×10⁻⁴ on 1,301 tiles ≈ "I have memorised every answer key"). It can echo training tokens given training rasters, but on Painter-generated rasters it produces invalid grammars.
3. **`constrained=False` in eval** removed the only safety net.

Architecture verdict: ✅ gradient signal extracted (both loss curves are healthy and monotone-down). End-to-end verdict: ❌ not green; samples are not "cities."

**Selected next action:** Plan 3b, scoped not as "add SG + LK" but as the structural fix this run uncovered:

| Plan 3b deliverable | Cost (node-h, est.) |
|---|---:|
| Train a fresh VAE on Sweden+Singapore+Sri Lanka (1,888 tiles) — replaces synth VAE in the Painter pipeline | ~10 |
| Enable mandatory-subset constrained decoding in `run_eval.py` (already implemented; just flip the flag) + plumb the slow path's mask through cached sampler | ~0.5 |
| Add country-conditioning to Painter & Writer (text/tag CFG branch, dropout 10 %); retrain both | ~30 |
| Re-run sample-from-ckpt eval | ~0.5 |
| **Plan 3b total** | **~40** |

## Known instrumentation gaps surfaced this Plan

1. `metrics.csv` only contains `train/loss`. The Lightning modules' `validation_step` is wired but isn't `self.log("val/loss", ...)`-ing under DDP — likely missing `sync_dist=True` or the call itself. Without val/loss we cannot detect overfitting in-flight; we caught it only at sample time.
2. ModelCheckpoint runs at Lightning's default cadence (one ckpt per epoch, overwriting). Acceptable for ~3-5 h runs, insufficient for the 30 h chained Writer the spec assumed.
3. Spec batch-size estimate for Writer (16/rank → 64 effective) overflowed 64 GiB at 8 k context. Real headroom is **batch=4/rank → 16 effective**. Plan 3b should either add gradient checkpointing on the Inker blocks or shrink `max_context_len` to 4096 if 8 k isn't actually needed for Sweden's typical tile size.

## Hand-off

- Plan 3b spec to be drafted next, addressing the three structural fixes above (fresh VAE, constrained eval, country conditioning) plus extending to Singapore + Sri Lanka.
- Painter / Writer / VAE checkpoints retained at `$WORK/bonzai-plan3a-sweden/` on Leonardo for Plan 3b warm-starts (or for ablations against fresh-VAE replays).
- Sample artefacts kept locally for the eyeball record at [`plan3a-sweden-samples/`](plan3a-sweden-samples/).
