# Plan 3 — Experiments 1 + 2 — Lean Real-Data De-Risking

> Sub-spec scoped to Phase 1 first half. Refers up to the global design at
> [`2026-05-03-genai-city-infrastructure-design.md`](2026-05-03-genai-city-infrastructure-design.md)
> and the Phase 0b spec at
> [`2026-05-04-phase-0b-modeling-and-smoke-harness-design.md`](2026-05-04-phase-0b-modeling-and-smoke-harness-design.md).
> Phase 0b (modeling layer + smoke run) is complete; this spec extends the
> existing `bonzai_genai` package with the smallest set of code changes
> needed to run Experiments 1 and 2 on the real Sweden + Singapore + Sri
> Lanka shards already on Leonardo.

**Branch:** `genai-city-model` (long-lived; never merge to main)
**Allocation:** `AIFAC_P02_222` (~50% project window elapsed, 2026-05-04)
**Cost target:** ~54 node-h ≈ 1,728 core-h ≈ 4.3 % of the 40k allocation.

---

## 1. Goal

Run **Experiment 1** (Painter on real Sweden + Singapore + Sri Lanka) and
**Experiment 2** (Writer on the same three countries' ground-truth rasters)
with the leanest possible code surface, to surface the first gradient
signal from real data. The goal is **not** a production-quality model — it
is evidence about which of several known failure modes actually fires when
the architecture meets real OSM data.

## 2. Why "lean" is the right shape

Phase 0a delivered an imbalanced corpus (Sweden 1,301 / Sri Lanka 384 /
Singapore 203 = 1,888 tiles total). We have enumerated ~8 ways to improve
that corpus (skip rule, NODE_REF cap, lat/lon distortion, H3
stratification, multipolygon support, density / Köppen annotation, etc.).
**Pre-solving all of them costs ~3 days of free-CPU prep and produces no
evidence.** Burning ~50 node-h to surface the actual failure mode produces
evidence and points at the right fix.

The architecture is robust at scale — the corpus is what is fragile,
deliberately, because de-risking is meant to be cheap. Each deferred item
has a known fix; the experiment tells us which one to land first. See §7
for the decision tree.

## 3. Scope

### In scope (Plan 3 ships)

- Bump `TinyConfig` → `Plan3Config` for Painter (~200 M params) and Writer
  (~300 M params). VAE reused from Phase 0b smoke, frozen.
- Add `WeightedRandomSampler` over country in the existing
  LightningDataModule (one weight per tile, `1 / count_in_country`).
- Wire 4-GPU DDP into `lit_stage_a.py` and `lit_stage_b.py`
  (`Trainer(devices=4, strategy="ddp", sync_batchnorm=True)`).
- TDD test verifying WebDataset + DDP shard splitting (no shard read by
  two ranks).
- Add `BONZAI_SAMPLE_FROM_CKPT` mode to `eval/run_eval.py`: load
  Painter / Writer / VAE checkpoints, run DPM-Solver++ + greedy decode,
  dump 64 PNGs + 64 GeoJSON files for inspection.
- New Slurm template `leonardo_plan3.sbatch` for `boost_usr_prod`,
  4×A100, 24-h walls, chained jobs.
- Smoke validation on Leonardo (1 epoch, 4 GPUs, both stages) before
  launching real training.
- Run Exp 1 + Exp 2 in parallel as two independent Slurm jobs.
- `bonzai_genai/results/PLAN_3_REPORT.md` with decision-tree outcome.
- Update `STATUS.md` + `PROJECT.md` with Plan 3 hand-off.

### Out of scope (deferred — re-trigger only on red eval signal)

- Coastline skip-rule fix (`<5 features` rule kills `>50 % water` tiles).
- Marina Bay NODE_REF cap lift (8,192 → 16,384).
- Sri Lanka `max_tiles` bump (2k → 6k).
- Lat/lon → UTM tile geometry (Sweden's ~30–50 % N-S area distortion).
- H3 cell-id annotation for stratification metadata.
- Multipolygon-relation support in the OSM sampler.
- Density bucket / Köppen / primary land-use per-tile annotation.
- Constrained-decoding rules deferred from Plan 2 Task 16: polygon
  non-self-intersection, road-edge node-ref bounds, building-field
  ordering.
- Experiment 3 (end-to-end domain gap) — Plan 4.
- Experiment 4 (tile stitching) — Plan 5.
- EasyControl LoRA conditioning training — Plan 6+.
- Beam-search decoding for Writer — greedy only in Plan 3.

Each item above has a known fix mapped to a specific failure signal in §7.

## 4. Module changes

```
bonzai_genai/
├── src/bonzai_genai/
│   ├── models/configs.py              ← + Plan3Config (Painter 200M, Writer 300M)
│   ├── training/data_module.py        ← + WeightedRandomSampler over country
│   ├── training/lit_stage_a.py        ← + DDP-aware logging (sync_dist=True)
│   ├── training/lit_stage_b.py        ← + DDP-aware logging
│   └── eval/run_eval.py               ← + BONZAI_SAMPLE_FROM_CKPT mode
├── scripts/
│   ├── leonardo_plan3.sbatch          ← NEW (boost_usr_prod, 4×A100, 24h)
│   └── README.md                      ← + Plan 3 command notes
└── tests/
    ├── test_data_module_ddp.py        ← NEW (no shard double-read across ranks)
    ├── test_models_plan3_config.py    ← NEW (param count sanity)
    └── test_eval_sample_from_ckpt.py  ← NEW (driver smoke on tiny checkpoints)
```

**Estimated diff:** ~400 LoC code + ~150 LoC tests, ~12–15 plan tasks.

## 5. Sizing — `Plan3Config`

| Component | Plan3Config | Phase 0b smoke | Production (Phase 4–5) |
|---|---|---|---|
| VAE | reuse Phase 0b smoke (frozen) | 5 M | 10 M |
| Painter (DiT) | **200 M**, 16 layers, hidden 768, 12 heads, patch 2 over 64×64 latent | 50 M | 400 M |
| Writer (Inker) | **300 M**, 16 layers, hidden 1024, 16 heads, ctx 8 k tokens, RoPE | 50 M | 750 M |
| Raster CNN encoder | 15 M, 4 strided conv, output 32×32×512 | 5 M | 30 M |

VAE reuse is intentional: training a new VAE is a separate exercise (Phase 3).
Reconstruction quality of the smoke VAE on **real** tiles is unmeasured;
verifying it during smoke is part of §6.0.

## 6. Training plan

### 6.0 Smoke pass (1 GPU-h, blocking)

Before launching real training, run a 1-epoch DDP smoke on the existing
synth shards to verify:

1. WebDataset + DDP path: each rank sees a disjoint shard subset.
2. `Plan3Config` instantiates and forwards on 4×A100 without OOM.
3. Lightning logging works under DDP (loss curves on rank 0 only, no NaN).
4. Reused smoke VAE produces sane reconstructions of one real-tile batch.

Pass: no crash, no NaN, sample images decode. Fail → fix before real run.

### 6.1 Painter (Experiment 1)

| Item | Value |
|---|---|
| Loss | EDM diffusion in latent space (VAE frozen) |
| Optimiser | AdamW, lr 1e-4, cosine decay, weight decay 0.01 |
| Epochs | 50 over the 1,888-tile real corpus |
| Sampler | `WeightedRandomSampler(weights=1/country_count, replacement=True)` |
| Conditioning | Country tag only (text + Köppen + density deferred) |
| CFG dropout | 10 % (drops conditioning to null) |
| In-loop sampling | 32 tiles via DPM-Solver++ every 1,000 steps; dumped to logs |
| DDP | 4 GPUs, batch 64 per rank → effective 256 |
| Checkpointing | every 30 min to `$WORK/bonzai-plan3/painter/` |
| Wall | ~20 h on 4×A100 = ~20 node-h |

### 6.2 Writer (Experiment 2)

| Item | Value |
|---|---|
| Loss | Cross-entropy on next-token, teacher-forcing |
| Optimiser | AdamW, lr 3e-4, cosine decay |
| Cross-attention | To **ground-truth** raster (no domain gap — that's Exp 3) |
| Epochs | 50 over the same 1,888 (raster, tokens) pairs |
| Sampler | Same `WeightedRandomSampler` over country |
| Constrained decoding | Plan 2 mandatory subset only — paired x/y, polygon closure, layer order |
| DDP | 4 GPUs, batch 16 per rank → effective 64 (memory-bounded by 8k context) |
| Checkpointing | every 30 min to `$WORK/bonzai-plan3/writer/` |
| Wall | ~30 h on 4×A100 = ~30 node-h (chained two 24-h slots, restart-on-checkpoint) |

### 6.3 Orchestration

Two Slurm jobs submitted in parallel after the smoke pass clears. They
share no state — safe to run concurrently. End-to-end wall ~30 h
(Writer-bound).

## 7. Eval — sample-from-checkpoint + decision tree

### 7.1 Sample-from-checkpoint driver (new in Plan 3)

`BONZAI_SAMPLE_FROM_CKPT=1 python -m bonzai_genai.eval.run_eval`:

1. Load latest Painter / Writer / VAE checkpoints.
2. Generate 64 samples — 16 unconditional + 48 conditional (16 per
   country tag).
3. Decode Painter's denoised latents through the VAE → 9-channel rasters
   → render PNGs.
4. Pipe each Painter raster through the Writer (greedy + constrained
   decoding) → token sequence → GeoJSON.
5. Run the §8 metric suite (already implemented in Phase 0b).
6. Dump artefacts to `bonzai_genai/results/plan3-samples/`.

This is the artefact that closes the deferred follow-up from
`EXPERIMENT_0_REPORT.md` (Phase 0b's eval ran on val ground-truth, not on
samples). **Without this driver Plan 3 has nothing to look at.**

### 7.2 Eyeball check

Render a 4×4 grid of unconditional samples + 3 separate 4×4 grids of
per-country samples. Embed in `PLAN_3_REPORT.md`. PI judgment whether the
samples "look like cities."

### 7.3 Decision tree

| Eval signal | Diagnosis | Fix | Cost |
|---|---|---|---:|
| Both green per global §10 Exp 1+2 criteria | Architecture extracts signal from real data | Proceed to Plan 4 (Exp 3) + Plan 5 (Exp 4) | — |
| Writer self-intersection > 30 % | Constrained decoding insufficient | Land deferred Plan 2 rules; retrain Writer | ~30 node-h |
| Painter samples are pure noise | 200 M too small | Bump to 400 M (production size); same code | ~50 node-h |
| Painter Singapore-conditioning ≈ Sweden-conditioning | Corpus imbalance overpowering CFG | Re-prep with NODE_REF 16k + Sri Lanka 6k; retrain | ~1 d prep + 30 node-h |
| Painter loses coastal morphology | Skip rule killed coastlines | Fix `<5 features` rule; re-prep; retrain | ~1 d + 30 node-h |
| Painter samples are visibly N-S squashed | Lat/lon distortion biting | Switch to UTM-projected tile geometry; re-prep; retrain | ~1 d + 30 node-h |

Each red signal points to a specific known fix. We re-trigger Plan 3 with
**that fix only**, not all of them simultaneously.

## 8. Slurm template (illustrative)

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/plan3-%x-%j.out
#SBATCH --error=logs/plan3-%x-%j.err

source .venv/bin/activate
srun python -m bonzai_genai.training.fit \
    --stage "$BONZAI_STAGE" \
    --config plan3 \
    --data $WORK/bonzai-tiles \
    --out $WORK/bonzai-plan3
```

Writer chained via `--dependency=afterok` for its 30-h wall (one slot is
24 h max). Resume from latest checkpoint on restart.

## 9. Budget

| Item | Burn (node-h) | Cumulative |
|---|---:|---:|
| Smoke pass on Leonardo | 1 | 1 |
| Exp 1 Painter | ~20 | 21 |
| Exp 2 Writer | ~30 | 51 |
| Sample-from-ckpt + eval | ~3 | 54 |
| **Total Plan 3** | **~54** | |

54 node-h × 32 core-h/node ≈ **1,728 core-h ≈ 4.3 %** of the 40k
allocation. Cumulative project burn after Plan 3: ~16 + 54 = ~70 node-h
≈ 5.6 % of the 1,250 GPU-h initial budget. Well inside.

## 10. Success criteria

**Plan 3 ships when:**

- All Phase A code lands with passing tests + ruff-clean.
- 4-GPU DDP smoke run completes on Leonardo without crash.
- Both training jobs run to completion (Painter 50 epochs, Writer 50
  epochs) without divergence.
- Sample-from-checkpoint driver produces 64 PNGs + 64 GeoJSON files.
- `PLAN_3_REPORT.md` is committed with the decision-tree outcome (green
  / yellow / red call per row of §7.3).
- `STATUS.md` + `PROJECT.md` updated with hand-off pointer.

**Plan 3 does NOT ship if:**

- Either training job NaNs / diverges and 50 % more compute can't fix
  it. (No-go signal — diagnose root cause; do not paper over.)
- Sample-from-checkpoint driver fails to produce parseable GeoJSON for
  > 50 % of samples. (Means the Writer's output is broken — a structural
  bug, not a quality issue.)

## 11. Open questions / risks

1. **WebDataset + DDP shard splitting.** Lightning's `nodesplitter`
   should handle this but multi-GPU has never run on this codebase.
   Mitigation: TDD test in `test_data_module_ddp.py` before launching
   real training.
2. **DDP-aware Lightning logging.** `self.log()` needs `sync_dist=True`
   for cross-rank averaging. Easy to forget. Mitigation: code-review
   pass; smoke run will surface logging glitches.
3. **Singapore overfit.** With ~200 unique Singapore tiles and the
   class-balanced sampler oversampling per epoch, the Painter sees each
   Singapore tile ~325 times by epoch 50. Risk of memorisation visible
   in samples. Acceptable for de-risking; flag in `PLAN_3_REPORT.md` if
   it appears.
4. **Reused smoke VAE quality on real tiles unmeasured.** §6.0 smoke
   pass adds a 1-batch reconstruction visualisation as the gate. If
   reconstructions are visibly poor, retrain VAE on real shards before
   launching Painter (~1 GPU-h adder).
5. **8-k Writer context vs real-tile token length.** Phase 0a tile token
   counts spanned ~5k–14k. The 8-k context will truncate the longest
   tiles. Mitigation: add a tile-length histogram to the smoke-pass
   output; flag if > 5 % of tiles overflow.

## 12. Self-review

- **Placeholders:** none. All numbers concrete or marked open.
- **Internal consistency:** §6 batch sizes match §9 burn estimates;
  §4 module changes match §6 / §7 file references; §7.3 decision tree
  rows map to §3 deferred items.
- **Scope check:** focused on lean Exp 1 + Exp 2 only. Explicit deferral
  list (§3) keeps scope bounded; decision tree (§7.3) keeps post-Plan-3
  fixes one at a time.
- **Ambiguity:** "looks like cities" (§7.2) is intentionally subjective
  and labelled as PI judgment. Numeric pass criteria live in global §10
  Exp 1 / Exp 2.
