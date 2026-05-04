# Phase 0b — Modeling Layer + Eval Harness + Experiment 0 Smoke Run

> Sub-spec scoped to Phase 0b. Refers up to the global design at
> [`2026-05-03-genai-city-infrastructure-design.md`](2026-05-03-genai-city-infrastructure-design.md).
> Phase 0a (data prep + tile shards) is complete; this spec extends the
> `bonzai_genai` package with the modeling and eval surface needed to
> run Experiment 0 and to unblock production training in Phases 4–5.

**Branch:** `genai-city-model` (long-lived; never merge to main)
**Allocation:** `AIFAC_P02_222`, ~50% project window elapsed (2026-05-04)
**Phase 0b cost target:** ~10 GPU-h Experiment 0 burn; up to ~25 if convergence needs it (no artificial cap, per "quality over thrift").

---

## 1. Goal

Add the model code, training loops, and eval harness needed to (a) run **Experiment 0** end-to-end on synthetic data and (b) be ready to launch Experiment 1 on Phase 0a's real-country tile shards without further code work. Confirm the architecture skeleton — VAE → DiT (Sketcher) → AR Inker with cross-attention to a raster CNN encoder, plus constrained decoding — closes geometrically before we commit production GPU-h.

## 2. Scope

### In scope (Plan 2 will deliver)

- Custom-from-scratch VAE, DiT (Sketcher), AR Inker (Stage B), and small CNN raster encoder. **Built in raw PyTorch — no `diffusers` / no `transformers` model imports** (decision logged 2026-05-04). Each model is config-driven with a "tiny" preset for Experiment 0 and a "production" preset for Phases 4–5.
- PyTorch Lightning training loops: `lit_vae.py`, `lit_stage_a.py`, `lit_stage_b.py`, sharing one `LightningDataModule` over the WebDataset shards.
- Full §8 eval harness from the global spec: Stage A metrics, Stage B metrics, end-to-end, and §8.2 baselines.
- Slurm GPU job templates for VAE / Stage A / Stage B / eval / Experiment 0 driver.
- Extension of `synth/procedural.py` to produce ~5,000 reasonably-varied synthetic tiles (grid roads at arbitrary angles, varied building blocks, occasional landuse + POIs).
- Run Experiment 0 on Leonardo `boost_usr_prod`. Write `bonzai_genai/results/EXPERIMENT_0_REPORT.md`.

### Out of scope (deferred to Plan 3+)

- EasyControl LoRA conditioning **training** (the path is *coded* in production-size DiT but disabled in Experiment 0 — Exp 0 is unconditional).
- Experiment 1 (Stage A on real data) and Experiments 2–4. They follow Plan 3+.
- Wave 1 production data prep (~150k Western Europe tiles). That's Phase 2.
- Cross-tile context conditioning (tile stitching). Experiment 4's problem.
- Marina Bay extreme-density tile pre-cropping. Phase 0a follow-up; not load-bearing for Phase 0b.

## 3. Module layout

```
bonzai_genai/
├── src/bonzai_genai/
│   ├── models/                              ← NEW
│   │   ├── configs.py                       (TinyConfig / ProductionConfig dataclasses for each model)
│   │   ├── vae.py                           (9-channel VAE encoder + decoder + sampling head)
│   │   ├── dit.py                           (DiT blocks, AdaLN-Zero, sinusoidal time embed, conditioning paths)
│   │   ├── inker.py                         (AR transformer + RoPE + cross-attention + constrained decoder)
│   │   └── raster_encoder.py                (4-layer strided CNN, frozen during Inker training)
│   ├── training/                            ← NEW
│   │   ├── lit_vae.py                       (LightningModule: reconstruction + KL)
│   │   ├── lit_stage_a.py                   (LightningModule: EDM diffusion, latent or pixel)
│   │   ├── lit_stage_b.py                   (LightningModule: cross-entropy AR + raster cross-attn)
│   │   ├── data_module.py                   (LightningDataModule wrapping WebDataset shards)
│   │   ├── samplers.py                      (DPM-Solver++ for DiT, greedy + beam for Inker)
│   │   └── callbacks.py                     (sample-dump, EMA, custom checkpoint policy)
│   ├── eval/                                ← NEW
│   │   ├── stage_a.py                       (channel IoU, FID, FID-clip, conditioning ablation)
│   │   ├── stage_b.py                       (Chamfer, road graph connectivity, POI placement, validity)
│   │   ├── end_to_end.py                    (combined: sketch → ink → decode → re-rasterize → IoU)
│   │   └── baselines.py                     (random crop / nearest neighbor / frequency-matched / perfect)
│   └── synth/procedural.py                  (extended; existing minimal version stays as fallback)
├── scripts/
│   ├── leonardo_data_prep.sbatch            (existing)
│   ├── leonardo_vae_train.sbatch            ← NEW (boost_usr_prod, 1×A100, 24h, EMA + resume)
│   ├── leonardo_stage_a_train.sbatch        ← NEW (boost_usr_prod, 1×A100 smoke / 4×A100 prod)
│   ├── leonardo_stage_b_train.sbatch        ← NEW (boost_usr_prod, 1×A100 smoke / 4×A100 prod)
│   ├── leonardo_eval.sbatch                 ← NEW (lrd_all_serial for CPU-bound metrics; boost for sampling)
│   └── leonardo_experiment_0.sbatch         ← NEW (driver: VAE → DiT → Inker → eval, single 24-h job)
└── tests/
    └── test_{models,training,eval,synth}_*  ← NEW
```

**Estimated diff:** ~5,000 LoC code + ~1,500 LoC tests, ~30–40 plan tasks.

## 4. Sizing presets

Two named presets per model, picked by config flag at construction time. Lives in `models/configs.py` as `TinyConfig` / `ProductionConfig` frozen dataclasses.

| Component | "tiny" (Experiment 0) | "production" (Phase 4/5) | Spec ref |
|---|---|---|---|
| **VAE** | ~5 M params, 4 down-blocks, base ch 32 | ~10 M params, 4 down-blocks, base ch 64 | global §5.3 |
| **DiT (Sketcher)** | ~50 M, 12 layers, hidden 512, 8 heads, patch 2 over 64×64 latent | ~400 M, 24 layers, hidden 1024, 16 heads, patch 2, AdaLN-Zero, EDM noise | global §5.4 |
| **Inker (Stage B)** | ~50 M, 12 layers, hidden 512, 8 heads, ctx 4 k tokens, RoPE | ~750 M, 24–32 layers, hidden 1024–1280, 16 heads, ctx 16 k, RoPE | global §6.3 |
| **Raster CNN encoder** | ~5 M, 3 strided conv layers, output 32×32×256 | ~30 M, 4 strided conv layers, output 32×32×768 | global §6.3 |

EasyControl-style LoRA conditioning paths (global §5.5) are **coded** in production-size DiT but **dropped to null** for Experiment 0 — Exp 0 trains unconditionally; the conditioning interface is exercised only at the type-checking level.

## 5. Training stages

Three independent LightningModules. Each is single-GPU-runnable for the smoke preset and 4×A100 for production.

### 5.1 VAE training (Plan 2 runs the smoke version; production deferred to Phase 3)

- Loss: BCE on binary masks (channels 1–5, 7–9), MSE on density channel (6), KL regularizer.
- Optimizer: AdamW, lr 1e-4, cosine decay.
- 50 epochs reconstruction-only on the synth shards (smoke). EMA weights.
- Visual: dump 16 reconstructions every 10 epochs to a `tensorboard` / `wandb` writer.
- Freeze and save when val PSNR > 20 dB or 50 epochs reached (whichever first).

### 5.2 Stage A — DiT (Sketcher) training

- Loss: EDM diffusion loss with AdaLN-Zero conditioning blocks.
- Optimizer: AdamW, lr 1e-4, cosine decay.
- 10 % classifier-free-guidance dropout (drops conditioning to null; for Exp 0 this is a no-op since unconditional).
- 1 epoch over the synth shards (smoke). EMA weights.
- Sample 32 tiles every 1,000 steps via DPM-Solver++, 50 denoising steps.
- Lightning checkpoint every 30 min to `$WORK/bonzai-checkpoints/`.

### 5.3 Stage B — Inker training

- Loss: cross-entropy on next-token prediction, teacher-forcing.
- Optimizer: AdamW, lr 3e-4, cosine decay.
- Cross-attention to the raster CNN encoder's 32×32×256 (smoke) / 32×32×768 (production) feature grid.
- For Exp 0: cross-attend to the **ground-truth** raster (no domain gap yet — that's Experiment 3).
- 1 epoch over `(raster, tokens)` pairs from the synth shards (smoke).
- Sample 32 tiles every 500 steps, greedy, with constrained-decoding logit masks (global §6.4) active.
- Lightning checkpoint every 30 min.

## 6. Eval harness

Full §8 suite from the global spec. Implemented in `bonzai_genai/eval/`. Each metric exposes a function with signature `metric(samples, ground_truth, **opts) -> dict[str, float]`.

### 6.1 Stage A metrics — `eval/stage_a.py`
- Per-channel IoU on binary channels (1–5, 7–9).
- MSE on density channel (6).
- FID-clip on a 1,000-tile val subset (low budget for Exp 0; bumps for Phase 1).
- Conditioning effectiveness: KL between conditional and unconditional sample distributions (no-op in Exp 0; live in Phase 1).

### 6.2 Stage B metrics — `eval/stage_b.py`
- Building Chamfer distance (sample-vs-ground-truth, average + p95).
- % non-self-intersecting buildings.
- Single-component fraction of the road graph.
- POI placement: distance to nearest same-class ground-truth POI.
- % well-formed GeoJSON outputs from constrained decoding.

### 6.3 End-to-end metrics — `eval/end_to_end.py`
- Pipe DiT-sampled raster → Inker → decode → re-rasterize → channel IoU vs original ground-truth raster.
- "Validity drop" = end-to-end Stage B validity rate vs Stage B-on-ground-truth validity rate (Experiment 3's measurement; computed but not load-bearing in Exp 0).

### 6.4 Baselines — `eval/baselines.py` (computed once before any training)
Per global §8.2:
- **Random crop** — random tile from the val set (lower bound).
- **Nearest neighbor** — nearest ground-truth tile by raster L2.
- **Frequency-matched random** — sample-from-class-prior baseline.
- **Perfect tile** — actual ground-truth (upper bound).

## 7. Experiment 0 protocol

End-to-end driver (`scripts/leonardo_experiment_0.sbatch`) runs these steps in one 24-h job slot:

| Step | What | Where | Budget |
|---|---|---|---:|
| 1 | Generate 5,000 synth tiles (4,500 / 500 split) | `lrd_all_serial` (free CPU) | ~10 min wall |
| 2 | Train tiny VAE | `boost_usr_prod`, 1×A100 | ~1 GPU-h baseline, up to 5 |
| 3 | Train tiny DiT (latent space, frozen VAE) | `boost_usr_prod`, 1×A100 | ~5 GPU-h baseline, up to 15 |
| 4 | Train tiny Inker (cross-attn to GT raster) | `boost_usr_prod`, 1×A100 | ~4 GPU-h baseline, up to 10 |
| 5 | Run §8 eval suite on val set | mixed (GPU sample, CPU metric) | ~2 GPU-h |
| 6 | Write `EXPERIMENT_0_REPORT.md` | local | 0 |

Total: **~12 GPU-h baseline; up to ~30 GPU-h if convergence needs it.**

### 7.1 Go signals (per global §10 Experiment 0)

All three soft-pass:

1. **Visual eyeball check** on 32 sampled DiT tiles: "looks like grid + rectangles". Subjective; PI judgment.
2. **Loss curves don't diverge** for any of VAE / DiT / Inker (no NaN, no hockey-stick up).
3. **≥ 90 % of decoded GeoJSON outputs from the Inker are well-formed** (parses cleanly, all polygons closed, layer order correct).

### 7.2 No-go signals

Any of:

- Training NaNs / loss explodes for any stage that 5+ GPU-h can't fix.
- Sampled DiT tiles are pure noise after >5 GPU-h.
- < 50 % well-formed GeoJSON output from constrained decoding.

→ Diagnose root cause, fix, re-run. Do **not** proceed to Plan 3 with red signals.

### 7.3 Mixed signals

Stage A green but Stage B partial (e.g. 60 % well-formed): record observations in `EXPERIMENT_0_REPORT.md`, decide case-by-case. Likely log as Plan 3 follow-up rather than blocking — Stage B has far more room to improve on real data than synthetic.

## 8. Open questions

1. **VAE PSNR threshold** — 20 dB chosen as a soft pass. May need calibration after first smoke run; bump if reconstructions are visibly poor.
2. **Inker context length for "tiny"** — 4 k tokens chosen. A dense Singapore-style synth tile *could* exceed this; need to confirm during data generation. If overflow > 5 % of synth tiles, bump to 8 k or simplify the synthetic generator.
3. **Constrained decoding implementation** — global §6.4 lists six masking rules. Which subset is mandatory for Exp 0 vs nice-to-have? Tentative answer: layer order + polygon closure + coordinate pair are mandatory; non-self-intersection + node-ref bounds + building field order are nice-to-have. Revisit during plan-writing.
4. **EMA decay** — 0.9999 by convention; is that right for short smoke runs (1 epoch)? Probably too slow; lower to 0.999 for smoke, 0.9999 for production.

## 9. Success criteria

**Plan 2 ships when:**
- All model + training + eval modules land with passing tests + ruff-clean.
- Slurm scripts run end-to-end on Leonardo (smoke preset).
- Experiment 0 runs to completion on Leonardo (`boost_usr_prod`).
- `EXPERIMENT_0_REPORT.md` is committed with go/no-go decision recorded.
- `STATUS.md` updated with Plan 3 hand-off pointer.

**Plan 2 does NOT ship if:**
- Experiment 0 hits a no-go signal that can't be reproducibly fixed.
- Production-size code paths exist only as type-stubs (must be runnable, even if not yet trained).

---

## Self-review

- Placeholders: none — all sections have concrete numbers or named open questions.
- Internal consistency: §4 sizing matches §7 budget (1 + 5 + 4 = 10 GPU-h baseline ≈ "up to ~25" with margin); §3 module layout matches §5 / §6 file references.
- Scope check: focused on Phase 0b's modeling + eval + Exp 0 run. Production training (Phases 4–5) and Experiment 1+ are explicitly out.
- Ambiguity: "tiny" preset numbers are concrete (50 M / 5 M / etc.); "≥ 90 % well-formed GeoJSON" is concrete; "looks like grid + rectangles" is intentionally subjective and labeled as such.
