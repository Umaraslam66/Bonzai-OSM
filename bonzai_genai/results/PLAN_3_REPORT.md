# Plan 3 — Report

**Status:** TEMPLATE — fill in after the Leonardo run completes.

**Spec:** [`docs/superpowers/specs/2026-05-04-plan-3-experiments-1-and-2-design.md`](../../docs/superpowers/specs/2026-05-04-plan-3-experiments-1-and-2-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-04-plan-3-experiments-1-and-2.md`](../../docs/superpowers/plans/2026-05-04-plan-3-experiments-1-and-2.md)
**Branch:** `genai-city-model`
**Cluster:** Leonardo Booster (CINECA)
**Output dir on Leonardo:** `$WORK/bonzai-plan3/`

## What this run was

Plan 3 — Experiments 1 (Painter on real Sweden + Singapore + Sri Lanka)
and 2 (Writer on perfect ground-truth raster). 4-GPU DDP. Country-balanced
streaming sampler. ~200 M Painter, ~300 M Writer.

## What ran

| Step | What | Where | Wall | Outcome |
|---|---|---|---:|---|
| 1 | Country count + weight compute | local Mac | <1 min | (fill) |
| 2 | Painter (stage_a) — 50 epochs, 4-GPU DDP | `boost_usr_prod` 4×A100 | (fill) | (fill) |
| 3 | Writer (stage_b) — 50 epochs, 4-GPU DDP | `boost_usr_prod` 4×A100 | (fill) | (fill) |
| 4 | Sample-from-ckpt (64 samples) | `boost_usr_prod` 1×A100 | (fill) | (fill) |
| 5 | Eyeball check + decision-tree call | local | (fill) | (fill) |

**Total node-h burned:** (fill) / ~54 budgeted.

## Country weights used

(fill from `scripts/run_plan3.py` output)

## Eval numbers

(fill from `eval_results.json`)

## Sample inspection

(embed grids from `bonzai_genai/results/plan3-samples/`)

## Decision-tree call

Per spec §7.3, the eval signal points to one of:

- [ ] Both green → proceed to Plan 4 (Exp 3 — domain gap).
- [ ] Writer self-intersection > 30% → land deferred Plan 2 rules; retrain Writer.
- [ ] Painter samples are pure noise → bump to 400 M (production size).
- [ ] Painter Singapore-conditioning ≈ Sweden-conditioning → re-prep with NODE_REF 16k + Sri Lanka 6k.
- [ ] Painter loses coastal morphology → fix `<5 features` skip rule.
- [ ] Painter samples visibly N-S squashed → switch to UTM tile geometry.

**Selected:** (fill)

**Next action:** (fill)

## Hand-off

(fill — point to Plan 4 spec or to the corrective re-prep plan)
