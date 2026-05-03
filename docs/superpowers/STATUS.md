# Bonzai-OSM — Live Status

> **Last updated:** 2026-05-03
> **Updater:** Claude (current session)
>
> If you are a new agent starting a session: read this file first, then [`PROJECT.md`](../../PROJECT.md), then the [spec](specs/2026-05-03-genai-city-infrastructure-design.md), then the [active plan](plans/2026-05-03-phase-0a-data-prep-pipeline.md). Pick up from the next unchecked task.

---

## Current state

| | |
|---|---|
| Branch | `genai-city-model` (long-lived dev branch — **never merge to main**) |
| Active phase | **Phase 0a — Data Prep Pipeline + 3-country tile dataset** |
| Active plan | [`plans/2026-05-03-phase-0a-data-prep-pipeline.md`](plans/2026-05-03-phase-0a-data-prep-pipeline.md) (20 tasks) |
| Last completed task | Cleanup + branching + doc updates (pre-Task 1) |
| Next action | **Plan Task 1 — Project scaffolding (pyproject.toml + src layout)** |
| GPU-h burned this session | 0 |
| GPU-h burned cumulative on project | 14 (from prior sessions, pre-v1) |

## Known good state

- Repository cleaned: `overture-map/` deleted, Luxembourg test SLURM jobs deleted, early exploration scripts deleted. Only the v1-relevant docs and helpers remain.
- Spec, plan, and brainstorm log are all committed and consistent. Spec spans 15 sections + 2 appendices; plan has 20 tasks.
- Memory entries up-to-date for: user role, project goal, allocation, data state (pre-v1), v1 user persona (AV / sim), quality-over-thrift philosophy, terminal-vs-visual preferences, simple-words explanation preference, branch + country selection.

## Phase 0a deliverable

A working `bonzai_genai/` Python package + tile shards on Leonardo `$WORK/bonzai-tiles/{sweden,singapore,sri_lanka}/`, validated via round-trip tests. **Zero GPU billing consumed.**

## Three countries for v1 de-risking

| Country | Köppen | Geofabrik URL | Centroid bbox |
|---|---|---|---|
| Sweden | Cfb / Dfb | `https://download.geofabrik.de/europe/sweden-latest.osm.pbf` | `55.0,10.5 → 69.5,24.5` (full country) |
| Singapore | Af | `https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf` (extract) | `1.20,103.60 → 1.48,104.05` (full island) |
| Sri Lanka | Af / Aw | `https://download.geofabrik.de/asia/sri-lanka-latest.osm.pbf` | `5.85,79.55 → 9.90,81.95` (full country) |

Note: Singapore's Geofabrik extract is bundled with Malaysia + Brunei; we'll use osmium to crop to the Singapore bbox during data prep.

## Plan progress (Phase 0a — 20 tasks)

- [ ] Task 1: Project scaffolding (pyproject.toml + src layout)
- [ ] Task 2: Install dev dependencies + verify pytest works
- [ ] Task 3: Global config module
- [ ] Task 4: Token type definitions
- [ ] Task 5: Attribute vocabulary
- [ ] Task 6: Tokeniser — encode primitives to token sequence
- [ ] Task 7: Tokeniser round-trip property test
- [ ] Task 8: Rasteriser — line and polygon painting
- [ ] Task 9: TileBundle dataclass + serialisation
- [ ] Task 10: WebDataset shard writer + reader
- [ ] Task 11: Synthetic procedural city generator
- [ ] Task 12: End-to-end synthetic round-trip test
- [ ] Task 13: Smoke test CLI — generate 100 synthetic shards locally
- [ ] Task 14: Real-data tile sampler (Overture / Geofabrik)
- [ ] Task 15: Generate small Sweden tile dataset locally
- [ ] Task 16: Slurm template for Leonardo data prep
- [ ] Task 17: Plan-level documentation in repo README
- [ ] Task 18: Run the full test suite + lint
- [ ] Task 19: Deploy to Leonardo and run all three country jobs
- [ ] Task 20: Plan completion summary

Per-task checkboxes are inside the plan file.

## Blockers / open questions

None at this moment. Awaiting Plan Task 1 execution.

## Session-handoff notes (for new agents)

- The active sub-skills in use are `superpowers:executing-plans` (inline execution) — **not** subagent-driven. PI has explicitly requested inline execution.
- All implementation runs through this `STATUS.md` file as the live tracker. Update it after each plan task completes, or whenever a non-trivial decision is made.
- Memory directory: `~/.claude/projects/-Users-umaraslam-Documents-dynamo-Bonzai-OSM/memory/`. `MEMORY.md` is the index.
- Commit cadence: at least once per plan task (the plan's TDD steps describe explicit `git commit` invocations). Commit messages should be descriptive enough that `git log --oneline` tells the implementation story.
- No subagents. PI explicitly requested inline execution by the main session.
