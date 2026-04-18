# Stockholm PoC — tokenization pipeline

Proof-of-concept for the "Spatial Foundation Model": turn a single-city
OpenStreetMap `.osm.pbf` into a 1D sequence of spatial tokens that a
standard Transformer can ingest.

This directory is self-contained: scripts here produce
`stockholm_tokens.parquet` (and a companion `stockholm_vocab.json`) from
the BBBike Stockholm extract.

## Files

| File | Purpose |
|------|---------|
| `setup_workspace.sh`    | Create `$WORK/stockholm_poc` on Leonardo and download `Stockholm.osm.pbf` from BBBike |
| `tokenize_stockholm.py` | PBF → filter → simplify → H3 anchor → relative-offset moves → Z-order sort → Parquet |
| `slurm_tokenize.sh`     | SLURM batch for `dcgp_usr_prod` (1 node, 128 GB RAM) — spec-compliant, **billed** |
| `slurm_tokenize_serial.sh` | SLURM batch for `lrd_all_serial` (1 core, 28 GB RAM, 4 h) — **budget-free**, default for iteration |
| `requirements.txt`      | Python deps (pyrosm, shapely, h3, pyarrow, numpy, pandas) |

## Token schema

Each object emits a bracketed list of tokens. The full corpus is one
big sequence obtained by concatenating objects in **Z-order (Morton)**
of their centroids so the 2D map becomes a 1D stream with locality.

```
<BUILDING_START>   <TAG_{class}>   <H3_{res11-cell}>   <MOVE_{DIR}_{Nm}M> ...   <BUILDING_END>
<ROAD_START>       <TAG_{class}>   <H3_{res11-cell}>   <MOVE_{DIR}_{Nm}M> ...   <ROAD_END>
<PART_SEP>         # separator between rings of a MultiPolygon
```

- **Anchor** — the first vertex of each (multi)polygon / line is turned
  into an **H3 index at resolution 11** (~25 m edge). This gives the
  model an absolute spatial "you are here" before any relative motion.
- **Moves** — every subsequent simplified vertex is expressed as a
  direction + distance delta from the previous vertex, in local meters
  via an equirectangular projection:
  - `DIR ∈ {N, NE, E, SE, S, SW, W, NW}`
  - distance snapped to the ladder `{5, 10, 15, 25, 50, 100, 250, 500, 1000} m`, greedily decomposed (long edges emit multiple tokens).
- **Tags** — `highway=*` and `building=*` values are bucketed into the
  most common OSM classes (see `HIGHWAY_CLASSES` / `BUILDING_CLASSES`
  in `tokenize_stockholm.py`). Everything else collapses to `<TAG_OTHER>`.
- **Simplification** — `shapely.simplify(1e-5 deg)` ≈ 1 m at Stockholm
  latitudes. Drops noisy micro-vertices while keeping corners.

## Parquet layout

One row per object (building or road segment). Chunked at 10 000 rows
per row-group so HuggingFace `datasets` can stream it efficiently.

| column         | type             | notes |
|----------------|------------------|-------|
| `kind`         | string           | `BUILDING` or `ROAD` |
| `tag_token`    | string           | e.g. `<TAG_RESIDENTIAL>` |
| `centroid_lon` | float64          | for debugging / viz |
| `centroid_lat` | float64          | for debugging / viz |
| `n_tokens`     | int32            | length of `tokens` |
| `tokens`       | list\<string\>   | full per-object token sequence |

The flat 1D corpus for training is `concat(row.tokens for row in file)`,
already in Z-order.

## How to run on Leonardo

All the below happens on Leonardo. Paste these into your Leonardo
terminal one block at a time.

### 1. Get the code onto `$WORK`

From your Mac (or wherever this repo lives):

```bash
git push origin feature/stockholm-poc
```

Then on Leonardo:

```bash
git clone https://github.com/Umaraslam66/Bonzai-OSM.git "$WORK/Bonzai-OSM" 2>/dev/null || \
  (cd "$WORK/Bonzai-OSM" && git fetch && git checkout feature/stockholm-poc && git pull)
mkdir -p "$WORK/stockholm_poc/scripts"
cp "$WORK/Bonzai-OSM/stockholm_poc/tokenize_stockholm.py" "$WORK/stockholm_poc/scripts/"
```

### 2. Create the workspace and download the PBF

```bash
bash "$WORK/Bonzai-OSM/stockholm_poc/setup_workspace.sh"
```

### 3. Create the Python venv (one-time, on a login node)

```bash
module purge && module load $(module avail 2>&1 | grep -oE 'python/3\.11[^ ]*' | head -n1)
python -m venv "$WORK/stockholm_poc/venv"
source "$WORK/stockholm_poc/venv/bin/activate"
pip install --upgrade pip
pip install -r "$WORK/Bonzai-OSM/stockholm_poc/requirements.txt"
python -c "import pyrosm, shapely, h3, pyarrow; print('ok')"
deactivate
```

If no Python 3.11 module is exposed on Leonardo, fall back to a user-space Miniforge:

```bash
curl -L -o "$WORK/miniforge.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$WORK/miniforge.sh" -b -p "$WORK/miniforge3"
source "$WORK/miniforge3/bin/activate"
conda create -y -n sfm python=3.11
conda activate sfm
pip install -r "$WORK/Bonzai-OSM/stockholm_poc/requirements.txt"
# In slurm_tokenize.sh, replace `source "${VENV}/bin/activate"` with
# `source "$WORK/miniforge3/bin/activate" && conda activate sfm`.
```

### 4. Submit the tokenization job

**Default: budget-free `lrd_all_serial`** (what we use for iteration):

```bash
sbatch "$WORK/Bonzai-OSM/stockholm_poc/slurm_tokenize_serial.sh"
squeue --me
# once it starts:
tail -f stockholm-tokenize-serial-*.out
```

**Spec-compliant 128 GB billed run** (use only for the final PoC, not for iteration):

```bash
sbatch "$WORK/Bonzai-OSM/stockholm_poc/slurm_tokenize.sh"
```

### 5. Verify the output

```bash
ls -lh "$WORK/stockholm_poc/outputs/"
python - <<'PY'
import os, pyarrow.parquet as pq
p = os.path.join(os.environ["WORK"], "stockholm_poc/outputs/stockholm_tokens.parquet")
pf = pq.ParquetFile(p)
print("num row groups:", pf.num_row_groups)
print("num rows      :", pf.metadata.num_rows)
tbl = pf.read_row_group(0)
print(tbl.column("kind")[:5].to_pylist())
print(tbl.column("tokens")[0][:12].as_py())
PY
```

## Budget notes

`dcgp_usr_prod` is the CPU partition with >30 GB RAM, but it bills at
**112 core-hours per node-hour**. A 30-minute run on one node costs
~56 core-hours (~0.14 % of the 40 000 allocation). That's fine for the
PoC but don't loop it unnecessarily.

For a budget-free smoke test, the same Python script runs comfortably
on `lrd_all_serial` (4 cores, 30.8 GB RAM, 4 h walltime) — Stockholm
has well under a million objects and the tokenizer never materialises
more than a few hundred MB in memory. Swap the SLURM header accordingly
if you need iteration speed on a near-empty budget.
