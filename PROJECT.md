# Bonzai-OSM — Project Log

> World-scale OpenStreetMap → unsupervised generative "map LLM" trained on Leonardo (CINECA EuroHPC).

**PI:** Umar Aslam — Leonardo username **`uaslam00`** (uid 132662, primary group `interactive`), Cineca contact email `aslamumar16@gmail.com`, personal `aslamumar012@gmail.com`
**Project account name on Leonardo:** **`AIFAC_P02_222`** — this IS the EuroHPC allocation. Confirmed on 2026-04-15: `saldo -b` shows start=`20260311`, end=`20260611`, total=`40000` local-h — matches the award letter exactly. Only 14 core-h burned (0.035%) by prior sentiment-action-transformer experiments. Previous unrelated work under `/leonardo_work/AIFAC_P02_222/{containers, bonzai}` and `/leonardo_scratch/fast/AIFAC_P02_222/bonzai_cache` was fully removed on 2026-04-15; the OSM workspace now starts from a clean slate.
**Auto-exported env vars on login:** `$HOME`, `$WORK=/leonardo_work/AIFAC_P02_222`, `$FAST=/leonardo_scratch/fast/AIFAC_P02_222`, `$CINECA_SCRATCH=/leonardo_scratch/large/userexternal/uaslam00`. All four are set on first login — no manual export needed.
**Today:** 2026-04-15
**Doc owner:** Claude + Umar. Every material decision lands here.

---

## 1. Goal (one paragraph)

Download all of OpenStreetMap (planet + full history optionally), convert the dense PBF graph into a tokenised representation of roads, buildings, land-use, POIs, and topology, then run unsupervised pre-training so the model learns the statistical structure of "what the world looks like on a map." Downstream: generate synthetic maps / new plausible worlds, conditional generation ("draw a coastal Scandinavian town"), and region completion.

## 2. Allocation (ground truth from the EuroHPC award letter)

| Item                     | Value                                                           |
| ------------------------ | --------------------------------------------------------------- |
| Cluster                  | **Leonardo Booster** (CINECA, Bologna)                          |
| Budget                   | **40,000 local core-hours** = **1,250 node-hours**              |
| Project window           | **2026-03-11 → 2026-06-11** (3 months)                          |
| Elapsed as of 2026-04-15 | ~35 days of 92 (≈38%)                                           |
| Default $WORK quota      | **1 TB** (extendable on request — see §5)                       |
| Archive storage          | **Not active by default** — request via `superc@cineca.it`      |
| Budget linearization     | Monthly quota = total / months; over-quota = lower Slurm priority (still runnable, just slower) |

**Spend model (what 1 node-hour costs):**
- `boost_usr_prod`: 1 node × 1 h = **32 core-hours** (the whole node, GPUs included; GPU ≈ ¼ node)
- `dcgp_usr_prod`: 1 node × 1 h = **112 core-hours** (CPU-only partition; burns budget ~3.5× faster per node-hour than Booster)
- `lrd_all_serial`: **budget-free** (see §4)

**Rule of thumb:** 40k core-hours on Booster = **~1,250 GPU-node-hours** = ~5,000 single-GPU hours. That's enough for a meaningful pre-training run if we pick architecture carefully, but it's not generous. **Do not burn any of it on PBF extraction, data wrangling, or tokenisation.**

## 3. Strategy summary (the plan in 5 bullets)

1. **PBF download goes through Leonardo datamovers; extraction runs on the budget-free `lrd_all_serial` partition**, both staged on `$CINECA_SCRATCH` (per-user, effectively 20 TB, files >40 days auto-deleted). This is the "20 TB free scratch" you asked about — it exists, it's yours, it just resets on a rolling 40-day window.
2. **Tokenised datasets** (the ones we actually want to keep) move to `$WORK` once stable. Request a $WORK quota bump to the max allowed under our allocation when we need it.
3. **GPU training** runs on `boost_usr_prod` only, with every run checkpointed so we never redo work that already cost budget.
4. **Data transfer from outside** (planet.osm.pbf is ~85 GB) should use Leonardo datamovers or GridFTP. Small files can go through login nodes, but the official docs note a **10-minute CPU-time limit** there, so datamovers are the safe default for planet-scale transfers.
5. **Keep the Booster budget for the one thing that actually needs A100s: training.**

## 4. Leonardo partitions & the "free CPU" answer

| Queue              | Cores/node | RAM    | GPUs | Max walltime | Billed? |
| ------------------ | ---------- | ------ | ---- | ------------ | ------- |
| **lrd_all_serial** | **4** (8 logical) | **30.8 GB** | 0 | **4h** | **BUDGET FREE** |
| boost_usr_prod     | 32         | 512 GB | 4    | 24h          | yes (Booster) |
| boost_qos_dbg      | 32         | 512 GB | 4    | 30 min       | yes (debug) |
| boost_qos_lprod    | 32         | 512 GB | 4    | 4 days       | yes (long) |
| dcgp_usr_prod      | 112        | 512 GB | 0    | 24h          | yes (CPU; expensive) |

**The free-CPU answer, direct:** `lrd_all_serial` is marked "Budget Free" in the official docs. It gives you 4 cores + 30.8 GB RAM + 4 h walltime per job, with **unlimited job submissions**, and usage is excluded from your 40k core-hour budget. It's explicitly intended for "limited post-production data analysis" and is available even after a project account expires. PBF inspection and region-scale parsing are I/O-bound enough to fit here. This is the only partition we should use for preprocessing unless we prove we genuinely need more RAM.

**Login-node work is also effectively free** as long as it stays under a few cores and doesn't hammer the shared node. But for large transfers, use the datamovers: the official docs say login nodes have a **10-minute CPU-time limit** that can interrupt long downloads. Small checks like checksum verification and short `osmium` commands are still fine there.

**No hidden DCGP free tier.** The DCGP (CPU) partition is billed at 112 core-hours per node-hour — running a 24 h job there burns 2,688 core-hours, ~7% of our entire allocation. **Avoid DCGP unless we have a task that genuinely needs >30 GB RAM or >4 cores AND is CPU-bound.** For everything else, use lrd_all_serial.

## 5. Storage tiers — the "20 TB scratch" answer

| Area             | Path / var              | Default quota | Scope       | Retention                        | Backup | How to grow                              |
| ---------------- | ----------------------- | ------------- | ----------- | -------------------------------- | ------ | ---------------------------------------- |
| `$HOME`          | per-user                | **50 GB**     | per user    | permanent                        | yes    | not expandable; don't store data here    |
| `$WORK`          | `/gpfs/work/<account>`  | **1 TB**      | per project | kept 6 months past project end   | no     | email `superc@cineca.it` for an increase |
| `$FAST`          | fast NVMe               | **1 TB**      | per project | kept 6 months past project end   | no     | **fixed — no extensions**                |
| **`$CINECA_SCRATCH`** | `/leonardo_scratch/...` | **unlimited (~20 TB practical)** | **per user** | **files older than 40 days auto-deleted** | no | — (already unlimited) |
| `$PUBLIC`        | per-user                | 50 GB         | per user    | permanent                        | no     | Help Desk                                |
| `$DRES`          | `/gss/gss_work/...`     | on request    | cross-project | 6 months past DRES end         | no     | must be explicitly requested             |

**So yes — the 20 TB scratch exists as `$CINECA_SCRATCH`**, it's per-user, and there's no form to fill out. The catch is the **40-day auto-delete** — any file not modified in 40 days gets garbage-collected. Rule: scratch is the workbench, `$WORK` is the vault. Do not `touch` files to evade the TTL (docs explicitly warn that gets you banned).

**For $WORK growth:** the award letter says we can "ask to CINECA User Support to increase your quota up to the amount granted in the allocation awarded by EuroHPC." Our proposal presumably justified some data footprint. When we know how much we need (tokenised planet ≈ 500 GB–2 TB depending on format), email `superc@cineca.it` from the address on the UserDB account.

**Quota-check commands (on Leonardo):**
```bash
cindata              # usage across all areas, loaded via cintools module (auto-loaded)
cinQuota             # filesystem quota + grace + files count
saldo -b             # budget: total, monthly, consumed, remaining
saldo --help         # all options
```

## 6. First-time access — copy-paste walkthrough

> Run these on your **macOS laptop**, not on Leonardo. The certificate lives locally and gets injected into your ssh session.

### 6.1 One-time: enroll 2FA on the CINECA Identity Provider

1. Open **https://sso.hpc.cineca.it** in a browser.
2. Sign in with the HPC credentials Cineca sent you after they added the EuroHPC project to `uaslam00`.
3. Verify email when prompted.
4. Follow the setup wizard — it shows a QR code. Scan it with **Google Authenticator**, **FreeOTP**, or **1Password** on your phone. Save the recovery codes somewhere safe (1Password, not Downloads).
5. Done. You now generate a 6-digit OTP on demand.

### 6.2 One-time: install smallstep CLI

```bash
# Homebrew (recommended on macOS)
brew install step
step version      # sanity check — should print something like "Smallstep CLI/0.27.x"
```

If `brew install step` errors with an "undefined method 'cellar'" message, run `brew update && brew tap homebrew/core` and retry.

### 6.3 One-time: bootstrap the CINECA CA

```bash
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

This writes the CA root to `~/.step/` and trusts the CINECA SSH CA for future requests. You only do this once per machine.

### 6.4 Every 12 hours: request an SSH certificate

```bash
step ssh login 'aslamumar16@gmail.com' --provisioner cineca-hpc
```

A browser tab opens → authenticate with your CINECA HPC username/password → enter the OTP from the authenticator → tab closes → a short-lived SSH certificate lands in your ssh agent, valid for **12 hours**.

> If you entered the wrong email when Cineca set up your account, use whichever address they actually linked to `uaslam00`. When in doubt, log in to https://userdb.hpc.cineca.it/ and check "email" on your profile — that's the one to pass here.

### 6.5 Connect

```bash
ssh uaslam00@login.leonardo.cineca.it
```

No password. If it asks for one, your certificate expired — rerun step 6.4.

### 6.6 Optional: `~/.ssh/config` shortcut

Add this to `~/.ssh/config` on your laptop so you can just type `ssh leo`:

```
Host leo leonardo
    HostName login.leonardo.cineca.it
    User uaslam00
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

Then `ssh leo` works any time your cert is fresh.

## 7. Cleanup plan for existing $WORK on Leonardo

You said "I have some things already on Leonardo but all are useless." **Do not mass-delete until we've seen what's there.** The policy:

1. **Inventory first, delete second.** Always.
2. **Never** `rm -rf $WORK` in one shot — GPFS doesn't give you an undo.
3. **Prefer `mv` to a `_trash/` dir** for anything over a few GB; actual deletion can wait until we're sure.

Run these in order, **from the Leonardo login node** (after SSH'ing per §6.5):

```bash
# 7.1 — Who are we, what accounts are we on
whoami                           # should print: uaslam00
id
saldo -b                         # lists every project account you belong to + budget

# 7.2 — Find the actual filesystem paths (NOT via $WORK/$FAST — those aren't auto-set on Leonardo)
ls -la /leonardo/home/userexternal/ 2>/dev/null | grep uaslam00
echo "HOME = $HOME"
ls -la /leonardo_work/ 2>/dev/null       # every project-work dir visible to you
ls -la /leonardo_scratch/fast/ 2>/dev/null | grep -E "uaslam00|AIFAC|$(id -gn)"
ls -la /leonardo_scratch/large/ 2>/dev/null | grep uaslam00

# 7.3 — Once saldo -b prints your account name(s), export the paths yourself
# (example using the old account — replace with what saldo -b actually shows for the new project)
export OLD_ACCOUNT=AIFAC_P02_222
export OLD_WORK=/leonardo_work/${OLD_ACCOUNT}
export OLD_FAST=/leonardo_scratch/fast/${OLD_ACCOUNT}
ls -la "$OLD_WORK" "$OLD_FAST" 2>/dev/null

# Then for the NEW EuroHPC project (replace <NEW_ACCOUNT> with what saldo -b shows):
# export NEW_ACCOUNT=<NEW_ACCOUNT>
# export WORK=/leonardo_work/${NEW_ACCOUNT}
# export FAST=/leonardo_scratch/fast/${NEW_ACCOUNT}
# export CINECA_SCRATCH=/leonardo_scratch/large/${USER}     # per-user, not per-project
# cd "$WORK" && pwd

# 7.4 — Inventory the OLD project (main cleanup target)
cd "$OLD_WORK" 2>/dev/null && pwd
du -sh -- */ .[!.]*/ 2>/dev/null | sort -h

# 7.5 — 30 biggest files anywhere under the old project
find "$OLD_WORK" -xdev -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -nr | head -30 \
  | awk '{ printf "%10.2f GB\t%s\n", $1/1024/1024/1024, substr($0, index($0,$2)) }'

# 7.6 — Quota / usage from Cineca tools (works regardless of exported vars)
cindata
cinQuota

# 7.7 — Create a staging trash dir (NOT deleting yet)
mkdir -p "$OLD_WORK/_trash_$(date +%Y%m%d)"

# 7.8 — (ONLY after reviewing 7.4 output) move obviously-useless items to trash
# Example — adjust the names to match what 7.4 actually printed:
# mv "$OLD_WORK/containers" "$OLD_WORK/_trash_$(date +%Y%m%d)/"
# mv "$OLD_WORK/bonzai"     "$OLD_WORK/_trash_$(date +%Y%m%d)/"

# 7.9 — After a day of "did I need that?" grace period, nuke the trash
# rm -rf "$OLD_WORK/_trash_YYYYMMDD"
```

**Note on env vars:** On Leonardo all four of `$HOME`, `$WORK`, `$FAST`, `$CINECA_SCRATCH` **are** auto-exported on login. No manual setup needed. Confirmed 2026-04-15 from `uaslam00`'s fresh shell.

**Paste the output of 7.2, 7.3, 7.4 back into this chat** and I'll tell you which dirs to move into `_trash_…` and which to keep. Don't delete blindly.

## 8. OSM data pipeline plan (first pass, will evolve)

### 8.1 What we're downloading

- **`planet-latest.osm.pbf`** from `https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf` — one file, **~85 GB**, snapshot updated weekly.
- **Geofabrik per-continent/country extracts** as a backup source if planet is too heavy to iterate on: `https://download.geofabrik.de/` — smaller, regionally partitioned.
- **(Maybe)** the full history file `planet-latest-internal.osh.pbf` (~180 GB) if we want temporal signal. Decide later.

### 8.2 Where it lives on Leonardo

| Stage                          | Location                        | Size      | Why                                                 |
| ------------------------------ | ------------------------------- | --------- | --------------------------------------------------- |
| Raw PBF download               | `$CINECA_SCRATCH/osm/raw/`      | ~85 GB    | scratch is free & huge; raw re-downloadable         |
| Regional PBF extracts          | `$CINECA_SCRATCH/osm/extracts/` | ~hundreds GB | intermediate; regenerated by GDAL / future `osmium` workflows |
| Tokenised training shards      | `$WORK/osm-tokens/`             | 0.5–2 TB  | **this is the artifact we cannot afford to lose**   |
| Model checkpoints              | `$WORK/checkpoints/`            | tens GB   | keep forever; back up manually to something durable |
| Active training working set    | `$FAST/osm/` (if helpful)       | ≤1 TB     | NVMe, good for hot shards during training          |

### 8.3 The extraction job (budget-free)

Current verified state on 2026-04-15:

- `planet-latest.osm.pbf` downloaded successfully to `$CINECA_SCRATCH/osm/raw/`
- Upstream redirect resolved to `planet-260406.osm.pbf`
- File size: `92,239,256,545` bytes
- `md5sum -c planet-latest.osm.pbf.md5` returned `planet-latest.osm.pbf: OK`
- `module avail` shows `gdal` and `proj`, but **no `osmium` module**

So the immediate free-path plan is:

1. Download tiny Geofabrik extracts such as Luxembourg and Iceland via the Leonardo datamover.
2. Probe parsing on `lrd_all_serial` using GDAL's OSM driver.
3. Defer full-planet custom extraction until either:
   - the GDAL probe is satisfactory, or
   - `osmium` is installed/built in user space.

Reference probe job:

```bash
#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=lux-probe
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

module load gdal/3.8.5--gcc--12.2.0

RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf
ogrinfo "$RAW" -so points
ogrinfo "$RAW" -so lines
ogrinfo "$RAW" -so multipolygons
```

Future `osmium`-based extraction template once the tool is available:

```bash
#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=osm-extract
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out

module load osmium-tool   # TODO: verify exact module name with `module avail osmium`

cd "$CINECA_SCRATCH/osm"
osmium extract \
    --config regions.conf \
    --strategy=simple \
    raw/planet-latest.osm.pbf
```

We chain jobs with `--dependency=afterok:<jobid>` so each 4-hour chunk picks up where the last one stopped. Cost to budget: **zero**.

### 8.4 Tokenisation strategy (open)

**Decide before we start extracting:** what is "a token" for a map?
- **Tile-based**: fixed-zoom Slippy tile → sequence of (tag, geometry) tuples inside it
- **Graph-based**: nodes + ways serialised as a sequence, similar to "Polygen" / "DeepMind's MeshGPT"
- **Hierarchical**: admin-region → way → node, learned BPE-style on OSM tag vocabulary

Likely we'll prototype 2–3 on a small extract (say, Iceland or Luxembourg) and pick one before touching the planet. This decision is TODO.

## 9. Open action items

| #  | Action | Owner | Status | Notes |
| -- | ------ | ----- | ------ | ----- |
| A1 | Enroll 2FA, install smallstep, first SSH | Umar | ✅ **done** 2026-04-15 | cert in agent |
| A2 | Run inventory on $WORK / $FAST / $HOME | Umar | ✅ **done** 2026-04-15 | see §10 budget/disk baseline |
| A3 | Confirm project account name | Umar | ✅ **done** | `AIFAC_P02_222` IS the EuroHPC allocation (dates + budget match) |
| A4 | Delete staged legacy trash from `$WORK` and `$FAST` | Umar | ✅ **done** 2026-04-15 | `$WORK` now ~1.4M, `$FAST` now 0k |
| A5 | Create clean OSM workspace skeleton under `$CINECA_SCRATCH/osm/` and `$WORK/osm/` | Umar | ✅ **done** 2026-04-15 | `raw/`, `extracts/`, `jobs/` created |
| A6 | Download `planet-latest.osm.pbf` via Leonardo datamover | Umar | ✅ **done** 2026-04-15 | resolved to `planet-260406.osm.pbf`, checksum OK |
| A7 | Probe available OSM modules on Leonardo | Umar | ✅ **done** 2026-04-15 | `gdal` and `proj` available, no `osmium` found |
| A8 | Download Luxembourg and Iceland extracts for prototyping | Umar | **next** | Geofabrik via datamover |
| A9 | Run first GDAL OSM probe on `lrd_all_serial` | Claude + Umar | **next** | use Luxembourg first |
| A10 | Decide tokenisation scheme on Iceland/Luxembourg extract | Claude + Umar | later | §8.4 |
| A11 | Email `superc@cineca.it` for $WORK quota bump once token size is known | Umar | later | |
| A12 | Decide archive storage need | Umar | later | |
| A13 | First end-to-end prototype run on small extract | Claude + Umar | later | |

## 10. Budget linearization math

- Total: 40,000 core-hours / 3 months = **~13,333 core-hours/month** at full priority
- Per month at Booster node-hour rate (32 core-hours/node-hour): **~416 node-hours/month**
- Going over-quota in any month is **not fatal** — jobs still run, just with lower Slurm priority. So front-loading extraction+tokenisation on lrd_all_serial (budget-free) in month 1–2 and burning the GPU budget in month 2–3 is fine.

**Current burn (2026-04-15 16:40 CET):**

| Metric                          | Value                      |
| ------------------------------- | -------------------------- |
| Budget total                    | 40,000 core-h              |
| Consumed on local cluster       | **14 core-h** (0.035%)     |
| Remaining                       | ~39,986 core-h             |
| Monthly quota (linearized)      | 13,043 core-h              |
| This month consumed             | 5 core-h                   |
| Project window                  | 2026-03-11 → 2026-06-11    |

**Disk state after cleanup and planet download (2026-04-15):**

| Area             | Used     | Quota | Files  | Notes                                        |
| ---------------- | -------- | ----- | ------ | -------------------------------------------- |
| `$HOME`          | 19.54 GB | 50 GB | 3,348  | 39% full — trim before it fills up |
| `$CINECA_SCRATCH`| ~86 GB   | ∞     | 3+     | planet file + checksum + workspace dirs |
| `$WORK`          | 1.422 MB | 1 TB  | 22     | cleaned; only minimal repo/workspace state |
| `$FAST`          | 0 KB     | 1 TB  | 0      | completely cleared |
| `$PUBLIC`        | 4 KB     | 50 GB | 1      | unused |

## 11. Change log

- **2026-04-15** — Initial project doc created. Allocation facts captured from award letter. Storage/partition facts verified against `docs.hpc.cineca.it`. Access walkthrough written. Cleanup procedure staged (pending inventory). OSM pipeline outlined.
- **2026-04-15 (later)** — First `step ssh login` succeeded on Umar's Mac; cert in agent. Username corrected `usaslam00` → `uaslam00`.
- **2026-04-15 16:40 CET** — First login on `login02`. Confirmed `AIFAC_P02_222` **is** the EuroHPC allocation (dates + budget match the award letter exactly). Budget baseline: 14/40000 core-h burned. Disk baseline: $WORK 25 GB (25 GB in `containers/`), $FAST 77 GB (77 GB in `bonzai_cache/`), $HOME 19.5/50 GB, $CINECA_SCRATCH empty. Corrected doc: all four filesystem env vars ARE auto-set on Leonardo (earlier guidance was wrong). Cleanup candidates identified, awaiting user confirmation to move to trash.
- **2026-04-15 17:00 CET** — Transfer plan corrected against current CINECA docs: the 85 GB planet download should use `data.leonardo.cineca.it` datamovers, not a long-lived login-node `wget`, because login nodes have a 10-minute CPU-time limit that can interrupt large transfers. `lrd_all_serial` remains the budget-free path for extraction and other light preprocessing.
- **2026-04-15 18:xx CET** — Legacy project trash was deleted for real. `cinQuota` now shows `$WORK=1.422M` and `$FAST=0k`, confirming a clean slate.
- **2026-04-15 19:04 CET** — `planet-latest.osm.pbf` finished downloading through `data.leonardo.cineca.it` to `/leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf`. Upstream redirected to `planet-260406.osm.pbf` (snapshot date 2026-04-06). `md5sum -c planet-latest.osm.pbf.md5` returned `OK`.
- **2026-04-15 19:05 CET** — Module probe shows `gdal/3.8.5--gcc--12.2.0` and `proj/9.2.1--gcc--12.2.0-spack0.22`, but no `osmium` module. Short-term plan changed to GDAL-first region probes on `lrd_all_serial`.

---

## Appendix A — Useful Leonardo one-liners

```bash
# See what modules exist for OSM tools
module avail 2>&1 | grep -i -E 'osmium|osmosis|gdal|pyrosm|proj|geos'

# Current verified module state on 2026-04-15
# gdal/3.8.5--gcc--12.2.0(default)
# proj/9.2.1--gcc--12.2.0-spack0.22(default)
# no osmium module found

# Submit a serial (free) job
sbatch my_free_job.sh

# Watch your queue
squeue -u $USER
squeue --me                 # same thing, newer Slurm

# Cancel a job
scancel <jobid>

# See node health on what you're scheduled onto
scontrol show job <jobid>

# Tail a job's output live
tail -f osm-extract-<jobid>.out

# How much scratch am I using right now (cheap estimate)
du -sh "$CINECA_SCRATCH"/* 2>/dev/null

# Pull planet.osm.pbf through the Leonardo datamover (free, safe for long transfers)
mkdir -p "$CINECA_SCRATCH/osm/raw"
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue --progress=dot:giga \
  -O /leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf \
  https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf
# Verify checksum on the login node after the transfer lands
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue \
  -O /leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf.md5 \
  https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf.md5
cd "$CINECA_SCRATCH/osm/raw"
md5sum -c planet-latest.osm.pbf.md5
```

## Appendix B — Sources

- Leonardo User Guide: https://docs.hpc.cineca.it/hpc/leonardo.html
- CINECA Filesystems: https://docs.hpc.cineca.it/hpc/hpc_data_storage.html
- Access & 2FA: https://docs.hpc.cineca.it/general/access.html
- UserDB (register, check email): https://userdb.hpc.cineca.it/
- CINECA IdP self-service (OTP enrollment): https://sso.hpc.cineca.it
- Smallstep CLI: https://smallstep.com/docs/step-cli/installation/
- Support email: `superc@cineca.it`
- OSM Planet: https://planet.openstreetmap.org/pbf/
- Geofabrik extracts: https://download.geofabrik.de/
