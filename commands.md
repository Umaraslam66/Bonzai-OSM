# Bonzai-OSM — Leonardo Command Cheatsheet

Copy-paste commands, grouped by phase. Run on **Mac** unless marked `# ON LEONARDO`.
Username: `uaslam00`. Cineca email: `aslamumar16@gmail.com`. Login host: `login.leonardo.cineca.it`.

> **Env vars on Leonardo (confirmed 2026-04-15 on uaslam00's login shell).** All four auto-export on login:
> ```
> HOME          = /leonardo/home/userexternal/uaslam00
> WORK          = /leonardo_work/AIFAC_P02_222
> FAST          = /leonardo_scratch/fast/AIFAC_P02_222
> CINECA_SCRATCH= /leonardo_scratch/large/userexternal/uaslam00
> ```
> No manual export needed. If `echo $WORK` looks empty, check for an unclosed quote in your `echo` command (bash will show `>` and wait for the closing `"`).

---

## Table of contents

1. [One-time Mac setup (2FA + smallstep)](#1-one-time-mac-setup-2fa--smallstep)
2. [Every-session login](#2-every-session-login)
3. [First-login inventory on Leonardo](#3-first-login-inventory-on-leonardo)
4. [$WORK cleanup (safe, staged)](#4-work-cleanup-safe-staged)
5. [Budget & quota monitoring](#5-budget--quota-monitoring)
6. [OSM data download (budget-free)](#6-osm-data-download-budget-free)
7. [OSM extraction jobs (budget-free serial partition)](#7-osm-extraction-jobs-budget-free-serial-partition)
8. [Job management (Slurm)](#8-job-management-slurm)
9. [Data transfer between Mac ↔ Leonardo](#9-data-transfer-between-mac--leonardo)
10. [Support & escalation](#10-support--escalation)

---

## 1. One-time Mac setup (2FA + smallstep)

Only do this section **once per machine**.

### 1.1 Enroll OTP on the CINECA Identity Provider

Open in browser: **https://sso.hpc.cineca.it**
→ sign in with CINECA HPC credentials
→ follow the wizard, scan the QR code with Google Authenticator / FreeOTP / 1Password
→ save the recovery codes somewhere durable (password manager, not Downloads)

### 1.2 Install smallstep CLI

```bash
brew install step
step version
```

If Homebrew errors with `undefined method 'cellar'`:

```bash
brew update
brew tap homebrew/core
brew install step
```

### 1.3 Bootstrap the CINECA CA

```bash
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

### 1.4 Optional: `~/.ssh/config` shortcut

Append to `~/.ssh/config` so you can type `ssh leo`:

```
Host leo leonardo
    HostName login.leonardo.cineca.it
    User uaslam00
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

---

## 2. Every-session login

Certificates last **12 hours**. When your cert expires, repeat these steps.

```bash
# Make sure ssh-agent is running in this shell
eval "$(ssh-agent -s)"

# Request a fresh 12-hour cert (opens browser → OTP → cert loaded into agent)
step ssh login 'aslamumar16@gmail.com' --provisioner cineca-hpc

# Connect
ssh uaslam00@login.leonardo.cineca.it
# or, with the ssh config shortcut:
ssh leo
```

> If `step ssh login` prints *"key ... is already present in the SSH agent"* — that's **success**, not an error. Your cert is still valid. Just run the `ssh` command.

---

## 3. First-login inventory on Leonardo

Run these **on Leonardo** (after SSH'ing). Paste the output back so we can plan cleanup.

### 3.1 Identity + budget

```bash
# ON LEONARDO
whoami                           # expect: uaslam00
id
saldo -b                         # <-- lists every project account you belong to + budget
saldo --help
```

Copy the **account name(s)** that `saldo -b` prints. You'll plug them into §3.4.

### 3.2 Discover what's on disk (absolute paths — no env vars needed yet)

```bash
# ON LEONARDO — $HOME is the only filesystem var that's auto-set
echo "HOME = $HOME"
ls -la "$HOME"

# ON LEONARDO — which project work dirs can you see?
ls -la /leonardo_work/ 2>/dev/null

# ON LEONARDO — which project FAST scratch dirs exist?
ls -la /leonardo_scratch/fast/ 2>/dev/null | head -30

# ON LEONARDO — your per-user LARGE scratch (the ~20 TB free area)
ls -la /leonardo_scratch/large/$USER 2>/dev/null
ls -la /leonardo_scratch/large/ 2>/dev/null | grep "$USER"

# ON LEONARDO — overall quota picture (cintools module is auto-loaded, no vars needed)
cindata
cinQuota
```

### 3.3 Inventory the old AIFAC_P02_222 project (main cleanup target)

```bash
# ON LEONARDO
export OLD_ACCOUNT=AIFAC_P02_222
export OLD_WORK=/leonardo_work/${OLD_ACCOUNT}
export OLD_FAST=/leonardo_scratch/fast/${OLD_ACCOUNT}

ls -la "$OLD_WORK" 2>/dev/null
ls -la "$OLD_FAST" 2>/dev/null

du -sh "$OLD_WORK"   2>/dev/null
du -sh "$OLD_WORK"/* 2>/dev/null | sort -h
du -sh "$OLD_FAST"   2>/dev/null
du -sh "$OLD_FAST"/* 2>/dev/null | sort -h

# 20 biggest files under the old project $WORK
find "$OLD_WORK" -xdev -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -nr | head -20 \
  | awk '{ printf "%10.2f GB\t%s\n", $1/1024/1024/1024, substr($0, index($0,$2)) }'
```

### 3.4 Sanity-check env vars (already auto-set, this is just verification)

```bash
# ON LEONARDO
echo "HOME=$HOME"
echo "WORK=$WORK"
echo "FAST=$FAST"
echo "CINECA_SCRATCH=$CINECA_SCRATCH"
echo "PUBLIC=$PUBLIC"
ls -la "$WORK" "$FAST" "$CINECA_SCRATCH" 2>/dev/null
```

The project account name (`AIFAC_P02_222` for Umar) is baked into `$WORK` and `$FAST` by Cineca's login scripts.

---

## 4. $WORK cleanup (safe, staged)

**Rule:** `mv` to a trash dir first, actual `rm -rf` only after a grace period. GPFS has no undo.

```bash
# ON LEONARDO — helper script from this repo
# ./scripts/leonardo_cleanup.sh --inventory-only
# ./scripts/leonardo_cleanup.sh --stage-defaults

# ON LEONARDO — stage a trash dir under the OLD project (we're cleaning AIFAC_P02_222)
TRASH="$OLD_WORK/_trash_$(date +%Y%m%d)"
mkdir -p "$TRASH"
echo "Trash: $TRASH"

# ON LEONARDO — move useless items to trash
#   ADJUST the names to match what du -sh printed in §3.3
# mv "$OLD_WORK/containers" "$TRASH/"
# mv "$OLD_WORK/bonzai"     "$TRASH/"

# ON LEONARDO — same trash treatment for FAST scratch
FAST_TRASH="$OLD_FAST/_trash_$(date +%Y%m%d)"
mkdir -p "$FAST_TRASH"
# mv "$OLD_FAST/bonzai_cache" "$FAST_TRASH/"

# ON LEONARDO — (after a grace period) nuke trash
# rm -rf "$TRASH"
# rm -rf "$FAST_TRASH"
```

---

## 5. Budget & quota monitoring

```bash
# ON LEONARDO — core-hour budget
saldo -b                         # total / monthly quota / consumed / remaining
saldo -ba                        # per-account breakdown
saldo --help

# ON LEONARDO — disk usage across all areas (no env vars needed)
cindata
cinQuota

# ON LEONARDO — scratch usage (requires the exports from §3.4)
du -sh "$CINECA_SCRATCH"/* 2>/dev/null | sort -h
du -sh "$FAST"/* 2>/dev/null           | sort -h

# ON LEONARDO — hunt files approaching the 40-day auto-delete on $CINECA_SCRATCH
find "$CINECA_SCRATCH" -type f -mtime +30 -printf '%T+ %p\n' 2>/dev/null | sort | head -30
```

---

## 6. OSM data download (budget-free)

Use the **Leonardo datamover** for the 85 GB planet file. CINECA allows small transfers on login nodes, but the official docs state login nodes have a **10-minute CPU-time limit** that can interrupt large downloads; datamovers are designed for long transfers and still do not burn project allocation.

```bash
# ON LEONARDO LOGIN NODE — prep staging area on scratch
mkdir -p "$CINECA_SCRATCH/osm/raw"

# ON LEONARDO LOGIN NODE — launch the long transfer on the datamover
# Use -O with an absolute destination path to avoid shell/path splitting issues.
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue --progress=dot:giga \
  -O /leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf \
  https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf

# ON LEONARDO LOGIN NODE — fetch checksum via datamover, then verify locally
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue \
  -O /leonardo_scratch/large/userexternal/uaslam00/osm/raw/planet-latest.osm.pbf.md5 \
  https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf.md5
cd "$CINECA_SCRATCH/osm/raw"
md5sum -c planet-latest.osm.pbf.md5

# ON LEONARDO LOGIN NODE — helper script from this repo
# ./scripts/leonardo_download_planet.sh

# ON LEONARDO LOGIN NODE — optional smaller Geofabrik extracts for prototyping
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue https://download.geofabrik.de/europe/iceland-latest.osm.pbf \
  -P "$CINECA_SCRATCH/osm/raw"
ssh -xt "$USER"@data.leonardo.cineca.it \
  wget --continue https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf \
  -P "$CINECA_SCRATCH/osm/raw"
```

---

## 7. OSM extraction jobs (budget-free serial partition)

`lrd_all_serial` is **budget-free**: 4 cores, 30.8 GB RAM, 4h walltime, unlimited submissions.

### 7.1 Discover available OSM tooling

```bash
# ON LEONARDO — what's already installed as a module
module avail 2>&1 | grep -i -E 'osmium|osmosis|gdal|pyrosm|proj|geos'

# ON LEONARDO — verified 2026-04-15:
#   gdal/3.8.5--gcc--12.2.0(default)
#   proj/9.2.1--gcc--12.2.0-spack0.22(default)
#   no osmium module found
module load gdal/3.8.5--gcc--12.2.0
ogrinfo --formats | grep -i OSM
```

### 7.2 First free probe job

Save as `$WORK/osm/jobs/luxembourg_probe.sbatch` or reuse [`jobs/luxembourg_probe.sbatch`](./jobs/luxembourg_probe.sbatch):

```bash
#!/bin/bash
#SBATCH --job-name=lux-probe
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load gdal/3.8.5--gcc--12.2.0

RAW=/leonardo_scratch/large/userexternal/uaslam00/osm/raw/luxembourg-latest.osm.pbf
ogrinfo "$RAW" -so points
ogrinfo "$RAW" -so lines
ogrinfo "$RAW" -so multipolygons
```

### 7.3 Future `osmium` extraction path

[`jobs/leonardo_osm_extract.sbatch`](./jobs/leonardo_osm_extract.sbatch) is still kept in the repo as the future template once `osmium` is installed or built in user space. Do not submit it yet on Leonardo, because no `osmium` module has been confirmed for this account.

### 7.4 Submit the probe

```bash
# ON LEONARDO
JOB1=$(sbatch --parsable "$WORK/osm/jobs/luxembourg_probe.sbatch")
echo "Submitted: $JOB1"

# Inspect queue / output
squeue --me
tail -f lux-probe-"$JOB1".out
```

---

## 8. Job management (Slurm)

```bash
# ON LEONARDO — your queue
squeue --me
squeue -u $USER

# ON LEONARDO — detailed view of one job
scontrol show job <jobid>

# ON LEONARDO — cancel
scancel <jobid>
scancel -u $USER                 # cancels ALL your jobs — careful

# ON LEONARDO — live tail of job output
tail -f osm-extract-<jobid>.out

# ON LEONARDO — historical queries after job finishes (squeue no longer shows it)
sacct -u $USER --starttime=$(date -d '1 day ago' +%Y-%m-%d) \
      --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode

# ON LEONARDO — cost per finished job
sacct -u $USER --format=JobID,JobName,Elapsed,AllocCPUS,State
```

---

## 9. Data transfer between Mac ↔ Leonardo

### 9.1 Mac → Leonardo (rsync, resumable)

```bash
# ON MAC — push the Bonzai-OSM repo (adjust paths once NEW_ACCOUNT is known)
rsync -avz --partial --progress \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude data \
  ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/<NEW_ACCOUNT>/bonzai-osm/
```

### 9.2 Leonardo → Mac (pull results)

```bash
# ON MAC — pull checkpoints / samples
rsync -avz --partial --progress \
  uaslam00@login.leonardo.cineca.it:/leonardo_work/<NEW_ACCOUNT>/bonzai-osm/outputs/ \
  ./outputs/
```

### 9.3 Single-file scp

```bash
# ON MAC
scp ./local-file.tar uaslam00@login.leonardo.cineca.it:/leonardo_work/<NEW_ACCOUNT>/
scp uaslam00@login.leonardo.cineca.it:/leonardo_work/<NEW_ACCOUNT>/remote-file ./
```

---

## 10. Support & escalation

| Need                                             | Contact                                     |
| ------------------------------------------------ | ------------------------------------------- |
| Quota increase on `$WORK` (up to allocation max) | `superc@cineca.it` — include project name, size, justification |
| Enable archive storage                           | `superc@cineca.it`                          |
| Add a collaborator to the project                | PI adds them at https://userdb.hpc.cineca.it/ |
| EuroHPC allocation extension                     | `access@eurohpc-ju.europa.eu`               |
| General CINECA User Support                      | `superc@cineca.it`                          |

### Email template — $WORK quota increase

> Subject: $WORK quota increase request — project `<PROJECT_NAME>`
>
> Dear CINECA User Support,
>
> I'm Umar Aslam (`uaslam00`), PI of EuroHPC project `<PROJECT_NAME>` on Leonardo. I'd like to increase the `$WORK` quota from the 1 TB default to `<N>` TB, which is within the storage budget granted in our allocation.
>
> The project trains a generative model on the full OpenStreetMap planet dataset. Tokenised training shards are estimated at `<N>` TB and must persist on `$WORK` throughout the project duration (raw PBFs live on `$CINECA_SCRATCH`).
>
> Thank you,
> Umar Aslam
