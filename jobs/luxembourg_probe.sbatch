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
