#!/bin/bash
#SBATCH --job-name=tile_cib
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0256
#SBATCH --output=tile_cib.log
#SBATCH --error=tile_cib.err

set -euo pipefail

# Edit only this block.
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
PROJECT_DIR="${STL_DEV_ROOT}/scattering_vi"
INPUT_MAP="${STL_DEV_ROOT}/data/test/100_Herschel_Lockman_250m_fullreso_MJysr_1500.npy"
OUTPUT_MAP="${STL_DEV_ROOT}/data/test/100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
METADATA="${OUTPUT_MAP%.npy}.json"
PYTHON_EXE="python3"
TILE_SIZE=256
TILES_PER_MAP=34

cd "$PROJECT_DIR"

if [[ ! -f "$INPUT_MAP" ]]; then
  echo "ERROR: input map does not exist: $INPUT_MAP" >&2
  exit 2
fi

echo "input=${INPUT_MAP}"
echo "output=${OUTPUT_MAP}"
echo "metadata=${METADATA}"

srun -N1 -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  "$PYTHON_EXE" -u tile_herschel_lockman.py \
    --input "$INPUT_MAP" \
    --output "$OUTPUT_MAP" \
    --metadata "$METADATA" \
    --tile-size "$TILE_SIZE" \
    --tiles-per-map "$TILES_PER_MAP"
