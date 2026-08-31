#!/bin/bash
#SBATCH --job-name=plot_val
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0298
#SBATCH --output=plot_val.log
#SBATCH --error=plot_val.err

set -euo pipefail

# Edit only this block.
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
PROJECT_DIR="${STL_DEV_ROOT}/scattering_vi"
PYTHON_EXE="python3"

INPUT_NPZ="${PROJECT_DIR}/results_validation/100_Herschel_Lockman_250m_tiles_3400x256x256_sampled_stat_synthesis.npz"
OUTPUT_PNG="${PROJECT_DIR}/results_validation/100_Herschel_Lockman_250m_tiles_3400x256x256_sampled_stat_synthesis_fixed_scale.png"
VMIN=-6
VMAX=6
SUBTRACT_MEAN=true

cd "$PROJECT_DIR"

if [[ ! -f "$INPUT_NPZ" ]]; then
  echo "ERROR: missing input NPZ: $INPUT_NPZ" >&2
  exit 2
fi

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --export=ALL \
  "$PYTHON_EXE" -u plot_validation_fixed_scale.py \
    --input "$INPUT_NPZ" \
    --output "$OUTPUT_PNG" \
    --vmin "$VMIN" \
    --vmax "$VMAX" \
    --subtract-mean "$SUBTRACT_MEAN"
