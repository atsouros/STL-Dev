#!/bin/bash
#SBATCH --job-name=syn
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0256
#SBATCH --output=syn.log
#SBATCH --error=syn.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Edit only this block.
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
PROJECT_DIR="${STL_DEV_ROOT}/scattering_vi"
OUTPUT_DIR="${PROJECT_DIR}/results_syn"
INPUT_MAP="${STL_DEV_ROOT}/data/test/100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
PYTHON_EXE="python3"

TARGET_SIZE=256
SUBTRACT_MEAN=false
TAKE_LOG=false
WHITEN=false
ISO=true
ANGULAR_FT=true
SCALE_FT=true
HARMONICS_ANGLE=2
HARMONICS_SCALE=3
COMPUTE_PS=false
FEWER_CONVOLUTIONS=false
PBC=true
MASK_SMALL_VARIANCE=true
VARIANCE_RELATIVE_TOL=1e-10
MAX_ITER=300
SEED=70001
DTYPE="float64"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "$INPUT_MAP" ]]; then
  echo "ERROR: input map does not exist: $INPUT_MAP" >&2
  exit 2
fi

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "$PYTHON_EXE" -u syn.py \
    --input "$INPUT_MAP" \
    --output-dir "$OUTPUT_DIR" \
    --target-size "$TARGET_SIZE" \
    --subtract-mean "$SUBTRACT_MEAN" \
    --take-log "$TAKE_LOG" \
    --whiten "$WHITEN" \
    --device cuda:0 \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --J 7 \
    --L 4 \
    --iso "$ISO" \
    --angular-ft "$ANGULAR_FT" \
    --harmonics-angle "$HARMONICS_ANGLE" \
    --scale-ft "$SCALE_FT" \
    --harmonics-scale "$HARMONICS_SCALE" \
    --dj 3 \
    --compute-ps "$COMPUTE_PS" \
    --fewer-convolutions "$FEWER_CONVOLUTIONS" \
    --pbc "$PBC" \
    --mask-small-variance "$MASK_SMALL_VARIANCE" \
    --variance-relative-tol "$VARIANCE_RELATIVE_TOL" \
    --max-iter "$MAX_ITER" \
    --lr 1.0 \
    --history-size 50 \
    --print-iter 10

echo "end=$(date)"
