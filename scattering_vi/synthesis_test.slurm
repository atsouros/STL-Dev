#!/bin/bash
#SBATCH --job-name=synthesis_test
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
#SBATCH --output=synthesis_test.log
#SBATCH --error=synthesis_test.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

export STL_DEV_ROOT="${STL_DEV_ROOT:-/obs/atsouros/projects/STL-Dev}"
export PROJECT_DIR="${PROJECT_DIR:-${STL_DEV_ROOT}/scattering_vi}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results_synthesis_test}"
export PYTHON_EXE="${PYTHON_EXE:-python3}"

export INPUT_MAP="${INPUT_MAP:-/obs/atsouros/projects/STL-Dev/data/test/lss.npy}"
INPUT_STEM="$(basename "$INPUT_MAP")"
INPUT_STEM="${INPUT_STEM%.npy}"
export RUN_NAME="${RUN_NAME:-${INPUT_STEM}_synthesis_test}"

export TARGET_SIZE="${TARGET_SIZE:-256}"
export TAKE_LOG="${TAKE_LOG:-false}"
export WHITEN="${WHITEN:-false}"
export MAX_ITER="${MAX_ITER:-300}"
export SEED="${SEED:-70001}"
export DTYPE="${DTYPE:-float64}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "$PYTHON_EXE" -u synthesis_test.py \
    --input "$INPUT_MAP" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --target-size "$TARGET_SIZE" \
    --take-log "$TAKE_LOG" \
    --whiten "$WHITEN" \
    --device cuda:0 \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    --J 7 \
    --L 4 \
    --iso \
    --angular-ft \
    --harmonics-angle 2 \
    --scale-ft \
    --harmonics-scale 3 \
    --dj 3 \
    --no-compute-ps \
    --pbc \
    --max-iter "$MAX_ITER" \
    --lr 1.0 \
    --history-size 50 \
    --print-iter 10

echo "end=$(date)"
