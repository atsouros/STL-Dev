#!/bin/bash
#SBATCH --job-name=validate_prior
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --qos=gpu_ens_normal
#SBATCH --output=validate_prior-%j.log
#SBATCH --error=validate_prior-%j.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Edit only this block.
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
PROJECT_DIR="${STL_DEV_ROOT}/scattering_vi"
OUTPUT_DIR="${PROJECT_DIR}/results_prior_validation"
PYTHON_EXE="${PYTHON_EXE:-python3}"

INPUT_MAP="${STL_DEV_ROOT}/langevin/results/refinement/signal_refined.npy"
COVARIANCE_NPZ="${PROJECT_DIR}/results_prior/mock_data_samples_prior_covariance.npz"
COVARIANCE_JSON="${PROJECT_DIR}/results_prior/mock_data_samples_prior_covariance.json"
RUN_NAME="mock_data_samples_prior_sampled_stat_synthesis"
MAX_ITER=600
SEED=60001

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "$INPUT_MAP" ]]; then
  echo "ERROR: input map does not exist: $INPUT_MAP" >&2
  exit 2
fi
if [[ ! -f "$COVARIANCE_NPZ" ]]; then
  echo "ERROR: covariance npz does not exist: $COVARIANCE_NPZ" >&2
  exit 2
fi
if [[ ! -f "$COVARIANCE_JSON" ]]; then
  echo "ERROR: covariance json does not exist: $COVARIANCE_JSON" >&2
  exit 2
fi

"$PYTHON_EXE" -u validate_covariance.py \
    --covariance-npz "$COVARIANCE_NPZ" \
    --covariance-json "$COVARIANCE_JSON" \
    --input "$INPUT_MAP" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --n-samples 7 \
    --seed "$SEED" \
    --device cuda:0 \
    --max-iter "$MAX_ITER" \
    --lr 1.0 \
    --history-size 50 \
    --print-iter 10

echo "end=$(date)"
