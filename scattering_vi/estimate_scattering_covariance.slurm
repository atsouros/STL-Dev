#!/bin/bash
#SBATCH --job-name=prior_cov
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --qos=gpu_ens_normal
#SBATCH --output=prior_covariance-%j.log
#SBATCH --error=prior_covariance-%j.err


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
OUTPUT_DIR="${PROJECT_DIR}/results_prior"
INPUT_MAP="${STL_DEV_ROOT}/data/test/mock_data_samples.npy"
REFERENCE_MAP="${STL_DEV_ROOT}/langevin/results/refinement/signal_refined.npy"
NOISE_INPUT="${STL_DEV_ROOT}/data/test/100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
PYTHON_EXE="${PYTHON_EXE:-python3}"

RUN_NAME="mock_data_samples_prior_covariance"
COMPUTE_BIAS=true
BIAS_N_NOISE=0
SAVE_BIAS_STATISTICS=false

TARGET_SIZE=256
TAKE_LOG=false
WHITEN=false
BATCH_SIZE=100
N_SAMPLES=0
SAMPLE_MULTIPLIER=20
SYNTHESIS_SIZE=256
MAX_ITER=300
SEED=0
DTYPE="float64"

# Reduce BATCH_SIZE if GPU memory is insufficient.

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_MAP" ]]; then
  echo "ERROR: input map does not exist: $INPUT_MAP" >&2
  exit 2
fi
if [[ ! -d "${STL_DEV_ROOT}/STL_main" ]]; then
  echo "ERROR: STL_main does not exist under $STL_DEV_ROOT" >&2
  exit 2
fi

BIAS_ARGS=(--no-compute-bias)
if [[ "$COMPUTE_BIAS" == "true" ]]; then
  for path in "$REFERENCE_MAP" "$NOISE_INPUT"; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: bias computation requires: $path" >&2
      exit 2
    fi
  done
  BIAS_ARGS=(
    --compute-bias
    --bias-reference-map "$REFERENCE_MAP"
    --noise-input "$NOISE_INPUT"
    --bias-n-noise "$BIAS_N_NOISE"
  )
  if [[ "$SAVE_BIAS_STATISTICS" == "true" ]]; then
    BIAS_ARGS+=(--save-bias-statistics)
  else
    BIAS_ARGS+=(--no-save-bias-statistics)
  fi
fi

export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=${PYTHON_EXE}"
"$PYTHON_EXE" --version
"$PYTHON_EXE" - <<'PY'
import numpy
import torch
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device=" + torch.cuda.get_device_name(0))
PY

echo "input=${INPUT_MAP}"
echo "output=${OUTPUT_DIR}"
echo "target_size=${TARGET_SIZE}"
echo "take_log=${TAKE_LOG}"
echo "whiten=${WHITEN}"
echo "synthesis_size=${SYNTHESIS_SIZE}"
echo "batch_size=${BATCH_SIZE}"
echo "n_samples=${N_SAMPLES}"
echo "sample_multiplier=${SAMPLE_MULTIPLIER}"
echo "compute_bias=${COMPUTE_BIAS}"
echo "reference_map=${REFERENCE_MAP}"
echo "noise_input=${NOISE_INPUT}"

"$PYTHON_EXE" -u estimate_scattering_covariance.py \
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
    --batch-size "$BATCH_SIZE" \
    --n-samples "$N_SAMPLES" \
    --sample-multiplier "$SAMPLE_MULTIPLIER" \
    --synthesis-size "$SYNTHESIS_SIZE" \
    --max-iter "$MAX_ITER" \
    --lr 1.0 \
    --history-size 50 \
    --print-iter 10 \
    --no-apply-nyquist-filter \
    --covariance-mode both \
    --max-full-cov-dim 4096 \
    --save-statistics \
    --no-save-syntheses \
    "${BIAS_ARGS[@]}"

echo "Results:"
ls -lh "${OUTPUT_DIR}/${RUN_NAME}.npz" "${OUTPUT_DIR}/${RUN_NAME}.json"
echo "end=$(date)"
