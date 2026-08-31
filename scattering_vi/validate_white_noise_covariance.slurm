#!/bin/bash
#SBATCH --job-name=grf_cov_val
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0256
#SBATCH --output=power_law_grf_covariance.log
#SBATCH --error=power_law_grf_covariance.err


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
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results_power_law_grf}"
export PYTHON_EXE="${PYTHON_EXE:-python3}"

export N_REFERENCE="${N_REFERENCE:-10000}"
export REFERENCE_CHUNK_SIZE="${REFERENCE_CHUNK_SIZE:-64}"
export BATCH_SIZE="${BATCH_SIZE:-100}"
export N_SAMPLES="${N_SAMPLES:-0}"
export SAMPLE_MULTIPLIER="${SAMPLE_MULTIPLIER:-20}"
export BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-200}"
export MAX_ITER="${MAX_ITER:-300}"
export DTYPE="${DTYPE:-float64}"
export FIELD_STD="${FIELD_STD:-1.0}"
export SPECTRAL_INDEX="${SPECTRAL_INDEX:--3.6666666666666665}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"
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

echo "reference_maps=${N_REFERENCE}"
echo "mean_field_batch=${BATCH_SIZE}"
echo "n_samples=${N_SAMPLES}"
echo "sample_multiplier=${SAMPLE_MULTIPLIER}"
echo "bootstrap_replicates=${BOOTSTRAP_REPLICATES}"
echo "spectral_index=${SPECTRAL_INDEX}"
echo "output=${OUTPUT_DIR}"

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "$PYTHON_EXE" -u validate_white_noise_covariance.py \
    --output-dir "$OUTPUT_DIR" \
    --run-name power_law_grf_covariance_validation \
    --size 128 \
    --field-std "$FIELD_STD" \
    --spectral-index "$SPECTRAL_INDEX" \
    --n-reference "$N_REFERENCE" \
    --reference-chunk-size "$REFERENCE_CHUNK_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --n-samples "$N_SAMPLES" \
    --sample-multiplier "$SAMPLE_MULTIPLIER" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --device cuda:0 \
    --dtype "$DTYPE" \
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
    --print-iter 10 \
    --no-apply-nyquist-filter \
    --max-full-cov-dim 2048 \
    --no-augment-symmetries \
    --sampled-stat-synthesis-test \
    --save-syntheses \
    --no-save-reference-statistics

echo "end=$(date)"
