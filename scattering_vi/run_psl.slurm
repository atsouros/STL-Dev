#!/bin/bash
#SBATCH --job-name=scat_psl
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
#SBATCH --output=scat_psl.log
#SBATCH --error=scat_psl.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Edit only this block.
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
PROJECT_DIR="${STL_DEV_ROOT}/scattering_vi"
PYTHON_EXE="python3"

COVARIANCE_NPZ="${PROJECT_DIR}/results/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.npz"
COVARIANCE_JSON="${PROJECT_DIR}/results/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.json"
DATA_INPUT="${STL_DEV_ROOT}/data/test/mock_data.npy"
INITIAL_SIGNAL="${STL_DEV_ROOT}/langevin/results/component_separation/recovered_signal.npy"

DATA_INDEX=0
# N_SAMPLES=2
# VI_ITERATIONS=1
# MAP_ITERATIONS=2
# KL_ITERATIONS=3
# SAMPLING_ITERATIONS=3

N_SAMPLES=12
VI_ITERATIONS=8
MAP_ITERATIONS=40
KL_ITERATIONS=100
SAMPLING_ITERATIONS=300
INITIALIZATION_ITERATIONS=40
INITIALIZATION_RELATIVE_NOISE=0.05

PRIOR_LOGLOG_SLOPE_MEAN=-1.53876
PRIOR_LOGLOG_SLOPE_STD=0.3
PRIOR_FLUCTUATIONS_MEAN=2.53272
PRIOR_FLUCTUATIONS_STD=1.26636

OUT_FILE="mock_data_initialized_cib_likelihood_posterior.npz"
PLOT_FILE="mock_data_initialized_cib_likelihood_posterior.png"
SUMMARY_FILE="mock_data_initialized_cib_likelihood_summary.json"

cd "$PROJECT_DIR"
export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

for path in "$COVARIANCE_NPZ" "$COVARIANCE_JSON" "$DATA_INPUT" "$INITIAL_SIGNAL"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 2
  fi
done

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=${PYTHON_EXE}"
"$PYTHON_EXE" --version
nvidia-smi

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "$PYTHON_EXE" -u microsample.py \
    --device cuda:0 \
    --dtype float64 \
    --covariance-npz "$COVARIANCE_NPZ" \
    --covariance-json "$COVARIANCE_JSON" \
    --data-input "$DATA_INPUT" \
    --initial-signal "$INITIAL_SIGNAL" \
    --initialization-iterations "$INITIALIZATION_ITERATIONS" \
    --initialization-relative-noise "$INITIALIZATION_RELATIVE_NOISE" \
    --data-index "$DATA_INDEX" \
    --n-samples "$N_SAMPLES" \
    --vi-iterations "$VI_ITERATIONS" \
    --map-iterations "$MAP_ITERATIONS" \
    --kl-iterations "$KL_ITERATIONS" \
    --sampling-iterations "$SAMPLING_ITERATIONS" \
    --run-map \
    --run-vi \
    --use-correlated-field-prior \
    --prior-loglog-slope-mean "$PRIOR_LOGLOG_SLOPE_MEAN" \
    --prior-loglog-slope-std "$PRIOR_LOGLOG_SLOPE_STD" \
    --prior-fluctuations-mean "$PRIOR_FLUCTUATIONS_MEAN" \
    --prior-fluctuations-std "$PRIOR_FLUCTUATIONS_STD" \
    --out "$OUT_FILE" \
    --plot "$PLOT_FILE" \
    --diagnostics-summary "$SUMMARY_FILE"

echo "end=$(date)"
