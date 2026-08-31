#!/bin/bash
#SBATCH --job-name=mn_post_cal
#SBATCH -A mp107d
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=mn_post_cal.log
#SBATCH --error=mn_post_cal.err
#SBATCH --export=NONE

set -Eeo pipefail
trap 'status=$?; echo "ERROR: command failed at line ${LINENO}: ${BASH_COMMAND} (exit ${status})" >&2; exit ${status}' ERR

export SLURM_CPU_BIND=cores

set +u
# With --export=NONE the batch job starts from Slurm's clean environment.
# Do not run module purge or the CPE restore script here: on CPU nodes the
# restore script can try to swap "nvidia" even when it is not loaded.
module load pytorch
module list
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
export PYTHONNOUSERSITE=1
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PYTHON_EXE="$(command -v python)"

# NERSC paths.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="${STL_DEV_ROOT}"
export DATASET_DIR="${STL_DEV_ROOT}/moment_network_dataset/version_2"
export SYNTHESIS_DIR="${STL_DEV_ROOT}/Moment Network/validation_syntheses"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export MODEL_DIR="${STL_DEV_ROOT}/moment_network_training/version_2_qu_i_splitstd_moment_fixedmean/models"
export CHECKPOINT_PATH="${MODEL_DIR}/moment_network_joint_best.pth"
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_validation/synthetic_posterior/version_2_qu_i_splitstd_moment_fixedmean"

# What to evaluate:
#   original  -> DATASET_DIR, files like p*_moment_dataset_full.npz
#   synthetic -> SYNTHESIS_DIR, files like patch_*_validation_synthesis.npz
export EVAL_SOURCE="original"

# Which patches to evaluate:
#   all
#   0-4,75-80,187-191
#   92
export PATCH_LIST="all"

export INTENSITY_FREQ="857"
export DEVICE="cpu"
export BATCH_SIZE="16"
export HISTOGRAM_BINS="120"
export Z_LIMIT="6.0"
export SAVE_ARRAYS="1"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python=${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" - <<'PY'
import matplotlib
import numpy
import torch
print("matplotlib=" + matplotlib.__version__)
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("torch_threads=" + str(torch.get_num_threads()))
PY
echo "start=$(date)"
echo "EVAL_SOURCE=${EVAL_SOURCE}"
echo "PATCH_LIST=${PATCH_LIST}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "SYNTHESIS_DIR=${SYNTHESIS_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"
echo "OUTPUT_DIR_CREATED=${OUTPUT_DIR}"
ls -ld "${OUTPUT_DIR}"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory does not exist: $1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file does not exist: $1" >&2
    exit 1
  fi
}

if [[ "${EVAL_SOURCE}" == "original" ]]; then
  require_dir "${DATASET_DIR}"
elif [[ "${EVAL_SOURCE}" == "synthetic" ]]; then
  require_dir "${SYNTHESIS_DIR}"
else
  echo "ERROR: EVAL_SOURCE must be original or synthetic, got ${EVAL_SOURCE}" >&2
  exit 1
fi
require_file "${CHECKPOINT_PATH}"
require_file "validate_synthetic_posterior.py"

srun -N1 \
  -n1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
  --export=ALL \
  "${PYTHON_EXE}" -u validate_synthetic_posterior.py \
  2>&1 | tee "${OUTPUT_DIR}/validate_synthetic_posterior_run.log"

echo "end=$(date)"
