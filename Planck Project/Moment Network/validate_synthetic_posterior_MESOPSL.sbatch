#!/bin/bash
#SBATCH --job-name=MN_synth_val
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0257
#SBATCH --output=MN_synth_val.log
#SBATCH --error=MN_synth_val.err

set -eo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Repository and data locations on MESOPSL.
export STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
export MOMENT_NETWORK_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks"
export RESULTS_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks/mn_synth"
export PLANCK_NUISANCE_DIR="/travail/atsouros/planck_data/BK_CMB_S4_north_patch_v4/nuisance"
export PLANCK_SIGNAL_DIR="/travail/atsouros/signal_I"
export MODEL_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks/moment_network_training/version_2_qu_i_splitstd_nll/models"
export OUTPUT_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks/moment_network_validation/synthetic_posterior_qu_i"

# Free variables.
export PATCH="3"
export CHECKPOINT_PATH="${MODEL_DIR}/moment_network_joint_latest.pth"

# Match the current dataset-generation setup for one new joint Q/U synthesis.
export FREQ="353"
export INTENSITY_FREQ="857"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"
export SYNTHESIS_PBC="1"
export CROSS_MATRIX="1,1;0,1"
export SYNTHESIS_COMPUTE_PS="0"
export SYNTHESIS_RUNNING_SHAPE="512"
export SYNTHESIS_MAX_ITER="500"
export SYNTHESIS_LR="1.0"
export SYNTHESIS_HISTORY_SIZE="40"
export PRINT_EVERY_SYNTHESIS="1"
export VERBOSE_SYNTHESIS="1"

# Use seeds different from make_dataset.py defaults so this is not one of the
# original training syntheses.
export SYNTHESIS_SEED="10007"
export NOISE_SEED="20007"
export SAVE_MAPS="1"

# Override this in the environment if the cluster needs a specific env/conda Python.
export PYTHON_EXE="${PYTHON_EXE:-python3}"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "PYTHON_EXE=${PYTHON_EXE}"
command -v "${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Need Python >= 3.10 for validate_synthetic_posterior_MESOPSL.py; got {}".format(sys.version.split()[0]))
import numpy
import scipy
import torch
print("numpy=" + numpy.__version__)
print("scipy=" + scipy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device=" + torch.cuda.get_device_name(0))
PY
echo "start=$(date)"
echo "PATCH=${PATCH}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "PLANCK_NUISANCE_DIR=${PLANCK_NUISANCE_DIR}"
echo "PLANCK_SIGNAL_DIR=${PLANCK_SIGNAL_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SYNTHESIS_RUNNING_SHAPE=${SYNTHESIS_RUNNING_SHAPE}"
echo "SYNTHESIS_MAX_ITER=${SYNTHESIS_MAX_ITER}"
echo "SYNTHESIS_SEED=${SYNTHESIS_SEED}"
echo "NOISE_SEED=${NOISE_SEED}"

mkdir -p "${OUTPUT_DIR}"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory does not exist: $1" >&2
    exit 2
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file does not exist: $1" >&2
    exit 2
  fi
}

require_dir "${RESULTS_DIR}"
require_dir "${PLANCK_NUISANCE_DIR}"
require_dir "${PLANCK_SIGNAL_DIR}"
require_dir "${MODEL_DIR}"
require_file "${CHECKPOINT_PATH}"
require_file "${MOMENT_NETWORK_DIR}/validate_synthetic_posterior_MESOPSL.py"

echo "Input path checks passed"
echo "Component-separation candidates under RESULTS_DIR:"
find "${RESULTS_DIR}" -maxdepth 3 -type f \( -name "p${PATCH}_*.npy" -o -name "p${PATCH}_*.json" \) -print

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  --export=ALL \
  "${PYTHON_EXE}" -u validate_synthetic_posterior.py

echo "end=$(date)"
