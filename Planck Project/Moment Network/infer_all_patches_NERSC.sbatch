#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=mn_full_infer
#SBATCH -o mn_full_infer.log
#SBATCH -e mn_full_infer.err
#SBATCH --export=NONE

set -eo pipefail

export SLURM_CPU_BIND=cores

set +u
module purge
module load pytorch
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
export PYTHONNOUSERSITE=1
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_EXE="$(command -v python)"

# Repository and data locations on NERSC.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"

# Full-sky patch inference target.
export PATCH_LIST="0-191"
export FREQ="353"
export MAP_SIZE="384"
export DEVICE="cuda"

# Legacy joint version_2 network by default.
export NETWORK_MODE="joint"
export MODEL_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/version_2/models"
export JOINT_MODEL_PATH="${MODEL_DIR}/moment_network_joint_best.pth"

# If JOINT_MODEL_PATH points to a Q/U/I two-stage model, the script will also
# load I857 from PLANCK_SIGNAL_DIR.
export INTENSITY_FREQ="857"

# Output: one mean and one std array per patch, each with shape (2,384,384).
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/mn_full_sky/version_2"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "PYTHON_EXE=${PYTHON_EXE}"
command -v "${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Need Python >= 3.10 for infer_all_patches.py; got {}".format(sys.version.split()[0]))
import numpy
import torch
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device=" + torch.cuda.get_device_name(0))
PY
echo "start=$(date)"
echo "PATCH_LIST=${PATCH_LIST}"
echo "FREQ=${FREQ}"
echo "MAP_SIZE=${MAP_SIZE}"
echo "NETWORK_MODE=${NETWORK_MODE}"
echo "PLANCK_SIGNAL_DIR=${PLANCK_SIGNAL_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "JOINT_MODEL_PATH=${JOINT_MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

test -d "${PLANCK_SIGNAL_DIR}"
test -d "${MODEL_DIR}"
if [[ "${NETWORK_MODE}" == "joint" ]]; then
  test -f "${JOINT_MODEL_PATH}"
elif [[ "${NETWORK_MODE}" == "separate" ]]; then
  export Q_MODEL_PATH="${MODEL_DIR}/moment_network_q_best.pth"
  export U_MODEL_PATH="${MODEL_DIR}/moment_network_u_best.pth"
  test -f "${Q_MODEL_PATH}"
  test -f "${U_MODEL_PATH}"
else
  echo "NETWORK_MODE must be joint or separate, got ${NETWORK_MODE}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "${PYTHON_EXE}" -u infer_all_patches.py

echo "end=$(date)"
