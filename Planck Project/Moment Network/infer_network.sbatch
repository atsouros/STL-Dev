#!/bin/bash

#SBATCH -A mp107d
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=mn_infer
#SBATCH -o mn_infer.log
#SBATCH -e mn_infer.err

set -eo pipefail

export SLURM_CPU_BIND=cores

set +u
module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# Repository and data locations.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export MODEL_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/version_2/models"
export Q_MODEL_PATH="${MODEL_DIR}/moment_network_q_best.pth"
export U_MODEL_PATH="${MODEL_DIR}/moment_network_u_best.pth"
export JOINT_MODEL_PATH="${MODEL_DIR}/moment_network_joint_best.pth"
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_inference/version_2"
export PLOTS_DIR="/pscratch/sd/a/atsouros/STL/mn_plots"

# Patch and map selection. Change PATCH here.
export PATCH="3"
export FREQ="353"
export MAP_SIZE="384"
export DEVICE="cpu"

# Use NETWORK_MODE="separate" for the current two-network implementation.
# Use NETWORK_MODE="joint" to run the old single Q/U network.
export NETWORK_MODE="separate"

# Leave empty to use NETWORK_MODE as the tag. This prevents joint/separate
# outputs from overwriting each other.
export OUTPUT_TAG=""

# Set to 1 only if you want the old Q/U std difference diagnostic figure.
export SAVE_STD_CHECK="0"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python3=$(which python3)"
python3 --version
echo "start=$(date)"
echo "PATCH=${PATCH}"
echo "FREQ=${FREQ}"
echo "MAP_SIZE=${MAP_SIZE}"
echo "NETWORK_MODE=${NETWORK_MODE}"
echo "OUTPUT_TAG=${OUTPUT_TAG:-${NETWORK_MODE}}"
echo "PLANCK_SIGNAL_DIR=${PLANCK_SIGNAL_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "Q_MODEL_PATH=${Q_MODEL_PATH}"
echo "U_MODEL_PATH=${U_MODEL_PATH}"
echo "JOINT_MODEL_PATH=${JOINT_MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PLOTS_DIR=${PLOTS_DIR}"

test -d "${PLANCK_SIGNAL_DIR}"
if [[ "${NETWORK_MODE}" == "separate" ]]; then
  test -f "${Q_MODEL_PATH}"
  test -f "${U_MODEL_PATH}"
elif [[ "${NETWORK_MODE}" == "joint" ]]; then
  test -f "${JOINT_MODEL_PATH}"
else
  echo "NETWORK_MODE must be separate or joint, got ${NETWORK_MODE}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PLOTS_DIR}"

srun -N1 \
  -n1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
  python3 -u infer_network.py

echo "end=$(date)"
