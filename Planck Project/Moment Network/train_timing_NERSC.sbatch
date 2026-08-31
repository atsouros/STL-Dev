#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=mn_timing_ddp
#SBATCH -o mn_timing.log
#SBATCH -e mn_timing.err
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
export OMP_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_EXE="$(command -v python)"

# Repository and data locations on NERSC.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export DATASET_DIR="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"

# Set to 1 to train with D=(d_Q,d_U,d_I). Set to 0 for the legacy Q/U input.
export USE_INTENSITY="0"
export INTENSITY_FREQ="857"
if [[ "${USE_INTENSITY}" == "1" ]]; then
  export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/timing_qu_i_splitstd_nll_ddp"
else
  export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/timing_qu_splitstd_nll_ddp"
fi
export LEGACY_MEAN_CHECKPOINT=""

# Full dataset smoke/timing test. This calls the same script and same DDP launch
# as production, but only runs one mean epoch and one full NLL epoch.
export PATCH_LIST="0-191"
export EXPECTED_PATCH_COUNT="192"

# BATCH_SIZE is per GPU/rank. With 4 GPUs, BATCH_SIZE=2 gives global batch size 8.
export TRAIN_FRACTION="0.9"
export SEED="11"
export BATCH_SIZE="2"
export NUM_WORKERS="4"
export N_EPOCHS_MEAN="1"
export N_EPOCHS_FULL="1"
export LEARNING_RATE="3e-5"
export VARIANCE_LEARNING_RATE="3e-5"
export WEIGHT_DECAY="1e-6"
export RESIDUAL_FLOOR="1e-8"
export NORMALIZE="1"
export SAVE_EVERY="1"

# Use 0 for no cap. This measures complete train/validation epochs.
export MAX_TRAIN_BATCHES="0"
export MAX_VAL_BATCHES="0"

export SAVE_DIAGNOSTIC_PLOTS="1"
export DIAGNOSTIC_PATCH="3"
export DIAGNOSTIC_SAMPLE_INDEX="0"

export RESUME=""

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "PYTHON_EXE=${PYTHON_EXE}"
command -v "${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Need Python >= 3.10 for train_network.py; got {}".format(sys.version.split()[0]))
import numpy
import torch
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
print("cuda_device_count=" + str(torch.cuda.device_count()))
if torch.cuda.is_available():
    print("cuda_device=" + torch.cuda.get_device_name(0))
PY
echo "start=$(date)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "USE_INTENSITY=${USE_INTENSITY}"
if [[ "${USE_INTENSITY}" == "1" ]]; then
  echo "PLANCK_SIGNAL_DIR=${PLANCK_SIGNAL_DIR}"
  echo "INTENSITY_FREQ=${INTENSITY_FREQ}"
fi
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LEGACY_MEAN_CHECKPOINT=${LEGACY_MEAN_CHECKPOINT}"
echo "PATCH_LIST=${PATCH_LIST}"
echo "EXPECTED_PATCH_COUNT=${EXPECTED_PATCH_COUNT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE=$((BATCH_SIZE * SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "N_EPOCHS_MEAN=${N_EPOCHS_MEAN}"
echo "N_EPOCHS_FULL=${N_EPOCHS_FULL}"
echo "VARIANCE_LEARNING_RATE=${VARIANCE_LEARNING_RATE}"
echo "MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES}"
echo "MAX_VAL_BATCHES=${MAX_VAL_BATCHES}"
echo "NORMALIZE=${NORMALIZE}"
echo "SAVE_DIAGNOSTIC_PLOTS=${SAVE_DIAGNOSTIC_PLOTS}"
echo "DIAGNOSTIC_PATCH=${DIAGNOSTIC_PATCH}"
echo "DIAGNOSTIC_SAMPLE_INDEX=${DIAGNOSTIC_SAMPLE_INDEX}"

test -d "${DATASET_DIR}"
if [[ "${USE_INTENSITY}" == "1" ]]; then
  test -d "${PLANCK_SIGNAL_DIR}"
fi
if [[ -n "${LEGACY_MEAN_CHECKPOINT}" ]]; then
  test -f "${LEGACY_MEAN_CHECKPOINT}"
fi
mkdir -p "${OUTPUT_DIR}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

srun --export=ALL \
  "${PYTHON_EXE}" -m torch.distributed.run \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --nproc-per-node="${SLURM_GPUS_PER_NODE}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  train_network.py

echo "end=$(date)"
