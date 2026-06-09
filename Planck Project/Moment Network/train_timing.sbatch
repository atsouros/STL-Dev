#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=mn_timing
#SBATCH -o mn_timing.log
#SBATCH -e mn_timing.err

set -eo pipefail

export SLURM_CPU_BIND=cores

set +u
module unload cudatoolkit || true
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load cray-mpich craype-accel-nvidia80

module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Repository and data locations.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export DATASET_DIR="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/timing_full_epoch"

# Use the full dataset discovery path so file opening and patch selection are realistic.
export PATCH_LIST="0-191"
export EXPECTED_PATCH_COUNT="192"

# Timing-test parameters. This runs two real full-dataset epochs, one mean
# and one full-NLL, so the wall time can be scaled directly to the full run.
export TRAIN_FRACTION="0.9"
export SEED="11"
export BATCH_SIZE="8"
export NUM_WORKERS="4"
export N_EPOCHS_MEAN="1"
export N_EPOCHS_FULL="1"
export LEARNING_RATE="3e-5"
export WEIGHT_DECAY="1e-6"
export NORMALIZE="1"
export SAVE_EVERY="1"
export STOKES="auto"

# Use 0 for no cap. This is the important change: the test now measures
# complete train/validation epochs over all selected patches.
export MAX_TRAIN_BATCHES="0"
export MAX_VAL_BATCHES="0"

export RESUME=""

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python3=$(which python3)"
python3 --version
echo "start=$(date)"
echo "DATASET_DIR=${DATASET_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PATCH_LIST=${PATCH_LIST}"
echo "EXPECTED_PATCH_COUNT=${EXPECTED_PATCH_COUNT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "N_EPOCHS_MEAN=${N_EPOCHS_MEAN}"
echo "N_EPOCHS_FULL=${N_EPOCHS_FULL}"
echo "STOKES=${STOKES}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-none}"
echo "MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES}"
echo "MAX_VAL_BATCHES=${MAX_VAL_BATCHES}"
echo "NORMALIZE=${NORMALIZE}"

test -d "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun -N1 \
  -n2 \
  --ntasks-per-node=2 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  python3 -u train_network.py

echo "end=$(date)"
