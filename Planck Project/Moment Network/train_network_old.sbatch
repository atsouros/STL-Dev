#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=mn_train
#SBATCH -o mn_train.log
#SBATCH -e mn_train.err

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
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_training/version_2"

# Patch selection. Full shared network over all generated patches.
export PATCH_LIST="0-191"
export EXPECTED_PATCH_COUNT="192"

# Training parameters.
export TRAIN_FRACTION="0.9"
export SEED="11"
export BATCH_SIZE="8"
export NUM_WORKERS="4"
export N_EPOCHS_MEAN="10"
export N_EPOCHS_FULL="90"
export LEARNING_RATE="3e-5"
export WEIGHT_DECAY="1e-6"
export NORMALIZE="1"
export SAVE_EVERY="5"

# Optional debug limits. Use 0 for no limit.
export MAX_TRAIN_BATCHES="0"
export MAX_VAL_BATCHES="0"

# Optional checkpoint path for continuing an interrupted run.
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
echo "MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES}"
echo "MAX_VAL_BATCHES=${MAX_VAL_BATCHES}"
echo "RESUME=${RESUME}"

test -d "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  python3 -u train_network.py

echo "end=$(date)"
