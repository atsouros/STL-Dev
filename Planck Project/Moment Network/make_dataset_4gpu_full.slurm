#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=mn_data_full
#SBATCH -o mn_dataset_full.log
#SBATCH -e mn_dataset_full.err

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
export RESULTS_DIR="/pscratch/sd/a/atsouros/STL/planck_results/version_2"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"

# Full patch selection. Patches 0-3 were already produced by the debug run.
export PATCH_LIST="4-191"
export PATCH_LIMIT="0"
export SHARD_BY_SLURM_ARRAY="0"
export SHARD_BY_SLURM_TASKS="1"

# Data-generation parameters.
export FREQ="353"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"
export CROP_SIZE=""
export SYNTHESES_PER_PATCH="10"
export SYNTHESIS_BATCH_SIZE="5"
export SYNTHESIS_PBC="1"
export SYNTHESIS_JOINT_QU="1"
export CROSS_MATRIX="1,1;0,1"
export SYNTHESIS_COMPUTE_PS="0"
export SYNTHESIS_RUNNING_SHAPE="512"
export SYNTHESIS_MAX_ITER="500"
export SYNTHESIS_LR="1.0"
export SYNTHESIS_HISTORY_SIZE="40"
export PRINT_EVERY_SYNTHESIS="1"
export VERBOSE_SYNTHESIS="1"
export N_AUGMENTATIONS="8"
export SHIFT_STEP="16"
export SEED="7"
export OVERWRITE="0"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python3=$(which python3)"
python3 --version
echo "start=$(date)"
echo "PATCH_LIST=${PATCH_LIST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-none}"
echo "SYNTHESES_PER_PATCH=${SYNTHESES_PER_PATCH}"
echo "SYNTHESIS_BATCH_SIZE=${SYNTHESIS_BATCH_SIZE}"
echo "SYNTHESIS_JOINT_QU=${SYNTHESIS_JOINT_QU}"
echo "CROSS_MATRIX=${CROSS_MATRIX}"
echo "SYNTHESIS_COMPUTE_PS=${SYNTHESIS_COMPUTE_PS}"
echo "SYNTHESIS_RUNNING_SHAPE=${SYNTHESIS_RUNNING_SHAPE}"

test -d "${RESULTS_DIR}"
test -d "${PLANCK_NUISANCE_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun -N1 \
  -n4 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  python3 -u make_dataset.py

echo "end=$(date)"
