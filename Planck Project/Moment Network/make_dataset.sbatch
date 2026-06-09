#!/bin/bash
#SBATCH --job-name=mn_dataset
#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --array=0-31
#SBATCH --output=mn_dataset.log
#SBATCH --error=mn_dataset.err

set -eo pipefail

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
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

# Repository and data locations.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export RESULTS_DIR="/pscratch/sd/a/atsouros/STL/planck_results/version_2"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"
export OUTPUT_DIR="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"

# Patch selection. PATCH_LIST accepts comma-separated ids/ranges, e.g. "0,1,2" or "0-15".
# PATCH_LIMIT is useful for quick debug runs after discovery/filtering.
export PATCH_LIST="0-11"
export PATCH_LIMIT="0"
export SHARD_BY_SLURM_ARRAY="1"

# Data-generation parameters.
export FREQ="353"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"
# Empty CROP_SIZE keeps the full map size. Set CROP_SIZE=256 only for explicit crop tests.
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
export OVERWRITE="1"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python3=$(which python3)"
python3 --version
echo "start=$(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-none}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "PLANCK_NUISANCE_DIR=${PLANCK_NUISANCE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PATCH_LIST=${PATCH_LIST}"
echo "PATCH_LIMIT=${PATCH_LIMIT}"
echo "SYNTHESES_PER_PATCH=${SYNTHESES_PER_PATCH}"
echo "SYNTHESIS_BATCH_SIZE=${SYNTHESIS_BATCH_SIZE}"
echo "SYNTHESIS_JOINT_QU=${SYNTHESIS_JOINT_QU}"
echo "CROSS_MATRIX=${CROSS_MATRIX}"
echo "SYNTHESIS_COMPUTE_PS=${SYNTHESIS_COMPUTE_PS}"
echo "SYNTHESIS_RUNNING_SHAPE=${SYNTHESIS_RUNNING_SHAPE}"
echo "CROP_SIZE=${CROP_SIZE:-full}"

test -d "${RESULTS_DIR}"
test -d "${PLANCK_NUISANCE_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun -u python3 -u make_dataset.py

echo "end=$(date)"
