#!/bin/bash

#SBATCH -A mp107d
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=plots
#SBATCH -o planck_plots.log
#SBATCH -e planck_plots.err

set -eo pipefail

export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PYTHONUNBUFFERED=1

set +u
module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate
set -u

export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export DATASET_ROOT="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"
export PATCH="3"
export PLOTS_DIR="/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2/plots"
export N_EXAMPLES="10"
export INDEX_MODE="first"
export SEED="0"

cd "${MOMENT_NETWORK_DIR}"

mkdir -p "$PLOTS_DIR"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python3=$(which python3)"
python3 --version
echo "MOMENT_NETWORK_DIR=${MOMENT_NETWORK_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PATCH=${PATCH}"
echo "PLOTS_DIR=${PLOTS_DIR}"
echo "Dataset convention: Q network uses yq -> xq, U network uses yu -> xu"

python3 -u plot_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --patch "$PATCH" \
  --out-dir "$PLOTS_DIR" \
  --n-examples "$N_EXAMPLES" \
  --index-mode "$INDEX_MODE" \
  --seed "$SEED"
