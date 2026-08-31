#!/bin/bash

#SBATCH -A mp107d
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=signal_I
#SBATCH -o signal_I.log
#SBATCH -e signal_I.err

set -eo pipefail

export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Keep Python startup clean. This preprocessing job only needs Python + NumPy,
# not cmbenv/STL/PyTorch.
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export MOMENT_NETWORK_DIR="/pscratch/sd/a/atsouros/STL/Moment Network"
export SOURCE_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export OUT_SIGNAL_I_DIR="/pscratch/sd/a/atsouros/STL/signal_I"
export PYTHON_EXE="${PYTHON_EXE:-python3}"

export PATCH_LIST="0-191"
export EXPECTED_PATCH_COUNT="192"
export INTENSITY_FREQ="857"
export MAP_SIZE="384"
export OUTPUT_DTYPE="float32"
export OVERWRITE="0"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "PYTHON_EXE=${PYTHON_EXE}"
command -v "${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" -c 'import numpy; print("numpy=" + numpy.__version__)'
echo "start=$(date)"
echo "SOURCE_SIGNAL_DIR=${SOURCE_SIGNAL_DIR}"
echo "OUT_SIGNAL_I_DIR=${OUT_SIGNAL_I_DIR}"
echo "PATCH_LIST=${PATCH_LIST}"
echo "EXPECTED_PATCH_COUNT=${EXPECTED_PATCH_COUNT}"
echo "INTENSITY_FREQ=${INTENSITY_FREQ}"
echo "MAP_SIZE=${MAP_SIZE}"
echo "OUTPUT_DTYPE=${OUTPUT_DTYPE}"
echo "OVERWRITE=${OVERWRITE}"

test -d "${SOURCE_SIGNAL_DIR}"
mkdir -p "${OUT_SIGNAL_I_DIR}"

overwrite_args=()
if [ "${OVERWRITE}" = "1" ]; then
  overwrite_args+=(--overwrite)
fi

"${PYTHON_EXE}" -u prepare_signal_i.py \
  --signal-dir "${SOURCE_SIGNAL_DIR}" \
  --out-dir "${OUT_SIGNAL_I_DIR}" \
  --patches "${PATCH_LIST}" \
  --expected-patch-count "${EXPECTED_PATCH_COUNT}" \
  --intensity-freq "${INTENSITY_FREQ}" \
  --map-size "${MAP_SIZE}" \
  --dtype "${OUTPUT_DTYPE}" \
  "${overwrite_args[@]}"

echo "end=$(date)"
