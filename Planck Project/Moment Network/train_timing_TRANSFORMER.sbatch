#!/bin/bash
#SBATCH --job-name=MN_transformer_timing
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --qos=gpu_ens_vlong
#SBATCH --nodelist=node0298
#SBATCH --output=MN_transformer_timing.log
#SBATCH --error=MN_transformer_timing.err

set -eo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Repository and data locations on MESOPSL.
export STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
export MOMENT_NETWORK_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks"
export DATASET_DIR="/travail/atsouros/version_2"
export PLANCK_SIGNAL_DIR="/travail/atsouros/signal_I"

# Set to 1 to train with D=(d_Q,d_U,d_I). Set to 0 for the legacy Q/U input.
export USE_INTENSITY="0"
export INTENSITY_FREQ="857"
if [[ "${USE_INTENSITY}" == "1" ]]; then
  export OUTPUT_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks/moment_network_training/timing_qu_i_transformer_splitstd_nll"
else
  export OUTPUT_DIR="/obs/atsouros/projects/STL-Dev/Moment Networks/moment_network_training/timing_qu_transformer_splitstd_nll"
fi
export LEGACY_MEAN_CHECKPOINT=""

# Full dataset timing/smoke test for the transformer split-std NLL run. If this passes, the
# full run uses the same paths, environment, normalization, and model code.
export PATCH_LIST="0-191"
export EXPECTED_PATCH_COUNT="192"

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

# Live diagnostics. These use the generated dataset file for this patch/sample
# and save inference-style PNGs next to the checkpoints.
export SAVE_DIAGNOSTIC_PLOTS="1"
export DIAGNOSTIC_PATCH="3"
export DIAGNOSTIC_SAMPLE_INDEX="0"

# Use 0 for no cap. This timing test measures complete train/validation epochs.
export MAX_TRAIN_BATCHES="0"
export MAX_VAL_BATCHES="0"

export RESUME=""

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
    raise SystemExit("Need Python >= 3.10 for train_network_TRANSFORMER.py; got {}".format(sys.version.split()[0]))
import numpy
import torch
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
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

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  --export=ALL \
  "${PYTHON_EXE}" -u train_network_TRANSFORMER.py

echo "end=$(date)"
