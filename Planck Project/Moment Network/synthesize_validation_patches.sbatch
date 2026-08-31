#!/bin/bash
#SBATCH --job-name=mn_val_synth
#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=mn_val_synth.log
#SBATCH --error=mn_val_synth.err
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

# NERSC paths.
export STL_DEV_ROOT="/pscratch/sd/a/atsouros/STL"
SCRIPT_DIR_CANDIDATES=(
  "${SLURM_SUBMIT_DIR:-}"
  "${STL_DEV_ROOT}"
  "${STL_DEV_ROOT}/Moment Network"
)
export MOMENT_NETWORK_DIR=""
for candidate in "${SCRIPT_DIR_CANDIDATES[@]}"; do
  if [[ -n "${candidate}" \
      && -f "${candidate}/synthesize_validation_patches.py" ]]; then
    export MOMENT_NETWORK_DIR="${candidate}"
    break
  fi
done
if [[ -z "${MOMENT_NETWORK_DIR}" ]]; then
  echo "ERROR: could not find synthesize_validation_patches.py." >&2
  echo "Checked: ${SCRIPT_DIR_CANDIDATES[*]}" >&2
  exit 1
fi
export RESULTS_DIR="/pscratch/sd/a/atsouros/STL/planck_results/version_2"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"
export OUTPUT_DIR="${STL_DEV_ROOT}/Moment Network/validation_syntheses"

# Synthesis settings, matching the existing single-patch validation.
export FREQ="353"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"
export SYNTHESIS_PBC="1"
export CROSS_MATRIX="1,1;0,1"
export SYNTHESIS_COMPUTE_PS="0"
export SYNTHESIS_RUNNING_SHAPE="512"
export SYNTHESIS_MAX_ITER="500"
export SYNTHESIS_LR="1.0"
export SYNTHESIS_HISTORY_SIZE="40"
export PRINT_EVERY_SYNTHESIS="1"
export VERBOSE_SYNTHESIS="1"
export SYNTHESIS_SEED="10007"
export NOISE_SEED="20007"
export OVERWRITE="0"
export EXPECTED_MAP_SIZE="384"

cd "${MOMENT_NETWORK_DIR}"

echo "pwd=$(pwd)"
echo "host=$(hostname)"
echo "python=${PYTHON_EXE}"
"${PYTHON_EXE}" --version
"${PYTHON_EXE}" - <<'PY'
import torch
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device=" + torch.cuda.get_device_name(0))
PY
echo "start=$(date)"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EXPECTED_MAP_SIZE=${EXPECTED_MAP_SIZE}"

test -d "${RESULTS_DIR}"
test -d "${PLANCK_NUISANCE_DIR}"
test -f "synthesize_validation_patches.py"
mkdir -p "${OUTPUT_DIR}"

# Four consecutive srun steps, each synthesizing four patches concurrently.
PATCH_GROUPS=(
  "0,1,2,3"
  "4,75,76,77"
  "78,79,80,187"
  "188,189,190,191"
)

for patch_group in "${PATCH_GROUPS[@]}"; do
  export PATCH_LIST="${patch_group}"
  echo "Launching PATCH_LIST=${PATCH_LIST}"
  srun -N1 \
    -n4 \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=32 \
    --export=ALL \
    "${PYTHON_EXE}" -u synthesize_validation_patches.py
done

unset PATCH_LIST
echo "end=$(date)"
