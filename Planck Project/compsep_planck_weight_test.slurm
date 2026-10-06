#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=compsep
#SBATCH -o planck_compsep.log
#SBATCH -e planck_compsep.err
#SBATCH --export=NONE

set -eo pipefail

# Submit from the login node with:
#   bash compsep_planck_weight_test.slurm
# Override without editing: NODES=2 bash compsep_planck_weight_test.slurm
# Shell variables are not expanded in #SBATCH directives. The bash entry point
# passes the chosen allocation to sbatch; the job uses the allocated node count.
# Direct sbatch submission uses the one-node default above (or sbatch --nodes=N).
NODES="${NODES:-1}"
GPUS_PER_NODE=4
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ ! "$NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NODES must be a positive integer." >&2
    exit 1
  fi
  exec sbatch --nodes="$NODES" --ntasks="$((NODES * GPUS_PER_NODE))" "$0"
fi

JOB_NODES="${SLURM_JOB_NUM_NODES:?Missing Slurm allocation node count}"
JOB_TASKS=$((JOB_NODES * GPUS_PER_NODE))
if [[ "${SLURM_NTASKS:-$JOB_TASKS}" -ne "$JOB_TASKS" ]]; then
  echo "ERROR: Expected $JOB_TASKS tasks ($GPUS_PER_NODE per node)." >&2
  exit 1
fi

export SLURM_CPU_BIND=cores

set +u
module load pytorch
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
export PYTHONNOUSERSITE=1
set -u

cd "$SLURM_SUBMIT_DIR" || exit 1

EXPECTED_COMPSEP_MARKER="2026-10-01_stable_relative_convergence_cosecant_mask_optional"
if ! grep -q "$EXPECTED_COMPSEP_MARKER" compsep_planck_weight_test.py; then
  echo "ERROR: compsep_planck_weight_test.py is a stale cluster copy." >&2
  echo "Expected marker: $EXPECTED_COMPSEP_MARKER" >&2
  echo "Copy the updated compsep_planck_weight_test.py into $SLURM_SUBMIT_DIR before submitting." >&2
  exit 1
fi

export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPICH_GPU_SUPPORT_ENABLED=0
export HOME="${HOME:-/pscratch/sd/a/atsouros}"
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-${SLURM_SUBMIT_DIR}/.torch_kernel_cache}"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
export PYTHON_EXE="$(command -v python3 || command -v python)"
if [[ -z "${PYTHON_EXE}" ]]; then
  echo "ERROR: could not find python3 or python after loading modules" >&2
  exit 1
fi

echo "python=${PYTHON_EXE}"
"${PYTHON_EXE}" --version

unset PATCH
unset N_PATCHES
export PATCH_LIST=""
export PATCH_START=0
export PATCH_END=191

export ROOT="/pscratch/sd/e/erussie/GNILC+ST/patches"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"

# Start each patch from the observed data, not from an earlier recovered result.
unset INITIAL_RESULTS_DIR
export OUTDIR="/pscratch/sd/a/atsouros/STL/planck_results/mask_test"

export FREQ=353
export MAP_SIZE=384

# True: mask all inputs and divide recovered outputs by the mask when saving.
# False: run without loading or applying the cosecant patch mask.
export USE_PATCH_MASK=True

export BACKEND=fft
export DTYPE=float64
export WTYPE="Bump-Steerable"

export WHITE_NOISE_INITIAL=False
export BATCH=15
export EPOCHS=1
export EPOCH_STEPS=120
export LBFGS_MAX_ITER=100
export SEED=2

export CONVERGENCE_THRESHOLD=1e-3
export CONVERGENCE_RELATIVE_TOLERANCE=1e-2
export CONVERGENCE_CHECK_EVERY=5
export CONVERGENCE_PATIENCE=3

export PBC=False

export ST_REDUCED=False
export ST_J=7
export ST_L=4
export ST_ISO=False
export ST_ANGULAR_FT=True
export ST_SCALE_FT=True
export ST_HARMONICS_ANGLE=2
export ST_HARMONICS_SCALE=3
export ST_DJ=3
export ST_COMPUTE_PS=True
export PS_LOSS_WEIGHT=50
export ST_PS_METHOD=gaussian_rings
unset ST_PS_N_BINS
export ST_FEWER_CONVOLUTIONS=False

export START_WITHOUT_NOISE_CHANNELS=False

mkdir -p "$OUTDIR"

echo "Launching PATCH_LIST=$PATCH_LIST on $JOB_NODES nodes / $JOB_TASKS GPUs"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Output dir: $OUTDIR"
echo "Power-spectrum loss weight: $PS_LOSS_WEIGHT"
echo "Convergence threshold: $CONVERGENCE_THRESHOLD"
echo "Use patch mask: $USE_PATCH_MASK"
echo "Python script: $SLURM_SUBMIT_DIR/compsep_planck_weight_test.py"

srun --export=ALL -N "$JOB_NODES" \
  -n "$JOB_TASKS" \
  --ntasks-per-node="$GPUS_PER_NODE" \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  "${PYTHON_EXE}" -u compsep_planck_weight_test.py

echo "Component separation completed."
