#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=compsep_de
#SBATCH -o planck_compsep_debug.log
#SBATCH -e planck_compsep_debug.err
#SBATCH --export=NONE

set -eo pipefail

export SLURM_CPU_BIND=cores

set +u
# With --export=NONE, Slurm starts this job from a clean environment. Avoid
# module purge here because on current Perlmutter CPE it can emit restore-lmod
# warnings and leave the environment in a bad state.
module load pytorch
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
export PYTHONNOUSERSITE=1
set -u

cd "$SLURM_SUBMIT_DIR" || exit 1

EXPECTED_COMPSEP_MARKER="2026-06-29_nonreduced_apply_no_scaleft_operator"
if ! grep -q "$EXPECTED_COMPSEP_MARKER" compsep_planck.py; then
  echo "ERROR: compsep_planck.py is a stale cluster copy." >&2
  echo "Expected marker: $EXPECTED_COMPSEP_MARKER" >&2
  echo "Copy the updated compsep_planck.py into $SLURM_SUBMIT_DIR before submitting." >&2
  exit 1
fi

export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
unset PATCH_START
unset PATCH_END
unset N_PATCHES
export PATCH_LIST="0,1,2,3"

export ROOT="/pscratch/sd/e/erussie/GNILC+ST/patches"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"

# NERSC input paths.
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"

# Use the same output directory as production so write-path behavior is also tested.
export OUTDIR="$SLURM_SUBMIT_DIR/planck_results/version_3"

# Data selection.
export FREQ=353
export MAP_SIZE=384

# Component-separation parameters.
export BACKEND=fft
export DTYPE=float64
export WTYPE="Bump-Steerable"

export WHITE_NOISE_INITIAL=False
export BATCH=15
export EPOCHS=1
export EPOCH_STEPS=60
export LBFGS_MAX_ITER=100
export SEED=2

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
# Power-spectrum estimator: "legacy" for the original backend, "gaussian_rings" for log-Gaussian bins.
export ST_PS_METHOD=gaussian_rings
# Optional override. Leave unset to use the backend default; try 12 to halve the 384x384 default.
# export ST_PS_N_BINS=12
export ST_FEWER_CONVOLUTIONS=False

export START_WITHOUT_NOISE_CHANNELS=False

mkdir -p "$OUTDIR"

echo "Launching production-equivalent debug run: PATCH_LIST=$PATCH_LIST"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Output dir: $OUTDIR"
echo "Python script: $SLURM_SUBMIT_DIR/compsep_planck.py"

srun --export=ALL -N1 \
  -n4 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  "${PYTHON_EXE}" -u compsep_planck.py

echo "Production-equivalent debug run completed."
