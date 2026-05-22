#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=compsep
#SBATCH -o planck_compsep.log
#SBATCH -e planck_compsep.err

set -eo pipefail

export SLURM_CPU_BIND=cores

module unload cudatoolkit
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load cray-mpich craype-accel-nvidia80

module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate

cd "$SLURM_SUBMIT_DIR" || exit 1

export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATCH_START=0
export PATCH_END=192


export PLANCK_NUISANCE_VERSION="v4_10_arcmin"


# NERSC input paths.
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"

# Save outputs next to this Slurm submission directory.
export OUTDIR="$SLURM_SUBMIT_DIR/planck_results/version_2"

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
export ST_PS_METHOD=legacy
# Optional override. Leave unset to use the backend default; try 12 to halve the 384x384 default.
# export ST_PS_N_BINS=12
export ST_FEWER_CONVOLUTIONS=False

export START_WITHOUT_NOISE_CHANNELS=False

echo "Launching production run: patches ${PATCH_START}-${PATCH_END} on 4 GPUs"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Output dir: $OUTDIR"
echo "Python script: $SLURM_SUBMIT_DIR/compsep_planck.py"

for chunk_start in $(seq "$PATCH_START" 4 "$PATCH_END"); do
  chunk_end=$((chunk_start + 3))
  if [ "$chunk_end" -gt "$PATCH_END" ]; then
    chunk_end="$PATCH_END"
  fi

  patch_list=""
  for patch in $(seq "$chunk_start" "$chunk_end"); do
    if [ -z "$patch_list" ]; then
      patch_list="$patch"
    else
      patch_list="${patch_list},${patch}"
    fi
  done

  export PATCH_LIST="$patch_list"
  echo "Launching patch chunk: PATCH_LIST=$PATCH_LIST"

  srun -N1 \
    -n4 \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=32 \
    python -u compsep_planck.py
done

unset PATCH_LIST

echo "Production run completed."
