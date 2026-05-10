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

export ROOT="/pscratch/sd/e/erussie/GNILC+ST/patches"
export PLANCK_SIGNAL_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/signal"
export PLANCK_NUISANCE_DIR="/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance"
export PLANCK_NUISANCE_VERSION="v4_10_arcmin"

unset PATCH
unset PATCH_START
unset PATCH_END
unset N_PATCHES
export PATCH_LIST="0, 1, 2, 3"

export OUTDIR="$SLURM_SUBMIT_DIR/planck_results_debug"

export FREQ="353"
export MAP_SIZE="384"

export BACKEND="fft"
export DTYPE="float64"
export WTYPE="Bump-Steerable"

export WHITE_NOISE_INITIAL="False"
export BATCH="15"
export EPOCHS="1"
export EPOCH_STEPS="60"
export LBFGS_MAX_ITER="100"
export SEED="2"

export PBC="False"

export ST_REDUCED="False"
export ST_J="7"
export ST_L="4"
export ST_ISO="False"
export ST_ANGULAR_FT="True"
export ST_SCALE_FT="True"
export ST_HARMONICS_ANGLE="2"
export ST_HARMONICS_SCALE="3"
export ST_DJ="3"
export ST_COMPUTE_PS="True"
export ST_FEWER_CONVOLUTIONS="False"

export START_WITHOUT_NOISE_CHANNELS="False"

mkdir -p "$OUTDIR"

echo "Launching one-patch run: PATCH_LIST=$PATCH_LIST"
echo "Output dir: $OUTDIR"
echo "Python script: $SLURM_SUBMIT_DIR/compsep_planck.py"

srun -N1 \
  -n4 \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  python -u compsep_planck.py
  
echo "Run completed."
