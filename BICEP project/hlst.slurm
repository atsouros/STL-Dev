#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hlst
#SBATCH -o hlst_debug.log
#SBATCH -e hlst_debug.err


set -eo pipefail

export SLURM_CPU_BIND=cores

module unload cudatoolkit
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12
module load cray-mpich craype-accel-nvidia80

module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate

export PROJECT_DIR="/pscratch/sd/a/atsouros/STL/BICEP"
export REPO_DIR="/pscratch/sd/a/atsouros/STL"

cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

export PATCH_DATA_DIR="/pscratch/sd/a/atsouros/STL/planck_results/version_2"
export OUTDIR="/pscratch/sd/a/atsouros/STL/BICEP"

# Run settings.
PATCH_IDS="129"
EXPECTED_HEIGHT="384"
EXPECTED_WIDTH="384"
ST_J="5"
ST_L="4"
ST_COMPUTE_PS="False"
ST_PS_METHOD="legacy"
MAP_NORMALIZE="False"
REF_PATCH_ID="178"
SYNTHESIS_PBC="True"
SYNTHESIS_MAX_ITER="100"
SYNTHESIS_LR="1.0"
SYNTHESIS_HISTORY_SIZE="50"
SYNTHESIS_PRINT_ITER="10"
SEED="26"

mkdir -p "$OUTDIR"

echo "Launching HL ST synthesis debug run"
echo "Patch data dir: $PATCH_DATA_DIR"
echo "Output dir: $OUTDIR"
echo "Python script: $PROJECT_DIR/hlst.py"
echo "Patch ids: $PATCH_IDS"
echo "Compute PS: $ST_COMPUTE_PS"
echo "PS method: $ST_PS_METHOD"
echo "Map normalization: $MAP_NORMALIZE"
echo "Synthesis PBC: $SYNTHESIS_PBC"

if [[ "$MAP_NORMALIZE" == "True" ]]; then
  MAP_NORM_ARG="--map-normalize"
else
  MAP_NORM_ARG="--no-map-normalize"
fi

if [[ "$SYNTHESIS_PBC" == "True" ]]; then
  SYNTH_PBC_ARG="--synthesis-pbc"
else
  SYNTH_PBC_ARG="--no-synthesis-pbc"
fi

if [[ "$ST_COMPUTE_PS" == "True" ]]; then
  PS_ARG="--compute-ps"
else
  PS_ARG=""
fi

srun -N1 \
  -n1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task=32 \
  python -u "$PROJECT_DIR/hlst.py" \
    --patch-data-dir "$PATCH_DATA_DIR" \
    --output-dir "$OUTDIR" \
    --patch-ids $PATCH_IDS \
    --expected-height "$EXPECTED_HEIGHT" \
    --expected-width "$EXPECTED_WIDTH" \
    --st-j "$ST_J" \
    --st-l "$ST_L" \
    $PS_ARG \
    --ps-method "$ST_PS_METHOD" \
    --ref-patch-id "$REF_PATCH_ID" \
    "$MAP_NORM_ARG" \
    "$SYNTH_PBC_ARG" \
    --synthesis-max-iter "$SYNTHESIS_MAX_ITER" \
    --synthesis-lr "$SYNTHESIS_LR" \
    --synthesis-history-size "$SYNTHESIS_HISTORY_SIZE" \
    --synthesis-print-iter "$SYNTHESIS_PRINT_ITER" \
    --seed "$SEED"

echo "Run completed."
