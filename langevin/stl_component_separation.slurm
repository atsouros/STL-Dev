#!/bin/bash
#SBATCH --job-name=stl_compsep
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
#SBATCH --output=component_separation.log
#SBATCH --error=component_separation.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
LANGEVIN_DIR="${STL_DEV_ROOT}/langevin"
DATA="${STL_DEV_ROOT}/data/test/mock_data.npy"
CIB_BANK="${STL_DEV_ROOT}/data/test/100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
OUTPUT_DIR="${LANGEVIN_DIR}/results/component_separation"
PYTHON_EXE="${PYTHON_EXE:-python3}"

for path in "$DATA" "$CIB_BANK" "${LANGEVIN_DIR}/stl_component_separation.py"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
cd "$LANGEVIN_DIR"
export PYTHONPATH="${STL_DEV_ROOT}:${PYTHONPATH:-}"

echo "host=$(hostname)"
echo "python=${PYTHON_EXE}"
echo "STL component separation with empirical CIB realizations"
"$PYTHON_EXE" --version
"$PYTHON_EXE" -c 'import numpy, torch, matplotlib; print("numpy", numpy.__version__); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("matplotlib", matplotlib.__version__)'
nvidia-smi

srun -N1 \
  -n1 \
  --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
  --export=ALL \
  "$PYTHON_EXE" -u stl_component_separation.py \
    --data "$DATA" \
    --data-index 0 \
    --cib-bank "$CIB_BANK" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --dtype float32 \
    --seed 240830 \
    --n-iterations 150 \
    --batch-size 15 \
    --n-noise 0 \
    --optimizer-lr 1.0 \
    --history-size 100 \
    --wtype Bump-Steerable \
    --pbc \
    --no-compute-ps \
    --apply-nyquist-after \
    --print-every 10

echo "Results:"
ls -lh "$OUTPUT_DIR"
echo "ULA initialization: ${OUTPUT_DIR}/recovered_signal.npy"
echo "end=$(date)"
