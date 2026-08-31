#!/bin/bash
#SBATCH --job-name=refine_compsep
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --qos=gpu_ens_normal
#SBATCH --output=refine_component_separation-%j.log
#SBATCH --error=refine_component_separation-%j.err

set -euo pipefail

export SLURM_CPU_BIND=cores
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
LANGEVIN_DIR="${STL_DEV_ROOT}/langevin"
COMPSEP_DIR="${LANGEVIN_DIR}/results/component_separation"
SCATTERING_RESULTS="${STL_DEV_ROOT}/scattering_vi/results"
OUTPUT_DIR="${LANGEVIN_DIR}/results/refinement"
PYTHON_EXE="${PYTHON_EXE:-python3}"

DATA="${COMPSEP_DIR}/observed_data.npy"
INITIAL_SIGNAL="${COMPSEP_DIR}/recovered_signal.npy"
COVARIANCE_NPZ="${SCATTERING_RESULTS}/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.npz"
COVARIANCE_JSON="${SCATTERING_RESULTS}/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.json"

for path in "$DATA" "$INITIAL_SIGNAL" "$COVARIANCE_NPZ" "$COVARIANCE_JSON" "${LANGEVIN_DIR}/refine_component_separation.py"; do
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
echo "Exact-likelihood refinement; original component map is read-only"
"$PYTHON_EXE" --version
"$PYTHON_EXE" -c 'import numpy, torch, matplotlib; print("numpy", numpy.__version__); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("matplotlib", matplotlib.__version__)'
nvidia-smi

"$PYTHON_EXE" -u refine_component_separation.py \
    --data "$DATA" \
    --initial-signal "$INITIAL_SIGNAL" \
    --covariance-npz "$COVARIANCE_NPZ" \
    --covariance-json "$COVARIANCE_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --dtype float64 \
    --max-iter 300 \
    --lr 1.0 \
    --history-size 50 \
    --anchor-precision 0.1 \
    --zero-mean-signal \
    --covariance-jitter-rel 1e-6 \
    --print-every 10 \
    --top-statistics 15

echo "Results:"
ls -lh "$OUTPUT_DIR"
echo "Original preserved: $INITIAL_SIGNAL"
echo "Refined signal: ${OUTPUT_DIR}/signal_refined.npy"
echo "end=$(date)"
