#!/bin/bash
#SBATCH --job-name=ula_anchor
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=node0256
#SBATCH --time=06:00:00
#SBATCH --qos=gpu_ens_normal
#SBATCH --output=langevin_anchor-%j.log
#SBATCH --error=langevin_anchor-%j.err


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

# MesoPSL project layout. Override PYTHON_EXE when submitting if needed:
#   sbatch --export=ALL,PYTHON_EXE=/path/to/environment/bin/python langevin_sample.slurm
STL_DEV_ROOT="/obs/atsouros/projects/STL-Dev"
LANGEVIN_DIR="${STL_DEV_ROOT}/langevin"
SCATTERING_RESULTS="${STL_DEV_ROOT}/scattering_vi/results"
PYTHON_EXE="${PYTHON_EXE:-python3}"

COMPSEP_OUTPUT="${LANGEVIN_DIR}/results/component_separation"
REFINEMENT_OUTPUT="${LANGEVIN_DIR}/results/refinement"
DATA="${COMPSEP_OUTPUT}/observed_data.npy"
ANCHOR_SIGNAL="${REFINEMENT_OUTPUT}/signal_refined.npy"
INITIAL_SIGNAL="${REFINEMENT_OUTPUT}/signal_refined.npy"
COVARIANCE_NPZ="${SCATTERING_RESULTS}/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.npz"
COVARIANCE_JSON="${SCATTERING_RESULTS}/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.json"
OUTPUT_DIR="${LANGEVIN_DIR}/results"

for path in "$DATA" "$ANCHOR_SIGNAL" "$INITIAL_SIGNAL" "$COVARIANCE_NPZ" "$COVARIANCE_JSON" "${LANGEVIN_DIR}/langevin_sample.py"; do
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
echo "Anchored ULA: component-separation anchor, refined initialization."
echo "Full run: 20000 steps, 10000 burn-in, 10 saved samples."
"$PYTHON_EXE" --version
"$PYTHON_EXE" -c 'import numpy, torch, matplotlib; print("numpy", numpy.__version__); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("matplotlib", matplotlib.__version__)'
nvidia-smi

"$PYTHON_EXE" -u langevin_sample.py \
    --data "$DATA" \
    --covariance-npz "$COVARIANCE_NPZ" \
    --covariance-json "$COVARIANCE_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --run-name mock_data_ula_anchor \
    --device cuda:0 \
    --dtype float64 \
    --seed 240829 \
    --prior-type anchor \
    --anchor-file "$ANCHOR_SIGNAL" \
    --anchor-precision 50 \
    --zero-mean-signal \
    --init file \
    --init-file "$INITIAL_SIGNAL" \
    --n-steps 20000 \
    --burn-in 10000 \
    --thin 1000 \
    --step-size 1e-4 \
    --covariance-jitter-rel 1e-6 \
    --print-every 100

echo "Results:"
ls -lh "$OUTPUT_DIR"/mock_data_ula_anchor_*
echo "end=$(date)"
