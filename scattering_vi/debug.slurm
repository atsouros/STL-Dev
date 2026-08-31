#!/bin/bash
#SBATCH --job-name=scat_debug
#SBATCH --account=pmr@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

source ~/.bashrc
conda activate science

cd /lustre/fswork/projects/rech/pmr/uee28se/scattering_vi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

nvidia-smi

python - <<'PY'
import nifty8
import torch
print("torch cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("nifty8:", getattr(nifty8, "__version__", "unknown"))
PY

srun python -u run.py \
    --device cuda:0 \
    --dtype float64 \
    --covariance-npz lss_cov/lss_covariance.npz \
    --covariance-json lss_cov/lss_covariance.json \
    --target-input lss_cov/lss.npy \
    --n-samples 1 \
    --vi-iterations 1 \
    --map-iterations 0 \
    --kl-iterations 1 \
    --sampling-iterations 5 \
    --run-vi \
    --no-run-mcmc \
    --use-correlated-field-prior \
    --prior-loglog-slope-mean -3.0 \
    --prior-loglog-slope-std 0.5 \
    --quiet \
    --out debug_samples.npz \
    --plot debug_samples.png
