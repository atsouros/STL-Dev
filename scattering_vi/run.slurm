#!/bin/bash
#SBATCH --job-name=scat_run
#SBATCH --account=pmr@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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

srun python -u microsample.py \
    --device cuda:0 \
    --dtype float64 \
    --covariance-npz covariances/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.npz \
    --covariance-json covariances/100_Herschel_Lockman_250m_tiles_3400x256x256_covariance.json \
    --data-input test_data.npy \
    --data-index 0 \
    --n-samples 10 \
    --vi-iterations 4 \
    --map-iterations 10 \
    --kl-iterations 80 \
    --sampling-iterations 250 \
    --run-map \
    --run-vi \
    --use-correlated-field-prior \
    --prior-loglog-slope-mean -3.0 \
    --prior-loglog-slope-std 0.5 \
    --out test_data_cib_likelihood_posterior.npz \
    --plot test_data_cib_likelihood_posterior.png \
    --diagnostics-summary test_data_cib_likelihood_summary.json
