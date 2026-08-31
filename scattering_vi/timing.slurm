#!/bin/bash
#SBATCH --job-name=scat_timing
#SBATCH --account=pmr@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=timing.out
#SBATCH --error=timing.err

set -euo pipefail

source ~/.bashrc
conda activate science

cd /lustre/fswork/projects/rech/pmr/uee28se/scattering_vi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

N_SAMPLES=2
VI_ITERATIONS=1
MAP_ITERATIONS=0
KL_ITERATIONS=10
SAMPLING_ITERATIONS=25

echo "started: $(date --iso-8601=seconds)"
echo "host: $(hostname)"
echo "job id: ${SLURM_JOB_ID}"
echo "timing parameters:"
echo "  n_samples=${N_SAMPLES}"
echo "  vi_iterations=${VI_ITERATIONS}"
echo "  map_iterations=${MAP_ITERATIONS}"
echo "  kl_iterations=${KL_ITERATIONS}"
echo "  sampling_iterations=${SAMPLING_ITERATIONS}"
echo "full run currently uses: n_samples=8 vi_iterations=4 kl_iterations=80 sampling_iterations=250"

nvidia-smi

/usr/bin/time -v srun python -u run.py \
    --device cuda:0 \
    --dtype float64 \
    --covariance-npz lss_cov/lss_covariance.npz \
    --covariance-json lss_cov/lss_covariance.json \
    --target-input lss_cov/lss.npy \
    --n-samples "${N_SAMPLES}" \
    --vi-iterations "${VI_ITERATIONS}" \
    --map-iterations "${MAP_ITERATIONS}" \
    --kl-iterations "${KL_ITERATIONS}" \
    --sampling-iterations "${SAMPLING_ITERATIONS}" \
    --run-vi \
    --no-run-mcmc \
    --use-correlated-field-prior \
    --prior-loglog-slope-mean -3.0 \
    --prior-loglog-slope-std 0.5 \
    --quiet \
    --out timing_samples.npz \
    --plot timing_samples.png \
    --diagnostics-csv timing_diagnostics.csv \
    --diagnostics-plot timing_diagnostics.png \
    --diagnostics-summary timing_diagnostics_summary.json

echo "finished: $(date --iso-8601=seconds)"
