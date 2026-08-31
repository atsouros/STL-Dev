#!/bin/bash
#SBATCH --job-name=nifty_q_truth
#SBATCH --account=pmr@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:59:59
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

source ~/.bashrc
conda activate science

cd /lustre/fswork/projects/rech/pmr/uee28se/planck

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

nvidia-smi

python - <<'PY'
import importlib.util
missing = [name for name in ("nifty8", "healpy", "ducc0") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "ERROR: missing required Python packages in this environment: "
        + ", ".join(missing)
    )
PY

# True field s. Set this to the full-sky truth map you want to use.
export TRUTH_FITS="/lustre/fswork/projects/rech/pmr/uee28se/fits/recovered_v2_white_noise_beam_matched_IQU_nside1024.fits"
export TRUTH_COMPONENT="1"
export SYNTH_NSIDE="1024"

# Synthetic white noise added to the true field.
export NOISE_SIGMA="8.630920e-02"
export SYNTH_SNR="1.0"
export SYNTH_SEED="11"
export BEAM_FWHM_ARCMIN="12.0"
export BEAM_MODE="noise-only"

export SYNTH_OUTDIR="/lustre/fswork/projects/rech/pmr/uee28se/nifty8_healpix_denoising/truth_q_nside${SYNTH_NSIDE}_beam12arcmin_noise8p630920em02_10samples"

export POSTERIOR_SAMPLES="10"
export SAVE_SAMPLES="10"
export VI_ITERATIONS="3"
export MAP_ITERATIONS="50"
export SAMPLE_CG_ITERATIONS="150"

srun python -u healpix_q.py \
  --truth-fits "${TRUTH_FITS}" \
  --truth-component "${TRUTH_COMPONENT}" \
  --nside "${SYNTH_NSIDE}" \
  --snr "${SYNTH_SNR}" \
  --noise-sigma "${NOISE_SIGMA}" \
  --beam-fwhm-arcmin "${BEAM_FWHM_ARCMIN}" \
  --beam-mode "${BEAM_MODE}" \
  --seed "${SYNTH_SEED}" \
  --outdir "${SYNTH_OUTDIR}" \
  --posterior-samples "${POSTERIOR_SAMPLES}" \
  --save-samples "${SAVE_SAMPLES}" \
  --vi-iterations "${VI_ITERATIONS}" \
  --map-iterations "${MAP_ITERATIONS}" \
  --sample-cg-iterations "${SAMPLE_CG_ITERATIONS}"
