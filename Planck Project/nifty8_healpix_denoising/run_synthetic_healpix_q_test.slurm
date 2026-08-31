#!/bin/bash
#SBATCH --job-name=nifty_q_synth
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

export SYNTH_NSIDE="128"
export SYNTH_SNR="1.0"
export SYNTH_SEED="11"
export BEAM_FWHM_ARCMIN="12.0"
export BEAM_MODE="both"
export SYNTH_OUTDIR="/lustre/fswork/projects/rech/pmr/uee28se/nifty8_healpix_denoising/synthetic_q_nside${SYNTH_NSIDE}_beam12arcmin_10samples"
export SYNTH_CL_AMPLITUDE="1e-5"
export SYNTH_CL_SLOPE="-2.4"
export SYNTH_ELL0="80.0"
export TRUTH_FITS=""
export TRUTH_COMPONENT="1"
export NOISE_SIGMA=""

export POSTERIOR_SAMPLES="10"
export SAVE_SAMPLES="10"
export VI_ITERATIONS="3"
export MAP_ITERATIONS="50"
export SAMPLE_CG_ITERATIONS="150"

TRUTH_ARGS=()
if [[ -n "${TRUTH_FITS}" ]]; then
  TRUTH_ARGS+=(--truth-fits "${TRUTH_FITS}" --truth-component "${TRUTH_COMPONENT}")
fi

NOISE_ARGS=()
if [[ -n "${NOISE_SIGMA}" ]]; then
  NOISE_ARGS+=(--noise-sigma "${NOISE_SIGMA}")
fi

srun python -u synthetic_healpix_q_test.py \
  "${TRUTH_ARGS[@]}" \
  --nside "${SYNTH_NSIDE}" \
  --snr "${SYNTH_SNR}" \
  "${NOISE_ARGS[@]}" \
  --beam-fwhm-arcmin "${BEAM_FWHM_ARCMIN}" \
  --beam-mode "${BEAM_MODE}" \
  --seed "${SYNTH_SEED}" \
  --outdir "${SYNTH_OUTDIR}" \
  --cl-amplitude "${SYNTH_CL_AMPLITUDE}" \
  --cl-slope "${SYNTH_CL_SLOPE}" \
  --ell0 "${SYNTH_ELL0}" \
  --posterior-samples "${POSTERIOR_SAMPLES}" \
  --save-samples "${SAVE_SAMPLES}" \
  --vi-iterations "${VI_ITERATIONS}" \
  --map-iterations "${MAP_ITERATIONS}" \
  --sample-cg-iterations "${SAMPLE_CG_ITERATIONS}"
