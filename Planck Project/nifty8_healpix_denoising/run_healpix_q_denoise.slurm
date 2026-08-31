#!/bin/bash
#SBATCH --job-name=nifty_q_denoise
#SBATCH --account=pmr@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
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
        + "\nInstall them in the activated science environment before submitting."
    )
PY

# Data. This recovered map is the datum d in d = s + n.
export DATA_FITS="/lustre/fswork/projects/rech/pmr/uee28se/fits/recovered_v2_white_noise_beam_matched_IQU_nside1024.fits"
export NOISE_SIGMA="8.630920e-02"
export COMPONENT="Q"
export NSIDE="1024"
# Full-resolution run. For debugging only, set this to 256 or 512.
export ANALYSIS_NSIDE="1024"
export NEST="False"

# White-noise likelihood.
# NOISE_SIGMA is the fitted scalar white-noise standard deviation for COMPONENT.

# Output.
export OUTDIR="/lustre/fswork/projects/rech/pmr/uee28se/nifty8_healpix_denoising/results_q_full_nside1024_10samples"
export SEED="2"

# Posterior approximation.
export POSTERIOR_SAMPLES="10"
export SAVE_SAMPLES="10"
export VI_ITERATIONS="3"
export MAP_ITERATIONS="50"
export MAP_TOL="1e-6"
export SAMPLE_CG_ITERATIONS="150"
export SAMPLE_CG_TOL="1e-3"
export GEOVI="False"
export GEOVI_ITERATIONS="12"
export GEOVI_TOL="1e-3"
export DRY_RUN="False"

# Correlated-field prior for polarized dust Q.
# The default slope is for C_ell, not D_ell: D_ell ~ ell^-0.4 implies roughly C_ell ~ ell^-2.4.
export PRIOR_LOGLOG_SLOPE_MEAN="-2.4"
export PRIOR_LOGLOG_SLOPE_STD="0.5"
export PRIOR_FLUCTUATIONS_MEAN="1.0"
export PRIOR_FLUCTUATIONS_STD="0.5"
export PRIOR_FLEXIBILITY_MEAN="0.5"
export PRIOR_FLEXIBILITY_STD="0.5"
export PRIOR_ASPERITY_MEAN="0.0"
export PRIOR_ASPERITY_STD="0.5"
export PRIOR_OFFSET_MEAN="0.0"
export PRIOR_OFFSET_STD_MEAN="1.0"
export PRIOR_OFFSET_STD_STD="0.5"

if [[ -z "${NOISE_SIGMA}" ]]; then
  echo "ERROR: set NOISE_SIGMA to the fitted white-noise standard deviation before submitting." >&2
  exit 1
fi

ANALYSIS_NSIDE_ARGS=()
if [[ -n "${ANALYSIS_NSIDE}" ]]; then
  ANALYSIS_NSIDE_ARGS+=(--analysis-nside "${ANALYSIS_NSIDE}")
fi

srun python -u healpix_q_denoise_nifty8.py \
  --data-fits "${DATA_FITS}" \
  --component "${COMPONENT}" \
  --nside "${NSIDE}" \
  "${ANALYSIS_NSIDE_ARGS[@]}" \
  $( [[ "${NEST}" == "True" ]] && echo --nest || echo --no-nest ) \
  --noise-sigma "${NOISE_SIGMA}" \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --posterior-samples "${POSTERIOR_SAMPLES}" \
  --save-samples "${SAVE_SAMPLES}" \
  --vi-iterations "${VI_ITERATIONS}" \
  --map-iterations "${MAP_ITERATIONS}" \
  --map-tol "${MAP_TOL}" \
  --sample-cg-iterations "${SAMPLE_CG_ITERATIONS}" \
  --sample-cg-tol "${SAMPLE_CG_TOL}" \
  $( [[ "${GEOVI}" == "True" ]] && echo --geovi || echo --no-geovi ) \
  --geovi-iterations "${GEOVI_ITERATIONS}" \
  --geovi-tol "${GEOVI_TOL}" \
  $( [[ "${DRY_RUN}" == "True" ]] && echo --dry-run || echo --no-dry-run ) \
  --prior-loglog-slope-mean "${PRIOR_LOGLOG_SLOPE_MEAN}" \
  --prior-loglog-slope-std "${PRIOR_LOGLOG_SLOPE_STD}" \
  --prior-fluctuations-mean "${PRIOR_FLUCTUATIONS_MEAN}" \
  --prior-fluctuations-std "${PRIOR_FLUCTUATIONS_STD}" \
  --prior-flexibility-mean "${PRIOR_FLEXIBILITY_MEAN}" \
  --prior-flexibility-std "${PRIOR_FLEXIBILITY_STD}" \
  --prior-asperity-mean "${PRIOR_ASPERITY_MEAN}" \
  --prior-asperity-std "${PRIOR_ASPERITY_STD}" \
  --prior-offset-mean "${PRIOR_OFFSET_MEAN}" \
  --prior-offset-std-mean "${PRIOR_OFFSET_STD_MEAN}" \
  --prior-offset-std-std "${PRIOR_OFFSET_STD_STD}"
