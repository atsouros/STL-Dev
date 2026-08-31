#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from healpix_q_denoise_nifty8 import main as run_denoising


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def positive_float(value: str) -> float:
    out = float(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def dust_like_cl(nside: int, amplitude: float, slope: float, ell0: float) -> np.ndarray:
    ell_max = 3 * nside - 1
    ell = np.arange(ell_max + 1, dtype=np.float64)
    cl = np.zeros_like(ell)
    valid = ell >= 2
    cl[valid] = amplitude * (ell[valid] / ell0) ** slope
    return cl


def read_truth_map(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    import healpy as hp

    if args.truth_fits is None:
        cl = dust_like_cl(
            args.nside,
            amplitude=args.cl_amplitude,
            slope=args.cl_slope,
            ell0=args.ell0,
        )
        # healpy.synfast uses NumPy's global RNG internally.
        np.random.seed(args.seed)
        signal = hp.synfast(cl, nside=args.nside, lmax=len(cl) - 1, new=True)
        source = {
            "truth_source": "generated_gaussian",
            "cl_amplitude": args.cl_amplitude,
            "cl_slope": args.cl_slope,
            "ell0": args.ell0,
        }
    else:
        signal = hp.read_map(args.truth_fits, field=args.truth_component, dtype=np.float64)
        expected = hp.nside2npix(args.nside)
        if signal.shape != (expected,):
            raise ValueError(
                f"{args.truth_fits} field {args.truth_component} has shape {signal.shape}; "
                f"expected ({expected},) for nside={args.nside}"
            )
        source = {
            "truth_source": "fits",
            "truth_fits": str(args.truth_fits),
            "truth_component": int(args.truth_component),
        }
    signal = np.asarray(signal, dtype=np.float64)
    signal -= np.mean(signal[np.isfinite(signal)])
    return signal, source


def make_synthetic_data(args: argparse.Namespace) -> tuple[Path, float, dict[str, object]]:
    import healpy as hp

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    signal, truth_source = read_truth_map(args)

    sigma_noise = float(args.noise_sigma) if args.noise_sigma is not None else float(np.std(signal) / args.snr)
    noise = rng.normal(0.0, sigma_noise, size=signal.shape)
    if args.beam_mode not in {"both", "noise-only", "none"}:
        raise ValueError("--beam-mode must be one of: both, noise-only, none")

    if args.beam_mode == "both":
        unobserved_data = signal + noise
    elif args.beam_mode == "noise-only":
        unobserved_data = signal
    else:
        unobserved_data = signal + noise

    if args.beam_fwhm_arcmin > 0 and args.beam_mode != "none":
        fwhm_rad = np.deg2rad(args.beam_fwhm_arcmin / 60.0)
        beamed_noise = hp.smoothing(noise, fwhm=fwhm_rad)
        if args.beam_mode == "both":
            data = hp.smoothing(signal + noise, fwhm=fwhm_rad)
        else:
            data = signal + beamed_noise
    else:
        data = signal + noise
        beamed_noise = noise

    # The inference model is intentionally identical across synthetic cases:
    # d_model = B s + B n. Only the generated data map differs.
    signal_beam_for_likelihood = args.beam_fwhm_arcmin
    noise_beam_for_likelihood = args.beam_fwhm_arcmin

    signal_path = args.outdir / "synthetic_signal_Q.fits"
    noise_path = args.outdir / "synthetic_noise_Q.fits"
    beamed_noise_path = args.outdir / "synthetic_beamed_noise_Q.fits"
    unbeamed_data_path = args.outdir / "synthetic_unbeamed_data_Q.fits"
    data_path = args.outdir / "synthetic_data_Q.fits"
    hp.write_map(str(signal_path), signal, overwrite=True)
    hp.write_map(str(noise_path), noise, overwrite=True)
    hp.write_map(str(beamed_noise_path), beamed_noise, overwrite=True)
    hp.write_map(str(unbeamed_data_path), unobserved_data, overwrite=True)
    hp.write_map(str(data_path), data, overwrite=True)

    metadata = {
        "nside": args.nside,
        "npix": int(data.size),
        "seed": args.seed,
        "snr": args.snr,
        "sigma_noise": sigma_noise,
        "noise_sigma_source": "cli" if args.noise_sigma is not None else "signal_std_over_snr",
        "beam_fwhm_arcmin": args.beam_fwhm_arcmin,
        "beam_mode": args.beam_mode,
        "signal_beam_for_likelihood_arcmin": signal_beam_for_likelihood,
        "noise_beam_for_likelihood_arcmin": noise_beam_for_likelihood,
        "signal_std": float(np.std(signal)),
        "noise_std": float(np.std(noise)),
        "beamed_data_std": float(np.std(data)),
        "signal_path": str(signal_path),
        "noise_path": str(noise_path),
        "beamed_noise_path": str(beamed_noise_path),
        "unbeamed_data_path": str(unbeamed_data_path),
        "data_path": str(data_path),
        **truth_source,
    }
    (args.outdir / "synthetic_truth_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(args.outdir / "synthetic_signal_Q.npy", signal, allow_pickle=False)
    np.save(args.outdir / "synthetic_noise_Q.npy", noise, allow_pickle=False)
    np.save(args.outdir / "synthetic_beamed_noise_Q.npy", beamed_noise, allow_pickle=False)
    np.save(args.outdir / "synthetic_data_Q.npy", data, allow_pickle=False)
    return data_path, sigma_noise, metadata


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Small-nside synthetic Healpix test for the NIFTy8 Q denoising script. "
            "This generates d=s+n and then calls healpix_q_denoise_nifty8.py."
        )
    )
    p.add_argument("--nside", type=positive_int, default=64)
    p.add_argument("--snr", type=positive_float, default=1.0)
    p.add_argument(
        "--noise-sigma",
        type=positive_float,
        default=None,
        help="Optional fixed white-noise standard deviation. If omitted, sigma=std(signal)/snr.",
    )
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--outdir", type=Path, default=Path("synthetic_nifty8_q_test"))
    p.add_argument(
        "--beam-fwhm-arcmin",
        type=float,
        default=10.0,
        help="Gaussian beam FWHM in arcmin.",
    )
    p.add_argument(
        "--beam-mode",
        choices=["both", "noise-only", "none"],
        default="both",
        help=(
            "'both': d=B(s+n), used for Gaussian generated truth. "
            "'noise-only': d=s+B n, used when the truth map is already beamed. "
            "'none': d=s+n."
        ),
    )
    p.add_argument(
        "--truth-fits",
        type=Path,
        default=None,
        help="Optional Healpix FITS map to use as the true signal s. If omitted, a Gaussian field is generated.",
    )
    p.add_argument(
        "--truth-component",
        type=int,
        default=0,
        help="FITS field index to read from --truth-fits.",
    )
    p.add_argument("--cl-amplitude", type=positive_float, default=1e-5)
    p.add_argument("--cl-slope", type=float, default=-2.4)
    p.add_argument("--ell0", type=positive_float, default=80.0)

    p.add_argument("--posterior-samples", type=positive_int, default=10)
    p.add_argument("--save-samples", type=positive_int, default=10)
    p.add_argument("--vi-iterations", type=positive_int, default=3)
    p.add_argument("--map-iterations", type=positive_int, default=50)
    p.add_argument("--sample-cg-iterations", type=positive_int, default=150)
    p.add_argument("--sample-cg-tol", type=positive_float, default=1e-3)
    p.add_argument("--map-tol", type=positive_float, default=1e-6)

    p.add_argument("--prior-loglog-slope-mean", type=float, default=-2.4)
    p.add_argument("--prior-loglog-slope-std", type=positive_float, default=0.5)
    p.add_argument("--prior-fluctuations-mean", type=positive_float, default=1.0)
    p.add_argument("--prior-fluctuations-std", type=positive_float, default=0.5)
    p.add_argument("--prior-flexibility-mean", type=float, default=0.5)
    p.add_argument("--prior-flexibility-std", type=positive_float, default=0.5)
    return p


def main() -> int:
    args = make_parser().parse_args()
    data_path, sigma_noise, metadata = make_synthetic_data(args)
    print(
        "synthetic_data_ready "
        f"nside={args.nside} sigma_noise={sigma_noise:.8e} "
        f"signal_std={metadata['signal_std']:.8e}",
        flush=True,
    )

    denoise_args = [
        "--data-fits", str(data_path),
        "--component", "I",
        "--nside", str(args.nside),
        "--noise-sigma", f"{sigma_noise:.17g}",
        "--beam-fwhm-arcmin", str(metadata["signal_beam_for_likelihood_arcmin"]),
        "--noise-beam-fwhm-arcmin", str(metadata["noise_beam_for_likelihood_arcmin"]),
        "--outdir", str(args.outdir / "posterior"),
        "--seed", str(args.seed),
        "--posterior-samples", str(args.posterior_samples),
        "--save-samples", str(args.save_samples),
        "--vi-iterations", str(args.vi_iterations),
        "--map-iterations", str(args.map_iterations),
        "--map-tol", str(args.map_tol),
        "--sample-cg-iterations", str(args.sample_cg_iterations),
        "--sample-cg-tol", str(args.sample_cg_tol),
        "--no-geovi",
        "--prior-loglog-slope-mean", str(args.prior_loglog_slope_mean),
        "--prior-loglog-slope-std", str(args.prior_loglog_slope_std),
        "--prior-fluctuations-mean", str(args.prior_fluctuations_mean),
        "--prior-fluctuations-std", str(args.prior_fluctuations_std),
        "--prior-flexibility-mean", str(args.prior_flexibility_mean),
        "--prior-flexibility-std", str(args.prior_flexibility_std),
    ]
    return run_denoising(denoise_args)


if __name__ == "__main__":
    raise SystemExit(main())
