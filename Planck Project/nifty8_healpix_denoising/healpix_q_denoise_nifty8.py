#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


IQU_FIELDS = {"I": 0, "Q": 1, "U": 2}


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


def parse_component(value: str) -> int:
    key = value.upper()
    if key in IQU_FIELDS:
        return IQU_FIELDS[key]
    idx = int(value)
    if idx not in (0, 1, 2):
        raise argparse.ArgumentTypeError("component must be I, Q, U, 0, 1, or 2")
    return idx


def read_healpix_component(path: Path, component: int, nside: int, nest: bool) -> np.ndarray:
    import healpy as hp

    if not path.exists():
        raise FileNotFoundError(path)
    try:
        values = hp.read_map(path, field=component, dtype=np.float64, nest=nest)
    except TypeError:
        values = hp.read_map(path, field=component, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    expected = hp.nside2npix(nside)
    if values.shape != (expected,):
        raise ValueError(
            f"{path} component {component} has shape {values.shape}; "
            f"expected ({expected},) for nside={nside}"
        )
    return values


def maybe_degrade_map(values: np.ndarray, nside_in: int, nside_out: int, nest: bool) -> np.ndarray:
    import healpy as hp

    if nside_out == nside_in:
        return values
    if nside_out > nside_in:
        raise ValueError(f"analysis nside {nside_out} cannot exceed input nside {nside_in}")
    print(f"degrading map from nside={nside_in} to analysis_nside={nside_out}", flush=True)
    return np.asarray(hp.ud_grade(values, nside_out=nside_out, order_in="NESTED" if nest else "RING", order_out="NESTED" if nest else "RING"), dtype=np.float64)


def finite_sky_mask(*maps: np.ndarray) -> np.ndarray:
    import healpy as hp

    mask = np.ones_like(maps[0], dtype=bool)
    for m in maps:
        mask &= np.isfinite(m)
        mask &= m != hp.UNSEEN
    return mask


def build_prior(position_space, args):
    import nifty8 as ift

    cfm = ift.CorrelatedFieldMaker(args.prior_prefix)
    cfm.set_amplitude_total_offset(
        args.prior_offset_mean,
        (args.prior_offset_std_mean, args.prior_offset_std_std),
    )
    cfm.add_fluctuations(
        position_space,
        fluctuations=(args.prior_fluctuations_mean, args.prior_fluctuations_std),
        flexibility=(
            None
            if args.prior_flexibility_mean <= 0
            else (args.prior_flexibility_mean, args.prior_flexibility_std)
        ),
        asperity=(
            None
            if args.prior_asperity_mean <= 0
            else (args.prior_asperity_mean, args.prior_asperity_std)
        ),
        loglogavgslope=(args.prior_loglog_slope_mean, args.prior_loglog_slope_std),
        harmonic_partner=position_space.get_default_codomain(),
    )
    return cfm.finalize(prior_info=args.prior_info)


def build_beam_response(position_space, fwhm_arcmin: float):
    import nifty8 as ift

    if fwhm_arcmin <= 0:
        return ift.Operator.identity_operator(position_space)
    harmonic_space = position_space.get_default_codomain()
    harmonic_transform = ift.HarmonicTransformOperator(harmonic_space, position_space)
    sigma_rad = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    ell = harmonic_space.get_k_length_array().val
    beam = ift.makeField(harmonic_space, np.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2))
    return harmonic_transform @ ift.makeOp(beam) @ harmonic_transform.adjoint


def build_noise_inverse_covariance(position_space, sigma_n: float, valid: np.ndarray, noise_beam_fwhm_arcmin: float):
    import nifty8 as ift

    if noise_beam_fwhm_arcmin <= 0:
        if np.all(valid):
            return ift.ScalingOperator(position_space, 1.0 / sigma_n**2, sampling_dtype=np.float64)
        precision = np.zeros(position_space.shape, dtype=np.float64)
        precision[valid] = 1.0 / sigma_n**2
        return ift.makeOp(ift.makeField(position_space, precision))

    if not np.all(valid):
        raise NotImplementedError(
            "Masked pixels with beamed-noise covariance are not implemented. "
            "Use full finite-sky maps or disable --noise-beam-fwhm-arcmin."
        )

    harmonic_space = position_space.get_default_codomain()
    harmonic_transform = ift.HarmonicTransformOperator(harmonic_space, position_space)
    sigma_rad = np.deg2rad(noise_beam_fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    ell = harmonic_space.get_k_length_array().val
    beam = np.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2)
    inv_noise_power = ift.makeField(harmonic_space, 1.0 / (sigma_n**2 * beam**2))
    return ift.SandwichOperator.make(
        harmonic_transform.adjoint,
        ift.DiagonalOperator(inv_noise_power, sampling_dtype=np.float64),
    )


def posterior_sample_array(sample_list, signal_op, n_save: int, shape: tuple[int, ...]) -> np.ndarray:
    samples = []
    for ii, latent in enumerate(sample_list.iterator()):
        if ii >= n_save:
            break
        samples.append(signal_op(latent).val.astype(np.float64, copy=False))
    if not samples:
        return np.empty((0,) + shape, dtype=np.float64)
    return np.stack(samples, axis=0)


def write_healpix_outputs(args, posterior_mean, posterior_std, samples: np.ndarray) -> None:
    import healpy as hp

    args.outdir.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(args.outdir / f"posterior_mean_{args.component_name}.fits"), posterior_mean, overwrite=True)
    hp.write_map(str(args.outdir / f"posterior_std_{args.component_name}.fits"), posterior_std, overwrite=True)
    if samples.size:
        hp.write_map(
            str(args.outdir / f"posterior_samples_{args.component_name}.fits"),
            samples,
            overwrite=True,
        )
    np.save(args.outdir / f"posterior_samples_{args.component_name}.npy", samples, allow_pickle=False)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "NIFTy8 Healpix denoising for one I/Q/U component with an analytic "
            "white-noise Gaussian likelihood and a hierarchical correlated-field prior."
        )
    )
    p.add_argument(
        "--data-fits",
        type=Path,
        default=Path(
            "/pscratch/sd/a/atsouros/STL/patches2map/output/minimal_noise_beam_match/"
            "recovered_v2_white_noise_beam_matched_IQU_nside1024.fits"
        ),
        help="IQU FITS map used as the data map d in d = s + n.",
    )
    p.add_argument("--component", default="Q", type=parse_component)
    p.add_argument("--nside", default=1024, type=positive_int)
    p.add_argument(
        "--analysis-nside",
        type=positive_int,
        default=None,
        help="Optionally downsample the input map before inference. Use 128/256 for smoke tests.",
    )
    p.add_argument("--nest", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--noise-sigma",
        type=positive_float,
        required=True,
        help="Fitted scalar white-noise standard deviation for the selected component.",
    )
    p.add_argument(
        "--beam-fwhm-arcmin",
        type=float,
        default=0.0,
        help="Gaussian beam FWHM in arcmin applied to the signal response B s. 0 disables signal beam.",
    )
    p.add_argument(
        "--noise-beam-fwhm-arcmin",
        type=float,
        default=0.0,
        help="Gaussian beam FWHM in arcmin applied to the white-noise covariance. 0 means pixel-white noise.",
    )
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--outdir", type=Path, default=Path("nifty8_healpix_denoising/results_q"))

    p.add_argument("--posterior-samples", type=int, default=8)
    p.add_argument("--save-samples", type=int, default=8)
    p.add_argument("--vi-iterations", type=positive_int, default=6)
    p.add_argument("--map-iterations", type=positive_int, default=50)
    p.add_argument("--map-tol", type=positive_float, default=1e-6)
    p.add_argument("--sample-cg-iterations", type=positive_int, default=250)
    p.add_argument("--sample-cg-tol", type=positive_float, default=1e-3)
    p.add_argument("--geovi", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--geovi-iterations", type=positive_int, default=12)
    p.add_argument("--geovi-tol", type=positive_float, default=1e-3)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--prior-prefix", default="dust_q")
    p.add_argument("--prior-info", type=int, default=0)
    p.add_argument("--prior-offset-mean", type=float, default=0.0)
    p.add_argument("--prior-offset-std-mean", type=positive_float, default=1.0)
    p.add_argument("--prior-offset-std-std", type=positive_float, default=0.5)
    p.add_argument("--prior-fluctuations-mean", type=positive_float, default=1.0)
    p.add_argument("--prior-fluctuations-std", type=positive_float, default=0.5)
    p.add_argument(
        "--prior-loglog-slope-mean",
        type=float,
        default=-2.4,
        help="Default is a broad polarized-dust-like C_ell slope.",
    )
    p.add_argument("--prior-loglog-slope-std", type=positive_float, default=0.5)
    p.add_argument("--prior-flexibility-mean", type=float, default=0.5)
    p.add_argument("--prior-flexibility-std", type=positive_float, default=0.5)
    p.add_argument("--prior-asperity-mean", type=float, default=0.0)
    p.add_argument("--prior-asperity-std", type=positive_float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.component_name = {0: "I", 1: "Q", 2: "U"}[args.component]

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import nifty8 as ift

    try:
        import cupy  # noqa: F401

        gpu_note = "cupy import succeeded"
    except Exception:
        gpu_note = "classic NIFTy8 path is NumPy-backed here"

    print(f"python={sys.executable}", flush=True)
    print(f"nifty8={ift.__file__}", flush=True)
    print(f"backend_note={gpu_note}", flush=True)
    print(f"component={args.component_name} nside={args.nside}", flush=True)

    ift.random.push_sseq_from_seed(args.seed)
    data_map = read_healpix_component(args.data_fits, args.component, args.nside, args.nest)
    analysis_nside = args.nside if args.analysis_nside is None else args.analysis_nside
    data_map = maybe_degrade_map(data_map, args.nside, analysis_nside, args.nest)
    sigma_n = float(args.noise_sigma)
    print(f"data_fits={args.data_fits}", flush=True)
    print(f"white_noise_sigma={sigma_n:.8e} source=cli", flush=True)
    print(f"signal_beam_fwhm_arcmin={args.beam_fwhm_arcmin:.6g}", flush=True)
    print(f"noise_beam_fwhm_arcmin={args.noise_beam_fwhm_arcmin:.6g}", flush=True)
    print(
        "run_config "
        f"analysis_nside={analysis_nside} npix={data_map.size} "
        f"posterior_samples={args.posterior_samples} vi_iterations={args.vi_iterations} "
        f"map_iterations={args.map_iterations} sample_cg_iterations={args.sample_cg_iterations}",
        flush=True,
    )

    valid = finite_sky_mask(data_map)
    if not np.all(valid):
        print(f"masking {np.size(valid) - int(valid.sum())} invalid pixels by setting them to zero", flush=True)
        data_map = data_map.copy()
        data_map[~valid] = 0.0

    position_space = ift.HPSpace(analysis_nside)
    data = ift.makeField(position_space, data_map.astype(np.float64, copy=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    status_path = args.outdir / "progress.jsonl"
    with status_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "start",
            "data_fits": str(args.data_fits),
            "component": args.component_name,
            "input_nside": args.nside,
            "analysis_nside": analysis_nside,
            "npix": int(data_map.size),
            "noise_sigma": sigma_n,
            "posterior_samples": args.posterior_samples,
            "vi_iterations": args.vi_iterations,
            "map_iterations": args.map_iterations,
            "sample_cg_iterations": args.sample_cg_iterations,
            "signal_beam_fwhm_arcmin": args.beam_fwhm_arcmin,
            "noise_beam_fwhm_arcmin": args.noise_beam_fwhm_arcmin,
        }) + "\n")

    signal_op = build_prior(position_space, args)
    response = build_beam_response(position_space, args.beam_fwhm_arcmin)
    observed_signal_op = response @ signal_op
    n_inv = build_noise_inverse_covariance(
        position_space,
        sigma_n,
        valid,
        args.noise_beam_fwhm_arcmin,
    )
    likelihood_energy = ift.GaussianEnergy(data=data, inverse_covariance=n_inv) @ observed_signal_op

    if args.dry_run:
        args.outdir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "data_fits": str(args.data_fits),
            "component": args.component_name,
            "input_nside": args.nside,
            "analysis_nside": analysis_nside,
            "noise_sigma": sigma_n,
            "noise_sigma_source": "cli",
            "signal_beam_fwhm_arcmin": args.beam_fwhm_arcmin,
            "noise_beam_fwhm_arcmin": args.noise_beam_fwhm_arcmin,
            "prior_loglog_slope_mean": args.prior_loglog_slope_mean,
            "dry_run": True,
        }
        (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("dry_run=done", flush=True)
        return 0

    kl_controller = ift.GradientNormController(
        tol_abs_gradnorm=args.map_tol,
        iteration_limit=args.map_iterations,
    )
    kl_minimizer = ift.NewtonCG(kl_controller, enable_logging=False)
    sampling_ic = ift.GradInfNormController(
        args.sample_cg_tol,
        iteration_limit=args.sample_cg_iterations,
    )

    if args.geovi:
        nl_controller = ift.GradientNormController(
            tol_abs_gradnorm=args.geovi_tol,
            iteration_limit=args.geovi_iterations,
        )
        nonlinear_sampling_minimizer = ift.NewtonCG(nl_controller, enable_logging=False)
    else:
        nonlinear_sampling_minimizer = None

    def inspect_callback(sample_list, iteration):
        with status_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "vi_iteration_done", "iteration": int(iteration)}) + "\n")
        print(f"vi_iteration_done={iteration}", flush=True)

    print("starting optimize_kl", flush=True)
    sample_list, mean_latent = ift.optimize_kl(
        likelihood_energy=likelihood_energy,
        total_iterations=args.vi_iterations,
        n_samples=args.posterior_samples,
        kl_minimizer=kl_minimizer,
        sampling_iteration_controller=sampling_ic,
        nonlinear_sampling_minimizer=nonlinear_sampling_minimizer,
        output_directory=str(args.outdir / "nifty_state"),
        inspect_callback=inspect_callback,
        plot_energy_history=False,
        plot_minisanity_history=False,
        save_strategy="all",
        return_final_position=True,
        sanity_checks=True,
    )
    print("optimize_kl_done", flush=True)

    posterior_mean, posterior_var = sample_list.sample_stat(op=signal_op)
    posterior_std = np.sqrt(np.maximum(posterior_var.val, 0.0))
    samples = posterior_sample_array(
        sample_list,
        signal_op,
        min(args.save_samples, args.posterior_samples),
        position_space.shape,
    )
    write_healpix_outputs(args, posterior_mean.val, posterior_std, samples)

    metadata = {
        "data_fits": str(args.data_fits),
        "component": args.component_name,
        "input_nside": args.nside,
        "analysis_nside": analysis_nside,
        "nest": args.nest,
        "noise_model": (
            "pixel-white Gaussian noise"
            if args.noise_beam_fwhm_arcmin <= 0
            else "Gaussian noise with harmonic covariance sigma_n^2 B_l^2"
        ),
        "noise_sigma": sigma_n,
        "noise_sigma_source": "cli",
        "signal_beam_fwhm_arcmin": args.beam_fwhm_arcmin,
        "noise_beam_fwhm_arcmin": args.noise_beam_fwhm_arcmin,
        "posterior_samples": args.posterior_samples,
        "vi_iterations": args.vi_iterations,
        "geovi": args.geovi,
        "prior": {
            "type": "NIFTy8 CorrelatedFieldMaker on HPSpace",
            "loglog_slope_mean": args.prior_loglog_slope_mean,
            "loglog_slope_std": args.prior_loglog_slope_std,
            "fluctuations": [args.prior_fluctuations_mean, args.prior_fluctuations_std],
            "flexibility": [args.prior_flexibility_mean, args.prior_flexibility_std],
            "asperity": [args.prior_asperity_mean, args.prior_asperity_std],
        },
        "latent_keys": list(mean_latent.keys()),
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved_outputs={args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())