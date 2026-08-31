#!/usr/bin/env python3
"""Direct image-space ULA sampling with a scattering-Gaussian likelihood.

The sampled model is d = s + n with a scattering-Gaussian likelihood. The
recommended proof-of-concept prior is an anchor around an existing component
separation map,

    U(s) = 0.5 (phi(d-s)-mu)^T C^{-1} (phi(d-s)-mu)
           + 0.5 * anchor_precision * ||s-s_compsep||^2.

This implementation uses unadjusted Langevin dynamics (ULA), not MALA/HMC,
so its step size must be checked empirically.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import STL_main.torch_backend as bk  # noqa: E402
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass  # noqa: E402


COVARIANCE_STEM = "100_Herschel_Lockman_250m_tiles_3400x256x256_covariance"
DEFAULT_COVARIANCE_NPZ = REPO_ROOT / "scattering_vi" / "results" / f"{COVARIANCE_STEM}.npz"
DEFAULT_COVARIANCE_JSON = REPO_ROOT / "scattering_vi" / "results" / f"{COVARIANCE_STEM}.json"
DEFAULT_DATA = REPO_ROOT / "data" / "test" / "test_data_2.npy"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--covariance-npz", type=Path, default=DEFAULT_COVARIANCE_NPZ)
    parser.add_argument("--covariance-json", type=Path, default=DEFAULT_COVARIANCE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="test_ula")
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--seed", type=int, default=240827)

    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--burn-in", type=int, default=10)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=1e-6)
    parser.add_argument(
        "--init", choices=("zeros", "data", "random", "prior", "file"), default="zeros"
    )
    parser.add_argument(
        "--init-file",
        type=Path,
        default=None,
        help="Two-dimensional .npy signal map used when --init file.",
    )
    parser.add_argument("--init-std", type=float, default=0.1)
    parser.add_argument(
        "--prior-type", choices=("anchor", "quadratic", "powerlaw"), default="anchor"
    )
    parser.add_argument(
        "--anchor-file",
        type=Path,
        default=None,
        help="Component-separation .npy map used as the centre of the anchor prior.",
    )
    parser.add_argument("--anchor-precision", type=float, default=0.1)
    parser.add_argument("--lambda-grad", type=float, default=1.0)
    parser.add_argument("--lambda-l2", type=float, default=1e-2)
    parser.add_argument(
        "--prior-spectral-index",
        type=float,
        default=-3.07751,
        help=(
            "Exponent alpha of the POWER spectrum P(k) proportional to k**alpha. "
            "This is twice the corresponding Fourier-amplitude slope."
        ),
    )
    parser.add_argument("--prior-rms", type=float, default=2.53272)
    parser.add_argument(
        "--zero-mean-signal", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--covariance-jitter-rel", type=float, default=1e-6)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--max-plot-samples", type=int, default=3)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def configure_backend(device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64 if dtype == torch.float32 else torch.complex128


def load_metadata(path: Path) -> tuple[dict[str, Any], Namespace]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"{path} does not contain a JSON object named 'config'")
    required = (
        "target_size", "J", "L", "wtype", "iso", "angular_ft",
        "harmonics_angle", "scale_ft", "harmonics_scale", "dj",
        "compute_ps", "pbc", "take_log", "whiten", "dtype",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Covariance metadata is missing configuration keys: {missing}")
    config.setdefault("subtract_mean", False)
    config.setdefault("fewer_convolutions", False)
    return metadata, Namespace(**config)


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.data, "observed data"),
        (args.covariance_npz, "covariance NPZ"),
        (args.covariance_json, "covariance JSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.n_steps <= 0 or args.burn_in < 0 or args.thin <= 0:
        raise ValueError("Require n_steps > 0, burn_in >= 0, and thin > 0")
    if args.burn_in >= args.n_steps:
        raise ValueError("--burn-in must be smaller than --n-steps")
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")
    if args.prior_type == "anchor":
        if args.anchor_file is None or not args.anchor_file.is_file():
            raise FileNotFoundError("--prior-type anchor requires an existing --anchor-file")
        if args.anchor_precision <= 0:
            raise ValueError("--anchor-precision must be positive")
    if args.prior_type == "quadratic" and (args.lambda_grad < 0 or args.lambda_l2 <= 0):
        raise ValueError("Require lambda_grad >= 0 and lambda_l2 > 0 for a proper prior")
    if args.prior_type == "powerlaw" and args.prior_rms <= 0:
        raise ValueError("--prior-rms must be positive")
    if args.init == "prior" and args.prior_type != "powerlaw":
        raise ValueError("--init prior requires --prior-type powerlaw")
    if args.init == "file" and (args.init_file is None or not args.init_file.is_file()):
        raise FileNotFoundError("--init file requires an existing --init-file .npy map")
    if args.covariance_jitter_rel <= 0:
        raise ValueError("--covariance-jitter-rel must be positive")


def load_observed(path: Path, config: Namespace, data_index: int) -> np.ndarray:
    data = np.load(path).astype(np.float64, copy=False)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (H,W) or (N,H,W), got {data.shape}")
    if not 0 <= data_index < data.shape[0]:
        raise IndexError(f"data index {data_index} is outside a stack of {data.shape[0]} maps")
    expected = (int(config.target_size), int(config.target_size))
    if data.shape[-2:] != expected:
        raise ValueError(f"Expected image size {expected}, got {data.shape[-2:]}")
    data = data[data_index].copy()
    if not np.isfinite(data).all():
        raise ValueError("Observed data contains NaN or infinite values")
    if config.take_log:
        if np.any(data <= 0):
            raise ValueError("Covariance take_log=True, but observed data is not strictly positive")
        data = np.log(data)
    if config.subtract_mean:
        data = data - data.mean()
    if config.whiten:
        std = data.std()
        if std <= 0:
            raise ValueError("Cannot whiten a zero-variance observed image")
        data = (data - data.mean()) / std
    return data


def build_st_operator(size: int, config: Namespace, device: torch.device, dtype: torch.dtype) -> Any:
    example = DataClass(torch.zeros((1, 1, size, size), device=device, dtype=dtype), pbc=config.pbc)
    kwargs: dict[str, Any] = {
        "J": config.J,
        "L": config.L,
        "WType": config.wtype,
        "iso": config.iso,
        "angular_ft": config.angular_ft,
        "harmonics_angle": config.harmonics_angle,
        "scale_ft": config.scale_ft,
        "harmonics_scale": config.harmonics_scale,
        "dj": config.dj,
        "compute_PS": config.compute_ps,
        "has_fewer_convolutions": config.fewer_convolutions,
    }
    return example.get_ST_op(**kwargs)


def apply_st_operator(st_op: Any, data: DataClass, **kwargs: Any) -> Any:
    parameters = inspect.signature(st_op.apply).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return st_op.apply(data, **kwargs)
    return st_op.apply(data, **{k: v for k, v in kwargs.items() if k in parameters})


def load_model(
    path: Path,
    st_op: Any,
    device: torch.device,
    dtype: torch.dtype,
    jitter_rel: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray], float]:
    with np.load(path) as payload:
        required = ("mean", "covariance", "active_statistic_mask")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"Covariance NPZ is missing arrays: {missing}")
        mean_full = np.asarray(payload["mean"], dtype=np.float64)
        covariance_full = np.asarray(payload["covariance"], dtype=np.float64)
        active = np.asarray(payload["active_statistic_mask"], dtype=bool)
        if mean_full.shape != active.shape or covariance_full.shape != (mean_full.size, mean_full.size):
            raise ValueError("Inconsistent mean, covariance, and active-mask dimensions")

        references: dict[str, np.ndarray] = {}
        reference_keys = {
            "field_mean": "stl_reference_field_mean",
            "field_std": "stl_reference_field_std",
            "S2_ref_sqrt_chan_diag": "stl_reference_S2_ref_sqrt_chan_diag",
            "var_ref": "stl_reference_var_ref",
            "PS_ref_sqrt_chan_diag": "stl_reference_PS_ref_sqrt_chan_diag",
        }
        for name, key in reference_keys.items():
            if key not in payload.files:
                raise KeyError(f"Covariance NPZ is missing STL reference array {key!r}")
            references[name] = np.array(payload[key], copy=True)

    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    st_op.S2_ref_sqrt_chan_diag = torch.as_tensor(
        references["S2_ref_sqrt_chan_diag"], device=device, dtype=complex_dtype
    )
    st_op.var_ref = torch.as_tensor(references["var_ref"], device=device, dtype=dtype)
    if bool(st_op.compute_PS):
        if references["PS_ref_sqrt_chan_diag"].size == 0:
            raise ValueError("compute_ps=True but the stored PS reference is empty")
        st_op.PS_ref_sqrt_chan_diag = torch.as_tensor(
            references["PS_ref_sqrt_chan_diag"], device=device, dtype=complex_dtype
        )

    covariance = covariance_full[np.ix_(active, active)]
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    positive_diagonal = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
    scale = float(np.median(positive_diagonal)) if positive_diagonal.size else 1.0
    jitter = jitter_rel * scale
    covariance = covariance + jitter * np.eye(covariance.shape[0])

    mean_t = torch.as_tensor(mean_full[active], device=device, dtype=dtype)
    active_t = torch.as_tensor(active, device=device, dtype=torch.bool)
    covariance_t = torch.as_tensor(covariance, device=device, dtype=dtype)
    try:
        covariance_cholesky_t = torch.linalg.cholesky(covariance_t)
    except RuntimeError as exc:
        raise RuntimeError(
            "Regularized active covariance is not positive definite; increase "
            "--covariance-jitter-rel"
        ) from exc
    return mean_t, covariance_cholesky_t, active_t, references, jitter


def scattering_vector(x: torch.Tensor, st_op: Any, config: Namespace) -> torch.Tensor:
    """Compute phi exactly as used for covariance samples: self-standardize first."""
    data = DataClass(x[None, None], pbc=config.pbc)
    standardized, _field_mean, _field_std = st_op.wavelet_op.standardize(
        data, mean_field=False, inplace=False
    )
    stats = apply_st_operator(
        st_op,
        standardized,
        norm="load_ref",
        norm_batch_mean=False,
        compute_PS=config.compute_ps,
    )
    return stats.to_flatten(
        keep_batch_dim=True,
        mean_along_batch=False,
        keepnans=False,
        flatten_complex=True,
    ).real[0]


def likelihood_energy(
    residual: torch.Tensor,
    st_op: Any,
    config: Namespace,
    mean: torch.Tensor,
    covariance_cholesky: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    phi_full = scattering_vector(residual, st_op, config)
    if phi_full.numel() != active.numel():
        raise RuntimeError(
            f"STL produced {phi_full.numel()} statistics but the stored mask has {active.numel()}"
        )
    phi = phi_full[active]
    diff = phi - mean
    solved = torch.cholesky_solve(diff[:, None], covariance_cholesky)[:, 0]
    return 0.5 * torch.dot(diff, solved)


def quadratic_prior(s: torch.Tensor, lambda_grad: float, lambda_l2: float) -> torch.Tensor:
    dx = torch.roll(s, shifts=-1, dims=0) - s
    dy = torch.roll(s, shifts=-1, dims=1) - s
    return 0.5 * lambda_grad * (dx.square().sum() + dy.square().sum()) + 0.5 * lambda_l2 * s.square().sum()


def build_powerlaw_spectrum(
    shape: tuple[int, int],
    spectral_index: float,
    target_rms: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed P(k) proportional to k**spectral_index on the rFFT grid."""
    height, width = shape
    kx = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype) * height
    ky = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype) * width
    radius = torch.sqrt(kx[:, None].square() + ky[None, :].square())
    nonzero = radius > 0
    raw = torch.zeros_like(radius)
    raw[nonzero] = radius[nonzero].pow(spectral_index)

    # Interior rFFT columns represent both positive and negative ky modes.
    weights = torch.ones_like(raw)
    if width % 2 == 0:
        weights[:, 1:-1] = 2.0
    else:
        weights[:, 1:] = 2.0
    normalization = target_rms**2 * (height * width) / torch.sum(weights * raw)
    spectrum = normalization * raw
    spectrum[0, 0] = 1.0  # placeholder only; the zero mode is excluded
    return spectrum, weights


def powerlaw_prior(
    s: torch.Tensor, spectrum: torch.Tensor, rfft_weights: torch.Tensor
) -> torch.Tensor:
    coefficients = torch.fft.rfft2(s, norm="ortho")
    inverse_spectrum = torch.reciprocal(spectrum).clone()
    inverse_spectrum[0, 0] = 0.0
    return 0.5 * torch.sum(
        rfft_weights * coefficients.abs().square() * inverse_spectrum
    )


def load_signal_map(path: Path, observed: torch.Tensor, label: str) -> torch.Tensor:
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != tuple(observed.shape):
        raise ValueError(f"{label} has shape {array.shape}; expected {tuple(observed.shape)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return torch.as_tensor(array, device=observed.device, dtype=observed.dtype).clone()


def initialize_signal(
    args: argparse.Namespace,
    observed: torch.Tensor,
    generator: torch.Generator,
    power_spectrum: torch.Tensor | None,
) -> torch.Tensor:
    if args.init == "zeros":
        signal = torch.zeros_like(observed)
    elif args.init == "data":
        signal = observed.clone()
    elif args.init == "file":
        assert args.init_file is not None
        signal = load_signal_map(args.init_file, observed, "Initial signal")
    else:
        white = torch.randn(
            observed.shape, device=observed.device, dtype=observed.dtype, generator=generator
        )
        if args.init == "prior":
            if power_spectrum is None:
                raise RuntimeError("Power-law prior spectrum was not constructed")
            coefficients = torch.fft.rfft2(white, norm="ortho") * torch.sqrt(power_spectrum)
            coefficients[0, 0] = 0.0
            signal = torch.fft.irfft2(coefficients, s=observed.shape, norm="ortho")
        else:
            signal = args.init_std * white
    if args.zero_mean_signal:
        signal = signal - signal.mean()
    return signal


def save_plot(
    path: Path,
    observed: np.ndarray,
    samples: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    max_samples: int,
) -> None:
    n_show = min(max_samples, samples.shape[0])
    columns = 2 + n_show
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 7), constrained_layout=True)
    ordinary = [observed, posterior_mean, *samples[:n_show]]
    vmin = float(min(np.percentile(image, 1) for image in ordinary))
    vmax = float(max(np.percentile(image, 99) for image in ordinary))

    top_images = [observed, posterior_mean, *samples[:n_show]]
    top_titles = ["observed d", "posterior mean", *[f"sample {i + 1}" for i in range(n_show)]]
    bottom_images = [posterior_std, observed - posterior_mean, *[observed - x for x in samples[:n_show]]]
    bottom_titles = ["posterior std", "mean residual", *[f"residual {i + 1}" for i in range(n_show)]]
    for ax, image, title in zip(axes[0], top_images, top_titles):
        im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    residuals = bottom_images[1:]
    rlim = float(max(np.percentile(np.abs(image), 99) for image in residuals))
    for index, (ax, image, title) in enumerate(zip(axes[1], bottom_images, bottom_titles)):
        if index == 0:
            im = ax.imshow(image, origin="lower", cmap="viridis")
        else:
            im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_diagnostics_plot(
    path: Path,
    total_energy: np.ndarray,
    likelihood_energy_values: np.ndarray,
    prior_energy: np.ndarray,
    gradient_rms: np.ndarray,
    rho: np.ndarray,
    anchor_distance_rms: np.ndarray,
    active_dimension: int,
    burn_in: int,
) -> None:
    steps = np.arange(1, total_energy.size + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, total_energy, label="total", lw=1.2)
    axes[0, 0].plot(steps, likelihood_energy_values, label="likelihood", lw=1.0)
    axes[0, 0].plot(steps, prior_energy, label="prior", lw=1.0)
    axes[0, 0].set_ylabel("energy")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, rho, lw=1.2)
    axes[0, 1].axhline(math.sqrt(active_dimension), color="black", ls="--", label=r"$\sqrt{d_{active}}$")
    axes[0, 1].set_ylabel(r"residual $\rho$")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, gradient_rms, lw=1.2)
    axes[1, 0].set_ylabel("gradient RMS")
    axes[1, 0].set_xlabel("ULA step")

    if np.isfinite(anchor_distance_rms).any():
        axes[1, 1].plot(steps, anchor_distance_rms, lw=1.2)
        axes[1, 1].set_ylabel("RMS distance from anchor")
    else:
        axes[1, 1].plot(steps, likelihood_energy_values / active_dimension, lw=1.2)
        axes[1, 1].axhline(0.5, color="black", ls="--", label="Gaussian reference")
        axes[1, 1].set_ylabel(r"$U_{like}/d_{active}$")
        axes[1, 1].legend()
    axes[1, 1].set_xlabel("ULA step")

    if 0 < burn_in < total_energy.size:
        for ax in axes.ravel():
            ax.axvline(burn_in, color="tab:red", ls=":", alpha=0.8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    validate_args(args)
    metadata, config = load_metadata(args.covariance_json)
    dtype_name = args.dtype or str(config.dtype)
    dtype = torch_dtype(dtype_name)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False; use --device cpu")
        torch.cuda.set_device(device)
    configure_backend(device, dtype)

    observed_np = load_observed(args.data, config, args.data_index)
    observed = torch.as_tensor(observed_np, device=device, dtype=dtype)
    st_op = build_st_operator(int(config.target_size), config, device, dtype)
    mean, covariance_cholesky, active, references, jitter = load_model(
        args.covariance_npz, st_op, device, dtype, args.covariance_jitter_rel
    )

    power_spectrum = None
    rfft_weights = None
    if args.prior_type == "powerlaw":
        power_spectrum, rfft_weights = build_powerlaw_spectrum(
            tuple(observed.shape), args.prior_spectral_index, args.prior_rms, device, dtype
        )

    anchor = None
    if args.prior_type == "anchor":
        assert args.anchor_file is not None
        anchor = load_signal_map(args.anchor_file, observed, "Anchor signal")
        if args.zero_mean_signal:
            anchor = anchor - anchor.mean()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    signal = initialize_signal(args, observed, generator, power_spectrum).detach()
    initial_signal_np = signal.detach().cpu().numpy().copy()
    samples: list[np.ndarray] = []
    total_history: list[float] = []
    likelihood_history: list[float] = []
    prior_history: list[float] = []
    gradient_norm_history: list[float] = []
    gradient_rms_history: list[float] = []
    rho_history: list[float] = []
    anchor_distance_history: list[float] = []
    elapsed_history: list[float] = []
    start_time = time.monotonic()

    print(f"device={device} dtype={dtype_name}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    print(f"data={args.data} shape={tuple(observed.shape)}", flush=True)
    print(f"covariance={args.covariance_npz}", flush=True)
    print(f"statistics={active.numel()} active={int(active.sum())} jitter={jitter:.6e}", flush=True)
    if args.prior_type == "anchor":
        assert args.anchor_file is not None
        print(
            f"prior=anchor centre={args.anchor_file} "
            f"precision={args.anchor_precision:.6g} zero_mean={args.zero_mean_signal}",
            flush=True,
        )
    elif args.prior_type == "powerlaw":
        print(
            f"prior=powerlaw spectral_index={args.prior_spectral_index:.6g} "
            f"target_rms={args.prior_rms:.6g} zero_mean={args.zero_mean_signal}",
            flush=True,
        )
    else:
        print(
            f"prior=quadratic lambda_grad={args.lambda_grad:.6g} "
            f"lambda_l2={args.lambda_l2:.6g}",
            flush=True,
        )
    print(
        f"ULA steps={args.n_steps} burn_in={args.burn_in} thin={args.thin} "
        f"epsilon={args.step_size:.3e}",
        flush=True,
    )

    for iteration in range(args.n_steps):
        signal.requires_grad_(True)
        residual = observed - signal
        like = likelihood_energy(residual, st_op, config, mean, covariance_cholesky, active)
        if args.prior_type == "anchor":
            assert anchor is not None
            prior = 0.5 * args.anchor_precision * (signal - anchor).square().sum()
        elif args.prior_type == "powerlaw":
            assert power_spectrum is not None and rfft_weights is not None
            prior = powerlaw_prior(signal, power_spectrum, rfft_weights)
        else:
            prior = quadratic_prior(signal, args.lambda_grad, args.lambda_l2)
        total = like + prior
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite energy at iteration {iteration}")
        (gradient,) = torch.autograd.grad(total, signal)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite gradient at iteration {iteration}")

        grad_norm = torch.linalg.vector_norm(gradient)
        grad_rms = grad_norm / math.sqrt(gradient.numel())
        with torch.no_grad():
            noise = torch.randn(signal.shape, device=device, dtype=dtype, generator=generator)
            if args.zero_mean_signal:
                noise = noise - noise.mean()
            signal = signal - args.step_size * gradient + math.sqrt(2.0 * args.step_size) * noise
            if args.zero_mean_signal:
                signal = signal - signal.mean()

        like_value = float(like.detach().cpu())
        prior_value = float(prior.detach().cpu())
        total_history.append(float(total.detach().cpu()))
        likelihood_history.append(like_value)
        prior_history.append(prior_value)
        gradient_norm_history.append(float(grad_norm.detach().cpu()))
        gradient_rms_history.append(float(grad_rms.detach().cpu()))
        rho_history.append(math.sqrt(max(2.0 * like_value, 0.0)))
        if anchor is None:
            anchor_distance_history.append(float("nan"))
        else:
            anchor_distance_history.append(
                float(torch.sqrt(torch.mean((signal.detach() - anchor).square())).cpu())
            )
        elapsed_history.append(time.monotonic() - start_time)

        step_number = iteration + 1
        if step_number > args.burn_in and (step_number - args.burn_in) % args.thin == 0:
            samples.append(signal.detach().cpu().numpy().copy())
        if step_number == 1 or step_number % args.print_every == 0 or step_number == args.n_steps:
            anchor_text = (
                "" if anchor is None else f" anchor_rms={anchor_distance_history[-1]:.6e}"
            )
            print(
                f"step={step_number:5d} U={total_history[-1]:.6e} "
                f"U_like={like_value:.6e} U_prior={prior_value:.6e} "
                f"grad_rms={gradient_rms_history[-1]:.6e} rho={rho_history[-1]:.4f} "
                f"saved={len(samples)}{anchor_text}",
                flush=True,
            )

    if not samples:
        raise RuntimeError("No samples were saved; adjust --n-steps, --burn-in, or --thin")
    sample_array = np.stack(samples)
    posterior_mean = sample_array.mean(axis=0)
    posterior_std = sample_array.std(axis=0)
    residual_array = observed_np[None] - sample_array

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"{args.run_name}_samples.npz"
    plot_path = args.output_dir / f"{args.run_name}_summary.png"
    diagnostics_plot_path = args.output_dir / f"{args.run_name}_diagnostics.png"
    json_path = args.output_dir / f"{args.run_name}_summary.json"
    np.savez_compressed(
        npz_path,
        observed_data=observed_np,
        initial_signal=initial_signal_np,
        anchor_signal=(
            np.array([]) if anchor is None else anchor.detach().cpu().numpy()
        ),
        samples=sample_array,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        residuals=residual_array,
        total_energy=np.asarray(total_history),
        likelihood_energy=np.asarray(likelihood_history),
        prior_energy=np.asarray(prior_history),
        gradient_norm=np.asarray(gradient_norm_history),
        gradient_rms=np.asarray(gradient_rms_history),
        rho=np.asarray(rho_history),
        anchor_distance_rms=np.asarray(anchor_distance_history),
        elapsed_seconds=np.asarray(elapsed_history),
        active_statistic_mask=active.detach().cpu().numpy(),
        scattering_mean=mean.detach().cpu().numpy(),
        stl_reference_field_mean=references["field_mean"],
        stl_reference_field_std=references["field_std"],
        prior_power_spectrum=(
            np.array([]) if power_spectrum is None else power_spectrum.detach().cpu().numpy()
        ),
    )
    save_plot(
        plot_path, observed_np, sample_array, posterior_mean, posterior_std, args.max_plot_samples
    )
    save_diagnostics_plot(
        diagnostics_plot_path,
        np.asarray(total_history),
        np.asarray(likelihood_history),
        np.asarray(prior_history),
        np.asarray(gradient_rms_history),
        np.asarray(rho_history),
        np.asarray(anchor_distance_history),
        int(active.sum().item()),
        args.burn_in,
    )
    summary = {
        "warning": "ULA samples remain step-size dependent and require diagnostic review.",
        "model": (
            "scattering likelihood plus component-separation-centred quadratic anchor"
            if args.prior_type == "anchor"
            else f"scattering likelihood plus {args.prior_type} Gaussian prior"
        ),
        "data": str(args.data),
        "covariance_npz": str(args.covariance_npz),
        "covariance_json": str(args.covariance_json),
        "npz_output": str(npz_path),
        "plot_output": str(plot_path),
        "diagnostics_plot_output": str(diagnostics_plot_path),
        "n_saved_samples": int(sample_array.shape[0]),
        "active_statistic_dimension": int(active.sum().item()),
        "covariance_jitter": jitter,
        "expected_gaussian_rho": math.sqrt(int(active.sum().item())),
        "final_rho": rho_history[-1],
        "final_anchor_distance_rms": (
            None if anchor is None else anchor_distance_history[-1]
        ),
        "elapsed_seconds": elapsed_history[-1],
        "ula": {key: json_value(value) for key, value in vars(args).items()},
        "statistic_config": metadata["config"],
        "standardization_note": (
            "Each residual is self-standardized before STL, matching the covariance-sample "
            "calculation. Stored field mean/std are loaded for provenance but not applied."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={npz_path}", flush=True)
    print(f"saved={plot_path}", flush=True)
    print(f"saved={diagnostics_plot_path}", flush=True)
    print(f"saved={json_path}", flush=True)


if __name__ == "__main__":
    main()
