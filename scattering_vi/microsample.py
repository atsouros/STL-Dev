#!/usr/bin/env python3
"""MGVI reconstruction with a scattering noise likelihood.

The observed image is d = s + n. The scattering Gaussian model is used for
the noise field n = d - s. NIFTy provides the signal model/prior for s.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nifty8 as ift
import numpy as np
import torch


def find_project_root() -> Path:
    here = Path.cwd().resolve()
    candidates = [here, *here.parents]
    for cand in candidates:
        if (cand / "STL-Dev" / "scattering_vi").exists():
            return cand
        if (cand / "scattering_vi").exists() and (cand / "STL_main").exists():
            return cand
        if cand.name == "scattering_vi" and (cand.parent / "STL_main").exists():
            return cand.parent
    raise FileNotFoundError(
        "Could not find project root containing either STL-Dev/scattering_vi "
        "or the Jean Zay layout with scattering_vi and STL_main side by side. "
        "Run this script from the repository root, scattering_vi, or a subdirectory."
    )


PROJECT_ROOT = find_project_root()
if (PROJECT_ROOT / "STL-Dev" / "scattering_vi").exists():
    SCATTERING_VI = PROJECT_ROOT / "STL-Dev" / "scattering_vi"
    STL_REPO = PROJECT_ROOT / "STL-Dev"
else:
    SCATTERING_VI = PROJECT_ROOT / "scattering_vi"
    STL_REPO = PROJECT_ROOT
for path in (str(SCATTERING_VI), str(STL_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

import STL_main.torch_backend as bk  # noqa: E402
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass  # noqa: E402
from nifty8.operators.simple_linear_operators import VdotOperator  # noqa: E402


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def configure_backend(device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64 if dtype == torch.float32 else torch.complex128


def load_data_stack(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(args.input).astype(np.float64, copy=False)
    if data.ndim == 2:
        data = data[None, :, :]
    if data.ndim != 3:
        raise ValueError(f"Expected data stack with shape (S,H,W), got {data.shape}")
    if data.shape[-2:] != (args.target_size, args.target_size):
        raise ValueError(
            f"Expected maps of size {args.target_size}x{args.target_size}, got {data.shape}"
        )
    if args.take_log:
        if np.any(data <= 0):
            raise ValueError("take_log=True requires strictly positive input values")
        data = np.log(data)
    if args.subtract_mean:
        data = data - data.mean(axis=(-2, -1), keepdims=True)
    if args.whiten:
        mean = data.mean(axis=(-2, -1), keepdims=True)
        std = data.std(axis=(-2, -1), keepdims=True)
        if np.any(std <= 0):
            raise ValueError("whiten=True encountered a zero-variance map")
        data = (data - mean) / std
    return data, {
        "input_path": str(args.input),
        "data_shape": list(data.shape),
        "n_data_maps": int(data.shape[0]),
        "take_log": args.take_log,
        "subtract_mean": args.subtract_mean,
        "whiten": args.whiten,
        "data_mean": float(data.mean()),
        "data_std": float(data.std()),
    }


def build_st_operator(data: DataClass, args: argparse.Namespace, *, replace_nan_value: Any = None) -> Any:
    kwargs: dict[str, Any] = {
        "J": args.J,
        "L": args.L,
        "WType": args.wtype,
        "iso": args.iso,
        "angular_ft": args.angular_ft,
        "harmonics_angle": args.harmonics_angle,
        "scale_ft": args.scale_ft,
        "harmonics_scale": args.harmonics_scale,
        "dj": args.dj,
        "compute_PS": args.compute_ps,
        "has_fewer_convolutions": args.fewer_convolutions,
    }
    if replace_nan_value is not None:
        kwargs["replace_nan_value"] = replace_nan_value
    return data.get_ST_op(**kwargs)


def apply_st_operator(st_op: Any, data: DataClass, **kwargs: Any) -> Any:
    signature = inspect.signature(st_op.apply)
    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return st_op.apply(data, **kwargs)
    supported = {k: v for k, v in kwargs.items() if k in parameters}
    return st_op.apply(data, **supported)


def apply_standardized_st(
    st_op: Any,
    data: DataClass,
    *,
    mean_field: bool,
    mark_standardized: bool,
    **kwargs: Any,
) -> Any:
    standardized_data, mean_pre_std, std_pre_std = st_op.wavelet_op.standardize(
        data, mean_field=mean_field, inplace=False
    )
    stats = apply_st_operator(st_op, standardized_data, **kwargs)
    if mark_standardized:
        stats.standardized = True
        stats.mean_pre_std = mean_pre_std
        stats.std_pre_std = std_pre_std
    return stats


def target_statistics(target: torch.Tensor, args: argparse.Namespace) -> tuple[Any, Any, torch.Tensor]:
    if target.ndim == 2:
        array = target[None, None, :, :]
    elif target.ndim == 3:
        array = target[:, None, :, :]
    else:
        raise ValueError(f"Expected target with ndim 2 or 3, got {target.shape}")
    data = DataClass(array, pbc=args.pbc)
    st_op = build_st_operator(data, args)
    with torch.no_grad():
        stats = apply_standardized_st(
            st_op,
            data,
            mean_field=True,
            mark_standardized=True,
            norm="store_ref",
            norm_batch_mean=True,
            compute_PS=args.compute_ps,
        )
        flat = stats.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=True,
            keepnans=False,
            flatten_complex=True,
        ).real[0]
    return stats, st_op, flat.detach()


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def require_existing(path: Path, label: str, candidates: tuple[Path, ...] = ()) -> Path:
    if path.exists():
        return path
    searched = [path, *candidates]
    lines = "\n".join(f"  - {p}" for p in searched)
    raise FileNotFoundError(
        f"Missing {label}: {path}\n"
        f"Searched:\n{lines}\n"
        "Pass the correct path with the corresponding command-line option."
    )


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIFTy8 MGVI for d=s+n using a scattering Gaussian model for n=d-s."
    )
    p.add_argument("--covariance-npz", type=Path, required=True)
    p.add_argument("--covariance-json", type=Path, required=True)
    p.add_argument("--data-input", type=Path, required=True)
    p.add_argument("--initial-signal", type=Path, default=None)
    p.add_argument("--initialization-iterations", type=int, default=40)
    p.add_argument("--initialization-relative-noise", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=SCATTERING_VI / "samples.npz")
    p.add_argument("--plot", type=Path, default=SCATTERING_VI / "samples.png")
    p.add_argument("--diagnostics-summary", type=Path, default=SCATTERING_VI / "diagnostics_summary.json")
    p.add_argument("--device", default="cuda:0", help="Use cuda:0 on a GPU node; use cpu only for debugging.")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-index", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--vi-iterations", type=int, default=4)
    p.add_argument("--map-iterations", type=int, default=30)
    p.add_argument("--kl-iterations", type=int, default=25)
    p.add_argument("--sampling-iterations", type=int, default=100)
    p.add_argument("--geovi-iterations", type=int, default=8)
    p.add_argument("--covariance-jitter-rel", type=float, default=1e-6)
    p.add_argument("--metric-strength", type=float, default=1.0)
    p.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--run-map", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-vi", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vi-start-from-map", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-geovi", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-correlated-field-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prior-loglog-slope-mean", type=float, default=-3.0)
    p.add_argument("--prior-loglog-slope-std", type=float, default=0.5)
    p.add_argument("--prior-fluctuations-mean", type=float, default=1.0)
    p.add_argument("--prior-fluctuations-std", type=float, default=0.5)
    args = p.parse_args()
    args.covariance_npz = require_existing(args.covariance_npz, "covariance NPZ")
    args.covariance_json = require_existing(args.covariance_json, "covariance JSON")
    args.data_input = require_existing(args.data_input, "observed data input")
    if args.initial_signal is not None:
        args.initial_signal = require_existing(args.initial_signal, "initial signal")
    return args


def args_from_covariance_json(json_path: Path, input_path: Path, *, device: str, dtype: str) -> Namespace:
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        config = dict(metadata.get("config", {}))
    else:
        config = {}

    defaults = {
        "input": input_path,
        "target_size": 256,
        "take_log": False,
        "subtract_mean": False,
        "whiten": False,
        "device": device,
        "dtype": dtype,
        "J": 7,
        "L": 4,
        "wtype": "Bump-Steerable",
        "iso": True,
        "angular_ft": True,
        "harmonics_angle": 2,
        "scale_ft": True,
        "harmonics_scale": 3,
        "dj": 3,
        "compute_ps": False,
        "fewer_convolutions": False,
        "pbc": True,
        "stats_chunk_size": 32,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    config["input"] = Path(input_path)
    config["device"] = device
    config["dtype"] = dtype
    return Namespace(**config)


def load_scattering_gaussian(npz_path: Path, jitter_rel: float = 1e-6):
    payload = np.load(npz_path)
    missing = [key for key in ("mean", "covariance") if key not in payload.files]
    if missing:
        raise KeyError(f"Covariance product is missing required keys: {missing}")
    mu = payload["mean"]
    covariance = payload["covariance"]

    active = payload["active_statistic_mask"] if "active_statistic_mask" in payload.files else np.ones_like(mu, dtype=bool)
    mu = np.asarray(mu, dtype=np.float64)[active]
    covariance = np.asarray(covariance, dtype=np.float64)[np.ix_(active, active)]
    covariance = 0.5 * (covariance + covariance.T)

    diag_scale = float(np.nanmedian(np.diag(covariance)))
    if not np.isfinite(diag_scale) or diag_scale <= 0:
        diag_scale = float(np.nanmax(np.diag(covariance)))
    if not np.isfinite(diag_scale) or diag_scale <= 0:
        diag_scale = 1.0
    covariance = covariance + (jitter_rel * diag_scale) * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance, hermitian=True)
    return mu, covariance, precision, active


def load_scattering_reference(st_op, npz_path: Path, device: torch.device, dtype: torch.dtype) -> None:
    payload = np.load(npz_path)
    aliases = {
        "S2_ref_sqrt_chan_diag": ("stl_reference_S2_ref_sqrt_chan_diag", "st_S2_ref_sqrt_chan_diag"),
        "var_ref": ("stl_reference_var_ref", "st_var_ref"),
        "PS_ref_sqrt_chan_diag": ("stl_reference_PS_ref_sqrt_chan_diag", "st_PS_ref_sqrt_chan_diag"),
    }
    for attr, keys in aliases.items():
        for key in keys:
            if key in payload.files and payload[key].size:
                setattr(st_op, attr, torch.as_tensor(payload[key], device=device, dtype=dtype))
                break


def scattering_vector_with_grad(x_2d: torch.Tensor, st_op, args: Namespace) -> torch.Tensor:
    if x_2d.ndim != 2:
        raise ValueError(f"expected a 2D map, got shape {tuple(x_2d.shape)}")
    data = DataClass(x_2d[None, None, :, :], pbc=args.pbc)
    stats = apply_standardized_st(
        st_op,
        data,
        mean_field=False,
        mark_standardized=False,
        norm="load_ref",
        compute_PS=args.compute_ps,
    )
    return stats.to_flatten(
        keep_batch_dim=True,
        mean_along_batch=False,
        keepnans=False,
        flatten_complex=True,
    ).real[0]


class _ScatteringWhitenedResidual(ift.Operator):
    def __init__(
        self,
        domain,
        st_op,
        args,
        mu_t,
        precision_sqrt_t,
        active_mask_t,
        device,
        dtype,
        observed_data_t=None,
    ):
        self._domain = ift.makeDomain(domain)
        self._target = ift.UnstructuredDomain(tuple(mu_t.shape))
        self._st_op = st_op
        self._args = args
        self._mu_t = mu_t
        self._precision_sqrt_t = precision_sqrt_t
        self._active_mask_t = active_mask_t
        self._device = device
        self._dtype = dtype
        self._observed_data_t = observed_data_t

    @staticmethod
    def _as_numeric_array(x_value) -> np.ndarray:
        if hasattr(x_value, "val"):
            x_value = x_value.val
        if isinstance(x_value, dict):
            if len(x_value) != 1:
                raise TypeError(f"expected one field value, got keys {list(x_value)}")
            x_value = next(iter(x_value.values()))
        return np.asarray(x_value)

    def apply(self, x):
        self._check_input(x)
        x_arr = self._as_numeric_array(x.val)
        with torch.no_grad():
            x_t = torch.as_tensor(np.array(x_arr, copy=True), device=self._device, dtype=self._dtype)
            field_t = self._observed_data_t - x_t if self._observed_data_t is not None else x_t
            phi = scattering_vector_with_grad(field_t, self._st_op, self._args)[self._active_mask_t]
            residual = self._precision_sqrt_t @ (phi - self._mu_t)
        return ift.makeField(self._target, residual.detach().cpu().numpy())


class ScatteringGaussianEnergy(ift.LikelihoodEnergyOperator):
    def __init__(
        self,
        domain,
        st_op,
        args,
        mu,
        precision,
        active_mask,
        *,
        device,
        dtype,
        metric_strength=1.0,
        observed_data=None,
    ):
        self._domain = ift.makeDomain(domain)
        self._st_op = st_op
        self._args = args
        self._device = device
        self._dtype = dtype
        self._active_mask_t = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
        self._mu_t = torch.as_tensor(mu, device=device, dtype=dtype)
        self._precision_t = torch.as_tensor(precision, device=device, dtype=dtype)
        evals, evecs = np.linalg.eigh(0.5 * (precision + precision.T))
        evals = np.clip(evals, 0.0, None)
        precision_sqrt = (evecs * np.sqrt(evals)[None, :]) @ evecs.T
        self._precision_sqrt_t = torch.as_tensor(precision_sqrt, device=device, dtype=dtype)
        self._observed_data_t = (
            None
            if observed_data is None
            else torch.as_tensor(observed_data, device=device, dtype=dtype)
        )
        self._data_domain = ift.UnstructuredDomain(mu.shape)
        self._res = _ScatteringWhitenedResidual(
            self._domain, st_op, args, self._mu_t, self._precision_sqrt_t,
            self._active_mask_t, device, dtype, observed_data_t=self._observed_data_t
        )
        self._sqrt_data_metric_at = lambda x: ift.ScalingOperator(self._data_domain, 1.0)
        self._name = "scattering_noise"
        self._metric_strength = float(metric_strength)

    @property
    def data_domain(self):
        return self._data_domain

    def normalized_residual(self, x):
        x_arr = self._as_numeric_array(x.val if hasattr(x, "val") else x)
        with torch.no_grad():
            x_t = torch.as_tensor(np.array(x_arr, copy=True), device=self._device, dtype=self._dtype)
            field_t = self._observed_data_t - x_t if self._observed_data_t is not None else x_t
            phi = scattering_vector_with_grad(field_t, self._st_op, self._args)[self._active_mask_t]
            residual = self._precision_sqrt_t @ (phi - self._mu_t)
        return ift.makeField(self._data_domain, residual.detach().cpu().numpy())

    @staticmethod
    def _as_numeric_array(x_value) -> np.ndarray:
        if hasattr(x_value, "val"):
            x_value = x_value.val
        if isinstance(x_value, dict):
            if len(x_value) != 1:
                raise TypeError(f"expected one field value, got keys {list(x_value)}")
            x_value = next(iter(x_value.values()))
        arr = np.asarray(x_value)
        if arr.dtype == object:
            raise TypeError(f"could not unwrap NIFTy value into a numeric array; got dtype={arr.dtype}")
        return arr

    def get_transformation(self):
        return (np.float64, self._res)

    def _value_and_gradient(self, x_np: np.ndarray):
        x_arr = self._as_numeric_array(x_np)
        x_t = torch.as_tensor(np.array(x_arr, copy=True), device=self._device, dtype=self._dtype)
        x_t.requires_grad_(True)
        field_t = self._observed_data_t - x_t if self._observed_data_t is not None else x_t
        phi_full = scattering_vector_with_grad(field_t, self._st_op, self._args)
        if phi_full.numel() != self._active_mask_t.numel() or not torch.isfinite(phi_full).all():
            return 1e100, np.zeros_like(x_arr, dtype=np.float64)
        phi = phi_full[self._active_mask_t]
        diff = phi - self._mu_t
        energy = 0.5 * diff @ (self._precision_t @ diff)
        if not torch.isfinite(energy):
            return 1e100, np.zeros_like(x_arr, dtype=np.float64)
        energy.backward()
        grad = x_t.grad.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.isfinite(grad).all():
            return 1e100, np.zeros_like(x_arr, dtype=np.float64)
        return float(energy.detach().cpu()), grad

    def apply(self, x):
        self._check_input(x)
        value, grad_np = self._value_and_gradient(x.val)
        value_field = ift.Field.scalar(value)
        if x.jac is None:
            return value_field
        grad_field = ift.makeField(self._domain, grad_np)
        jac = VdotOperator(grad_field)
        res = x.new(value_field, jac)
        if x.want_metric:
            metric = ift.ScalingOperator(
                self._domain,
                self._metric_strength,
                sampling_dtype=np.float64,
            )
            res = res.add_metric(metric)
        return res


def make_plot(
    path: Path,
    observed_data: np.ndarray,
    data: np.ndarray,
    map_signal,
    posterior_mean,
    posterior_std,
    signal_samples: np.ndarray,
) -> None:
    images: list[tuple[np.ndarray, str, bool]] = [
        (observed_data, "observed data", False),
    ]
    if map_signal is not None:
        images.append((np.asarray(map_signal.val), "MAP", False))
    if posterior_mean is not None:
        images.append((np.asarray(posterior_mean.val), "MGVI mean", False))
    if posterior_std is not None:
        images.append((np.asarray(posterior_std.val), "MGVI std", True))
    for ii in range(min(3, signal_samples.shape[0])):
        images.append((signal_samples[ii], f"posterior sample {ii + 1}", False))

    ordinary = [image for image, _title, is_std in images if not is_std]
    vmin = float(min(np.percentile(image, 1) for image in ordinary))
    vmax = float(max(np.percentile(image, 99) for image in ordinary))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (image, title, is_std) in zip(axes, images):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="viridis" if is_std else "RdBu_r",
            vmin=None if is_std else vmin,
            vmax=None if is_std else vmax,
        )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(images):]:
        ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def initial_position(domain, *, seed: int, xi_std: float = 0.1):
    values = {}
    rng = np.random.default_rng(seed)
    for key in domain.keys():
        shape = domain[key].shape
        values[key] = np.zeros(shape, dtype=np.float64)
        if key.endswith("xi"):
            values[key] = rng.normal(0.0, xi_std, size=shape)
    return ift.makeField(domain, values)


def fit_initial_signal(signal_op, initial_signal: np.ndarray, *, iterations: int, relative_noise: float, seed: int):
    if initial_signal.shape != signal_op.target.shape:
        raise ValueError(
            f"Initial signal shape {initial_signal.shape} does not match "
            f"the signal domain {signal_op.target.shape}."
        )
    sigma = relative_noise * float(np.std(initial_signal))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("The initial signal must have finite, non-zero variance.")

    target = ift.makeField(signal_op.target, initial_signal)
    inverse_covariance = ift.ScalingOperator(
        signal_op.target, 1.0 / sigma**2, sampling_dtype=np.float64
    )
    matching_likelihood = ift.GaussianEnergy(
        data=target, inverse_covariance=inverse_covariance
    ) @ signal_op
    hamiltonian = ift.StandardHamiltonian(matching_likelihood)
    start = initial_position(matching_likelihood.domain, seed=seed)
    energy = ift.EnergyAdapter(start, hamiltonian, want_metric=True, nanisinf=True)
    controller = ift.GradientNormController(
        tol_abs_gradnorm=1e-5, iteration_limit=iterations
    )
    minimizer = ift.NewtonCG(controller, enable_logging=True)
    solution, status = minimizer(energy)
    fitted_signal = signal_op(solution.position)
    relative_error = float(
        np.linalg.norm(fitted_signal.val - initial_signal)
        / max(np.linalg.norm(initial_signal), np.finfo(float).eps)
    )
    print("initial-signal latent fit status:", status, flush=True)
    print("initial-signal latent fit relative L2 error:", relative_error, flush=True)
    return solution.position, fitted_signal, relative_error


def main() -> None:
    cli = parse_cli()
    if cli.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        torch.cuda.set_device(int(cli.device.split(":", 1)[1]) if ":" in cli.device else 0)
        print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)

    args = args_from_covariance_json(cli.covariance_json, cli.data_input, device=cli.device, dtype=cli.dtype)
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)

    ift.random.push_sseq_from_seed(cli.seed)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cli.seed)

    mu_np, covariance_np, precision_np, active_mask_np = load_scattering_gaussian(
        cli.covariance_npz, jitter_rel=cli.covariance_jitter_rel
    )
    data_args = Namespace(**vars(args))
    data_args.input = cli.data_input
    data_stack, input_info = load_data_stack(data_args)
    if not 0 <= cli.data_index < data_stack.shape[0]:
        raise IndexError(
            f"--data-index={cli.data_index} is outside a stack of "
            f"{data_stack.shape[0]} maps"
        )
    data_example_t = torch.as_tensor(data_stack[:1].copy(), device=device, dtype=dtype)
    _data_stats, st_op, data_flat_t = target_statistics(data_example_t, args)
    load_scattering_reference(st_op, cli.covariance_npz, device, dtype)

    data_np = np.asarray(data_stack[cli.data_index], dtype=np.float64)

    print(f"project root: {PROJECT_ROOT}", flush=True)
    print(f"observed data: {cli.data_input}", flush=True)
    print(f"data shape: {data_stack.shape}", flush=True)
    print(f"full statistic dimension: {data_flat_t.numel()}", flush=True)
    print(f"active statistic dimension: {mu_np.size}", flush=True)
    print(f"covariance: {cli.covariance_npz}", flush=True)
    print(f"data index: {cli.data_index}", flush=True)
    print("likelihood: scattering model on n = d - s", flush=True)
    print(f"observed data std: {float(data_np.std())}", flush=True)

    position_space = ift.RGSpace(tuple(data_stack.shape[-2:]))
    scattering_energy_x = ScatteringGaussianEnergy(
        position_space, st_op, args, mu_np, precision_np, active_mask_np,
        device=device, dtype=dtype, metric_strength=cli.metric_strength,
    )

    if cli.use_correlated_field_prior:
        harmonic_partner = position_space.get_default_codomain()
        cfm = ift.CorrelatedFieldMaker("cf")
        cfm.set_amplitude_total_offset(0.0, (1.0, 0.5))
        cfm.add_fluctuations(
            position_space,
            fluctuations=(cli.prior_fluctuations_mean, cli.prior_fluctuations_std),
            flexibility=(1.0, 0.5),
            asperity=None,
            loglogavgslope=(cli.prior_loglog_slope_mean, cli.prior_loglog_slope_std),
            harmonic_partner=harmonic_partner,
        )
        signal_op = cfm.finalize(prior_info=0)
        print(
            "using correlated-field prior with log-log slope "
            f"{cli.prior_loglog_slope_mean} +/- {cli.prior_loglog_slope_std}",
            flush=True,
        )
    else:
        signal_op = ift.FieldAdapter(position_space, "x")
        print("using an identity signal model with NIFTy's white Gaussian prior", flush=True)

    latent_start = None
    fitted_initial_signal = None
    initial_fit_relative_error = None
    initial_signal_np = None
    if cli.initial_signal is not None:
        initial_stack = np.load(cli.initial_signal)
        if initial_stack.ndim == 3:
            if initial_stack.shape[0] != 1:
                raise ValueError(
                    "--initial-signal must be a 2D map or a stack containing exactly one map."
                )
            initial_stack = initial_stack[0]
        initial_signal_np = np.asarray(initial_stack, dtype=np.float64)
        latent_start, fitted_initial_signal, initial_fit_relative_error = fit_initial_signal(
            signal_op,
            initial_signal_np,
            iterations=cli.initialization_iterations,
            relative_noise=cli.initialization_relative_noise,
            seed=cli.seed,
        )
        print("using latent fit to", cli.initial_signal, "as inference initialization", flush=True)

    combined_energy_x = ScatteringGaussianEnergy(
        position_space,
        st_op,
        args,
        mu_np,
        precision_np,
        active_mask_np,
        device=device,
        dtype=dtype,
        metric_strength=cli.metric_strength,
        observed_data=data_np,
    )
    effective_likelihood = combined_energy_x @ signal_op

    print("inference domain:", effective_likelihood.domain, flush=True)
    print("signal target:", signal_op.target, flush=True)

    def scattering_energy(image: np.ndarray) -> float:
        return float(scattering_energy_x(ift.makeField(position_space, image)).val)

    def noise_likelihood_energy(image: np.ndarray) -> float:
        return float(combined_energy_x(ift.makeField(position_space, image)).val)

    print("zero signal noise-likelihood energy:", noise_likelihood_energy(np.zeros_like(data_np)), flush=True)

    map_position = None
    map_signal = None
    if cli.run_map:
        H = ift.StandardHamiltonian(effective_likelihood)
        start = latent_start
        if start is None:
            start = initial_position(effective_likelihood.domain, seed=cli.seed)
        energy = ift.EnergyAdapter(start, H, want_metric=True, nanisinf=True)
        controller = ift.GradientNormController(
            tol_abs_gradnorm=1e-5,
            iteration_limit=cli.map_iterations,
        )
        minimizer = ift.NewtonCG(controller, enable_logging=not cli.quiet)
        solution_energy, status = minimizer(energy)
        map_position = solution_energy.position
        map_signal = signal_op(map_position)
        print("MAP status:", status, flush=True)
        print("MAP energy:", float(solution_energy.value), flush=True)
        print("MAP gradient norm:", float(solution_energy.gradient.norm()), flush=True)
        print("MAP noise-likelihood energy:", noise_likelihood_energy(map_signal.val), flush=True)

    posterior_samples = None
    posterior_mean = None
    posterior_std = None
    if cli.run_vi:
        if "_minisanity" in ift.optimize_kl.__globals__:
            ift.optimize_kl.__globals__["_minisanity"] = lambda *args, **kwargs: None
            print("disabled NIFTy internal minisanity for custom scattering likelihood", flush=True)
        kl_controller = ift.GradientNormController(tol_abs_gradnorm=1e-5, iteration_limit=cli.kl_iterations)
        kl_minimizer = ift.NewtonCG(kl_controller, enable_logging=not cli.quiet)
        sampling_ic = ift.GradInfNormController(1e-3, iteration_limit=cli.sampling_iterations)
        if cli.use_geovi:
            nl_controller = ift.GradientNormController(tol_abs_gradnorm=1e-3, iteration_limit=cli.geovi_iterations)
            nonlinear_sampling_minimizer = ift.NewtonCG(nl_controller, enable_logging=False)
        else:
            nonlinear_sampling_minimizer = None

        vi_initial_position = None
        if cli.vi_start_from_map:
            vi_initial_position = map_position if map_position is not None else latent_start

        posterior_samples, _mean_latent = ift.optimize_kl(
            likelihood_energy=effective_likelihood,
            total_iterations=cli.vi_iterations,
            n_samples=cli.n_samples,
            kl_minimizer=kl_minimizer,
            sampling_iteration_controller=sampling_ic,
            nonlinear_sampling_minimizer=nonlinear_sampling_minimizer,
            plot_energy_history=False,
            plot_minisanity_history=False,
            initial_position=vi_initial_position,
            return_final_position=True,
            sanity_checks=False,
        )

        posterior_mean, posterior_var = posterior_samples.sample_stat(op=signal_op)
        posterior_std = ift.makeField(posterior_var.domain, np.sqrt(np.maximum(posterior_var.val, 0.0)))
        print("VI complete", flush=True)
        print("posterior mean noise-likelihood energy:", noise_likelihood_energy(posterior_mean.val), flush=True)

    signal_samples = []
    if posterior_samples is not None:
        for i, xi in enumerate(posterior_samples.iterator()):
            if i >= cli.n_samples:
                break
            signal_samples.append(np.asarray(signal_op(xi).val))
        signal_samples = np.stack(signal_samples, axis=0) if signal_samples else np.empty((0,) + position_space.shape)
    else:
        signal_samples = np.empty((0,) + position_space.shape)

    posterior_mean_np = None if posterior_mean is None else np.asarray(posterior_mean.val)
    posterior_std_np = None if posterior_std is None else np.asarray(posterior_std.val)
    map_signal_np = None if map_signal is None else np.asarray(map_signal.val)

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cli.out,
        observed_data=data_np,
        posterior_samples=signal_samples,
        posterior_mean=posterior_mean_np,
        posterior_std=posterior_std_np,
        map_signal=map_signal_np,
        mu=mu_np,
        covariance=covariance_np,
        active_statistic_mask=active_mask_np,
        initial_signal=initial_signal_np,
        fitted_initial_signal=None if fitted_initial_signal is None else np.asarray(fitted_initial_signal.val),
    )
    print("saved", cli.out, flush=True)

    if cli.plot is not None:
        make_plot(
            cli.plot,
            data_np,
            data_np,
            map_signal,
            posterior_mean,
            posterior_std,
            signal_samples,
        )
        print("saved", cli.plot, flush=True)

    if cli.diagnostics_summary is not None:
        summary = {
            "data_input": str(cli.data_input),
            "covariance_npz": str(cli.covariance_npz),
            "covariance_json": str(cli.covariance_json),
            "preprocessing": input_info,
            "data_index": cli.data_index,
            "initial_signal": None if cli.initial_signal is None else str(cli.initial_signal),
            "initial_fit_relative_l2": initial_fit_relative_error,
            "observed_data_std": float(data_np.std()),
            "active_statistic_dimension": int(mu_np.size),
            "n_posterior_samples": int(signal_samples.shape[0]),
        }
        write_summary(cli.diagnostics_summary, summary)
        print("saved", cli.diagnostics_summary, flush=True)


if __name__ == "__main__":
    main()
