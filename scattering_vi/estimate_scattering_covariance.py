#!/usr/bin/env python3
"""Estimate scattering-statistic covariance from an input map stack.

The target can be a single map (H,W), a singleton stack (1,H,W), or a stack
(S,H,W). All available maps are used to define the mean scattering target.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import LBFGS


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import STL_main.torch_backend as bk
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass
from STL_main.Synthesis import apply_nyquist_filter


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_lower = value.lower()
    if value_lower in {"1", "true", "yes", "y"}:
        return True
    if value_lower in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate scattering covariance from input-map syntheses."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--subtract-mean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--take-log", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--whiten", type=parse_bool, nargs="?", const=True, default=False)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--wtype", default="Bump-Steerable")
    parser.add_argument("--iso", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--angular-ft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--harmonics-angle", type=int, default=2)
    parser.add_argument("--scale-ft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--harmonics-scale", type=int, default=3)
    parser.add_argument("--dj", type=int, default=3)
    parser.add_argument("--compute-ps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fewer-convolutions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pbc", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--sample-multiplier", type=int, default=20)
    parser.add_argument("--synthesis-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--print-iter", type=int, default=10)
    parser.add_argument("--apply-nyquist-filter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stats-chunk-size", type=int, default=32)

    parser.add_argument("--covariance-mode", choices=("diagonal", "full", "both"), default="both")
    parser.add_argument("--max-full-cov-dim", type=int, default=4096)
    parser.add_argument("--variance-relative-tol", type=float, default=1e-10)
    parser.add_argument("--save-syntheses", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-statistics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compute-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also estimate the additive-noise statistic bias "
            "<phi(reference + noise)> - phi(reference)."
        ),
    )
    parser.add_argument(
        "--bias-reference-map",
        type=Path,
        default=None,
        help="Reference signal map used for the optional statistic-bias estimate.",
    )
    parser.add_argument(
        "--noise-input",
        type=Path,
        default=None,
        help="Noise-map stack (N,H,W) used when --compute-bias is enabled.",
    )
    parser.add_argument(
        "--bias-n-noise",
        type=int,
        default=0,
        help="Number of noise maps used for the bias; 0 uses the complete stack.",
    )
    parser.add_argument(
        "--save-bias-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save every phi(reference + noise) vector in addition to their mean bias.",
    )
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


def load_target(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.load(args.input).astype(np.float64, copy=False)
    if target.ndim == 2:
        target = target[None, :, :]
    if target.ndim != 3:
        raise ValueError(f"Expected target stack with shape (S,H,W), got {target.shape}")
    if target.shape[-2:] != (args.target_size, args.target_size):
        raise ValueError(
            f"Expected maps of size {args.target_size}x{args.target_size}, got {target.shape}"
        )
    if args.take_log:
        if np.any(target <= 0):
            raise ValueError("--take-log requires strictly positive input values")
        target = np.log(target)
    if args.subtract_mean:
        target = target - target.mean(axis=(-2, -1), keepdims=True)
    if args.whiten:
        mean = target.mean(axis=(-2, -1), keepdims=True)
        std = target.std(axis=(-2, -1), keepdims=True)
        if np.any(std <= 0):
            raise ValueError("--whiten encountered a zero-variance map")
        target = (target - mean) / std
    return target, {
        "input_path": str(args.input),
        "target_shape": list(target.shape),
        "n_target_maps": int(target.shape[0]),
        "take_log": args.take_log,
        "subtract_mean": args.subtract_mean,
        "whiten": args.whiten,
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
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
    stats_chunk_size = getattr(args, "stats_chunk_size", 32)
    if target.ndim == 2:
        array = target[None, None, :, :]
    elif target.ndim == 3:
        array = target[:, None, :, :]
    else:
        raise ValueError(f"Expected target with ndim 2 or 3, got {target.shape}")
    data = DataClass(array, pbc=args.pbc)
    st_op = build_st_operator(data, args)
    if array.shape[0] > stats_chunk_size:
        global_mean = array.mean(dim=(-2, -1), keepdim=False).mean(dim=0, keepdim=True)
        centered = array - global_mean[..., None, None]
        global_std = torch.sqrt((centered.abs() ** 2).mean(dim=(-2, -1)).mean(dim=0, keepdim=True))
        flats: list[torch.Tensor] = []
        first_stats = None
        with torch.no_grad():
            for start in range(0, array.shape[0], stats_chunk_size):
                chunk = array[start : start + stats_chunk_size]
                chunk_data = DataClass(chunk, pbc=args.pbc)
                chunk_data.array = (chunk_data.array - global_mean[..., None, None]) / global_std[..., None, None]
                norm = "store_ref" if first_stats is None else "load_ref"
                if first_stats is not None:
                    st_op.S2_ref_sqrt_chan_diag = first_stats.S2_ref_sqrt_chan_diag
                    st_op.var_ref = first_stats.var_ref
                    if st_op.compute_PS:
                        st_op.PS_ref_sqrt_chan_diag = first_stats.PS_ref_sqrt_chan_diag
                stats = apply_st_operator(
                    st_op,
                    chunk_data,
                    norm=norm,
                    norm_batch_mean=True,
                    compute_PS=args.compute_ps,
                )
                if first_stats is None:
                    first_stats = stats
                    first_stats.standardized = True
                    first_stats.mean_pre_std = global_mean
                    first_stats.std_pre_std = global_std
                flat = stats.to_flatten(
                    keep_batch_dim=True,
                    mean_along_batch=True,
                    keepnans=False,
                    flatten_complex=True,
                ).real[0]
                flats.append(flat.detach())
        if first_stats is None:
            raise ValueError("No target statistics were computed")
        weights = torch.tensor(
            [
                min(stats_chunk_size, array.shape[0] - start)
                for start in range(0, array.shape[0], stats_chunk_size)
            ],
            device=array.device,
            dtype=array.real.dtype,
        )
        return first_stats, st_op, (torch.stack(flats, dim=0) * weights[:, None]).sum(dim=0) / weights.sum()
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


def make_running_operator(size: int, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Any:
    example = torch.zeros((1, 1, size, size), device=device, dtype=dtype)
    data = DataClass(example, pbc=args.pbc)
    return build_st_operator(data, args, replace_nan_value=None)


def synthesize_from_flat_statistics(
    *,
    target_flat: torch.Tensor,
    target_stats: Any,
    st_op_running: Any,
    running_shape: tuple[int, int],
    pbc_running: bool,
    lr: float,
    max_iter: int,
    history_size: int,
    print_iter: int,
    verbose: bool,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    n_channels = target_stats.Nc
    u = torch.randn((1, n_channels, *running_shape), device=device, dtype=dtype)
    u.requires_grad_()
    target_flat = target_flat.to(device=device, dtype=dtype).detach()

    if target_stats.S2_ref_sqrt_chan_diag is not None:
        st_op_running.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = target_stats.var_ref
    if st_op_running.compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    optimizer = LBFGS([u], lr=lr, max_iter=max_iter, history_size=history_size, line_search_fn="strong_wolfe")
    loss_history: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        data_u = target_stats.DataClass(u, pbc=pbc_running)
        stats_u = apply_st_operator(
            st_op_running,
            data_u,
            has_fewer_convolutions=target_stats.has_fewer_convolutions,
            compute_cross_matrix=target_stats.compute_cross_matrix,
            compute_PS=st_op_running.compute_PS,
            norm="load_ref",
        )
        flat_u = stats_u.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=True,
            keepnans=False,
            flatten_complex=True,
        ).real[0]
        loss = ((flat_u - target_flat) ** 2).sum()
        loss.backward()
        loss_history.append(float(loss.detach().cpu()))
        if verbose and (len(loss_history) == 1 or len(loss_history) % print_iter == 0):
            print(f"[sampled-stat LBFGS] iter {len(loss_history)}, loss={loss.item():.6e}")
        return loss

    optimizer.step(closure)
    print(f"Sampled-stat synthesis final loss: {loss_history[-1]:.6e}")
    u_opt = u.detach()
    if target_stats.standardized:
        data_u_opt = target_stats.DataClass(u_opt, pbc=pbc_running)
        st_op_running.wavelet_op.unstandardize(
            data_u_opt,
            mean=target_stats.mean_pre_std.mean(dim=0),
            std=target_stats.std_pre_std.mean(dim=0),
            inplace=True,
        )
        u_opt = data_u_opt.array
    if n_channels == 1:
        u_opt = u_opt[:, 0, ...]
    return u_opt[0]


def call_optimize_from_stats(
    *,
    target_stats: Any,
    st_op_running: Any,
    batch_size: int,
    running_shape: tuple[int, int],
    pbc_running: bool,
    init_running: Any,
    mean_field: bool,
    lr: float,
    max_iter: int,
    history_size: int,
    print_iter: int,
    verbose: bool,
    seed: int,
    target_flat_override: torch.Tensor | None = None,
) -> torch.Tensor:
    torch.manual_seed(seed)
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    n_channels = target_stats.Nc
    u = torch.randn((batch_size, n_channels, *running_shape), device=device, dtype=dtype)
    if init_running is not None:
        u = torch.as_tensor(init_running, device=device, dtype=dtype).expand_as(u).clone()
    u.requires_grad_()

    if target_flat_override is None:
        target_flat = target_stats.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=mean_field,
            keepnans=True,
        ).detach()
        target_flat = target_flat[~target_flat.isnan()]
    else:
        target_flat = target_flat_override.detach()

    if target_stats.S2_ref_sqrt_chan_diag is not None:
        st_op_running.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = target_stats.var_ref
    if st_op_running.compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    optimizer = LBFGS([u], lr=lr, max_iter=max_iter, history_size=history_size, line_search_fn="strong_wolfe")
    loss_history: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        data_u = target_stats.DataClass(u, pbc=pbc_running)
        stats_u = apply_st_operator(
            st_op_running,
            data_u,
            has_fewer_convolutions=target_stats.has_fewer_convolutions,
            compute_cross_matrix=target_stats.compute_cross_matrix,
            compute_PS=st_op_running.compute_PS,
            norm="load_ref",
        )
        flat_u = stats_u.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=mean_field,
            keepnans=False,
            flatten_complex=True,
        )
        loss = ((flat_u - target_flat).abs() ** 2).sum()
        loss.backward()
        loss_history.append(float(loss.detach().cpu()))
        if verbose and (len(loss_history) == 1 or len(loss_history) % print_iter == 0):
            print(f"[direct LBFGS] iter {len(loss_history)}, loss={loss.item():.6e}")
        return loss

    optimizer.step(closure)
    print(f"Direct synthesis final loss: {loss_history[-1]:.6e}")

    u_opt = u.detach()
    if target_stats.standardized:
        data_u_opt = target_stats.DataClass(u_opt, pbc=pbc_running)
        st_op_running.wavelet_op.unstandardize(
            data_u_opt,
            mean=target_stats.mean_pre_std.mean(dim=0),
            std=target_stats.std_pre_std.mean(dim=0),
            inplace=True,
        )
        u_opt = data_u_opt.array
    if n_channels == 1:
        u_opt = u_opt[:, 0, ...]
    return u_opt[0] if batch_size == 1 else u_opt


def tile_maps(maps: torch.Tensor, tile_size: int) -> torch.Tensor:
    if maps.ndim == 2:
        maps = maps[None, :, :]
    if maps.shape[-2:] == (tile_size, tile_size):
        return maps
    tiles = maps.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    return tiles.contiguous().reshape(-1, tile_size, tile_size)


def scattering_vectors(maps: torch.Tensor, st_op: Any, args: argparse.Namespace) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, maps.shape[0], args.stats_chunk_size):
            chunk = maps[start : start + args.stats_chunk_size, None, :, :]
            data = DataClass(chunk, pbc=args.pbc)
            stats = apply_standardized_st(
                st_op,
                data,
                mean_field=False,
                mark_standardized=False,
                norm="load_ref",
                compute_PS=args.compute_ps,
            )
            flat = stats.to_flatten(
                keep_batch_dim=True,
                mean_along_batch=False,
                keepnans=False,
                flatten_complex=True,
            ).real
            vectors.append(flat.detach().cpu())
    return torch.cat(vectors, dim=0)


def preprocess_map_stack(
    maps: np.ndarray, args: argparse.Namespace, *, label: str
) -> np.ndarray:
    """Apply the same map-level preprocessing used for the covariance input."""
    array = np.asarray(maps, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected {label} shape (H,W) or (N,H,W), got {array.shape}")
    expected = (args.target_size, args.target_size)
    if array.shape[-2:] != expected:
        raise ValueError(f"Expected {label} maps of size {expected}, got {array.shape[-2:]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    if args.take_log:
        if np.any(array <= 0):
            raise ValueError(f"--take-log requires strictly positive {label} values")
        array = np.log(array)
    if args.subtract_mean:
        array = array - array.mean(axis=(-2, -1), keepdims=True)
    if args.whiten:
        mean = array.mean(axis=(-2, -1), keepdims=True)
        std = array.std(axis=(-2, -1), keepdims=True)
        if np.any(std <= 0):
            raise ValueError(f"--whiten encountered a zero-variance {label} map")
        array = (array - mean) / std
    return array


def compute_statistic_bias(
    args: argparse.Namespace,
    st_op: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Estimate <phi(reference + noise)> - phi(reference) using noise draws."""
    if args.bias_reference_map is None or args.noise_input is None:
        raise ValueError(
            "--compute-bias requires both --bias-reference-map and --noise-input"
        )
    if not args.bias_reference_map.is_file():
        raise FileNotFoundError(f"Missing bias reference map: {args.bias_reference_map}")
    if not args.noise_input.is_file():
        raise FileNotFoundError(f"Missing noise stack: {args.noise_input}")
    if args.bias_n_noise < 0:
        raise ValueError("--bias-n-noise must be non-negative")

    reference_raw = np.load(args.bias_reference_map).astype(np.float64, copy=False)
    if reference_raw.ndim == 3 and reference_raw.shape[0] == 1:
        reference_raw = reference_raw[0]
    if reference_raw.ndim != 2:
        raise ValueError(
            f"Expected one reference map with shape (H,W), got {reference_raw.shape}"
        )
    expected = (args.target_size, args.target_size)
    if reference_raw.shape != expected:
        raise ValueError(f"Expected reference map size {expected}, got {reference_raw.shape}")
    if not np.isfinite(reference_raw).all():
        raise ValueError("Bias reference map contains NaN or infinite values")

    noise_stack = np.load(args.noise_input, mmap_mode="r")
    if noise_stack.ndim == 2:
        noise_stack = noise_stack[None, :, :]
    if noise_stack.ndim != 3 or noise_stack.shape[-2:] != expected:
        raise ValueError(
            f"Expected noise stack shape (N,{expected[0]},{expected[1]}), got {noise_stack.shape}"
        )
    n_available = int(noise_stack.shape[0])
    n_used = n_available if args.bias_n_noise == 0 else min(args.bias_n_noise, n_available)
    if n_used < 1:
        raise ValueError("The noise stack contains no usable maps")
    rng = np.random.default_rng(args.seed)
    if n_used == n_available:
        indices = np.arange(n_available, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(n_available, size=n_used, replace=False)).astype(np.int64)

    reference_processed = preprocess_map_stack(
        reference_raw, args, label="bias reference"
    )
    reference_tensor = torch.as_tensor(reference_processed, device=device, dtype=dtype)
    reference_statistics = (
        scattering_vectors(reference_tensor, st_op, args)
        .numpy()
        .astype(np.float64, copy=False)[0]
    )

    mixed_groups: list[np.ndarray] = []
    chunk_size = max(1, int(args.stats_chunk_size))
    print(
        f"Computing statistic bias with {n_used}/{n_available} noise maps "
        f"and reference {args.bias_reference_map}"
    )
    for start in range(0, n_used, chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        noise_chunk = np.asarray(noise_stack[chunk_indices], dtype=np.float64)
        if not np.isfinite(noise_chunk).all():
            raise ValueError("Noise stack contains NaN or infinite values")
        mixed_raw = reference_raw[None, :, :] + noise_chunk
        mixed_processed = preprocess_map_stack(
            mixed_raw, args, label="reference-plus-noise"
        )
        mixed_tensor = torch.as_tensor(mixed_processed, device=device, dtype=dtype)
        mixed_groups.append(
            scattering_vectors(mixed_tensor, st_op, args)
            .numpy()
            .astype(np.float64, copy=False)
        )
        print(f"Bias statistics: {min(start + chunk_size, n_used)}/{n_used}", flush=True)

    mixed_statistics = np.concatenate(mixed_groups, axis=0)
    mixed_mean = mixed_statistics.mean(axis=0)
    bias = mixed_mean - reference_statistics
    bias_samples = mixed_statistics - reference_statistics[None, :]
    arrays: dict[str, np.ndarray] = {
        "bias": bias,
        "bias_reference_statistics": reference_statistics,
        "bias_mixed_mean": mixed_mean,
        "bias_noise_indices": indices,
        "bias_variance": bias_samples.var(axis=0, ddof=1) if n_used > 1 else np.zeros_like(bias),
    }
    if args.save_bias_statistics:
        arrays["bias_mixed_statistic_samples"] = mixed_statistics
        arrays["bias_statistic_samples"] = bias_samples
    info = {
        "computed": True,
        "definition": "mean_noise(phi(reference + noise)) - phi(reference)",
        "reference_map": str(args.bias_reference_map),
        "noise_input": str(args.noise_input),
        "n_noise_available": n_available,
        "n_noise_used": n_used,
        "noise_selection_seed": int(args.seed),
        "saved_bias_statistics": bool(args.save_bias_statistics),
        "bias_l2_norm": float(np.linalg.norm(bias)),
    }
    return arrays, info


def diagonal_variance(centered: np.ndarray, denominator: int) -> np.ndarray:
    return np.einsum("nd,nd->d", centered, centered) / denominator


def compute_covariance_products(groups: list[np.ndarray], mode: str, max_full_cov_dim: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_stats = np.concatenate(groups, axis=0).astype(np.float64, copy=False)
    n_samples, dimension = all_stats.shape
    pooled_mean = all_stats.mean(axis=0)
    pooled_centered = all_stats - pooled_mean
    pooled_variance = diagonal_variance(pooled_centered, n_samples - 1)

    within_ss_diag = np.zeros(dimension, dtype=np.float64)
    within_denominator = 0
    for group in groups:
        if group.shape[0] < 2:
            continue
        centered = group - group.mean(axis=0)
        within_ss_diag += np.einsum("nd,nd->d", centered, centered)
        within_denominator += group.shape[0] - 1
    within_variance = within_ss_diag / within_denominator

    arrays: dict[str, np.ndarray] = {
        "synthesized_mean": pooled_mean,
        "variance_within_batch": within_variance,
        "variance_pooled": pooled_variance,
    }
    full_requested = mode in {"full", "both"}
    full_computed = full_requested and dimension <= max_full_cov_dim
    if full_computed:
        arrays["covariance_pooled"] = pooled_centered.T @ pooled_centered / (n_samples - 1)
        within_ss = np.zeros((dimension, dimension), dtype=np.float64)
        for group in groups:
            if group.shape[0] < 2:
                continue
            centered = group - group.mean(axis=0)
            within_ss += centered.T @ centered
        arrays["covariance_within_batch"] = within_ss / within_denominator

    metadata = {
        "n_statistic_samples": int(n_samples),
        "statistic_dimension": int(dimension),
        "n_groups": len(groups),
        "group_sizes": [int(group.shape[0]) for group in groups],
        "within_covariance_degrees_of_freedom": int(within_denominator),
        "pooled_covariance_rank_upper_bound": int(min(dimension, n_samples - 1)),
        "within_covariance_rank_upper_bound": int(min(dimension, within_denominator)),
        "nominal_relative_std_error_variance": math.sqrt(2.0 / within_denominator),
        "full_covariance_requested": full_requested,
        "full_covariance_computed": full_computed,
        "max_full_cov_dim": int(max_full_cov_dim),
    }
    return arrays, metadata


def json_ready(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def save_checkpoint(path: Path, groups: list[np.ndarray], batch_ids: list[np.ndarray], target_flat: np.ndarray) -> None:
    np.savez_compressed(
        path,
        synthesized_statistics=np.concatenate(groups, axis=0),
        synthesis_batch_id=np.concatenate(batch_ids, axis=0),
        target_statistics=target_flat,
    )


def tensor_to_numpy_or_empty(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"{args.input.stem}_covariance"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_name}.npz"
    metadata_path = output_dir / f"{args.run_name}.json"
    checkpoint_path = output_dir / f"{args.run_name}_checkpoint.npz"

    target_np, input_info = load_target(args)
    target = torch.as_tensor(target_np, device=device, dtype=dtype)
    target_stats, target_op, target_flat_t = target_statistics(target, args)
    target_flat = target_flat_t.cpu().numpy().astype(np.float64, copy=False)
    dimension = int(target_flat.shape[0])
    n_total_samples = args.n_samples if args.n_samples > 0 else args.sample_multiplier * dimension

    print(f"Input: {input_info['input_path']}")
    print(f"Preprocessed target shape: {target_np.shape}")
    print(f"Real flattened statistic dimension: {dimension}")
    print(f"Requested covariance samples total: {n_total_samples}")

    groups: list[np.ndarray] = []
    batch_ids: list[np.ndarray] = []
    saved_map_files: list[str] = []
    batch_mean_errors: list[float] = []
    print("Computing statistics of input maps")
    vectors = scattering_vectors(target, target_op, args).numpy().astype(np.float64, copy=False)
    target_flat = vectors.mean(axis=0)
    groups.append(vectors)
    batch_ids.append(np.zeros(vectors.shape[0], dtype=np.int32))
    n_existing_samples = vectors.shape[0]
    n_synthesis_samples = max(0, n_total_samples - n_existing_samples)
    print(f"Input maps used for covariance: {n_existing_samples}")
    print(f"Additional synthesis samples needed: {n_synthesis_samples}")
    save_checkpoint(checkpoint_path, groups, batch_ids, target_flat)

    batch_index = 1
    if n_synthesis_samples > 0:
        target_mean_pre_std = target_stats.mean_pre_std.detach().clone()
        target_std_pre_std = target_stats.std_pre_std.detach().clone()
        running_op = make_running_operator(args.synthesis_size, args, device=device, dtype=dtype)
        print(f"Synthesis: {n_synthesis_samples} extra maps, batch_size<={args.batch_size}")
        n_completed = 0
        while n_completed < n_synthesis_samples:
            current_batch_size = min(args.batch_size, n_synthesis_samples - n_completed)
            batch_seed = args.seed + batch_index
            print(f"\n=== Synthesis batch {batch_index} ({n_completed + current_batch_size}/{n_synthesis_samples}, seed={batch_seed}) ===")
            target_stats.mean_pre_std = target_mean_pre_std.clone()
            target_stats.std_pre_std = target_std_pre_std.clone()
            synthesized = call_optimize_from_stats(
                target_stats=target_stats,
                st_op_running=running_op,
                batch_size=current_batch_size,
                running_shape=(args.synthesis_size, args.synthesis_size),
                pbc_running=args.pbc,
                init_running=None,
                mean_field=True,
                lr=args.lr,
                max_iter=args.max_iter,
                history_size=args.history_size,
                print_iter=args.print_iter,
                verbose=True,
                seed=batch_seed,
                target_flat_override=target_flat_t,
            )
            if synthesized.ndim == 2:
                synthesized = synthesized[None, :, :]
            if args.apply_nyquist_filter:
                synthesized = apply_nyquist_filter(synthesized)
            if args.save_syntheses:
                map_path = output_dir / f"{args.run_name}_synthesis_batch_{batch_index:03d}.npy"
                np.save(map_path, synthesized.detach().cpu().numpy())
                saved_map_files.append(map_path.name)

            vectors = scattering_vectors(tile_maps(synthesized, args.target_size), target_op, args).numpy().astype(np.float64, copy=False)
            group_mean = vectors.mean(axis=0)
            relative_error = float(np.linalg.norm(group_mean - target_flat) / max(float(np.linalg.norm(target_flat)), 1e-30))
            print(f"Batch mean relative statistic error: {relative_error:.6e}")
            batch_mean_errors.append(relative_error)
            groups.append(vectors)
            batch_ids.append(np.full(vectors.shape[0], batch_index, dtype=np.int32))
            n_completed += current_batch_size
            batch_index += 1
            save_checkpoint(checkpoint_path, groups, batch_ids, target_flat)

    covariance_arrays, covariance_info = compute_covariance_products(groups, args.covariance_mode, args.max_full_cov_dim)
    all_stats = np.concatenate(groups, axis=0)
    variance_threshold = args.variance_relative_tol * float(covariance_arrays["variance_pooled"].max())
    active = covariance_arrays["variance_pooled"] > variance_threshold
    bias_arrays: dict[str, np.ndarray] = {}
    bias_info: dict[str, Any] = {"computed": False}
    if args.compute_bias:
        bias_arrays, bias_info = compute_statistic_bias(
            args, target_op, device=device, dtype=dtype
        )

    payload: dict[str, np.ndarray] = {
        "target_example": target_np[: min(target_np.shape[0], 4)],
        "mean": target_flat,
        "covariance": covariance_arrays.get(
            "covariance_within_batch",
            covariance_arrays.get("covariance_pooled", np.diag(covariance_arrays["variance_pooled"])),
        ),
        "variance": covariance_arrays["variance_pooled"],
        "stl_reference_field_mean": tensor_to_numpy_or_empty(target_stats.mean_pre_std),
        "stl_reference_field_std": tensor_to_numpy_or_empty(target_stats.std_pre_std),
        "stl_reference_S2_ref_sqrt_chan_diag": tensor_to_numpy_or_empty(target_stats.S2_ref_sqrt_chan_diag),
        "stl_reference_var_ref": tensor_to_numpy_or_empty(target_stats.var_ref),
        "stl_reference_PS_ref_sqrt_chan_diag": tensor_to_numpy_or_empty(
            getattr(target_stats, "PS_ref_sqrt_chan_diag", None)
        ),
        "active_statistic_mask": active,
        "active_statistic_indices": np.flatnonzero(active),
        **covariance_arrays,
        **bias_arrays,
    }
    if args.save_statistics:
        payload["statistic_samples"] = all_stats
        payload["synthesis_batch_id"] = np.concatenate(batch_ids, axis=0)
    np.savez_compressed(result_path, **payload)

    metadata = {
        "config": json_ready(args),
        "input": input_info,
        "output_npz": str(result_path),
        "checkpoint_npz": str(checkpoint_path),
        "saved_synthesis_files": saved_map_files,
        "target_statistic_dimension": dimension,
        "n_requested_covariance_samples_total": int(n_total_samples),
        "n_input_covariance_samples": int(n_existing_samples),
        "n_requested_synthesis_samples": int(n_synthesis_samples),
        "n_synthesis_batches": int(batch_index),
        "batch_mean_relative_errors": batch_mean_errors,
        "mean_batch_mean_relative_error": (
            float(np.mean(batch_mean_errors)) if batch_mean_errors else None
        ),
        "max_batch_mean_relative_error": (
            float(np.max(batch_mean_errors)) if batch_mean_errors else None
        ),
        "n_active_statistics": int(active.sum()),
        "variance_threshold": float(variance_threshold),
        "bias": bias_info,
        **covariance_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {result_path}")
    print(f"Saved {metadata_path}")
    print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    main()
