#!/usr/bin/env python3
"""Direct synthesis check: match the input-map mean scattering statistics."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.optim import LBFGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import STL_main.torch_backend as bk
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize one map from input-map mean statistics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_syn")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--subtract-mean", type=parse_bool, default=False)
    parser.add_argument("--take-log", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--whiten", type=parse_bool, nargs="?", const=True, default=False)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=70001)

    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--wtype", default="Bump-Steerable")
    parser.add_argument("--iso", type=parse_bool, default=True)
    parser.add_argument("--angular-ft", type=parse_bool, default=True)
    parser.add_argument("--harmonics-angle", type=int, default=2)
    parser.add_argument("--scale-ft", type=parse_bool, default=True)
    parser.add_argument("--harmonics-scale", type=int, default=3)
    parser.add_argument("--dj", type=int, default=3)
    parser.add_argument("--compute-ps", type=parse_bool, default=False)
    parser.add_argument("--fewer-convolutions", type=parse_bool, default=False)
    parser.add_argument("--pbc", type=parse_bool, default=True)

    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--print-iter", type=int, default=10)
    parser.add_argument("--mask-small-variance", type=parse_bool, default=False)
    parser.add_argument("--variance-relative-tol", type=float, default=1e-10)
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
        raise ValueError(f"Expected input shape (S,H,W), got {target.shape}")
    if target.shape[-2:] != (args.target_size, args.target_size):
        raise ValueError(f"Expected maps of size {args.target_size}x{args.target_size}, got {target.shape}")
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
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return st_op.apply(data, **kwargs)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return st_op.apply(data, **supported)


def target_statistics(target: torch.Tensor, args: argparse.Namespace) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    array = target[:, None, :, :] if target.ndim == 3 else target[None, None, :, :]
    data = DataClass(array, pbc=args.pbc)
    st_op = build_st_operator(data, args)
    standardized, mean_pre_std, std_pre_std = st_op.wavelet_op.standardize(
        data, mean_field=True, inplace=False
    )
    stats_chunk_size = 32
    flats: list[torch.Tensor] = []
    per_map_flats: list[torch.Tensor] = []
    first_stats = None
    with torch.no_grad():
        for start in range(0, standardized.array.shape[0], stats_chunk_size):
            chunk = standardized.array[start : start + stats_chunk_size]
            chunk_data = DataClass(chunk, pbc=args.pbc)
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
                first_stats.mean_pre_std = mean_pre_std
                first_stats.std_pre_std = std_pre_std
            flat = stats.to_flatten(
                keep_batch_dim=True,
                mean_along_batch=True,
                keepnans=False,
                flatten_complex=True,
            ).real[0]
            flats.append(flat.detach())
            if args.mask_small_variance:
                per_map_flat = stats.to_flatten(
                    keep_batch_dim=True,
                    mean_along_batch=False,
                    keepnans=False,
                    flatten_complex=True,
                ).real
                per_map_flats.append(per_map_flat.detach())
    if first_stats is None:
        raise ValueError("No statistics computed")
    weights = torch.tensor(
        [min(stats_chunk_size, array.shape[0] - start) for start in range(0, array.shape[0], stats_chunk_size)],
        device=array.device,
        dtype=array.real.dtype,
    )
    target_flat = (torch.stack(flats, dim=0) * weights[:, None]).sum(dim=0) / weights.sum()
    active_mask = None
    if args.mask_small_variance:
        all_stats = torch.cat(per_map_flats, dim=0)
        variance = all_stats.var(dim=0, unbiased=True)
        threshold = args.variance_relative_tol * variance.max()
        active_mask = variance > threshold
    return first_stats, target_flat, active_mask


def make_running_operator(size: int, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Any:
    data = DataClass(torch.zeros((1, 1, size, size), device=device, dtype=dtype), pbc=args.pbc)
    return build_st_operator(data, args, replace_nan_value=None)


def synthesize_from_flat_statistics(
    target_flat: torch.Tensor,
    target_stats: Any,
    st_op_running: Any,
    args: argparse.Namespace,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    torch.manual_seed(args.seed)
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    u = torch.randn((1, target_stats.Nc, args.target_size, args.target_size), device=device, dtype=dtype)
    u.requires_grad_()
    target_flat = target_flat.to(device=device, dtype=dtype).detach()

    if target_stats.S2_ref_sqrt_chan_diag is not None:
        st_op_running.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = target_stats.var_ref
    if st_op_running.compute_PS:
        st_op_running.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    optimizer = LBFGS([u], lr=args.lr, max_iter=args.max_iter, history_size=args.history_size, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        stats_u = apply_st_operator(
            st_op_running,
            target_stats.DataClass(u, pbc=args.pbc),
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
        if active_mask is not None:
            flat_u = flat_u[active_mask]
        loss = ((flat_u - target_flat) ** 2).sum()
        loss.backward()
        if closure.calls == 0 or closure.calls % args.print_iter == 0:
            print(f"[LBFGS] iter {closure.calls + 1}, loss={loss.item():.6e}")
        closure.calls += 1
        return loss

    closure.calls = 0
    optimizer.step(closure)
    u_opt = u.detach()
    if target_stats.standardized:
        data_u_opt = target_stats.DataClass(u_opt, pbc=args.pbc)
        st_op_running.wavelet_op.unstandardize(
            data_u_opt,
            mean=target_stats.mean_pre_std.mean(dim=0),
            std=target_stats.std_pre_std.mean(dim=0),
            inplace=True,
        )
        u_opt = data_u_opt.array
    return u_opt[0, 0]


def save_plot(path: Path, reference: np.ndarray, synthesis: np.ndarray) -> None:
    vmin = float(min(np.percentile(reference, 1), np.percentile(synthesis, 1)))
    vmax = float(max(np.percentile(reference, 99), np.percentile(synthesis, 99)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    im = axes[0].imshow(reference, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(synthesis, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("synthesis")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.input.stem}_syn"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_np, input_info = load_target(args)
    target = torch.as_tensor(target_np, device=device, dtype=dtype)
    target_stats, target_flat, active_mask = target_statistics(target, args)
    full_dimension = int(target_flat.numel())
    if active_mask is not None:
        print(f"statistic dimension d = {full_dimension}", flush=True)
        print(f"active statistic dimension d_active = {int(active_mask.sum().item())}", flush=True)
        target_flat = target_flat[active_mask]
    else:
        print(f"statistic dimension d = {full_dimension}", flush=True)
    running_op = make_running_operator(args.target_size, args, device=device, dtype=dtype)

    image = synthesize_from_flat_statistics(
        target_flat=target_flat,
        target_stats=target_stats,
        st_op_running=running_op,
        args=args,
        active_mask=active_mask,
    )
    synthesis = image.detach().cpu().numpy()
    reference = target_np[0]

    plot_path = args.output_dir / f"{run_name}.png"
    npz_path = args.output_dir / f"{run_name}.npz"
    json_path = args.output_dir / f"{run_name}.json"
    save_plot(plot_path, reference, synthesis)
    np.savez_compressed(
        npz_path,
        original=reference,
        synthesis=synthesis,
        mean_statistic=target_flat.detach().cpu().numpy(),
        active_statistic_mask=np.array([]) if active_mask is None else active_mask.detach().cpu().numpy(),
    )
    json_path.write_text(
        json.dumps(
            {
                "input": input_info,
                "plot": plot_path.name,
                "npz": npz_path.name,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {plot_path}")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
