#!/usr/bin/env python3
"""Refine an STL component-separation map with the exact CIB likelihood.

Starting from a recovered signal s0, minimize

    U_refine(s) = U_like(d - s) + 0.5 * lambda_anchor * ||s - s0||^2.

The anchor protects the visually successful component-separation solution.
The input map is read-only and is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from langevin_sample import (
    build_st_operator,
    configure_backend,
    likelihood_energy,
    load_metadata,
    load_model,
    load_observed,
    scattering_vector,
    torch_dtype,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COVARIANCE_STEM = "100_Herschel_Lockman_250m_tiles_3400x256x256_covariance"
DEFAULT_COMPSEP_DIR = SCRIPT_DIR / "results" / "component_separation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_COMPSEP_DIR / "observed_data.npy")
    parser.add_argument(
        "--initial-signal", type=Path, default=DEFAULT_COMPSEP_DIR / "recovered_signal.npy"
    )
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument(
        "--covariance-npz",
        type=Path,
        default=REPO_ROOT / "scattering_vi" / "results" / f"{COVARIANCE_STEM}.npz",
    )
    parser.add_argument(
        "--covariance-json",
        type=Path,
        default=REPO_ROOT / "scattering_vi" / "results" / f"{COVARIANCE_STEM}.json",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "refinement")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument(
        "--anchor-precision",
        type=float,
        default=0.1,
        help="lambda_anchor in 0.5*lambda_anchor*sum((s-s0)^2).",
    )
    parser.add_argument(
        "--zero-mean-signal", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--covariance-jitter-rel", type=float, default=1e-6)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--top-statistics", type=int, default=15)
    return parser.parse_args()


def load_initial(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    signal = np.load(path)
    if signal.ndim == 3 and signal.shape[0] == 1:
        signal = signal[0]
    if signal.shape != expected_shape:
        raise ValueError(f"Initial signal shape {signal.shape}; expected {expected_shape}")
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("Initial signal contains NaN or infinite values")
    return signal


def posterior_diagnostics(
    signal: torch.Tensor,
    initial: torch.Tensor,
    observed: torch.Tensor,
    st_op: Any,
    config: argparse.Namespace,
    mean: torch.Tensor,
    covariance_cholesky: torch.Tensor,
    active: torch.Tensor,
    anchor_precision: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    likelihood = likelihood_energy(
        observed - signal, st_op, config, mean, covariance_cholesky, active
    )
    correction = signal - initial
    anchor = 0.5 * anchor_precision * correction.square().sum()
    return likelihood + anchor, likelihood, anchor, correction


def whitened_residual(
    residual: torch.Tensor,
    st_op: Any,
    config: argparse.Namespace,
    mean: torch.Tensor,
    covariance_cholesky: torch.Tensor,
    active: torch.Tensor,
) -> np.ndarray:
    statistic = scattering_vector(residual, st_op, config)[active]
    difference = statistic - mean
    whitened = torch.linalg.solve_triangular(
        covariance_cholesky, difference[:, None], upper=False
    )[:, 0]
    return whitened.detach().cpu().numpy()


def image_panel(ax: Any, image: np.ndarray, title: str) -> None:
    center = float(np.median(image))
    limit = float(np.percentile(np.abs(image - center), 99))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    shown = ax.imshow(
        image,
        origin="lower",
        cmap="RdBu_r",
        vmin=center - limit,
        vmax=center + limit,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.data, "observed data"),
        (args.initial_signal, "component-separation signal"),
        (args.covariance_npz, "covariance NPZ"),
        (args.covariance_json, "covariance JSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.max_iter <= 0 or args.lr <= 0 or args.history_size <= 0:
        raise ValueError("Require max_iter, lr, and history_size to be positive")
    if args.anchor_precision < 0 or args.covariance_jitter_rel <= 0:
        raise ValueError("Require anchor_precision >= 0 and positive covariance jitter")
    if args.print_every <= 0 or args.top_statistics < 0:
        raise ValueError("Print frequency must be positive and top-statistics non-negative")

    _metadata, config = load_metadata(args.covariance_json)
    dtype_name = args.dtype or str(config.dtype)
    dtype = torch_dtype(dtype_name)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
        torch.cuda.set_device(device)
    configure_backend(device, dtype)

    observed_np = load_observed(args.data, config, args.data_index)
    initial_np = load_initial(args.initial_signal, tuple(observed_np.shape))
    if args.zero_mean_signal:
        initial_np = initial_np - initial_np.mean()
    observed = torch.as_tensor(observed_np, device=device, dtype=dtype)
    initial = torch.as_tensor(initial_np, device=device, dtype=dtype)
    signal = initial.clone().detach().requires_grad_(True)

    st_op = build_st_operator(int(config.target_size), config, device, dtype)
    mean, covariance_cholesky, active, _references, jitter = load_model(
        args.covariance_npz,
        st_op,
        device,
        dtype,
        args.covariance_jitter_rel,
    )
    expected_rho = math.sqrt(int(active.sum()))
    print(f"device={device} dtype={dtype_name}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    print(f"data={args.data}", flush=True)
    print(f"initial_signal={args.initial_signal} (read-only)", flush=True)
    print(
        f"active_dimension={int(active.sum())} expected_rho={expected_rho:.4f} "
        f"anchor_precision={args.anchor_precision:.6g} jitter={jitter:.6e}",
        flush=True,
    )

    optimizer = torch.optim.LBFGS(
        [signal],
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )
    total_history: list[float] = []
    likelihood_history: list[float] = []
    anchor_history: list[float] = []
    rho_history: list[float] = []
    correction_rms_history: list[float] = []
    gradient_rms_history: list[float] = []
    elapsed_history: list[float] = []
    start_time = time.monotonic()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        total, likelihood, anchor, correction = posterior_diagnostics(
            signal,
            initial,
            observed,
            st_op,
            config,
            mean,
            covariance_cholesky,
            active,
            args.anchor_precision,
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite refinement objective")
        total.backward()
        if signal.grad is None or not torch.isfinite(signal.grad).all():
            raise FloatingPointError("Non-finite refinement gradient")
        likelihood_value = float(likelihood.detach().cpu())
        total_history.append(float(total.detach().cpu()))
        likelihood_history.append(likelihood_value)
        anchor_history.append(float(anchor.detach().cpu()))
        rho_history.append(math.sqrt(max(2.0 * likelihood_value, 0.0)))
        correction_rms_history.append(float(torch.sqrt(torch.mean(correction.square())).detach().cpu()))
        gradient_rms_history.append(float(torch.sqrt(torch.mean(signal.grad.square())).detach().cpu()))
        elapsed_history.append(time.monotonic() - start_time)
        call = len(total_history)
        if call == 1 or call % args.print_every == 0:
            print(
                f"call={call:5d} total={total_history[-1]:.6e} "
                f"U_like={likelihood_value:.6e} rho={rho_history[-1]:.4f} "
                f"U_anchor={anchor_history[-1]:.6e} "
                f"delta_rms={correction_rms_history[-1]:.6e} "
                f"grad_rms={gradient_rms_history[-1]:.6e}",
                flush=True,
            )
        return total

    optimizer.step(closure)
    with torch.no_grad():
        if args.zero_mean_signal:
            signal -= signal.mean()
    final_total, final_like, final_anchor, final_correction = posterior_diagnostics(
        signal,
        initial,
        observed,
        st_op,
        config,
        mean,
        covariance_cholesky,
        active,
        args.anchor_precision,
    )
    initial_whitened = whitened_residual(
        observed - initial, st_op, config, mean, covariance_cholesky, active
    )
    final_whitened = whitened_residual(
        observed - signal, st_op, config, mean, covariance_cholesky, active
    )
    initial_rho = float(np.linalg.norm(initial_whitened))
    final_rho = float(np.linalg.norm(final_whitened))
    print(
        f"complete calls={len(total_history)} initial_rho={initial_rho:.4f} "
        f"final_rho={final_rho:.4f} final_delta_rms="
        f"{float(torch.sqrt(torch.mean(final_correction.square())).detach().cpu()):.6e}",
        flush=True,
    )

    refined_np = signal.detach().cpu().numpy()
    correction_np = refined_np - initial_np
    residual_before = observed_np - initial_np
    residual_after = observed_np - refined_np
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "signal_before": args.output_dir / "signal_before.npy",
        "signal_refined": args.output_dir / "signal_refined.npy",
        "correction": args.output_dir / "signal_correction.npy",
        "residual_before": args.output_dir / "residual_before.npy",
        "residual_after": args.output_dir / "residual_refined.npy",
        "diagnostics": args.output_dir / "refinement_diagnostics.npz",
        "maps_plot": args.output_dir / "refinement_maps.png",
        "diagnostics_plot": args.output_dir / "refinement_diagnostics.png",
        "summary": args.output_dir / "refinement_summary.json",
    }
    np.save(paths["signal_before"], initial_np)
    np.save(paths["signal_refined"], refined_np)
    np.save(paths["correction"], correction_np)
    np.save(paths["residual_before"], residual_before)
    np.save(paths["residual_after"], residual_after)
    np.savez_compressed(
        paths["diagnostics"],
        total_energy=np.asarray(total_history),
        likelihood_energy=np.asarray(likelihood_history),
        anchor_energy=np.asarray(anchor_history),
        rho=np.asarray(rho_history),
        correction_rms=np.asarray(correction_rms_history),
        gradient_rms=np.asarray(gradient_rms_history),
        elapsed_seconds=np.asarray(elapsed_history),
        whitened_residual_before=initial_whitened,
        whitened_residual_after=final_whitened,
        active_statistic_mask=active.detach().cpu().numpy(),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    image_panel(axes[0, 0], observed_np, "observed data d")
    image_panel(axes[0, 1], initial_np, "signal before refinement")
    image_panel(axes[0, 2], residual_before, f"residual before (rho={initial_rho:.2f})")
    image_panel(axes[1, 0], correction_np, "signal correction")
    image_panel(axes[1, 1], refined_np, "signal after refinement")
    image_panel(axes[1, 2], residual_after, f"residual after (rho={final_rho:.2f})")
    fig.savefig(paths["maps_plot"], dpi=180)
    plt.close(fig)

    calls = np.arange(1, len(total_history) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(calls, total_history, label="total")
    axes[0, 0].plot(calls, likelihood_history, label="likelihood")
    axes[0, 0].plot(calls, anchor_history, label="anchor")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("energy")
    axes[0, 0].legend()
    axes[0, 1].plot(calls, rho_history)
    axes[0, 1].axhline(expected_rho, color="black", ls="--", label=r"$\sqrt{D}$")
    axes[0, 1].set_ylabel(r"$\rho$")
    axes[0, 1].legend()
    axes[1, 0].plot(calls, correction_rms_history)
    axes[1, 0].set_ylabel("correction RMS")
    axes[1, 0].set_xlabel("L-BFGS closure call")
    axes[1, 1].plot(calls, gradient_rms_history)
    axes[1, 1].set_ylabel("gradient RMS")
    axes[1, 1].set_xlabel("L-BFGS closure call")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.savefig(paths["diagnostics_plot"], dpi=180)
    plt.close(fig)

    active_indices = np.flatnonzero(active.detach().cpu().numpy())
    before_contribution = initial_whitened**2
    after_contribution = final_whitened**2
    order = np.argsort(before_contribution)[::-1]
    n_top = min(args.top_statistics, order.size)
    print("largest initial contributions to rho^2 (before -> after):", flush=True)
    print("rank  full_statistic_index  before_z_squared  after_z_squared", flush=True)
    for rank, position in enumerate(order[:n_top], start=1):
        print(
            f"{rank:4d}  {active_indices[position]:20d} "
            f"{before_contribution[position]:17.6f} {after_contribution[position]:16.6f}",
            flush=True,
        )

    summary = {
        "method": "exact CIB scattering likelihood plus L2 anchor to component map",
        "input_component_map_preserved": str(args.initial_signal),
        "data": str(args.data),
        "covariance_npz": str(args.covariance_npz),
        "covariance_json": str(args.covariance_json),
        "output_files": {key: str(value) for key, value in paths.items()},
        "active_dimension": int(active.sum()),
        "expected_rho": expected_rho,
        "initial_rho": initial_rho,
        "final_rho": final_rho,
        "final_total_energy": float(final_total.detach().cpu()),
        "final_likelihood_energy": float(final_like.detach().cpu()),
        "final_anchor_energy": float(final_anchor.detach().cpu()),
        "final_correction_rms": float(
            torch.sqrt(torch.mean(final_correction.square())).detach().cpu()
        ),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for path in paths.values():
        print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
