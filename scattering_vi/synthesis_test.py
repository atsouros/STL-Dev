#!/usr/bin/env python3
"""Run one mean-field synthesis and plot loss/history plus map comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.optim import LBFGS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimate_scattering_covariance import (
    apply_st_operator,
    configure_backend,
    load_target,
    make_running_operator,
    target_statistics,
    torch_dtype,
)


SCRIPT_DIR = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="Single synthesis diagnostic.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_synthesis_test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--subtract-mean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--take-log", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--whiten", type=parse_bool, nargs="?", const=True, default=False)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=70001)

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

    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--print-iter", type=int, default=10)
    return parser.parse_args()


def run_single_synthesis(
    target_stats: Any,
    running_op: Any,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, np.ndarray]:
    torch.manual_seed(args.seed)
    device = running_op.wavelet_op.device
    dtype = running_op.wavelet_op.dtype
    u = torch.randn(
        (1, target_stats.Nc, args.target_size, args.target_size),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    target_flat = target_stats.to_flatten(
        keep_batch_dim=True,
        mean_along_batch=True,
        keepnans=True,
    ).detach()
    target_flat = target_flat[~target_flat.isnan()]

    if target_stats.S2_ref_sqrt_chan_diag is not None:
        running_op.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    running_op.var_ref = target_stats.var_ref
    if running_op.compute_PS:
        running_op.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    optimizer = LBFGS(
        [u],
        lr=args.lr,
        max_iter=args.max_iter,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )
    losses: list[float] = []

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        data_u = target_stats.DataClass(u, pbc=args.pbc)
        stats_u = apply_st_operator(
            running_op,
            data_u,
            has_fewer_convolutions=target_stats.has_fewer_convolutions,
            compute_cross_matrix=target_stats.compute_cross_matrix,
            compute_PS=running_op.compute_PS,
            norm="load_ref",
        )
        flat_u = stats_u.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=True,
            keepnans=False,
        )
        loss = ((flat_u - target_flat).abs() ** 2).sum()
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        if len(losses) == 1 or len(losses) % args.print_iter == 0:
            print(f"[LBFGS] iter {len(losses)}, loss={loss.item():.6e}")
        return loss

    optimizer.step(closure)
    u_opt = u.detach()
    if target_stats.standardized:
        data_u_opt = target_stats.DataClass(u_opt, pbc=args.pbc)
        running_op.wavelet_op.unstandardize(
            data_u_opt,
            mean=target_stats.mean_pre_std.mean(dim=0),
            std=target_stats.std_pre_std.mean(dim=0),
            inplace=True,
        )
        u_opt = data_u_opt.array
    return u_opt[0, 0], np.asarray(losses, dtype=np.float64)


def save_plots(
    output_dir: Path,
    run_name: str,
    reference: np.ndarray,
    synthesis: np.ndarray,
    losses: np.ndarray,
) -> tuple[Path, Path]:
    comparison_path = output_dir / f"{run_name}_comparison.png"
    loss_path = output_dir / f"{run_name}_loss.png"

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
    fig.savefig(comparison_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(np.arange(1, losses.size + 1), losses)
    ax.set_yscale("log")
    ax.set_xlabel("LBFGS closure evaluation")
    ax.set_ylabel("synthesis loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(loss_path, dpi=170)
    plt.close(fig)
    return comparison_path, loss_path


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.input.stem}_synthesis_test"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_np, input_info = load_target(args)
    target = torch.as_tensor(target_np, device=device, dtype=dtype)
    target_stats, _, _ = target_statistics(target, args)
    running_op = make_running_operator(args.target_size, args, device=device, dtype=dtype)
    synthesis_t, losses = run_single_synthesis(target_stats, running_op, args)
    synthesis = synthesis_t.detach().cpu().numpy()
    reference = target_np[0]

    comparison_path, loss_path = save_plots(
        args.output_dir,
        run_name,
        reference,
        synthesis,
        losses,
    )
    npz_path = args.output_dir / f"{run_name}.npz"
    json_path = args.output_dir / f"{run_name}.json"
    np.savez_compressed(
        npz_path,
        reference_map=reference,
        synthesized_map=synthesis,
        loss_history=losses,
    )
    json_path.write_text(
        json.dumps(
            {
                "input": input_info,
                "comparison_plot": comparison_path.name,
                "loss_plot": loss_path.name,
                "npz_file": npz_path.name,
                "final_loss": float(losses[-1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {comparison_path}")
    print(f"Saved {loss_path}")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
