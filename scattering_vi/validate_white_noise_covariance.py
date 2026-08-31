#!/usr/bin/env python3
"""Validate mean-field synthesis covariance on 128x128 Gaussian random fields.

Two covariance estimates are compared:

1. A high-precision reference covariance from many independent GRF maps.
2. The covariance of synthesized maps, optionally augmented by rotations and
   random recenterings, from one GRF realization.

Bootstrap subsets of the reference ensemble, each with the same size as the
synthesis batch, provide the finite-sample compatibility baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimate_scattering_covariance import (
    DataClass,
    build_st_operator,
    call_optimize_from_stats,
    configure_backend,
    augment_maps_by_rotations_and_recentering,
    make_running_operator,
    scattering_vectors,
    synthesize_from_flat_statistics,
    target_statistics,
    torch_dtype,
)
from STL_main.Synthesis import apply_nyquist_filter


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct and mean-field-synthesis covariance on GRFs."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "results_power_law_grf"
    )
    parser.add_argument("--run-name", default="power_law_grf_covariance_validation")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--field-std", type=float, default=1.0)
    parser.add_argument("--spectral-index", type=float, default=-11.0 / 3.0)
    parser.add_argument("--target-seed", type=int, default=10001)
    parser.add_argument("--reference-seed", type=int, default=20001)
    parser.add_argument("--synthesis-seed", type=int, default=30001)
    parser.add_argument("--bootstrap-seed", type=int, default=40001)
    parser.add_argument("--stat-sample-seed", type=int, default=50001)
    parser.add_argument(
        "--n-reference",
        type=int,
        default=10000,
        help="Independent GRF maps used for the reference covariance.",
    )
    parser.add_argument("--reference-chunk-size", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum maps optimized jointly in each mean-field synthesis batch.",
    )
    parser.add_argument("--n-samples", type=int, default=0)
    parser.add_argument("--sample-multiplier", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )

    # Requested reduced representation.
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--wtype", default="Bump-Steerable")
    parser.add_argument(
        "--iso", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--angular-ft", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--harmonics-angle", type=int, default=2)
    parser.add_argument(
        "--scale-ft", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--harmonics-scale", type=int, default=3)
    parser.add_argument("--dj", type=int, default=3)
    parser.add_argument(
        "--compute-ps", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fewer-convolutions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pbc", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stats-chunk-size", type=int, default=64)

    # Synthesis.
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--print-iter", type=int, default=10)
    parser.add_argument(
        "--apply-nyquist-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Off by default because filtering after optimization changes the matched statistics.",
    )
    parser.add_argument(
        "--augment-symmetries",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--augment-random-centers", type=int, default=4)

    # Covariance diagnostics.
    parser.add_argument(
        "--variance-relative-tol",
        type=float,
        default=1e-10,
        help="Drop features whose reference variance is below this fraction of the maximum.",
    )
    parser.add_argument(
        "--max-full-cov-dim",
        type=int,
        default=2048,
        help="Compute full covariance/bootstrap metrics only below this dimension.",
    )
    parser.add_argument(
        "--save-reference-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-syntheses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sampled-stat-synthesis-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def generator(device: torch.device, seed: int) -> torch.Generator:
    result = torch.Generator(device=device)
    result.manual_seed(seed)
    return result


def power_law_grf(
    count: int,
    size: int,
    std: float,
    spectral_index: float,
    rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gaussian = torch.randn(
        (count, size, size),
        generator=rng,
        device=device,
        dtype=dtype,
    )
    freq = torch.fft.fftfreq(size, device=device, dtype=dtype) * size
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    k = torch.sqrt(kx**2 + ky**2)
    amplitude = torch.zeros_like(k)
    nonzero = k > 0
    amplitude[nonzero] = k[nonzero] ** (0.5 * spectral_index)
    field = torch.fft.ifft2(torch.fft.fft2(gaussian) * amplitude).real
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    field_std = field.std(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    return field * (std / field_std)


def reference_statistics(
    args: argparse.Namespace,
    st_op: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    rng = generator(device, args.reference_seed)
    chunks: list[np.ndarray] = []
    completed = 0
    while completed < args.n_reference:
        count = min(args.reference_chunk_size, args.n_reference - completed)
        maps = power_law_grf(
            count,
            args.size,
            args.field_std,
            args.spectral_index,
            rng,
            device,
            dtype,
        )
        flat = scattering_vectors(maps, st_op, args)
        chunks.append(flat.numpy().astype(np.float64, copy=False))
        completed += count
        print(f"Reference statistics: {completed}/{args.n_reference}")
    return np.concatenate(chunks, axis=0)


def sample_covariance(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    return mean, covariance


def sample_statistic_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    evals, evecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
    evals = np.clip(evals, 0.0, None)
    normal = rng.standard_normal(mean.shape[0])
    return mean + evecs @ (np.sqrt(evals) * normal)


def plot_sampled_stat_synthesis(
    path: Path,
    synthesized: np.ndarray,
    true_field: np.ndarray,
) -> None:
    vmin = float(min(np.percentile(synthesized, 1), np.percentile(true_field, 1)))
    vmax = float(max(np.percentile(synthesized, 99), np.percentile(true_field, 99)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(synthesized, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Synthesis from sampled statistics")
    axes[0].axis("off")
    axes[1].imshow(true_field, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("True GRF realization")
    axes[1].axis("off")
    fig.colorbar(im0, ax=axes, shrink=0.82)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(estimate - reference) / denominator)


def percentile_rank(values: np.ndarray, observation: float) -> float:
    return float(100.0 * np.mean(values <= observation))


def bootstrap_compatibility(
    reference_stats: np.ndarray,
    reference_covariance: np.ndarray,
    synthesis_covariance: np.ndarray,
    batch_size: int,
    replicates: int,
    seed: int,
    full_covariance: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    dimension = reference_stats.shape[1]
    reference_variance = np.diag(reference_covariance)
    synthesis_variance = np.diag(synthesis_covariance)

    variance_samples = np.empty((replicates, dimension), dtype=np.float64)
    diagonal_errors = np.empty(replicates, dtype=np.float64)
    full_errors = (
        np.empty(replicates, dtype=np.float64) if full_covariance else None
    )

    for index in range(replicates):
        selection = rng.integers(
            0, reference_stats.shape[0], size=batch_size
        )
        _, covariance = sample_covariance(reference_stats[selection])
        variance_samples[index] = np.diag(covariance)
        diagonal_errors[index] = relative_l2(
            variance_samples[index], reference_variance
        )
        if full_errors is not None:
            full_errors[index] = relative_l2(
                covariance, reference_covariance
            )

    lower = np.percentile(variance_samples, 2.5, axis=0)
    upper = np.percentile(variance_samples, 97.5, axis=0)
    bootstrap_std = variance_samples.std(axis=0, ddof=1)
    safe_std = np.where(bootstrap_std > 0, bootstrap_std, np.inf)
    synthesis_z = (synthesis_variance - reference_variance) / safe_std

    observed_diagonal_error = relative_l2(
        synthesis_variance, reference_variance
    )
    metrics = {
        "observed_diagonal_relative_l2": observed_diagonal_error,
        "bootstrap_diagonal_relative_l2_median": float(
            np.median(diagonal_errors)
        ),
        "bootstrap_diagonal_relative_l2_p95": float(
            np.percentile(diagonal_errors, 95)
        ),
        "observed_diagonal_error_percentile": percentile_rank(
            diagonal_errors, observed_diagonal_error
        ),
        "fraction_synthesis_variances_in_bootstrap_95_interval": float(
            np.mean(
                (synthesis_variance >= lower)
                & (synthesis_variance <= upper)
            )
        ),
        "rms_bootstrap_standardized_variance_error": float(
            np.sqrt(np.mean(synthesis_z**2))
        ),
        "median_abs_bootstrap_standardized_variance_error": float(
            np.median(np.abs(synthesis_z))
        ),
    }

    arrays: dict[str, np.ndarray] = {
        "bootstrap_variance_lower_95": lower,
        "bootstrap_variance_upper_95": upper,
        "bootstrap_variance_std": bootstrap_std,
        "synthesis_variance_bootstrap_z": synthesis_z,
        "bootstrap_diagonal_relative_l2": diagonal_errors,
    }

    if full_errors is not None:
        observed_full_error = relative_l2(
            synthesis_covariance, reference_covariance
        )
        metrics.update(
            {
                "observed_full_covariance_relative_frobenius": observed_full_error,
                "bootstrap_full_covariance_relative_frobenius_median": float(
                    np.median(full_errors)
                ),
                "bootstrap_full_covariance_relative_frobenius_p95": float(
                    np.percentile(full_errors, 95)
                ),
                "observed_full_covariance_error_percentile": percentile_rank(
                    full_errors, observed_full_error
                ),
            }
        )
        arrays["bootstrap_full_covariance_relative_frobenius"] = full_errors

    return arrays, metrics


def diagnostic_plot(
    path: Path,
    reference_variance: np.ndarray,
    synthesis_variance: np.ndarray,
    bootstrap_lower: np.ndarray,
    bootstrap_upper: np.ndarray,
) -> None:
    ref = reference_variance
    syn = synthesis_variance
    lower = bootstrap_lower
    upper = bootstrap_upper
    ratio = syn / ref

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(ref, syn, ".", alpha=0.65)
    minimum = min(float(ref.min()), float(syn.min()))
    maximum = max(float(ref.max()), float(syn.max()))
    axes[0].plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1)
    axes[0].set_xlabel("Direct GRF variance")
    axes[0].set_ylabel("Mean-field synthesis variance")
    axes[0].grid(True, alpha=0.3)

    indices = np.arange(ref.shape[0])
    axes[1].fill_between(
        indices,
        lower / ref,
        upper / ref,
        alpha=0.25,
        label="95% interval: 100 direct WN maps",
    )
    axes[1].plot(indices, ratio, ".", markersize=3, label="synthesis/reference")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Active reduced-statistic index")
    axes[1].set_ylabel("Variance ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_config(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.n_reference <= args.batch_size:
        raise ValueError("--n-reference must be larger than --batch-size")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.n_samples < 0:
        raise ValueError("--n-samples must be non-negative")
    if args.sample_multiplier < 1:
        raise ValueError("--sample-multiplier must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_name}.npz"
    metadata_path = output_dir / f"{args.run_name}.json"
    plot_path = output_dir / f"{args.run_name}.png"
    sampled_stat_plot_path = output_dir / f"{args.run_name}_sampled_stat_synthesis.png"

    target_rng = generator(device, args.target_seed)
    target = power_law_grf(
        1,
        args.size,
        args.field_std,
        args.spectral_index,
        target_rng,
        device,
        dtype,
    )[0]
    target_stats, target_op, target_flat_t = target_statistics(target, args)
    target_flat = target_flat_t.cpu().numpy().astype(np.float64, copy=False)
    statistic_dimension = int(target_flat.shape[0])
    n_synthesis_samples = (
        int(args.n_samples)
        if args.n_samples > 0
        else int(args.sample_multiplier * statistic_dimension)
    )
    target_mean_pre_std = target_stats.mean_pre_std.detach().clone()
    target_std_pre_std = target_stats.std_pre_std.detach().clone()

    print(f"Device: {device} | dtype: {dtype}")
    print(
        f"Power-law GRF target: {args.size}x{args.size}, "
        f"P(k)~k^{args.spectral_index}, std={args.field_std}"
    )
    print(f"Reduced-statistic dimension: {target_flat.shape[0]}")
    print(
        f"Reference covariance: {args.n_reference} independent maps; "
        f"synthesis covariance: {n_synthesis_samples} maps"
    )

    reference_stats = reference_statistics(
        args, target_op, device=device, dtype=dtype
    )
    reference_mean, reference_covariance = sample_covariance(reference_stats)
    reference_variance = np.diag(reference_covariance)

    running_op = make_running_operator(
        args.size, args, device=device, dtype=dtype
    )
    synthesized_batches: list[torch.Tensor] = []
    synthesis_stats_batches: list[np.ndarray] = []
    n_completed = 0
    batch_index = 0
    while n_completed < n_synthesis_samples:
        current_batch_size = min(args.batch_size, n_synthesis_samples - n_completed)
        batch_seed = args.synthesis_seed + batch_index
        target_stats.mean_pre_std = target_mean_pre_std.clone()
        target_stats.std_pre_std = target_std_pre_std.clone()
        synthesized = call_optimize_from_stats(
            target_stats=target_stats,
            st_op_running=running_op,
            batch_size=current_batch_size,
            running_shape=(args.size, args.size),
            pbc_running=args.pbc,
            init_running=None,
            mean_field=True,
            lr=args.lr,
            max_iter=args.max_iter,
            history_size=args.history_size,
            print_iter=args.print_iter,
            verbose=True,
            seed=batch_seed,
        )
        if synthesized.ndim == 2:
            synthesized = synthesized[None, :, :]
        if args.apply_nyquist_filter:
            synthesized = apply_nyquist_filter(synthesized)
        synthesized_batches.append(synthesized.detach().cpu())

        synthesis_maps_for_stats = synthesized
        if args.augment_symmetries:
            synthesis_maps_for_stats = augment_maps_by_rotations_and_recentering(
                synthesis_maps_for_stats,
                n_centers=args.augment_random_centers,
                seed=batch_seed + 1_000_000,
            )
            print(
                f"Augmented synthesized tensor shape: {tuple(synthesis_maps_for_stats.shape)}"
            )
        synthesis_stats_batches.append(
            scattering_vectors(synthesis_maps_for_stats, target_op, args)
            .numpy()
            .astype(np.float64, copy=False)
        )
        n_completed += current_batch_size
        batch_index += 1

    synthesized = torch.cat(synthesized_batches, dim=0)
    synthesis_stats = np.concatenate(synthesis_stats_batches, axis=0)
    synthesis_stats = synthesis_stats.astype(np.float64, copy=False)
    synthesis_mean, synthesis_covariance = sample_covariance(synthesis_stats)
    synthesis_variance = np.diag(synthesis_covariance)

    sampled_statistic = None
    sampled_stat_synthesis = None
    sampled_stat_true_field = None
    if args.sampled_stat_synthesis_test:
        sampled_statistic = sample_statistic_gaussian(
            synthesis_mean,
            synthesis_covariance,
            args.stat_sample_seed,
        )
        running_op_sample = make_running_operator(
            args.size, args, device=device, dtype=dtype
        )
        target_stats.mean_pre_std = target_mean_pre_std.clone()
        target_stats.std_pre_std = target_std_pre_std.clone()
        sampled_stat_synthesis_t = synthesize_from_flat_statistics(
            target_flat=torch.as_tensor(sampled_statistic, device=device, dtype=dtype),
            target_stats=target_stats,
            st_op_running=running_op_sample,
            running_shape=(args.size, args.size),
            pbc_running=args.pbc,
            lr=args.lr,
            max_iter=args.max_iter,
            history_size=args.history_size,
            print_iter=args.print_iter,
            verbose=True,
            seed=args.stat_sample_seed,
        )
        sampled_stat_synthesis = sampled_stat_synthesis_t.detach().cpu().numpy()
        true_rng = generator(device, args.stat_sample_seed + 1)
        sampled_stat_true_field = power_law_grf(
            1,
            args.size,
            args.field_std,
            args.spectral_index,
            true_rng,
            device,
            dtype,
        )[0].detach().cpu().numpy()
        plot_sampled_stat_synthesis(
            sampled_stat_plot_path,
            sampled_stat_synthesis,
            sampled_stat_true_field,
        )

    variance_threshold = (
        args.variance_relative_tol * float(reference_variance.max())
    )
    active = reference_variance > variance_threshold
    if not active.any():
        raise RuntimeError("All reduced statistics were classified as degenerate")

    ref_cov_active = reference_covariance[np.ix_(active, active)]
    syn_cov_active = synthesis_covariance[np.ix_(active, active)]
    ref_stats_active = reference_stats[:, active]
    full_covariance = int(active.sum()) <= args.max_full_cov_dim

    bootstrap_arrays, compatibility = bootstrap_compatibility(
        reference_stats=ref_stats_active,
        reference_covariance=ref_cov_active,
        synthesis_covariance=syn_cov_active,
        batch_size=synthesis_stats.shape[0],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        full_covariance=full_covariance,
    )

    synthesis_mean_error = synthesis_mean - target_flat
    compatibility.update(
        {
            "n_active_statistics": int(active.sum()),
            "reference_covariance_rank": int(
                np.linalg.matrix_rank(ref_cov_active)
            ),
            "synthesis_covariance_rank": int(
                np.linalg.matrix_rank(syn_cov_active)
            ),
            "synthesis_rank_upper_bound": min(
                int(active.sum()), synthesis_stats.shape[0] - 1
            ),
            "n_synthesized_maps": int(n_synthesis_samples),
            "n_synthesis_batches": int(batch_index),
            "n_augmented_synthesis_maps": int(synthesis_stats.shape[0]),
            "target_to_reference_mean_relative_l2": relative_l2(
                target_flat, reference_mean
            ),
            "synthesis_mean_to_target_relative_l2": float(
                np.linalg.norm(synthesis_mean_error)
                / max(float(np.linalg.norm(target_flat)), 1e-30)
            ),
            "median_synthesis_to_reference_variance_ratio": float(
                np.median(
                    synthesis_variance[active]
                    / reference_variance[active]
                )
            ),
            "fraction_variance_ratios_within_20_percent": float(
                np.mean(
                    np.abs(
                        synthesis_variance[active]
                        / reference_variance[active]
                        - 1.0
                    )
                    <= 0.2
                )
            ),
        }
    )

    ratios = synthesis_variance[active] / reference_variance[active]
    compatibility["fraction_variance_ratios_within_factor_two"] = float(
        np.mean((ratios >= 0.5) & (ratios <= 2.0))
    )

    payload: dict[str, np.ndarray] = {
        "target_map": target.detach().cpu().numpy(),
        "target_statistics": target_flat,
        "reference_mean": reference_mean,
        "reference_covariance": reference_covariance,
        "reference_variance": reference_variance,
        "synthesis_mean": synthesis_mean,
        "synthesis_covariance": synthesis_covariance,
        "synthesis_variance": synthesis_variance,
        "active_statistic_mask": active,
        "synthesis_statistics": synthesis_stats,
        **bootstrap_arrays,
    }
    if sampled_statistic is not None:
        payload["sampled_gaussian_statistic"] = sampled_statistic
        payload["sampled_stat_synthesis_map"] = sampled_stat_synthesis
        payload["sampled_stat_true_grf_map"] = sampled_stat_true_field
    if args.save_reference_statistics:
        payload["reference_statistics"] = reference_stats
    if args.save_syntheses:
        payload["synthesized_maps"] = synthesized.detach().cpu().numpy()
    np.savez_compressed(result_path, **payload)

    diagnostic_plot(
        plot_path,
        reference_variance[active],
        synthesis_variance[active],
        bootstrap_arrays["bootstrap_variance_lower_95"],
        bootstrap_arrays["bootstrap_variance_upper_95"],
    )

    compatible_diagonal = (
        compatibility["observed_diagonal_error_percentile"] <= 95.0
    )
    compatible_full = (
        compatibility.get(
            "observed_full_covariance_error_percentile", 0.0
        )
        <= 95.0
        if full_covariance
        else None
    )
    metadata = {
        "experiment": (
            "Direct GRF covariance versus mean-field "
            "synthesis covariance"
        ),
        "config": json_config(args),
        "statistic_dimension": int(target_flat.shape[0]),
        "active_statistic_dimension": int(active.sum()),
        "variance_threshold": variance_threshold,
        "compatibility_metrics": compatibility,
        "compatible_at_bootstrap_95_percent_diagonal": compatible_diagonal,
        "compatible_at_bootstrap_95_percent_full": compatible_full,
        "result_file": result_path.name,
        "plot_file": plot_path.name,
        "sampled_stat_synthesis_plot_file": (
            sampled_stat_plot_path.name if args.sampled_stat_synthesis_test else None
        ),
        "notes": [
            (
                "The reference ensemble and synthesized maps are evaluated "
                "with the same target-derived scattering normalization."
            ),
            (
                "The synthesis covariance has rank at most batch_size - 1; "
                "full-matrix compatibility is therefore interpreted against "
                "size-matched direct-GRF bootstrap covariances."
            ),
            (
                "The 100 synthesized maps are coupled through their constrained "
                "batch-mean statistics and are not strictly independent draws."
            ),
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    print("\nCompatibility summary")
    for key, value in compatibility.items():
        print(f"{key}: {value}")
    print(
        "Diagonal covariance compatible at bootstrap 95% level: "
        f"{compatible_diagonal}"
    )
    if full_covariance:
        print(
            "Full covariance compatible at bootstrap 95% level: "
            f"{compatible_full}"
        )
    else:
        print(
            "Full covariance bootstrap comparison skipped because active "
            f"dimension {active.sum()} exceeds {args.max_full_cov_dim}."
        )
    print(f"Saved: {result_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
