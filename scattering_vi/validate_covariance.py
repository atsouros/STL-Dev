#!/usr/bin/env python3
"""Sample a learned statistic Gaussian and synthesize validation maps.

If the covariance product contains ``bias``, the sampled distribution is the
signal-side approximation phi(s) ~ N(mean - bias, covariance). Otherwise the
original N(mean, covariance) validation is used unchanged.
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
    configure_backend,
    load_target,
    make_running_operator,
    synthesize_from_flat_statistics,
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
    parser = argparse.ArgumentParser(
        description="Synthesize maps from sampled learned statistics."
    )
    parser.add_argument("--covariance-npz", type=Path, required=True)
    parser.add_argument("--covariance-json", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_validation")
    parser.add_argument("--run-name", default="sampled_stat_synthesis")
    parser.add_argument("--n-samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=60001)

    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--subtract-mean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--take-log", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--whiten", type=parse_bool, nargs="?", const=True, default=False)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")

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


def apply_estimation_config(args: argparse.Namespace) -> None:
    json_path = args.covariance_json or args.covariance_npz.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing covariance metadata JSON: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    config = metadata.get("config", {})
    for key in (
        "target_size",
        "subtract_mean",
        "take_log",
        "whiten",
        "J",
        "L",
        "wtype",
        "iso",
        "angular_ft",
        "harmonics_angle",
        "scale_ft",
        "harmonics_scale",
        "dj",
        "compute_ps",
        "fewer_convolutions",
        "pbc",
        "dtype",
    ):
        if key in config:
            setattr(args, key, config[key])
    args.covariance_json = json_path


def sample_gaussian(mean: np.ndarray, covariance: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cov = 0.5 * (covariance + covariance.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    normals = rng.standard_normal((n_samples, mean.shape[0]))
    return mean[None, :] + normals @ (np.sqrt(evals)[:, None] * evecs.T)


def covariance_payload(path: Path) -> tuple[np.ndarray | None, np.ndarray]:
    payload = np.load(path)
    if "mean" in payload.files:
        mean = payload["mean"]
    elif "target_statistics" in payload.files:
        mean = payload["target_statistics"]
    elif "synthesized_mean" in payload.files:
        mean = payload["synthesized_mean"]
    else:
        mean = None
    if "covariance" in payload.files:
        covariance = payload["covariance"]
    elif "covariance_within_batch" in payload.files:
        covariance = payload["covariance_within_batch"]
    elif "covariance_pooled" in payload.files:
        covariance = payload["covariance_pooled"]
    else:
        stats = payload["statistic_samples"] if "statistic_samples" in payload.files else payload["synthesized_statistics"]
        covariance = np.cov(stats, rowvar=False)
    return mean, covariance


def load_scattering_reference(target_stats: Any, payload: np.lib.npyio.NpzFile, device: torch.device, dtype: torch.dtype) -> None:
    aliases = {
        "mean_pre_std": ("stl_reference_field_mean", "target_mean_pre_std"),
        "std_pre_std": ("stl_reference_field_std", "target_std_pre_std"),
        "S2_ref_sqrt_chan_diag": ("stl_reference_S2_ref_sqrt_chan_diag", "st_S2_ref_sqrt_chan_diag"),
        "var_ref": ("stl_reference_var_ref", "st_var_ref"),
        "PS_ref_sqrt_chan_diag": ("stl_reference_PS_ref_sqrt_chan_diag", "st_PS_ref_sqrt_chan_diag"),
    }
    for attr, keys in aliases.items():
        for key in keys:
            if key in payload.files and payload[key].size:
                setattr(target_stats, attr, torch.as_tensor(payload[key], device=device, dtype=dtype))
                break


def plot_grid(
    path: Path,
    syntheses: list[np.ndarray],
    references: np.ndarray,
    reference_title: str,
) -> None:
    images = [references[0]] + syntheses
    titles = [reference_title] + [f"synthesis {i + 1}" for i in range(len(syntheses))]
    if len(images) != 8:
        raise ValueError("The 2x4 plot requires exactly one original and seven syntheses")
    all_images = images
    vmin = float(min(np.percentile(image, 1) for image in all_images))
    vmax = float(max(np.percentile(image, 99) for image in all_images))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_samples != 7:
        raise ValueError("This script currently produces a 2x4 plot, so --n-samples must be 7")
    apply_estimation_config(args)

    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    configure_backend(device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_np, input_info = load_target(args)
    target = torch.as_tensor(target_np[:1], device=device, dtype=dtype)
    target_stats, _, target_flat_t = target_statistics(target, args)
    cov_payload = np.load(args.covariance_npz)
    load_scattering_reference(target_stats, cov_payload, device, dtype)
    target_flat = (
        cov_payload["mean"].astype(np.float64, copy=False)
        if "mean" in cov_payload.files
        else cov_payload["target_statistics"].astype(np.float64, copy=False)
        if "target_statistics" in cov_payload.files
        else target_flat_t.cpu().numpy().astype(np.float64, copy=False)
    )
    target_mean_pre_std = target_stats.mean_pre_std.detach().clone()
    target_std_pre_std = target_stats.std_pre_std.detach().clone()

    mean, covariance = covariance_payload(args.covariance_npz)
    if mean is None:
        mean = target_flat
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    bias = None
    distribution_mean = mean
    distribution_mode = "learned_statistic_gaussian"
    if "bias" in cov_payload.files:
        bias = np.asarray(cov_payload["bias"], dtype=np.float64)
        if bias.shape != mean.shape:
            raise ValueError(
                f"Bias shape {bias.shape} does not match mean shape {mean.shape}"
            )
        distribution_mean = mean - bias
        distribution_mode = "bias_corrected_signal_prior"
        print("Bias detected: sampling phi(s) ~ N(mean - bias, covariance)")
    else:
        print("No bias found: sampling the original N(mean, covariance)")
    samples = sample_gaussian(
        distribution_mean, covariance, args.n_samples, args.seed
    )

    syntheses: list[np.ndarray] = []
    for index, sampled_stat in enumerate(samples):
        target_stats.mean_pre_std = target_mean_pre_std.clone()
        target_stats.std_pre_std = target_std_pre_std.clone()
        running_op = make_running_operator(args.target_size, args, device=device, dtype=dtype)
        image = synthesize_from_flat_statistics(
            target_flat=torch.as_tensor(sampled_stat, device=device, dtype=dtype),
            target_stats=target_stats,
            st_op_running=running_op,
            running_shape=(args.target_size, args.target_size),
            pbc_running=args.pbc,
            lr=args.lr,
            max_iter=args.max_iter,
            history_size=args.history_size,
            print_iter=args.print_iter,
            verbose=True,
            seed=args.seed + index,
        )
        syntheses.append(image.detach().cpu().numpy())

    plot_path = args.output_dir / f"{args.run_name}_2x4.png"
    npz_path = args.output_dir / f"{args.run_name}.npz"
    json_path = args.output_dir / f"{args.run_name}.json"
    reference_title = "reference signal" if bias is not None else "original"
    plot_grid(plot_path, syntheses, target_np, reference_title)
    np.savez_compressed(
        npz_path,
        target_map=target_np,
        mean=target_flat,
        source_statistic_mean=mean,
        statistic_bias=(np.array([]) if bias is None else bias),
        gaussian_statistic_mean=distribution_mean,
        gaussian_statistic_covariance=covariance,
        sampled_statistics=samples,
        synthesized_maps=np.stack(syntheses, axis=0),
    )
    metadata: dict[str, Any] = {
        "input": input_info,
        "covariance_npz": str(args.covariance_npz),
        "covariance_json": str(args.covariance_json),
        "plot_file": plot_path.name,
        "npz_file": npz_path.name,
        "n_samples": args.n_samples,
        "distribution_mode": distribution_mode,
        "bias_detected": bias is not None,
        "sampled_statistic_mean_definition": (
            "mean - bias" if bias is not None else "mean"
        ),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {plot_path}")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
