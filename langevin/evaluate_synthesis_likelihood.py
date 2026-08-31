#!/usr/bin/env python3
"""Evaluate the learned scattering likelihood on covariance-validation syntheses.

The validation NPZ contains both Gaussian statistic draws and maps synthesized
to reproduce those draws.  This script evaluates both with exactly the same
JSON configuration, active mask, covariance, and STL reference state used by
``langevin_sample.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from langevin_sample import (
    build_st_operator,
    configure_backend,
    load_metadata,
    load_model,
    scattering_vector,
    torch_dtype,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COVARIANCE_STEM = "100_Herschel_Lockman_250m_tiles_3400x256x256_covariance"
SYNTHESIS_STEM = "100_Herschel_Lockman_250m_tiles_3400x256x256_sampled_stat_synthesis"
DEFAULT_CIB_BANK = (
    REPO_ROOT / "data" / "test" / "100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthesis-npz",
        type=Path,
        default=REPO_ROOT / "scattering_vi" / "results_validation" / f"{SYNTHESIS_STEM}.npz",
    )
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
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "results" / "likelihood_validation"
    )
    parser.add_argument("--cib-bank", type=Path, default=DEFAULT_CIB_BANK)
    parser.add_argument(
        "--cib-seed",
        type=int,
        default=60001,
        help="Seed selecting real CIB maps; the number selected matches the syntheses.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--covariance-jitter-rel", type=float, default=1e-6)
    parser.add_argument("--top-statistics", type=int, default=10)
    return parser.parse_args()


def whitened_discrepancy(
    statistic: torch.Tensor,
    mean: torch.Tensor,
    covariance_cholesky: torch.Tensor,
) -> torch.Tensor:
    difference = statistic - mean
    return torch.linalg.solve_triangular(
        covariance_cholesky, difference[:, None], upper=False
    )[:, 0]


def active_statistic(
    statistic: np.ndarray,
    active: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    statistic_t = torch.as_tensor(statistic, device=device, dtype=dtype).real.flatten()
    if statistic_t.numel() == active.numel():
        return statistic_t[active]
    if statistic_t.numel() == int(active.sum()):
        return statistic_t
    raise ValueError(
        f"Stored statistic has length {statistic_t.numel()}, but expected "
        f"{active.numel()} full or {int(active.sum())} active values"
    )


def summarize(name: str, rho: np.ndarray, energy: np.ndarray) -> None:
    print(
        f"{name}: n={rho.size} "
        f"rho mean={rho.mean():.4f} median={np.median(rho):.4f} "
        f"min={rho.min():.4f} max={rho.max():.4f} "
        f"U_like mean={energy.mean():.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.synthesis_npz, "validation synthesis NPZ"),
        (args.covariance_npz, "covariance NPZ"),
        (args.covariance_json, "covariance JSON"),
        (args.cib_bank, "real CIB map bank"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.covariance_jitter_rel <= 0 or args.top_statistics < 0:
        raise ValueError("Require positive covariance jitter and non-negative top-statistics")

    _metadata, config = load_metadata(args.covariance_json)
    dtype_name = args.dtype or str(config.dtype)
    dtype = torch_dtype(dtype_name)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
        torch.cuda.set_device(device)
    configure_backend(device, dtype)

    st_op = build_st_operator(int(config.target_size), config, device, dtype)
    mean, covariance_cholesky, active, _references, jitter = load_model(
        args.covariance_npz,
        st_op,
        device,
        dtype,
        args.covariance_jitter_rel,
    )

    with np.load(args.synthesis_npz) as payload:
        required = ("sampled_statistics", "synthesized_maps")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"Validation NPZ is missing arrays: {missing}")
        sampled_statistics = np.asarray(payload["sampled_statistics"])
        synthesized_maps = np.asarray(payload["synthesized_maps"])

    if sampled_statistics.ndim != 2:
        raise ValueError(f"Expected sampled_statistics shape (N,D), got {sampled_statistics.shape}")
    if synthesized_maps.ndim != 3:
        raise ValueError(f"Expected synthesized_maps shape (N,H,W), got {synthesized_maps.shape}")
    if sampled_statistics.shape[0] != synthesized_maps.shape[0]:
        raise ValueError("The numbers of sampled statistics and synthesized maps differ")
    expected_shape = (int(config.target_size), int(config.target_size))
    if tuple(synthesized_maps.shape[-2:]) != expected_shape:
        raise ValueError(f"Expected synthesized map size {expected_shape}, got {synthesized_maps.shape[-2:]}")

    cib_bank = np.load(args.cib_bank, mmap_mode="r")
    if cib_bank.ndim == 2:
        cib_bank = cib_bank[None]
    if cib_bank.ndim != 3 or tuple(cib_bank.shape[-2:]) != expected_shape:
        raise ValueError(
            f"Expected CIB bank shape (N,{expected_shape[0]},{expected_shape[1]}), "
            f"got {cib_bank.shape}"
        )
    n_samples = sampled_statistics.shape[0]
    if cib_bank.shape[0] < n_samples:
        raise ValueError(
            f"CIB bank has {cib_bank.shape[0]} maps, but {n_samples} are required"
        )
    rng = np.random.default_rng(args.cib_seed)
    cib_indices = rng.choice(cib_bank.shape[0], size=n_samples, replace=False)

    target_whitened: list[np.ndarray] = []
    realized_whitened: list[np.ndarray] = []
    cib_whitened: list[np.ndarray] = []
    print(
        f"active_dimension={int(active.sum())} expected_rho={math.sqrt(int(active.sum())):.4f} "
        f"jitter={jitter:.6e}",
        flush=True,
    )
    print(
        "index  target_U  target_rho  map_U  map_rho  cib_index  cib_U  cib_rho",
        flush=True,
    )
    for index, (sampled_statistic, synthesized_map, cib_index) in enumerate(
        zip(sampled_statistics, synthesized_maps, cib_indices)
    ):
        target = active_statistic(sampled_statistic, active, device, dtype)
        target_z = whitened_discrepancy(target, mean, covariance_cholesky)
        image_t = torch.as_tensor(synthesized_map, device=device, dtype=dtype)
        realized_full = scattering_vector(image_t, st_op, config)
        realized = realized_full[active]
        realized_z = whitened_discrepancy(realized, mean, covariance_cholesky)
        cib_t = torch.as_tensor(
            np.array(cib_bank[cib_index], copy=True), device=device, dtype=dtype
        )
        cib_full = scattering_vector(cib_t, st_op, config)
        cib_z = whitened_discrepancy(cib_full[active], mean, covariance_cholesky)
        target_whitened.append(target_z.detach().cpu().numpy())
        realized_whitened.append(realized_z.detach().cpu().numpy())
        cib_whitened.append(cib_z.detach().cpu().numpy())
        target_rho = float(torch.linalg.vector_norm(target_z).cpu())
        realized_rho = float(torch.linalg.vector_norm(realized_z).cpu())
        cib_rho = float(torch.linalg.vector_norm(cib_z).cpu())
        print(
            f"{index:5d}  {0.5 * target_rho**2:8.3f}  {target_rho:10.4f}  "
            f"{0.5 * realized_rho**2:8.3f}  {realized_rho:8.4f}  "
            f"{cib_index:9d}  {0.5 * cib_rho**2:8.3f}  {cib_rho:8.4f}",
            flush=True,
        )

    target_z_array = np.stack(target_whitened)
    realized_z_array = np.stack(realized_whitened)
    cib_z_array = np.stack(cib_whitened)
    target_rho_array = np.linalg.norm(target_z_array, axis=1)
    realized_rho_array = np.linalg.norm(realized_z_array, axis=1)
    target_energy = 0.5 * target_rho_array**2
    realized_energy = 0.5 * realized_rho_array**2
    cib_rho_array = np.linalg.norm(cib_z_array, axis=1)
    cib_energy = 0.5 * cib_rho_array**2
    summarize("Gaussian statistic draws", target_rho_array, target_energy)
    summarize("Synthesized maps", realized_rho_array, realized_energy)
    summarize("Real CIB maps", cib_rho_array, cib_energy)

    active_indices = np.flatnonzero(active.detach().cpu().numpy())
    mean_contribution = np.mean(realized_z_array**2, axis=0)
    order = np.argsort(mean_contribution)[::-1]
    n_top = min(args.top_statistics, order.size)
    if n_top:
        print("largest mean synthesized-map contributions to rho^2:", flush=True)
        print("rank  active_position  full_statistic_index  mean_z_squared", flush=True)
        for rank, position in enumerate(order[:n_top], start=1):
            print(
                f"{rank:4d}  {position:15d}  {active_indices[position]:20d}  "
                f"{mean_contribution[position]:14.6f}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "sampled_synthesis_likelihood.png"
    result_path = args.output_dir / "sampled_synthesis_likelihood.npz"
    summary_path = args.output_dir / "sampled_synthesis_likelihood.json"
    indices = np.arange(sampled_statistics.shape[0])
    reference_rho = math.sqrt(int(active.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(indices, target_rho_array, "o-", label="sampled statistic target")
    axes[0].plot(indices, realized_rho_array, "s-", label="synthesized map")
    axes[0].plot(indices, cib_rho_array, "^-", label="real CIB map")
    axes[0].axhline(reference_rho, color="black", ls="--", label=r"$\sqrt{D}$")
    axes[0].set_xlabel("validation sample index")
    axes[0].set_ylabel(r"$\rho$")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    width = 0.26
    axes[1].bar(indices - width, target_energy, width, label="target")
    axes[1].bar(indices, realized_energy, width, label="synthesized map")
    axes[1].bar(indices + width, cib_energy, width, label="real CIB map")
    axes[1].axhline(int(active.sum()) / 2, color="black", ls="--", label=r"$D/2$")
    axes[1].set_xlabel("validation sample index")
    axes[1].set_ylabel(r"$U_{\rm like}=\rho^2/2$")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    np.savez_compressed(
        result_path,
        target_rho=target_rho_array,
        synthesized_map_rho=realized_rho_array,
        cib_indices=cib_indices,
        real_cib_rho=cib_rho_array,
        target_likelihood_energy=target_energy,
        synthesized_map_likelihood_energy=realized_energy,
        real_cib_likelihood_energy=cib_energy,
        target_whitened_discrepancy=target_z_array,
        synthesized_map_whitened_discrepancy=realized_z_array,
        real_cib_whitened_discrepancy=cib_z_array,
        active_statistic_mask=active.detach().cpu().numpy(),
        mean_rho_squared_contribution=mean_contribution,
    )
    summary = {
        "synthesis_npz": str(args.synthesis_npz),
        "covariance_npz": str(args.covariance_npz),
        "covariance_json": str(args.covariance_json),
        "cib_bank": str(args.cib_bank),
        "cib_seed": args.cib_seed,
        "cib_indices": cib_indices.tolist(),
        "active_dimension": int(active.sum()),
        "expected_rho": reference_rho,
        "target_rho": target_rho_array.tolist(),
        "synthesized_map_rho": realized_rho_array.tolist(),
        "real_cib_rho": cib_rho_array.tolist(),
        "target_rho_mean": float(target_rho_array.mean()),
        "synthesized_map_rho_mean": float(realized_rho_array.mean()),
        "real_cib_rho_mean": float(cib_rho_array.mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={plot_path}", flush=True)
    print(f"saved={result_path}", flush=True)
    print(f"saved={summary_path}", flush=True)


if __name__ == "__main__":
    main()
