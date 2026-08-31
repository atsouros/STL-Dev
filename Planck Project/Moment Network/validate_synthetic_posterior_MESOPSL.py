#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PLANCK_PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PLANCK_PROJECT_DIR.parent
REPO_ROOT = Path(os.environ.get("STL_DEV_ROOT", str(DEFAULT_REPO_ROOT))).expanduser()
for path in (REPO_ROOT, PLANCK_PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_dataset import (  # noqa: E402
    RunRecord,
    build_joint_pairs,
    center_crop_to_hw,
    discover_runs,
    list_nuisance_paths,
    load_config,
    load_nuisance_bank,
    parse_cross_matrix,
    parse_hw,
    parse_patch_selector,
    str_to_bool,
    synthesize_joint_qu,
)
from train_network import LSSUNet  # noqa: E402


DEFAULT_RESULTS_DIR = Path("/pscratch/sd/a/atsouros/STL/planck_results/version_2")
DEFAULT_NUISANCE_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance")
DEFAULT_SIGNAL_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/signal")
DEFAULT_MODEL_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_training/version_2_qu_splitstd_nll/models")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_validation/synthetic_posterior")


def select_no_bonus_signal_path(signal_dir: Path, pattern: str, label: str) -> Path:
    candidates = sorted(signal_dir.glob(pattern))
    no_bonus = [
        path
        for path in candidates
        if re.search(r"_(?:hr|hm)_\d+\.npy$", path.name) is None
    ]
    if not no_bonus:
        candidate_names = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            f"Could not find no-bonus {label} signal file in {signal_dir}. "
            f"Pattern: {pattern}. Candidates: {candidate_names}"
        )
    canonical = [path for path in no_bonus if path.name.endswith("_v4_10_arcmin.npy")]
    return canonical[0] if canonical else no_bonus[0]


def downsample_by_four(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D intensity map, got {image.shape}")
    h, w = image.shape
    if h % 2 or w % 2:
        raise ValueError(f"Intensity dimensions must be even for 2x2 averaging, got {image.shape}")
    return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def center_crop(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    h, w = image.shape
    if out_h > h or out_w > w:
        raise ValueError(f"Cannot crop intensity map {image.shape} to {out_hw}")
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return image[y0 : y0 + out_h, x0 : x0 + out_w]


def load_intensity_map(
    signal_dir: Path,
    *,
    patch: str,
    intensity_freq: int,
    final_hw: tuple[int, int],
) -> tuple[np.ndarray, Path]:
    path = select_no_bonus_signal_path(
        signal_dir,
        f"patch_{patch}_I{intensity_freq}_*.npy",
        f"I{intensity_freq}",
    )
    raw = np.load(path).astype(np.float64)
    expected_raw_shape = (2 * final_hw[0], 2 * final_hw[1])
    if raw.shape == final_hw:
        image = raw
    elif raw.shape == expected_raw_shape:
        image = center_crop(downsample_by_four(raw), final_hw)
    else:
        raise RuntimeError(
            f"Expected I{intensity_freq} patch {patch} to have shape {final_hw} "
            f"or {expected_raw_shape}, got {raw.shape}"
        )
    return image.astype(np.float32), path


def discover_single_patch_run(results_dir: Path, patch: str) -> RunRecord:
    patch_selector = parse_patch_selector(patch)
    records = discover_runs(results_dir, patch_selector)
    if not records:
        nested_records: list[RunRecord] = []
        visited_dirs: set[Path] = set()
        for npy_path in sorted(results_dir.rglob(f"p{int(patch)}_*.npy")):
            parent = npy_path.parent
            if parent in visited_dirs:
                continue
            visited_dirs.add(parent)
            nested_records.extend(discover_runs(parent, patch_selector))
        records = nested_records

    unique = {record.npy_path.resolve(): record for record in records}
    records = sorted(unique.values(), key=lambda record: str(record.npy_path))
    if len(records) == 1:
        return records[0]

    npy_candidates = sorted(results_dir.rglob(f"*{patch}*.npy"))
    candidate_text = "\n".join(f"  - {path}" for path in npy_candidates[:30])
    if len(npy_candidates) > 30:
        candidate_text += f"\n  ... and {len(npy_candidates) - 30} more"
    if not candidate_text:
        candidate_text = "  (no .npy filenames containing the patch number)"

    if not records:
        raise RuntimeError(
            f"Could not find a component-separation result pair for patch {patch} under {results_dir}.\n"
            f"Expected an .npy named p{int(patch)}_* and a same-stem .json file.\n"
            f"Candidate .npy files containing {patch}:\n{candidate_text}"
        )

    record_text = "\n".join(
        f"  - npy={record.npy_path} json={record.json_path}"
        for record in records
    )
    raise RuntimeError(
        f"Found {len(records)} component-separation result pairs for patch {patch}; expected one:\n{record_text}"
    )


def load_state_with_optional_prefix(checkpoint: dict[str, object], key: str, prefix: str) -> dict[str, torch.Tensor]:
    state = checkpoint.get(key)
    if state is not None:
        return state

    full_state = checkpoint.get("model_state_dict")
    if full_state is None:
        raise RuntimeError(f"Checkpoint does not contain {key} or model_state_dict")
    prefix_dot = f"{prefix}."
    stripped = {
        name[len(prefix_dot) :]: value
        for name, value in full_state.items()
        if name.startswith(prefix_dot)
    }
    if not stripped:
        raise RuntimeError(f"Could not extract {prefix_dot}* parameters from model_state_dict")
    return stripped


def load_joint_model(checkpoint_path: Path, device: torch.device) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    norm = checkpoint.get("normalization")
    if norm is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain normalization statistics")

    stats = {
        "input_mean": np.asarray(norm["input_mean"], dtype=np.float32),
        "input_std": np.asarray(norm["input_std"], dtype=np.float32),
        "target_mean": np.asarray(norm["target_mean"], dtype=np.float32),
        "target_std": np.asarray(norm["target_std"], dtype=np.float32),
    }
    n_input = int(stats["input_mean"].size)
    if n_input not in (2, 3):
        raise RuntimeError(f"Checkpoint input normalization has {n_input} channels; expected 2 or 3")
    if stats["input_std"].shape != (n_input,):
        raise RuntimeError(
            f"Checkpoint normalization field input_std has shape {stats['input_std'].shape}; expected ({n_input},)"
        )
    for key in ("target_mean", "target_std"):
        if stats[key].shape != (2,):
            raise RuntimeError(f"Checkpoint normalization field {key} has shape {stats[key].shape}; expected (2,)")

    model_format = checkpoint.get("model_format", "")
    has_split_std = (
        "std_q_model_state_dict" in checkpoint
        or "std_u_model_state_dict" in checkpoint
        or str(model_format) in {"joint_qu_split_std_nll", "joint_qu_i_split_std_nll"}
    )
    if has_split_std:
        mean_model = LSSUNet(in_channels=n_input, out_channels=2).to(device)
        std_q_model = LSSUNet(in_channels=n_input, out_channels=1).to(device)
        std_u_model = LSSUNet(in_channels=n_input, out_channels=1).to(device)
        mean_model.load_state_dict(load_state_with_optional_prefix(checkpoint, "mean_model_state_dict", "mean_model"))
        std_q_model.load_state_dict(load_state_with_optional_prefix(checkpoint, "std_q_model_state_dict", "std_q_model"))
        std_u_model.load_state_dict(load_state_with_optional_prefix(checkpoint, "std_u_model_state_dict", "std_u_model"))
        mean_model.eval()
        std_q_model.eval()
        std_u_model.eval()
        return {
            "kind": "split_std",
            "label": "splitstd_i" if n_input == 3 else "splitstd",
            "n_input": n_input,
            "mean_model": mean_model,
            "std_q_model": std_q_model,
            "std_u_model": std_u_model,
        }, stats, checkpoint

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is not a recognized joint model. "
            "Expected either split-std keys or legacy model_state_dict."
        )

    if n_input != 2:
        raise RuntimeError(
            f"Legacy four-output checkpoint expects two input channels, but normalization contains {n_input}"
        )
    model = LSSUNet(in_channels=n_input, out_channels=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"kind": "legacy", "label": "legacy", "model": model}, stats, checkpoint


@torch.no_grad()
def infer_joint(
    model_info: dict[str, object],
    observed: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_input = int(stats["input_mean"].size)
    if observed.shape[0] != n_input:
        raise RuntimeError(f"Checkpoint expects {n_input} input channels, got observed shape {observed.shape}")
    input_mean = stats["input_mean"].reshape(n_input, 1, 1)
    input_std = stats["input_std"].reshape(n_input, 1, 1)
    target_mean = stats["target_mean"].reshape(2, 1, 1)
    target_std = stats["target_std"].reshape(2, 1, 1)

    observed_norm = (observed - input_mean) / input_std
    tensor = torch.from_numpy(observed_norm[None].astype(np.float32)).to(device=device)

    if model_info["kind"] == "split_std":
        mean_model = model_info["mean_model"]
        std_q_model = model_info["std_q_model"]
        std_u_model = model_info["std_u_model"]
        mean_norm = mean_model(tensor).detach().cpu().numpy()[0][:2]
        logvar_q = std_q_model(tensor).detach().cpu().numpy()[0][:1]
        logvar_u = std_u_model(tensor).detach().cpu().numpy()[0][:1]
        logvar_norm = np.concatenate([logvar_q, logvar_u], axis=0)
    elif model_info["kind"] == "legacy":
        output = model_info["model"](tensor).detach().cpu().numpy()[0]
        mean_norm = output[:2]
        logvar_norm = output[2:4]
    else:
        raise RuntimeError(f"Unsupported model kind {model_info['kind']!r}")

    logvar_norm = np.clip(logvar_norm, -12.0, 12.0)
    std_norm = np.exp(0.5 * logvar_norm)
    mean = mean_norm * target_std + target_mean
    std = std_norm * target_std
    return mean.astype(np.float32), std.astype(np.float32), logvar_norm.astype(np.float32)


def robust_hist_limits(values: np.ndarray, q: float = 99.5) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-5.0, 5.0)
    lim = float(np.nanpercentile(np.abs(finite), q))
    if not np.isfinite(lim) or lim <= 0:
        lim = 5.0
    lim = max(3.0, min(12.0, lim))
    return (-lim, lim)


def save_histogram_figure(out_path: Path, *, residual_z: np.ndarray, patch: str, checkpoint_path: Path) -> None:
    labels = ["Q", "U"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), constrained_layout=True)
    for ax, label, values in zip(axes, labels, residual_z):
        flat = values.ravel()
        flat = flat[np.isfinite(flat)]
        hist_min, hist_max = robust_hist_limits(flat)
        ax.hist(flat, bins=120, range=(hist_min, hist_max), density=True, color="#3b6ea8", alpha=0.82)
        x_grid = np.linspace(hist_min, hist_max, 600)
        normal_pdf = np.exp(-0.5 * x_grid**2) / np.sqrt(2.0 * np.pi)
        ax.plot(x_grid, normal_pdf, color="black", linewidth=1.6, label=r"$\mathcal{N}(0,1)$")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(
            f"{label}: (truth - posterior mean) / posterior std "
            f"| mean={np.mean(flat):.3f}, std={np.std(flat):.3f}"
        )
        ax.set_xlabel("standardized residual")
        ax.set_ylabel("density")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Synthetic posterior calibration check, patch {patch}\n{checkpoint_path.name}", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_map_figure(
    out_path: Path,
    *,
    truth: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    residual_z: np.ndarray,
    patch: str,
) -> None:
    labels = ["Q", "U"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.0), constrained_layout=True)
    for row, label in enumerate(labels):
        fields = [
            (f"{label} truth", truth[row], "coolwarm"),
            (f"{label} data", observed[row], "coolwarm"),
            (f"{label} mean", mean[row], "coolwarm"),
            (f"{label} std", std[row], "viridis"),
            (f"{label} z", residual_z[row], "coolwarm"),
        ]
        for col, (title, arr, cmap) in enumerate(fields):
            ax = axes[row, col]
            if title.endswith("std"):
                vmin, vmax = 0.0, float(np.nanpercentile(arr, 99.0))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = float(np.nanmax(arr)) if np.size(arr) else 1.0
            elif title.endswith("z"):
                vmax = float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 99.0))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 5.0
                vmax = min(max(vmax, 3.0), 8.0)
                vmin = -vmax
            else:
                combined = np.concatenate([truth[row].ravel(), observed[row].ravel(), mean[row].ravel()])
                vmax = float(np.nanpercentile(np.abs(combined[np.isfinite(combined)]), 99.0))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                vmin = -vmax
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Synthetic posterior calibration maps, patch {patch}", fontsize=13)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize one Q/U patch, infer with a trained joint moment network, and histogram z residuals.")
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR))))
    parser.add_argument("--nuisance-dir", type=Path, default=Path(os.environ.get("PLANCK_NUISANCE_DIR", str(DEFAULT_NUISANCE_DIR))))
    parser.add_argument("--signal-dir", type=Path, default=Path(os.environ.get("PLANCK_SIGNAL_DIR", str(DEFAULT_SIGNAL_DIR))))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ["CHECKPOINT_PATH"]) if os.environ.get("CHECKPOINT_PATH") else None)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--patch", default=os.environ.get("PATCH", "3"))
    parser.add_argument("--freq", type=int, default=int(os.environ.get("FREQ", "353")))
    parser.add_argument("--intensity-freq", type=int, default=int(os.environ.get("INTENSITY_FREQ", "857")))
    parser.add_argument("--nuisance-version", default=os.environ.get("PLANCK_NUISANCE_VERSION", "v4_10_arcmin"))
    parser.add_argument("--synthesis-pbc", type=str_to_bool, default=str_to_bool(os.environ.get("SYNTHESIS_PBC", "1")))
    parser.add_argument("--synthesis-compute-ps", type=str_to_bool, default=str_to_bool(os.environ.get("SYNTHESIS_COMPUTE_PS", "0")))
    parser.add_argument("--synthesis-running-shape", type=parse_hw, default=parse_hw(os.environ.get("SYNTHESIS_RUNNING_SHAPE", "512")))
    parser.add_argument("--cross-matrix", type=parse_cross_matrix, default=parse_cross_matrix(os.environ.get("CROSS_MATRIX", "1,1;0,1")))
    parser.add_argument("--synthesis-max-iter", type=int, default=int(os.environ.get("SYNTHESIS_MAX_ITER", "500")))
    parser.add_argument("--synthesis-lr", type=float, default=float(os.environ.get("SYNTHESIS_LR", "1.0")))
    parser.add_argument("--synthesis-history-size", type=int, default=int(os.environ.get("SYNTHESIS_HISTORY_SIZE", "40")))
    parser.add_argument("--print-every-synthesis", type=int, default=int(os.environ.get("PRINT_EVERY_SYNTHESIS", "1")))
    parser.add_argument("--verbose-synthesis", type=str_to_bool, default=str_to_bool(os.environ.get("VERBOSE_SYNTHESIS", "1")))
    parser.add_argument("--synthesis-seed", type=int, default=int(os.environ.get("SYNTHESIS_SEED", "10007")))
    parser.add_argument("--noise-seed", type=int, default=int(os.environ.get("NOISE_SEED", "20007")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    parser.add_argument("--save-maps", type=str_to_bool, default=str_to_bool(os.environ.get("SAVE_MAPS", "1")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch = str(int(args.patch))
    args.results_dir = args.results_dir.expanduser()
    args.nuisance_dir = args.nuisance_dir.expanduser()
    args.signal_dir = args.signal_dir.expanduser()
    args.model_dir = args.model_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser() if args.checkpoint is not None else args.model_dir / "moment_network_joint_best.pth"

    if args.device.strip():
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    record = discover_single_patch_run(args.results_dir, patch)

    print(f"patch={patch}", flush=True)
    print(f"device={device}", flush=True)
    print(f"results={record.npy_path}", flush=True)
    print(f"metadata={record.json_path}", flush=True)
    print(f"checkpoint={checkpoint_path}", flush=True)

    config = load_config(record.json_path)
    recovered_qu = np.load(record.npy_path).astype(np.float64)
    if recovered_qu.ndim != 3 or recovered_qu.shape[0] != 2:
        raise RuntimeError(f"Expected recovered Q/U map with shape (2,H,W), got {recovered_qu.shape}")
    final_hw = tuple(int(v) for v in recovered_qu.shape[-2:])

    q_noise_paths = list_nuisance_paths(args.nuisance_dir, patch, "Q", args.freq, args.nuisance_version)
    u_noise_paths = list_nuisance_paths(args.nuisance_dir, patch, "U", args.freq, args.nuisance_version)
    q_noise_bank = load_nuisance_bank(q_noise_paths, crop_size=None)
    u_noise_bank = load_nuisance_bank(u_noise_paths, crop_size=None)
    if tuple(q_noise_bank.shape[-2:]) != final_hw or tuple(u_noise_bank.shape[-2:]) != final_hw:
        raise RuntimeError(f"Nuisance shape mismatch: Q={q_noise_bank.shape[-2:]} U={u_noise_bank.shape[-2:]} expected={final_hw}")

    qu_synth_full = synthesize_joint_qu(
        recovered_qu,
        config,
        device=device,
        seed=args.synthesis_seed + int(patch) * 10,
        total_syntheses=1,
        synthesis_batch_size=1,
        synthesis_pbc=args.synthesis_pbc,
        running_shape=args.synthesis_running_shape,
        compute_ps=args.synthesis_compute_ps,
        cross_matrix=args.cross_matrix,
        max_iter=args.synthesis_max_iter,
        lr=args.synthesis_lr,
        history_size=args.synthesis_history_size,
        print_iter=args.print_every_synthesis,
        verbose=args.verbose_synthesis,
    )

    rng = np.random.default_rng(args.noise_seed + int(patch))
    q_clean, q_data, u_clean, u_data = build_joint_pairs(
        qu_synth_full[:, 0],
        qu_synth_full[:, 1],
        q_noise_bank,
        u_noise_bank,
        n_augmentations=1,
        shift_step=16,
        final_hw=final_hw,
        rng=rng,
    )
    truth = np.stack([q_clean[0], u_clean[0]], axis=0).astype(np.float32)
    observed_qu = np.stack([q_data[0], u_data[0]], axis=0).astype(np.float32)

    model_info, stats, checkpoint = load_joint_model(checkpoint_path, device)
    n_input = int(stats["input_mean"].size)
    intensity_path: Path | None = None
    if n_input == 3:
        intensity, intensity_path = load_intensity_map(
            args.signal_dir,
            patch=patch,
            intensity_freq=args.intensity_freq,
            final_hw=final_hw,
        )
        observed = np.concatenate([observed_qu, intensity[None]], axis=0)
        print(f"intensity={intensity_path}", flush=True)
    else:
        observed = observed_qu
    print(
        f"loaded_model_kind={model_info['kind']} "
        f"input_channels={n_input} "
        f"model_format={checkpoint.get('model_format')} "
        f"posterior_parameterization={checkpoint.get('posterior_parameterization')}",
        flush=True,
    )
    mean, std, logvar_norm = infer_joint(model_info, observed, stats, device=device)
    residual_z = (truth - mean) / np.maximum(std, 1e-12)

    model_label = str(model_info["label"])
    tag = f"patch_{patch}_seed{args.synthesis_seed}_noise{args.noise_seed}"
    hist_path = args.output_dir / f"{tag}_{model_label}_z_hist.png"
    save_histogram_figure(hist_path, residual_z=residual_z, patch=patch, checkpoint_path=checkpoint_path)
    map_path = args.output_dir / f"{tag}_{model_label}_maps.png"
    if args.save_maps:
        save_map_figure(map_path, truth=truth, observed=observed, mean=mean, std=std, residual_z=residual_z, patch=patch)

    npz_path = args.output_dir / f"{tag}_{model_label}_validation.npz"
    np.savez_compressed(
        npz_path,
        truth=truth,
        observed=observed,
        mean=mean,
        std=std,
        logvar_norm=logvar_norm,
        residual_z=residual_z.astype(np.float32),
        patch=np.asarray(patch),
    )

    summary = {
        "patch": patch,
        "results": str(record.npy_path),
        "metadata": str(record.json_path),
        "checkpoint": str(checkpoint_path),
        "input_channels": n_input,
        "signal_dir": str(args.signal_dir) if n_input == 3 else None,
        "intensity_freq": args.intensity_freq if n_input == 3 else None,
        "intensity_path": str(intensity_path) if intensity_path is not None else None,
        "model_kind": str(model_info["kind"]),
        "model_format": checkpoint.get("model_format"),
        "posterior_parameterization": checkpoint.get("posterior_parameterization"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        "output_histogram": str(hist_path),
        "output_maps": str(map_path) if args.save_maps else None,
        "output_npz": str(npz_path),
        "truth_shape": list(truth.shape),
        "observed_shape": list(observed.shape),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "synthesis_seed": args.synthesis_seed,
        "noise_seed": args.noise_seed,
        "synthesis_pbc": args.synthesis_pbc,
        "synthesis_running_shape": list(args.synthesis_running_shape) if args.synthesis_running_shape is not None else None,
        "synthesis_compute_ps": args.synthesis_compute_ps,
        "synthesis_max_iter": args.synthesis_max_iter,
        "cross_matrix": args.cross_matrix,
        "q_z_mean": float(np.mean(residual_z[0])),
        "q_z_std": float(np.std(residual_z[0])),
        "u_z_mean": float(np.mean(residual_z[1])),
        "u_z_std": float(np.std(residual_z[1])),
    }
    summary_path = args.output_dir / f"{tag}_{model_label}_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")

    print(f"Q z mean={summary['q_z_mean']:.6f} std={summary['q_z_std']:.6f}", flush=True)
    print(f"U z mean={summary['u_z_mean']:.6f} std={summary['u_z_std']:.6f}", flush=True)
    print(f"Saved {hist_path}", flush=True)
    if args.save_maps:
        print(f"Saved {map_path}", flush=True)
    print(f"Saved {npz_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
