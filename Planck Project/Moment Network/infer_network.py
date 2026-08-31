#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
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

from train_network import LSSUNet
from utils import SIGNAL_DIR, _center_crop, _downsample_by_four, _select_no_bonus_signal_path


DEFAULT_MODEL_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_training/version_2_qu_twostage/models")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_inference/version_2_qu_twostage")
DEFAULT_PLOTS_DIR = Path("/pscratch/sd/a/atsouros/STL/mn_plots")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def robust_limits(arrays: list[np.ndarray], q: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lim = float(np.nanpercentile(np.abs(values), q))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return (-lim, lim)


def positive_limits(arrays: list[np.ndarray], q: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    vmax = float(np.nanpercentile(values, q))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(values)) if values.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return (0.0, vmax)


def load_signal_qu(
    signal_dir: Path,
    *,
    patch: str,
    freq: int,
    map_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    q_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_Q{freq}_*.npy", f"Q{freq}")
    u_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_U{freq}_*.npy", f"U{freq}")

    q_raw = np.load(q_path).astype(np.float64)
    u_raw = np.load(u_path).astype(np.float64)
    if q_raw.shape != u_raw.shape:
        raise RuntimeError(f"Q/U raw shape mismatch: Q={q_raw.shape} U={u_raw.shape}")

    expected_raw_shape = (2 * map_size, 2 * map_size)
    if q_raw.shape != expected_raw_shape:
        raise RuntimeError(
            f"Expected raw signal maps to have shape {expected_raw_shape} before downgrading, "
            f"got {q_raw.shape}"
        )

    q_map = _downsample_by_four(q_raw)
    u_map = _downsample_by_four(u_raw)
    q_map = _center_crop(q_map, out_hw=(map_size, map_size))
    u_map = _center_crop(u_map, out_hw=(map_size, map_size))

    observed = np.stack([q_map, u_map], axis=0).astype(np.float32)
    paths = {
        "q_signal": str(q_path),
        "u_signal": str(u_path),
        "raw_shape": list(q_raw.shape),
        "downgraded_shape": list(observed.shape),
        "downgrade": "2x2 block average, matching make_dataset.py downsample_by_four",
    }
    return observed, paths


def load_signal_i(
    signal_dir: Path,
    *,
    patch: str,
    intensity_freq: int,
    map_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    i_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_I{intensity_freq}_*.npy", f"I{intensity_freq}")
    i_raw = np.load(i_path).astype(np.float64)
    expected_raw_shape = (2 * map_size, 2 * map_size)
    if i_raw.shape != expected_raw_shape:
        raise RuntimeError(
            f"Expected raw I{intensity_freq} map to have shape {expected_raw_shape} before downgrading, got {i_raw.shape}"
        )
    i_map = _downsample_by_four(i_raw)
    i_map = _center_crop(i_map, out_hw=(map_size, map_size))
    return i_map.astype(np.float32), {"i_signal": str(i_path), "i_raw_shape": list(i_raw.shape)}


def load_model(checkpoint_path: Path, device: torch.device, expected_stokes: str) -> tuple[LSSUNet, dict[str, np.ndarray], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    parameterization = checkpoint.get("posterior_parameterization")
    if parameterization != "gaussian_logvar":
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has posterior_parameterization={parameterization!r}. "
            "This inference script expects separate Gaussian log-variance Q/U checkpoints."
        )
    if str(checkpoint.get("stokes", "")).upper() != expected_stokes.upper():
        raise RuntimeError(f"Checkpoint {checkpoint_path} is not a {expected_stokes} checkpoint")

    model = LSSUNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm = checkpoint.get("normalization")
    if norm is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain normalization statistics")
    stats = {
        "input_mean": np.asarray(norm["input_mean"], dtype=np.float32),
        "input_std": np.asarray(norm["input_std"], dtype=np.float32),
        "target_mean": np.asarray(norm["target_mean"], dtype=np.float32),
        "target_std": np.asarray(norm["target_std"], dtype=np.float32),
    }
    return model, stats, checkpoint


def load_joint_model(checkpoint_path: Path, device: torch.device) -> tuple[object, dict[str, np.ndarray], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    parameterization = checkpoint.get("posterior_parameterization")
    model_format = checkpoint.get("model_format")

    norm = checkpoint.get("normalization")
    if norm is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain normalization statistics")
    stats = {
        "input_mean": np.asarray(norm["input_mean"], dtype=np.float32),
        "input_std": np.asarray(norm["input_std"], dtype=np.float32),
        "target_mean": np.asarray(norm["target_mean"], dtype=np.float32),
        "target_std": np.asarray(norm["target_std"], dtype=np.float32),
    }

    if model_format == "joint_qu_two_stage_mean_variance":
        for key in ("input_mean", "input_std", "target_mean", "target_std"):
            if stats[key].size != 2:
                raise RuntimeError(f"Two-stage Q/U checkpoint {checkpoint_path} has {key} shape {stats[key].shape}; expected 2")

        mean_model = LSSUNet(in_channels=2, out_channels=2).to(device)
        variance_model = LSSUNet(in_channels=2, out_channels=2).to(device)
        mean_model.load_state_dict(checkpoint["mean_model_state_dict"])
        variance_model.load_state_dict(checkpoint["variance_model_state_dict"])
        mean_model.eval()
        variance_model.eval()
        return {"kind": "two_stage_qu", "mean_model": mean_model, "variance_model": variance_model}, stats, checkpoint

    if model_format == "joint_qu_i_two_stage_mean_variance" or parameterization == "two_stage_residual_logvar":
        for key in ("input_mean", "input_std"):
            if stats[key].size != 3:
                raise RuntimeError(f"Two-stage Q/U/I checkpoint {checkpoint_path} has {key} shape {stats[key].shape}; expected 3")
        for key in ("target_mean", "target_std"):
            if stats[key].size != 2:
                raise RuntimeError(f"Two-stage Q/U/I checkpoint {checkpoint_path} has {key} shape {stats[key].shape}; expected 2")

        mean_model = LSSUNet(in_channels=3, out_channels=2).to(device)
        variance_model = LSSUNet(in_channels=3, out_channels=2).to(device)
        mean_model.load_state_dict(checkpoint["mean_model_state_dict"])
        variance_model.load_state_dict(checkpoint["variance_model_state_dict"])
        mean_model.eval()
        variance_model.eval()
        return {"kind": "two_stage", "mean_model": mean_model, "variance_model": variance_model}, stats, checkpoint

    if parameterization is None:
        print(
            f"Warning: checkpoint {checkpoint_path} has no posterior_parameterization metadata; "
            "treating it as a legacy joint Gaussian log-variance checkpoint.",
            flush=True,
        )
    elif parameterization != "gaussian_logvar":
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has posterior_parameterization={parameterization!r}. "
            "Joint inference expects the old Gaussian log-variance Q/U checkpoint."
        )

    model = LSSUNet(in_channels=2, out_channels=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for key, value in stats.items():
        if value.size != 2:
            raise RuntimeError(
                f"Joint checkpoint {checkpoint_path} has normalization field {key} with shape {value.shape}; "
                "expected two Q/U values."
            )
    return {"kind": "legacy", "model": model}, stats, checkpoint


@torch.no_grad()
def infer_single_map(
    model: LSSUNet,
    observed: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_mean = float(stats["input_mean"][0])
    input_std = float(stats["input_std"][0])
    target_mean = float(stats["target_mean"][0])
    target_std = float(stats["target_std"][0])

    observed_norm = (observed - input_mean) / input_std
    tensor = torch.from_numpy(observed_norm[None, None].astype(np.float32)).to(device=device)
    output = model(tensor).detach().cpu().numpy()[0]

    mean_norm = output[0]
    logvar_norm = np.clip(output[1], -12.0, 12.0)
    std_norm = np.exp(0.5 * logvar_norm)
    mean = mean_norm * target_std + target_mean
    std = std_norm * target_std
    return mean.astype(np.float32), std.astype(np.float32), logvar_norm.astype(np.float32)


@torch.no_grad()
def infer_joint_map(
    model: LSSUNet,
    observed: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_mean = stats["input_mean"].reshape(2, 1, 1)
    input_std = stats["input_std"].reshape(2, 1, 1)
    target_mean = stats["target_mean"].reshape(2, 1, 1)
    target_std = stats["target_std"].reshape(2, 1, 1)

    observed_norm = (observed - input_mean) / input_std
    tensor = torch.from_numpy(observed_norm[None].astype(np.float32)).to(device=device)
    output = model(tensor).detach().cpu().numpy()[0]

    mean_norm = output[:2]
    logvar_norm = np.clip(output[2:4], -12.0, 12.0)
    std_norm = np.exp(0.5 * logvar_norm)
    mean = mean_norm * target_std + target_mean
    std = std_norm * target_std
    return mean.astype(np.float32), std.astype(np.float32), logvar_norm.astype(np.float32)


@torch.no_grad()
def infer_two_stage_joint_map(
    models: dict[str, object],
    observed: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_input = int(stats["input_mean"].size)
    if observed.shape[0] != n_input:
        raise RuntimeError(f"Two-stage checkpoint expects {n_input} input channels, got observed shape {observed.shape}")
    input_mean = stats["input_mean"].reshape(n_input, 1, 1)
    input_std = stats["input_std"].reshape(n_input, 1, 1)
    target_mean = stats["target_mean"].reshape(2, 1, 1)
    target_std = stats["target_std"].reshape(2, 1, 1)

    observed_norm = (observed - input_mean) / input_std
    tensor = torch.from_numpy(observed_norm[None].astype(np.float32)).to(device=device)
    mean_norm = models["mean_model"](tensor).detach().cpu().numpy()[0]
    logvar_norm = np.clip(models["variance_model"](tensor).detach().cpu().numpy()[0], -12.0, 12.0)
    std_norm = np.exp(0.5 * logvar_norm)
    mean = mean_norm * target_std + target_mean
    std = std_norm * target_std
    return mean.astype(np.float32), std.astype(np.float32), logvar_norm.astype(np.float32)


def save_figure(
    out_path: Path,
    *,
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    patch: str,
    freq: int,
    network_mode: str,
) -> None:
    labels = ["Q", "U"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)

    for row, label in enumerate(labels):
        map_vmin, map_vmax = robust_limits([observed[row], mean[row]])
        std_vmin, std_vmax = positive_limits([std[row]])
        input_title = f"{label}{freq} data d" if network_mode == "separate" else f"{label}{freq} input"
        panels = [
            (input_title, observed[row], "coolwarm", map_vmin, map_vmax),
            (f"{label} posterior mean", mean[row], "coolwarm", map_vmin, map_vmax),
            (f"{label} posterior std", std[row], "viridis", std_vmin, std_vmax),
        ]
        for col, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.82)

    if network_mode == "separate":
        fig.suptitle(f"Separate Q/U moment-network inference for patch {patch}", fontsize=14)
    else:
        fig.suptitle(f"Moment-network inference for patch {patch}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_std_check_figure(out_path: Path, *, std: np.ndarray, patch: str) -> None:
    diff = std[1] - std[0]
    std_vmin, std_vmax = positive_limits([std[0], std[1]])
    diff_vmin, diff_vmax = robust_limits([diff])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = [
        ("Q posterior std", std[0], "viridis", std_vmin, std_vmax),
        ("U posterior std", std[1], "viridis", std_vmin, std_vmax),
        ("U std - Q std", diff, "coolwarm", diff_vmin, diff_vmax),
    ]
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle(f"Posterior std channel check for patch {patch}", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def channel_diagnostics(mean: np.ndarray, std: np.ndarray, logvar_norm: np.ndarray) -> dict[str, float]:
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        av = np.asarray(a).ravel()
        bv = np.asarray(b).ravel()
        if np.std(av) == 0 or np.std(bv) == 0:
            return float("nan")
        return float(np.corrcoef(av, bv)[0, 1])

    return {
        "mean_q_u_corr": corr(mean[0], mean[1]),
        "std_q_u_corr": corr(std[0], std[1]),
        "std_u_minus_q_abs_mean": float(np.mean(np.abs(std[1] - std[0]))),
        "std_u_minus_q_abs_max": float(np.max(np.abs(std[1] - std[0]))),
        "logvar_norm_q_u_corr": corr(logvar_norm[0], logvar_norm[1]),
        "logvar_norm_u_minus_q_abs_mean": float(np.mean(np.abs(logvar_norm[1] - logvar_norm[0]))),
        "logvar_norm_u_minus_q_abs_max": float(np.max(np.abs(logvar_norm[1] - logvar_norm[0]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained moment network(s) on a Planck signal patch.")
    parser.add_argument("--patch", default=os.environ.get("PATCH", "3"))
    parser.add_argument("--freq", type=int, default=int(os.environ.get("FREQ", "353")))
    parser.add_argument("--intensity-freq", type=int, default=int(os.environ.get("INTENSITY_FREQ", "857")))
    parser.add_argument("--map-size", type=int, default=int(os.environ.get("MAP_SIZE", "384")))
    parser.add_argument("--signal-dir", type=Path, default=Path(os.environ.get("PLANCK_SIGNAL_DIR", str(SIGNAL_DIR))))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))))
    parser.add_argument("--network-mode", choices=("separate", "joint"), default=os.environ.get("NETWORK_MODE", "separate"))
    parser.add_argument("--q-model", type=Path, default=Path(os.environ["Q_MODEL_PATH"]) if os.environ.get("Q_MODEL_PATH") else None)
    parser.add_argument("--u-model", type=Path, default=Path(os.environ["U_MODEL_PATH"]) if os.environ.get("U_MODEL_PATH") else None)
    parser.add_argument("--joint-model", type=Path, default=Path(os.environ["JOINT_MODEL_PATH"]) if os.environ.get("JOINT_MODEL_PATH") else None)
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--plots-dir", type=Path, default=Path(os.environ.get("PLOTS_DIR", str(DEFAULT_PLOTS_DIR))))
    parser.add_argument("--output-tag", default=os.environ.get("OUTPUT_TAG", ""))
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    parser.add_argument("--save-std-check", action="store_true", default=env_flag("SAVE_STD_CHECK", False))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch = str(int(args.patch))
    signal_dir = args.signal_dir.expanduser()
    model_dir = args.model_dir.expanduser()
    q_model_path = args.q_model.expanduser() if args.q_model is not None else model_dir / "moment_network_q_best.pth"
    u_model_path = args.u_model.expanduser() if args.u_model is not None else model_dir / "moment_network_u_best.pth"
    joint_model_path = args.joint_model.expanduser() if args.joint_model is not None else model_dir / "moment_network_joint_best.pth"
    if args.network_mode == "separate" and q_model_path == u_model_path:
        raise RuntimeError(
            f"Q and U model paths are identical: {q_model_path}. "
            "The current implementation expects two separate polarization-channel checkpoints."
        )
    output_tag = args.output_tag.strip() or args.network_mode
    output_suffix = f"_{output_tag}" if output_tag else ""

    base_out_dir = args.out_dir.expanduser() / f"patch_{patch}"
    out_dir = base_out_dir / output_tag if output_tag else base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir.expanduser()
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.device.strip():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observed, signal_paths = load_signal_qu(signal_dir, patch=patch, freq=args.freq, map_size=args.map_size)
    if observed.shape != (2, args.map_size, args.map_size):
        raise RuntimeError(f"Expected observed shape (2, {args.map_size}, {args.map_size}), got {observed.shape}")

    q_checkpoint = None
    u_checkpoint = None
    joint_checkpoint = None
    if args.network_mode == "separate":
        q_model, q_stats, q_checkpoint = load_model(q_model_path, device=device, expected_stokes="Q")
        u_model, u_stats, u_checkpoint = load_model(u_model_path, device=device, expected_stokes="U")
        mean_q, std_q, logvar_q = infer_single_map(q_model, observed[0], q_stats, device=device)
        mean_u, std_u, logvar_u = infer_single_map(u_model, observed[1], u_stats, device=device)
        mean = np.stack([mean_q, mean_u], axis=0)
        std = np.stack([std_q, std_u], axis=0)
        logvar_norm = np.stack([logvar_q, logvar_u], axis=0)
        network_architecture = "two independent scalar moment networks: Q data -> Q posterior moments, U data -> U posterior moments"
    else:
        joint_model, joint_stats, joint_checkpoint = load_joint_model(joint_model_path, device=device)
        if joint_model["kind"] == "two_stage":
            i_map, i_paths = load_signal_i(signal_dir, patch=patch, intensity_freq=args.intensity_freq, map_size=args.map_size)
            signal_paths.update(i_paths)
            observed_joint = np.concatenate([observed, i_map[None]], axis=0)
            mean, std, logvar_norm = infer_two_stage_joint_map(joint_model, observed_joint, joint_stats, device=device)
            network_architecture = "joint Q/U/I two-stage moment network: (Q data, U data, I data) -> Q/U posterior mean and residual variance"
        elif joint_model["kind"] == "two_stage_qu":
            mean, std, logvar_norm = infer_two_stage_joint_map(joint_model, observed, joint_stats, device=device)
            network_architecture = "joint Q/U two-stage moment network: old-style (Q data, U data) -> Q/U posterior mean, then frozen-mean residual variance"
        else:
            mean, std, logvar_norm = infer_joint_map(joint_model["model"], observed, joint_stats, device=device)
            network_architecture = "old joint Q/U moment network: (Q data, U data) -> Q/U posterior moments"

    npz_path = out_dir / f"patch_{patch}{output_suffix}_moment_network_inference.npz"
    np.savez_compressed(
        npz_path,
        observed=observed,
        mean=mean,
        std=std,
        logvar_norm=logvar_norm,
        patch=np.asarray(patch),
        freq=np.asarray(args.freq),
    )
    np.save(out_dir / f"patch_{patch}{output_suffix}_mean_qu.npy", mean)
    np.save(out_dir / f"patch_{patch}{output_suffix}_std_qu.npy", std)
    np.save(out_dir / f"patch_{patch}{output_suffix}_mean_q.npy", mean[0])
    np.save(out_dir / f"patch_{patch}{output_suffix}_mean_u.npy", mean[1])
    np.save(out_dir / f"patch_{patch}{output_suffix}_std_q.npy", std[0])
    np.save(out_dir / f"patch_{patch}{output_suffix}_std_u.npy", std[1])

    fig_path = plots_dir / f"patch_{patch}{output_suffix}_moment_network_inference.png"
    save_figure(fig_path, observed=observed, mean=mean, std=std, patch=patch, freq=args.freq, network_mode=args.network_mode)
    std_check_fig_path = None
    if args.save_std_check:
        std_check_fig_path = plots_dir / f"patch_{patch}{output_suffix}_posterior_std_channel_check.png"
        save_std_check_figure(std_check_fig_path, std=std, patch=patch)

    summary = {
        "patch": patch,
        "freq": args.freq,
        "intensity_freq": args.intensity_freq,
        "network_mode": args.network_mode,
        "output_tag": output_tag,
        "network_architecture": network_architecture,
        "posterior_parameterization": "per-pixel Gaussian log-variance",
        "signal_dir": str(signal_dir),
        "signal_paths": signal_paths,
        "q_model": str(q_model_path) if args.network_mode == "separate" else None,
        "u_model": str(u_model_path) if args.network_mode == "separate" else None,
        "joint_model": str(joint_model_path) if args.network_mode == "joint" else None,
        "q_checkpoint_epoch": q_checkpoint.get("epoch") if q_checkpoint is not None else None,
        "u_checkpoint_epoch": u_checkpoint.get("epoch") if u_checkpoint is not None else None,
        "joint_checkpoint_epoch": joint_checkpoint.get("epoch") if joint_checkpoint is not None else None,
        "q_checkpoint_best_val_loss": q_checkpoint.get("best_val_loss") if q_checkpoint is not None else None,
        "u_checkpoint_best_val_loss": u_checkpoint.get("best_val_loss") if u_checkpoint is not None else None,
        "joint_checkpoint_best_val_loss": joint_checkpoint.get("best_val_loss") if joint_checkpoint is not None else None,
        "device": str(device),
        "observed_shape": list(observed.shape),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "output_npz": str(npz_path),
        "mean_npy": str(out_dir / f"patch_{patch}{output_suffix}_mean_qu.npy"),
        "std_npy": str(out_dir / f"patch_{patch}{output_suffix}_std_qu.npy"),
        "figure": str(fig_path),
        "std_check_figure": str(std_check_fig_path) if std_check_fig_path is not None else None,
        "channel_diagnostics": channel_diagnostics(mean, std, logvar_norm),
    }
    summary_path = out_dir / f"patch_{patch}{output_suffix}_moment_network_inference.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")

    print(f"Saved {npz_path}", flush=True)
    print(f"Saved {fig_path}", flush=True)
    if std_check_fig_path is not None:
        print(f"Saved {std_check_fig_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
