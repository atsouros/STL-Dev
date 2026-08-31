#!/usr/bin/env python3

"""Minimal synthetic posterior calibration check.

Loads synthesized validation maps, runs the trained split-std U-Net checkpoint,
computes z = (truth - posterior_mean) / posterior_std, and saves the requested
histograms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SYNTHESIS_DIR_DEFAULT = Path(
    "/pscratch/sd/a/atsouros/STL/Moment Network/validation_syntheses"
)
DATASET_DIR_DEFAULT = Path(
    "/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"
)
CHECKPOINT_DEFAULT = Path(
    "/pscratch/sd/a/atsouros/STL/moment_network_training/"
    "version_2_qu_i_splitstd_moment_fixedmean/models/moment_network_joint_best.pth"
)
OUTPUT_DIR_DEFAULT = Path(
    "/pscratch/sd/a/atsouros/STL/moment_network_validation/"
    "synthetic_posterior/version_2_qu_i_splitstd_moment_fixedmean"
)
SIGNAL_DIR_DEFAULT = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/signal")
PATCH_LIST_DEFAULT = "all"
CHANNELS = ("Q", "U")


class ConvGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        if out_ch % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(groups, out_ch, eps=1e-5, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LSSUNet(nn.Module):
    """Architecture needed to load the trained checkpoint weights."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = ConvGN(in_channels, 16)
        self.p1 = nn.AvgPool2d(2)
        self.c2 = ConvGN(16, 32)
        self.p2 = nn.AvgPool2d(2)
        self.c3 = ConvGN(32, 64)
        self.p3 = nn.AvgPool2d(2)
        self.c4 = ConvGN(64, 64)
        self.p4 = nn.AvgPool2d(2)
        self.c5 = ConvGN(64, 64)
        self.c6 = ConvGN(128, 64)
        self.c7 = ConvGN(128, 64)
        self.c8 = ConvGN(96, 32)
        self.c9 = ConvGN(48, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.c1(x)
        b2 = self.c2(self.p1(b1))
        b3 = self.c3(self.p2(b2))
        b4 = self.c4(self.p3(b3))
        b5 = self.c5(self.p4(b4))
        u6 = F.interpolate(b5, size=b4.shape[-2:], mode="nearest")
        b6 = self.c6(torch.cat([b4, u6], dim=1))
        u7 = F.interpolate(b6, size=b3.shape[-2:], mode="nearest")
        b7 = self.c7(torch.cat([b3, u7], dim=1))
        u8 = F.interpolate(b7, size=b2.shape[-2:], mode="nearest")
        b8 = self.c8(torch.cat([b2, u8], dim=1))
        u9 = F.interpolate(b8, size=b1.shape[-2:], mode="nearest")
        b9 = self.c9(torch.cat([b1, u9], dim=1))
        return self.final(b9)


def parse_patch_list(text: str) -> list[int]:
    if text.strip().lower() in {"", "all", "*"}:
        return []
    patches: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            patches.update(range(int(left), int(right) + 1))
        else:
            patches.add(int(item))
    return sorted(patches)


def downsample_2x2(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def center_crop(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    out_h, out_w = out_hw
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return image[y0 : y0 + out_h, x0 : x0 + out_w]


def load_intensity(
    signal_dir: Path,
    patch: int,
    out_hw: tuple[int, int],
    freq: int,
) -> np.ndarray:
    candidates = sorted(signal_dir.glob(f"patch_{patch}_I{freq}_*.npy"))
    candidates = [
        p for p in candidates if re.search(r"_(?:hr|hm)_\d+\.npy$", p.name) is None
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No I{freq} map found for patch {patch} in {signal_dir}"
        )
    canonical = [p for p in candidates if p.name.endswith("_v4_10_arcmin.npy")]
    path = canonical[0] if canonical else candidates[0]
    image = np.load(path).astype(np.float32)
    if image.shape == out_hw:
        return image
    if image.shape == (2 * out_hw[0], 2 * out_hw[1]):
        return center_crop(downsample_2x2(image), out_hw).astype(np.float32)
    raise RuntimeError(f"{path}: unexpected intensity shape {image.shape}")


def load_syntheses(
    synthesis_dir: Path,
    patches: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    truths = []
    observed = []
    found = []
    for patch in patches:
        path = synthesis_dir / f"patch_{patch}_validation_synthesis.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing synthesized validation file: {path}")
        with np.load(path, allow_pickle=False) as data:
            truth = data["truth"].astype(np.float32)
            obs = data["observed_qu"].astype(np.float32)
        if truth.shape[0] != 2 or obs.shape != truth.shape:
            raise RuntimeError(f"{path}: bad shapes truth={truth.shape}, obs={obs.shape}")
        truths.append(truth)
        observed.append(obs)
        found.append(patch)
    return np.stack(truths), np.stack(observed), found


def patch_from_dataset_name(path: Path) -> int:
    match = re.match(r"p(\d+)_", path.name)
    if match is None:
        raise RuntimeError(f"Could not infer patch number from dataset file {path}")
    return int(match.group(1))


def patch_from_synthesis_name(path: Path) -> int:
    match = re.fullmatch(r"patch_(\d+)_validation_synthesis\.npz", path.name)
    if match is None:
        raise RuntimeError(f"Could not infer patch number from synthesis file {path}")
    return int(match.group(1))


def select_original_dataset_files(dataset_dir: Path, patches: list[int]) -> list[Path]:
    patch_filter = set(patches)
    files = []
    for path in dataset_dir.glob("*_moment_dataset_*.npz"):
        patch = patch_from_dataset_name(path)
        if patch_filter and patch not in patch_filter:
            continue
        files.append(path)
    files.sort(key=patch_from_dataset_name)
    if not files:
        selection = "all patches" if not patch_filter else sorted(patch_filter)
        raise FileNotFoundError(f"No original dataset files found in {dataset_dir} for {selection}")
    return files


def select_synthesis_patches(synthesis_dir: Path, patches: list[int]) -> list[int]:
    if patches:
        return sorted(patches)
    files = sorted(
        synthesis_dir.glob("patch_*_validation_synthesis.npz"),
        key=patch_from_synthesis_name,
    )
    if not files:
        raise FileNotFoundError(f"No synthetic validation files found in {synthesis_dir}")
    return [patch_from_synthesis_name(path) for path in files]


def patch_selection_tag(patch_list: str) -> str:
    normalized = patch_list.strip().lower()
    if normalized in {"", "all", "*"}:
        return "all"
    return (
        patch_list.strip()
        .replace(" ", "")
        .replace(",", "_")
        .replace("-", "to")
    )


def augment_image_np(
    image: np.ndarray,
    *,
    n_augmentations: int,
    shift_step: int,
) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []
    arr = np.asarray(image)
    for k in range(4):
        rotated = np.rot90(arr, k=k, axes=(-2, -1))
        for flip in (False, True):
            flipped = np.flip(rotated, axis=-1) if flip else rotated
            for shift_h in range(2):
                for shift_w in range(4):
                    shifted = np.roll(
                        flipped,
                        shift=(shift_h * shift_step, shift_w * shift_step),
                        axis=(-2, -1),
                    )
                    augmented.append(shifted.astype(np.float32, copy=True))
                    if len(augmented) >= n_augmentations:
                        return augmented
    return augmented


def build_dataset_intensity_samples(
    data: np.lib.npyio.NpzFile,
    dataset_path: Path,
    *,
    signal_dir: Path,
    intensity_freq: int,
    map_hw: tuple[int, int],
) -> np.ndarray:
    if "yi" in data:
        return data["yi"].astype(np.float32, copy=False)

    patch = int(np.asarray(data["p"]).item()) if "p" in data else patch_from_dataset_name(dataset_path)
    n_total = int(data["xq"].shape[0])
    n_aug = int(np.asarray(data["naug"]).item()) if "naug" in data else 0
    shift_step = int(np.asarray(data["shift"]).item()) if "shift" in data else 16
    if n_aug <= 0:
        raise RuntimeError(f"{dataset_path}: missing usable naug value")
    if n_total % n_aug != 0:
        raise RuntimeError(f"{dataset_path}: sample count {n_total} not divisible by naug={n_aug}")

    base = load_intensity(signal_dir, patch, map_hw, intensity_freq)
    block = np.stack(
        augment_image_np(base, n_augmentations=n_aug, shift_step=shift_step),
        axis=0,
    )
    n_syntheses = n_total // n_aug
    return np.tile(block[None, ...], (n_syntheses, 1, 1, 1)).reshape(
        n_total,
        map_hw[0],
        map_hw[1],
    ).astype(np.float32)


def checkpoint_dataset_files(
    checkpoint: dict,
    *,
    dataset_dir: Path,
    split: str,
    train_limit: int,
) -> list[Path]:
    if split == "all":
        paths = sorted(
            dataset_dir.glob("*_moment_dataset_*.npz"),
            key=lambda path: patch_from_dataset_name(path),
        )
        if not paths:
            raise FileNotFoundError(
                f"No *_moment_dataset_*.npz files found in {dataset_dir}"
            )
        return paths

    sample_split = checkpoint.get("sample_split")
    if isinstance(sample_split, dict):
        key = "val_indices" if split == "val" else "train_indices"
        indices_by_file = sample_split.get(key)
        if isinstance(indices_by_file, dict) and indices_by_file:
            names = sorted(
                (str(name) for name in indices_by_file.keys()),
                key=lambda name: patch_from_dataset_name(Path(name)),
            )
            if split == "train" and train_limit > 0:
                names = names[:train_limit]
            paths = [dataset_dir / name for name in names]
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Missing dataset files:\n"
                    + "\n".join(str(path) for path in missing[:20])
                )
            return paths

    key = "val_files" if split == "val" else "train_files"
    names = checkpoint.get(key)
    if not isinstance(names, list) or not names:
        raise RuntimeError(f"Checkpoint does not contain non-empty {key}")
    if split == "train" and train_limit > 0:
        names = names[:train_limit]
    paths = [dataset_dir / str(name) for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files:\n" + "\n".join(str(path) for path in missing[:20])
        )
    return paths


def checkpoint_sample_indices(
    checkpoint: dict,
    *,
    split: str,
) -> dict[str, np.ndarray] | None:
    if split == "all":
        return None
    sample_split = checkpoint.get("sample_split")
    if not isinstance(sample_split, dict):
        return None
    key = "val_indices" if split == "val" else "train_indices"
    indices_by_file = sample_split.get(key)
    if not isinstance(indices_by_file, dict):
        return None
    return {
        str(name): np.asarray(indices, dtype=np.int64)
        for name, indices in indices_by_file.items()
    }


def norm_vector(checkpoint: dict, key: str) -> np.ndarray:
    normalization = checkpoint["normalization"]
    value = np.asarray(normalization[key], dtype=np.float32)
    if value.ndim != 1:
        raise RuntimeError(f"normalization[{key}] must be 1D, got {value.shape}")
    return value


def state_from_checkpoint(checkpoint: dict, direct_key: str, prefix: str) -> dict:
    if direct_key in checkpoint:
        return checkpoint[direct_key]
    state = checkpoint["model_state_dict"]
    prefix = prefix + "."
    return {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }


def load_models(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_mean = norm_vector(checkpoint, "input_mean")
    input_std = norm_vector(checkpoint, "input_std")
    target_mean = norm_vector(checkpoint, "target_mean")
    target_std = norm_vector(checkpoint, "target_std")

    in_channels = int(input_mean.size)
    mean_model = LSSUNet(in_channels, 2).to(device)
    std_q_model = LSSUNet(in_channels, 1).to(device)
    std_u_model = LSSUNet(in_channels, 1).to(device)

    mean_model.load_state_dict(
        state_from_checkpoint(checkpoint, "mean_model_state_dict", "mean_model")
    )
    std_q_model.load_state_dict(
        state_from_checkpoint(checkpoint, "std_q_model_state_dict", "std_q_model")
    )
    std_u_model.load_state_dict(
        state_from_checkpoint(checkpoint, "std_u_model_state_dict", "std_u_model")
    )
    mean_model.eval()
    std_q_model.eval()
    std_u_model.eval()
    return checkpoint, mean_model, std_q_model, std_u_model, input_mean, input_std, target_mean, target_std


def plot_one_patch_z(
    patch_z: np.ndarray,
    patch: int,
    path: Path,
    bins: int,
    z_limit: float,
) -> None:
    x = np.linspace(-z_limit, z_limit, 500)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, ax in enumerate(axes):
        values = patch_z[channel].ravel()
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, range=(-z_limit, z_limit), density=True, alpha=0.7)
        ax.plot(x, pdf, "k--", lw=2, label="N(0,1)")
        ax.set_title(
            f"patch {patch} {CHANNELS[channel]} z "
            f"(mean={np.mean(values):.3g}, std={np.std(values):.3g})"
        )
        ax.set_xlabel("(truth - mean) / std")
        ax.set_ylabel("density")
        ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_patch_z(
    z: np.ndarray,
    patches: list[int],
    output_dir: Path,
    bins: int,
    z_limit: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, patch in enumerate(patches):
        path = output_dir / f"patch_{patch}_z_histogram.png"
        plot_one_patch_z(z[index], patch, path, bins, z_limit)
        paths.append(path)
    return paths


def plot_channel_hist(values: np.ndarray, path: Path, title: str, bins: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, ax in enumerate(axes):
        flat = values[:, channel].ravel()
        ax.hist(flat[np.isfinite(flat)], bins=bins, alpha=0.8)
        ax.set_title(f"all patches {CHANNELS[channel]} posterior {title}")
        ax.set_xlabel(f"posterior {title}")
        ax.set_ylabel("pixels")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_patch_scalar_hist(
    values: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    bins: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, ax in enumerate(axes):
        channel_values = values[:, channel]
        channel_values = channel_values[np.isfinite(channel_values)]
        ax.hist(channel_values, bins=bins, alpha=0.8)
        ax.set_title(f"{CHANNELS[channel]} {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("patches")
    fig.savefig(path, dpi=180)
    plt.close(fig)


class DynamicHistogram:
    def __init__(self, bins: int):
        self.bins = int(bins)
        self.edges: np.ndarray | None = None
        self.counts: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        vmin = float(flat.min())
        vmax = float(flat.max())
        if vmin == vmax:
            delta = max(abs(vmin) * 1e-3, 1e-6)
            vmin -= delta
            vmax += delta
        if self.edges is None or self.counts is None:
            self.edges = np.linspace(vmin, vmax, self.bins + 1)
            self.counts = np.zeros(self.bins, dtype=np.float64)
        elif vmin < self.edges[0] or vmax > self.edges[-1]:
            new_min = min(vmin, float(self.edges[0]))
            new_max = max(vmax, float(self.edges[-1]))
            old_centers = 0.5 * (self.edges[:-1] + self.edges[1:])
            new_edges = np.linspace(new_min, new_max, self.bins + 1)
            self.counts = np.histogram(
                old_centers,
                bins=new_edges,
                weights=self.counts,
            )[0].astype(np.float64)
            self.edges = new_edges
        self.counts += np.histogram(flat, bins=self.edges)[0]


def plot_count_hist(
    hists: list[DynamicHistogram],
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, ax in enumerate(axes):
        hist = hists[channel]
        if hist.edges is None or hist.counts is None:
            continue
        centers = 0.5 * (hist.edges[:-1] + hist.edges[1:])
        widths = np.diff(hist.edges)
        ax.bar(centers, hist.counts, width=widths, align="center", alpha=0.8)
        ax.set_title(f"all patches {CHANNELS[channel]} {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pixels")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_patch_z_counts(
    z_counts: np.ndarray,
    z_edges: np.ndarray,
    patch: int,
    path: Path,
    z_mean: np.ndarray,
    z_std: np.ndarray,
) -> None:
    x = np.linspace(float(z_edges[0]), float(z_edges[-1]), 500)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    widths = np.diff(z_edges)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, ax in enumerate(axes):
        total = float(z_counts[channel].sum())
        density = z_counts[channel] / max(total, 1.0) / widths
        ax.bar(centers, density, width=widths, align="center", alpha=0.7)
        ax.plot(x, pdf, "k--", lw=2, label="N(0,1)")
        ax.set_title(
            f"patch {patch} {CHANNELS[channel]} z "
            f"(mean={z_mean[channel]:.3g}, std={z_std[channel]:.3g})"
        )
        ax.set_xlabel("(truth - mean) / std")
        ax.set_ylabel("density")
        ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_patch_statistics(
    path: Path,
    patches: list[int],
    z: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patch",
                "channel",
                "z_pixel_mean",
                "z_pixel_std",
                "posterior_mean_pixel_mean",
                "posterior_mean_pixel_std",
                "posterior_std_pixel_mean",
                "posterior_std_pixel_std",
            ],
        )
        writer.writeheader()
        for index, patch in enumerate(patches):
            for channel, label in enumerate(CHANNELS):
                writer.writerow(
                    {
                        "patch": patch,
                        "channel": label,
                        "z_pixel_mean": f"{np.mean(z[index, channel]):.10g}",
                        "z_pixel_std": f"{np.std(z[index, channel]):.10g}",
                        "posterior_mean_pixel_mean": (
                            f"{np.mean(posterior_mean[index, channel]):.10g}"
                        ),
                        "posterior_mean_pixel_std": (
                            f"{np.std(posterior_mean[index, channel]):.10g}"
                        ),
                        "posterior_std_pixel_mean": (
                            f"{np.mean(posterior_std[index, channel]):.10g}"
                        ),
                        "posterior_std_pixel_std": (
                            f"{np.std(posterior_std[index, channel]):.10g}"
                        ),
                    }
                )


def infer_batch(
    observed: np.ndarray,
    *,
    mean_model: LSSUNet,
    std_q_model: LSSUNet,
    std_u_model: LSSUNet,
    input_mean_t: torch.Tensor,
    input_std_t: torch.Tensor,
    target_mean_t: torch.Tensor,
    target_std_t: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch = torch.from_numpy(observed).to(device)
    batch = (batch - input_mean_t) / input_std_t
    mean_norm = mean_model(batch)
    logvar_norm = torch.cat([std_q_model(batch), std_u_model(batch)], dim=1)
    logvar_norm = logvar_norm.clamp(-12.0, 12.0)
    mean = mean_norm * target_std_t + target_mean_t
    std = torch.exp(0.5 * logvar_norm) * target_std_t
    return mean.cpu().numpy().astype(np.float32), std.cpu().numpy().astype(np.float32)


def run_dataset_diagnostic(
    *,
    args: argparse.Namespace,
    checkpoint: dict,
    mean_model: LSSUNet,
    std_q_model: LSSUNet,
    std_u_model: LSSUNet,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> dict:
    requested_patches = parse_patch_list(args.patch_list)
    dataset_files = select_original_dataset_files(args.dataset_dir, requested_patches)
    sample_indices_by_file = None
    patch_tag = patch_selection_tag(args.patch_list)
    result_dir = args.output_dir / f"original_patches_{patch_tag}"
    result_dir.mkdir(parents=True, exist_ok=True)
    per_patch_z_dir = result_dir / "per_patch_z_histograms"
    per_patch_z_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [path.name for path in dataset_files]
    expected_patches = [patch_from_dataset_name(path) for path in dataset_files]
    progress_path = result_dir / "original_dataset_progress.json"
    progress = {
        "status": "running",
        "eval_source": "original",
        "patch_list": args.patch_list,
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "expected_file_count": len(dataset_files),
        "expected_files": expected_files,
        "expected_patches": expected_patches,
        "sample_split_present": sample_indices_by_file is not None,
        "processed_file_count": 0,
        "processed_files": [],
        "processed_patches": [],
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    z_edges = np.linspace(-args.z_limit, args.z_limit, args.bins + 1)
    posterior_mean_hists = [DynamicHistogram(args.bins), DynamicHistogram(args.bins)]
    posterior_std_hists = [DynamicHistogram(args.bins), DynamicHistogram(args.bins)]
    patch_rows: list[dict[str, object]] = []
    patch_z_paths: list[str] = []

    input_mean_t = torch.tensor(input_mean[None, :, None, None], device=device)
    input_std_t = torch.tensor(input_std[None, :, None, None], device=device)
    target_mean_t = torch.tensor(target_mean[None, :, None, None], device=device)
    target_std_t = torch.tensor(target_std[None, :, None, None], device=device)

    for dataset_path in dataset_files:
        patch = patch_from_dataset_name(dataset_path)
        print(f"dataset_file={dataset_path.name} patch={patch}", flush=True)
        with np.load(dataset_path, allow_pickle=False) as data:
            xq = data["xq"].astype(np.float32)
            xu = data["xu"].astype(np.float32)
            yq = data["yq"].astype(np.float32)
            yu = data["yu"].astype(np.float32)
            n_samples, h, w = xq.shape
            if input_mean.size == 2:
                intensity = None
            elif input_mean.size == 3:
                intensity = build_dataset_intensity_samples(
                    data,
                    dataset_path,
                    signal_dir=args.signal_dir,
                    intensity_freq=args.intensity_freq,
                    map_hw=(h, w),
                )
            else:
                raise RuntimeError(
                    f"Unsupported checkpoint input channel count: {input_mean.size}"
                )
            if sample_indices_by_file is not None:
                indices = sample_indices_by_file.get(dataset_path.name)
                if indices is None or len(indices) == 0:
                    continue
                xq = xq[indices]
                xu = xu[indices]
                yq = yq[indices]
                yu = yu[indices]
                if intensity is not None:
                    intensity = intensity[indices]
                n_samples = int(len(indices))

            z_counts = np.zeros((2, args.bins), dtype=np.float64)
            z_sum = np.zeros(2, dtype=np.float64)
            z_sumsq = np.zeros(2, dtype=np.float64)
            z_count = np.zeros(2, dtype=np.float64)
            mean_sum = np.zeros(2, dtype=np.float64)
            mean_sumsq = np.zeros(2, dtype=np.float64)
            std_sum = np.zeros(2, dtype=np.float64)
            std_sumsq = np.zeros(2, dtype=np.float64)

            with torch.no_grad():
                for start in range(0, n_samples, args.batch_size):
                    end = min(start + args.batch_size, n_samples)
                    truth = np.stack([xq[start:end], xu[start:end]], axis=1)
                    observed_qu = np.stack([yq[start:end], yu[start:end]], axis=1)
                    if intensity is None:
                        observed = observed_qu
                    else:
                        observed = np.concatenate(
                            [observed_qu, intensity[start:end, None]],
                            axis=1,
                        )
                    posterior_mean, posterior_std = infer_batch(
                        observed,
                        mean_model=mean_model,
                        std_q_model=std_q_model,
                        std_u_model=std_u_model,
                        input_mean_t=input_mean_t,
                        input_std_t=input_std_t,
                        target_mean_t=target_mean_t,
                        target_std_t=target_std_t,
                        device=device,
                    )
                    z = (truth - posterior_mean) / posterior_std
                    if not np.all(np.isfinite(z)):
                        raise RuntimeError(f"{dataset_path}: z contains non-finite values")

                    for channel in range(2):
                        z_flat = z[:, channel].ravel()
                        z_counts[channel] += np.histogram(z_flat, bins=z_edges)[0]
                        z_sum[channel] += float(z_flat.sum())
                        z_sumsq[channel] += float(np.square(z_flat).sum())
                        z_count[channel] += z_flat.size
                        mean_flat = posterior_mean[:, channel].ravel()
                        std_flat = posterior_std[:, channel].ravel()
                        mean_sum[channel] += float(mean_flat.sum())
                        mean_sumsq[channel] += float(np.square(mean_flat).sum())
                        std_sum[channel] += float(std_flat.sum())
                        std_sumsq[channel] += float(np.square(std_flat).sum())
                        posterior_mean_hists[channel].update(mean_flat)
                        posterior_std_hists[channel].update(std_flat)

        z_mean = z_sum / np.maximum(z_count, 1)
        z_std = np.sqrt(np.maximum(z_sumsq / np.maximum(z_count, 1) - z_mean**2, 0.0))
        mean_pixel_mean = mean_sum / np.maximum(z_count, 1)
        mean_pixel_std = np.sqrt(
            np.maximum(mean_sumsq / np.maximum(z_count, 1) - mean_pixel_mean**2, 0.0)
        )
        std_pixel_mean = std_sum / np.maximum(z_count, 1)
        std_pixel_std = np.sqrt(
            np.maximum(std_sumsq / np.maximum(z_count, 1) - std_pixel_mean**2, 0.0)
        )
        z_path = per_patch_z_dir / f"patch_{patch}_z_histogram.png"
        plot_patch_z_counts(z_counts, z_edges, patch, z_path, z_mean, z_std)
        patch_z_paths.append(str(z_path))
        for channel, label in enumerate(CHANNELS):
            patch_rows.append(
                {
                    "patch": patch,
                    "dataset_file": dataset_path.name,
                    "channel": label,
                    "sample_count": n_samples,
                    "z_pixel_mean": float(z_mean[channel]),
                    "z_pixel_std": float(z_std[channel]),
                    "posterior_mean_pixel_mean": float(mean_pixel_mean[channel]),
                    "posterior_mean_pixel_std": float(mean_pixel_std[channel]),
                    "posterior_std_pixel_mean": float(std_pixel_mean[channel]),
                    "posterior_std_pixel_std": float(std_pixel_std[channel]),
                    }
                )
        progress["processed_files"].append(dataset_path.name)
        progress["processed_patches"].append(patch)
        progress["processed_file_count"] = len(progress["processed_files"])
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    z_patch_mean = np.asarray(
        [[row["z_pixel_mean"] for row in patch_rows if row["channel"] == label] for label in CHANNELS],
        dtype=np.float64,
    ).T
    z_patch_std = np.asarray(
        [[row["z_pixel_std"] for row in patch_rows if row["channel"] == label] for label in CHANNELS],
        dtype=np.float64,
    ).T
    patch_scalar_bins = min(args.bins, max(5, len(dataset_files)))
    z_patch_mean_path = result_dir / "original_dataset_z_patch_mean_histogram.png"
    z_patch_std_path = result_dir / "original_dataset_z_patch_std_histogram.png"
    mean_path = result_dir / "original_dataset_posterior_mean_histogram.png"
    std_path = result_dir / "original_dataset_posterior_std_histogram.png"
    plot_patch_scalar_hist(
        z_patch_mean,
        z_patch_mean_path,
        "z-map mean across patches",
        "mean of z pixels in one patch",
        patch_scalar_bins,
    )
    plot_patch_scalar_hist(
        z_patch_std,
        z_patch_std_path,
        "z-map std across patches",
        "std of z pixels in one patch",
        patch_scalar_bins,
    )
    plot_count_hist(posterior_mean_hists, mean_path, "posterior mean", "posterior mean")
    plot_count_hist(posterior_std_hists, std_path, "posterior std", "posterior std")

    csv_path = result_dir / "original_dataset_patch_statistics.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patch_rows)

    summary = {
        "data_source": "dataset",
        "eval_source": "original",
        "patch_list": args.patch_list,
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(result_dir),
        "dataset_files": expected_files,
        "patches": expected_patches,
        "sample_split_present": sample_indices_by_file is not None,
        "input_channels": int(input_mean.size),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "outputs": {
            "per_patch_z_histograms": patch_z_paths,
            "z_patch_mean_histogram": str(z_patch_mean_path),
            "z_patch_std_histogram": str(z_patch_std_path),
            "posterior_mean_histogram_all_patches": str(mean_path),
            "posterior_std_histogram_all_patches": str(std_path),
            "patch_statistics_csv": str(csv_path),
        },
    }
    summary_path = result_dir / "original_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    progress["status"] = "complete"
    progress["summary"] = str(summary_path)
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-source", choices=("original", "synthetic"), default=os.environ.get("EVAL_SOURCE", os.environ.get("DATA_SOURCE", "original")).replace("dataset", "original"))
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("DATASET_DIR", DATASET_DIR_DEFAULT)))
    parser.add_argument("--synthesis-dir", type=Path, default=Path(os.environ.get("SYNTHESIS_DIR", SYNTHESIS_DIR_DEFAULT)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("CHECKPOINT_PATH", CHECKPOINT_DEFAULT)))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", OUTPUT_DIR_DEFAULT)))
    parser.add_argument("--signal-dir", type=Path, default=Path(os.environ.get("PLANCK_SIGNAL_DIR", SIGNAL_DIR_DEFAULT)))
    parser.add_argument("--patch-list", default=os.environ.get("PATCH_LIST", PATCH_LIST_DEFAULT))
    parser.add_argument("--intensity-freq", type=int, default=int(os.environ.get("INTENSITY_FREQ", "857")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--bins", type=int, default=int(os.environ.get("HISTOGRAM_BINS", "120")))
    parser.add_argument("--z-limit", type=float, default=float(os.environ.get("Z_LIMIT", "6.0")))
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "64"))))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, mean_model, std_q_model, std_u_model, input_mean, input_std, target_mean, target_std = load_models(args.checkpoint, device)
    print(f"eval_source={args.eval_source}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"input_channels={input_mean.size}", flush=True)

    if args.eval_source == "original":
        summary = run_dataset_diagnostic(
            args=args,
            checkpoint=checkpoint,
            mean_model=mean_model,
            std_q_model=std_q_model,
            std_u_model=std_u_model,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )
        print(f"dataset_files={len(summary['dataset_files'])}", flush=True)
        print(f"patches={summary['patches']}", flush=True)
        for key, value in summary["outputs"].items():
            if isinstance(value, list):
                print(f"saved_{key}_dir={Path(value[0]).parent if value else ''}", flush=True)
            else:
                print(f"saved={value}", flush=True)
        return

    patches = select_synthesis_patches(args.synthesis_dir, parse_patch_list(args.patch_list))
    truth, observed_qu, patches = load_syntheses(args.synthesis_dir, patches)
    _, _, h, w = observed_qu.shape
    result_dir = args.output_dir / f"synthetic_patches_{patch_selection_tag(args.patch_list)}"
    result_dir.mkdir(parents=True, exist_ok=True)

    if input_mean.size == 2:
        observed = observed_qu
    elif input_mean.size == 3:
        intensity = np.stack(
            [load_intensity(args.signal_dir, p, (h, w), args.intensity_freq) for p in patches]
        )
        observed = np.concatenate([observed_qu, intensity[:, None]], axis=1)
    else:
        raise RuntimeError(f"Unsupported checkpoint input channel count: {input_mean.size}")

    posterior_mean_chunks = []
    posterior_std_chunks = []
    input_mean_t = torch.tensor(input_mean[None, :, None, None], device=device)
    input_std_t = torch.tensor(input_std[None, :, None, None], device=device)
    target_mean_t = torch.tensor(target_mean[None, :, None, None], device=device)
    target_std_t = torch.tensor(target_std[None, :, None, None], device=device)

    with torch.no_grad():
        for start in range(0, observed.shape[0], args.batch_size):
            batch = torch.from_numpy(observed[start : start + args.batch_size]).to(device)
            batch = (batch - input_mean_t) / input_std_t
            mean_norm = mean_model(batch)
            logvar_norm = torch.cat([std_q_model(batch), std_u_model(batch)], dim=1)
            logvar_norm = logvar_norm.clamp(-12.0, 12.0)
            mean = mean_norm * target_std_t + target_mean_t
            std = torch.exp(0.5 * logvar_norm) * target_std_t
            posterior_mean_chunks.append(mean.cpu().numpy())
            posterior_std_chunks.append(std.cpu().numpy())

    posterior_mean = np.concatenate(posterior_mean_chunks, axis=0).astype(np.float32)
    posterior_std = np.concatenate(posterior_std_chunks, axis=0).astype(np.float32)
    z = (truth - posterior_mean) / posterior_std
    if not np.all(np.isfinite(z)):
        raise RuntimeError("z contains non-finite values")
    z_patch_mean = z.mean(axis=(2, 3))
    z_patch_std = z.std(axis=(2, 3))

    per_patch_z_dir = result_dir / "per_patch_z_histograms"
    z_patch_mean_path = result_dir / "synthetic_posterior_z_patch_mean_histogram.png"
    z_patch_std_path = result_dir / "synthetic_posterior_z_patch_std_histogram.png"
    mean_path = result_dir / "synthetic_posterior_mean_histogram.png"
    std_path = result_dir / "synthetic_posterior_std_histogram.png"
    patch_z_paths = plot_per_patch_z(
        z,
        patches,
        per_patch_z_dir,
        args.bins,
        args.z_limit,
    )
    patch_scalar_bins = min(args.bins, max(5, len(patches)))
    plot_patch_scalar_hist(
        z_patch_mean,
        z_patch_mean_path,
        "z-map mean across patches",
        "mean of z pixels in one patch",
        patch_scalar_bins,
    )
    plot_patch_scalar_hist(
        z_patch_std,
        z_patch_std_path,
        "z-map std across patches",
        "std of z pixels in one patch",
        patch_scalar_bins,
    )
    plot_channel_hist(posterior_mean, mean_path, "mean", args.bins)
    plot_channel_hist(posterior_std, std_path, "std", args.bins)
    csv_path = result_dir / "synthetic_posterior_patch_statistics.csv"
    write_patch_statistics(csv_path, patches, z, posterior_mean, posterior_std)

    summary = {
        "synthesis_dir": str(args.synthesis_dir),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(result_dir),
        "eval_source": "synthetic",
        "patch_list": args.patch_list,
        "patches": patches,
        "input_channels": int(input_mean.size),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "z": {
            "Q_mean": float(np.mean(z[:, 0])),
            "Q_std": float(np.std(z[:, 0])),
            "U_mean": float(np.mean(z[:, 1])),
            "U_std": float(np.std(z[:, 1])),
        },
        "outputs": {
            "per_patch_z_histograms": [str(path) for path in patch_z_paths],
            "z_patch_mean_histogram": str(z_patch_mean_path),
            "z_patch_std_histogram": str(z_patch_std_path),
            "posterior_mean_histogram_all_patches": str(mean_path),
            "posterior_std_histogram_all_patches": str(std_path),
            "patch_statistics_csv": str(csv_path),
        },
    }
    summary_path = result_dir / "synthetic_posterior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"loaded_patches={patches}", flush=True)
    print(f"input_channels={input_mean.size}", flush=True)
    print(f"Q z mean/std = {summary['z']['Q_mean']:.6g} / {summary['z']['Q_std']:.6g}", flush=True)
    print(f"U z mean/std = {summary['z']['U_mean']:.6g} / {summary['z']['U_std']:.6g}", flush=True)
    print(f"saved_per_patch_z_dir={per_patch_z_dir}", flush=True)
    print(f"saved={z_patch_mean_path}", flush=True)
    print(f"saved={z_patch_std_path}", flush=True)
    print(f"saved={mean_path}", flush=True)
    print(f"saved={std_path}", flush=True)
    print(f"saved={csv_path}", flush=True)
    print(f"saved={summary_path}", flush=True)


if __name__ == "__main__":
    main()
