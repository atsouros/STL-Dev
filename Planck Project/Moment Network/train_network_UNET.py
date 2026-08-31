#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PLANCK_PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PLANCK_PROJECT_DIR.parent
REPO_ROOT = Path(os.environ.get("STL_DEV_ROOT", str(DEFAULT_REPO_ROOT))).expanduser()
for path in (REPO_ROOT, PLANCK_PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_DATASET_DIR = Path("/travail/atsouros/version_2")
DEFAULT_OUTPUT_DIR = Path("/obs/atsouros/projects/STL-Dev/Moment Networks/moment_network_training/version_2_qu_splitstd_nll")
DEFAULT_SIGNAL_DIR = Path("/travail/atsouros/signal_I")
DIAGNOSTIC_PATCH = "20"
DIAGNOSTIC_Q_PATH = Path(
    "/travail/atsouros/planck_data/BK_CMB_S4_north_patch_v4/signal/patch_20/"
    "patch_20_Q353_PR4_no_PS_map_zero_adjust_cen_pix_lon_254.970703125_"
    "lat_54.340912303861245_v4_10_arcmin.npy"
)
DIAGNOSTIC_U_PATH = Path(
    "/travail/atsouros/planck_data/BK_CMB_S4_north_patch_v4/signal/patch_20/"
    "patch_20_U353_PR4_no_PS_map_zero_adjust_cen_pix_lon_254.970703125_"
    "lat_54.340912303861245_v4_10_arcmin.npy"
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def parse_patch_selector(selector: str | None) -> set[str] | None:
    if selector is None or selector.strip() == "":
        return None

    patches: set[str] = set()
    for item in selector.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid descending patch range: {item}")
            patches.update(str(p) for p in range(start, end + 1))
        else:
            patches.add(str(int(item)))
    return patches


def patch_from_name(path: Path) -> str | None:
    match = re.match(r"p(\d+)_", path.name)
    return match.group(1) if match else None


def downsample_by_four(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image for downsampling, got {image.shape}")
    h, w = image.shape
    if h % 2 or w % 2:
        raise ValueError(f"Image dimensions must be even for 2x2 downsampling, got {h}x{w}")
    return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def center_crop(image: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image for cropping, got {image.shape}")
    out_h, out_w = out_hw
    h, w = image.shape
    if out_h > h or out_w > w:
        raise ValueError(f"Cannot crop {h}x{w} image to {out_h}x{out_w}")
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return image[y0 : y0 + out_h, x0 : x0 + out_w]


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


def augment_image_np(img: np.ndarray, n_augmentations: int, shift_step: int) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []
    arr = np.asarray(img)
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


def load_intensity_map(
    signal_dir: Path,
    *,
    patch: str,
    intensity_freq: int,
    map_size: int,
) -> np.ndarray:
    path = select_no_bonus_signal_path(signal_dir, f"patch_{patch}_I{intensity_freq}_*.npy", f"I{intensity_freq}")
    raw = np.load(path).astype(np.float64)
    expected_raw_shape = (2 * map_size, 2 * map_size)
    expected_preprocessed_shape = (map_size, map_size)
    if raw.shape == expected_preprocessed_shape:
        image = raw
    elif raw.shape == expected_raw_shape:
        image = downsample_by_four(raw)
        image = center_crop(image, out_hw=expected_preprocessed_shape)
    else:
        raise RuntimeError(
            f"Expected I{intensity_freq} map for patch {patch} to have raw shape {expected_raw_shape} "
            f"or preprocessed shape {expected_preprocessed_shape}, got {raw.shape}"
        )
    return image.astype(np.float32)


def build_intensity_samples(
    data: np.lib.npyio.NpzFile,
    dataset_path: Path,
    signal_dir: Path,
    *,
    intensity_freq: int,
) -> np.ndarray:
    if "yi" in data:
        return data["yi"].astype(np.float32, copy=False)

    patch = str(data["p"].item()) if "p" in data else patch_from_name(dataset_path)
    if patch is None:
        raise RuntimeError(f"Could not infer patch number for intensity channel from {dataset_path}")

    xq = data["xq"]
    n_total, h, w = xq.shape
    n_aug = int(data["naug"].item()) if "naug" in data else 0
    shift_step = int(data["shift"].item()) if "shift" in data else 16
    if n_aug <= 0:
        raise RuntimeError(f"Dataset {dataset_path} does not contain a usable naug value")
    if n_total % n_aug != 0:
        raise RuntimeError(f"Dataset {dataset_path} has {n_total} samples, not divisible by naug={n_aug}")

    intensity = load_intensity_map(signal_dir, patch=patch, intensity_freq=intensity_freq, map_size=h)
    if intensity.shape != (h, w):
        raise RuntimeError(f"Intensity shape mismatch for {dataset_path}: {intensity.shape} vs {(h, w)}")

    augmented_i = augment_image_np(intensity, n_augmentations=n_aug, shift_step=shift_step)
    block = np.stack(augmented_i, axis=0).astype(np.float32)
    n_syntheses = n_total // n_aug
    return np.tile(block[None, ...], (n_syntheses, 1, 1, 1)).reshape(n_total, h, w)


def build_input_samples(
    data: np.lib.npyio.NpzFile,
    dataset_path: Path,
    *,
    use_intensity: bool,
    signal_dir: Path,
    intensity_freq: int,
) -> np.ndarray:
    if not use_intensity:
        return np.stack([data["yq"], data["yu"]], axis=1).astype(np.float32, copy=False)
    yi = build_intensity_samples(data, dataset_path, signal_dir, intensity_freq=intensity_freq)
    return np.stack([data["yq"], data["yu"], yi], axis=1).astype(np.float32, copy=False)


def discover_dataset_files(dataset_dir: Path, patch_selector: set[str] | None) -> list[Path]:
    files = []
    for path in sorted(dataset_dir.glob("*_moment_dataset_*.npz")):
        patch = patch_from_name(path)
        if patch is None:
            continue
        if patch_selector is not None and patch not in patch_selector:
            continue
        files.append(path)
    files.sort(key=lambda p: int(patch_from_name(p) or -1))
    return files


def dataset_sample_count(path: Path) -> int:
    with open_dataset_npz(path) as data:
        return int(data["xq"].shape[0])


def split_samples(
    files: list[Path],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"TRAIN_FRACTION must be between 0 and 1, got {train_fraction}")

    counts = {path.name: dataset_sample_count(path) for path in files}
    total = int(sum(counts.values()))
    if total < 2:
        raise ValueError("Need at least two samples for a train/validation split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    split = int(round(train_fraction * total))
    split = min(max(1, split), total - 1)
    train_mask = np.zeros(total, dtype=bool)
    train_mask[order[:split]] = True

    train_indices: dict[str, np.ndarray] = {}
    val_indices: dict[str, np.ndarray] = {}
    offset = 0
    for path in files:
        count = counts[path.name]
        local = np.arange(count, dtype=np.int64)
        mask = train_mask[offset : offset + count]
        train_indices[path.name] = local[mask]
        val_indices[path.name] = local[~mask]
        offset += count

    return train_indices, val_indices, counts


def serializable_sample_split(
    train_indices: dict[str, np.ndarray],
    val_indices: dict[str, np.ndarray],
    sample_counts: dict[str, int],
) -> dict[str, object]:
    return {
        "sample_counts": {name: int(count) for name, count in sample_counts.items()},
        "train_sample_count": int(sum(len(indices) for indices in train_indices.values())),
        "val_sample_count": int(sum(len(indices) for indices in val_indices.values())),
        "train_indices": {
            name: indices.astype(np.int64).tolist()
            for name, indices in train_indices.items()
            if len(indices) > 0
        },
        "val_indices": {
            name: indices.astype(np.int64).tolist()
            for name, indices in val_indices.items()
            if len(indices) > 0
        },
    }


def sample_split_records(
    files: list[Path],
    train_indices: dict[str, np.ndarray],
    val_indices: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in files:
        patch = patch_from_name(path)
        if patch is None:
            raise RuntimeError(f"Could not infer patch number from {path}")
        for split, indices_by_file in (
            ("train", train_indices),
            ("val", val_indices),
        ):
            for sample_index in indices_by_file[path.name]:
                records.append(
                    {
                        "dataset_file": path.name,
                        "patch": int(patch),
                        "sample_index": int(sample_index),
                        "split": split,
                    }
                )
    return records


def save_sample_split_manifest(
    path: Path,
    *,
    files: list[Path],
    train_indices: dict[str, np.ndarray],
    val_indices: dict[str, np.ndarray],
    sample_counts: dict[str, int],
    train_fraction: float,
    seed: int,
) -> None:
    serializable = serializable_sample_split(
        train_indices,
        val_indices,
        sample_counts,
    )
    payload = {
        "split_unit": "sample",
        "description": (
            "Random split over individual (x,y) pairs pooled across all patch "
            "dataset files. Patches are not held out as units."
        ),
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        **serializable,
        "records": sample_split_records(files, train_indices, val_indices),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


@dataclass(frozen=True)
class ChannelStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def compute_channel_stats(
    files: list[Path],
    *,
    use_intensity: bool,
    signal_dir: Path,
    intensity_freq: int,
    sample_indices_by_file: dict[str, np.ndarray] | None = None,
    eps: float = 1e-6,
) -> ChannelStats:
    n_input = 3 if use_intensity else 2
    input_sum = np.zeros(n_input, dtype=np.float64)
    input_sumsq = np.zeros(n_input, dtype=np.float64)
    target_sum = np.zeros(2, dtype=np.float64)
    target_sumsq = np.zeros(2, dtype=np.float64)
    count = 0

    for file_index, path in enumerate(files, start=1):
        with open_dataset_npz(path) as data:
            x = np.stack([data["xq"], data["xu"]], axis=1).astype(np.float64, copy=False)
            y = build_input_samples(
                data,
                path,
                use_intensity=use_intensity,
                signal_dir=signal_dir,
                intensity_freq=intensity_freq,
            ).astype(np.float64, copy=False)
        if sample_indices_by_file is not None:
            indices = sample_indices_by_file.get(path.name)
            if indices is None or len(indices) == 0:
                continue
            x = x[indices]
            y = y[indices]
        pixels = int(np.prod(x.shape[0:1] + x.shape[2:]))
        target_sum += x.sum(axis=(0, 2, 3))
        target_sumsq += np.square(x).sum(axis=(0, 2, 3))
        input_sum += y.sum(axis=(0, 2, 3))
        input_sumsq += np.square(y).sum(axis=(0, 2, 3))
        count += pixels
        if file_index == 1 or file_index == len(files) or file_index % 25 == 0:
            print(f"stats {file_index:04d}/{len(files):04d}: {path.name}", flush=True)

    if count == 0:
        raise RuntimeError("No samples available for channel-stat computation")

    input_mean = input_sum / count
    target_mean = target_sum / count
    input_var = np.maximum(input_sumsq / count - input_mean**2, eps**2)
    target_var = np.maximum(target_sumsq / count - target_mean**2, eps**2)
    return ChannelStats(
        input_mean=input_mean.astype(np.float32),
        input_std=np.sqrt(input_var).astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=np.sqrt(target_var).astype(np.float32),
    )


def identity_channel_stats(n_input: int) -> ChannelStats:
    return ChannelStats(
        input_mean=np.zeros(n_input, dtype=np.float32),
        input_std=np.ones(n_input, dtype=np.float32),
        target_mean=np.zeros(2, dtype=np.float32),
        target_std=np.ones(2, dtype=np.float32),
    )


def save_stats(path: Path, stats: ChannelStats) -> None:
    np.savez(
        path,
        input_mean=stats.input_mean,
        input_std=stats.input_std,
        target_mean=stats.target_mean,
        target_std=stats.target_std,
        split_unit=np.asarray("sample"),
    )


def open_dataset_npz(path: Path) -> np.lib.npyio.NpzFile:
    try:
        return np.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open dataset file as a valid .npz archive: {path}. "
            "The file is probably corrupt, incomplete, or not actually an npz file."
        ) from exc


class MomentPatchIterableDataset(IterableDataset):
    def __init__(
        self,
        files: list[Path],
        *,
        stats: ChannelStats,
        use_intensity: bool,
        signal_dir: Path,
        intensity_freq: int,
        sample_indices_by_file: dict[str, np.ndarray] | None,
        seed: int,
        shuffle_files: bool,
        shuffle_samples: bool,
    ):
        super().__init__()
        self.files = list(files)
        self.stats = stats
        self.use_intensity = bool(use_intensity)
        self.signal_dir = signal_dir
        self.intensity_freq = int(intensity_freq)
        self.sample_indices_by_file = sample_indices_by_file
        self.seed = int(seed)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_samples = bool(shuffle_samples)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker.id)
            num_workers = int(worker.num_workers)

        rng = np.random.default_rng(self.seed + 1009 * self.epoch + worker_id)
        files = self.files[worker_id::num_workers]
        if self.shuffle_files:
            files = [files[i] for i in rng.permutation(len(files))]

        input_mean = self.stats.input_mean[None, :, None, None]
        input_std = self.stats.input_std[None, :, None, None]
        target_mean = self.stats.target_mean[None, :, None, None]
        target_std = self.stats.target_std[None, :, None, None]

        for path in files:
            with open_dataset_npz(path) as data:
                x = np.stack([data["xq"], data["xu"]], axis=1).astype(np.float32, copy=False)
                y = build_input_samples(
                    data,
                    path,
                    use_intensity=self.use_intensity,
                    signal_dir=self.signal_dir,
                    intensity_freq=self.intensity_freq,
                )
            if self.sample_indices_by_file is not None:
                indices = self.sample_indices_by_file.get(path.name)
                if indices is None or len(indices) == 0:
                    continue
                x = x[indices]
                y = y[indices]

            x = (x - target_mean) / target_std
            y = (y - input_mean) / input_std
            order = np.arange(len(x))
            if self.shuffle_samples:
                rng.shuffle(order)
            for idx in order:
                yield torch.from_numpy(y[idx]), torch.from_numpy(x[idx])


class ConvGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        if out_ch % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_ch, eps=1e-5, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LSSUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 2):
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


class SplitStdMomentModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.mean_model = LSSUNet(in_channels=in_channels, out_channels=2)
        self.std_q_model = LSSUNet(in_channels=in_channels, out_channels=1)
        self.std_u_model = LSSUNet(in_channels=in_channels, out_channels=1)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_model(x)

    def logvar(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.std_q_model(x), self.std_u_model(x)], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(x), self.logvar(x)


def moment_variance_loss(target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Regress the predicted variance onto the squared reconstruction residual."""
    logvar = logvar.clamp(min=-12.0, max=12.0)
    var = torch.exp(logvar).clamp_min(1e-8)
    squared_residual = (target - mu) ** 2
    return (0.5 * (squared_residual - var) ** 2).mean()


def train_mean_epoch(
    model: SplitStdMomentModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.mean_model.train()
    model.std_q_model.eval()
    model.std_u_model.eval()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mu = model.mean(y)
        loss = F.mse_loss(mu, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mean_model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate_mean_epoch(
    model: SplitStdMomentModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        mu = model.mean(y)
        loss = F.mse_loss(mu, x)
        running_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return running_loss / max(1, n_batches)


def train_full_moment_epoch(
    model: SplitStdMomentModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.mean_model.eval()
    model.std_q_model.train()
    model.std_u_model.train()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            mu = model.mean(y)
        logvar = model.logvar(y)
        loss = moment_variance_loss(x, mu, logvar)
        loss.backward()
        variance_parameters = list(model.std_q_model.parameters()) + list(model.std_u_model.parameters())
        torch.nn.utils.clip_grad_norm_(variance_parameters, 1.0)
        optimizer.step()
        running_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate_full_moment_epoch(
    model: SplitStdMomentModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        mu = model.mean(y)
        logvar = model.logvar(y)
        loss = moment_variance_loss(x, mu, logvar)
        running_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return running_loss / max(1, n_batches)


def make_loader(
    dataset: MomentPatchIterableDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def save_checkpoint(
    path: Path,
    *,
    model: SplitStdMomentModel,
    mean_optimizer: torch.optim.Optimizer,
    full_optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: str,
    best_val_loss: float,
    args: argparse.Namespace,
    train_files: list[Path],
    val_files: list[Path],
    stats: ChannelStats,
    train_losses: list[float],
    val_losses: list[float],
    stages: list[str],
) -> None:
    torch.save(
        {
            "model_format": "joint_qu_i_split_std_moment_mse" if args.use_intensity else "joint_qu_split_std_moment_mse",
            "model_state_dict": model.state_dict(),
            "mean_model_state_dict": model.mean_model.state_dict(),
            "std_q_model_state_dict": model.std_q_model.state_dict(),
            "std_u_model_state_dict": model.std_u_model.state_dict(),
            "mean_optimizer_state_dict": mean_optimizer.state_dict(),
            "full_optimizer_state_dict": full_optimizer.state_dict(),
            "epoch": int(epoch),
            "stage": stage,
            "best_val_loss": float(best_val_loss),
            "input_channels": ["Q_contaminated", "U_contaminated", f"I{args.intensity_freq}"] if args.use_intensity else ["Q_contaminated", "U_contaminated"],
            "target_channels": ["Q_clean", "U_clean"],
            "output_channels": ["Q_mean", "U_mean", "Q_logvar", "U_logvar"],
            "mean_model_output_channels": int(args.mean_output_channels),
            "mean_channels_used": [0, 1],
            "posterior_parameterization": "moment_variance_regression_split_std_trunks",
            "stage_2_objective": "0.5 * mean(((target - mean)**2 - exp(logvar))**2)",
            "stage_2_updates_mean": False,
            "stage_2_mean_fixed": True,
            "std_architecture": "independent_q_u_unet_trunks",
            "dataset_dir": str(args.dataset_dir),
            "split_unit": "sample",
            "train_fraction": float(args.train_fraction),
            "sample_split": getattr(args, "sample_split", None),
            "sample_split_manifest": getattr(args, "sample_split_manifest", None),
            "use_intensity": bool(args.use_intensity),
            "signal_dir": str(args.signal_dir) if args.use_intensity else None,
            "intensity_freq": int(args.intensity_freq) if args.use_intensity else None,
            "train_files": [p.name for p in train_files],
            "val_files": [p.name for p in val_files],
            "normalization": {
                "input_mean": stats.input_mean.tolist(),
                "input_std": stats.input_std.tolist(),
                "target_mean": stats.target_mean.tolist(),
                "target_std": stats.target_std.tolist(),
            },
            "train_losses": train_losses,
            "val_losses": val_losses,
            "stages": stages,
            "config": vars(args),
        },
        path,
    )


def save_history(path: Path, train_losses: list[float], val_losses: list[float], stages: list[str]) -> None:
    np.savez(
        path,
        train_losses=np.asarray(train_losses, dtype=np.float64),
        val_losses=np.asarray(val_losses, dtype=np.float64),
        stages=np.asarray(stages),
    )


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


def save_loss_plot(path: Path, train_losses: list[float], val_losses: list[float], stages: list[str]) -> None:
    if not train_losses and not val_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    if train_losses:
        ax.plot(epochs, train_losses, marker="o", markersize=2.5, linewidth=1.2, label="train")
    if val_losses:
        ax.plot(epochs, val_losses, marker="o", markersize=2.5, linewidth=1.2, label="validation")
    ax.set_yscale("log")
    for idx, stage in enumerate(stages):
        if idx == 0 or stages[idx - 1] != stage:
            ax.axvline(idx + 1, color="0.75", linewidth=0.8, linestyle="--")
            ax.text(idx + 1, ax.get_ylim()[1], stage, rotation=90, va="top", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Moment-network training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cleanup_intermediate_checkpoints(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return 0
    removed = 0
    for path in sorted(checkpoint_dir.glob("moment_network_joint_epoch_*.pth")):
        path.unlink()
        removed += 1
    return removed


def load_fixed_diagnostic_signal(
    *,
    use_intensity: bool,
    intensity_freq: int,
) -> tuple[np.ndarray, str]:
    maps = []
    for label, path in (("Q", DIAGNOSTIC_Q_PATH), ("U", DIAGNOSTIC_U_PATH)):
        if not path.is_file():
            raise FileNotFoundError(f"Could not find diagnostic {label} signal map: {path}")
        image = np.load(path).astype(np.float64)
        if image.shape != (768, 768):
            raise RuntimeError(
                f"Expected diagnostic {label} map to have shape (768, 768), got {image.shape}"
            )
        downgraded = downsample_by_four(image)
        if downgraded.shape != (384, 384):
            raise RuntimeError(
                f"Expected downgraded diagnostic {label} map to have shape (384, 384), "
                f"got {downgraded.shape}"
            )
        maps.append(downgraded.astype(np.float32))

    source_paths = [f"Q={DIAGNOSTIC_Q_PATH}", f"U={DIAGNOSTIC_U_PATH}"]
    if use_intensity:
        intensity_path = select_no_bonus_signal_path(
            DIAGNOSTIC_Q_PATH.parent,
            f"patch_{DIAGNOSTIC_PATCH}_I{intensity_freq}_*.npy",
            f"diagnostic I{intensity_freq}",
        )
        intensity = np.load(intensity_path).astype(np.float64)
        if intensity.shape != (768, 768):
            raise RuntimeError(
                f"Expected diagnostic intensity map to have shape (768, 768), got {intensity.shape}"
            )
        intensity = downsample_by_four(intensity)
        if intensity.shape != (384, 384):
            raise RuntimeError(
                f"Expected downgraded diagnostic intensity map to have shape (384, 384), "
                f"got {intensity.shape}"
            )
        maps.append(intensity.astype(np.float32))
        source_paths.append(f"I{intensity_freq}={intensity_path}")

    source = "; ".join(source_paths)
    return np.stack(maps, axis=0), source


@torch.no_grad()
def infer_diagnostic_sample(
    model: SplitStdMomentModel,
    observed: np.ndarray,
    stats: ChannelStats,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_input = int(stats.input_mean.size)
    input_mean = stats.input_mean.reshape(n_input, 1, 1)
    input_std = stats.input_std.reshape(n_input, 1, 1)
    target_mean = stats.target_mean.reshape(2, 1, 1)
    target_std = stats.target_std.reshape(2, 1, 1)
    observed_norm = (observed - input_mean) / input_std
    tensor = torch.from_numpy(observed_norm[None].astype(np.float32)).to(device=device)
    model.eval()
    mean_norm, logvar_norm = model(tensor)
    mean_norm_np = mean_norm.detach().cpu().numpy()[0]
    logvar_norm_np = np.clip(logvar_norm.detach().cpu().numpy()[0], -12.0, 12.0)
    std_norm_np = np.exp(0.5 * logvar_norm_np)
    mean = mean_norm_np * target_std + target_mean
    std = std_norm_np * target_std
    return mean.astype(np.float32), std.astype(np.float32), logvar_norm_np.astype(np.float32)


def save_diagnostic_inference_plot(
    path: Path,
    *,
    model: SplitStdMomentModel,
    observed: np.ndarray,
    stats: ChannelStats,
    device: torch.device,
    patch: str,
    stage: str,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
) -> None:
    mean, std, _ = infer_diagnostic_sample(model, observed, stats, device=device)
    labels = ["Q", "U"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for row, label in enumerate(labels):
        map_vmin, map_vmax = robust_limits([observed[row], mean[row]])
        std_vmin, std_vmax = positive_limits([std[row]])
        panels = [
            (f"{label} true signal input", observed[row], "coolwarm", map_vmin, map_vmax),
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
    fig.suptitle(
        f"Patch {patch} fixed true signal | {stage} epoch {epoch:04d}\n"
        f"train={train_loss:.6e} val={val_loss:.6e}" if train_loss is not None and val_loss is not None
        else f"Patch {patch} fixed true signal | {stage} epoch {epoch:04d}",
        fontsize=13,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_legacy_mean_model(mean_model: nn.Module, checkpoint_path: Path, *, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location=device)
    legacy_state = checkpoint.get("model_state_dict")
    if legacy_state is None:
        raise RuntimeError(f"Legacy mean checkpoint {checkpoint_path} does not contain model_state_dict")

    mean_state = mean_model.state_dict()
    converted_state = {}
    for key, value in legacy_state.items():
        if key not in mean_state:
            continue
        if key == "final.weight":
            if value.ndim != 4 or value.shape[0] < 2 or mean_state[key].shape[0] != 2:
                raise RuntimeError(
                    f"Cannot copy legacy final.weight with shape {tuple(value.shape)} "
                    f"into mean model shape {tuple(mean_state[key].shape)}"
                )
            converted_state[key] = value[0:2].clone()
        elif key == "final.bias":
            if value.ndim != 1 or value.shape[0] < 2 or mean_state[key].shape[0] != 2:
                raise RuntimeError(
                    f"Cannot copy legacy final.bias with shape {tuple(value.shape)} "
                    f"into mean model shape {tuple(mean_state[key].shape)}"
                )
            converted_state[key] = value[0:2].clone()
        elif tuple(value.shape) == tuple(mean_state[key].shape):
            converted_state[key] = value
        else:
            raise RuntimeError(
                f"Legacy checkpoint tensor {key} has shape {tuple(value.shape)}, "
                f"but the current mean model expects {tuple(mean_state[key].shape)}. "
                "Use USE_INTENSITY=0 with the legacy version_2 mean checkpoint."
            )

    missing = sorted(set(mean_state) - set(converted_state))
    if missing:
        raise RuntimeError(f"Legacy mean checkpoint {checkpoint_path} is missing tensors: {missing}")
    mean_model.load_state_dict(converted_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a legacy-style Q/U moment network with independent Q/U std trunks.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("DATASET_DIR", str(DEFAULT_DATASET_DIR))))
    parser.add_argument("--signal-dir", type=Path, default=Path(os.environ.get("PLANCK_SIGNAL_DIR", str(DEFAULT_SIGNAL_DIR))))
    parser.add_argument("--intensity-freq", type=int, default=int(os.environ.get("INTENSITY_FREQ", "857")))
    parser.add_argument("--use-intensity", type=str_to_bool, default=str_to_bool(os.environ.get("USE_INTENSITY", "0")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--patches", default=os.environ.get("PATCH_LIST", ""))
    parser.add_argument("--expected-patch-count", type=int, default=int(os.environ.get("EXPECTED_PATCH_COUNT", "0")))
    parser.add_argument("--train-fraction", type=float, default=float(os.environ.get("TRAIN_FRACTION", "0.9")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "11")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    parser.add_argument("--epochs-mean", type=int, default=int(os.environ.get("N_EPOCHS_MEAN", "10")))
    parser.add_argument("--epochs-full", type=int, default=int(os.environ.get("N_EPOCHS_FULL", "90")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "3e-5")))
    parser.add_argument("--variance-learning-rate", type=float, default=float(os.environ.get("VARIANCE_LEARNING_RATE", os.environ.get("LEARNING_RATE", "3e-5"))))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "1e-6")))
    parser.add_argument("--residual-floor", type=float, default=float(os.environ.get("RESIDUAL_FLOOR", "1e-8")))
    parser.add_argument("--normalize", type=str_to_bool, default=str_to_bool(os.environ.get("NORMALIZE", "1")))
    parser.add_argument("--max-train-batches", type=int, default=int(os.environ.get("MAX_TRAIN_BATCHES", "0")))
    parser.add_argument("--max-val-batches", type=int, default=int(os.environ.get("MAX_VAL_BATCHES", "0")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "5")))
    parser.add_argument("--resume", type=Path, default=Path(os.environ["RESUME"]) if os.environ.get("RESUME") else None)
    parser.add_argument("--legacy-mean-checkpoint", type=Path, default=Path(os.environ["LEGACY_MEAN_CHECKPOINT"]) if os.environ.get("LEGACY_MEAN_CHECKPOINT") else None)
    parser.add_argument("--save-diagnostic-plots", type=str_to_bool, default=str_to_bool(os.environ.get("SAVE_DIAGNOSTIC_PLOTS", "1")))
    parser.add_argument("--diagnostic-patch", default=os.environ.get("DIAGNOSTIC_PATCH", "3"))
    parser.add_argument("--diagnostic-sample-index", type=int, default=int(os.environ.get("DIAGNOSTIC_SAMPLE_INDEX", "0")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser()
    args.signal_dir = args.signal_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.mean_output_channels = 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    patch_selector = parse_patch_selector(args.patches)
    files = discover_dataset_files(args.dataset_dir, patch_selector)
    if not files:
        raise FileNotFoundError(f"No *_moment_dataset_*.npz files found under {args.dataset_dir}")
    if args.expected_patch_count > 0 and len(files) != args.expected_patch_count:
        raise RuntimeError(
            f"Expected {args.expected_patch_count} dataset files for PATCH_LIST={args.patches!r}, "
            f"but found {len(files)} under {args.dataset_dir}"
        )
    train_sample_indices_by_file, val_sample_indices_by_file, sample_counts_by_file = split_samples(
        files,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    train_files = [
        path for path in files
        if len(train_sample_indices_by_file.get(path.name, [])) > 0
    ]
    val_files = [
        path for path in files
        if len(val_sample_indices_by_file.get(path.name, [])) > 0
    ]
    args.sample_split = serializable_sample_split(
        train_sample_indices_by_file,
        val_sample_indices_by_file,
        sample_counts_by_file,
    )
    sample_split_manifest_path = args.output_dir / "sample_split_manifest.json"
    save_sample_split_manifest(
        sample_split_manifest_path,
        files=files,
        train_indices=train_sample_indices_by_file,
        val_indices=val_sample_indices_by_file,
        sample_counts=sample_counts_by_file,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    args.sample_split_manifest = str(sample_split_manifest_path)
    n_input = 3 if args.use_intensity else 2

    stats_path = args.output_dir / "normalization_stats.npz"
    if args.normalize:
        if stats_path.exists():
            loaded = np.load(stats_path)
            loaded_split_unit = (
                str(np.asarray(loaded["split_unit"]).item())
                if "split_unit" in loaded
                else None
            )
            if (
                loaded_split_unit == "sample"
                and loaded["input_mean"].shape == (n_input,)
                and loaded["target_mean"].shape == (2,)
            ):
                stats = ChannelStats(
                    input_mean=loaded["input_mean"].astype(np.float32),
                    input_std=loaded["input_std"].astype(np.float32),
                    target_mean=loaded["target_mean"].astype(np.float32),
                    target_std=loaded["target_std"].astype(np.float32),
                )
                print(f"Loaded normalization stats from {stats_path}", flush=True)
            else:
                print(f"Ignoring incompatible normalization stats at {stats_path}; recomputing", flush=True)
                stats = compute_channel_stats(
                    train_files,
                    use_intensity=args.use_intensity,
                    signal_dir=args.signal_dir,
                    intensity_freq=args.intensity_freq,
                    sample_indices_by_file=train_sample_indices_by_file,
                )
                save_stats(stats_path, stats)
                print(f"Saved normalization stats to {stats_path}", flush=True)
        else:
            print("Computing normalization stats from training files", flush=True)
            stats = compute_channel_stats(
                train_files,
                use_intensity=args.use_intensity,
                signal_dir=args.signal_dir,
                intensity_freq=args.intensity_freq,
                sample_indices_by_file=train_sample_indices_by_file,
            )
            save_stats(stats_path, stats)
            print(f"Saved normalization stats to {stats_path}", flush=True)
    else:
        stats = identity_channel_stats(n_input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_dataset = MomentPatchIterableDataset(
        train_files,
        stats=stats,
        use_intensity=args.use_intensity,
        signal_dir=args.signal_dir,
        intensity_freq=args.intensity_freq,
        sample_indices_by_file=train_sample_indices_by_file,
        seed=args.seed,
        shuffle_files=True,
        shuffle_samples=True,
    )
    val_dataset = MomentPatchIterableDataset(
        val_files,
        stats=stats,
        use_intensity=args.use_intensity,
        signal_dir=args.signal_dir,
        intensity_freq=args.intensity_freq,
        sample_indices_by_file=val_sample_indices_by_file,
        seed=args.seed + 100000,
        shuffle_files=False,
        shuffle_samples=False,
    )
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    model = SplitStdMomentModel(in_channels=n_input).to(device)
    mean_optimizer = torch.optim.Adam(model.mean_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    variance_parameters = list(model.std_q_model.parameters()) + list(model.std_u_model.parameters())
    full_optimizer = torch.optim.Adam(
        variance_parameters,
        lr=args.variance_learning_rate,
        weight_decay=args.weight_decay,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    stages: list[str] = []
    best_val_loss = math.inf
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume.expanduser(), map_location=device)
        if checkpoint.get("split_unit") != "sample" or "sample_split" not in checkpoint:
            raise RuntimeError(
                f"Refusing to resume {args.resume}: checkpoint was not produced "
                "by the sample-level split training code. Start from scratch or "
                "resume a sample-split checkpoint."
            )
        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise RuntimeError(f"Resume checkpoint {args.resume} is not a split-std Q/U checkpoint")
        try:
            model.load_state_dict(model_state)
        except RuntimeError as exc:
            raise RuntimeError(f"Resume checkpoint {args.resume} is not compatible with split-std moment training") from exc
        if "mean_optimizer_state_dict" in checkpoint:
            mean_optimizer.load_state_dict(checkpoint["mean_optimizer_state_dict"])
        if "full_optimizer_state_dict" in checkpoint:
            full_optimizer.load_state_dict(checkpoint["full_optimizer_state_dict"])
        train_losses = [float(v) for v in checkpoint.get("train_losses", [])]
        val_losses = [float(v) for v in checkpoint.get("val_losses", [])]
        stages = [str(v) for v in checkpoint.get("stages", [])]
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from {args.resume} at completed epoch {start_epoch}", flush=True)
    elif args.legacy_mean_checkpoint is not None:
        if args.use_intensity:
            raise RuntimeError("LEGACY_MEAN_CHECKPOINT from version_2 is compatible only with USE_INTENSITY=0")
        load_legacy_mean_model(model.mean_model, args.legacy_mean_checkpoint, device=device)
        print(
            f"Loaded legacy mean from {args.legacy_mean_checkpoint}; "
            "using only legacy output channels 0:2 to initialize the mean trunk",
            flush=True,
        )

    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None

    best_path = model_dir / "moment_network_joint_best.pth"
    final_path = model_dir / "moment_network_joint_final.pth"
    latest_path = model_dir / "moment_network_joint_latest.pth"
    mean_stage_path = model_dir / "moment_network_joint_mean_stage.pth"
    history_path = model_dir / "moment_network_joint_history.npz"
    loss_plot_path = model_dir / "moment_network_joint_loss.png"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_observed: np.ndarray | None = None
    diagnostic_source: str | None = None
    if args.save_diagnostic_plots:
        try:
            diagnostic_observed, diagnostic_source = load_fixed_diagnostic_signal(
                use_intensity=args.use_intensity,
                intensity_freq=args.intensity_freq,
            )
            print(
                f"diagnostic_plots=enabled patch={DIAGNOSTIC_PATCH} source={diagnostic_source}",
                flush=True,
            )
        except Exception as exc:
            diagnostic_observed = None
            print(f"diagnostic_plots=disabled reason={exc}", flush=True)

    def save_state_plot(plot_path: Path, *, stage: str, epoch: int, train_loss: float | None, val_loss: float | None) -> None:
        if diagnostic_observed is None:
            return
        save_diagnostic_inference_plot(
            plot_path,
            model=model,
            observed=diagnostic_observed,
            stats=stats,
            device=device,
            patch=DIAGNOSTIC_PATCH,
            stage=stage,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"dataset_dir={args.dataset_dir}", flush=True)
    print(f"use_intensity={args.use_intensity}", flush=True)
    if args.use_intensity:
        print(f"signal_dir={args.signal_dir}", flush=True)
        print(f"intensity_freq={args.intensity_freq}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print("split_unit=sample", flush=True)
    print(f"patch_files={len(files)} train_files={len(train_files)} val_files={len(val_files)}", flush=True)
    print(
        f"train_samples={args.sample_split['train_sample_count']} "
        f"val_samples={args.sample_split['val_sample_count']}",
        flush=True,
    )
    print(f"sample_split_manifest={args.sample_split_manifest}", flush=True)
    print(f"batch_size={args.batch_size} workers={args.num_workers}", flush=True)
    print(f"mean_model_output_channels={args.mean_output_channels}", flush=True)
    print("std_architecture=independent_q_u_unet_trunks", flush=True)
    print("stage_2_objective=0.5 * mean(((target - mean)**2 - exp(logvar))**2)", flush=True)
    print("stage_2_updates_mean=False", flush=True)
    print("stage_2_mean_fixed=True", flush=True)
    print(f"epochs_mean={args.epochs_mean} epochs_full={args.epochs_full}", flush=True)

    epoch_number = start_epoch
    schedule = [("mean", args.epochs_mean), ("full", args.epochs_full)]
    completed_by_stage = {stage_name: stages.count(stage_name) for stage_name, _ in schedule}
    for stage, n_epochs in schedule:
        remaining_epochs = max(0, n_epochs - completed_by_stage.get(stage, 0))
        for _ in range(remaining_epochs):
            epoch_number += 1
            train_dataset.set_epoch(epoch_number)
            val_dataset.set_epoch(epoch_number)
            epoch_start = time.perf_counter()
            print(f"[{stage}] epoch={epoch_number:04d} start={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            if stage == "mean":
                train_loss = train_mean_epoch(
                    model,
                    train_loader,
                    mean_optimizer,
                    device=device,
                    max_batches=max_train_batches,
                )
                val_loss = validate_mean_epoch(
                    model,
                    val_loader,
                    device=device,
                    max_batches=max_val_batches,
                )
            elif stage == "full":
                train_loss = train_full_moment_epoch(
                    model,
                    train_loader,
                    full_optimizer,
                    device=device,
                    max_batches=max_train_batches,
                )
                val_loss = validate_full_moment_epoch(
                    model,
                    val_loader,
                    device=device,
                    max_batches=max_val_batches,
                )
            else:
                raise ValueError(f"Unsupported stage: {stage}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            stages.append(stage)
            save_history(history_path, train_losses, val_losses, stages)
            save_loss_plot(loss_plot_path, train_losses, val_losses, stages)

            status = ""
            if (stage == "full" or args.epochs_full == 0) and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path,
                    model=model,
                    mean_optimizer=mean_optimizer,
                    full_optimizer=full_optimizer,
                    epoch=epoch_number,
                    stage=stage,
                    best_val_loss=best_val_loss,
                    args=args,
                    train_files=train_files,
                    val_files=val_files,
                    stats=stats,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    stages=stages,
                )
                save_state_plot(
                    model_dir / f"moment_network_joint_best_patch{DIAGNOSTIC_PATCH}.png",
                    stage=stage,
                    epoch=epoch_number,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                status = " new_best"

            save_checkpoint(
                latest_path,
                model=model,
                mean_optimizer=mean_optimizer,
                full_optimizer=full_optimizer,
                epoch=epoch_number,
                stage=stage,
                best_val_loss=best_val_loss,
                args=args,
                train_files=train_files,
                val_files=val_files,
                stats=stats,
                train_losses=train_losses,
                val_losses=val_losses,
                stages=stages,
            )
            save_state_plot(
                model_dir / f"moment_network_joint_latest_patch{DIAGNOSTIC_PATCH}.png",
                stage=stage,
                epoch=epoch_number,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            if args.save_every > 0 and epoch_number % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"moment_network_joint_epoch_{epoch_number:04d}_{stage}.pth"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    mean_optimizer=mean_optimizer,
                    full_optimizer=full_optimizer,
                    epoch=epoch_number,
                    stage=stage,
                    best_val_loss=best_val_loss,
                    args=args,
                    train_files=train_files,
                    val_files=val_files,
                    stats=stats,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    stages=stages,
                )
                print(f"Saved intermediate checkpoint to {checkpoint_path}", flush=True)
                save_state_plot(
                    checkpoint_path.with_name(f"{checkpoint_path.stem}_patch{DIAGNOSTIC_PATCH}.png"),
                    stage=stage,
                    epoch=epoch_number,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

            if stage == "mean" and stages.count("mean") == args.epochs_mean:
                save_checkpoint(
                    mean_stage_path,
                    model=model,
                    mean_optimizer=mean_optimizer,
                    full_optimizer=full_optimizer,
                    epoch=epoch_number,
                    stage=stage,
                    best_val_loss=best_val_loss,
                    args=args,
                    train_files=train_files,
                    val_files=val_files,
                    stats=stats,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    stages=stages,
                )
                print(f"Saved mean-stage checkpoint to {mean_stage_path}", flush=True)
                save_state_plot(
                    model_dir / f"moment_network_joint_mean_stage_patch{DIAGNOSTIC_PATCH}.png",
                    stage=stage,
                    epoch=epoch_number,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

            print(
                f"[{stage}] epoch={epoch_number:04d} train={train_loss:.6e} val={val_loss:.6e} "
                f"elapsed_s={time.perf_counter() - epoch_start:.1f}{status}",
                flush=True,
            )

    save_checkpoint(
        final_path,
        model=model,
        mean_optimizer=mean_optimizer,
        full_optimizer=full_optimizer,
        epoch=epoch_number,
        stage=stages[-1] if stages else "none",
        best_val_loss=best_val_loss,
        args=args,
        train_files=train_files,
        val_files=val_files,
        stats=stats,
        train_losses=train_losses,
        val_losses=val_losses,
        stages=stages,
    )
    save_history(history_path, train_losses, val_losses, stages)
    save_loss_plot(loss_plot_path, train_losses, val_losses, stages)
    save_state_plot(
        model_dir / f"moment_network_joint_final_patch{DIAGNOSTIC_PATCH}.png",
        stage=stages[-1] if stages else "none",
        epoch=epoch_number,
        train_loss=train_losses[-1] if train_losses else None,
        val_loss=val_losses[-1] if val_losses else None,
    )
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_format": "joint_qu_i_split_std_moment_mse" if args.use_intensity else "joint_qu_split_std_moment_mse",
                "dataset_dir": str(args.dataset_dir),
                "use_intensity": bool(args.use_intensity),
                "signal_dir": str(args.signal_dir) if args.use_intensity else None,
                "intensity_freq": args.intensity_freq if args.use_intensity else None,
                "output_dir": str(args.output_dir),
                "input_channels": ["Q_contaminated", "U_contaminated", f"I{args.intensity_freq}"] if args.use_intensity else ["Q_contaminated", "U_contaminated"],
                "target_channels": ["Q_clean", "U_clean"],
                "variance_target": "MSE between squared residuals and predicted Q/U variances",
                "stage_2_objective": "0.5 * mean(((target - mean)**2 - exp(logvar))**2)",
                "stage_2_updates_mean": False,
                "stage_2_mean_fixed": True,
                "std_architecture": "independent_q_u_unet_trunks",
                "split_unit": "sample",
                "train_fraction": args.train_fraction,
                "n_files": len(files),
                "n_train_files": len(train_files),
                "n_val_files": len(val_files),
                "n_total_samples": int(sum(sample_counts_by_file.values())) if sample_counts_by_file else None,
                "n_train_samples": args.sample_split["train_sample_count"],
                "n_val_samples": args.sample_split["val_sample_count"],
                "sample_split_manifest": args.sample_split_manifest,
                "train_files": [p.name for p in train_files],
                "val_files": [p.name for p in val_files],
                "best_model": str(best_path),
                "mean_stage_model": str(mean_stage_path),
                "final_model": str(final_path),
                "latest_model": str(latest_path),
                "history": str(history_path),
                "loss_plot": str(loss_plot_path),
                "diagnostic_plots": {
                    "enabled": diagnostic_observed is not None,
                    "patch": DIAGNOSTIC_PATCH,
                    "sample_index": None,
                    "source": diagnostic_source,
                    "best_plot": str(model_dir / f"moment_network_joint_best_patch{DIAGNOSTIC_PATCH}.png"),
                    "latest_plot": str(model_dir / f"moment_network_joint_latest_patch{DIAGNOSTIC_PATCH}.png"),
                    "final_plot": str(model_dir / f"moment_network_joint_final_patch{DIAGNOSTIC_PATCH}.png"),
                    "mean_stage_plot": str(model_dir / f"moment_network_joint_mean_stage_patch{DIAGNOSTIC_PATCH}.png"),
                },
                "normalization_stats": str(stats_path) if args.normalize else None,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "config": vars(args),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="ascii",
    )
    removed_checkpoints = cleanup_intermediate_checkpoints(checkpoint_dir)
    if removed_checkpoints:
        print(f"Deleted {removed_checkpoints} intermediate checkpoints from {checkpoint_dir}", flush=True)
    final_stage = stages[-1] if stages else "none"
    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None
    print(
        f"Final loss: stage={final_stage} train={final_train_loss} val={final_val_loss} best_val={best_val_loss}",
        flush=True,
    )
    print(f"Saved final model to {final_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
