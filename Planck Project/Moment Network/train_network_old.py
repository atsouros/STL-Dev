#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


SCRIPT_DIR = Path(__file__).resolve().parent
PLANCK_PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PLANCK_PROJECT_DIR.parent
REPO_ROOT = Path(os.environ.get("STL_DEV_ROOT", str(DEFAULT_REPO_ROOT))).expanduser()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_training/version_2")


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


def split_files(
    files: list[Path],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"TRAIN_FRACTION must be between 0 and 1, got {train_fraction}")
    if len(files) < 2:
        raise ValueError("Need at least two dataset files for a train/validation split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    split = int(round(train_fraction * len(files)))
    split = min(max(1, split), len(files) - 1)
    train_idx = sorted(int(i) for i in order[:split])
    val_idx = sorted(int(i) for i in order[split:])
    return [files[i] for i in train_idx], [files[i] for i in val_idx]


@dataclass(frozen=True)
class ChannelStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def compute_channel_stats(files: list[Path], eps: float = 1e-6) -> ChannelStats:
    input_sum = np.zeros(2, dtype=np.float64)
    input_sumsq = np.zeros(2, dtype=np.float64)
    target_sum = np.zeros(2, dtype=np.float64)
    target_sumsq = np.zeros(2, dtype=np.float64)
    count = 0

    for file_index, path in enumerate(files, start=1):
        with np.load(path) as data:
            x = np.stack([data["xq"], data["xu"]], axis=1).astype(np.float64, copy=False)
            y = np.stack([data["yq"], data["yu"]], axis=1).astype(np.float64, copy=False)
        pixels = int(np.prod(x.shape[0:1] + x.shape[2:]))
        target_sum += x.sum(axis=(0, 2, 3))
        target_sumsq += np.square(x).sum(axis=(0, 2, 3))
        input_sum += y.sum(axis=(0, 2, 3))
        input_sumsq += np.square(y).sum(axis=(0, 2, 3))
        count += pixels
        print(f"stats {file_index:04d}/{len(files):04d}: {path.name}", flush=True)

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


def identity_channel_stats() -> ChannelStats:
    return ChannelStats(
        input_mean=np.zeros(2, dtype=np.float32),
        input_std=np.ones(2, dtype=np.float32),
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
    )


class MomentPatchIterableDataset(IterableDataset):
    def __init__(
        self,
        files: list[Path],
        *,
        stats: ChannelStats,
        seed: int,
        shuffle_files: bool,
        shuffle_samples: bool,
    ):
        super().__init__()
        self.files = list(files)
        self.stats = stats
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
            with np.load(path) as data:
                x = np.stack([data["xq"], data["xu"]], axis=1).astype(np.float32, copy=False)
                y = np.stack([data["yq"], data["yu"]], axis=1).astype(np.float32, copy=False)

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
    def __init__(self, in_channels: int = 2, out_channels: int = 4):
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


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = logvar.clamp(min=-12.0, max=12.0)
    var = torch.exp(logvar).clamp_min(1e-8)
    return (0.5 * (logvar + (target - mu) ** 2 / var)).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    stage: str,
    max_batches: int | None,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(y)
        mu = output[:, 0:2]
        if stage == "mean":
            loss = F.mse_loss(mu, x)
        elif stage == "full":
            logvar = output[:, 2:4]
            loss = gaussian_nll(x, mu, logvar)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    stage: str,
    max_batches: int | None,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for y, x in loader:
        y = y.to(device=device, non_blocking=True)
        x = x.to(device=device, non_blocking=True)
        output = model(y)
        mu = output[:, 0:2]
        if stage == "mean":
            loss = F.mse_loss(mu, x)
        elif stage == "full":
            logvar = output[:, 2:4]
            loss = gaussian_nll(x, mu, logvar)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "stage": stage,
            "best_val_loss": float(best_val_loss),
            "input_channels": ["Q_contaminated", "U_contaminated"],
            "target_channels": ["Q_clean", "U_clean"],
            "output_channels": ["Q_mean", "U_mean", "Q_logvar", "U_logvar"],
            "dataset_dir": str(args.dataset_dir),
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


def cleanup_intermediate_checkpoints(checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        return 0
    removed = 0
    for path in sorted(checkpoint_dir.glob("moment_network_joint_epoch_*.pth")):
        path.unlink()
        removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one joint Q/U moment network on generated Planck patch datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get("DATASET_DIR", str(DEFAULT_DATASET_DIR))))
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
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "1e-6")))
    parser.add_argument("--normalize", type=str_to_bool, default=str_to_bool(os.environ.get("NORMALIZE", "1")))
    parser.add_argument("--max-train-batches", type=int, default=int(os.environ.get("MAX_TRAIN_BATCHES", "0")))
    parser.add_argument("--max-val-batches", type=int, default=int(os.environ.get("MAX_VAL_BATCHES", "0")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "5")))
    parser.add_argument("--resume", type=Path, default=Path(os.environ["RESUME"]) if os.environ.get("RESUME") else None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
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
    train_files, val_files = split_files(files, train_fraction=args.train_fraction, seed=args.seed)

    stats_path = args.output_dir / "normalization_stats.npz"
    if args.normalize:
        if stats_path.exists():
            loaded = np.load(stats_path)
            stats = ChannelStats(
                input_mean=loaded["input_mean"].astype(np.float32),
                input_std=loaded["input_std"].astype(np.float32),
                target_mean=loaded["target_mean"].astype(np.float32),
                target_std=loaded["target_std"].astype(np.float32),
            )
            print(f"Loaded normalization stats from {stats_path}", flush=True)
        else:
            print("Computing normalization stats from training files", flush=True)
            stats = compute_channel_stats(train_files)
            save_stats(stats_path, stats)
            print(f"Saved normalization stats to {stats_path}", flush=True)
    else:
        stats = identity_channel_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_dataset = MomentPatchIterableDataset(
        train_files,
        stats=stats,
        seed=args.seed,
        shuffle_files=True,
        shuffle_samples=True,
    )
    val_dataset = MomentPatchIterableDataset(
        val_files,
        stats=stats,
        seed=args.seed + 100000,
        shuffle_files=False,
        shuffle_samples=False,
    )
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = make_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    model = LSSUNet(in_channels=2, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_losses: list[float] = []
    val_losses: list[float] = []
    stages: list[str] = []
    best_val_loss = math.inf
    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume.expanduser(), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = [float(v) for v in checkpoint.get("train_losses", [])]
        val_losses = [float(v) for v in checkpoint.get("val_losses", [])]
        stages = [str(v) for v in checkpoint.get("stages", [])]
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from {args.resume} at completed epoch {start_epoch}", flush=True)

    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None

    best_path = model_dir / "moment_network_joint_best.pth"
    final_path = model_dir / "moment_network_joint_final.pth"
    latest_path = model_dir / "moment_network_joint_latest.pth"
    history_path = model_dir / "moment_network_joint_history.npz"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}", flush=True)
    print(f"dataset_dir={args.dataset_dir}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"patch_files={len(files)} train_files={len(train_files)} val_files={len(val_files)}", flush=True)
    print(f"batch_size={args.batch_size} workers={args.num_workers}", flush=True)
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
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                stage=stage,
                max_batches=max_train_batches,
            )
            val_loss = validate_epoch(
                model,
                val_loader,
                device=device,
                stage=stage,
                max_batches=max_val_batches,
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            stages.append(stage)
            save_history(history_path, train_losses, val_losses, stages)

            status = ""
            if stage == "full" and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
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
                status = " new_best"

            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
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

            if args.save_every > 0 and epoch_number % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"moment_network_joint_epoch_{epoch_number:04d}_{stage}.pth"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
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

            print(
                f"[{stage}] epoch={epoch_number:04d} train={train_loss:.6e} val={val_loss:.6e}{status}",
                flush=True,
            )

    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
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
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir),
                "output_dir": str(args.output_dir),
                "n_files": len(files),
                "n_train_files": len(train_files),
                "n_val_files": len(val_files),
                "train_files": [p.name for p in train_files],
                "val_files": [p.name for p in val_files],
                "best_model": str(best_path),
                "final_model": str(final_path),
                "latest_model": str(latest_path),
                "history": str(history_path),
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
    print(f"Saved final model to {final_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()