#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PLANCK_PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PLANCK_PROJECT_DIR.parent
REPO_ROOT = Path(os.environ.get("STL_DEV_ROOT", str(DEFAULT_REPO_ROOT))).expanduser()
for path in (REPO_ROOT, PLANCK_PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infer_network import (  # noqa: E402
    infer_joint_map,
    infer_single_map,
    infer_two_stage_joint_map,
    load_joint_model,
    load_model,
    load_signal_i,
    load_signal_qu,
)
from utils import SIGNAL_DIR  # noqa: E402


DEFAULT_MODEL_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_training/version_2/models")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/mn_full_sky/version_2")


def parse_patch_list(text: str) -> list[str]:
    patches: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            patches.extend(range(start, end + step, step))
        else:
            patches.append(int(chunk))
    unique = sorted(set(patches))
    if not unique:
        raise ValueError(f"Empty patch list from {text!r}")
    return [str(patch) for patch in unique]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run moment-network inference over many Planck patches and save one "
            "(2,H,W) mean array and one (2,H,W) std array per patch."
        )
    )
    parser.add_argument("--patch-list", default=os.environ.get("PATCH_LIST", "0-191"))
    parser.add_argument("--freq", type=int, default=int(os.environ.get("FREQ", "353")))
    parser.add_argument("--intensity-freq", type=int, default=int(os.environ.get("INTENSITY_FREQ", "857")))
    parser.add_argument("--map-size", type=int, default=int(os.environ.get("MAP_SIZE", "384")))
    parser.add_argument("--signal-dir", type=Path, default=Path(os.environ.get("PLANCK_SIGNAL_DIR", str(SIGNAL_DIR))))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))))
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--network-mode", choices=("joint", "separate"), default=os.environ.get("NETWORK_MODE", "joint"))
    parser.add_argument("--q-model", type=Path, default=Path(os.environ["Q_MODEL_PATH"]) if os.environ.get("Q_MODEL_PATH") else None)
    parser.add_argument("--u-model", type=Path, default=Path(os.environ["U_MODEL_PATH"]) if os.environ.get("U_MODEL_PATH") else None)
    parser.add_argument(
        "--joint-model",
        type=Path,
        default=Path(os.environ["JOINT_MODEL_PATH"]) if os.environ.get("JOINT_MODEL_PATH") else None,
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patches = parse_patch_list(args.patch_list)
    signal_dir = args.signal_dir.expanduser()
    model_dir = args.model_dir.expanduser()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.strip():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_model_path = args.q_model.expanduser() if args.q_model is not None else model_dir / "moment_network_q_best.pth"
    u_model_path = args.u_model.expanduser() if args.u_model is not None else model_dir / "moment_network_u_best.pth"
    joint_model_path = args.joint_model.expanduser() if args.joint_model is not None else model_dir / "moment_network_joint_best.pth"

    q_model = u_model = None
    q_stats = u_stats = None
    joint_model = None
    joint_stats = None
    checkpoint_info: dict[str, object] = {}

    if args.network_mode == "separate":
        q_model, q_stats, q_checkpoint = load_model(q_model_path, device=device, expected_stokes="Q")
        u_model, u_stats, u_checkpoint = load_model(u_model_path, device=device, expected_stokes="U")
        checkpoint_info = {
            "q_model": str(q_model_path),
            "u_model": str(u_model_path),
            "q_checkpoint_epoch": q_checkpoint.get("epoch"),
            "u_checkpoint_epoch": u_checkpoint.get("epoch"),
            "q_checkpoint_best_val_loss": q_checkpoint.get("best_val_loss"),
            "u_checkpoint_best_val_loss": u_checkpoint.get("best_val_loss"),
        }
    else:
        joint_model, joint_stats, joint_checkpoint = load_joint_model(joint_model_path, device=device)
        checkpoint_info = {
            "joint_model": str(joint_model_path),
            "joint_model_kind": joint_model["kind"],
            "joint_checkpoint_epoch": joint_checkpoint.get("epoch"),
            "joint_checkpoint_best_val_loss": joint_checkpoint.get("best_val_loss"),
        }

    print(f"device={device}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"network_mode={args.network_mode}", flush=True)
    print(f"patch_count={len(patches)}", flush=True)
    print(f"signal_dir={signal_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    started = time.time()
    failures: list[dict[str, str]] = []
    completed: list[str] = []

    for index, patch in enumerate(patches, start=1):
        patch_start = time.time()
        try:
            observed, signal_paths = load_signal_qu(signal_dir, patch=patch, freq=args.freq, map_size=args.map_size)

            if args.network_mode == "separate":
                assert q_model is not None and u_model is not None
                assert q_stats is not None and u_stats is not None
                mean_q, std_q, _ = infer_single_map(q_model, observed[0], q_stats, device=device)
                mean_u, std_u, _ = infer_single_map(u_model, observed[1], u_stats, device=device)
                mean = np.stack([mean_q, mean_u], axis=0).astype(np.float32)
                std = np.stack([std_q, std_u], axis=0).astype(np.float32)
            else:
                assert joint_model is not None and joint_stats is not None
                if joint_model["kind"] == "two_stage":
                    i_map, i_paths = load_signal_i(
                        signal_dir,
                        patch=patch,
                        intensity_freq=args.intensity_freq,
                        map_size=args.map_size,
                    )
                    signal_paths.update(i_paths)
                    observed_for_model = np.concatenate([observed, i_map[None]], axis=0)
                    mean, std, _ = infer_two_stage_joint_map(joint_model, observed_for_model, joint_stats, device=device)
                elif joint_model["kind"] == "two_stage_qu":
                    mean, std, _ = infer_two_stage_joint_map(joint_model, observed, joint_stats, device=device)
                else:
                    mean, std, _ = infer_joint_map(joint_model["model"], observed, joint_stats, device=device)

            if mean.shape != (2, args.map_size, args.map_size):
                raise RuntimeError(f"Mean has shape {mean.shape}; expected (2, {args.map_size}, {args.map_size})")
            if std.shape != (2, args.map_size, args.map_size):
                raise RuntimeError(f"Std has shape {std.shape}; expected (2, {args.map_size}, {args.map_size})")

            mean_path = out_dir / f"patch_{patch}_mean.npy"
            std_path = out_dir / f"patch_{patch}_std.npy"
            np.save(mean_path, mean.astype(np.float32))
            np.save(std_path, std.astype(np.float32))
            completed.append(patch)
            print(
                f"[{index:03d}/{len(patches):03d}] patch={patch} saved "
                f"{mean_path.name} {std_path.name} elapsed_s={time.time() - patch_start:.1f}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"patch": patch, "error": repr(exc)})
            print(f"[{index:03d}/{len(patches):03d}] patch={patch} FAILED: {exc!r}", flush=True)

    summary = {
        "patch_list": patches,
        "completed": completed,
        "failures": failures,
        "freq": args.freq,
        "intensity_freq": args.intensity_freq,
        "map_size": args.map_size,
        "network_mode": args.network_mode,
        "signal_dir": str(signal_dir),
        "output_dir": str(out_dir),
        "device": str(device),
        "runtime_seconds": time.time() - started,
        **checkpoint_info,
    }
    summary_path = out_dir / "mn_full_sky_inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="ascii")
    print(f"Saved summary {summary_path}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} patches failed; see {summary_path}")


if __name__ == "__main__":
    main()
