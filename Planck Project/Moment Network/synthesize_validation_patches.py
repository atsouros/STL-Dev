#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
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

from make_dataset import (  # noqa: E402
    RunRecord,
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


DEFAULT_RESULTS_DIR = Path("/pscratch/sd/a/atsouros/STL/planck_results/version_2")
DEFAULT_NUISANCE_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/Moment Network/validation_syntheses")
DEFAULT_PATCH_LIST = "0-4,75-80,187-191"
SCRIPT_VERSION = "2026-06-23-validation-synthesis-v4"


def select_device(requested: str) -> torch.device:
    if requested.strip():
        device = torch.device(requested)
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        raise RuntimeError("Validation synthesis requires a CUDA GPU")
    if device.type != "cuda":
        raise RuntimeError(f"Validation synthesis requires CUDA, got {device}")
    torch.cuda.set_device(device)
    return device


def slurm_shard(patches: list[int]) -> list[int]:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))
    if size < 1 or rank < 0 or rank >= size:
        raise RuntimeError(f"Invalid Slurm rank configuration: rank={rank}, size={size}")
    assigned = patches[rank::size]
    print(f"rank={rank}/{size} assigned_patches={assigned}", flush=True)
    return assigned


def nuisance_pair_key(path: Path, *, stokes: str, freq: int) -> str:
    token = f"{stokes}{freq}"
    if token not in path.name:
        raise RuntimeError(
            f"Could not identify {token} in nuisance filename {path.name}"
        )
    return path.name.replace(token, "{STOKES}", 1)


def select_source_record(
    records: list[RunRecord],
    *,
    patch: int,
) -> RunRecord:
    if not records:
        raise RuntimeError(
            f"No component-separation result found for patch {patch}"
        )
    if len(records) == 1:
        return records[0]

    candidate_text = "\n".join(
        f"  - {record.npy_path.name}" for record in records
    )
    selected = max(
        records,
        key=lambda record: (record.json_path.stat().st_mtime_ns, record.stem),
    )
    print(
        f"WARNING patch={patch}: found multiple component-separation results; "
        f"selecting newest result "
        f"{selected.npy_path.name}.\n"
        f"Candidates:\n{candidate_text}",
        flush=True,
    )
    return selected


def save_product(
    output_dir: Path,
    *,
    patch: int,
    truth: np.ndarray,
    observed_qu: np.ndarray,
    source_npy: Path,
    source_json: Path,
    q_noise_paths: list[Path],
    u_noise_paths: list[Path],
    noise_index: int,
    noise_qu: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    output_path = output_dir / f"patch_{patch}_validation_synthesis.npz"
    temporary_path = output_dir / f".patch_{patch}_validation_synthesis.tmp.npz"
    np.savez_compressed(
        temporary_path,
        truth=truth.astype(np.float32),
        noise_qu=noise_qu.astype(np.float32),
        observed_qu=observed_qu.astype(np.float32),
        patch=np.asarray(patch, dtype=np.int32),
        source_result=np.asarray(str(source_npy)),
        source_metadata=np.asarray(str(source_json)),
        q_noise_file=np.asarray(str(q_noise_paths[noise_index])),
        u_noise_file=np.asarray(str(u_noise_paths[noise_index])),
        noise_index=np.asarray(noise_index, dtype=np.int32),
        synthesis_seed=np.asarray(args.synthesis_seed + patch * 10, dtype=np.int64),
        noise_seed=np.asarray(args.noise_seed + patch, dtype=np.int64),
    )
    temporary_path.replace(output_path)

    summary = {
        "script_version": SCRIPT_VERSION,
        "patch": patch,
        "output": str(output_path),
        "truth_shape": list(truth.shape),
        "observed_qu_shape": list(observed_qu.shape),
        "source_result": str(source_npy),
        "source_metadata": str(source_json),
        "noise_preprocessing": "2x2 mean pooling from native nuisance resolution",
        "noise_index": noise_index,
        "q_noise_file": str(q_noise_paths[noise_index]),
        "u_noise_file": str(u_noise_paths[noise_index]),
        "q_noise_files": [str(path) for path in q_noise_paths],
        "u_noise_files": [str(path) for path in u_noise_paths],
        "synthesis_seed": args.synthesis_seed + patch * 10,
        "noise_seed": args.noise_seed + patch,
        "synthesis_pbc": args.synthesis_pbc,
        "synthesis_running_shape": list(args.synthesis_running_shape)
        if args.synthesis_running_shape is not None
        else None,
        "synthesis_compute_ps": args.synthesis_compute_ps,
        "synthesis_max_iter": args.synthesis_max_iter,
        "cross_matrix": args.cross_matrix,
    }
    summary_path = output_dir / f"patch_{patch}_validation_synthesis.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return output_path


def synthesize_patch(patch: int, args: argparse.Namespace, device: torch.device) -> Path:
    output_path = args.output_dir / f"patch_{patch}_validation_synthesis.npz"
    if output_path.exists() and not args.overwrite:
        print(f"patch={patch} already exists; skipping {output_path}", flush=True)
        return output_path

    records = discover_runs(args.results_dir, parse_patch_selector(str(patch)))
    record = select_source_record(
        records,
        patch=patch,
    )
    config = load_config(record.json_path)
    recovered_qu = np.load(record.npy_path).astype(np.float64)
    if recovered_qu.ndim != 3 or recovered_qu.shape[0] != 2:
        raise RuntimeError(
            f"Expected recovered Q/U map with shape (2,H,W), got {recovered_qu.shape}"
        )
    final_hw = tuple(int(value) for value in recovered_qu.shape[-2:])
    if args.expected_map_size > 0:
        expected_hw = (args.expected_map_size, args.expected_map_size)
        if final_hw != expected_hw:
            raise RuntimeError(
                f"Expected recovered patch {patch} to have spatial shape "
                f"{expected_hw}, got {final_hw}"
            )

    q_noise_paths = list_nuisance_paths(
        args.nuisance_dir, str(patch), "Q", args.freq, args.nuisance_version
    )
    u_noise_paths = list_nuisance_paths(
        args.nuisance_dir, str(patch), "U", args.freq, args.nuisance_version
    )
    q_noise_keys = [
        nuisance_pair_key(path, stokes="Q", freq=args.freq)
        for path in q_noise_paths
    ]
    u_noise_keys = [
        nuisance_pair_key(path, stokes="U", freq=args.freq)
        for path in u_noise_paths
    ]
    if q_noise_keys != u_noise_keys:
        raise RuntimeError(
            f"Patch {patch} Q/U nuisance files do not form aligned pairs:\n"
            f"Q={q_noise_keys}\nU={u_noise_keys}"
        )
    q_noise_bank = load_nuisance_bank(q_noise_paths, crop_size=None)
    u_noise_bank = load_nuisance_bank(u_noise_paths, crop_size=None)
    if tuple(q_noise_bank.shape[-2:]) != final_hw:
        raise RuntimeError(
            f"Patch {patch} Q nuisance shape {q_noise_bank.shape[-2:]} does not match {final_hw}"
        )
    if tuple(u_noise_bank.shape[-2:]) != final_hw:
        raise RuntimeError(
            f"Patch {patch} U nuisance shape {u_noise_bank.shape[-2:]} does not match {final_hw}"
        )

    print(
        f"patch={patch} device={device} result={record.npy_path.name} final_hw={final_hw}",
        flush=True,
    )
    synthesized = synthesize_joint_qu(
        recovered_qu,
        config,
        device=device,
        seed=args.synthesis_seed + patch * 10,
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

    if synthesized.shape[0] != 1:
        raise RuntimeError(
            f"Expected one joint synthesis for patch {patch}, got {synthesized.shape}"
        )
    truth = np.stack(
        [
            center_crop_to_hw(synthesized[0, 0], final_hw),
            center_crop_to_hw(synthesized[0, 1], final_hw),
        ],
        axis=0,
    ).astype(np.float32)
    rng = np.random.default_rng(args.noise_seed + patch)
    noise_index = int(rng.integers(0, len(q_noise_bank)))
    noise_qu = np.stack(
        [q_noise_bank[noise_index], u_noise_bank[noise_index]],
        axis=0,
    ).astype(np.float32)
    observed_qu = (truth + noise_qu).astype(np.float32)
    if not np.allclose(
        observed_qu.astype(np.float64),
        truth.astype(np.float64) + noise_qu.astype(np.float64),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            f"Internal validation failed for patch {patch}: "
            "observed_qu is not truth + noise_qu"
        )
    print(
        f"patch={patch} selected_noise_index={noise_index} "
        f"q_noise={q_noise_paths[noise_index].name} "
        f"u_noise={u_noise_paths[noise_index].name}",
        flush=True,
    )
    saved = save_product(
        args.output_dir,
        patch=patch,
        truth=truth,
        observed_qu=observed_qu,
        source_npy=record.npy_path,
        source_json=record.json_path,
        q_noise_paths=q_noise_paths,
        u_noise_paths=u_noise_paths,
        noise_index=noise_index,
        noise_qu=noise_qu,
        args=args,
    )
    print(f"patch={patch} saved={saved}", flush=True)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reusable synthetic truth/data pairs for posterior validation."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
    )
    parser.add_argument(
        "--nuisance-dir",
        type=Path,
        default=Path(os.environ.get("PLANCK_NUISANCE_DIR", str(DEFAULT_NUISANCE_DIR))),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument(
        "--patch-list",
        default=os.environ.get("PATCH_LIST", DEFAULT_PATCH_LIST),
    )
    parser.add_argument(
        "--expected-map-size",
        type=int,
        default=int(os.environ.get("EXPECTED_MAP_SIZE", "384")),
        help="Expected square output size; set to 0 to accept any recovered-map size.",
    )
    parser.add_argument("--freq", type=int, default=int(os.environ.get("FREQ", "353")))
    parser.add_argument(
        "--nuisance-version",
        default=os.environ.get("PLANCK_NUISANCE_VERSION", "v4_10_arcmin"),
    )
    parser.add_argument(
        "--synthesis-pbc",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("SYNTHESIS_PBC", "1")),
    )
    parser.add_argument(
        "--synthesis-compute-ps",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("SYNTHESIS_COMPUTE_PS", "0")),
    )
    parser.add_argument(
        "--synthesis-running-shape",
        type=parse_hw,
        default=parse_hw(os.environ.get("SYNTHESIS_RUNNING_SHAPE", "512")),
    )
    parser.add_argument(
        "--cross-matrix",
        type=parse_cross_matrix,
        default=parse_cross_matrix(os.environ.get("CROSS_MATRIX", "1,1;0,1")),
    )
    parser.add_argument(
        "--synthesis-max-iter",
        type=int,
        default=int(os.environ.get("SYNTHESIS_MAX_ITER", "500")),
    )
    parser.add_argument(
        "--synthesis-lr",
        type=float,
        default=float(os.environ.get("SYNTHESIS_LR", "1.0")),
    )
    parser.add_argument(
        "--synthesis-history-size",
        type=int,
        default=int(os.environ.get("SYNTHESIS_HISTORY_SIZE", "40")),
    )
    parser.add_argument(
        "--print-every-synthesis",
        type=int,
        default=int(os.environ.get("PRINT_EVERY_SYNTHESIS", "1")),
    )
    parser.add_argument(
        "--verbose-synthesis",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("VERBOSE_SYNTHESIS", "1")),
    )
    parser.add_argument(
        "--synthesis-seed",
        type=int,
        default=int(os.environ.get("SYNTHESIS_SEED", "10007")),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=int(os.environ.get("NOISE_SEED", "20007")),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    parser.add_argument(
        "--overwrite",
        type=str_to_bool,
        default=str_to_bool(os.environ.get("OVERWRITE", "0")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser()
    args.nuisance_dir = args.nuisance_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = parse_patch_selector(args.patch_list)
    if not selected:
        raise RuntimeError("PATCH_LIST did not select any patches")
    patches = sorted(int(patch) for patch in selected)
    device = select_device(args.device)

    print(f"script_version={SCRIPT_VERSION}", flush=True)
    print(f"all_selected_patches={patches}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    assigned = slurm_shard(patches)
    for patch in assigned:
        synthesize_patch(patch, args, device)


if __name__ == "__main__":
    main()
