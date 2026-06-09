#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PLANCK_PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_REPO_ROOT = PLANCK_PROJECT_DIR.parent
REPO_ROOT = Path(os.environ.get("STL_DEV_ROOT", str(DEFAULT_REPO_ROOT))).expanduser()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from STL_main.Synthesis import optimize_from_maps
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as FFT_DataClass
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch as Kernel_DataClass
import STL_main.torch_backend as bk


DEFAULT_RESULTS_DIR = Path("/pscratch/sd/a/atsouros/STL/planck_results/version_2")
DEFAULT_NUISANCE_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance")
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2")


@dataclass(frozen=True)
class RunRecord:
    patch: str
    stem: str
    npy_path: Path
    json_path: Path


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def parse_hw(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip() == "":
        return None
    value = value.strip().lower()
    if "x" in value:
        left, right = value.split("x", 1)
        return (int(left), int(right))
    size = int(value)
    return (size, size)


def parse_cross_matrix(value: str) -> list[list[int]]:
    rows = []
    for row in value.strip().split(";"):
        row = row.strip()
        if not row:
            continue
        rows.append([int(token.strip()) for token in row.split(",") if token.strip()])
    if not rows:
        raise ValueError("Cross matrix cannot be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Cross matrix rows must have the same length: {value!r}")
    return rows


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


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


def discover_runs(results_dir: Path, patch_selector: set[str] | None) -> list[RunRecord]:
    records: list[RunRecord] = []
    for npy_path in sorted(results_dir.glob("*.npy")):
        if npy_path.name.endswith("_checkpoint.npy"):
            continue
        patch = patch_from_name(npy_path)
        if patch is None:
            continue
        if patch_selector is not None and patch not in patch_selector:
            continue
        json_path = npy_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Skipping {npy_path.name}: missing {json_path.name}", flush=True)
            continue
        records.append(RunRecord(patch=patch, stem=npy_path.stem, npy_path=npy_path, json_path=json_path))

    records.sort(key=lambda rec: int(rec.patch))
    return records


def shard_records(records: list[RunRecord], enabled: bool) -> list[RunRecord]:
    if not enabled:
        return records

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_count = int(
        os.environ.get(
            "SLURM_ARRAY_TASK_COUNT",
            os.environ.get("SLURM_ARRAY_TASK_MAX", "0"),
        )
    )
    if task_count <= 0:
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", str(task_id)))
        task_count = task_max - task_min + 1
    task_index = task_id - task_min

    if task_index < 0 or task_index >= task_count:
        raise RuntimeError(
            f"Invalid Slurm array shard: index={task_index}, count={task_count}, task_id={task_id}"
        )

    return records[task_index::task_count]


def shard_records_by_slurm_tasks(records: list[RunRecord], enabled: bool) -> list[RunRecord]:
    if not enabled:
        return records

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))
    if size <= 1:
        return records
    if rank < 0 or rank >= size:
        raise RuntimeError(f"Invalid Slurm task shard: rank={rank}, size={size}")
    return records[rank::size]


def configure_backend_defaults(device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64 if dtype == torch.float32 else torch.complex128


def choose_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in {"float64", "fp64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_name}")


def maybe_set_wtype(st_op, ref_dc, wtype: str) -> None:
    try:
        st_op.wavelet_op = ref_dc.get_wavelet_op(J=st_op.J, L=st_op.L, WType=wtype)
        st_op.WType = getattr(st_op.wavelet_op, "WType", wtype)
    except TypeError:
        pass


def build_single_channel_st_op(
    config: dict[str, object],
    example_map: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pbc: bool,
    J: int | None = None,
    n_bins: int | None = None,
):
    configure_backend_defaults(device, dtype)
    backend = str(config.get("backend", "fft")).lower()
    data_class = FFT_DataClass if backend == "fft" else Kernel_DataClass

    example_tensor = torch.from_numpy(np.asarray(example_map)).to(device=device, dtype=dtype)
    ref_dc = data_class(example_tensor, pbc=pbc)

    ps_kwargs = {}
    if n_bins is not None:
        ps_kwargs["n_bins"] = int(n_bins)
    elif config.get("st_ps_n_bins") is not None:
        ps_kwargs["n_bins"] = int(config["st_ps_n_bins"])
    if backend == "fft":
        ps_kwargs["power_spectrum_method"] = str(config.get("st_ps_method", "legacy"))

    J_value = J if J is not None else int(config.get("st_j", 7))

    if bool(config.get("st_reduced", False)):
        st_op = ref_dc.get_ST_op(
            J=J_value,
            L=int(config.get("st_l", 4)),
            iso=bool(config.get("st_iso", False)),
            angular_ft=bool(config.get("st_angular_ft", False)),
            scale_ft=bool(config.get("st_scale_ft", False)),
            harmonics_angle=int(config.get("st_harmonics_angle", 2)),
            harmonics_scale=int(config.get("st_harmonics_scale", 3)),
            dj=int(config.get("st_dj", 3)),
            compute_PS=bool(config.get("st_compute_ps", True)),
            has_fewer_convolutions=bool(config.get("st_has_fewer_convolutions", False)),
            **ps_kwargs,
        )
    else:
        st_op = ref_dc.get_ST_op(
            J=J,
            compute_PS=bool(config.get("st_compute_ps", True)),
            has_fewer_convolutions=bool(config.get("st_has_fewer_convolutions", False)),
            **ps_kwargs,
        )

    maybe_set_wtype(st_op=st_op, ref_dc=ref_dc, wtype=str(config.get("wtype", "Bump-Steerable")))
    return data_class, ref_dc, st_op


def synthesize_single_channel(
    reference_map: np.ndarray,
    config: dict[str, object],
    *,
    device: torch.device,
    seed: int,
    total_syntheses: int,
    synthesis_batch_size: int,
    synthesis_pbc: bool,
    running_shape: tuple[int, int] | None,
    compute_ps: bool,
    max_iter: int,
    lr: float,
    history_size: int,
    print_iter: int,
    verbose: bool,
) -> np.ndarray:
    dtype = choose_torch_dtype(str(config.get("dtype", "float64")))
    target_pbc = bool(config.get("pbc", False))
    _, ref_dc, st_op_target = build_single_channel_st_op(
        config,
        reference_map,
        device=device,
        dtype=dtype,
        pbc=target_pbc,
    )
    running_example = (
        np.zeros(running_shape, dtype=np.float64)
        if running_shape is not None
        else reference_map
    )
    _, running_dc, st_op_running = build_single_channel_st_op(
        config,
        running_example,
        device=device,
        dtype=dtype,
        pbc=synthesis_pbc,
        J=st_op_target.J,
        n_bins=st_op_target.n_bins,
    )

    del running_dc
    compute_cross_matrix = torch.ones((1, 1), dtype=torch.bool, device=device)
    syntheses: list[np.ndarray] = []
    synth_seed = seed

    while len(syntheses) < total_syntheses:
        remaining = total_syntheses - len(syntheses)
        current_batch_size = min(synthesis_batch_size, remaining)
        generated = optimize_from_maps(
            target=ref_dc,
            st_op_target=st_op_target,
            st_op_running=st_op_running,
            nbatch=current_batch_size,
            pbc_running=synthesis_pbc,
            running_shape=running_shape,
            has_fewer_convolutions=bool(config.get("st_has_fewer_convolutions", False)),
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_ps,
            mean_field=True,
            max_iter=max_iter,
            lr=lr,
            history_size=history_size,
            print_iter=max(1, print_iter),
            verbose=verbose,
            seed=synth_seed,
        )
        generated_np = generated.detach().cpu().numpy() if isinstance(generated, torch.Tensor) else np.asarray(generated)
        if generated_np.ndim == 2:
            generated_np = generated_np[None, ...]
        syntheses.extend(list(generated_np))
        print(
            f"synthesis seed={synth_seed} batch={generated_np.shape[0]} total={len(syntheses)}/{total_syntheses}",
            flush=True,
        )
        synth_seed += 1

    return np.stack(syntheses[:total_syntheses], axis=0)


def synthesize_joint_qu(
    reference_qu: np.ndarray,
    config: dict[str, object],
    *,
    device: torch.device,
    seed: int,
    total_syntheses: int,
    synthesis_batch_size: int,
    synthesis_pbc: bool,
    running_shape: tuple[int, int] | None,
    compute_ps: bool,
    cross_matrix: list[list[int]],
    max_iter: int,
    lr: float,
    history_size: int,
    print_iter: int,
    verbose: bool,
) -> np.ndarray:
    if reference_qu.shape[0] != 2 or reference_qu.ndim != 3:
        raise ValueError(f"Expected reference_qu shape (2,H,W), got {reference_qu.shape}")

    dtype = choose_torch_dtype(str(config.get("dtype", "float64")))
    target_pbc = bool(config.get("pbc", False))
    _, ref_dc, st_op_target = build_single_channel_st_op(
        config,
        reference_qu,
        device=device,
        dtype=dtype,
        pbc=target_pbc,
    )
    running_example = (
        np.zeros((2, *running_shape), dtype=np.float64)
        if running_shape is not None
        else reference_qu
    )
    _, running_dc, st_op_running = build_single_channel_st_op(
        config,
        running_example,
        device=device,
        dtype=dtype,
        pbc=synthesis_pbc,
        J=st_op_target.J,
        n_bins=st_op_target.n_bins,
    )

    del running_dc
    cross_matrix_np = np.asarray(cross_matrix, dtype=bool)
    if cross_matrix_np.shape != (2, 2):
        raise ValueError(f"Joint Q/U synthesis requires a 2x2 cross matrix, got {cross_matrix_np.shape}")
    compute_cross_matrix = torch.tensor(cross_matrix_np, dtype=torch.bool, device=device)
    syntheses: list[np.ndarray] = []
    synth_seed = seed

    while len(syntheses) < total_syntheses:
        remaining = total_syntheses - len(syntheses)
        current_batch_size = min(synthesis_batch_size, remaining)
        generated = optimize_from_maps(
            target=ref_dc,
            st_op_target=st_op_target,
            st_op_running=st_op_running,
            nbatch=current_batch_size,
            pbc_running=synthesis_pbc,
            running_shape=running_shape,
            has_fewer_convolutions=bool(config.get("st_has_fewer_convolutions", False)),
            compute_cross_matrix=compute_cross_matrix,
            compute_PS=compute_ps,
            mean_field=True,
            max_iter=max_iter,
            lr=lr,
            history_size=history_size,
            print_iter=max(1, print_iter),
            verbose=verbose,
            seed=synth_seed,
        )
        generated_np = generated.detach().cpu().numpy() if isinstance(generated, torch.Tensor) else np.asarray(generated)
        if generated_np.ndim == 3:
            generated_np = generated_np[None, ...]
        if generated_np.ndim != 4 or generated_np.shape[1] != 2:
            raise RuntimeError(f"Expected generated joint syntheses shape (N,2,H,W), got {generated_np.shape}")
        syntheses.extend(list(generated_np))
        print(
            f"joint synthesis seed={synth_seed} batch={generated_np.shape[0]} total={len(syntheses)}/{total_syntheses}",
            flush=True,
        )
        synth_seed += 1

    return np.stack(syntheses[:total_syntheses], axis=0)


def maybe_center_crop_2d(image: np.ndarray, crop_size: int | None) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")
    if crop_size is None:
        return image
    h, w = image.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"Cannot crop {h}x{w} image to {crop_size}")
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return image[y0 : y0 + crop_size, x0 : x0 + crop_size]


def center_crop_to_hw(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")
    out_h, out_w = out_hw
    h, w = image.shape
    if out_h > h or out_w > w:
        raise ValueError(f"Cannot crop {h}x{w} image to {out_h}x{out_w}")
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return image[y0 : y0 + out_h, x0 : x0 + out_w]


def downsample_by_four(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")
    h, w = image.shape
    if h % 2 or w % 2:
        raise ValueError(f"Image dimensions must be even, got {h}x{w}")
    return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def augment_image(img: np.ndarray, n_augmentations: int, shift_step: int) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []
    img_tensor = torch.from_numpy(np.asarray(img)).unsqueeze(0).unsqueeze(0).float()

    for k in range(4):
        rotated = torch.rot90(img_tensor, k, dims=[2, 3])
        for flip in [False, True]:
            flipped = torch.flip(rotated, dims=[3]) if flip else rotated
            for shift_h in range(2):
                for shift_w in range(4):
                    shifted = torch.roll(
                        flipped,
                        shifts=(shift_h * shift_step, shift_w * shift_step),
                        dims=(2, 3),
                    )
                    augmented.append(shifted.squeeze(0).squeeze(0).cpu().numpy())
                    if len(augmented) >= n_augmentations:
                        return augmented

    return augmented


def augment_pair(q_img: np.ndarray, u_img: np.ndarray, n_augmentations: int, shift_step: int) -> list[tuple[np.ndarray, np.ndarray]]:
    augmented: list[tuple[np.ndarray, np.ndarray]] = []
    pair = np.stack([q_img, u_img], axis=0)
    pair_tensor = torch.from_numpy(np.asarray(pair)).unsqueeze(0).float()

    for k in range(4):
        rotated = torch.rot90(pair_tensor, k, dims=[2, 3])
        for flip in [False, True]:
            flipped = torch.flip(rotated, dims=[3]) if flip else rotated
            for shift_h in range(2):
                for shift_w in range(4):
                    shifted = torch.roll(
                        flipped,
                        shifts=(shift_h * shift_step, shift_w * shift_step),
                        dims=(2, 3),
                    )
                    arr = shifted.squeeze(0).cpu().numpy()
                    augmented.append((arr[0], arr[1]))
                    if len(augmented) >= n_augmentations:
                        return augmented

    return augmented


def nuisance_version_suffix(version: str) -> str | None:
    return None if version == "all" else f"_{version}.npy"


def filter_nuisance_version(paths: list[Path], version: str) -> list[Path]:
    suffix = nuisance_version_suffix(version)
    if suffix is None:
        return paths
    return [path for path in paths if path.name.endswith(suffix)]


def list_nuisance_paths(nuisance_dir: Path, patch: str, stokes: str, freq: int, version: str) -> list[Path]:
    patterns = [
        f"patch_{patch}_noise_{stokes}{freq}_*.npy",
        f"*patch_{patch}*noise*{stokes}{freq}*.npy",
    ]
    for pattern in patterns:
        matches = filter_nuisance_version(sorted(nuisance_dir.glob(pattern)), version)
        if matches:
            return matches
    raise FileNotFoundError(
        f"No nuisance maps found for patch={patch}, stokes={stokes}, freq={freq} in {nuisance_dir}"
    )


def load_nuisance_bank(paths: list[Path], crop_size: int | None) -> np.ndarray:
    maps = []
    for path in paths:
        arr = downsample_by_four(np.load(path).astype(np.float64))
        maps.append(maybe_center_crop_2d(arr, crop_size).astype(np.float32))
    return np.stack(maps, axis=0)


def build_pairs(
    clean_maps: np.ndarray,
    nuisance_bank: np.ndarray,
    *,
    n_augmentations: int,
    shift_step: int,
    final_hw: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    clean_samples: list[np.ndarray] = []
    contaminated_samples: list[np.ndarray] = []

    for clean_map in clean_maps:
        for augmented in augment_image(clean_map, n_augmentations=n_augmentations, shift_step=shift_step):
            augmented = center_crop_to_hw(augmented, final_hw)
            noise_idx = int(rng.integers(0, len(nuisance_bank)))
            clean_samples.append(augmented.astype(np.float32))
            contaminated_samples.append((augmented + nuisance_bank[noise_idx]).astype(np.float32))

    return np.stack(clean_samples, axis=0), np.stack(contaminated_samples, axis=0)


def build_joint_pairs(
    q_clean_maps: np.ndarray,
    u_clean_maps: np.ndarray,
    q_nuisance_bank: np.ndarray,
    u_nuisance_bank: np.ndarray,
    *,
    n_augmentations: int,
    shift_step: int,
    final_hw: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(q_clean_maps) != len(u_clean_maps):
        raise ValueError(f"Q/U clean synthesis count mismatch: {len(q_clean_maps)} vs {len(u_clean_maps)}")
    if len(q_nuisance_bank) != len(u_nuisance_bank):
        raise ValueError(f"Q/U nuisance count mismatch: {len(q_nuisance_bank)} vs {len(u_nuisance_bank)}")

    q_clean_samples: list[np.ndarray] = []
    q_contaminated_samples: list[np.ndarray] = []
    u_clean_samples: list[np.ndarray] = []
    u_contaminated_samples: list[np.ndarray] = []

    for q_map, u_map in zip(q_clean_maps, u_clean_maps):
        for q_aug, u_aug in augment_pair(
            q_map,
            u_map,
            n_augmentations=n_augmentations,
            shift_step=shift_step,
        ):
            q_aug = center_crop_to_hw(q_aug, final_hw)
            u_aug = center_crop_to_hw(u_aug, final_hw)
            noise_idx = int(rng.integers(0, len(q_nuisance_bank)))
            q_clean_samples.append(q_aug.astype(np.float32))
            u_clean_samples.append(u_aug.astype(np.float32))
            q_contaminated_samples.append((q_aug + q_nuisance_bank[noise_idx]).astype(np.float32))
            u_contaminated_samples.append((u_aug + u_nuisance_bank[noise_idx]).astype(np.float32))

    return (
        np.stack(q_clean_samples, axis=0),
        np.stack(q_contaminated_samples, axis=0),
        np.stack(u_clean_samples, axis=0),
        np.stack(u_contaminated_samples, axis=0),
    )


def load_config(json_path: Path) -> dict[str, object]:
    metadata = json.loads(json_path.read_text())
    config = metadata.get("config", metadata)
    if not isinstance(config, dict):
        raise ValueError(f"Could not read config dictionary from {json_path}")
    return config


def process_record(record: RunRecord, args: argparse.Namespace, device: torch.device) -> Path:
    print(f"Processing patch {record.patch}: {record.npy_path.name}", flush=True)
    config = load_config(record.json_path)
    recovered_qu = np.load(record.npy_path).astype(np.float64)
    if recovered_qu.shape[0] != 2 or recovered_qu.ndim != 3:
        raise RuntimeError(f"Expected recovered Q/U shape (2,H,W), got {recovered_qu.shape}")
    final_hw = (
        (args.crop_size, args.crop_size)
        if args.crop_size is not None
        else tuple(int(v) for v in recovered_qu.shape[-2:])
    )

    patch_out_dir = args.output_dir
    patch_out_dir.mkdir(parents=True, exist_ok=True)
    crop_tag = "full" if args.crop_size is None else f"crop{args.crop_size}"
    out_path = patch_out_dir / f"{record.stem}_moment_dataset_{crop_tag}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"Skipping patch {record.patch}: output exists at {out_path}", flush=True)
        return out_path

    rng = np.random.default_rng(args.seed + int(record.patch))
    q_noise_paths = list_nuisance_paths(args.nuisance_dir, record.patch, "Q", args.freq, args.nuisance_version)
    u_noise_paths = list_nuisance_paths(args.nuisance_dir, record.patch, "U", args.freq, args.nuisance_version)
    q_noise_bank = load_nuisance_bank(q_noise_paths, crop_size=args.crop_size)
    u_noise_bank = load_nuisance_bank(u_noise_paths, crop_size=args.crop_size)
    if tuple(q_noise_bank.shape[-2:]) != final_hw or tuple(u_noise_bank.shape[-2:]) != final_hw:
        raise RuntimeError(
            f"Nuisance shape mismatch: Q={q_noise_bank.shape[-2:]} U={u_noise_bank.shape[-2:]} expected={final_hw}"
        )

    if args.synthesis_joint_qu:
        qu_synth_full = synthesize_joint_qu(
            recovered_qu,
            config,
            device=device,
            seed=args.seed + int(record.patch) * 10,
            total_syntheses=args.syntheses_per_patch,
            synthesis_batch_size=args.synthesis_batch_size,
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
        q_synth_full = qu_synth_full[:, 0]
        u_synth_full = qu_synth_full[:, 1]
        q_synth = np.stack([center_crop_to_hw(arr, final_hw) for arr in q_synth_full], axis=0)
        u_synth = np.stack([center_crop_to_hw(arr, final_hw) for arr in u_synth_full], axis=0)
        q_clean, q_contaminated, u_clean, u_contaminated = build_joint_pairs(
            q_synth_full,
            u_synth_full,
            q_noise_bank,
            u_noise_bank,
            n_augmentations=args.n_augmentations,
            shift_step=args.shift_step,
            final_hw=final_hw,
            rng=rng,
        )
    else:
        q_synth_full = synthesize_single_channel(
            recovered_qu[0],
            config,
            device=device,
            seed=args.seed + int(record.patch) * 10 + 0,
            total_syntheses=args.syntheses_per_patch,
            synthesis_batch_size=args.synthesis_batch_size,
            synthesis_pbc=args.synthesis_pbc,
            running_shape=args.synthesis_running_shape,
            compute_ps=args.synthesis_compute_ps,
            max_iter=args.synthesis_max_iter,
            lr=args.synthesis_lr,
            history_size=args.synthesis_history_size,
            print_iter=args.print_every_synthesis,
            verbose=args.verbose_synthesis,
        )
        u_synth_full = synthesize_single_channel(
            recovered_qu[1],
            config,
            device=device,
            seed=args.seed + int(record.patch) * 10 + 1,
            total_syntheses=args.syntheses_per_patch,
            synthesis_batch_size=args.synthesis_batch_size,
            synthesis_pbc=args.synthesis_pbc,
            running_shape=args.synthesis_running_shape,
            compute_ps=args.synthesis_compute_ps,
            max_iter=args.synthesis_max_iter,
            lr=args.synthesis_lr,
            history_size=args.synthesis_history_size,
            print_iter=args.print_every_synthesis,
            verbose=args.verbose_synthesis,
        )
        q_synth = np.stack([center_crop_to_hw(arr, final_hw) for arr in q_synth_full], axis=0)
        u_synth = np.stack([center_crop_to_hw(arr, final_hw) for arr in u_synth_full], axis=0)
        q_clean, q_contaminated = build_pairs(
            q_synth_full,
            q_noise_bank,
            n_augmentations=args.n_augmentations,
            shift_step=args.shift_step,
            final_hw=final_hw,
            rng=rng,
        )
        u_clean, u_contaminated = build_pairs(
            u_synth_full,
            u_noise_bank,
            n_augmentations=args.n_augmentations,
            shift_step=args.shift_step,
            final_hw=final_hw,
            rng=rng,
        )

    np.savez_compressed(
        out_path,
        # Augmented training targets and inputs.
        xq=q_clean,
        yq=q_contaminated,
        xu=u_clean,
        yu=u_contaminated,
        # Raw periodic syntheses after optional center crop. These are before augmentation.
        sq=q_synth.astype(np.float32),
        su=u_synth.astype(np.float32),
        rq=q_synth_full.astype(np.float32),
        ru=u_synth_full.astype(np.float32),
        p=np.asarray(record.patch),
        stem=np.asarray(record.stem),
        naug=np.asarray(args.n_augmentations),
        shift=np.asarray(args.shift_step),
        aug=np.asarray("rot90_flip_periodic_roll"),
    )

    summary = {
        "patch": record.patch,
        "stem": record.stem,
        "source_npy": str(record.npy_path),
        "source_json": str(record.json_path),
        "output_npz": str(out_path),
        "syntheses_per_patch": args.syntheses_per_patch,
        "n_augmentations": args.n_augmentations,
        "crop_size": args.crop_size,
        "output_shape": list(q_clean.shape[-2:]),
        "synthesis_pbc": args.synthesis_pbc,
        "synthesis_joint_qu": args.synthesis_joint_qu,
        "cross_matrix": args.cross_matrix,
        "synthesis_compute_ps": args.synthesis_compute_ps,
        "synthesis_running_shape": list(args.synthesis_running_shape)
        if args.synthesis_running_shape is not None
        else None,
        "target_pbc_from_metadata": bool(config.get("pbc", False)),
        "q_pairs": int(len(q_clean)),
        "u_pairs": int(len(u_clean)),
        "q_noise_count": int(len(q_noise_bank)),
        "u_noise_count": int(len(u_noise_bank)),
    }
    (patch_out_dir / f"{record.stem}_moment_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(f"Saved {out_path}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-patch moment-network dataset shards from Planck compsep outputs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path(env_default("RESULTS_DIR", str(DEFAULT_RESULTS_DIR))))
    parser.add_argument("--nuisance-dir", type=Path, default=Path(env_default("PLANCK_NUISANCE_DIR", str(DEFAULT_NUISANCE_DIR))))
    parser.add_argument("--output-dir", type=Path, default=Path(env_default("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--patches", default=os.environ.get("PATCH_LIST", ""))
    parser.add_argument("--patch-limit", type=int, default=int(os.environ.get("PATCH_LIMIT", "0")))
    parser.add_argument("--shard-by-slurm-array", type=str_to_bool, default=str_to_bool(os.environ.get("SHARD_BY_SLURM_ARRAY", "1")))
    parser.add_argument("--shard-by-slurm-tasks", type=str_to_bool, default=str_to_bool(os.environ.get("SHARD_BY_SLURM_TASKS", "0")))
    parser.add_argument("--overwrite", action="store_true", default=str_to_bool(os.environ.get("OVERWRITE", "0")))

    parser.add_argument("--freq", type=int, default=int(os.environ.get("FREQ", "353")))
    parser.add_argument("--nuisance-version", default=os.environ.get("PLANCK_NUISANCE_VERSION", "v4_10_arcmin"))
    crop_size_env = os.environ.get("CROP_SIZE", "").strip()
    parser.add_argument(
        "--crop-size",
        type=int,
        default=int(crop_size_env) if crop_size_env else None,
        help="Optional center crop size. Default keeps full map dimensions.",
    )
    parser.add_argument("--syntheses-per-patch", type=int, default=int(os.environ.get("SYNTHESES_PER_PATCH", "10")))
    parser.add_argument("--synthesis-batch-size", type=int, default=int(os.environ.get("SYNTHESIS_BATCH_SIZE", "5")))
    parser.add_argument("--synthesis-pbc", type=str_to_bool, default=str_to_bool(os.environ.get("SYNTHESIS_PBC", "1")))
    parser.add_argument("--synthesis-joint-qu", type=str_to_bool, default=str_to_bool(os.environ.get("SYNTHESIS_JOINT_QU", "1")))
    parser.add_argument("--synthesis-compute-ps", type=str_to_bool, default=str_to_bool(os.environ.get("SYNTHESIS_COMPUTE_PS", "0")))
    parser.add_argument("--cross-matrix", type=parse_cross_matrix, default=parse_cross_matrix(os.environ.get("CROSS_MATRIX", "1,1;0,1")))
    parser.add_argument(
        "--synthesis-running-shape",
        type=parse_hw,
        default=parse_hw(os.environ.get("SYNTHESIS_RUNNING_SHAPE", "400")),
        help="Running synthesis shape, e.g. 400 or 400x400. Empty uses target shape.",
    )
    parser.add_argument("--synthesis-max-iter", type=int, default=int(os.environ.get("SYNTHESIS_MAX_ITER", "150")))
    parser.add_argument("--synthesis-lr", type=float, default=float(os.environ.get("SYNTHESIS_LR", "1.0")))
    parser.add_argument("--synthesis-history-size", type=int, default=int(os.environ.get("SYNTHESIS_HISTORY_SIZE", "40")))
    parser.add_argument("--print-every-synthesis", type=int, default=int(os.environ.get("PRINT_EVERY_SYNTHESIS", "10")))
    parser.add_argument("--verbose-synthesis", type=str_to_bool, default=str_to_bool(os.environ.get("VERBOSE_SYNTHESIS", "1")))

    parser.add_argument("--n-augmentations", type=int, default=int(os.environ.get("N_AUGMENTATIONS", "64")))
    parser.add_argument("--shift-step", type=int, default=int(os.environ.get("SHIFT_STEP", "16")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "7")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser()
    args.nuisance_dir = args.nuisance_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")

    patch_selector = parse_patch_selector(args.patches)
    records = discover_runs(args.results_dir, patch_selector)
    if args.patch_limit > 0:
        records = records[: args.patch_limit]
    records = shard_records(records, args.shard_by_slurm_array)
    records = shard_records_by_slurm_tasks(records, args.shard_by_slurm_tasks)

    print(f"REPO_ROOT={REPO_ROOT}", flush=True)
    print(f"results_dir={args.results_dir}", flush=True)
    print(f"nuisance_dir={args.nuisance_dir}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"synthesis_running_shape={args.synthesis_running_shape}", flush=True)
    print(
        f"slurm_procid={os.environ.get('SLURM_PROCID', '0')} "
        f"slurm_ntasks={os.environ.get('SLURM_NTASKS', '1')}",
        flush=True,
    )
    print(f"records_this_task={len(records)}", flush=True)
    print(f"patches_this_task={[record.patch for record in records]}", flush=True)

    written = []
    for record in records:
        written.append(str(process_record(record, args, device)))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest_path = args.output_dir / f"manifest_task_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "nuisance_dir": str(args.nuisance_dir),
                "output_dir": str(args.output_dir),
                "written": written,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(f"Wrote manifest {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
