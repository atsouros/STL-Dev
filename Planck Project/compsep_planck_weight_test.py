#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import inspect
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_VERSION = "2026-10-01_stable_relative_convergence_cosecant_mask_optional"
PATCH_MASK_PATH = Path(
    "/pscratch/sd/s/shamikg/polarized_dust_STGNILC/resources/"
    "cosecant_mask_eps0.05.npy"
)


# Repo root (for imports)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as FFT_DataClass
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch as Kernel_DataClass
import STL_main.ST_Operator as st_operator_module
import STL_main.ST_Statistics as st_statistics_module
from utils import (
    NUISANCE_DIR,
    SIGNAL_DIR,
    _build_identity_config,
    _build_run_config,
    _build_run_stem,
    _center_crop,
    _configure_backend_defaults,
    _downsample_by_four,
    _filter_nuisance_version,
    _load_noise_batch,
    _maybe_set_wtype,
    _nuisance_version_suffix,
    _parse_patch_list,
    _select_no_bonus_signal_path,
)


def _load_patch_mask(patch: str, out_hw: tuple[int, int]) -> np.ndarray:
    """Read one already-downsampled mask and align it with the input crop."""
    masks = np.load(PATCH_MASK_PATH, mmap_mode="r", allow_pickle=False)
    if masks.shape != (192, 384, 384):
        raise ValueError(f"Expected mask shape (192, 384, 384), got {masks.shape}")
    patch_index = int(patch)
    if not 0 <= patch_index < masks.shape[0]:
        raise ValueError(f"Mask patch index must be in [0, 191], got {patch!r}")
    patch_mask = np.array(
        _center_crop(masks[patch_index], out_hw=out_hw), dtype=np.float64, copy=True
    )
    if not np.isfinite(patch_mask).all() or np.any(patch_mask <= 0):
        raise ValueError(
            f"Patch {patch}: mask values must be finite and strictly positive "
            "to divide recovered maps by the mask when saving."
        )
    return patch_mask


def _load_masked_noise_batch(
    paths: list[Path], *, patch_mask: np.ndarray | None,
    expected_hw: tuple[int, int], crop_hw: tuple[int, int] | None
) -> np.ndarray:
    """Optionally mask nuisance realizations after the usual preprocessing."""
    batch = _load_noise_batch(paths, expected_hw=expected_hw, crop_hw=crop_hw)
    return batch if patch_mask is None else batch * patch_mask[None, :, :]


def _unmask_recovered(signal_qu: torch.Tensor, patch_mask: np.ndarray | None) -> np.ndarray:
    """Save maps in original units, including intermediate checkpoints."""
    recovered = signal_qu.detach().cpu().numpy()
    return recovered if patch_mask is None else recovered / patch_mask[None, :, :]


def _build_default_config() -> dict[str, object]:
    return {
        "patch": "3",
        "map_size": None,
        "backend": "fft",
        "dtype": "float64",
        "wtype": "Bump-Steerable",
        "seed": 0,
        "batch": 15,
        "epochs": 10,
        "epoch_steps": 1,
        "lbfgs_max_iter": 100,
        "pbc": False,
        "white_noise_initial": False,
        "start_without_noise_channels": False,
        "nuisance_version": "v4_10_arcmin",
        "st_reduced": True,
        "st_j": 7,
        "st_l": 4,
        "st_iso": False,
        "st_angular_ft": True,
        "st_scale_ft": True,
        "st_harmonics_angle": 2,
        "st_harmonics_scale": 3,
        "st_dj": 3,
        "st_compute_ps": True,
        "st_ps_method": "legacy",
        "st_ps_n_bins": None,
        "ps_loss_weight": 1.0,
        "loss_normalization": "mean_per_coefficient",
        "convergence_threshold": 1e-3,
        "convergence_relative_tolerance": 1e-2,
        "convergence_check_every": 5,
        "convergence_patience": 3,
        "st_has_fewer_convolutions": True,
        "patch_mask_path": str(PATCH_MASK_PATH),
        "use_patch_mask": True,
    }


def _run_patch(patch: str, *, rank: int = 0) -> None:
    patch_start_time = time.perf_counter()
    signal_dir = Path(os.environ.get("PLANCK_SIGNAL_DIR", str(SIGNAL_DIR))).expanduser()
    nuisance_dir = Path(os.environ.get("PLANCK_NUISANCE_DIR", str(NUISANCE_DIR))).expanduser()
    root = signal_dir.parent

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    seed = int(os.environ.get("SEED", "0"))
    mask_flag = os.environ.get("USE_PATCH_MASK", "True").strip().lower()
    if mask_flag not in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
        raise ValueError("USE_PATCH_MASK must be True or False")
    use_patch_mask = mask_flag in {"true", "1", "yes", "y"}
    n_batch = int(os.environ.get("BATCH", "15"))
    epochs = int((os.environ.get("EPOCHS") or os.environ.get("OUTER_ITERS") or "10").strip())
    epoch_steps = int((os.environ.get("EPOCH_STEPS") or "1").strip())
    lbfgs_max_iter = int(os.environ.get("LBFGS_MAX_ITER", "100"))
    convergence_threshold = float(
        (os.environ.get("CONVERGENCE_THRESHOLD") or "1e-3").strip()
    )
    convergence_relative_tolerance = float(
        (os.environ.get("CONVERGENCE_RELATIVE_TOLERANCE") or "1e-2").strip()
    )
    convergence_check_every = int(
        (os.environ.get("CONVERGENCE_CHECK_EVERY") or "5").strip()
    )
    convergence_patience = int(
        (os.environ.get("CONVERGENCE_PATIENCE") or "3").strip()
    )
    if not np.isfinite(convergence_threshold) or convergence_threshold <= 0:
        raise ValueError("CONVERGENCE_THRESHOLD must be a finite positive number")
    if (
        not np.isfinite(convergence_relative_tolerance)
        or convergence_relative_tolerance <= 0
    ):
        raise ValueError(
            "CONVERGENCE_RELATIVE_TOLERANCE must be a finite positive number"
        )
    if convergence_check_every <= 0:
        raise ValueError("CONVERGENCE_CHECK_EVERY must be a positive integer")
    if convergence_patience <= 0:
        raise ValueError("CONVERGENCE_PATIENCE must be a positive integer")
    start_without_noise_channels = (
        os.environ.get("START_WITHOUT_NOISE_CHANNELS") or "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    initial_results_dir_env = (os.environ.get("INITIAL_RESULTS_DIR") or "").strip()
    pbc = (os.environ.get("PBC") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    white_noise_initial = (os.environ.get("WHITE_NOISE_INITIAL") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_reduced = (os.environ.get("ST_REDUCED") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_j = int((os.environ.get("ST_J") or "7").strip())
    st_l = int((os.environ.get("ST_L") or "4").strip())
    st_iso = (os.environ.get("ST_ISO") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_angular_ft = (os.environ.get("ST_ANGULAR_FT") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_scale_ft = (os.environ.get("ST_SCALE_FT") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_harmonics_angle = int((os.environ.get("ST_HARMONICS_ANGLE") or "2").strip())
    st_harmonics_scale = int((os.environ.get("ST_HARMONICS_SCALE") or "3").strip())
    st_dj = int((os.environ.get("ST_DJ") or "3").strip())
    st_compute_ps = (os.environ.get("ST_COMPUTE_PS") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_ps_method = (os.environ.get("ST_PS_METHOD") or "legacy").strip().lower()
    if st_ps_method not in {"legacy", "gaussian_rings"}:
        raise ValueError("ST_PS_METHOD must be 'legacy' or 'gaussian_rings'")
    st_ps_n_bins_env = (os.environ.get("ST_PS_N_BINS") or "").strip()
    st_ps_n_bins = int(st_ps_n_bins_env) if st_ps_n_bins_env else None
    if st_ps_n_bins is not None and st_ps_n_bins <= 0:
        raise ValueError("ST_PS_N_BINS must be a positive integer when set")
    ps_loss_weight = float((os.environ.get("PS_LOSS_WEIGHT") or "1").strip())
    if not np.isfinite(ps_loss_weight) or ps_loss_weight <= 0:
        raise ValueError("PS_LOSS_WEIGHT must be a finite positive number")
    if not st_compute_ps and ps_loss_weight != 1.0:
        raise ValueError("PS_LOSS_WEIGHT requires ST_COMPUTE_PS=True")
    ps_residual_scale = ps_loss_weight**0.5
    st_has_fewer_convolutions = (
        os.environ.get("ST_FEWER_CONVOLUTIONS") or "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    wtype = os.environ.get("WTYPE", "Bump-Steerable")
    map_size_env = (os.environ.get("MAP_SIZE") or "").strip()
    map_size = int(map_size_env) if map_size_env else None
    nuisance_version = (os.environ.get("PLANCK_NUISANCE_VERSION") or "v4_10_arcmin").strip()
    if nuisance_version not in {"v4_10_arcmin", "v2", "all"}:
        raise ValueError("PLANCK_NUISANCE_VERSION must be 'v4_10_arcmin', 'v2', or 'all'")

    backend = (os.environ.get("BACKEND") or "fft").strip().lower()
    if backend not in {"fft", "kernel"}:
        raise ValueError("BACKEND must be 'fft' or 'kernel'")
    if backend != "fft" and st_ps_method != "legacy":
        raise ValueError(
            "ST_PS_METHOD='gaussian_rings' is currently implemented only for BACKEND='fft'"
        )
    DataClass = FFT_DataClass if backend == "fft" else Kernel_DataClass

    # -------------------------------------------------------------------------
    # Device / dtype
    # -------------------------------------------------------------------------
    device_override = (os.environ.get("DEVICE") or "").strip()
    if device_override:
        device = torch.device(device_override)
    else:
        local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
    print(f"Rank {rank} | patch {patch} | using device: {device} (backend={backend})")

    dtype_str = (os.environ.get("DTYPE") or "float64").strip().lower()
    if dtype_str in {"float64", "fp64", "double"}:
        dtype = torch.float64
    elif dtype_str in {"float32", "fp32", "single"}:
        dtype = torch.float32
    else:
        raise ValueError("DTYPE must be float64 or float32")

    _configure_backend_defaults(device=device, dtype=dtype)
    torch.manual_seed(seed)
    print(
        "code_provenance: "
        f"version={CODE_VERSION} | "
        f"compsep={Path(__file__).resolve()} | "
        f"ST_Operator={inspect.getsourcefile(st_operator_module)} | "
        f"ST_Statistics={inspect.getsourcefile(st_statistics_module)}"
    )
    print(f"PBC: {pbc}")
    print(f"Epochs: {epochs} | steps per epoch: {epoch_steps}")
    print("Optimizer: lbfgs | one optimizer iteration per logged step")
    print(
        "Convergence: "
        f"normalized validation loss <= {convergence_threshold:g}, then "
        f"{convergence_patience} consecutive relative changes <= "
        f"{convergence_relative_tolerance:g} | "
        f"check every {convergence_check_every} steps"
    )
    print(
        "Loss weighting: "
        f"L = L_other + {ps_loss_weight:g} * L_PS "
        f"(PS residual scale={ps_residual_scale:g})"
    )
    if st_reduced:
        print(
            "Reduced ST config: "
            f"J={st_j} | L={st_l} | iso={st_iso} | angular_ft={st_angular_ft} | "
            f"scale_ft={st_scale_ft} | harmonics_angle={st_harmonics_angle} | "
            f"harmonics_scale={st_harmonics_scale} | dj={st_dj} | "
            f"compute_PS={st_compute_ps} | ps_method={st_ps_method} | "
            f"ps_n_bins={st_ps_n_bins} | "
            f"fewer_convolutions={st_has_fewer_convolutions}"
        )
    else:
        print(
            "Standard ST config: "
            f"compute_PS={st_compute_ps} | ps_method={st_ps_method} | "
            f"ps_n_bins={st_ps_n_bins} | "
            f"fewer_convolutions={st_has_fewer_convolutions}"
        )

    # -------------------------------------------------------------------------
    # Load NERSC Planck patch inputs
    # -------------------------------------------------------------------------
    freq = int(os.environ.get("FREQ", "353"))
    q353_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_Q{freq}_*.npy", f"Q{freq}")
    u353_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_U{freq}_*.npy", f"U{freq}")
    i857_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_I857_*.npy", "I857")
    print(f"Q{freq}:", q353_path.name)
    print(f"U{freq}:", u353_path.name)
    print("I857:", i857_path.name)

    d_q = _downsample_by_four(np.load(q353_path).astype(np.float64))
    d_u = _downsample_by_four(np.load(u353_path).astype(np.float64))
    aux = _downsample_by_four(np.load(i857_path).astype(np.float64))

    if map_size is not None:
        d_q = _center_crop(d_q, out_hw=(map_size, map_size))
        d_u = _center_crop(d_u, out_hw=(map_size, map_size))
        aux = _center_crop(aux, out_hw=(map_size, map_size))

    aux = aux - float(np.mean(aux))  # fixed auxiliary map, mean-subtracted

    H, W = d_q.shape
    if d_u.shape != (H, W) or aux.shape != (H, W):
        raise RuntimeError(
            f"Shape mismatch: d_q={d_q.shape}, d_u={d_u.shape}, aux={aux.shape}"
        )
    expected_map_size = int(os.environ.get("EXPECTED_MAP_SIZE", "384"))
    if map_size is None and (H, W) != (expected_map_size, expected_map_size):
        raise RuntimeError(
            f"Expected downsampled maps to be {expected_map_size}x{expected_map_size}, got {H}x{W}."
        )

    patch_mask = None
    patch_mask_t = None
    if use_patch_mask:
        patch_mask = _load_patch_mask(patch, out_hw=(H, W))
        print(
            f"Patch mask: {PATCH_MASK_PATH} | patch={patch} | "
            f"shape={patch_mask.shape} | range=[{patch_mask.min():.6g}, {patch_mask.max():.6g}]"
        )
        # Center I857 before masking so subtracting a constant does not undo its taper.
        d_q *= patch_mask
        d_u *= patch_mask
        aux *= patch_mask
        patch_mask_t = torch.from_numpy(patch_mask).to(device=device, dtype=dtype)
    else:
        print("Patch mask: disabled (no mask loading, multiplication, or division)")

    initial_signal_qu = None
    if initial_results_dir_env:
        initial_results_dir = Path(initial_results_dir_env).expanduser()
        initial_candidates = sorted(
            path
            for path in initial_results_dir.glob(f"p{patch}_*.npy")
            if "_checkpoint" not in path.stem
        )
        if len(initial_candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one previous result for patch {patch} in "
                f"{initial_results_dir}, found {len(initial_candidates)}: "
                f"{[path.name for path in initial_candidates]}"
            )
        initial_signal_qu = np.load(initial_candidates[0], allow_pickle=False).astype(np.float64)
        if initial_signal_qu.shape != (2, H, W):
            raise RuntimeError(
                f"Previous result {initial_candidates[0]} has shape {initial_signal_qu.shape}; "
                f"expected (2, {H}, {W})."
            )
        if patch_mask is not None:
            initial_signal_qu *= patch_mask[None, :, :]
        print(f"Continuing from previous result: {initial_candidates[0]}")

    # -------------------------------------------------------------------------
    # Pair nuisance samples by (noise_seed, CMB_res_seed)
    # -------------------------------------------------------------------------
    pat = re.compile(r"noise_seed_(\d+)_CMB_res_seed_(\d+)")

    def parse_key(path: Path) -> tuple[int, int]:
        m = pat.search(path.name)
        if not m:
            raise ValueError(f"Could not parse seeds from: {path.name}")
        return (int(m.group(1)), int(m.group(2)))

    q_pattern = f"patch_{patch}_noise_Q{freq}_*.npy"
    u_pattern = f"patch_{patch}_noise_U{freq}_*.npy"
    q_paths = _filter_nuisance_version(sorted(nuisance_dir.glob(q_pattern)), nuisance_version)
    u_paths = _filter_nuisance_version(sorted(nuisance_dir.glob(u_pattern)), nuisance_version)
    if not q_paths or not u_paths:
        suffix = _nuisance_version_suffix(nuisance_version)
        version_msg = "" if suffix is None else f" with suffix *{suffix}"
        raise FileNotFoundError(
            f"Could not find nuisance samples{version_msg}. Looked for "
            f"{nuisance_dir / q_pattern} and {nuisance_dir / u_pattern}."
        )
    print(f"Nuisance version: {nuisance_version}")
    try:
        q_by_key = {parse_key(p): p for p in q_paths}
        u_by_key = {parse_key(p): p for p in u_paths}
        if len(q_by_key) != len(q_paths) or len(u_by_key) != len(u_paths):
            raise RuntimeError(
                "Nuisance seed keys are not unique. Set PLANCK_NUISANCE_VERSION to "
                "'v4_10_arcmin' or 'v2' instead of 'all'."
            )
        paired_keys = sorted(set(q_by_key).intersection(u_by_key))
        if not paired_keys:
            raise RuntimeError("No paired nuisance samples found for Q/U (seed keys empty).")
        noise_q_paths = [q_by_key[k] for k in paired_keys]
        noise_u_paths = [u_by_key[k] for k in paired_keys]
        print("Paired nuisance samples:", len(paired_keys))
        print("First key:", paired_keys[0], "->", noise_q_paths[0].name, "|", noise_u_paths[0].name)
    except ValueError:
        if len(q_paths) != len(u_paths):
            raise RuntimeError(
                f"Cannot pair nuisance Q/U samples: Q has {len(q_paths)} files, U has {len(u_paths)} files, "
                "and filenames do not match the expected seed pattern."
            )
        noise_q_paths = q_paths
        noise_u_paths = u_paths
        print("Paired nuisance samples by index:", len(q_paths))

    n_noise = len(noise_q_paths)

    # Use a fixed validation subset for convergence checks. Keep it out of the
    # training pool whenever there are enough nuisance realizations to do so.
    validation_rng = np.random.default_rng(seed + 10_000_019)
    if n_noise > n_batch:
        validation_count = min(n_batch, n_noise - n_batch)
        validation_indices = np.sort(
            validation_rng.choice(n_noise, size=validation_count, replace=False)
        )
        training_indices = np.setdiff1d(
            np.arange(n_noise), validation_indices, assume_unique=True
        )
        validation_is_reserved = True
    else:
        validation_count = min(n_batch, n_noise)
        validation_indices = np.arange(validation_count)
        training_indices = np.arange(n_noise)
        validation_is_reserved = False
    print(
        f"Validation nuisance samples: {validation_count} fixed | "
        f"reserved from training: {validation_is_reserved} | "
        f"training samples: {training_indices.size}"
    )

    # -------------------------------------------------------------------------
    # Phase 1 (optional): 3 channels [dQ, dU, aux], no explicit noise channels
    # Phase 2: 5 channels [dQ, nQ, dU, nU, aux]
    # -------------------------------------------------------------------------
    CROSS_MATRIX_NO_NOISE_CHANNELS = torch.tensor(
        [
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    CROSS_MATRIX_WITH_NOISE_CHANNELS = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    CROSS_MATRIX_REF_3 = torch.eye(3, dtype=torch.bool, device=device)
    CROSS_MATRIX_REF_5 = torch.eye(5, dtype=torch.bool, device=device)

    # -------------------------------------------------------------------------
    # Operator + normalization references
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    ref_i = int(rng.choice(training_indices))
    n_q_ref, n_u_ref = _load_masked_noise_batch(
        [noise_q_paths[ref_i], noise_u_paths[ref_i]],
        patch_mask=patch_mask,
        expected_hw=(H, W),
        crop_hw=(H, W) if map_size is not None else None,
    )

    ref_tensor_3 = torch.from_numpy(np.stack([d_q, d_u, aux], axis=0)).to(device, dtype=dtype)
    ref_dc_3 = DataClass(ref_tensor_3[None, ...], pbc=pbc)  # (1, 3, H, W)

    ref_tensor_5 = torch.from_numpy(np.stack([d_q, n_q_ref, d_u, n_u_ref, aux], axis=0)).to(
        device, dtype=dtype
    )  # (5, H, W)
    ref_dc_5 = DataClass(ref_tensor_5[None, ...], pbc=pbc)  # (1, 5, H, W)

    def build_st_op(ref_dc):
        ps_kwargs = {"n_bins": st_ps_n_bins} if st_ps_n_bins is not None else {}
        if backend == "fft":
            ps_kwargs["power_spectrum_method"] = st_ps_method
        if st_reduced:
            st_op_local = ref_dc.get_ST_op(
                J=st_j,
                L=st_l,
                iso=st_iso,
                angular_ft=st_angular_ft,
                scale_ft=st_scale_ft,
                harmonics_angle=st_harmonics_angle,
                harmonics_scale=st_harmonics_scale,
                dj=st_dj,
                compute_PS=st_compute_ps,
                has_fewer_convolutions=st_has_fewer_convolutions,
                **ps_kwargs,
            )
        else:
            st_op_local = ref_dc.get_ST_op(
                compute_PS=st_compute_ps,
                has_fewer_convolutions=st_has_fewer_convolutions,
                **ps_kwargs,
            )
            # Legacy non-reduced Planck runs recorded st_angular_ft/st_scale_ft in
            # metadata but did not pass those options into the standard ST operator.
            # Keep that behavior explicit so this path cannot accidentally enter
            # STL_main.ST_Statistics.to_scale_ft(), whose non-isotropic branch is
            # not part of the validated Planck production configuration.
            st_op_local.angular_ft = False
            st_op_local.scale_ft = False
        _maybe_set_wtype(st_op=st_op_local, ref_dc=ref_dc, wtype=wtype)
        print(
            "actual_st_operator: "
            f"requested_reduced={st_reduced} | "
            f"metadata_angular_ft={st_angular_ft} | metadata_scale_ft={st_scale_ft} | "
            f"operator_angular_ft={getattr(st_op_local, 'angular_ft', None)} | "
            f"operator_scale_ft={getattr(st_op_local, 'scale_ft', None)}"
        )
        return st_op_local

    st_op_3 = build_st_op(ref_dc_3)
    st_op_5 = build_st_op(ref_dc_5)
    apply_transform_kwargs = {} if st_reduced else {"angular_ft": False, "scale_ft": False}

    with torch.no_grad():
        st_op_3.apply(
            ref_dc_3,
            norm="store_ref",
            compute_cross_matrix=CROSS_MATRIX_REF_3,
            **apply_transform_kwargs,
        )
        st_op_5.apply(
            ref_dc_5,
            norm="store_ref",
            compute_cross_matrix=CROSS_MATRIX_REF_5,
            **apply_transform_kwargs,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    d_q_t = torch.from_numpy(np.copy(d_q)).to(device, dtype=dtype)
    d_u_t = torch.from_numpy(np.copy(d_u)).to(device, dtype=dtype)
    aux_t = torch.from_numpy(np.copy(aux)).to(device, dtype=dtype)
    validation_q_np = _load_masked_noise_batch(
        [noise_q_paths[int(i)] for i in validation_indices],
        patch_mask=patch_mask,
        expected_hw=(H, W),
        crop_hw=(H, W) if map_size is not None else None,
    )
    validation_u_np = _load_masked_noise_batch(
        [noise_u_paths[int(i)] for i in validation_indices],
        patch_mask=patch_mask,
        expected_hw=(H, W),
        crop_hw=(H, W) if map_size is not None else None,
    )
    validation_nq = torch.from_numpy(validation_q_np).to(device, dtype=dtype)
    validation_nu = torch.from_numpy(validation_u_np).to(device, dtype=dtype)
    validation_nb = int(validation_indices.size)
    printed_ps_block_size_by_phase = {
        "without_noise_channels": False,
        "with_noise_channels": False,
    }

    def stats_flat(dc: DataClass, *, phase: str) -> torch.Tensor:
        if phase == "without_noise_channels":
            statistics = st_op_3.apply(
                dc,
                norm="load_ref",
                compute_cross_matrix=CROSS_MATRIX_NO_NOISE_CHANNELS,
                **apply_transform_kwargs,
            )
        else:
            statistics = st_op_5.apply(
                dc,
                norm="load_ref",
                compute_cross_matrix=CROSS_MATRIX_WITH_NOISE_CHANNELS,
                **apply_transform_kwargs,
            )

        flat = statistics.to_flatten(mean_along_batch=True, keepnans=False)
        if not st_compute_ps:
            return flat

        # ST_Statistics.to_flatten appends PS last. Reproduce its batch mean and
        # NaN removal to identify that exact suffix, then multiply its residual
        # by sqrt(lambda_PS). The squared-L2 objective is consequently
        # L_other + lambda_PS * L_PS.
        ps_flat = statistics.PS.mean(dim=0, keepdim=True).reshape(-1)
        ps_count = int((~torch.isnan(ps_flat)).sum().item())
        if ps_count <= 0 or ps_count > flat.numel():
            raise RuntimeError(
                f"Invalid power-spectrum block size: PS={ps_count}, total={flat.numel()}"
            )
        if not printed_ps_block_size_by_phase[phase]:
            print(
                f"Power-spectrum coefficient count for phase '{phase}': {ps_count} "
                f"of {flat.numel()}"
            )
            printed_ps_block_size_by_phase[phase] = True
        if ps_loss_weight == 1.0:
            return flat
        return torch.cat(
            [flat[:-ps_count], flat[-ps_count:] * ps_residual_scale]
        )

    def make_target_batch(
        nb: int,
        *,
        phase: str,
        batch_nq: torch.Tensor | None = None,
        batch_nu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if phase == "without_noise_channels":
            return torch.stack(
                [
                    d_q_t.expand(nb, -1, -1),
                    d_u_t.expand(nb, -1, -1),
                    aux_t.expand(nb, -1, -1),
                ],
                dim=1,
            )  # (Nb, 3, H, W)
        if batch_nq is None or batch_nu is None:
            raise RuntimeError("batch_nq and batch_nu must be provided in the noise-channel phase.")
        return torch.stack(
            [
                d_q_t.expand(nb, -1, -1),
                batch_nq,
                d_u_t.expand(nb, -1, -1),
                batch_nu,
                aux_t.expand(nb, -1, -1),
            ],
            dim=1,
        )  # (Nb, 5, H, W)

    def make_running_batch(
        signal_qu: torch.Tensor,
        *,
        phase: str,
        batch_nq: torch.Tensor,
        batch_nu: torch.Tensor,
    ) -> torch.Tensor:
        nb = int(batch_nq.shape[0])
        s_q = signal_qu[0]
        s_u = signal_qu[1]
        if phase == "without_noise_channels":
            return torch.stack(
                [
                    s_q[None, :, :] + batch_nq,
                    s_u[None, :, :] + batch_nu,
                    aux_t.expand(nb, -1, -1),
                ],
                dim=1,
            )
        return torch.stack(
            [
                s_q[None, :, :] + batch_nq,
                (d_q_t - s_q)[None, :, :].expand(nb, -1, -1),
                s_u[None, :, :] + batch_nu,
                (d_u_t - s_u)[None, :, :].expand(nb, -1, -1),
                aux_t.expand(nb, -1, -1),
            ],
            dim=1,
        )

    def squared_l2(diff: torch.Tensor) -> torch.Tensor:
        """Mean squared residual per retained statistic coefficient."""
        if diff.numel() == 0:
            raise RuntimeError("Cannot calculate loss from an empty coefficient vector")
        return diff.abs().square().sum() / diff.numel()

    # -------------------------------------------------------------------------
    # Optimization (jointly optimize Q and U)
    # -------------------------------------------------------------------------
    init_std_q = float(torch.std(d_q_t).detach().cpu())
    init_std_u = float(torch.std(d_u_t).detach().cpu())
    if white_noise_initial:
        print(
            f"Initialization per epoch: white noise | std(Q)={init_std_q:.6g} | std(U)={init_std_u:.6g}"
        )
    else:
        print("Initialization per epoch: data maps")

    loss_calls: list[float] = []
    validation_steps: list[int] = []
    validation_losses: list[float] = []
    validation_relative_changes: list[float | None] = []
    validation_target_by_phase: dict[str, torch.Tensor] = {}
    convergence_hits = 0
    previous_convergence_loss: float | None = None
    converged = False
    completed_steps = 0
    print(f"Start without explicit noise channels: {start_without_noise_channels}")
    printed_target_size_by_phase = {
        "without_noise_channels": False,
        "with_noise_channels": False,
    }

    def format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # -------------------------------------------------------------------------
    # Output setup
    # -------------------------------------------------------------------------
    out_dir_env = (os.environ.get("OUTDIR") or "").strip()
    results_dir = Path(out_dir_env).expanduser() if out_dir_env else REPO_ROOT / "planck_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_config = _build_run_config(
        root=root,
        patch=patch,
        map_size=map_size,
        device=device,
        backend=backend,
        dtype_str=dtype_str,
        wtype=wtype,
        seed=seed,
        n_batch=n_batch,
        epochs=epochs,
        epoch_steps=epoch_steps,
        lbfgs_max_iter=lbfgs_max_iter,
        pbc=pbc,
        white_noise_initial=white_noise_initial,
        start_without_noise_channels=start_without_noise_channels,
        nuisance_version=nuisance_version,
        st_reduced=st_reduced,
        st_j=st_j,
        st_l=st_l,
        st_iso=st_iso,
        st_angular_ft=st_angular_ft,
        st_scale_ft=st_scale_ft,
        st_harmonics_angle=st_harmonics_angle,
        st_harmonics_scale=st_harmonics_scale,
        st_dj=st_dj,
        st_compute_ps=st_compute_ps,
        st_ps_method=st_ps_method,
        st_ps_n_bins=st_ps_n_bins,
        st_has_fewer_convolutions=st_has_fewer_convolutions,
    )
    run_config["ps_loss_weight"] = ps_loss_weight
    run_config["loss_normalization"] = "mean_per_coefficient"
    run_config["convergence_threshold"] = convergence_threshold
    run_config["convergence_relative_tolerance"] = convergence_relative_tolerance
    run_config["convergence_check_every"] = convergence_check_every
    run_config["convergence_patience"] = convergence_patience
    run_config["validation_batch"] = validation_count
    run_config["validation_reserved_from_training"] = validation_is_reserved
    run_config["use_patch_mask"] = use_patch_mask
    run_config["patch_mask_path"] = str(PATCH_MASK_PATH) if use_patch_mask else None
    run_config["patch_mask_sha256"] = (
        hashlib.sha256(patch_mask.tobytes()).hexdigest() if use_patch_mask else None
    )
    run_config["patch_mask_min"] = float(patch_mask.min()) if use_patch_mask else None
    run_config["patch_mask_max"] = float(patch_mask.max()) if use_patch_mask else None
    run_config["aux_preprocessing"] = (
        "subtract_mean_then_multiply_patch_mask" if use_patch_mask else "subtract_mean"
    )
    run_config["saved_map_space"] = "unmasked"
    # Keep the existing output names so identical run settings replace results.
    # Mask provenance is recorded in both config and identity_config instead.
    weight_tag = f"{ps_loss_weight:.12g}".replace(".", "p").replace("-", "m")
    threshold_tag = f"{convergence_threshold:.12g}".replace(".", "p").replace("-", "m")
    relative_tolerance_tag = f"{convergence_relative_tolerance:.12g}".replace(".", "p").replace("-", "m")
    out_stem = (
        f"{_build_run_stem(run_config)}_psw{weight_tag}"
        f"_ct{threshold_tag}_crt{relative_tolerance_tag}"
        f"_ce{convergence_check_every}_cp{convergence_patience}"
    )
    checkpoint_every = 10

    def weight_test_identity_config() -> dict[str, object]:
        identity = _build_identity_config(run_config)
        identity["ps_loss_weight"] = ps_loss_weight
        identity["loss_normalization"] = "mean_per_coefficient"
        identity["convergence_threshold"] = convergence_threshold
        identity["convergence_relative_tolerance"] = convergence_relative_tolerance
        identity["convergence_check_every"] = convergence_check_every
        identity["convergence_patience"] = convergence_patience
        identity["patch_mask_sha256"] = run_config["patch_mask_sha256"]
        identity["use_patch_mask"] = use_patch_mask
        identity["aux_preprocessing"] = run_config["aux_preprocessing"]
        return identity

    def save_loss_curve(loss_values: list[float], loss_path: Path) -> None:
        finite_loss = np.asarray(loss_values, dtype=float)
        finite_mask = np.isfinite(finite_loss)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if np.any(finite_mask):
            training_steps = np.arange(1, finite_loss.size + 1)[finite_mask]
            ax.plot(
                training_steps,
                finite_loss[finite_mask],
                linewidth=1,
                label="Training",
            )
        else:
            ax.plot([1], [1e-1], linewidth=1, label="Training")
        if validation_losses:
            ax.plot(
                validation_steps,
                validation_losses,
                marker="o",
                linewidth=1.5,
                label="Fixed validation",
            )
        ax.axhline(
            convergence_threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Convergence threshold",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e5)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Mean squared residual per coefficient")
        ax.set_title("Normalized loss")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        try:
            fig.tight_layout()
            fig.savefig(loss_path, dpi=200)
        except Exception as exc:
            print(f"Warning: could not save loss curve for patch {patch}: {exc}")
        plt.close(fig)

    def save_checkpoint(global_step: int) -> None:
        recovered = _unmask_recovered(running_signal_qu, patch_mask)
        checkpoint_stem = f"{out_stem}_checkpoint"
        checkpoint_path = results_dir / f"{checkpoint_stem}.npy"
        checkpoint_metadata_path = results_dir / f"{checkpoint_stem}.json"
        checkpoint_loss_path = results_dir / f"{checkpoint_stem}_loss_curve_planck.png"
        np.save(checkpoint_path, recovered)
        checkpoint_metadata = {
            "run_stem": checkpoint_stem,
            "final_file": checkpoint_path.name,
            "stage1_file": None,
            "stage2_file": None,
            "loss_curve_file": checkpoint_loss_path.name,
            "checkpoint_every": checkpoint_every,
            "checkpoint_step": global_step,
            "converged": converged,
            "validation_steps": validation_steps,
            "validation_losses": validation_losses,
            "validation_relative_changes": validation_relative_changes,
            "elapsed_seconds": time.perf_counter() - patch_start_time,
            "defaults": _build_default_config(),
            "identity_config": weight_test_identity_config(),
            "config": run_config,
        }
        checkpoint_metadata_path.write_text(
            json.dumps(checkpoint_metadata, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        save_loss_curve(loss_calls, checkpoint_loss_path)
        print("Checkpoint saved:", checkpoint_path)
        print("Checkpoint saved:", checkpoint_metadata_path)
        print("Checkpoint saved:", checkpoint_loss_path)

    def remove_checkpoint() -> None:
        checkpoint_stem = f"{out_stem}_checkpoint"
        for path in (
            results_dir / f"{checkpoint_stem}.npy",
            results_dir / f"{checkpoint_stem}.json",
            results_dir / f"{checkpoint_stem}_loss_curve_planck.png",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def make_running_signal_qu() -> torch.Tensor:
        if initial_signal_qu is not None:
            running_signal_qu_local = torch.from_numpy(initial_signal_qu).to(
                device=device, dtype=dtype
            )
        elif white_noise_initial:
            running_signal_qu_local = torch.stack(
                [
                    torch.randn_like(d_q_t) * init_std_q,
                    torch.randn_like(d_u_t) * init_std_u,
                ],
                dim=0,
            )
            if patch_mask_t is not None:
                running_signal_qu_local *= patch_mask_t
        else:
            running_signal_qu_local = torch.stack([d_q_t.clone(), d_u_t.clone()], dim=0)
        running_signal_qu_local.requires_grad_()
        return running_signal_qu_local

    def make_optimizer() -> torch.optim.LBFGS:
        return torch.optim.LBFGS(
            [running_signal_qu],
            lr=1,
            max_iter=1,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None,
        )

    def evaluate_validation_loss(phase: str) -> float:
        with torch.no_grad():
            if phase not in validation_target_by_phase:
                validation_target_batch = make_target_batch(
                    validation_nb,
                    phase=phase,
                    batch_nq=validation_nq,
                    batch_nu=validation_nu,
                )
                validation_target_dc = DataClass(validation_target_batch, pbc=pbc)
                validation_target_by_phase[phase] = stats_flat(
                    validation_target_dc, phase=phase
                )

            validation_running_batch = make_running_batch(
                running_signal_qu,
                phase=phase,
                batch_nq=validation_nq,
                batch_nu=validation_nu,
            )
            validation_running_dc = DataClass(validation_running_batch, pbc=pbc)
            validation_running_flat = stats_flat(
                validation_running_dc, phase=phase
            )
            validation_target_flat = validation_target_by_phase[phase]
            if validation_running_flat.numel() != validation_target_flat.numel():
                raise RuntimeError(
                    "Validation statistic length mismatch: "
                    f"running={validation_running_flat.numel()} "
                    f"target={validation_target_flat.numel()}."
                )
            return float(
                squared_l2(validation_running_flat - validation_target_flat)
                .detach()
                .cpu()
            )

    optimization_start_time = time.perf_counter()
    for epoch_idx in range(epochs):
        if epoch_idx > 0:
            convergence_hits = 0
            previous_convergence_loss = None
        running_signal_qu = make_running_signal_qu()
        optimizer = make_optimizer()
        for step_idx in range(epoch_steps):
            use_without_noise_phase = (
                start_without_noise_channels
                and step_idx < (epoch_steps // 2)
            )
            phase = "without_noise_channels" if use_without_noise_phase else "with_noise_channels"
            idx = rng.choice(
                training_indices,
                size=min(n_batch, training_indices.size),
                replace=False,
            )
            q_batch_np = _load_masked_noise_batch(
                [noise_q_paths[int(i)] for i in idx],
                patch_mask=patch_mask,
                expected_hw=(H, W),
                crop_hw=(H, W) if map_size is not None else None,
            )
            u_batch_np = _load_masked_noise_batch(
                [noise_u_paths[int(i)] for i in idx],
                patch_mask=patch_mask,
                expected_hw=(H, W),
                crop_hw=(H, W) if map_size is not None else None,
            )
            batch_nq = torch.from_numpy(q_batch_np).to(device, dtype=dtype)
            batch_nu = torch.from_numpy(u_batch_np).to(device, dtype=dtype)
            nb = int(idx.shape[0])

            with torch.no_grad():
                target_batch = make_target_batch(
                    nb,
                    phase=phase,
                    batch_nq=batch_nq,
                    batch_nu=batch_nu,
                )
                target_dc = DataClass(target_batch, pbc=pbc)
                target_flat = stats_flat(target_dc, phase=phase)
                if not printed_target_size_by_phase[phase]:
                    print(f"ST coefficient count for phase '{phase}': {target_flat.numel()}")
                    printed_target_size_by_phase[phase] = True

            def closure():
                optimizer.zero_grad()
                running_batch = make_running_batch(
                    running_signal_qu,
                    phase=phase,
                    batch_nq=batch_nq,
                    batch_nu=batch_nu,
                )

                running_dc = DataClass(running_batch, pbc=pbc)
                running_flat = stats_flat(running_dc, phase=phase)

                if running_flat.numel() != target_flat.numel():
                    raise RuntimeError(
                        f"Flattened statistic length mismatch: running={running_flat.numel()} target={target_flat.numel()}."
                    )

                loss = squared_l2(running_flat - target_flat)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_value = float(loss.detach().cpu())
            loss_calls.append(loss_value)
            print(
                f"Epoch {epoch_idx+1}/{epochs} | step {step_idx+1}/{epoch_steps} | phase {phase} | minibatch {nb} | loss: {loss_value:.6g}"
            )
            global_step = epoch_idx * epoch_steps + step_idx + 1
            completed_steps = global_step
            if global_step % checkpoint_every == 0:
                save_checkpoint(global_step)

            should_check_convergence = (
                global_step % convergence_check_every == 0
                or global_step == epochs * epoch_steps
            )
            if should_check_convergence:
                validation_loss = evaluate_validation_loss(phase)
                validation_steps.append(global_step)
                validation_losses.append(validation_loss)
                eligible = phase == "with_noise_channels"

                relative_change = None
                if eligible and previous_convergence_loss is not None:
                    relative_change = abs(
                        validation_loss - previous_convergence_loss
                    ) / max(abs(previous_convergence_loss), np.finfo(float).eps)
                validation_relative_changes.append(relative_change)

                below_threshold = eligible and validation_loss <= convergence_threshold
                previous_below_threshold = (
                    previous_convergence_loss is not None
                    and previous_convergence_loss <= convergence_threshold
                )
                stable = (
                    relative_change is not None
                    and relative_change <= convergence_relative_tolerance
                )
                if below_threshold and previous_below_threshold and stable:
                    convergence_hits += 1
                else:
                    convergence_hits = 0
                previous_convergence_loss = validation_loss if eligible else None

                elapsed = time.perf_counter() - optimization_start_time
                seconds_per_step = elapsed / global_step
                remaining_steps = max(0, epochs * epoch_steps - global_step)
                relative_change_text = (
                    "n/a" if relative_change is None else f"{relative_change:.6g}"
                )
                print(
                    f"Validation | global step {global_step}/{epochs * epoch_steps} | "
                    f"normalized loss: {validation_loss:.6g} | "
                    f"relative change: {relative_change_text} "
                    f"(limit {convergence_relative_tolerance:g}) | "
                    f"stable hits: {convergence_hits}/{convergence_patience} | "
                    f"elapsed: {format_duration(elapsed)} | "
                    f"hard-cap ETA: {format_duration(seconds_per_step * remaining_steps)}"
                )

                if eligible and convergence_hits >= convergence_patience:
                    converged = True
                    print(
                        f"Converged patch {patch} at global step {global_step}: "
                        f"validation loss {validation_loss:.6g} <= "
                        f"{convergence_threshold:g}, with {convergence_patience} "
                        f"consecutive relative changes <= "
                        f"{convergence_relative_tolerance:g}."
                    )
                    break
        if converged:
            break

    optimization_elapsed_seconds = time.perf_counter() - optimization_start_time
    if not converged:
        final_validation = validation_losses[-1] if validation_losses else float("nan")
        print(
            f"Patch {patch} reached the hard maximum of {completed_steps} steps "
            f"without satisfying convergence; final validation loss: "
            f"{final_validation:.6g}."
        )

    recovered_qu = _unmask_recovered(running_signal_qu, patch_mask)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    out_path = results_dir / f"{out_stem}.npy"
    np.save(out_path, recovered_qu)

    metadata_path = results_dir / f"{out_stem}.json"
    metadata = {
        "run_stem": out_stem,
        "final_file": out_path.name,
        "stage1_file": None,
        "stage2_file": None,
        "loss_curve_file": f"{out_stem}_loss_curve_planck.png",
        "checkpoint_every": checkpoint_every,
        "completed_steps": completed_steps,
        "converged": converged,
        "validation_steps": validation_steps,
        "validation_losses": validation_losses,
        "validation_relative_changes": validation_relative_changes,
        "optimization_elapsed_seconds": optimization_elapsed_seconds,
        "defaults": _build_default_config(),
        "identity_config": weight_test_identity_config(),
        "config": run_config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")

    loss_path = results_dir / f"{out_stem}_loss_curve_planck.png"
    save_loss_curve(loss_calls, loss_path)
    remove_checkpoint()

    print("Saved:", out_path)
    print("Saved:", metadata_path)
    print("Saved:", loss_path)
    total_elapsed_seconds = time.perf_counter() - patch_start_time
    print(
        f"Patch {patch} timing | optimization: "
        f"{format_duration(optimization_elapsed_seconds)} "
        f"({optimization_elapsed_seconds:.1f} s) | total: "
        f"{format_duration(total_elapsed_seconds)} ({total_elapsed_seconds:.1f} s) | "
        f"completed steps: {completed_steps}/{epochs * epoch_steps} | "
        f"converged: {converged}"
    )


def main() -> None:
    try:
        from mpi4py import MPI
    except ImportError:
        comm = None
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        size = int(os.environ.get("SLURM_NTASKS", "1"))
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    patches = _parse_patch_list()
    if rank == 0:
        print(f"Patch count: {len(patches)} | MPI ranks: {size}")
        print(f"Patch range/list starts with: {patches[:min(8, len(patches))]}")

    for patch in patches[rank::size]:
        _run_patch(patch, rank=rank)
        torch.cuda.empty_cache()

    if comm is not None:
        comm.Barrier()
    if rank == 0:
        print("All assigned patches completed.")


if __name__ == "__main__":
    main()
