from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import STL_main.torch_backend as bk


# NERSC patch layout. Override with PLANCK_SIGNAL_DIR / PLANCK_NUISANCE_DIR if needed.
SIGNAL_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/signal")
NUISANCE_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance")


def _maybe_set_wtype(st_op, ref_dc, wtype: str) -> None:
    """
    Try to set the wavelet type on backends that support it.
    """
    try:
        st_op.wavelet_op = ref_dc.get_wavelet_op(J=st_op.J, L=st_op.L, WType=wtype)
        st_op.WType = getattr(st_op.wavelet_op, "WType", wtype)
    except TypeError:
        pass


def _configure_backend_defaults(device: torch.device, dtype: torch.dtype) -> None:
    """
    Support both backend variants:
    - older API exposing bk.set_default_device(...)
    - newer API exposing only bk._DEFAULT_* globals
    """
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64 if dtype == torch.float32 else torch.complex128


def _downsample_by_four(image: np.ndarray) -> np.ndarray:
    """
    Downsample by averaging over 2x2 blocks (reduces each dimension by 2).

    This matches the "downsample_by_four" style you shared (area reduced by 4).
    """
    H, W = image.shape[-2:]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Image dimensions must be even for 2x2 downsampling (got {H}x{W}).")
    if image.ndim == 2:
        return image.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    raise ValueError(f"Unsupported image shape for downsampling: {image.shape} (expected 2D).")


def _center_crop(image: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Center-crop a 2D image to (out_h, out_w).
    """
    if image.ndim != 2:
        raise ValueError(f"Unsupported image shape for cropping: {image.shape} (expected 2D).")
    out_h, out_w = out_hw
    H, W = image.shape
    if out_h > H or out_w > W:
        raise ValueError(f"Cannot crop {H}x{W} to larger size {out_h}x{out_w}.")
    y0 = (H - out_h) // 2
    x0 = (W - out_w) // 2
    return image[y0 : y0 + out_h, x0 : x0 + out_w]


def _load_noise_batch(
    paths: list[Path],
    *,
    expected_hw: tuple[int, int] | None = None,
    crop_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Load a batch of nuisance/noise maps from .npy paths.

    Returns
    -------
    np.ndarray
        Array of shape (Nb, H, W), dtype float64.
    """
    maps: list[np.ndarray] = []
    for p in paths:
        x = _downsample_by_four(np.load(p).astype(np.float64))
        if crop_hw is not None:
            x = _center_crop(x, out_hw=crop_hw)
        maps.append(x)

    arr = np.stack(maps, axis=0)
    if expected_hw is not None and arr.shape[1:] != expected_hw:
        raise RuntimeError(f"Noise batch has wrong shape: {arr.shape} vs expected (*, {expected_hw[0]}, {expected_hw[1]})")
    return arr


def _select_no_bonus_signal_path(signal_dir: Path, pattern: str, label: str) -> Path:
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


def _nuisance_version_suffix(version: str) -> str | None:
    if version == "all":
        return None
    return f"_{version}.npy"


def _filter_nuisance_version(paths: list[Path], version: str) -> list[Path]:
    suffix = _nuisance_version_suffix(version)
    if suffix is None:
        return paths
    return [path for path in paths if path.name.endswith(suffix)]


def _short_bool(value: bool) -> str:
    return "1" if value else "0"


def _short_dtype(dtype_str: str) -> str:
    aliases = {
        "float64": "f64",
        "fp64": "f64",
        "double": "f64",
        "float32": "f32",
        "fp32": "f32",
        "single": "f32",
    }
    return aliases.get(dtype_str.lower(), dtype_str.lower())


def _short_wtype(wtype: str) -> str:
    return "".join(ch.lower() for ch in wtype if ch.isalnum())


def _build_run_config(
    *,
    root: Path,
    patch: str,
    map_size: int | None,
    device: torch.device,
    backend: str,
    dtype_str: str,
    wtype: str,
    seed: int,
    n_batch: int,
    epochs: int,
    epoch_steps: int,
    lbfgs_max_iter: int,
    pbc: bool,
    white_noise_initial: bool,
    start_without_noise_channels: bool,
    nuisance_version: str,
    st_reduced: bool,
    st_j: int,
    st_l: int,
    st_iso: bool,
    st_angular_ft: bool,
    st_scale_ft: bool,
    st_harmonics_angle: int,
    st_harmonics_scale: int,
    st_dj: int,
    st_compute_ps: bool,
    st_ps_method: str,
    st_ps_n_bins: int | None,
    st_has_fewer_convolutions: bool,
) -> dict[str, object]:
    return {
        "root": str(root),
        "patch": patch,
        "map_size": map_size,
        "device": str(device),
        "backend": backend,
        "dtype": dtype_str,
        "wtype": wtype,
        "seed": seed,
        "batch": n_batch,
        "epochs": epochs,
        "epoch_steps": epoch_steps,
        "lbfgs_max_iter": lbfgs_max_iter,
        "pbc": pbc,
        "white_noise_initial": white_noise_initial,
        "start_without_noise_channels": start_without_noise_channels,
        "nuisance_version": nuisance_version,
        "st_reduced": st_reduced,
        "st_j": st_j,
        "st_l": st_l,
        "st_iso": st_iso,
        "st_angular_ft": st_angular_ft,
        "st_scale_ft": st_scale_ft,
        "st_harmonics_angle": st_harmonics_angle,
        "st_harmonics_scale": st_harmonics_scale,
        "st_dj": st_dj,
        "st_compute_ps": st_compute_ps,
        "st_ps_method": st_ps_method,
        "st_ps_n_bins": st_ps_n_bins,
        "st_has_fewer_convolutions": st_has_fewer_convolutions,
    }


def _build_default_config() -> dict[str, object]:
    return {
        "patch": "3",
        "map_size": 384,
        "backend": "fft",
        "dtype": "float64",
        "wtype": "Bump-Steerable",
        "seed": 2,
        "batch": 15,
        "epochs": 2,
        "epoch_steps": 50,
        "lbfgs_max_iter": 100,
        "pbc": False,
        "white_noise_initial": False,
        "start_without_noise_channels": False,
        "nuisance_version": "v4_10_arcmin",
        "st_reduced": False,
        "st_j": 7,
        "st_l": 4,
        "st_iso": False,
        "st_angular_ft": True,
        "st_scale_ft": True,
        "st_harmonics_angle": 2,
        "st_harmonics_scale": 3,
        "st_dj": 3,
        "st_compute_ps": False,
        "st_ps_method": "legacy",
        "st_ps_n_bins": None,
        "st_has_fewer_convolutions": False,
    }


def _build_identity_config(config: dict[str, object]) -> dict[str, object]:
    identity_keys = (
        "patch",
        "map_size",
        "backend",
        "dtype",
        "wtype",
        "seed",
        "batch",
        "epochs",
        "epoch_steps",
        "lbfgs_max_iter",
        "pbc",
        "white_noise_initial",
        "start_without_noise_channels",
        "nuisance_version",
        "st_reduced",
        "st_j",
        "st_l",
        "st_iso",
        "st_angular_ft",
        "st_scale_ft",
        "st_harmonics_angle",
        "st_harmonics_scale",
        "st_dj",
        "st_compute_ps",
        "st_ps_method",
        "st_ps_n_bins",
        "st_has_fewer_convolutions",
    )
    return {key: config[key] for key in identity_keys}


def _build_run_stem(config: dict[str, object]) -> str:
    payload = json.dumps(_build_identity_config(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    map_size_tag = "full" if config["map_size"] is None else str(config["map_size"])
    return (
        f"p{config['patch']}_m{map_size_tag}_{config['backend']}_{_short_dtype(str(config['dtype']))}"
        f"_s{config['seed']}_r{_short_bool(bool(config['st_reduced']))}"
        f"_i{_short_bool(bool(config['st_iso']))}_p{_short_bool(bool(config['pbc']))}"
        f"_n{_short_bool(bool(config['start_without_noise_channels']))}"
        f"_{_short_wtype(str(config['wtype']))}_{digest}"
    )


def _parse_patch_list() -> list[str]:
    patch_list_env = (os.environ.get("PATCH_LIST") or "").strip()
    if patch_list_env:
        return [p.strip() for p in patch_list_env.split(",") if p.strip()]

    single_patch = (os.environ.get("PATCH") or "").strip()
    if single_patch:
        return [single_patch]

    patch_start = int(os.environ.get("PATCH_START", "4"))
    n_patches_env = (os.environ.get("N_PATCHES") or "").strip()
    if n_patches_env:
        patch_end = patch_start + int(n_patches_env) - 1
    else:
        patch_end = int(os.environ.get("PATCH_END", "192"))
    return [str(p) for p in range(patch_start, patch_end + 1)]
