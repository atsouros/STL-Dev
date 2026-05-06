#!/usr/bin/env python3

from __future__ import annotations

import hashlib
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


# -----------------------------------------------------------------------------
# Set ROOT
# -----------------------------------------------------------------------------
# This should point to the directory that contains:
#   ROOT / "signal" / "patch_3" / ...
#   ROOT / "nuisance" / "patch_3" / ...
#
# You said you'll fill this in; you can either:
# - edit this constant, or
# - set ROOT=/path/to/root at runtime.
ROOT = Path("SET_ROOT_HERE")


# Repo root (for imports)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import STL_main.torch_backend as bk

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as FFT_DataClass
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch as Kernel_DataClass


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
        "st_has_fewer_convolutions": st_has_fewer_convolutions,
    }


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
        "st_has_fewer_convolutions": True,
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


def main() -> None:
    # Resolve ROOT (either from env var or by editing the constant above).
    root_env = (os.environ.get("ROOT") or "").strip()
    root = Path(root_env).expanduser().resolve() if root_env else ROOT
    if str(root) == "SET_ROOT_HERE":
        raise RuntimeError(
            "Please set ROOT or export ROOT "
            "to the directory that contains `signal/` and `nuisance/`."
        )

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    seed = int(os.environ.get("SEED", "0"))
    n_batch = int(os.environ.get("BATCH", "15"))
    epochs = int((os.environ.get("EPOCHS") or os.environ.get("OUTER_ITERS") or "10").strip())
    epoch_steps = int((os.environ.get("EPOCH_STEPS") or "1").strip())
    lbfgs_max_iter = int(os.environ.get("LBFGS_MAX_ITER", "100"))
    start_without_noise_channels = (
        os.environ.get("START_WITHOUT_NOISE_CHANNELS") or "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    pbc = (os.environ.get("PBC") or os.environ.get("PBC") or "0").strip().lower() in {
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
    st_has_fewer_convolutions = (
        os.environ.get("ST_FEWER_CONVOLUTIONS") or "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    wtype = os.environ.get("WTYPE", "Bump-Steerable")
    patch = str(os.environ.get("PATCH", "3")).strip()
    map_size_env = (os.environ.get("MAP_SIZE") or "").strip()
    map_size = int(map_size_env) if map_size_env else None

    backend = (os.environ.get("BACKEND") or "fft").strip().lower()
    if backend not in {"fft", "kernel"}:
        raise ValueError("BACKEND must be 'fft' or 'kernel'")
    DataClass = FFT_DataClass if backend == "fft" else Kernel_DataClass

    # -------------------------------------------------------------------------
    # Device / dtype
    # -------------------------------------------------------------------------
    device_override = (os.environ.get("DEVICE") or "").strip()
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (backend={backend})")

    dtype_str = (os.environ.get("DTYPE") or "float64").strip().lower()
    if dtype_str in {"float64", "fp64", "double"}:
        dtype = torch.float64
    elif dtype_str in {"float32", "fp32", "single"}:
        dtype = torch.float32
    else:
        raise ValueError("DTYPE must be float64 or float32")

    _configure_backend_defaults(device=device, dtype=dtype)
    torch.manual_seed(seed)
    print(f"PBC: {pbc}")
    print(f"Epochs: {epochs} | steps per epoch: {epoch_steps}")
    print("Optimizer: lbfgs | one optimizer iteration per logged step")
    if st_reduced:
        print(
            "Reduced ST config: "
            f"J={st_j} | L={st_l} | iso={st_iso} | angular_ft={st_angular_ft} | "
            f"scale_ft={st_scale_ft} | harmonics_angle={st_harmonics_angle} | "
            f"harmonics_scale={st_harmonics_scale} | dj={st_dj} | "
            f"compute_PS={st_compute_ps} | fewer_convolutions={st_has_fewer_convolutions}"
        )
    else:
        print(
            "Standard ST config: "
            f"compute_PS={st_compute_ps} | fewer_convolutions={st_has_fewer_convolutions}"
        )

    # -------------------------------------------------------------------------
    # Load Planck patch inputs
    # -------------------------------------------------------------------------
    signal_dir = root / "signal" / f"patch_{patch}"
    nuis_q_dir = root / "nuisance" / f"patch_{patch}" / "Stokes_Q"
    nuis_u_dir = root / "nuisance" / f"patch_{patch}" / "Stokes_U"

    q353_path = next(signal_dir.glob(f"patch_{patch}_Q353*.npy"), None)
    u353_path = next(signal_dir.glob(f"patch_{patch}_U353*.npy"), None)
    i857_path = next(signal_dir.glob(f"patch_{patch}_I857*.npy"), None)
    if q353_path is None or u353_path is None or i857_path is None:
        raise FileNotFoundError(
            "Could not find one of Q353/U353/I857 in "
            f"{signal_dir}. Expected files matching patch_{patch}_Q353*.npy, patch_{patch}_U353*.npy, patch_{patch}_I857*.npy."
        )

    print("Q353:", q353_path.name)
    print("U353:", u353_path.name)
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

    # -------------------------------------------------------------------------
    # Pair nuisance samples by (noise_seed, CMB_res_seed)
    # -------------------------------------------------------------------------
    pat = re.compile(r"noise_seed_(\d+)_CMB_res_seed_(\d+)")

    def parse_key(path: Path) -> tuple[int, int]:
        m = pat.search(path.name)
        if not m:
            raise ValueError(f"Could not parse seeds from: {path.name}")
        return (int(m.group(1)), int(m.group(2)))

    q_paths = sorted(nuis_q_dir.glob(f"patch_{patch}_noise_Q353*.npy"))
    u_paths = sorted(nuis_u_dir.glob(f"patch_{patch}_noise_U353*.npy"))
    if not q_paths or not u_paths:
        raise FileNotFoundError(
            "Could not find nuisance samples. Looked for "
            f"{nuis_q_dir}/'patch_{patch}_noise_Q353*.npy' and {nuis_u_dir}/'patch_{patch}_noise_U353*.npy'."
        )

    # Prefer seed-pairing when filename pattern is available; otherwise fall back to zip-by-index.
    try:
        q_by_key = {parse_key(p): p for p in q_paths}
        u_by_key = {parse_key(p): p for p in u_paths}
        paired_keys = sorted(set(q_by_key).intersection(u_by_key))
        if not paired_keys:
            raise RuntimeError("No paired nuisance samples found for Q353/U353 (seed keys empty).")
        noise_q_paths = [q_by_key[k] for k in paired_keys]
        noise_u_paths = [u_by_key[k] for k in paired_keys]
        print("Paired nuisance samples:", len(paired_keys))
        print("First key:", paired_keys[0], "->", noise_q_paths[0].name)
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

    # -------------------------------------------------------------------------
    # Phase 1 (optional): 3 channels [dQ, dU, aux], no explicit noise channels
    # Phase 2: 5 channels [dQ, nQ, dU, nU, aux]
    # -------------------------------------------------------------------------
    CROSS_MATRIX_NO_NOISE_CHANNELS = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    CROSS_MATRIX_WITH_NOISE_CHANNELS = torch.tensor(
        [
            [1, 0, 0, 0, 1],
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
    ref_i = int(rng.integers(0, n_noise))
    n_q_ref = _downsample_by_four(np.load(noise_q_paths[ref_i]).astype(np.float64))
    n_u_ref = _downsample_by_four(np.load(noise_u_paths[ref_i]).astype(np.float64))
    if map_size is not None:
        n_q_ref = _center_crop(n_q_ref, out_hw=(H, W))
        n_u_ref = _center_crop(n_u_ref, out_hw=(H, W))

    ref_tensor_3 = torch.from_numpy(np.stack([d_q, d_u, aux], axis=0)).to(device, dtype=dtype)
    ref_dc_3 = DataClass(ref_tensor_3[None, ...], pbc=pbc)  # (1, 3, H, W)

    ref_tensor_5 = torch.from_numpy(np.stack([d_q, n_q_ref, d_u, n_u_ref, aux], axis=0)).to(
        device, dtype=dtype
    )  # (5, H, W)
    ref_dc_5 = DataClass(ref_tensor_5[None, ...], pbc=pbc)  # (1, 5, H, W)

    def build_st_op(ref_dc):
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
            )
        else:
            st_op_local = ref_dc.get_ST_op(
                compute_PS=st_compute_ps,
                has_fewer_convolutions=st_has_fewer_convolutions,
            )
        _maybe_set_wtype(st_op=st_op_local, ref_dc=ref_dc, wtype=wtype)
        return st_op_local

    st_op_3 = build_st_op(ref_dc_3)
    st_op_5 = build_st_op(ref_dc_5)

    with torch.no_grad():
        st_op_3.apply(ref_dc_3, norm="store_ref", compute_cross_matrix=CROSS_MATRIX_REF_3)
        st_op_5.apply(ref_dc_5, norm="store_ref", compute_cross_matrix=CROSS_MATRIX_REF_5)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    d_q_t = torch.from_numpy(np.copy(d_q)).to(device, dtype=dtype)
    d_u_t = torch.from_numpy(np.copy(d_u)).to(device, dtype=dtype)
    aux_t = torch.from_numpy(np.copy(aux)).to(device, dtype=dtype)
    def stats_flat(dc: DataClass, *, phase: str) -> torch.Tensor:
        if phase == "without_noise_channels":
            return st_op_3.apply(
                dc, norm="load_ref", compute_cross_matrix=CROSS_MATRIX_NO_NOISE_CHANNELS
            ).to_flatten(mean_along_batch=True, keepnans=False)
        return st_op_5.apply(
            dc, norm="load_ref", compute_cross_matrix=CROSS_MATRIX_WITH_NOISE_CHANNELS
        ).to_flatten(mean_along_batch=True, keepnans=False)

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

    def squared_l2(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().square().sum()

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
    print(f"Start without explicit noise channels: {start_without_noise_channels}")
    printed_target_size_by_phase = {
        "without_noise_channels": False,
        "with_noise_channels": False,
    }

    def make_running_signal_qu() -> torch.Tensor:
        if white_noise_initial:
            running_signal_qu_local = torch.stack(
                [
                    torch.randn_like(d_q_t) * init_std_q,
                    torch.randn_like(d_u_t) * init_std_u,
                ],
                dim=0,
            )
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

    for epoch_idx in range(epochs):
        running_signal_qu = make_running_signal_qu()
        optimizer = make_optimizer()
        for step_idx in range(epoch_steps):
            use_without_noise_phase = (
                start_without_noise_channels
                and step_idx < (epoch_steps // 2)
            )
            phase = "without_noise_channels" if use_without_noise_phase else "with_noise_channels"
            idx = rng.choice(n_noise, size=min(n_batch, n_noise), replace=False)
            q_batch_np = _load_noise_batch(
                [noise_q_paths[int(i)] for i in idx],
                expected_hw=(H, W),
                crop_hw=(H, W) if map_size is not None else None,
            )
            u_batch_np = _load_noise_batch(
                [noise_u_paths[int(i)] for i in idx],
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
                s_q = running_signal_qu[0]
                s_u = running_signal_qu[1]

                if phase == "without_noise_channels":
                    running_batch = torch.stack(
                        [
                            s_q[None, :, :] + batch_nq,
                            s_u[None, :, :] + batch_nu,
                            aux_t.expand(nb, -1, -1),
                        ],
                        dim=1,
                    )
                else:
                    running_batch = torch.stack(
                        [
                            s_q[None, :, :] + batch_nq,
                            (d_q_t - s_q)[None, :, :].expand(nb, -1, -1),
                            s_u[None, :, :] + batch_nu,
                            (d_u_t - s_u)[None, :, :].expand(nb, -1, -1),
                            aux_t.expand(nb, -1, -1),
                        ],
                        dim=1,
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

    recovered_q = running_signal_qu.detach().cpu().numpy()[0]
    recovered_u = running_signal_qu.detach().cpu().numpy()[1]

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    out_dir_env = (os.environ.get("OUTDIR") or "results").strip()
    results_dir = Path(out_dir_env)
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
        st_has_fewer_convolutions=st_has_fewer_convolutions,
    )
    out_stem = _build_run_stem(run_config)
    out_path = results_dir / f"{out_stem}.npy"
    np.save(out_path, np.stack([recovered_q, recovered_u], axis=0))

    metadata_path = results_dir / f"{out_stem}.json"
    metadata = {
        "run_stem": out_stem,
        "final_file": out_path.name,
        "stage1_file": None,
        "stage2_file": None,
        "loss_curve_file": "loss_curve_planck.png",
        "defaults": _build_default_config(),
        "identity_config": _build_identity_config(run_config),
        "config": run_config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(loss_calls, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("LBFGS closure call index")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs time (closure calls)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = results_dir / "loss_curve_planck.png"
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)

    print("Saved:", out_path)
    print("Saved:", metadata_path)
    print("Saved:", loss_path)


if __name__ == "__main__":
    main()