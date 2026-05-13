#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import STL_main_old.torch_backend as bk
from STL_main_old.STL_2D_FFT_Torch import STL_2D_FFT_Torch
from STL_main_old.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA_POS = float(
    os.environ.get(
        "LAMBDA_POS",
        os.environ.get("LAMB", os.environ.get("COMPSEP_LAMB_POS", os.environ.get("COMPSEP_LAMB", "0"))),
    ).strip()
)
LAMBDA_NOISE = float(
    os.environ.get(
        "LAMBDA_NOISE", os.environ.get("LAMB_NOISE", os.environ.get("COMPSEP_LAMB_NOISE", "0"))
    ).strip()
)

# Integer-attractor schedule for the noise term:
#   penalty(noise) = mean( 1 - |sin(pi * (noise + 0.5))|^m ) = mean( 1 - |cos(pi * noise)|^m ),
#   with m = (outer_step_index + 1).
NOISE_INT_M_MAX_ENV = (os.environ.get("XRAY_NOISE_INT_M_MAX") or os.environ.get("COMPSEP_NOISE_INT_M_MAX") or "").strip()
NOISE_INT_M_MAX = int(NOISE_INT_M_MAX_ENV) if NOISE_INT_M_MAX_ENV else None
NOISE_INT_EPS = float(
    (os.environ.get("XRAY_NOISE_INT_EPS") or os.environ.get("COMPSEP_NOISE_INT_EPS") or "1e-12").strip()
)

MAP_SIZE = int(os.environ.get("XRAY_MAP_SIZE", "512"))
if MAP_SIZE not in (512, 1024):
    raise ValueError(f"XRAY_MAP_SIZE must be 512 or 1024, got {MAP_SIZE}")

# Optional radial low-pass filtering (post-processing of the recovered map only)
# to remove checker-grid artifacts by suppressing power near/above Nyquist.
APPLY_NYQUIST_FILTER = (os.environ.get("XRAY_APPLY_NYQUIST_FILTER", "0")).strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)
NYQUIST_CUTOFF_FRAC = float(os.environ.get("XRAY_NYQUIST_CUTOFF_FRAC", "0.8"))
if not (0.0 < NYQUIST_CUTOFF_FRAC <= 1.0):
    raise ValueError(f"XRAY_NYQUIST_CUTOFF_FRAC must be in (0,1], got {NYQUIST_CUTOFF_FRAC}")

# Optional smooth Nyquist taper during optimization (frequency-domain).
# Uses radial k = sqrt(kx^2 + ky^2) with kx,ky in cycles/pixel from torch.fft.fftfreq(·, d=1.0).
# The 2D radial maximum on the square Nyquist grid is k_Nyq = sqrt((1/2)^2 + (1/2)^2).
APPLY_NYQUIST_TAPER_DURING_OPT = (
    os.environ.get("XRAY_APPLY_NYQUIST_TAPER_DURING_OPT", "0").strip().lower()
    in ("1", "true", "yes", "y")
)
NYQUIST_TAPER_FRAC_DURING_OPT = float(os.environ.get("XRAY_NYQUIST_TAPER_FRAC_DURING_OPT", "1.0"))
if not (0.0 < NYQUIST_TAPER_FRAC_DURING_OPT <= 1.0):
    raise ValueError(
        "XRAY_NYQUIST_TAPER_FRAC_DURING_OPT must be in (0,1], got "
        f"{NYQUIST_TAPER_FRAC_DURING_OPT}"
    )


def _filter_radial_np(img: np.ndarray, filt_shifted: np.ndarray) -> np.ndarray:
    """
    Apply a radial filter in Fourier space (NumPy).

    filt_shifted is assumed to be centered (DC at center). We ifftshift it before multiplying.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={img.shape}")
    if filt_shifted.shape != img.shape:
        raise ValueError(f"Filter shape {filt_shifted.shape} != image shape {img.shape}")
    img_f = np.fft.fft2(img)
    filt = np.fft.ifftshift(filt_shifted, axes=(-2, -1))
    return np.fft.ifft2(filt * img_f).real


def _make_radial_lowpass_np(h: int, w: int, frac: float) -> np.ndarray:
    """
    Low-pass filter with cosine taper between r0 and Nyquist.

    Matches the helper used in plot.ipynb (in spirit).
    """
    r_ny = 0.5 * float(min(h, w))
    r0 = float(frac) * r_ny

    x = np.arange(h)[:, None]
    y = np.arange(w)[None, :]
    r = np.sqrt((x - h // 2) ** 2 + (y - w // 2) ** 2)

    filt = np.zeros_like(r, dtype=np.float64)
    filt[r <= r0] = 1.0
    mid = (r > r0) & (r < r_ny)
    if np.any(mid):
        filt[mid] = 0.5 * (1.0 + np.cos(np.pi * (r[mid] - r0) / (r_ny - r0)))
    return filt


def _make_nyquist_taper_mask_torch(
    h: int, w: int, frac: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Smooth radial taper mask in *unshifted* FFT coordinates.

    The mask equals 1 for k <= k0 and smoothly tapers to 0 at k_Nyq, where
      k = sqrt(kx^2 + ky^2),  kx,ky in cycles/pixel (d=1),
      k_Nyq = sqrt((1/2)^2 + (1/2)^2).
    """
    kx = torch.fft.fftfreq(w, d=1.0, device=device)
    ky = torch.fft.fftfreq(h, d=1.0, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k = torch.sqrt(KX**2 + KY**2).to(dtype)

    k_nyq = float(0.5 * (2.0**0.5))  # sqrt((1/2)^2 + (1/2)^2)
    k0 = float(frac) * k_nyq

    mask = torch.zeros((h, w), device=device, dtype=dtype)
    mask = torch.where(k <= k0, torch.ones_like(mask), mask)
    mid = (k > k0) & (k < k_nyq)
    if bool(torch.any(mid)):
        mask = torch.where(
            mid,
            0.5 * (1.0 + torch.cos(torch.pi * (k - k0) / (k_nyq - k0))),
            mask,
        )
    return mask


def _apply_nyquist_taper_torch(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed Nyquist taper mask in Fourier space (Torch)."""
    img_f = torch.fft.fft2(img)
    return torch.fft.ifft2(img_f * mask).real


def _maybe_set_wtype(st_op, ref_dc, wtype: str) -> None:
    """
    Try to set the wavelet type on backends that support it.
    """
    try:
        st_op.wavelet_op = ref_dc.get_wavelet_op(J=st_op.J, L=st_op.L, WType=wtype)
        st_op.WType = getattr(st_op.wavelet_op, "WType", wtype)
    except TypeError:
        pass


def _load_fits_as_npy(fits_path: Path, npy_path: Path) -> np.ndarray:
    try:
        from astropy.io import fits
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading FITS requires astropy. Install it or pre-convert the FITS to a .npy file."
        ) from exc

    arr = fits.getdata(fits_path)
    arr = np.asarray(arr).squeeze()
    np.save(npy_path, arr)
    return arr


def _load_data_map(map_size: int) -> tuple[Path, np.ndarray]:
    fits_path = ROOT / f"x-ray data/eROSITA_coadd_LANCZOS3_img_{map_size}_8deg.fits"
    npy_path = fits_path.with_suffix(".npy")
    if npy_path.exists():
        data = np.load(npy_path)
    else:
        data = _load_fits_as_npy(fits_path, npy_path)
    data = np.asarray(data).squeeze()
    # Make native-endian contiguous float64 (FITS can be big-endian)
    data = np.array(data, dtype=np.float64, copy=True, order="C")
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D map; got shape={data.shape}")
    if data.shape != (map_size, map_size):
        raise ValueError(f"Expected a {map_size}x{map_size} map; got shape={data.shape}")

    # IMPORTANT: no additional processing here. The eROSITA coadd is assumed to be already meaningful/positive.
    return (npy_path if npy_path.exists() else fits_path), data


def _load_noise_bank(map_size: int) -> tuple[list[Path], np.ndarray]:
    noise_dir = ROOT / f"x-ray data/{map_size}"
    noise_files = sorted(noise_dir.glob(f"poisson_image_{map_size}_*.npy"))
    if not noise_files:
        raise FileNotFoundError(f"No noise files found in {noise_dir}")

    n_limit_env = os.environ.get("COMPSEP_N_NOISE")
    if n_limit_env:
        n_limit = int(n_limit_env)
        noise_files = noise_files[:n_limit]

    noise_samples = np.empty((len(noise_files), map_size, map_size), dtype=np.float32)
    for i, p in enumerate(noise_files):
        n = np.load(p)
        n = np.asarray(n).squeeze()
        # Make native-endian contiguous float32
        n = np.array(n, dtype=np.float32, copy=True, order="C")
        if n.shape != (map_size, map_size):
            raise ValueError(f"Unexpected noise shape in {p.name}: {n.shape}")
        noise_samples[i] = n
    return noise_files, noise_samples


def main() -> None:
    # -----------------
    # Device (GPU)
    # -----------------
    device_override = os.environ.get("COMPSEP_DEVICE")
    if device_override:
        device = torch.device(device_override)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Set COMPSEP_DEVICE to a CUDA device (e.g. cuda:0) "
                "or explicitly override with COMPSEP_DEVICE=cpu for debugging."
            )
        device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # -----------------
    # Dtype (reduce GPU memory)
    # -----------------
    dtype_override = (os.environ.get("COMPSEP_DTYPE") or "").strip().lower()
    if dtype_override in ("float32", "fp32"):
        torch_dtype = torch.float32
    elif dtype_override in ("float64", "fp64", "double"):
        torch_dtype = torch.float64
    else:
        # Default: float32 on CUDA to reduce memory; float64 otherwise.
        torch_dtype = torch.float32 if device.type == "cuda" else torch.float64
    print(f"Using dtype: {torch_dtype}")

    # -----------------
    # DataClass backend (FFT or Kernel)
    # -----------------
    dataclass_kind = (
        os.environ.get("COMPSEP_DATACLASS") or os.environ.get("XRAY_DATACLASS") or "FFT"
    ).strip().lower()
    if dataclass_kind in ("fft", "stl_2d_fft_torch", "stl_2d_fft"):
        DataClass = STL_2D_FFT_Torch
    elif dataclass_kind in ("kernel", "stl_2d_kernel_torch", "stl_2d_kernel"):
        DataClass = STL_2D_Kernel_Torch
    else:
        raise ValueError(
            "COMPSEP_DATACLASS must be 'FFT' or 'Kernel' (case-insensitive), got "
            f"{dataclass_kind!r}"
        )
    print(f"Using DataClass: {DataClass.__name__}")

    # STL uses its own backend defaults (bk.zeros, bk.from_numpy, ...) to pick dtypes
    # for intermediate statistic tensors. If those defaults remain float64 while we feed
    # float32 inputs, ST_Operator will hit dtype-mismatch errors during indexed writes.
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = torch_dtype
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64 if torch_dtype == torch.float32 else torch.complex128

    # -----------------
    # Observed data map d (no post-processing)
    # -----------------
    data_path, data = _load_data_map(MAP_SIZE)  # (S, S), float64
    H, W = data.shape
    print(f"Loaded data map from: {data_path}")

    rng = np.random.default_rng(int(os.environ.get("COMPSEP_SEED", "0")))

    # Noise samples loaded from disk (S maps) with no post-processing.
    noise_paths, noise_samples = _load_noise_bank(MAP_SIZE)  # (N, S, S), float32
    n_noise = int(noise_samples.shape[0])
    print(f"Loaded {n_noise} noise samples of shape {(H, W)}")
    if noise_paths:
        print(f"Example noise file: {noise_paths[0]}")

    # -----------------
    # 2-channel setup + cross-matrix
    # -----------------
    # Requested: include (0,1) cross terms but not (1,0) (ignored by STL for c1>c2).
    CROSS_MATRIX = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool, device=device)

    # -----------------
    # Normalization reference
    # -----------------
    n_ref = noise_samples[int(rng.integers(0, n_noise))].astype(np.float64, copy=False)
    ref_tensor = torch.from_numpy(np.stack([data, n_ref], axis=0)).to(device, dtype=torch_dtype)  # (2, H, W)
    ref_dc = DataClass(ref_tensor[None, ...], pbc=True)  # (1, 2, H, W)

    st_op = ref_dc.get_ST_op(compute_PS=True, has_fewer_convolutions=True)

    wtype = os.environ.get("COMPSEP_WTYPE", "Bump-Steerable")
    _maybe_set_wtype(st_op, ref_dc, wtype)

    with torch.no_grad():
        st_op.apply(ref_dc, norm="store_ref", compute_cross_matrix=CROSS_MATRIX)

    # -----------------
    # Helpers
    # -----------------
    data_t = torch.from_numpy(np.copy(data)).to(device, dtype=torch_dtype)  # (H, W)
    nyquist_taper_mask = None
    if APPLY_NYQUIST_TAPER_DURING_OPT:
        nyquist_taper_mask = _make_nyquist_taper_mask_torch(
            H, W, frac=NYQUIST_TAPER_FRAC_DURING_OPT, device=device, dtype=torch_dtype
        )

    def finite_mask(z: torch.Tensor) -> torch.Tensor:
        # torch.isfinite supports complex tensors; it returns true if both real/imag are finite.
        return torch.isfinite(z)

    def stats_flat(dc) -> torch.Tensor:
        return st_op.apply(dc, norm="load_ref", compute_cross_matrix=CROSS_MATRIX).to_flatten(
            # IMPORTANT: keep a fixed feature length. If we drop NaNs (keepnans=False),
            # numerical NaNs/Infs during optimization can change the flattened length and
            # crash the loss with a length mismatch. We keep NaNs and mask them in the loss.
            mean_along_batch=True,
            keepnans=True,
        )

    def squared_l2(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().square().sum()

    def make_target_batch(batch_noise: torch.Tensor) -> torch.Tensor:
        # X_target = [d, n_i]
        nb = int(batch_noise.shape[0])
        batch_data = data_t.expand(nb, -1, -1)  # (Nb, H, W)
        return torch.stack([batch_data, batch_noise], dim=1)  # (Nb, 2, H, W)

    def make_running_batch(signal_hat: torch.Tensor, batch_noise: torch.Tensor) -> torch.Tensor:
        # X_running = [s_hat + n_i, d - s_hat]
        nb = int(batch_noise.shape[0])
        term1 = signal_hat[None, :, :] + batch_noise  # (Nb, H, W)
        term2 = (data_t - signal_hat)[None, :, :].expand(nb, -1, -1)  # (Nb, H, W)
        return torch.stack([term1, term2], dim=1)  # (Nb, 2, H, W)

    # -----------------
    # Optimization (LBFGS), following CompSep_notebook.ipynb logic:
    #   Stage 1: unconstrained (no positivity)
    #   Stage 2: positivity via s_hat = exp(u), initialized from stage 1
    # -----------------
    n_batch = int(os.environ.get("COMPSEP_BATCH", "25"))
    outer_steps = int(os.environ.get("COMPSEP_OUTER_ITERS", "10"))
    lbfgs_max_iter = int(os.environ.get("COMPSEP_LBFGS_MAX_ITER", "100"))
    microbatch_env = (os.environ.get("COMPSEP_MICROBATCH") or "").strip()
    microbatch = int(microbatch_env) if microbatch_env else n_batch
    microbatch = max(1, microbatch)
    _msg = (
        f"Batch size: {n_batch} | Microbatch: {microbatch} | Outer steps: {outer_steps} | "
        f"LBFGS max_iter: {lbfgs_max_iter}"
    )
    if LAMBDA_POS != 0.0:
        _msg += f" | Lambda_pos: {LAMBDA_POS}"
    if LAMBDA_NOISE != 0.0:
        _msg += f" | Lambda_noise: {LAMBDA_NOISE} (m=step+1" + (
            f", max={NOISE_INT_M_MAX}" if NOISE_INT_M_MAX is not None else ""
        ) + ")"
    print(_msg)

    # ---- Stage 1 (lin) ----
    running_signal = torch.from_numpy(np.copy(data)).to(device, dtype=torch_dtype)
    running_signal.requires_grad_()

    optimizer = torch.optim.LBFGS(
        [running_signal],
        lr=1,
        max_iter=lbfgs_max_iter,
        tolerance_grad=1e-17,
        tolerance_change=1e-17,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_calls_lin: list[float] = []

    for step in range(outer_steps):
        idx = rng.choice(n_noise, size=min(n_batch, n_noise), replace=False)
        idx = np.asarray(idx, dtype=np.int64)
        m_noise = int(step + 1)
        if NOISE_INT_M_MAX is not None:
            m_noise = min(m_noise, int(NOISE_INT_M_MAX))

        # Precompute target statistics mean over the batch, using microbatches to limit GPU memory.
        with torch.no_grad():
            n_total = int(idx.shape[0])
            if microbatch >= n_total:
                batch_noise = torch.from_numpy(noise_samples[idx].astype(np.float64, copy=False)).to(
                    device, dtype=torch_dtype
                )
                target_batch = make_target_batch(batch_noise)
                target_dc = DataClass(target_batch, pbc=True)
                target_flat = stats_flat(target_dc)
            else:
                target_flat = None
                for start in range(0, n_total, microbatch):
                    sub = idx[start : start + microbatch]
                    weight = float(sub.shape[0]) / float(n_total)
                    batch_noise = torch.from_numpy(
                        noise_samples[sub].astype(np.float64, copy=False)
                    ).to(device, dtype=torch_dtype)
                    target_batch = make_target_batch(batch_noise)
                    target_dc = DataClass(target_batch, pbc=True)
                    flat = stats_flat(target_dc)
                    target_flat = (
                        flat.mul(weight) if target_flat is None else target_flat.add(flat, alpha=weight)
                    )

        def closure():
            optimizer.zero_grad()
            signal_hat = running_signal
            if nyquist_taper_mask is not None:
                signal_hat = _apply_nyquist_taper_torch(signal_hat, nyquist_taper_mask)
            # Compute running statistics mean over the batch using microbatches (same weighting as target_flat).
            n_total = int(idx.shape[0])
            if microbatch >= n_total:
                batch_noise = torch.from_numpy(noise_samples[idx].astype(np.float64, copy=False)).to(
                    device, dtype=torch_dtype
                )
                running_batch = make_running_batch(signal_hat, batch_noise)
                running_dc = DataClass(running_batch, pbc=True)
                running_flat = stats_flat(running_dc)
            else:
                running_flat = None
                for start in range(0, n_total, microbatch):
                    sub = idx[start : start + microbatch]
                    weight = float(sub.shape[0]) / float(n_total)
                    batch_noise = torch.from_numpy(
                        noise_samples[sub].astype(np.float64, copy=False)
                    ).to(device, dtype=torch_dtype)
                    running_batch = make_running_batch(signal_hat, batch_noise)
                    running_dc = DataClass(running_batch, pbc=True)
                    flat = stats_flat(running_dc)
                    running_flat = (
                        flat.mul(weight)
                        if running_flat is None
                        else running_flat.add(flat, alpha=weight)
                    )
            if running_flat.numel() != target_flat.numel():
                raise RuntimeError(
                    f"Flattened statistic length mismatch: running={running_flat.numel()} target={target_flat.numel()}. "
                    "This should not happen when using the same operator and compute_cross_matrix."
                )
            diff = running_flat - target_flat
            mask = finite_mask(diff)
            diff = torch.where(mask, diff, torch.zeros_like(diff))
            loss = squared_l2(diff)
            if LAMBDA_POS != 0.0:
                loss = loss + LAMBDA_POS * torch.mean(
                    (signal_hat[:, :] - torch.abs(signal_hat[:, :])) ** 2
                )
            if LAMBDA_NOISE != 0.0:
                noise_hat = data_t - signal_hat
                # Integer-attractor with exponent m = (outer step index + 1):
                #   mean( 1 - |sin(pi * (noise_hat + 0.5))|^m ) = mean( 1 - |cos(pi * noise_hat)|^m )
                c = torch.cos(np.pi * noise_hat)
                c_mag = torch.sqrt(c.square() + float(NOISE_INT_EPS))
                loss = loss + LAMBDA_NOISE * torch.mean(1.0 - c_mag.pow(m_noise))
            loss.backward()
            loss_value = float(loss.detach().cpu())
            loss_calls_lin.append(loss_value)
            print(f"[lin] Outer step {step+1}/{outer_steps} | loss={loss_value:.4e}")
            return loss

        optimizer.step(closure)

    recovered_lin = running_signal.detach()

    # -----------------
    # Save recovered signal and loss curve
    # -----------------
    results_dir = ROOT / "XRAY_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _slug(s: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in s)

    def _fmt_float_tag(x: float) -> str:
        s = f"{float(x):g}"
        return s.replace("-", "m").replace(".", "p")

    dataclass_tag = _slug(dataclass_kind)
    wtype_tag = _slug(getattr(st_op, "WType", wtype))
    lpos_tag = _fmt_float_tag(LAMBDA_POS)
    lnoise_tag = _fmt_float_tag(LAMBDA_NOISE)
    batch_tag = f"batch{int(n_batch)}"
    micro_tag = f"micro{int(microbatch)}"
    noise_m_tag = (
        "mstep1" + (f"_mmax{int(NOISE_INT_M_MAX)}" if NOISE_INT_M_MAX is not None else "")
    )
    opt_nyq_tag = (
        f"optNyq{int(APPLY_NYQUIST_TAPER_DURING_OPT)}"
        + (f"a{_fmt_float_tag(NYQUIST_TAPER_FRAC_DURING_OPT)}" if APPLY_NYQUIST_TAPER_DURING_OPT else "")
    )
    post_nyq_tag = (
        f"postNyq{int(APPLY_NYQUIST_FILTER)}"
        + (f"c{_fmt_float_tag(NYQUIST_CUTOFF_FRAC)}" if APPLY_NYQUIST_FILTER else "")
    )

    base_name = (
        f"recovered_signal_XRays_size{MAP_SIZE}_dc{dataclass_tag}_wtype{wtype_tag}"
        f"_lpos{lpos_tag}_lnoise{lnoise_tag}_{noise_m_tag}"
        f"_{batch_tag}_{micro_tag}_{opt_nyq_tag}_{post_nyq_tag}"
    )
    out_path = results_dir / f"{base_name}.npy"
    recovered = recovered_lin.detach().cpu().numpy()
    if APPLY_NYQUIST_FILTER:
        out_path_unf = results_dir / f"{base_name}_unfiltered.npy"
        np.save(out_path_unf, recovered)
        print(f"Saved recovered signal (unfiltered) to: {out_path_unf}")

        lowpass = _make_radial_lowpass_np(H, W, frac=NYQUIST_CUTOFF_FRAC)
        recovered_filt = _filter_radial_np(recovered, lowpass)
        np.save(out_path, recovered_filt)
        print(f"Saved recovered signal (low-pass, cutoff={NYQUIST_CUTOFF_FRAC:.2f} Nyq) to: {out_path}")

        _delta = recovered - recovered_filt
        print("Nyquist filter RMS:", float(np.std(_delta)))
        print("Nyquist filter max|delta|:", float(np.max(np.abs(_delta))))
    else:
        np.save(out_path, recovered)
        print(f"Saved recovered signal to: {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(loss_calls_lin, linewidth=1, label="lin")
    ax.set_yscale("log")
    ax.set_xlabel("LBFGS closure call index")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs time (closure calls)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = results_dir / f"loss_curve_XRays_{base_name}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved loss figure to: {fig_path}")


if __name__ == "__main__":
    main()
