#!/usr/bin/env python3
"""STL component separation of one image corrupted by empirical CIB noise.

This follows ``compsep_simple_pedagogical.ipynb``.  For a candidate signal u
and independent CIB maps n_i, it matches the batch-averaged STL statistics of

    target:  [d,       n_i]
    running: [u + n_i, d - u].

Only the recovered map initializes the later Langevin run; this program does
not perform posterior sampling.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import STL_main.torch_backend as bk  # noqa: E402
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass  # noqa: E402


DEFAULT_DATA = REPO_ROOT / "data" / "test" / "mock_data.py"
DEFAULT_CIB_BANK = (
    REPO_ROOT / "data" / "test" / "100_Herschel_Lockman_250m_tiles_3400x256x256.npy"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "component_separation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--cib-bank", type=Path, default=DEFAULT_CIB_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=240830)
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--n-noise",
        type=int,
        default=0,
        help="Number of leading CIB maps available to sampling; 0 uses the full bank.",
    )
    parser.add_argument("--optimizer-lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=100)
    parser.add_argument("--wtype", default="Bump-Steerable")
    parser.add_argument("--pbc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compute-ps", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--apply-nyquist-after", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def configure_backend(device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = (
        torch.complex64 if dtype == torch.float32 else torch.complex128
    )


def select_image(array: Any, index: int, label: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim == 3:
        if not 0 <= index < image.shape[0]:
            raise IndexError(f"{label} index {index} outside stack of size {image.shape[0]}")
        image = image[index]
    if image.ndim != 2:
        raise ValueError(f"Expected {label} shape (H,W) or (N,H,W), got {image.shape}")
    image = np.asarray(image, dtype=np.float64)
    if not np.isfinite(image).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return image


def load_observed(path: Path, index: int) -> np.ndarray:
    """Load a NumPy image or a Python file defining d/data/mock_data."""
    if path.suffix.lower() == ".npy":
        return select_image(np.load(path), index, "observed data")
    if path.suffix.lower() != ".py":
        raise ValueError("--data must be a .npy file or a .py file defining an image array")

    namespace = runpy.run_path(str(path))
    for name in ("mock_data", "data", "d", "observed_data"):
        value = namespace.get(name)
        if isinstance(value, (np.ndarray, list, tuple)):
            print(f"loaded observed array from {path}:{name}", flush=True)
            return select_image(value, index, f"{path.name}:{name}")
    candidates = [
        (name, value)
        for name, value in namespace.items()
        if isinstance(value, np.ndarray) and value.ndim in (2, 3)
    ]
    if len(candidates) == 1:
        name, value = candidates[0]
        print(f"loaded sole image array from {path}:{name}", flush=True)
        return select_image(value, index, f"{path.name}:{name}")
    names = [name for name, _value in candidates]
    raise KeyError(
        f"Could not identify the observed image in {path}. Define one of "
        f"mock_data, data, d, observed_data; candidate arrays were {names}."
    )


def open_cib_bank(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    bank = np.load(path, mmap_mode="r")
    if bank.ndim == 2:
        bank = bank[None]
    if bank.ndim != 3 or tuple(bank.shape[-2:]) != expected_shape:
        raise ValueError(
            f"Expected CIB bank shape (N,{expected_shape[0]},{expected_shape[1]}), "
            f"got {bank.shape}"
        )
    if bank.shape[0] < 1:
        raise ValueError("CIB bank is empty")
    return bank


def nyquist_mask(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = shape
    ky = height * torch.fft.fftfreq(height, device=device)
    kx = width * torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    return torch.sqrt(xx.square() + yy.square()) <= min(height, width) / 2


def apply_nyquist_filter(image: torch.Tensor) -> torch.Tensor:
    mask = nyquist_mask(tuple(image.shape[-2:]), image.device)
    coefficients = torch.fft.fft2(image, norm="ortho").masked_fill(~mask, 0)
    return torch.fft.ifft2(coefficients, norm="ortho").real


def add_image(ax: Any, image: np.ndarray, title: str) -> None:
    limit = float(np.percentile(np.abs(image - np.median(image)), 99))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    center = float(np.median(image))
    shown = ax.imshow(
        image,
        origin="lower",
        cmap="RdBu_r",
        vmin=center - limit,
        vmax=center + limit,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    for path, label in ((args.data, "observed data"), (args.cib_bank, "CIB bank")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.n_iterations <= 0 or args.batch_size <= 0 or args.n_noise < 0:
        raise ValueError("Require n_iterations > 0, batch_size > 0, and n_noise >= 0")
    if args.optimizer_lr <= 0 or args.history_size <= 0 or args.print_every <= 0:
        raise ValueError("Optimizer lr, history size, and print frequency must be positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
        torch.cuda.set_device(device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    configure_backend(device, dtype)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    observed = load_observed(args.data, args.data_index)
    shape = tuple(observed.shape)
    cib_bank = open_cib_bank(args.cib_bank, shape)
    n_available = cib_bank.shape[0] if args.n_noise == 0 else min(args.n_noise, cib_bank.shape[0])
    batch_size = min(args.batch_size, n_available)
    reference_index = int(rng.integers(0, n_available))
    cib_example = np.asarray(cib_bank[reference_index], dtype=np.float64)

    print(f"device={device} dtype={args.dtype}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    print(f"data={args.data} shape={shape}", flush=True)
    print(
        f"cib_bank={args.cib_bank} available={cib_bank.shape[0]} "
        f"used={n_available} batch_size={batch_size}",
        flush=True,
    )
    print(
        f"iterations={args.n_iterations} lr={args.optimizer_lr} "
        f"pbc={args.pbc} compute_ps={args.compute_ps}",
        flush=True,
    )

    cross_matrix = torch.eye(2, dtype=torch.bool, device=device)
    reference = torch.from_numpy(np.stack([observed, cib_example]))
    reference = reference.to(device=device, dtype=dtype)
    reference_dc = DataClass(reference[None], pbc=args.pbc)
    st_op = reference_dc.get_ST_op(compute_PS=args.compute_ps)
    try:
        st_op.wavelet_op = reference_dc.get_wavelet_op(
            J=st_op.J, L=st_op.L, WType=args.wtype
        )
    except TypeError:
        st_op.wavelet_op = reference_dc.get_wavelet_op(J=st_op.J, L=st_op.L)
    st_op.WType = getattr(st_op.wavelet_op, "WType", args.wtype)
    with torch.no_grad():
        st_op.apply(reference_dc, norm="store_ref", compute_cross_matrix=cross_matrix)

    observed_t = torch.as_tensor(observed, device=device, dtype=dtype)

    def cib_batch(indices: np.ndarray) -> torch.Tensor:
        # The mmap keeps the 3,400-map bank out of GPU memory.  Only this batch moves.
        contiguous = np.array(cib_bank[indices], copy=True)
        return torch.as_tensor(contiguous, device=device, dtype=dtype)

    def statistics(batch: torch.Tensor) -> torch.Tensor:
        dc = DataClass(batch, pbc=args.pbc)
        return st_op.apply(
            dc, norm="load_ref", compute_cross_matrix=cross_matrix
        ).to_flatten(mean_along_batch=True, keepnans=False)

    recovered_t = observed_t.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [recovered_t],
        lr=args.optimizer_lr,
        max_iter=1,
        tolerance_grad=-1,
        tolerance_change=-1,
        history_size=args.history_size,
        line_search_fn=None,
    )
    losses: list[float] = []

    for iteration in range(args.n_iterations):
        indices = rng.choice(n_available, size=batch_size, replace=False)
        noise_t = cib_batch(indices)
        data_batch = observed_t[None].expand(batch_size, -1, -1)
        with torch.no_grad():
            target = statistics(torch.stack([data_batch, noise_t], dim=1))

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            signal_plus_noise = recovered_t[None] + noise_t
            residual = (observed_t - recovered_t)[None].expand(batch_size, -1, -1)
            running = torch.stack([signal_plus_noise, residual], dim=1)
            difference = statistics(running) - target
            loss = difference.abs().square().sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at iteration {iteration + 1}")
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        step = iteration + 1
        if step == 1 or step % args.print_every == 0 or step == args.n_iterations:
            print(f"iteration={step:4d}/{args.n_iterations} loss={loss_value:.6e}", flush=True)

    if args.apply_nyquist_after:
        with torch.no_grad():
            recovered_t.copy_(apply_nyquist_filter(recovered_t))

    recovered = recovered_t.detach().cpu().numpy().astype(np.float64)
    residual = observed - recovered
    losses_array = np.asarray(losses, dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovered_path = args.output_dir / "recovered_signal.npy"
    observed_path = args.output_dir / "observed_data.npy"
    residual_path = args.output_dir / "recovered_cib.npy"
    loss_path = args.output_dir / "loss.npy"
    diagnostic_path = args.output_dir / "diagnostic.png"
    loss_plot_path = args.output_dir / "loss.png"
    summary_path = args.output_dir / "summary.json"
    np.save(recovered_path, recovered)
    np.save(observed_path, observed)
    np.save(residual_path, residual)
    np.save(loss_path, losses_array)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    add_image(axes[0, 0], observed, "observed data d")
    add_image(axes[0, 1], cib_example, f"CIB example (index {reference_index})")
    add_image(axes[1, 0], recovered, "recovered signal s")
    add_image(axes[1, 1], residual, "recovered CIB d - s")
    fig.savefig(diagnostic_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(np.arange(1, losses_array.size + 1), losses_array, lw=1.2)
    if np.all(losses_array > 0):
        ax.set_yscale("log")
    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("STL loss")
    ax.grid(alpha=0.25)
    fig.savefig(loss_plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "method": "stochastic STL component separation with empirical CIB batches",
        "data": str(args.data),
        "cib_bank": str(args.cib_bank),
        "recovered_signal": str(recovered_path),
        "observed_data": str(observed_path),
        "recovered_cib": str(residual_path),
        "diagnostic_plot": str(diagnostic_path),
        "loss_plot": str(loss_plot_path),
        "final_loss": float(losses_array[-1]),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for path in (
        recovered_path,
        observed_path,
        residual_path,
        loss_path,
        diagnostic_path,
        loss_plot_path,
        summary_path,
    ):
        print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
