#!/usr/bin/env python3
"""Plot bump-steerable wavelet convolution magnitudes for Planck patches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch
from utils import _center_crop, _configure_backend_defaults, _downsample_by_four


DEFAULT_SIGNAL_ROOT = Path(
    "/Users/tsouros/Desktop/Projects/Planck data/BK_CMB_S4_north_patch_v4/signal"
)


def _find_channel_file(patch_dir: Path, patch: int, channel: str) -> Path:
    matches = sorted(patch_dir.glob(f"patch_{patch}_{channel}_*.npy"))
    if not matches:
        raise FileNotFoundError(f"No file found for {channel} in {patch_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file for {channel} in {patch_dir}, found {len(matches)}")
    return matches[0]


def _load_compsep_map(path: Path, map_size: int | None) -> np.ndarray:
    image = _downsample_by_four(np.load(path).astype(np.float64))
    if map_size is not None:
        image = _center_crop(image, out_hw=(map_size, map_size))
    return image


def _load_patch_file_maps(path: Path, map_size: int | None) -> dict[str, np.ndarray]:
    patch = np.load(path).astype(np.float64)
    if patch.ndim != 3 or patch.shape[0] != 2:
        raise ValueError(f"Expected patch file shape (2, H, W), got {patch.shape} from {path}")

    maps = {"Q353": patch[0], "U353": patch[1]}
    if map_size is not None:
        maps = {
            channel: _center_crop(image, out_hw=(map_size, map_size))
            for channel, image in maps.items()
        }
    return maps


def _first_layer_moduli(
    image: np.ndarray,
    *,
    j: int,
    l: int,
    wtype: str,
    pbc: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[np.ndarray], object]:
    _configure_backend_defaults(device=device, dtype=dtype)

    image_t = torch.from_numpy(np.copy(image)).to(device=device, dtype=dtype)
    data = STL_2D_FFT_Torch(image_t, pbc=pbc)
    wavelet_op = data.get_wavelet_op(J=j, L=l, WType=wtype)

    moduli: list[np.ndarray] = []
    working = data.copy(empty=False)
    with torch.no_grad():
        for scale in range(j):
            convolved = wavelet_op.apply(
                working,
                j=scale,
                target_fourier_status=False,
            )
            moduli.append(convolved.array.abs().detach().cpu().numpy())
            if scale < j - 1:
                wavelet_op.downsample(
                    data=working,
                    dg_out=wavelet_op.j_to_dg[scale + 1],
                    inplace=True,
                )
    return moduli, wavelet_op


def _second_layer_from_mother(
    mother: np.ndarray,
    wavelet_op,
    *,
    start_scale: int,
    device: torch.device,
    dtype: torch.dtype,
    pbc: bool,
) -> list[np.ndarray]:
    mother_t = torch.from_numpy(np.copy(mother)).to(device=device, dtype=dtype)
    data = STL_2D_FFT_Torch(
        mother_t,
        dg=wavelet_op.j_to_dg[start_scale],
        N0=wavelet_op.N0,
        pbc=pbc,
        conv_history=[start_scale],
    )

    moduli: list[np.ndarray] = []
    working = data.copy(empty=False)
    with torch.no_grad():
        for scale in range(start_scale, wavelet_op.J):
            convolved = wavelet_op.apply(
                working,
                j=scale,
                target_fourier_status=False,
            )
            moduli.append(convolved.array.abs().detach().cpu().numpy())
            if scale < wavelet_op.J - 1:
                wavelet_op.downsample(
                    data=working,
                    dg_out=wavelet_op.j_to_dg[scale + 1],
                    inplace=True,
                )
    return moduli


def _positive_vmax(arrays: list[np.ndarray]) -> float:
    values = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        if finite.size:
            values.append(np.percentile(finite, 99.5))
    if not values:
        return 1.0
    vmax = float(max(values))
    return vmax if vmax > 0 else 1.0


def _plot_channel(
    image: np.ndarray,
    moduli: list[np.ndarray],
    *,
    patch: int,
    channel: str,
    layer_label: str,
    operation_label: str,
    mother_label: str,
    output_path: Path,
) -> None:
    n_scales = len(moduli)
    n_orientations = moduli[0].shape[0]

    fig = plt.figure(figsize=(4.2 * n_orientations, 3.0 * (n_scales + 1)))
    grid = fig.add_gridspec(n_scales + 1, n_orientations, height_ratios=[1.25] + [1] * n_scales)

    original_ax = fig.add_subplot(grid[0, :])
    original_limit = float(np.percentile(np.abs(image[np.isfinite(image)]), 99.5))
    if original_limit <= 0:
        original_limit = 1.0
    original_ax.imshow(image, cmap="coolwarm", origin="lower", vmin=-original_limit, vmax=original_limit)
    original_ax.set_xticks([])
    original_ax.set_yticks([])
    original_ax.set_ylabel(f"{channel}\n{mother_label}", rotation=0, ha="right", va="center")

    vmax = _positive_vmax(moduli)
    for scale, scale_moduli in enumerate(moduli):
        for orientation in range(n_orientations):
            ax = fig.add_subplot(grid[scale + 1, orientation])
            ax.imshow(scale_moduli[orientation], cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if scale == 0:
                ax.set_xlabel(f"orientation {orientation}")
                ax.xaxis.set_label_position("top")
            if orientation == 0:
                ax.set_ylabel(f"scale {scale}", rotation=0, ha="right", va="center")

    fig.suptitle(f"Patch {patch} | {layer_label} | {operation_label}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Q/U 353 GHz bump-steerable wavelet convolution magnitudes for a Planck patch."
    )
    parser.add_argument("--patch", type=int, default=187, help="Patch number to load.")
    parser.add_argument(
        "--patch-file",
        type=Path,
        default=None,
        help="Optional .npy file with shape (2, H, W), ordered as Q then U.",
    )
    parser.add_argument(
        "--signal-root",
        type=Path,
        default=DEFAULT_SIGNAL_ROOT,
        help="Directory containing patch_<N> signal directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "conv_test",
        help="Directory where figures are saved.",
    )
    parser.add_argument("--map-size", type=int, default=384, help="Center-crop size after 2x2 downsampling.")
    parser.add_argument("--j", type=int, default=7, help="Number of wavelet scales.")
    parser.add_argument("--l", type=int, default=4, help="Number of wavelet orientations.")
    parser.add_argument("--second-layer-mother-scale", type=int, default=0, help="First-layer mother scale for second-layer plots.")
    parser.add_argument("--second-layer-mother-orientation", type=int, default=1, help="First-layer mother orientation for second-layer plots.")
    parser.add_argument("--wtype", default="Bump-Steerable", help="Wavelet type.")
    parser.add_argument("--pbc", action="store_true", help="Use periodic boundary conditions.")
    parser.add_argument("--device", default="cpu", help="Torch device to use.")
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Torch dtype used for the convolutions.",
    )
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_file is None:
        patch_dir = args.signal_root / f"patch_{args.patch}"
        if not patch_dir.is_dir():
            raise FileNotFoundError(f"Patch directory does not exist: {patch_dir}")
        images = {
            channel: _load_compsep_map(
                _find_channel_file(patch_dir, args.patch, channel),
                map_size=args.map_size,
            )
            for channel in ("Q353", "U353")
        }
    else:
        images = _load_patch_file_maps(args.patch_file.expanduser(), map_size=args.map_size)

    for channel, image in images.items():
        moduli, wavelet_op = _first_layer_moduli(
            image,
            j=args.j,
            l=args.l,
            wtype=args.wtype,
            pbc=args.pbc,
            device=device,
            dtype=dtype,
        )
        output_path = args.output_dir / f"patch_{args.patch}_{channel}_bump_steerable_convolutions.png"
        _plot_channel(
            image,
            moduli,
            patch=args.patch,
            channel=channel,
            layer_label="Layer 1",
            operation_label=r"$|X \star \psi_{j,\ell}|$",
            mother_label="original",
            output_path=output_path,
        )
        print("Saved:", output_path)

        mother_scale = args.second_layer_mother_scale
        mother_orientation = args.second_layer_mother_orientation
        if not 0 <= mother_scale < args.j:
            raise ValueError(f"second-layer mother scale must be in [0, {args.j - 1}]")
        if not 0 <= mother_orientation < args.l:
            raise ValueError(f"second-layer mother orientation must be in [0, {args.l - 1}]")

        mother = moduli[mother_scale][mother_orientation]
        second_moduli = _second_layer_from_mother(
            mother,
            wavelet_op,
            start_scale=mother_scale,
            device=device,
            dtype=dtype,
            pbc=args.pbc,
        )
        second_output_path = (
            args.output_dir
            / f"patch_{args.patch}_{channel}_bump_steerable_second_layer_from_j{mother_scale}_l{mother_orientation}.png"
        )
        _plot_channel(
            mother,
            second_moduli,
            patch=args.patch,
            channel=channel,
            layer_label=f"Layer 2 from scale {mother_scale}, orientation {mother_orientation}",
            operation_label=rf"$||X \star \psi_{{{mother_scale},{mother_orientation}}}| \star \psi_{{j,\ell}}|$",
            mother_label=f"scale {mother_scale}\norientation {mother_orientation}",
            output_path=second_output_path,
        )
        print("Saved:", second_output_path)


if __name__ == "__main__":
    main()
