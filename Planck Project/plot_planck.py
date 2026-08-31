#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import blackmanharris, tukey

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = Path("/pscratch/sd/a/atsouros/STL/planck_results")
DEFAULT_SIGNAL_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/signal")
DEFAULT_NUISANCE_DIR = Path("/pscratch/sd/e/erussie/GNILC+ST/patches/nuisance")
DEFAULT_OUT_DIR = Path("/pscratch/sd/a/atsouros/STL/plots")
DEFAULT_NEW_PROJECTION_PATH = Path(
    "/pscratch/sd/s/shamikg/polarized_dust_STGNILC/output/"
    "NPIPE_PR4_QU_tiles_tilenside256_margin64_hpxnside1024.npy"
)

TARGET_SHAPE = (384, 384)
PATCH_SIZE_DEG = 10.0
ELL_BINS = 60
PK_WINDOW = "blackmanharris"
TUKEY_ALPHA = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Planck component-separation result maps and power spectra."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--nuisance-dir", type=Path, default=DEFAULT_NUISANCE_DIR)
    parser.add_argument("--new-projection-path", type=Path, default=DEFAULT_NEW_PROJECTION_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--result-file",
        action="append",
        type=Path,
        default=None,
        help="Exact result .npy file to plot. May be repeated. When set, results-dir scanning is skipped.",
    )
    parser.add_argument("--freq", default="353")
    parser.add_argument(
        "--nuisance-version",
        choices=("v4_10_arcmin", "v2", "all"),
        default="v4_10_arcmin",
        help="Which nuisance map version to use. Default matches the local notebook.",
    )
    parser.add_argument("--map-size", type=int, default=384)
    parser.add_argument("--patch", action="append", default=None, help="Only plot this patch. May be repeated.")
    parser.add_argument("--patch-start", type=int, default=None, help="First patch to plot, inclusive.")
    parser.add_argument("--patch-end", type=int, default=None, help="Last patch to plot, inclusive.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_from_stem(stem: str) -> str:
    match = re.match(r"(?:newproj_)?p(\d+)_", stem)
    if not match:
        raise ValueError(f"Could not parse patch from result stem: {stem}")
    return match.group(1)


def is_new_projection_result(stem: str) -> bool:
    return stem.startswith("newproj_")


def patch_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(patch_from_stem(path.stem)), path.stem)
    except ValueError:
        return (10**9, path.stem)


def patch_number(path: Path) -> int:
    return int(patch_from_stem(path.stem))


def downsample_by_four(image: np.ndarray, sigma: float | None = None) -> np.ndarray:
    image = np.asarray(image)
    h, w = image.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"Image dimensions must be even for 2x2 downsampling, got {image.shape}.")
    if sigma is not None:
        if image.ndim == 2:
            image = gaussian_filter(image, sigma=float(sigma))
        elif image.ndim == 3:
            image = gaussian_filter(image, sigma=(float(sigma), float(sigma), 0.0))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.ndim == 2:
        return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    if image.ndim == 3:
        return image.reshape(h // 2, 2, w // 2, 2, image.shape[2]).mean(axis=(1, 3))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def center_crop(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    out_h, out_w = out_hw
    h, w = image.shape[:2]
    if out_h > h or out_w > w:
        raise ValueError(f"Cannot crop {image.shape} to {out_hw}.")
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    if image.ndim == 2:
        return image[y0 : y0 + out_h, x0 : x0 + out_w]
    if image.ndim == 3:
        return image[y0 : y0 + out_h, x0 : x0 + out_w, :]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def ensure_map_size(image: np.ndarray, name: str, map_size: int, sigma: float | None = None) -> np.ndarray:
    target_shape = (map_size, map_size)
    image = np.asarray(image)
    if image.shape == target_shape:
        return image
    if image.shape[:2] == (2 * map_size, 2 * map_size):
        image = downsample_by_four(image, sigma=sigma)
    if image.shape != target_shape and image.shape[0] >= map_size and image.shape[1] >= map_size:
        image = center_crop(image, target_shape)
    if image.shape != target_shape:
        raise ValueError(f"{name} has shape {image.shape}; expected {target_shape}.")
    return image


def load_map(path: Path, name: str, map_size: int) -> np.ndarray:
    return ensure_map_size(np.load(path, allow_pickle=False), name, map_size)


def load_new_projection_signal(path: Path, patch: str, map_size: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        patch_index = int(patch)
    except ValueError as exc:
        raise ValueError(f"Patch must be an integer tile index for {path}: {patch!r}") from exc

    patches = np.load(path, mmap_mode="r")
    if patches.ndim != 4 or patches.shape[1] != 2:
        raise ValueError(
            f"Expected new-projection patches to have shape (N, 2, H, W), got {patches.shape}."
        )
    if patch_index < 0 or patch_index >= patches.shape[0]:
        raise IndexError(
            f"Patch index {patch_index} is outside available range 0-{patches.shape[0] - 1} "
            f"for {path}."
        )

    qu = np.asarray(patches[patch_index], dtype=np.float64)
    signal_q = ensure_map_size(qu[0], "new_projection_signal_q", map_size)
    signal_u = ensure_map_size(qu[1], "new_projection_signal_u", map_size)
    return signal_q, signal_u


def select_no_bonus_signal_path(signal_dir: Path, pattern: str, label: str) -> Path:
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


def signal_path(signal_dir: Path, patch: str, pattern: str, label: str) -> Path:
    try:
        return select_no_bonus_signal_path(signal_dir, pattern, label)
    except FileNotFoundError:
        nested_dir = signal_dir / f"patch_{patch}"
        return select_no_bonus_signal_path(nested_dir, pattern, label)


def nuisance_version_suffix(version: str) -> str | None:
    if version == "all":
        return None
    return f"_{version}.npy"


def nuisance_sample_key(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_noise_seed_(\d+)_CMB_res_seed_(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def filter_nuisance_version(paths: list[Path], version: str) -> list[Path]:
    suffix = nuisance_version_suffix(version)
    if suffix is None:
        return paths
    return [path for path in paths if path.name.endswith(suffix)]


def nuisance_paths(nuisance_dir: Path, patch: str, stokes: str, freq: str, version: str) -> list[Path]:
    pattern = f"patch_{patch}_noise_{stokes}{freq}_*.npy"
    paths = filter_nuisance_version(sorted(nuisance_dir.glob(pattern)), version)
    if paths:
        return paths
    nested = nuisance_dir / f"patch_{patch}" / f"Stokes_{stokes}"
    paths = filter_nuisance_version(sorted(nested.glob(pattern)), version)
    if paths:
        return paths
    suffix = nuisance_version_suffix(version)
    version_msg = "" if suffix is None else f" with suffix *{suffix}"
    raise FileNotFoundError(
        f"No nuisance {stokes}{freq} files{version_msg} found for patch {patch}. "
        f"Looked for {nuisance_dir / pattern} and {nested / pattern}."
    )


def align_nuisance_paths(q_paths: list[Path], u_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    q_keys = [nuisance_sample_key(path) for path in q_paths]
    u_keys = [nuisance_sample_key(path) for path in u_paths]
    q_keys_are_unique = len(set(q_keys)) == len(q_keys)
    u_keys_are_unique = len(set(u_keys)) == len(u_keys)
    if any(key is None for key in q_keys + u_keys) or not q_keys_are_unique or not u_keys_are_unique:
        if len(q_paths) != len(u_paths):
            raise ValueError(
                f"Cannot align nuisance Q/U samples by index: Q has {len(q_paths)} files, "
                f"U has {len(u_paths)} files."
            )
        return q_paths, u_paths

    q_by_key = dict(zip(q_keys, q_paths))
    u_by_key = dict(zip(u_keys, u_paths))
    common = sorted(q_by_key.keys() & u_by_key.keys())
    if not common:
        raise ValueError("No paired nuisance Q/U samples found with matching noise and CMB residual seeds.")
    missing_q = sorted(u_by_key.keys() - q_by_key.keys())
    missing_u = sorted(q_by_key.keys() - u_by_key.keys())
    if missing_q or missing_u:
        raise ValueError(
            f"Cannot pair all nuisance Q/U samples: missing Q keys={missing_q[:5]}, "
            f"missing U keys={missing_u[:5]}."
        )
    return [q_by_key[key] for key in common], [u_by_key[key] for key in common]


def load_nuisance_stack(paths: list[Path], stokes: str, map_size: int) -> np.ndarray:
    return np.stack(
        [load_map(path, f"nuisance_{stokes}[{i}]", map_size) for i, path in enumerate(paths)],
        axis=0,
    )


def build_window(npix: int) -> tuple[np.ndarray, float]:
    if PK_WINDOW in {"blackmanharris", "bh"}:
        window_1d = blackmanharris(npix, sym=False)
    elif PK_WINDOW == "tukey":
        window_1d = tukey(npix, TUKEY_ALPHA)
    else:
        raise ValueError(f"Unknown PK_WINDOW={PK_WINDOW!r}")
    window_2d = np.outer(window_1d, window_1d)
    return window_2d, float(np.mean(window_2d**2))


def power_spectrum(image: np.ndarray, window_2d: np.ndarray, window_norm: float) -> tuple[np.ndarray, np.ndarray]:
    npix = image.shape[0]
    patch_size_rad = np.radians(PATCH_SIZE_DEG)
    delta_theta = patch_size_rad / npix

    image = image - np.mean(image)
    fft_image = np.fft.fft2(image * window_2d)
    ps2d = np.real(fft_image * np.conj(fft_image)) * (delta_theta**2) / (npix**2)

    freq = np.fft.fftfreq(npix, d=delta_theta)
    kx, ky = np.meshgrid(freq, freq)
    ell = (2 * np.pi * np.sqrt(kx**2 + ky**2)).ravel()
    ps = ps2d.ravel()

    bins = np.linspace(0.0, np.max(ell), ELL_BINS + 1)
    pk, _, _ = stats.binned_statistic(ell, ps, bins=bins, statistic="mean")
    ell_centers = 0.5 * (bins[1:] + bins[:-1])
    return ell_centers, pk / window_norm


def power_spectrum_stack(images: np.ndarray, window_2d: np.ndarray, window_norm: float) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    ell = None
    for image in images:
        ell, pk = power_spectrum(image, window_2d, window_norm)
        spectra.append(pk)
    if ell is None:
        raise ValueError("Cannot compute spectra for an empty stack.")
    return ell, np.stack(spectra, axis=0)


def save_maps_figure(
    out_path: Path,
    signal_q: np.ndarray,
    signal_u: np.ndarray,
    recovered_q: np.ndarray,
    recovered_u: np.ndarray,
    nuisance_q_example: np.ndarray,
    nuisance_u_example: np.ndarray,
    patch: str,
) -> None:
    residual_q = signal_q - recovered_q
    residual_u = signal_u - recovered_u
    recovered_plus_nuisance_q = recovered_q + nuisance_q_example
    recovered_plus_nuisance_u = recovered_u + nuisance_u_example

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    maps = [
        (signal_q, "Q signal"),
        (recovered_q, "Q recovered"),
        (residual_q, "Q signal - recovered"),
        (recovered_plus_nuisance_q, "Q recovered + nuisance"),
        (signal_u, "U signal"),
        (recovered_u, "U recovered"),
        (residual_u, "U signal - recovered"),
        (recovered_plus_nuisance_u, "U recovered + nuisance"),
    ]

    plot_arrays = [arr - np.mean(arr) for arr, _ in maps]
    scale = np.nanpercentile(np.abs(np.concatenate([arr.ravel() for arr in plot_arrays])), 99)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    im = None
    for ax, arr, (_, title) in zip(axes.ravel(), plot_arrays, maps):
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Patch {patch}", fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_power_spectrum_figure(
    out_path: Path,
    stokes: str,
    ell: np.ndarray,
    signal_pk: np.ndarray,
    recovered_pk: np.ndarray,
    residual_pk: np.ndarray,
    nuisance_pks: np.ndarray,
    recovered_plus_nuisance_pks: np.ndarray,
    patch: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    base = np.isfinite(ell) & (ell > 0)

    def plot_line(pk: np.ndarray, label: str, **kwargs) -> None:
        valid = base & np.isfinite(pk) & (pk > 0)
        ax.plot(ell[valid], pk[valid], label=label, **kwargs)

    plot_line(signal_pk, "Signal", linewidth=2.2, color="C0")
    plot_line(recovered_pk, "Recovered", linewidth=2.0, linestyle="--", color="C1")
    plot_line(residual_pk, "Signal - recovered", linewidth=2.0, linestyle="--", color="C3")

    for pks, label, color in [
        (nuisance_pks, "Nuisance envelope", "C4"),
        (recovered_plus_nuisance_pks, "Recovered + nuisance envelope", "C2"),
    ]:
        lo = np.nanmin(pks, axis=0)
        hi = np.nanmax(pks, axis=0)
        valid = base & np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
        ax.fill_between(ell[valid], lo[valid], hi[valid], alpha=0.22, color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin=1e-15)
    ax.set_xlabel(r"$\ell$ [rad$^{-1}$]")
    ax.set_ylabel(r"$P(\ell)$")
    ax.set_title(f"Patch {patch}: {stokes} power spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def process_result(path: Path, args: argparse.Namespace) -> None:
    stem = path.stem
    patch = patch_from_stem(stem)
    out_dir = args.out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    maps_path = out_dir / f"{stem}_maps.png"
    ps_q_path = out_dir / f"{stem}_power_spectrum_Q.png"
    ps_u_path = out_dir / f"{stem}_power_spectrum_U.png"
    if not args.overwrite and maps_path.exists() and ps_q_path.exists() and ps_u_path.exists():
        print(f"[patch {patch}] skipping existing plots for {stem}")
        return

    recovered = np.load(path, allow_pickle=False)
    if recovered.shape[0] != 2:
        raise ValueError(f"{path} has shape {recovered.shape}; expected first axis to be Q/U.")
    recovered_q = ensure_map_size(recovered[0], "recovered_q", args.map_size)
    recovered_u = ensure_map_size(recovered[1], "recovered_u", args.map_size)

    if is_new_projection_result(stem):
        signal_q, signal_u = load_new_projection_signal(
            args.new_projection_path,
            patch,
            args.map_size,
        )
        print(f"[patch {patch}] signal Q/U: {args.new_projection_path}")
    else:
        signal_q_path = signal_path(args.signal_dir, patch, f"patch_{patch}_Q{args.freq}_*.npy", f"Q{args.freq}")
        signal_u_path = signal_path(args.signal_dir, patch, f"patch_{patch}_U{args.freq}_*.npy", f"U{args.freq}")
        signal_q = load_map(signal_q_path, "signal_q", args.map_size)
        signal_u = load_map(signal_u_path, "signal_u", args.map_size)
        print(f"[patch {patch}] signal Q={signal_q_path.name}; U={signal_u_path.name}")

    nuisance_q_paths = nuisance_paths(args.nuisance_dir, patch, "Q", args.freq, args.nuisance_version)
    nuisance_u_paths = nuisance_paths(args.nuisance_dir, patch, "U", args.freq, args.nuisance_version)
    nuisance_q_paths, nuisance_u_paths = align_nuisance_paths(nuisance_q_paths, nuisance_u_paths)
    print(
        f"[patch {patch}] nuisance {args.freq} {args.nuisance_version}: "
        f"{len(nuisance_q_paths)} paired samples; first Q={nuisance_q_paths[0].name}; "
        f"first U={nuisance_u_paths[0].name}"
    )
    nuisance_q = load_nuisance_stack(nuisance_q_paths, "Q", args.map_size)
    nuisance_u = load_nuisance_stack(nuisance_u_paths, "U", args.map_size)

    save_maps_figure(
        maps_path,
        signal_q,
        signal_u,
        recovered_q,
        recovered_u,
        nuisance_q[0],
        nuisance_u[0],
        patch,
    )

    residual_q = signal_q - recovered_q
    residual_u = signal_u - recovered_u
    window_2d, window_norm = build_window(args.map_size)

    ell, signal_q_pk = power_spectrum(signal_q, window_2d, window_norm)
    _, recovered_q_pk = power_spectrum(recovered_q, window_2d, window_norm)
    _, residual_q_pk = power_spectrum(residual_q, window_2d, window_norm)
    _, nuisance_q_pks = power_spectrum_stack(nuisance_q, window_2d, window_norm)
    _, recovered_plus_nuisance_q_pks = power_spectrum_stack(
        recovered_q[None, :, :] + nuisance_q,
        window_2d,
        window_norm,
    )

    _, signal_u_pk = power_spectrum(signal_u, window_2d, window_norm)
    _, recovered_u_pk = power_spectrum(recovered_u, window_2d, window_norm)
    _, residual_u_pk = power_spectrum(residual_u, window_2d, window_norm)
    _, nuisance_u_pks = power_spectrum_stack(nuisance_u, window_2d, window_norm)
    _, recovered_plus_nuisance_u_pks = power_spectrum_stack(
        recovered_u[None, :, :] + nuisance_u,
        window_2d,
        window_norm,
    )

    save_power_spectrum_figure(
        ps_q_path,
        "Q",
        ell,
        signal_q_pk,
        recovered_q_pk,
        residual_q_pk,
        nuisance_q_pks,
        recovered_plus_nuisance_q_pks,
        patch,
    )
    save_power_spectrum_figure(
        ps_u_path,
        "U",
        ell,
        signal_u_pk,
        recovered_u_pk,
        residual_u_pk,
        nuisance_u_pks,
        recovered_plus_nuisance_u_pks,
        patch,
    )

    print(f"[patch {patch}] saved {maps_path.name}, {ps_q_path.name}, {ps_u_path.name}")


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser()
    args.signal_dir = args.signal_dir.expanduser()
    args.nuisance_dir = args.nuisance_dir.expanduser()
    args.new_projection_path = args.new_projection_path.expanduser()
    if args.result_file is not None:
        args.result_file = [path.expanduser() for path in args.result_file]
    if args.out_dir is not None:
        args.out_dir = args.out_dir.expanduser()

    requested_patches = set(args.patch or [])
    if args.result_file:
        result_paths = sorted(args.result_file, key=patch_sort_key)
    else:
        result_paths = sorted(
            [
                *args.results_dir.glob("p*_*.npy"),
                *args.results_dir.glob("newproj_p*_*.npy"),
            ],
            key=patch_sort_key,
        )
    if requested_patches:
        result_paths = [path for path in result_paths if patch_from_stem(path.stem) in requested_patches]
    if args.patch_start is not None:
        result_paths = [path for path in result_paths if patch_number(path) >= args.patch_start]
    if args.patch_end is not None:
        result_paths = [path for path in result_paths if patch_number(path) <= args.patch_end]
    if not result_paths:
        raise FileNotFoundError(f"No result .npy files found in {args.results_dir}.")

    print(f"Results dir: {args.results_dir}")
    print(f"Signal dir: {args.signal_dir}")
    print(f"Nuisance dir: {args.nuisance_dir}")
    print(f"Plotting {len(result_paths)} result files")

    failures: list[tuple[Path, Exception]] = []
    for path in result_paths:
        try:
            process_result(path, args)
        except Exception as exc:
            failures.append((path, exc))
            print(f"[error] {path.name}: {exc}")

    if failures:
        print(f"Completed with {len(failures)} failures.")
        raise SystemExit(1)
    print("All plots completed.")


if __name__ == "__main__":
    main()
