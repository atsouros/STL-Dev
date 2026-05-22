#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_INPUT_PATH = Path(
    "/pscratch/sd/s/shamikg/polarized_dust_STGNILC/output/"
    "NPIPE_PR4_QU_tiles_tilenside256_margin64_hpxnside1024.npy"
)
DEFAULT_OUTPUT_DIR = Path("/pscratch/sd/a/atsouros")
EXPECTED_SHAPE = (192, 2, 378, 378)
DEFAULT_PATCH_INDEX = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one patch from a consolidated Q/U .npy tile array."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input .npy file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=DEFAULT_PATCH_INDEX,
        help=f"Patch index to extract. Default: {DEFAULT_PATCH_INDEX}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the extracted patch. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output .npy path. Defaults to <output-dir>/patch_<patch>.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_dir = args.output_dir.expanduser()
    output_path = args.output.expanduser() if args.output is not None else output_dir / f"patch_{args.patch}.npy"

    tiles = np.load(input_path, mmap_mode="r")
    if tiles.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected input shape {EXPECTED_SHAPE}, got {tiles.shape} from {input_path}")
    if args.patch < 0 or args.patch >= tiles.shape[0]:
        raise IndexError(f"Patch index {args.patch} is outside available range 0-{tiles.shape[0] - 1}")

    patch = np.asarray(tiles[args.patch], dtype=tiles.dtype)
    if patch.shape != EXPECTED_SHAPE[1:]:
        raise RuntimeError(f"Expected patch shape {EXPECTED_SHAPE[1:]}, got {patch.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, patch)
    print(f"Saved patch {args.patch} with shape {patch.shape} to {output_path}")


if __name__ == "__main__":
    main()
