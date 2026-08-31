#!/usr/bin/env python3
"""Tile Herschel Lockman maps into 256x256 samples for covariance estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert (N,H,W) maps into non-overlapping square tiles."
    )
    parser.add_argument("--input", required=True, help="Input .npy file, shape (N,H,W).")
    parser.add_argument("--output", required=True, help="Output .npy file.")
    parser.add_argument("--metadata", default=None, help="Optional output JSON metadata.")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--tiles-per-map",
        type=int,
        default=34,
        help="Number of tiles kept per map. For 1500 and 256 this should be 34.",
    )
    parser.add_argument(
        "--layout",
        choices=("row-major",),
        default="row-major",
        help="Tile ordering. Currently row-major over the non-overlapping grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".json")

    maps = np.load(input_path, mmap_mode="r")
    if maps.ndim != 3:
        raise ValueError(f"Expected input shape (N,H,W), got {maps.shape}")

    n_maps, height, width = maps.shape
    tile = args.tile_size
    n_y = height // tile
    n_x = width // tile
    available = n_y * n_x
    if args.tiles_per_map > available:
        raise ValueError(
            f"Requested {args.tiles_per_map} tiles per map, but only {available} "
            f"non-overlapping {tile}x{tile} tiles fit in {height}x{width}."
        )

    positions = []
    for iy in range(n_y):
        for ix in range(n_x):
            positions.append((iy * tile, ix * tile))
    positions = positions[: args.tiles_per_map]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_shape = (n_maps * len(positions), tile, tile)
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=maps.dtype,
        shape=out_shape,
    )

    k = 0
    for m in range(n_maps):
        for y0, x0 in positions:
            out[k] = maps[m, y0 : y0 + tile, x0 : x0 + tile]
            k += 1
    out.flush()

    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "input_shape": list(maps.shape),
        "output_shape": list(out_shape),
        "tile_size": tile,
        "tiles_per_map": len(positions),
        "grid_shape": [n_y, n_x],
        "remainder": [height - n_y * tile, width - n_x * tile],
        "positions_yx": positions,
        "layout": args.layout,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {output_path} with shape {out_shape}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
