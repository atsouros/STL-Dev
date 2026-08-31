#!/usr/bin/env python3
"""Prepare downgraded I857 signal patches for moment-network training."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_patch_selector(selector):
    patches = []
    for item in str(selector).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            patches.extend(str(p) for p in range(int(start), int(end) + 1))
        else:
            patches.append(str(int(item)))
    return patches


def downsample_by_four(image):
    if image.ndim != 2:
        raise ValueError("Expected 2D image, got shape={}".format(image.shape))
    h, w = image.shape
    if h % 2 or w % 2:
        raise ValueError("Image dimensions must be even, got {}x{}".format(h, w))
    return image.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def center_crop(image, map_size):
    h, w = image.shape
    if map_size > h or map_size > w:
        raise ValueError("Cannot crop {}x{} image to {}".format(h, w, map_size))
    y0 = (h - map_size) // 2
    x0 = (w - map_size) // 2
    return image[y0 : y0 + map_size, x0 : x0 + map_size]


def select_i_file(signal_dir, patch, intensity_freq):
    pattern = "patch_{}_I{}_*".format(patch, intensity_freq)
    candidates = sorted(signal_dir.glob(pattern))
    candidates = [
        path
        for path in candidates
        if path.suffix == ".npy" and re.search(r"_(?:hr|hm)_\d+\.npy$", path.name) is None
    ]
    canonical = [path for path in candidates if path.name.endswith("_v4_10_arcmin.npy")]
    candidates = canonical or candidates
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            "Expected exactly one I{} file for patch {} in {}; got {}".format(
                intensity_freq, patch, signal_dir, names
            )
        )
    return candidates[0]


def process_one(path, out_dir, map_size, dtype, overwrite):
    out_path = out_dir / path.name
    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        return {
            "source": str(path),
            "output": str(out_path),
            "skipped": True,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    raw = np.load(path).astype(np.float64)
    downgraded = downsample_by_four(raw)
    cropped = center_crop(downgraded, map_size)
    saved = cropped.astype(dtype, copy=False)
    np.save(out_path, saved)
    return {
        "source": str(path),
        "output": str(out_path),
        "skipped": False,
        "raw_shape": list(raw.shape),
        "shape": list(saved.shape),
        "dtype": str(saved.dtype),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Copy only I857 signal patches after downgrading/cropping them.")
    parser.add_argument("--signal-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--patches", default="0-191")
    parser.add_argument("--expected-patch-count", type=int, default=0)
    parser.add_argument("--intensity-freq", type=int, default=857)
    parser.add_argument("--map-size", type=int, default=384)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.signal_dir = args.signal_dir.expanduser()
    args.out_dir = args.out_dir.expanduser()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patches = parse_patch_selector(args.patches)
    if args.expected_patch_count > 0 and len(patches) != args.expected_patch_count:
        raise RuntimeError(
            "Expected {} patches from --patches={}, got {}".format(
                args.expected_patch_count, args.patches, len(patches)
            )
        )

    records = []
    for index, patch in enumerate(patches, start=1):
        src = select_i_file(args.signal_dir, patch, args.intensity_freq)
        record = process_one(src, args.out_dir, args.map_size, np.dtype(args.dtype), args.overwrite)
        records.append(record)
        print(
            "{:04d}/{:04d} patch={} {} -> shape={}{}".format(
                index,
                len(patches),
                patch,
                src.name,
                tuple(record["shape"]),
                " skipped" if record["skipped"] else "",
            ),
            flush=True,
        )

    manifest = {
        "signal_dir": str(args.signal_dir),
        "out_dir": str(args.out_dir),
        "patches": patches,
        "intensity_freq": args.intensity_freq,
        "map_size": args.map_size,
        "dtype": args.dtype,
        "n_files": len(records),
        "files": records,
    }
    manifest_path = args.out_dir / "signal_I_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print("Wrote {}".format(manifest_path), flush=True)


if __name__ == "__main__":
    main()
