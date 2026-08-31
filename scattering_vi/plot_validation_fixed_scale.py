#!/usr/bin/env python3
"""Plot validation syntheses with a fixed color scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation maps with fixed vmin/vmax.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vmin", type=float, default=-6.0)
    parser.add_argument("--vmax", type=float, default=6.0)
    parser.add_argument("--max-syntheses", type=int, default=7)
    parser.add_argument("--subtract-mean", type=str, default="true")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    payload = np.load(args.input)
    subtract_mean = parse_bool(args.subtract_mean)

    images: list[tuple[str, np.ndarray]] = []
    if "target_map" in payload.files:
        target = payload["target_map"]
        if target.ndim == 3:
            target = target[0]
        images.append(("original", target))
    if "synthesized_maps" not in payload.files:
        raise KeyError("Expected key 'synthesized_maps' in validation NPZ")
    synth = payload["synthesized_maps"]
    for i in range(min(args.max_syntheses, synth.shape[0])):
        images.append((f"synthesis {i + 1}", synth[i]))

    ncols = 4
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (title, image) in zip(axes, images):
        image = np.asarray(image, dtype=float)
        if subtract_mean:
            image = image - np.nanmean(image)
        im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[: len(images)], shrink=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
