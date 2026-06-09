#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_dataset(dataset_root: Path, patch: str) -> Path:
    patch = str(int(patch))
    matches = sorted(dataset_root.glob(f"p{patch}_*_moment_dataset_full.npz"))
    if not matches:
        patch_dir = dataset_root / f"patch_{patch}"
        matches = sorted(patch_dir.glob(f"p{patch}_*_moment_dataset_full.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No dataset shard found for patch {patch} under {dataset_root} "
            f"or {dataset_root / f'patch_{patch}'}. "
            f"Expected pattern: p{patch}_*_moment_dataset_full.npz"
        )
    if len(matches) > 1:
        print(
            f"Warning: found {len(matches)} dataset shards for patch {patch}; using {matches[0].name}",
            flush=True,
        )
    return matches[0]


def robust_limits(arrays: list[np.ndarray], q: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    lim = float(np.nanpercentile(np.abs(values), q))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return (-lim, lim)


def load_indices(n_available: int, n_examples: int, seed: int, mode: str) -> np.ndarray:
    n = min(n_examples, n_available)
    if mode == "first":
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_available, size=n, replace=False))


def plot_training_pairs(
    dataset_path: Path,
    out_path: Path,
    *,
    n_examples: int,
    seed: int,
    index_mode: str,
) -> dict[str, object]:
    data = np.load(dataset_path)
    # Dataset convention: x* are synthesized clean signal maps s', y* are data maps d=s'+n.
    required = ("xq", "yq", "xu", "yu")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset {dataset_path} is missing required keys: {missing}")

    xq = data["xq"]
    yq = data["yq"]
    xu = data["xu"]
    yu = data["yu"]
    if not (xq.shape == yq.shape == xu.shape == yu.shape):
        raise RuntimeError(
            f"Shape mismatch: xq={xq.shape} yq={yq.shape} xu={xu.shape} yu={yu.shape}"
        )

    idx = load_indices(xq.shape[0], n_examples, seed, index_mode)
    q_noise = yq[idx] - xq[idx]
    u_noise = yu[idx] - xu[idx]

    q_vmin, q_vmax = robust_limits([xq[idx], yq[idx]])
    u_vmin, u_vmax = robust_limits([xu[idx], yu[idx]])
    qn_vmin, qn_vmax = robust_limits([q_noise])
    un_vmin, un_vmax = robust_limits([u_noise])

    columns = [
        ("Q data d_Q", yq[idx], q_vmin, q_vmax),
        ("Q synth s'_Q", xq[idx], q_vmin, q_vmax),
        ("Q nuisance n_Q", q_noise, qn_vmin, qn_vmax),
        ("U data d_U", yu[idx], u_vmin, u_vmax),
        ("U synth s'_U", xu[idx], u_vmin, u_vmax),
        ("U nuisance n_U", u_noise, un_vmin, un_vmax),
    ]

    fig, axes = plt.subplots(len(idx), len(columns), figsize=(18, max(2.2 * len(idx), 4.0)))
    if len(idx) == 1:
        axes = axes[None, :]

    for col, (title, stack, vmin, vmax) in enumerate(columns):
        axes[0, col].set_title(title)
        for row in range(len(idx)):
            ax = axes[row, col]
            ax.imshow(stack[row], cmap="coolwarm", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"idx {int(idx[row])}", rotation=0, ha="right", va="center")

    fig.suptitle(dataset_path.name, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    summary = {
        "dataset": str(dataset_path),
        "figure": str(out_path),
        "training_convention": {
            "q_network_pair": "d_Q -> s'_Q, stored as yq -> xq",
            "u_network_pair": "d_U -> s'_U, stored as yu -> xu",
            "nuisance": "n = d - s'",
        },
        "keys": {key: list(data[key].shape) for key in data.files if hasattr(data[key], "shape")},
        "indices": [int(i) for i in idx],
        "n_examples": int(len(idx)),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot moment-network dataset training pairs.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("/pscratch/sd/a/atsouros/STL/moment_network_dataset/version_2"))
    parser.add_argument("--patch", default="3")
    parser.add_argument("--out-dir", type=Path, default=Path("plots_dataset"))
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-mode", choices=("first", "random"), default="first")
    parser.add_argument("--name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = (
        args.dataset.expanduser()
        if args.dataset is not None
        else find_dataset(args.dataset_root.expanduser(), args.patch)
    )
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name.strip() or dataset_path.with_suffix("").name
    out_path = out_dir / f"{name}_training_pairs.png"
    summary_path = out_dir / f"{name}_training_pairs.json"

    summary = plot_training_pairs(
        dataset_path,
        out_path,
        n_examples=args.n_examples,
        seed=args.seed,
        index_mode=args.index_mode,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(f"Saved {out_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
