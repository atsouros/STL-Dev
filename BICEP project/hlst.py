#!/usr/bin/env python3
"""Average high-latitude ST targets from compsep maps and synthesize Q/U."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATCH_IDS = [129]
MAP_NORMALIZE = False
SYNTHESIS_PBC = False


def parse_patch_ids(values: list[str]) -> list[int]:
    patch_ids = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                patch_ids.append(int(token))
    return patch_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load compsep Q/U patch outputs, average one-channel ST statistics "
            "separately for Q and U, synthesize new Q/U maps with pbc=False, "
            "and save a PNG image."
        )
    )
    # Backward compatibility with older slurm wrappers. This is intentionally ignored.
    parser.add_argument("--contents", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--patch-data-dir",
        type=Path,
        default=Path("/pscratch/sd/a/atsouros/STL/planck_results/version_2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/pscratch/sd/a/atsouros/STL/BICEP"),
    )
    parser.add_argument("--patch-ids", nargs="+", default=None)
    parser.add_argument("--expected-height", type=int, default=384)
    parser.add_argument("--expected-width", type=int, default=384)
    parser.add_argument("--st-j", type=int, default=5)
    parser.add_argument("--st-l", type=int, default=4)
    parser.add_argument("--ref-patch-id", type=int, default=178)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--map-normalize",
        dest="map_normalize",
        action="store_true",
        help="Normalize Q/U map amplitudes to the reference patch before computing STs.",
    )
    norm_group.add_argument(
        "--no-map-normalize",
        dest="map_normalize",
        action="store_false",
        help="Use the Q/U maps as-is before computing STs.",
    )
    parser.set_defaults(map_normalize=MAP_NORMALIZE)
    parser.add_argument("--compute-ps", action="store_true")
    parser.add_argument(
        "--ps-method",
        choices=["legacy", "gaussian_rings"],
        default="legacy",
    )
    parser.add_argument("--has-fewer-convolutions", action="store_true")
    parser.add_argument("--synthesis-max-iter", type=int, default=100)
    parser.add_argument("--synthesis-lr", type=float, default=1.0)
    parser.add_argument("--synthesis-history-size", type=int, default=50)
    parser.add_argument("--synthesis-print-iter", type=int, default=10)
    pbc_group = parser.add_mutually_exclusive_group()
    pbc_group.add_argument(
        "--synthesis-pbc",
        dest="synthesis_pbc",
        action="store_true",
        help="Use periodic boundary conditions for the synthesized map only.",
    )
    pbc_group.add_argument(
        "--no-synthesis-pbc",
        dest="synthesis_pbc",
        action="store_false",
        help="Use non-periodic boundary conditions for the synthesized map.",
    )
    parser.set_defaults(synthesis_pbc=SYNTHESIS_PBC)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.patch_ids = PATCH_IDS if args.patch_ids is None else parse_patch_ids(args.patch_ids)
    return args


def filename_for_patch(patch_data_dir: Path, patch_id: int) -> Path:
    pattern = re.compile(rf"^p{patch_id}_.*\.npy$")
    matches = sorted(path for path in patch_data_dir.iterdir() if pattern.match(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one compsep .npy file for patch {patch_id}; found {len(matches)}."
        )
    return matches[0]


def load_compsep_patch(
    patch_data_dir: Path,
    patch_id: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    path = filename_for_patch(patch_data_dir, patch_id)
    patch = np.load(path).astype(np.float32, copy=False)
    expected = (2, *expected_shape)
    if patch.shape != expected:
        raise ValueError(f"Patch {patch_id} has shape {patch.shape}; expected {expected}.")
    return patch


def map_std(map_array: np.ndarray) -> float:
    return float(np.nanstd(map_array))


def normalize_maps_to_reference(
    patch_batch: np.ndarray,
    patch_ids: list[int],
    ref_patch: np.ndarray,
    ref_patch_id: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    q_ref_std = map_std(ref_patch[0])
    u_ref_std = map_std(ref_patch[1])
    if not np.isfinite(q_ref_std) or q_ref_std <= 0:
        raise ValueError(f"Reference patch {ref_patch_id} has invalid Q std: {q_ref_std}")
    if not np.isfinite(u_ref_std) or u_ref_std <= 0:
        raise ValueError(f"Reference patch {ref_patch_id} has invalid U std: {u_ref_std}")

    normalized = patch_batch.copy()
    rows = []
    for index, patch_id in enumerate(patch_ids):
        q_std = map_std(patch_batch[index, 0])
        u_std = map_std(patch_batch[index, 1])
        if not np.isfinite(q_std) or q_std <= 0:
            raise ValueError(f"Patch {patch_id} has invalid Q std: {q_std}")
        if not np.isfinite(u_std) or u_std <= 0:
            raise ValueError(f"Patch {patch_id} has invalid U std: {u_std}")
        q_scale = 1.0 if patch_id == ref_patch_id else q_ref_std / q_std
        u_scale = 1.0 if patch_id == ref_patch_id else u_ref_std / u_std
        normalized[index, 0] *= q_scale
        normalized[index, 1] *= u_scale
        rows.append(
            {
                "patch_id": patch_id,
                "q_std_before": q_std,
                "u_std_before": u_std,
                "q_scale": q_scale,
                "u_scale": u_scale,
                "q_std_after": q_scale * q_std,
                "u_std_after": u_scale * u_std,
            }
        )
    return normalized, rows


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_synthesis_png(
    path: Path,
    q_map: np.ndarray,
    u_map: np.ndarray,
    history: dict[str, np.ndarray],
) -> None:
    p_map = np.sqrt(q_map * q_map + u_map * u_map)
    panels = [q_map, u_map, p_map]
    titles = ["Q synthesis", "U synthesis", "P synthesis"]

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.4])
    image_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    loss_axis = fig.add_subplot(grid[1, :])

    for axis, panel, title in zip(image_axes, panels, titles):
        finite = panel[np.isfinite(panel)]
        if finite.size:
            vmax = float(np.percentile(np.abs(finite), 99))
            if vmax == 0:
                vmax = 1.0
            vmin = -vmax if title[0] != "P" else 0.0
        else:
            vmin, vmax = 0.0, 1.0
        cmap = "magma" if title[0] == "P" else "coolwarm"
        im = axis.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    if history["time"].size:
        loss_axis.plot(history["time"], history["loss"], label="joint Q/U", lw=1.8)
    loss_axis.set_xlabel("time [s]")
    loss_axis.set_ylabel("loss")
    loss_axis.set_yscale("log")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(frameon=False)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_target_stats(
    maps: np.ndarray,
    *,
    data_class,
    device: torch.device,
    st_j: int,
    st_l: int,
    compute_ps: bool,
    ps_method: str,
    has_fewer_convolutions: bool,
):
    """Compute two-channel Q/U ST statistics for a batch of maps."""
    data_tensor = torch.from_numpy(maps).to(device=device, dtype=torch.float32)
    data = data_class(data_tensor, pbc=False)
    st_op = data.get_ST_op(
        J=st_j,
        L=st_l,
        compute_PS=compute_ps,
        power_spectrum_method=ps_method,
        has_fewer_convolutions=has_fewer_convolutions,
        norm="store_ref",
    )
    cross_matrix = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool, device=device)
    with torch.no_grad():
        target_stats = st_op.apply(
            data,
            norm="store_ref",
            norm_batch_mean=True,
            compute_cross_matrix=cross_matrix,
            compute_PS=compute_ps,
            has_fewer_convolutions=has_fewer_convolutions,
        )
        if target_stats.mean_pre_std is None:
            target_stats.mean_pre_std = torch.zeros(
                (target_stats.Nb, target_stats.Nc),
                device=device,
                dtype=torch.float32,
            )
        if target_stats.std_pre_std is None:
            target_stats.std_pre_std = torch.ones(
                (target_stats.Nb, target_stats.Nc),
                device=device,
                dtype=torch.float32,
            )
        per_patch = target_stats.to_flatten(
            keep_batch_dim=True,
            keepnans=False,
            flatten_complex=True,
        )
        mean_vector = target_stats.to_flatten(
            keep_batch_dim=True,
            mean_along_batch=True,
            keepnans=False,
            flatten_complex=True,
        )[0]
    return target_stats, tensor_to_numpy(per_patch), tensor_to_numpy(mean_vector)


def synthesize_channels(
    target_stats,
    *,
    scattering_match_model,
    apply_nyquist_filter,
    shape: tuple[int, int],
    synthesis_pbc: bool,
    max_iter: int,
    lr: float,
    history_size: int,
    print_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch.manual_seed(seed)

    device = target_stats.device
    dtype = target_stats.dtype
    nbatch = 1
    mean_field = True
    nc = target_stats.Nc
    init_shape = (nbatch, nc, *shape)
    compute_ps = target_stats.compute_PS

    print("Running synthesis on device:", device, "dtype:", dtype)
    print(f"Initial shape for u: {init_shape}")

    data_running = target_stats.DataClass(
        torch.zeros(shape, device=device, dtype=dtype),
        pbc=synthesis_pbc,
    )
    running_has_nan = data_running.array.isnan().any()
    if not target_stats.compute_PS or running_has_nan:
        print(
            "Power spectrum optimization is disabled because it is not included "
            "in target_stats or because the running field has NaNs."
        )
        compute_ps = False

    st_op_running = data_running.get_ST_op(
        J=target_stats.J,
        n_bins=target_stats.n_bins,
        has_fewer_convolutions=target_stats.has_fewer_convolutions,
        compute_PS=compute_ps,
        replace_nan_value=None,
    )
    st_op_running.S2_ref_sqrt_chan_diag = target_stats.S2_ref_sqrt_chan_diag
    st_op_running.var_ref = target_stats.var_ref
    if compute_ps:
        st_op_running.PS_ref_sqrt_chan_diag = target_stats.PS_ref_sqrt_chan_diag

    target_stats_flat = target_stats.to_flatten(
        keep_batch_dim=True,
        mean_along_batch=mean_field,
    ).detach()

    model = scattering_match_model(
        st_op=st_op_running,
        DataClass=target_stats.DataClass,
        pbc=synthesis_pbc,
        init_shape=init_shape,
        init_map=None,
        device=device,
        dtype=dtype,
        has_fewer_convolutions=target_stats.has_fewer_convolutions,
        compute_cross_matrix=target_stats.compute_cross_matrix,
        compute_PS=compute_ps,
        keep_batch_dim=True,
        mean_field=mean_field,
    )

    optimizer = torch.optim.LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )
    loss_history: list[float] = []
    time_history: list[float] = []
    start = time.perf_counter()

    def closure():
        optimizer.zero_grad()
        output = model()
        loss = ((output - target_stats_flat).abs() ** 2).sum() / target_stats.Nb
        loss.backward()
        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        time_history.append(time.perf_counter() - start)
        if len(loss_history) % print_iter == 0:
            print(f"[LBFGS] iter {len(loss_history)}, loss = {loss_value:.6e}")
        return loss

    optimizer.step(closure)
    print(f"{len(loss_history)} iterations of synthesis.")
    print(f"Execution time: {time.perf_counter() - start:.3f} s")

    u_opt = model.u.detach()
    if target_stats.standardized:
        dc_u_opt = target_stats.DataClass(u_opt, pbc=synthesis_pbc)
        st_op_running.wavelet_op.unstandardize(
            dc_u_opt,
            mean=target_stats.mean_pre_std,
            std=target_stats.std_pre_std,
            inplace=True,
        )
        u_opt = dc_u_opt.array
    if st_op_running.wavelet_op.mask_full_res is not None:
        u_opt[..., st_op_running.wavelet_op.mask_full_res.array] = torch.nan
    if nc == 1:
        u_opt = u_opt[:, 0, ...]
    if nbatch == 1:
        u_opt = u_opt[0]
    if not u_opt.isnan().any():
        u_opt = apply_nyquist_filter(u_opt)

    history = {
        "time": np.asarray(time_history, dtype=np.float64),
        "loss": np.asarray(loss_history, dtype=np.float64),
    }
    return tensor_to_numpy(u_opt), history


def main() -> None:
    args = parse_args()

    repo_candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd().resolve().parent,
        Path.cwd().resolve(),
    ]
    repo_root = next(
        (candidate for candidate in repo_candidates if (candidate / "STL_main").is_dir()),
        None,
    )
    if repo_root is None:
        raise ModuleNotFoundError(
            "Could not locate STL_main. Expected it to be a sibling of the BICEP "
            "directory or available through PYTHONPATH."
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import STL_main.torch_backend as bk
    from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as FFTDataClass
    from STL_main.Synthesis import ScatteringMatchModel, apply_nyquist_filter

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    device = torch.device(args.device)
    bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = torch.float32
    bk._DEFAULT_COMPLEX_DTYPE = torch.complex64
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_shape = (args.expected_height, args.expected_width)

    print(f"Patch data directory: {args.patch_data_dir}")
    print(f"Patch ids: {args.patch_ids}")
    print(f"Expected compsep shape: {(2, *expected_shape)}")
    print("Target PBC: False")
    print("Synthesis PBC:", args.synthesis_pbc)
    print("ST coefficient normalization: enabled")
    print("Power spectrum:", args.compute_ps, "| method:", args.ps_method)
    print("Map amplitude normalization:", "enabled" if args.map_normalize else "disabled")

    patches = [
        load_compsep_patch(args.patch_data_dir, patch_id, expected_shape)
        for patch_id in args.patch_ids
    ]
    patch_batch = np.stack(patches, axis=0)
    norm_rows = []
    if args.map_normalize:
        ref_patch = load_compsep_patch(args.patch_data_dir, args.ref_patch_id, expected_shape)
        patch_batch, norm_rows = normalize_maps_to_reference(
            patch_batch=patch_batch,
            patch_ids=args.patch_ids,
            ref_patch=ref_patch,
            ref_patch_id=args.ref_patch_id,
        )
        print(f"Map amplitudes normalized to patch {args.ref_patch_id}.")
    q_maps = patch_batch[:, 0]
    u_maps = patch_batch[:, 1]
    print("Loaded Q batch:", q_maps.shape)
    print("Loaded U batch:", u_maps.shape)

    target_stats, per_patch, mu = compute_target_stats(
        patch_batch,
        data_class=FFTDataClass,
        device=device,
        st_j=args.st_j,
        st_l=args.st_l,
        compute_ps=args.compute_ps,
        ps_method=args.ps_method,
        has_fewer_convolutions=args.has_fewer_convolutions,
    )

    stats_path = args.output_dir / "stats.npz"
    np.savez(
        stats_path,
        patch_ids=np.asarray(args.patch_ids, dtype=np.int64),
        mu=mu,
        per_patch=per_patch,
        st_j=np.asarray(args.st_j, dtype=np.int64),
        st_l=np.asarray(args.st_l, dtype=np.int64),
        compute_ps=np.asarray(args.compute_ps),
        ps_method=np.asarray(args.ps_method),
        map_normalized=np.asarray(args.map_normalize),
        ref_patch_id=np.asarray(args.ref_patch_id, dtype=np.int64),
        target_pbc=np.asarray(False),
        synthesis_pbc=np.asarray(args.synthesis_pbc),
    )
    torch.save(target_stats, args.output_dir / "qu_stats.pt")
    print(f"Wrote {stats_path}")
    print("Joint Q/U mean ST vector:", mu.shape)

    print("Synthesizing Q/U jointly from average cross-channel statistics.")
    synth, history = synthesize_channels(
        target_stats,
        scattering_match_model=ScatteringMatchModel,
        apply_nyquist_filter=apply_nyquist_filter,
        shape=expected_shape,
        synthesis_pbc=args.synthesis_pbc,
        max_iter=args.synthesis_max_iter,
        lr=args.synthesis_lr,
        history_size=args.synthesis_history_size,
        print_iter=args.synthesis_print_iter,
        seed=args.seed,
    )
    q_synth, u_synth = synth[0], synth[1]

    image_path = args.output_dir / "synthesis.png"
    save_synthesis_png(image_path, q_synth, u_synth, history)
    metadata = {
        "patch_data_dir": str(args.patch_data_dir),
        "patch_ids": args.patch_ids,
        "shape": list(expected_shape),
        "target_pbc": False,
        "synthesis_pbc": args.synthesis_pbc,
        "map_normalized": args.map_normalize,
        "ref_patch_id": args.ref_patch_id,
        "normalization": norm_rows,
        "st_j": args.st_j,
        "st_l": args.st_l,
        "compute_ps": args.compute_ps,
        "ps_method": args.ps_method,
        "synthesis_max_iter": args.synthesis_max_iter,
        "synthesis_lr": args.synthesis_lr,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {image_path}")
    print(f"Outputs written under {args.output_dir}")


if __name__ == "__main__":
    main()
