#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Repo root (for imports)
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as FFT_DataClass
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch as Kernel_DataClass
from utils import (
    NUISANCE_DIR,
    SIGNAL_DIR,
    _build_identity_config,
    _build_run_config,
    _build_run_stem,
    _center_crop,
    _configure_backend_defaults,
    _downsample_by_four,
    _filter_nuisance_version,
    _load_noise_batch,
    _maybe_set_wtype,
    _nuisance_version_suffix,
    _parse_patch_list,
    _select_no_bonus_signal_path,
)


def _build_default_config() -> dict[str, object]:
    return {
        "patch": "3",
        "map_size": None,
        "backend": "fft",
        "dtype": "float64",
        "wtype": "Bump-Steerable",
        "seed": 0,
        "batch": 15,
        "epochs": 10,
        "epoch_steps": 1,
        "lbfgs_max_iter": 100,
        "pbc": False,
        "white_noise_initial": False,
        "start_without_noise_channels": False,
        "nuisance_version": "v4_10_arcmin",
        "st_reduced": True,
        "st_j": 7,
        "st_l": 4,
        "st_iso": False,
        "st_angular_ft": True,
        "st_scale_ft": True,
        "st_harmonics_angle": 2,
        "st_harmonics_scale": 3,
        "st_dj": 3,
        "st_compute_ps": True,
        "st_has_fewer_convolutions": True,
    }


def _run_patch(patch: str, *, rank: int = 0) -> None:
    signal_dir = Path(os.environ.get("PLANCK_SIGNAL_DIR", str(SIGNAL_DIR))).expanduser()
    nuisance_dir = Path(os.environ.get("PLANCK_NUISANCE_DIR", str(NUISANCE_DIR))).expanduser()
    root = signal_dir.parent

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    seed = int(os.environ.get("SEED", "0"))
    n_batch = int(os.environ.get("BATCH", "15"))
    epochs = int((os.environ.get("EPOCHS") or os.environ.get("OUTER_ITERS") or "10").strip())
    epoch_steps = int((os.environ.get("EPOCH_STEPS") or "1").strip())
    lbfgs_max_iter = int(os.environ.get("LBFGS_MAX_ITER", "100"))
    start_without_noise_channels = (
        os.environ.get("START_WITHOUT_NOISE_CHANNELS") or "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    pbc = (os.environ.get("PBC") or os.environ.get("PBC") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    white_noise_initial = (os.environ.get("WHITE_NOISE_INITIAL") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_reduced = (os.environ.get("ST_REDUCED") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_j = int((os.environ.get("ST_J") or "7").strip())
    st_l = int((os.environ.get("ST_L") or "4").strip())
    st_iso = (os.environ.get("ST_ISO") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_angular_ft = (os.environ.get("ST_ANGULAR_FT") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_scale_ft = (os.environ.get("ST_SCALE_FT") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_harmonics_angle = int((os.environ.get("ST_HARMONICS_ANGLE") or "2").strip())
    st_harmonics_scale = int((os.environ.get("ST_HARMONICS_SCALE") or "3").strip())
    st_dj = int((os.environ.get("ST_DJ") or "3").strip())
    st_compute_ps = (os.environ.get("ST_COMPUTE_PS") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    st_has_fewer_convolutions = (
        os.environ.get("ST_FEWER_CONVOLUTIONS") or "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    wtype = os.environ.get("WTYPE", "Bump-Steerable")
    map_size_env = (os.environ.get("MAP_SIZE") or "").strip()
    map_size = int(map_size_env) if map_size_env else None
    nuisance_version = (os.environ.get("PLANCK_NUISANCE_VERSION") or "v4_10_arcmin").strip()
    if nuisance_version not in {"v4_10_arcmin", "v2", "all"}:
        raise ValueError("PLANCK_NUISANCE_VERSION must be 'v4_10_arcmin', 'v2', or 'all'")

    backend = (os.environ.get("BACKEND") or "fft").strip().lower()
    if backend not in {"fft", "kernel"}:
        raise ValueError("BACKEND must be 'fft' or 'kernel'")
    DataClass = FFT_DataClass if backend == "fft" else Kernel_DataClass

    # -------------------------------------------------------------------------
    # Device / dtype
    # -------------------------------------------------------------------------
    device_override = (os.environ.get("DEVICE") or "").strip()
    if device_override:
        device = torch.device(device_override)
    else:
        local_rank = int(os.environ.get("SLURM_LOCALID", str(rank)))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
    print(f"Rank {rank} | patch {patch} | using device: {device} (backend={backend})")

    dtype_str = (os.environ.get("DTYPE") or "float64").strip().lower()
    if dtype_str in {"float64", "fp64", "double"}:
        dtype = torch.float64
    elif dtype_str in {"float32", "fp32", "single"}:
        dtype = torch.float32
    else:
        raise ValueError("DTYPE must be float64 or float32")

    _configure_backend_defaults(device=device, dtype=dtype)
    torch.manual_seed(seed)
    print(f"PBC: {pbc}")
    print(f"Epochs: {epochs} | steps per epoch: {epoch_steps}")
    print("Optimizer: lbfgs | one optimizer iteration per logged step")
    if st_reduced:
        print(
            "Reduced ST config: "
            f"J={st_j} | L={st_l} | iso={st_iso} | angular_ft={st_angular_ft} | "
            f"scale_ft={st_scale_ft} | harmonics_angle={st_harmonics_angle} | "
            f"harmonics_scale={st_harmonics_scale} | dj={st_dj} | "
            f"compute_PS={st_compute_ps} | fewer_convolutions={st_has_fewer_convolutions}"
        )
    else:
        print(
            "Standard ST config: "
            f"compute_PS={st_compute_ps} | fewer_convolutions={st_has_fewer_convolutions}"
        )

    # -------------------------------------------------------------------------
    # Load NERSC Planck patch inputs
    # -------------------------------------------------------------------------
    freq = int(os.environ.get("FREQ", "353"))
    q353_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_Q{freq}_*.npy", f"Q{freq}")
    u353_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_U{freq}_*.npy", f"U{freq}")
    i857_path = _select_no_bonus_signal_path(signal_dir, f"patch_{patch}_I857_*.npy", "I857")
    print(f"Q{freq}:", q353_path.name)
    print(f"U{freq}:", u353_path.name)
    print("I857:", i857_path.name)

    d_q = _downsample_by_four(np.load(q353_path).astype(np.float64))
    d_u = _downsample_by_four(np.load(u353_path).astype(np.float64))
    aux = _downsample_by_four(np.load(i857_path).astype(np.float64))

    if map_size is not None:
        d_q = _center_crop(d_q, out_hw=(map_size, map_size))
        d_u = _center_crop(d_u, out_hw=(map_size, map_size))
        aux = _center_crop(aux, out_hw=(map_size, map_size))

    aux = aux - float(np.mean(aux))  # fixed auxiliary map, mean-subtracted

    H, W = d_q.shape
    if d_u.shape != (H, W) or aux.shape != (H, W):
        raise RuntimeError(
            f"Shape mismatch: d_q={d_q.shape}, d_u={d_u.shape}, aux={aux.shape}"
        )
    expected_map_size = int(os.environ.get("EXPECTED_MAP_SIZE", "384"))
    if map_size is None and (H, W) != (expected_map_size, expected_map_size):
        raise RuntimeError(
            f"Expected downsampled maps to be {expected_map_size}x{expected_map_size}, got {H}x{W}."
        )

    # -------------------------------------------------------------------------
    # Pair nuisance samples by (noise_seed, CMB_res_seed)
    # -------------------------------------------------------------------------
    pat = re.compile(r"noise_seed_(\d+)_CMB_res_seed_(\d+)")

    def parse_key(path: Path) -> tuple[int, int]:
        m = pat.search(path.name)
        if not m:
            raise ValueError(f"Could not parse seeds from: {path.name}")
        return (int(m.group(1)), int(m.group(2)))

    q_pattern = f"patch_{patch}_noise_Q{freq}_*.npy"
    u_pattern = f"patch_{patch}_noise_U{freq}_*.npy"
    q_paths = _filter_nuisance_version(sorted(nuisance_dir.glob(q_pattern)), nuisance_version)
    u_paths = _filter_nuisance_version(sorted(nuisance_dir.glob(u_pattern)), nuisance_version)
    if not q_paths or not u_paths:
        suffix = _nuisance_version_suffix(nuisance_version)
        version_msg = "" if suffix is None else f" with suffix *{suffix}"
        raise FileNotFoundError(
            f"Could not find nuisance samples{version_msg}. Looked for "
            f"{nuisance_dir / q_pattern} and {nuisance_dir / u_pattern}."
        )
    print(f"Nuisance version: {nuisance_version}")
    try:
        q_by_key = {parse_key(p): p for p in q_paths}
        u_by_key = {parse_key(p): p for p in u_paths}
        if len(q_by_key) != len(q_paths) or len(u_by_key) != len(u_paths):
            raise RuntimeError(
                "Nuisance seed keys are not unique. Set PLANCK_NUISANCE_VERSION to "
                "'v4_10_arcmin' or 'v2' instead of 'all'."
            )
        paired_keys = sorted(set(q_by_key).intersection(u_by_key))
        if not paired_keys:
            raise RuntimeError("No paired nuisance samples found for Q/U (seed keys empty).")
        noise_q_paths = [q_by_key[k] for k in paired_keys]
        noise_u_paths = [u_by_key[k] for k in paired_keys]
        print("Paired nuisance samples:", len(paired_keys))
        print("First key:", paired_keys[0], "->", noise_q_paths[0].name, "|", noise_u_paths[0].name)
    except ValueError:
        if len(q_paths) != len(u_paths):
            raise RuntimeError(
                f"Cannot pair nuisance Q/U samples: Q has {len(q_paths)} files, U has {len(u_paths)} files, "
                "and filenames do not match the expected seed pattern."
            )
        noise_q_paths = q_paths
        noise_u_paths = u_paths
        print("Paired nuisance samples by index:", len(q_paths))

    n_noise = len(noise_q_paths)

    # -------------------------------------------------------------------------
    # Phase 1 (optional): 3 channels [dQ, dU, aux], no explicit noise channels
    # Phase 2: 5 channels [dQ, nQ, dU, nU, aux]
    # -------------------------------------------------------------------------
    CROSS_MATRIX_NO_NOISE_CHANNELS = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    CROSS_MATRIX_WITH_NOISE_CHANNELS = torch.tensor(
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    CROSS_MATRIX_REF_3 = torch.eye(3, dtype=torch.bool, device=device)
    CROSS_MATRIX_REF_5 = torch.eye(5, dtype=torch.bool, device=device)

    # -------------------------------------------------------------------------
    # Operator + normalization references
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    ref_i = int(rng.integers(0, n_noise))
    n_q_ref = _downsample_by_four(np.load(noise_q_paths[ref_i]).astype(np.float64))
    n_u_ref = _downsample_by_four(np.load(noise_u_paths[ref_i]).astype(np.float64))
    if map_size is not None:
        n_q_ref = _center_crop(n_q_ref, out_hw=(H, W))
        n_u_ref = _center_crop(n_u_ref, out_hw=(H, W))

    ref_tensor_3 = torch.from_numpy(np.stack([d_q, d_u, aux], axis=0)).to(device, dtype=dtype)
    ref_dc_3 = DataClass(ref_tensor_3[None, ...], pbc=pbc)  # (1, 3, H, W)

    ref_tensor_5 = torch.from_numpy(np.stack([d_q, n_q_ref, d_u, n_u_ref, aux], axis=0)).to(
        device, dtype=dtype
    )  # (5, H, W)
    ref_dc_5 = DataClass(ref_tensor_5[None, ...], pbc=pbc)  # (1, 5, H, W)

    def build_st_op(ref_dc):
        if st_reduced:
            st_op_local = ref_dc.get_ST_op(
                J=st_j,
                L=st_l,
                iso=st_iso,
                angular_ft=st_angular_ft,
                scale_ft=st_scale_ft,
                harmonics_angle=st_harmonics_angle,
                harmonics_scale=st_harmonics_scale,
                dj=st_dj,
                compute_PS=st_compute_ps,
                has_fewer_convolutions=st_has_fewer_convolutions,
            )
        else:
            st_op_local = ref_dc.get_ST_op(
                compute_PS=st_compute_ps,
                has_fewer_convolutions=st_has_fewer_convolutions,
            )
        _maybe_set_wtype(st_op=st_op_local, ref_dc=ref_dc, wtype=wtype)
        return st_op_local

    st_op_3 = build_st_op(ref_dc_3)
    st_op_5 = build_st_op(ref_dc_5)

    with torch.no_grad():
        st_op_3.apply(
            ref_dc_3,
            norm="store_ref",
            compute_cross_matrix=CROSS_MATRIX_REF_3,
        )
        st_op_5.apply(
            ref_dc_5,
            norm="store_ref",
            compute_cross_matrix=CROSS_MATRIX_REF_5,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    d_q_t = torch.from_numpy(np.copy(d_q)).to(device, dtype=dtype)
    d_u_t = torch.from_numpy(np.copy(d_u)).to(device, dtype=dtype)
    aux_t = torch.from_numpy(np.copy(aux)).to(device, dtype=dtype)
    def stats_flat(dc: DataClass, *, phase: str) -> torch.Tensor:
        if phase == "without_noise_channels":
            return st_op_3.apply(
                dc,
                norm="load_ref",
                compute_cross_matrix=CROSS_MATRIX_NO_NOISE_CHANNELS,
            ).to_flatten(mean_along_batch=True, keepnans=False)
        return st_op_5.apply(
            dc,
            norm="load_ref",
            compute_cross_matrix=CROSS_MATRIX_WITH_NOISE_CHANNELS,
        ).to_flatten(mean_along_batch=True, keepnans=False)

    def make_target_batch(
        nb: int,
        *,
        phase: str,
        batch_nq: torch.Tensor | None = None,
        batch_nu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if phase == "without_noise_channels":
            return torch.stack(
                [
                    d_q_t.expand(nb, -1, -1),
                    d_u_t.expand(nb, -1, -1),
                    aux_t.expand(nb, -1, -1),
                ],
                dim=1,
            )  # (Nb, 3, H, W)
        if batch_nq is None or batch_nu is None:
            raise RuntimeError("batch_nq and batch_nu must be provided in the noise-channel phase.")
        return torch.stack(
            [
                d_q_t.expand(nb, -1, -1),
                batch_nq,
                d_u_t.expand(nb, -1, -1),
                batch_nu,
                aux_t.expand(nb, -1, -1),
            ],
            dim=1,
        )  # (Nb, 5, H, W)

    def squared_l2(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().square().sum()

    # -------------------------------------------------------------------------
    # Optimization (jointly optimize Q and U)
    # -------------------------------------------------------------------------
    init_std_q = float(torch.std(d_q_t).detach().cpu())
    init_std_u = float(torch.std(d_u_t).detach().cpu())
    if white_noise_initial:
        print(
            f"Initialization per epoch: white noise | std(Q)={init_std_q:.6g} | std(U)={init_std_u:.6g}"
        )
    else:
        print("Initialization per epoch: data maps")

    loss_calls: list[float] = []
    print(f"Start without explicit noise channels: {start_without_noise_channels}")
    printed_target_size_by_phase = {
        "without_noise_channels": False,
        "with_noise_channels": False,
    }

    def make_running_signal_qu() -> torch.Tensor:
        if white_noise_initial:
            running_signal_qu_local = torch.stack(
                [
                    torch.randn_like(d_q_t) * init_std_q,
                    torch.randn_like(d_u_t) * init_std_u,
                ],
                dim=0,
            )
        else:
            running_signal_qu_local = torch.stack([d_q_t.clone(), d_u_t.clone()], dim=0)
        running_signal_qu_local.requires_grad_()
        return running_signal_qu_local

    def make_optimizer() -> torch.optim.LBFGS:
        return torch.optim.LBFGS(
            [running_signal_qu],
            lr=1,
            max_iter=1,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None,
        )

    for epoch_idx in range(epochs):
        running_signal_qu = make_running_signal_qu()
        optimizer = make_optimizer()
        for step_idx in range(epoch_steps):
            use_without_noise_phase = (
                start_without_noise_channels
                and step_idx < (epoch_steps // 2)
            )
            phase = "without_noise_channels" if use_without_noise_phase else "with_noise_channels"
            idx = rng.choice(n_noise, size=min(n_batch, n_noise), replace=False)
            q_batch_np = _load_noise_batch(
                [noise_q_paths[int(i)] for i in idx],
                expected_hw=(H, W),
                crop_hw=(H, W) if map_size is not None else None,
            )
            u_batch_np = _load_noise_batch(
                [noise_u_paths[int(i)] for i in idx],
                expected_hw=(H, W),
                crop_hw=(H, W) if map_size is not None else None,
            )
            batch_nq = torch.from_numpy(q_batch_np).to(device, dtype=dtype)
            batch_nu = torch.from_numpy(u_batch_np).to(device, dtype=dtype)
            nb = int(idx.shape[0])

            with torch.no_grad():
                target_batch = make_target_batch(
                    nb,
                    phase=phase,
                    batch_nq=batch_nq,
                    batch_nu=batch_nu,
                )
                target_dc = DataClass(target_batch, pbc=pbc)
                target_flat = stats_flat(target_dc, phase=phase)
                if not printed_target_size_by_phase[phase]:
                    print(f"ST coefficient count for phase '{phase}': {target_flat.numel()}")
                    printed_target_size_by_phase[phase] = True

            def closure():
                optimizer.zero_grad()
                s_q = running_signal_qu[0]
                s_u = running_signal_qu[1]

                if phase == "without_noise_channels":
                    running_batch = torch.stack(
                        [
                            s_q[None, :, :] + batch_nq,
                            s_u[None, :, :] + batch_nu,
                            aux_t.expand(nb, -1, -1),
                        ],
                        dim=1,
                    )
                else:
                    running_batch = torch.stack(
                        [
                            s_q[None, :, :] + batch_nq,
                            (d_q_t - s_q)[None, :, :].expand(nb, -1, -1),
                            s_u[None, :, :] + batch_nu,
                            (d_u_t - s_u)[None, :, :].expand(nb, -1, -1),
                            aux_t.expand(nb, -1, -1),
                        ],
                        dim=1,
                    )

                running_dc = DataClass(running_batch, pbc=pbc)
                running_flat = stats_flat(running_dc, phase=phase)

                if running_flat.numel() != target_flat.numel():
                    raise RuntimeError(
                        f"Flattened statistic length mismatch: running={running_flat.numel()} target={target_flat.numel()}."
                    )

                loss = squared_l2(running_flat - target_flat)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            loss_value = float(loss.detach().cpu())
            loss_calls.append(loss_value)
            print(
                f"Epoch {epoch_idx+1}/{epochs} | step {step_idx+1}/{epoch_steps} | phase {phase} | minibatch {nb} | loss: {loss_value:.6g}"
            )

    recovered_q = running_signal_qu.detach().cpu().numpy()[0]
    recovered_u = running_signal_qu.detach().cpu().numpy()[1]

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    out_dir_env = (os.environ.get("OUTDIR") or "").strip()
    results_dir = Path(out_dir_env).expanduser() if out_dir_env else REPO_ROOT / "planck_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_config = _build_run_config(
        root=root,
        patch=patch,
        map_size=map_size,
        device=device,
        backend=backend,
        dtype_str=dtype_str,
        wtype=wtype,
        seed=seed,
        n_batch=n_batch,
        epochs=epochs,
        epoch_steps=epoch_steps,
        lbfgs_max_iter=lbfgs_max_iter,
        pbc=pbc,
        white_noise_initial=white_noise_initial,
        start_without_noise_channels=start_without_noise_channels,
        nuisance_version=nuisance_version,
        st_reduced=st_reduced,
        st_j=st_j,
        st_l=st_l,
        st_iso=st_iso,
        st_angular_ft=st_angular_ft,
        st_scale_ft=st_scale_ft,
        st_harmonics_angle=st_harmonics_angle,
        st_harmonics_scale=st_harmonics_scale,
        st_dj=st_dj,
        st_compute_ps=st_compute_ps,
        st_has_fewer_convolutions=st_has_fewer_convolutions,
    )
    out_stem = _build_run_stem(run_config)
    out_path = results_dir / f"{out_stem}.npy"
    np.save(out_path, np.stack([recovered_q, recovered_u], axis=0))

    metadata_path = results_dir / f"{out_stem}.json"
    metadata = {
        "run_stem": out_stem,
        "final_file": out_path.name,
        "stage1_file": None,
        "stage2_file": None,
        "loss_curve_file": f"{out_stem}_loss_curve_planck.png",
        "defaults": _build_default_config(),
        "identity_config": _build_identity_config(run_config),
        "config": run_config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")

    finite_loss = np.asarray(loss_calls, dtype=float)
    finite_loss = finite_loss[np.isfinite(finite_loss)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if finite_loss.size:
        ax.plot(finite_loss, linewidth=1)
    else:
        ax.plot([0.0], linewidth=1)
    if finite_loss.size and np.any(finite_loss > 0):
        ax.set_yscale("log")
    ax.set_xlabel("LBFGS closure call index")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs time (closure calls)")
    ax.grid(True, alpha=0.3)
    loss_path = results_dir / f"{out_stem}_loss_curve_planck.png"
    try:
        fig.tight_layout()
        fig.savefig(loss_path, dpi=200)
    except Exception as exc:
        print(f"Warning: could not save loss curve for patch {patch}: {exc}")
    plt.close(fig)

    print("Saved:", out_path)
    print("Saved:", metadata_path)
    print("Saved:", loss_path)


def main() -> None:
    try:
        from mpi4py import MPI
    except ImportError:
        comm = None
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        size = int(os.environ.get("SLURM_NTASKS", "1"))
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    patches = _parse_patch_list()
    if rank == 0:
        print(f"Patch count: {len(patches)} | MPI ranks: {size}")
        print(f"Patch range/list starts with: {patches[:min(8, len(patches))]}")

    for patch in patches[rank::size]:
        _run_patch(patch, rank=rank)
        torch.cuda.empty_cache()

    if comm is not None:
        comm.Barrier()
    if rank == 0:
        print("All assigned patches completed.")


if __name__ == "__main__":
    main()
