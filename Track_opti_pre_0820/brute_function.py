# brute_force_id_parallel.py  (UPDATED: data loading matches trainMain_sparse_torch style)
# -----------------------------------------------------------------------------
# Key change requested:
#   - Load traj_info_{itr}.npz exactly like your GP training loop:
#       for itr in range(iteration):
#           data = np.load(os.path.join(file_dir,'Data', f"traj_info_{itr}.npz"))
#           append x_true_prev / x_u / x_true
#   - Use the last file (iteration-1) for "prev residual plot" baseline, same 느낌으로.
# -----------------------------------------------------------------------------

import os
import json
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from Track.track import Track
from Optimization.dynamics import VehicleDynamics, VehicleDynamicsTrue

_G = {}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(a.shape[0], -1)

def _precompute_s_inds(track_s, s_center_len, s_query_1d):
    s_ind = np.searchsorted(track_s, s_query_1d, side="left")
    s_ind = np.where(s_ind >= s_center_len, s_ind - s_center_len, s_ind)
    s_ind = np.where(s_ind < 0, 0, s_ind)
    return s_ind.astype(int)

def _make_grid(bounds, grid_sizes):
    return {
        "B":  np.linspace(bounds["B"][0],  bounds["B"][1],  grid_sizes["B"]),
        "C":  np.linspace(bounds["C"][0],  bounds["C"][1],  grid_sizes["C"]),
        "mu": np.linspace(bounds["mu"][0], bounds["mu"][1], grid_sizes["mu"]),
        "Iz": np.linspace(bounds["Iz"][0], bounds["Iz"][1], grid_sizes["Iz"]),
    }

def _ensure_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "Data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "Figure"), exist_ok=True)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_residual_triplet(
    y_vx, y_vx_prev, y_vx_true,
    y_vy, y_vy_prev, y_vy_true,
    y_om, y_om_prev, y_om_true,
    file_dir, iteration, prefix="brute"
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(y_vx,      lw=1.8, label="tuned (grid)")
    axes[0].plot(y_vx_prev, lw=1.5, ls="--", label="initial (prev)")
    axes[0].plot(y_vx_true, lw=2.0, ls="--", label="initial (true)")
    axes[0].set_title("Residual: vx (x_true - x_pred)")
    axes[0].set_ylabel("vx resid")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(y_vy,      lw=1.8, label="tuned (grid)")
    axes[1].plot(y_vy_prev, lw=1.5, ls="--", label="initial (prev)")
    axes[1].plot(y_vy_true, lw=2.0, ls="--", label="initial (true)")
    axes[1].set_title("Residual: vy (x_true - x_pred)")
    axes[1].set_ylabel("vy resid")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    axes[2].plot(y_om,      lw=1.8, label="tuned (grid)")
    axes[2].plot(y_om_prev, lw=1.5, ls="--", label="initial (prev)")
    axes[2].plot(y_om_true, lw=2.0, ls="--", label="initial (true)")
    axes[2].set_title("Residual: omega (x_true - x_pred)")
    axes[2].set_xlabel("Timestep (0 ... T-1)")
    axes[2].set_ylabel("omega resid")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    plt.tight_layout()
    out = os.path.join(file_dir, "Figure", f"{prefix}_residual_iter{iteration-1}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out

# -----------------------------------------------------------------------------
# Worker init + evaluation
# -----------------------------------------------------------------------------
def _init_worker(track_dir, track_file, traj_type, dt, N,
                 x_true_prev, x_u, x_true, s_inds, idx):
    track = Track(track_dir, track_file=track_file, traj_type=traj_type)
    dyn = VehicleDynamics(track, dt=dt, N=N)

    _G["track"] = track
    _G["dyn"] = dyn
    _G["x_true_prev"] = x_true_prev
    _G["x_u"] = x_u
    _G["x_true"] = x_true
    _G["s_inds"] = s_inds
    _G["idx"] = idx

def _eval_one_paramset(param_vec):
    B, C, mu, Iz = map(float, param_vec)

    track = _G["track"]
    dyn = _G["dyn"]
    x_prev = _G["x_true_prev"]
    x_u = _G["x_u"]
    x_true = _G["x_true"]
    s_inds = _G["s_inds"]
    idx = _G["idx"]

    dyn.B = B
    dyn.C = C
    dyn.wheel_friction = mu
    dyn.Iz = Iz

    sse = 0.0
    for i in range(x_prev.shape[0]):
        curv = track.curv_center[s_inds[i]]
        x_pred = dyn.f_d_rk4_time(x_prev[i], x_u[i], curv)
        r = np.asarray(x_pred).reshape(-1)[idx] - x_true[i, idx]
        sse += float(np.sum(r * r))
    return (sse, B, C, mu, Iz)

def brute_force_parallel(
    x_true_prev, x_u, x_true,
    track_dir, track_file, traj_type,
    dt, N,
    bounds, grid_sizes,
    idx=np.array([3, 4, 5]),
    n_workers=8,
    chunk_size=256,
    verbose_every=5000
):
    x_true_prev = _to_2d(x_true_prev).astype(np.float64, copy=False)
    x_u = _to_2d(x_u).astype(np.float64, copy=False)
    x_true = _to_2d(x_true).astype(np.float64, copy=False)
    print(track_dir,track_file)
    track_dir=track_dir+"/"
    track_tmp = Track(track_dir, track_file=track_file, traj_type=traj_type)
    s_inds = _precompute_s_inds(track_tmp.s, len(track_tmp.s_center), x_true_prev[:, 0])

    grids = _make_grid(bounds, grid_sizes)
    combos = itertools.product(grids["B"], grids["C"], grids["mu"], grids["Iz"])

    best = {"sse": np.inf, "params": None, "n_eval": 0}

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(track_dir, track_file, traj_type, dt, N,
                  x_true_prev, x_u, x_true, s_inds, np.asarray(idx, dtype=int))
    ) as pool:

        n_eval = 0
        while True:
            batch = list(itertools.islice(combos, chunk_size))
            if not batch:
                break

            futures = [pool.submit(_eval_one_paramset, p) for p in batch]

            for f in futures:
                sse, B, C, mu, Iz = f.result()
                n_eval += 1

                if sse < best["sse"]:
                    best["sse"] = float(sse)
                    best["params"] = np.array([B, C, mu, Iz], dtype=float)

                if verbose_every and (n_eval % verbose_every == 0):
                    print(f"[grid {n_eval}] best SSE={best['sse']:.6g} | params={best['params']}")

    best["n_eval"] = n_eval
    return best

# -----------------------------------------------------------------------------
# Data loading in the SAME FORMAT as your torch_sparse_gp.py
# -----------------------------------------------------------------------------
def load_traj_npz_stack(file_dir, iteration):
    """
    Matches:
        for itr in range(iteration):
            data = np.load(os.path.join(file_dir,'Data', f"traj_info_{itr}.npz"))
            x_true = data['x_true'][2:, :]
            x_true_prev = data['x_true'][1:-1, :]
            x_u = data['u'][1:]
            append...
    Returns stacked arrays across itr=0..iteration-1
    """
    x_prev_all, x_u_all, x_true_all = [], [], []

    for itr in range(iteration):
        data = np.load(os.path.join(file_dir, "Data", f"traj_info_{itr}.npz"), allow_pickle=True)
        x_true = data["x_true"][2:, :]
        x_true_prev = data["x_true"][1:-1, :]
        x_u = data["u"][1:]
        x_prev_all.append(x_true_prev)
        x_u_all.append(x_u)
        x_true_all.append(x_true)

    X_prev = np.concatenate(x_prev_all, axis=0)
    U = np.concatenate(x_u_all, axis=0)
    X_true = np.concatenate(x_true_all, axis=0)
    return X_prev, U, X_true

def load_prev_residual_last(file_dir, iteration):
    """
    For plotting baseline "initial(prev)" we mimic your existing usage:
      use the LAST dataset traj_info_{iteration-1}.npz
      y_prev = x_true[2:, 3:] - x_pred[1:, 3:]
    """
    assert iteration >= 1
    data_last = np.load(os.path.join(file_dir, "Data", f"traj_info_{iteration-1}.npz"), allow_pickle=True)

    x_true_345 = data_last["x_true"][2:, 3:]
    x_pred_345 = data_last["x_pred"][1:, 3:]
    y_prev = x_true_345 - x_pred_345
    return y_prev[:, 0], y_prev[:, 1], y_prev[:, 2], data_last

# -----------------------------------------------------------------------------
# High-level entrypoint (same "iteration means #files" semantics as trainMain_sparse_torch)
# -----------------------------------------------------------------------------
def runMain_bruteforce_id(
    iteration,                 # NOTE: iteration = number of datasets to use (0..iteration-1)
    file_dir="./MAP0/GP",
    map_dir="MAP0",
    file_name="optimized_traj",
    track_idx=0,               # track file suffix (matches your usual {0})
    traj_type="opt_traj",
    dt=0.1,
    N=15,
    bounds=None,
    grid_sizes=None,
    n_workers=8,
    chunk_size=256,
    verbose_every=2000,
    save=True,
    plot=True
):
    _ensure_dirs(file_dir)

    if bounds is None:
        bounds = {
            "B":  (1.0, 1.2),
            "C":  (1.1, 1.3),
            "mu": (0.7, 0.9),
            "Iz": (0.014, 0.024),
        }
    if grid_sizes is None:
        grid_sizes = {"B": 10, "C": 10, "mu": 10, "Iz": 10}

    # ---- load stacked identification data (SAME FORMAT) ----
    x_true_prev_all, x_u_all, x_true_all = load_traj_npz_stack(file_dir, iteration)

    # ---- load last-file residual baselines for plotting ----
    y_vx_prev, y_vy_prev, y_om_prev, data_last = load_prev_residual_last(file_dir, iteration)

    # for tuned/true residual plotting, we also use the last-file trajectories (consistent baseline)
    x_true_last = data_last["x_true"][2:, :]
    x_true_prev_last = data_last["x_true"][1:-1, :]
    x_u_last = data_last["u"][1:]

    # ---- track setup ----
    track_dir = os.path.join(map_dir, "Traj")
    track_file = f"{file_name}{track_idx}"
    print(track_dir)
 

    # ---- grid search on ALL stacked data ----
    best = brute_force_parallel(
        x_true_prev=x_true_prev_all,
        x_u=x_u_all,
        x_true=x_true_all,
        track_dir=track_dir,
        track_file=track_file,
        traj_type=traj_type,
        dt=dt,
        N=N,
        bounds=bounds,
        grid_sizes=grid_sizes,
        n_workers=n_workers,
        chunk_size=chunk_size,
        verbose_every=verbose_every,
    )

    B, C, mu, Iz = best["params"]
    print("\nBEST (GRID SEARCH):")
    print("  used files: 0 ...", iteration - 1)
    print("  n_eval:", best["n_eval"])
    print("  SSE:", best["sse"])
    print("  params [B, C, mu, Iz]:", best["params"])

    # ---- compute tuned residual on LAST file (for a clean apples-to-apples plot) ----
    track_dir=track_dir+"/"
    track = Track(track_dir, track_file=track_file, traj_type=traj_type)
    dyn = VehicleDynamics(track, dt=dt, N=N)
    dyn.B = float(B)
    dyn.C = float(C)
    dyn.wheel_friction = float(mu)
    dyn.Iz = float(Iz)

    dyn_true = VehicleDynamicsTrue(track, dt=dt, N=N)

    s_inds = _precompute_s_inds(track.s, len(track.s_center), np.asarray(x_true_prev_last)[:, 0])

    res = []
    res_true = []
    for i in range(len(x_true_prev_last)):
        curv = track.curv_center[s_inds[i]]
        x_pred = dyn.f_d_rk4_time(x_true_prev_last[i], x_u_last[i], curv)
        x_True = dyn_true.f_d_rk4_time(x_true_prev_last[i], x_u_last[i], curv)
        res.append(np.asarray(x_true_last[i, :] - x_pred).reshape(1, -1))
        res_true.append(np.asarray(x_true_last[i, :] - x_True).reshape(1, -1))

    res_mat = np.vstack(res)
    res_true_mat = np.vstack(res_true)

    vx_idx, vy_idx, om_idx = 3, 4, 5
    y_vx = res_mat[:, vx_idx]
    y_vy = res_mat[:, vy_idx]
    y_om = res_mat[:, om_idx]

    y_vx_true = res_true_mat[:, vx_idx]
    y_vy_true = res_true_mat[:, vy_idx]
    y_om_true = res_true_mat[:, om_idx]

    # ---- save results ----
    out_bundle = {
        "best_sse": float(best["sse"]),
        "best_params": {"B": float(B), "C": float(C), "mu": float(mu), "Iz": float(Iz)},
        "bounds": bounds,
        "grid_sizes": grid_sizes,
        "n_eval": int(best["n_eval"]),
        "used_files": [int(i) for i in range(iteration)],
        "dt": float(dt),
        "N": int(N),
        "track_file": track_file,
        "traj_type": traj_type,
    }

    if save:
        meta_path = os.path.join(file_dir, "Data", f"brute_best_iter{iteration-1}.json")
        with open(meta_path, "w") as f:
            json.dump(out_bundle, f, indent=2)

        npz_out = os.path.join(file_dir, "Data", f"brute_residual_iter{iteration-1}.npz")
        np.savez(
            npz_out,
            y_vx=y_vx, y_vy=y_vy, y_om=y_om,
            y_vx_prev=y_vx_prev, y_vy_prev=y_vy_prev, y_om_prev=y_om_prev,
            y_vx_true=y_vx_true, y_vy_true=y_vy_true, y_om_true=y_om_true,
            params=np.array([B, C, mu, Iz], dtype=float),
            sse=float(best["sse"]),
        )
        print(f"[Saved] {meta_path}")
        print(f"[Saved] {npz_out}")

    fig_path = None
    if plot:
        fig_path = plot_residual_triplet(
            y_vx, y_vx_prev, y_vx_true,
            y_vy, y_vy_prev, y_vy_true,
            y_om, y_om_prev, y_om_true,
            file_dir=file_dir,
            iteration=iteration,
            prefix="brute",
        )
        print(f"[Saved] {fig_path}")

    return B,C,mu,Iz,out_bundle, fig_path


# -----------------------------------------------------------------------------
# Example run (same semantics as trainMain_sparse_torch)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(os.cpu_count)
    import time
    start_time=time.time()
    runMain_bruteforce_id(
        iteration=1,                 # uses traj_info_0.npz
        file_dir="./MAP0/GP",
        map_dir="MAP0",
        file_name="optimized_traj",
        track_idx=0,
        traj_type="opt_traj",
        dt=0.1,
        N=15,
        bounds={
            "B":  (1.0, 1.2),
            "C":  (1.1, 1.3),
            "mu": (0.7, 0.9),
            "Iz": (0.014, 0.024),
        },
        grid_sizes={"B": 10, "C": 10, "mu": 10, "Iz": 10},
        n_workers=24,
        chunk_size=256,
        verbose_every=2000,
        save=True,
        plot=True,
    )
    print(time.time()-start_time)