import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
import warnings
warnings.filterwarnings("ignore")

from Track.track import Track
from Optimization.dynamics import *

_G = {}

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

def _init_worker(track_dir, track_file, traj_type, dt, N,
                 x_true_prev, x_u, x_true, s_inds):
    track = Track(track_dir, track_file=track_file, traj_type=traj_type)
    dyn = VehicleDynamics(track, dt=dt, N=N)
    _G["track"] = track
    _G["dyn"] = dyn
    _G["x_true_prev"] = x_true_prev
    _G["x_u"] = x_u
    _G["x_true"] = x_true
    _G["s_inds"] = s_inds

def _eval_one_paramset(param_vec):
    B, C, mu, Iz = map(float, param_vec)

    track = _G["track"]
    dyn = _G["dyn"]
    x_prev = _G["x_true_prev"]
    x_u = _G["x_u"]
    x_true = _G["x_true"]
    s_inds = _G["s_inds"]

    dyn.B = B
    dyn.C = C
    dyn.wheel_friction = mu
    dyn.Iz = Iz
 

    idx = np.array([3, 4, 5])  # vx, vy, omega (상태 순서 가정)
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
    bounds,
    grid_sizes,        # {"B":nB, "C":nC, "mu":nmu, "Iz":nIz}
    n_workers=8,
    chunk_size=256,    # 너무 작은 작업 오버헤드 줄이기
    verbose_every=5000
):
    x_true_prev = _to_2d(x_true_prev).astype(np.float64, copy=False)
    x_u = _to_2d(x_u).astype(np.float64, copy=False)
    x_true = _to_2d(x_true).astype(np.float64, copy=False)

    track_tmp = Track(track_dir, track_file=track_file, traj_type=traj_type)
    s_inds = _precompute_s_inds(track_tmp.s, len(track_tmp.s_center), x_true_prev[:, 0])

    # 그리드 생성
    B_grid  = np.linspace(bounds["B"][0],  bounds["B"][1],  grid_sizes["B"])
    C_grid  = np.linspace(bounds["C"][0],  bounds["C"][1],  grid_sizes["C"])
    mu_grid = np.linspace(bounds["mu"][0], bounds["mu"][1], grid_sizes["mu"])
    Iz_grid = np.linspace(bounds["Iz"][0], bounds["Iz"][1], grid_sizes["Iz"])

    # 전체 조합 iterator (메모리 폭발 방지: 리스트로 만들지 않음)
    combos = itertools.product(B_grid, C_grid, mu_grid, Iz_grid)

    best = {"sse": np.inf, "params": None}
    n_eval = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(track_dir, track_file, traj_type, dt, N,
                  x_true_prev, x_u, x_true, s_inds)
    ) as pool:

        # chunk 단위로 submit해서 scheduler 오버헤드 줄이기
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
                    print(f"[brute {n_eval}] best SSE={best['sse']:.6g} | params={best['params']}")

    return best
if __name__ == "__main__":
    # ---- load data ----
    data = np.load(os.path.join("./MAP0/GP", "Data", f"traj_info_{0}.npz"), allow_pickle=True)

    # prev residual (파일의 x_pred 기반)
    x_true_345 = data['x_true'][2:, 3:]
    x_pred_345 = data['x_pred'][1:, 3:]
    x_u = data['u'][1:]

    y_all = [(x_true_345[:,i] - x_pred_345[:,i]).reshape(-1,1) for i in range(3)]
    y_np_prev = np.concatenate(y_all, axis=0)

    T = x_true_345.shape[0]
    y_vx_prev = y_np_prev[0:T, 0]
    y_vy_prev = y_np_prev[T:2*T, 0]
    y_om_prev = y_np_prev[2*T:3*T, 0]

    # ---- brute force uses full state for x_true/x_prev ----
    x_true = data['x_true'][2:, :]       # (T,6)
    x_true_prev = data['x_true'][1:-1, :]# (T,6)
    x_u = data['u'][1:]                  # (T,2) or similar

    # ---- track/dynamics ----
    map_dir = "MAP0"
    track_dir = map_dir + "/Traj/"
    file_name = "optimized_traj"
    track_file = f"{file_name}{0}"
    traj_type = "opt_traj"
    dt = 0.1
    N = 15
    track = Track(track_dir, track_file=track_file, traj_type=traj_type)

    bounds = {
        "B":  (1.0, 1.2),
        "C":  (1.1, 1.3),
        "mu": (0.7, 0.9),
        "Iz": (0.014, 0.024),
    }

    # 그리드 해상도: 조합 수 = nB*nC*nmu*nIz
    grid_sizes = {"B": 10, "C": 10, "mu": 10, "Iz": 10}  # 예: 11*11*9*11 = 11979

    best = brute_force_parallel(
        x_true_prev=x_true_prev,
        x_u=x_u,
        x_true=x_true,
        track_dir=track_dir,
        track_file=track_file,
        traj_type=traj_type,
        dt=dt,
        N=N,
        bounds=bounds,
        grid_sizes=grid_sizes,
        n_workers=8,
        chunk_size=256,
        verbose_every=2000,
    )

    B, C, mu, Iz = best["params"]
    print("\nBEST (BRUTE FORCE):")
    print("  SSE:", best["sse"])
    print("  params [B, C, mu, Iz]:", best["params"])

    # ---- compute tuned residual with best params ----
    dyn = VehicleDynamics(track, dt=dt, N=N)
    dyn.B = float(B); dyn.C = float(C); dyn.wheel_friction = float(mu); dyn.Iz = float(Iz)
    dyn_true= VehicleDynamicsTrue(track, dt=dt, N=N)

    s_inds = _precompute_s_inds(track.s, len(track.s_center), np.asarray(x_true_prev)[:, 0])

    res = []
    res_true=[]
    for i in range(len(x_true_prev)):
        curv = track.curv_center[s_inds[i]]
        x_pred = dyn.f_d_rk4_time(x_true_prev[i], x_u[i], curv)
        x_True = dyn_true.f_d_rk4_time(x_true_prev[i], x_u[i], curv)
        res.append(np.asarray(x_true[i, :] - x_pred).reshape(1, -1))
        res_true.append(np.asarray(x_true[i, :] - x_True).reshape(1, -1))
    print(dyn_true.B,dyn_true.C,dyn_true.wheel_friction,dyn_true.Iz)
        
    res_mat = np.vstack(res)  # (T,6)
    res_true_mat=np.vstack(res_true)

    vx_idx, vy_idx, om_idx = 3, 4, 5
    y_vx = res_mat[:, vx_idx]
    y_vy = res_mat[:, vy_idx]
    y_om = res_mat[:, om_idx]
    y_vx_true = res_true_mat[:, vx_idx]
    y_vy_true = res_true_mat[:, vy_idx]
    y_om_true = res_true_mat[:, om_idx]

    # ---- plot: tuned vs prev ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(y_vx,      linestyle='-',  linewidth=1.8, label='tuned (brute)')
    axes[0].plot(y_vx_prev, linestyle='--', linewidth=1.5, label='initial (prev)')
    axes[0].plot(y_vx_true, linestyle='--', linewidth=2.1, label='initial (true)')
    axes[0].set_title("Residual: vx (x_true - x_pred)")
    axes[0].set_ylabel("vx resid")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(y_vy,      linestyle='-',  linewidth=1.8, label='tuned (brute)')
    axes[1].plot(y_vy_prev, linestyle='--', linewidth=1.5, label='initial (prev)')
    axes[1].plot(y_vy_true, linestyle='--', linewidth=2.1, label='initial (true)')
    axes[1].set_title("Residual: vy (x_true - x_pred)")
    axes[1].set_ylabel("vy resid")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    axes[2].plot(y_om,      linestyle='-',  linewidth=1.8, label='tuned (brute)')
    axes[2].plot(y_om_prev, linestyle='--', linewidth=1.5, label='initial (prev)')
    axes[2].plot(y_om_true, linestyle='--', linewidth=1.5, label='initial (true)')
    axes[2].set_title("Residual: omega (x_true - x_pred)")
    axes[2].set_xlabel("Timestep (0 ... T-1)")
    axes[2].set_ylabel("omega resid")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.show()
