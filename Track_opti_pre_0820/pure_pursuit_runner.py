#!/usr/bin/env python3


from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from Track.track import Track
from Optimization.dynamics import VehicleDynamicsTrue, VehicleDynamics
from Optimization.params import *
from Optimization.model_params import *
import math

# ---------------------------- Utilities ----------------------------
mpc_prams = mpc_prams_dyn
def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    a = (a + np.pi) % (2*np.pi) - np.pi
    return a

def pure_pursuit_control(
    xy: np.ndarray, psi: float, path_xy: np.ndarray,
    idx_near: int, Ld: float, delta_max: float,
    L_wb: float = 0.5,           
    return_target: bool = False
):
    N = len(path_xy)
    i_target = idx_near
    dist = 0.0
    while dist < Ld and i_target < idx_near + N - 1:
        j  =  i_target      % N
        jn = (i_target + 1) % N
        seg = np.linalg.norm(path_xy[jn] - path_xy[j])
        dist += seg
        i_target += 1
    i_target = i_target % N

    # 2) target point in vehicle frame
    pt = path_xy[i_target]
    dx = pt[0] - xy[0]
    dy = pt[1] - xy[1]
  
    psi=wrap_angle(psi)
    
    alpha = wrap_angle(math.atan2(dy, dx) - psi)
    alpha=wrap_angle(alpha)
    # Reverse steering angle when reversing
    delta = math.atan2(2.0 * L_wb * math.sin(alpha) / Ld,1.0)



    delta = np.clip(delta, -delta_max, delta_max)


    return delta


def nearest_index(xy: np.ndarray, path_xy: np.ndarray) -> int:
    d = np.linalg.norm(path_xy - xy.reshape(1,2), axis=1)
    return int(np.argmin(d))

def to_numpy_vec(v):
    """CasADi DM/SX or numpy -> 1D numpy float array."""
    if hasattr(v, "full"):           # DM: has .full()
        return v.full().ravel()
    try:
        return np.array(v, dtype=float).ravel()
    except Exception:
        # SX/MX fallback (rare here): index-wise cast
        return np.array([float(v[i]) for i in range(v.shape[0])], dtype=float)

# -------------------------- Main Runner ----------------------------
def run_pure_pursuit(iteration: int,
                     map_dir: str = './MAP',
                     file_name: str = 'optimized_true_traj',
                     file_iter: int | None = None,
                     plot: bool = False,
                     label: str | None = 'PP',
                     dt: float = 0.1,
                     L0: float = 0.0,
                     kLd: float = 0.5,
                     delta_max_deg: float = np.radians(30),
                     ax_max: float = 3.0,
                     ax_min: float = -5.0,
                     vx_gain: float = 0.8):
   
    track_dir = './MAP0/Traj/'

    track = Track(track_dir, track_file=f'optimized_traj{0}', traj_type='opt_traj')
   

    # Plant model (true dynamics)
    dynamics_true = VehicleDynamicsTrue(track, dt=dt, N=1)
    dynamics = VehicleDynamics(track, dt=dt, N=1)
    mpc_prams = mpc_prams_dyn

    # Initial state in Frenet: [s, ey, epsi, vx, vy, omega]
    x = np.zeros(6, dtype=float)
    x[0] = 0.0
    x[1] = track.ey[0] if hasattr(track, 'ey') else 0.0
    x[2] = 0.0
    x[3] = float(getattr(track, 'v', [5.0])[0])

    # Extract global path (use optimal line if available, else center)
    if hasattr(track, 'opt') and track.opt is not None and track.opt.shape[1] >= 2:
        path_xy = track.opt[:, :2].copy()
    else:
        path_xy = track.center[:, :2].copy()
    Np = len(path_xy)

    # Logging
    t = 0.0
    lap_time = 0.0
    lap_number = 0
    lap_time_arr = []

    xs_true = [x.copy()]
    xs_pred = []
    us = []
    xy_log = []
    xy_ref_log = []
    vx_ref_log = []

    delta_max = delta_max_deg
    s=0
    max_time = 120.0  
    while t < max_time:
        xg, yg, psi = track.local_to_global(np.array([x[0], x[1], x[2]]))
        xy = np.array([xg, yg])
        idx_near = nearest_index(xy, path_xy)

 
        v_ref = float(track.v[idx_near])


        Ld = float(L0 + kLd * max(x[3], 0.0))

        delta = pure_pursuit_control(xy, psi, path_xy, idx_near, Ld, delta_max)


        ax_cmd = np.clip(vx_gain * (v_ref - x[3]), ax_min, ax_max)
        u = np.array([ax_cmd, delta], dtype=float)
        s_ind = np.searchsorted(track.s, x[0], side='left')
        if s_ind >= len(track.s_center):
            s_ind = len(track.s_center) - 1
        curv = track.curv_center[s_ind]

        x_next = dynamics_true.f_d_rk4_time(x, u, curv)
        x_pred = dynamics.f_d_rk4_time(x, u, curv)
       

        x_next_np = to_numpy_vec(x_next)
        x_pred_np = to_numpy_vec(x_pred)

        if x_next_np[0] >= track.track_length:
            x_next_np[0] -= track.track_length
            

        x = x_next_np
        x[3] = np.clip(x[3], 0.0, vx_max)

        xs_true.append(x.copy())
        xs_pred.append(x_pred_np.copy())
        us.append(u.copy())
        xy_log.append(xy.copy())
        xy_ref_log.append(path_xy[idx_near].copy())
        vx_ref_log.append(v_ref)
        t += dt
        lap_time += dt
        
        if x[0] >= track.track_length or s-x[0]>0 :
            print('turned')
            x[0] -= track.track_length
            lap_number += 1
            lap_time_arr.append(lap_time)
            lap_time = 0.0
        if lap_number >= 1:
            break
        s=x[0]

        plot=True
        if plot: 
            plt.figure(1, figsize=(6,5)); plt.clf()
            plt.plot(track.inner[:,0], track.inner[:,1], 'k')
            plt.plot(track.outer[:,0], track.outer[:,1], 'k')
            if hasattr(track, 'opt'):
                plt.plot(track.opt[:,0], track.opt[:,1], '--r', label='optimal')
            plt.plot(np.array(xy_log)[:,0], np.array(xy_log)[:,1], 'b', label='PP traj')
            plt.plot(path_xy[:,0], path_xy[:,1], 'g.', alpha=0.3, label='ref path')
            plt.plot(path_xy[idx_near,0], path_xy[idx_near,1], 'r.', alpha=0.3, label='target')
            plt.legend(); plt.axis('equal'); plt.pause(0.001)

    xs_true = np.array(xs_true)
    xs_pred = np.array(xs_pred)
    us = np.array(us)
    xy_log = np.array(xy_log)

    out_dir_data = os.path.join(map_dir, 'GP', 'Data')
    out_dir_fig  = os.path.join(map_dir, 'GP', 'Figure')
    out_dir_mpc  = os.path.join(map_dir, 'GP', 'MPC')
    os.makedirs(out_dir_data, exist_ok=True)
    os.makedirs(out_dir_fig, exist_ok=True)
    os.makedirs(out_dir_mpc, exist_ok=True)

    try:
        traj_true_xy, traj_true_fren = track.interpolate(xs_true)
        traj_pred_xy, traj_pred_fren = track.interpolate(xs_pred)
    except Exception:
        traj_true_xy = xy_log
        traj_true_fren = xs_true

    plt.figure(102); plt.clf()
    plt.plot(track.inner[:,0], track.inner[:,1], 'k')
    plt.plot(track.outer[:,0], track.outer[:,1], 'k')
    if hasattr(track, 'opt'):
        plt.plot(track.opt[:,0], track.opt[:,1], '--r')
    plt.plot(traj_true_xy[1:,0], traj_true_xy[1:,1], 'b')
    plt.axis('equal'); plt.title('Pure Pursuit Trajectory')
    plt.savefig(f"{out_dir_fig}/traj_iter{iteration}_{label}.png", dpi=300, bbox_inches='tight')



    np.savez(f'./MAP0/GP/Data/traj_info_{iteration}.npz',
        
        x_true=xs_true, x_pred=xs_pred,
        u=us
    )
    print(traj_true_xy.shape)
    print(traj_pred_xy.shape)
    # np.savez(f'./MAP0/GP/Data/traj_info_{iteration}.npz',
    #          traj_true=traj_true_xy,
    #          traj_pred=traj_pred_xy,
    #          traj_true_fren=traj_true_fren,
    #          traj_pred_fren=traj_pred_fren,
    #          time_arr=lap_time_arr)


    if len(lap_time_arr)>0:
        print(f'Pure Pursuit Lap Time: {lap_time_arr[-1]:.3f} s')
    else:
        print('Pure Pursuit finished without completing a full lap.')

    return traj_true_xy, traj_true_fren, (lap_time_arr[-1] if len(lap_time_arr)>0 else np.nan)


def main():
    mpc_prams = mpc_prams_dyn

    run_pure_pursuit(iteration=0,
                     map_dir='./MAP0',
                     file_name='./Traj',
                     file_iter='./GP',
                     plot=True,
                     L0=2.0, kLd=0.0,
                     delta_max_deg=mpc_prams_dyn.u_steer_max,
                     ax_max=mpc_prams_dyn.u_a_max, ax_min=mpc_prams_dyn.u_a_min,
                     vx_gain=0.8)
import time
if __name__ == '__main__':
    start_time=time.time()
    main()
    print(time.time()-start_time)

