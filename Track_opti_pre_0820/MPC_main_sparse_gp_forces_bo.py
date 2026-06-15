#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from Track.track import Track
from Optimization.MPC_Forces_pro_dyn import MPC
from Optimization.dynamics import *
from Optimization.params import *
from Optimization.model_params import *
from gp_utils import *
import warnings
warnings.filterwarnings('ignore')


import _pickle as pickle


'''initialize the track and node'''


def MPCmain_bo(iteration, gp_iter = 0, gp_train = False, map_dir = './MAP', file_dir='/Bayes/', file_name = 'optimized_traj', return_ = False, plot_ = False, label= None):

    track_dir = map_dir + "/Traj/"
    if file_name == 'optimized_traj' or file_name == 'optimized_true_traj' :
        track = Track(track_dir, track_file=f'{file_name}{0}', traj_type='opt_traj')
    else:
        track = Track(track_dir, track_file=f'{file_name}{iteration}', traj_type='opt_traj')

    dt = 0.1
    N = 15
    dynamics_type = 'Dyn'
    if dynamics_type == 'Kin':
        dynamics = VehicleKinematics(track, dt=dt, N=N)
        mpc_prams = mpc_prams_kin
        x0 = [0]*(dynamics.n_q+2)

    elif dynamics_type == 'Dyn':
        dynamics = VehicleDynamics(track, dt=dt, N=N)
        mpc_prams = mpc_prams_dyn
        x0 = [0]*(dynamics.n_q)


    # Intialize the states

    x0[0], x0[1], x0[2], x0[3] = 0.0, track.ey[0], 0.0, track.v[0]
    u_mpc = np.array([0,0])

    if gp_train:
        state_names = ['vx', 'vy', 'omega']
        
        # with open(f'{map_dir}{file_dir}Data/gp_iter{iteration-1}_{state_names[0]}.pickle', 'rb') as f:
        #         (vxmodel, vxxscaler, vxyscaler, gpmodel) = pickle.load(f)

        with open(f'{map_dir}{file_dir}/Data/gp_iter{gp_iter-1}_{state_names[1]}.pickle', 'rb') as f:
            (vymodel, vyxscaler, vyyscaler, gpmodel) = pickle.load(f)
        
        with open(f'{map_dir}{file_dir}/Data/gp_iter{gp_iter-1}_{state_names[2]}.pickle', 'rb') as f:
            (omegamodel, omegaxscaler, omegayscaler, gpmodel) = pickle.load(f)


        if gpmodel == 'fullgp':
            # vxgp = loadGPModel(state_names[0], vxmodel, vxxscaler, vxyscaler)
            vygp = loadGPModel(state_names[1], vymodel, vyxscaler, vyyscaler)
            omegagp = loadGPModel(state_names[2], omegamodel, omegaxscaler, omegayscaler)

        elif gpmodel == 'sparsegp':

            # vx_Z = vymodel["Z"]                            # (M, D)
            # vx_alpha = vymodel["alpha"]          
            # vx_Kmm_inv = vymodel["Kmm_inv"]             # (M, 1)
            # vx_lengthscale = vymodel["lengthscale"]        # (D,) or scalar
            # vx_variance = vymodel["variance"] 
        
            vy_Z = vymodel["Z"]                            # (M, D)
            vy_alpha = vymodel["alpha"]          
            vy_Kmm_inv = vymodel["Kmm_inv"]             # (M, 1)
            vy_lengthscale = vymodel["lengthscale"]        # (D,) or scalar
            vy_variance = vymodel["variance"]              # scalar

            omega_Z = omegamodel["Z"]                     
            omega_alpha = omegamodel["alpha"]       
            omega_Kmm_inv = omegamodel["Kmm_inv"]             
            omega_lengthscale = omegamodel["lengthscale"]       
            omega_variance = omegamodel["variance"]             

            # vxgp = loadSparseGPModel('vx',vx_Z,vx_alpha,vx_Kmm_inv,vx_variance,vx_lengthscale, vxxscaler ,vxyscaler)
            vygp = loadSparseGPModel('vy',vy_Z,vy_alpha,vy_Kmm_inv,vy_variance,vy_lengthscale, vyxscaler ,vyyscaler)
            omegagp = loadSparseGPModel('omega',omega_Z,omega_alpha,omega_Kmm_inv,omega_variance,omega_lengthscale, omegaxscaler, omegayscaler)
        else:
            print("GP model type is error")
        
        gpmodels = {
            # 'vx' : vxgp,
            'vy': vygp,
            'omega': omegagp,
            'xscaler': vyxscaler,
            }
        
        if label is not None:
            opt = MPC(dynamics, track, mpc_prams, x0, dir = map_dir, iteration=gp_iter, gpmodels = gpmodels, label = label)
        else:
            opt = MPC(dynamics, track, mpc_prams, x0, dir = map_dir, iteration=gp_iter, gpmodels = gpmodels)
        
        # vxgpfull = vxgp
        vygpfull = vygp
        omegagpfull = omegagp
        xsm = vyxscaler.mean_
        xss = vyxscaler.scale_
    else:
        opt = MPC(dynamics, track, mpc_prams, x0, dir = map_dir,)
    

    lap_time = 0.0
    lap_time_arr = []
    lap_number = 0

    traj_pred = []
    traj_true = []
    traj_true.append(x0)
    u = []
    
    fail = False
    while(1):

        if gp_train and lap_time >= dt:
            x_mpc, u_mpc, ref = opt.solve_optimization_gp_dyn(x0, u_mpc[0])
        else:
            x_mpc, u_mpc, ref = opt.solve_optimization_gp_dyn(x0, np.array([0,0]))
        
        # Predicted Trajectory
        pred_ = np.zeros((len(x_mpc),3))
        for i in range(0,len(x_mpc)):
            pred_[i, 0], pred_[i, 1], pred_[i, 2] = track.local_to_global(np.array([x_mpc[i,0], x_mpc[i,1], x_mpc[i,2]]))
        
        # Reference 
        ref_ = np.zeros((len(ref),3))
        for i in range(0,len(ref_)):
            ref_[i,0], ref_[i,1], ref_[i,2] = track.local_to_global(np.array([ref[i,0], ref[i,1], ref[i,2]]))

        # Dynamic Update
        s_ind = np.where(track.s>=x0[0])[0]
        if len(s_ind) >= 1:
            s_ind = s_ind[0]
            if s_ind >= len(track.s_center):
                s_ind -= len(track.s_center)
        else:
            s_ind = 0

        error = np.zeros((6,))
        if gp_train == True:
            gp_input = x0[3:] + list(u_mpc[0])
            gpinput_ = (gp_input - xsm) / xss
            error[4] = vygpfull(gpinput_)[0]
            error[5] = omegagpfull(gpinput_)[0]

        x_next = dynamics.f_d_rk4_time(x0, u_mpc[0],track.curv_center[s_ind]) + error
        # x_next_ = dynamics_true.f_d_rk4_time(x0, u_mpc[0],track.curv_center[s_ind])
        # x0_ = [x_next_[0].__float__(), x_next_[1].__float__(), x_next_[2].__float__(),x_next_[3].__float__(),x_next_[4].__float__(),x_next_[5].__float__()]

        x0 = [x_next[0].__float__(), x_next[1].__float__(), x_next[2].__float__(),x_next[3].__float__(),x_next[4].__float__(),x_next[5].__float__()]
        
        # if np.sum( abs(np.array(x0)-np.array(x0_)) ) >= 0.1:
        #     print(np.array(x0)-np.array(x0_))
        
        x0[3] = np.clip(x0[3], 0.0, mpc_prams.vx_max)
        
        
        if plot_:
            fig = plt.figure(100)
            plt.clf()
            plt.plot(track.opt[:,0], track.opt[:,1],'--r', label='optimal line')
            plt.plot(track.inner[:,0], track.inner[:,1], 'k')
            plt.plot(track.outer[:,0], track.outer[:,1], 'k')
            plt.plot(pred_[:,0], pred_[:,1], 'g', label='planned')
            plt.plot(pred_[0,0], pred_[0,1], 'bo', label='start')
            plt.plot(ref_[:,0], ref_[:,1], 'y.', label='reference')
            plt.legend()
            plt.pause(0.001)

        lap_time += dt    
        if lap_number >= 2:  
            break
        elif lap_number >= 1:  
            break
            # traj_true.append(x0)
            # u.append(u_mpc[0])
        else:
            traj_true.append(x0)
            u.append(u_mpc[0])

        if x0[0] >= track.track_length:
            x0[0] = x0[0] - track.track_length
            lap_number += 1
            lap_time_arr.append(lap_time)
            lap_time = 0.0
        
        if lap_time >= 30:
            print("Soveler is failed")
            lap_time_arr.append(lap_time)
            fail = True
            break

    
    # LAP 1
    traj_true = np.array(traj_true)
    traj_pred = np.array(traj_pred)
    u = np.array(u)

    if not fail:
        traj_true_xy, traj_true_fren = track.interpolate(traj_true)

        # plt.figure(2)
        # plt.clf()
        # plt.plot(track.inner[:,0], track.inner[:,1], 'k')
        # plt.plot(track.outer[:,0], track.outer[:,1], 'k')
        # plt.plot(track.opt[:,0], track.opt[:,1],'--r')
        # plt.plot(traj_true_xy[1:,0], traj_true_xy[1:,1],'b')
        # plt.draw()
        # plt.savefig(f"{file_dir}Figure/botraj_iter{iteration}.png", dpi=300, bbox_inches='tight')

        np.savez(f'{map_dir}{file_dir}MPC/botraj_iter{iteration}.npz', 
                    traj_true = traj_true_xy,
                    traj_true_fren = traj_true_fren,
                    time_arr = lap_time_arr
                    )

        sdot  = (traj_true_fren[:,3]* np.cos(traj_true_fren[:,2]) - traj_true_fren[:,4] * np.sin(traj_true_fren[:,2])) / (1 - traj_true_fren[:,1]* track.curv_center)
        act_dt = track.d_dist/sdot
        print('Actual Lap Time', lap_time_arr[-1])
        lap_time2 = np.cumsum(act_dt)[-1]
    else:
        traj_true_xy = [] 
        traj_true_fren = []
        lap_time2 = 30.0

    return traj_true_xy, traj_true_fren, lap_time2, lap_time_arr[-1]

# MPCmain(0, gp_train=False, file_dir='./GP/')
