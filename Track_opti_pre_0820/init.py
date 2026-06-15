import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from Track.track import Track
from Optimization.track_opt import MiminumTimeOptimization
from Optimization.dynamics import VehicleDynamics, VehicleKinematics, VehicleDynamicsTrue
from Optimization.params import *
from Optimization.model_params import *
from gp_utils import *
import _pickle as pickle

class TrackOptimization():
    def __init__(self, dir):
        '''initialize the track and node'''
        self.track_dir = f"{dir}/Traj/"
        self.dir = dir
        self.Ns = 300
        self.track = Track(self.track_dir)
        self.track.linspace_s(N=self.Ns)

        self.opt_traj = None
    
    
    def track_opt(self, model, iter = 0, model_param=None):

        if model == 'Dyn':
            dynamics = VehicleKinematics(self.track, dt=0.1, N=self.Ns) 
            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_kin, 'Kin')

            x_pred, u_pred = opt.solve_optimization()
            dyn_state = np.zeros((len(x_pred), 3))
            dyn_state[:, -1] = x_pred[:, -1]

            dynamics = VehicleDynamics(self.track, dt=0.1, N=self.Ns) 
            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_dyn, model)
            x_pred, u_pred = opt.solve_optimization(np.concatenate([x_pred[:, :-1], dyn_state], axis=1), u_pred)

        elif model == 'DynTrue':
            dynamics = VehicleKinematics(self.track, dt=0.1, N=self.Ns) 
            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_kin, 'Kin')

            x_pred, u_pred = opt.solve_optimization()
            dyn_state = np.zeros((len(x_pred), 3))
            dyn_state[:, -1] = x_pred[:, -1]

            dynamics = VehicleDynamicsTrue(self.track, dt=0.1, N=self.Ns) 
            if model_param is not None:
                # dynamics.m = model_param["mass"]
                dynamics.wheel_friction = model_param["wheel_friction"]
                # dynamics.L_f = model_param["L_f"]
                # dynamics.L_r = model_param["L_r"]
                dynamics.C = model_param["C"]
                dynamics.B = model_param["B"]
                dynamics.Iz = model_param["Iz"]

            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_dyn, model)
            x_pred, u_pred = opt.solve_optimization(np.concatenate([x_pred[:, :-1], dyn_state], axis=1), u_pred)
        
        elif model == 'DynGP':
            
            self.track = Track(self.track_dir, track_file='optimized_traj0', traj_type='opt_traj')
            dynamics = VehicleDynamics(self.track, dt=0.1, N=self.Ns) 
            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_dyn, model)

            state_names = ['vx', 'vy', 'omega']

            # with open(f'{self.dir}/GP/Data/gp_iter{iter-1}_{state_names[0]}.pickle', 'rb') as f:
            #     (vxmodel, vxxscaler, vxyscaler, gpmodel) = pickle.load(f)

            with open(f'{self.dir}/GP/Data/gp_iter{iter-1}_{state_names[1]}.pickle', 'rb') as f:
                (vymodel, vyxscaler, vyyscaler, gpmodel) = pickle.load(f)
            
            with open(f'{self.dir}/GP/Data/gp_iter{iter-1}_{state_names[2]}.pickle', 'rb') as f:
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

            # if iter == 0:
            #     track = Track("./Traj/", track_file='optimized_traj0', traj_type='opt_traj')
            # else:
            #     track = Track("./Traj/", track_file=f'optimized_gp_traj{iter}', traj_type='opt_traj')

            opt_traj = self.track.opt_traj
            gp_input = opt_traj[:,3:-1]
            error = np.zeros((len(gp_input), 6))
            xsm = vyxscaler.mean_
            xss = vyxscaler.scale_
            for i in range(len(gp_input)):
                gpinput_ = (gp_input[i] - xsm) / xss

                # error[i,2] = vxgp(gpinput_)[0]
                error[i,3] = vygp(gpinput_)[0]*vyyscaler.scale_ + vyyscaler.mean_
                error[i,4] = omegagp(gpinput_)[0]*omegayscaler.scale_ + omegayscaler.mean_

            x_pred = np.zeros((len(gp_input)+1, 6)) 
            x_pred[:-1,:-1] = opt_traj[:,1:6]
            x_pred[:-1, -1] = np.cumsum(self.track.d_dist/opt_traj[:,-1])
            x_pred_ = dynamics.f_d_rk4(x_pred[-2], opt_traj[-1,6:8], self.track.s_center[1]-self.track.s_center[0], self.track.curv_center[-1])
            x_pred[-1] = [x_pred_[0].__float__(),x_pred_[1].__float__(),x_pred_[2].__float__(),x_pred_[3].__float__(),x_pred_[4].__float__(),x_pred_[5].__float__()]

            u_pred = opt_traj[:,6:8]
            
            x_pred, u_pred = opt.solve_optimization(x_ws = x_pred, u_ws = u_pred, error = error)

        elif model == 'Kin':
            dynamics = VehicleKinematics(self.track, dt=0.1, N=self.Ns) 
            opt = MiminumTimeOptimization(dynamics, self.track, opt_params_kin,model)

            x_pred, u_pred = opt.solve_optimization()
        else:
            print('Wrong Dynamic Model')

        if model == 'Kin':
            sdot = x_pred[:-1,2] * np.cos(x_pred[:-1,1]) / (1 - x_pred[:-1,0]* self.track.curv_center)
        else:
            sdot = (x_pred[:-1,2] * np.cos(x_pred[:-1,1]) - x_pred[:-1,3] * np.sin(x_pred[:-1,1])) / (1 - x_pred[:-1,0]* self.track.curv_center)

        dt = 1/sdot*self.track.d_dist
        print("optimal:", np.cumsum(dt)[-1])

        self.opt_traj_fren = np.zeros((len(x_pred)-1,x_pred.shape[1]+3)) # state + u
        self.opt_traj_fren[:, 0] = self.track.s_center
        self.opt_traj_fren[:, 1:x_pred.shape[1]] = x_pred[:-1,:-1]
        self.opt_traj_fren[:, x_pred.shape[1]:-1] = u_pred
        self.opt_traj_fren[:, -1] = sdot
        x = np.zeros((len(self.track.s_center),))
        y = np.zeros((len(self.track.s_center),))
        psi = np.zeros((len(self.track.s_center),))
        for i in range(len(self.track.s_center)):
            x[i], y[i], psi[i] = self.track.local_to_global(np.array([self.track.s_center[i], x_pred[i,0], x_pred[i,1]]))

        # Plot the otimized trajectory
        fig = plt.figure(100)
        plt.clf()
        ax = plt.gca()
        ax.axis('equal')
        points = np.zeros((len(self.track.s_center), 1, 2))
        points[:,0,0], points[:,0,1] = x, y 
        speed = x_pred[:-1,2] 
        plt.plot(self.track.center[:,0], self.track.center[:,1],'--k')
        plt.plot(self.track.inner[:,0],self.track.inner[:,1], 'k')
        plt.plot(self.track.outer[:,0],self.track.outer[:,1], 'k')
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(min(speed), max(speed))
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(speed)
        lc.set_linewidth(5)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        plt.draw()
        plt.pause(0.1)
        plt.savefig(f"{self.dir}/GP/Figure/optimal_traj_{iter}_{model}.png", dpi=300, bbox_inches='tight')
        
            
        if model == 'Kin':
            self.laptime = x_pred[-1, 3]
        else:
            self.laptime = x_pred[-1,-1]
            
        print("Optimized Lap Time:", self.laptime)

        self.opt_traj = np.zeros((len(self.track.s_center),4))
        self.opt_traj[:,0] = x
        self.opt_traj[:,1] = y
        self.opt_traj[:,2] = psi
        self.opt_traj[:,3] = speed

def SolveMinimumLapTime(dir, model, iter, model_param = None):
    trackoptimization = TrackOptimization(dir)
    trackoptimization.track_opt(model=model, iter = iter, model_param = model_param) 

    if model == 'DynTrue':
        np.savetxt(f'{dir}/Traj/optimized_true_traj0.txt', trackoptimization.opt_traj, delimiter=",")
        np.savetxt(f'{dir}/Traj/optimized_true_traj0_frenet.txt', trackoptimization.opt_traj_fren, delimiter=",")
    elif model == 'DynGP':
        np.savetxt(f'{dir}/Traj/optimized_gp_traj{iter}.txt', trackoptimization.opt_traj, delimiter=",")
        np.savetxt(f'{dir}/Traj/optimized_gp_traj{iter}_frenet.txt', trackoptimization.opt_traj_fren, delimiter=",")
    else:
        np.savetxt(f'{dir}/Traj/optimized_traj0.txt', trackoptimization.opt_traj, delimiter=",")
        np.savetxt(f'{dir}/Traj/optimized_traj0_frenet.txt', trackoptimization.opt_traj_fren, delimiter=",")
    print('saved')

    return trackoptimization.laptime

# SolveMinimumLapTime('./MAP0', 'DynTrue', 0, model_param = None)