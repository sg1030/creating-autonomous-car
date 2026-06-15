import numpy as np
from init import *
from MPC_main_sparse_gp_forces import *
from train_model import *
from train_model_gpflow import *
from bayes_opt_track_gp import *


trial_bo = 500

# SolveMinimumLapTime(model = 'Dyn', iter = 0)
lap_time_arr = []
lap_time_arr2 = []
collision = []

data_len = 0
_, _, lap_time2, lap_time, col = MPCmain_forces(0, gp_train=False, file_dir='./Bayes/', file_name = 'optimized_traj', return_ = True, plot_ = True, label = 'Dyn')
data= np.load(f'./Bayes/Data/traj_info_{0}.npz', allow_pickle=True)
data_len += len(data['x_true'])-2

lap_time_arr.append(lap_time)
lap_time_arr2.append(lap_time2)
collision.append(col)

for itr in range(1, 5):
    print("--------------------------------------------")
    print("Current iteration is", itr)
    if data_len >= 200:
        print(".....Sparse GP Training.....")
        trainMain_sparse(itr, file_dir = './Bayes/')
    else:
        print(".....GP Training.....")
        trainMain(itr, file_dir = './Bayes/')

    print(".....Bayesian Optimization.....")
    if itr == 1:
        file_name = 'optimized_traj'
        trackoptimization = TrackOptimization_bo(file_name = file_name, generate = True, gptrain = True, gpiter = itr, trial = trial_bo, model = 'ucb', weight = 0.1, label = 'ucb')
    else:
        file_name = 'bayes_traj_best'
        trackoptimization = TrackOptimization_bo(file_name = file_name, generate = True, gptrain = True, gpiter = itr, trial = trial_bo, model = 'ucb', weight = 0.1, label = 'ucb')
    trackoptimization.select_points()
    print(".....MPC.....")
    _, _, lap_time2, lap_time, col =MPCmain_forces(itr, gp_train=True, file_dir='./Bayes/', file_name = 'bayes_traj_best', return_ = True, plot_ = True, label = 'true_ucb')
    lap_time_arr.append(lap_time)
    lap_time_arr2.append(lap_time2)
    collision.append(col)

    data= np.load(f'./Bayes/Data/traj_info_{itr}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    print("Collision: ", col)

    print("--------------------------------------------")

print(lap_time_arr)
print(lap_time_arr2)
print(col)


