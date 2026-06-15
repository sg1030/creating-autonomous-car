import numpy as np
from init import *
from MPC_main_sparse_gp_forces import *
from train_model import *
from train_model_gpflow import *
import warnings
warnings.filterwarnings('ignore')

lap_time_arr = []
lap_time_arr2 = []

lap_time_arr_sdot = []
lap_time_arr2_sdot = []

collision = []
collision2 = []
true_plannedtime = SolveMinimumLapTime(model = 'DynTrue', iter = 0)
_, _, true_lap_time2, true_lap_time, truecol =  MPCmain_forces(-1, gp_train=False,  file_name = 'optimized_true_traj', file_dir='./GP/', return_ = True, plot_ = True, label = 'DynTrue_GP')


data_len = 0
defined_plannedtime = SolveMinimumLapTime(model = 'Dyn', iter = 0)
_, _, define_lap_time2, define_lap_time, definecol = MPCmain_forces(-1, gp_train=False,  file_dir='./GP/', return_ = True, plot_ = True, label = 'DynTrue_GP')
_, _, nogplap_time2, nogplap_time, nogpcol = MPCmain_forces(0, gp_train=False,  file_dir='./GP/', return_ = True, plot_ = True, label = 'DynGP')
data= np.load(f'./GP/Data/traj_info_{0}.npz', allow_pickle=True)
data_len += len(data['x_true'])-2
lap_time_arr.append(nogplap_time)
lap_time_arr_sdot.append(nogplap_time2)
collision.append(nogpcol)
for itr in range(1, 5):
    if data_len >= 200:
        print(".....Sparse GP Training.....")
        trainMain_sparse(itr)
    else:
        print(".....GP Training.....")
        trainMain(itr)
    
    traj, _, lap_time2, lap_time,col = MPCmain_forces(itr, gp_train=True,  file_name = 'optimized_traj', file_dir='./GP/', return_ = True, plot_ = True, label = 'DynGP')
    lap_time_arr.append(lap_time)
    lap_time_arr_sdot.append(lap_time2)
    collision.append(col)

    data= np.load(f'./GP/Data/traj_info_{itr}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    print("Collision: ", col)

doublegp_time = []
data_len = 0
data= np.load(f'./GP/Data/traj_info_{0}.npz', allow_pickle=True)
data_len += len(data['x_true'])-2
lap_time_arr2.append(nogplap_time)
lap_time_arr2_sdot.append(nogplap_time2)
collision2.append(nogpcol)
for itr in range(1, 5):
    if data_len >= 200:
        print(".....Sparse GP Training.....")
        trainMain_sparse(itr)
    else:
        print(".....GP Training.....")
        trainMain(itr)
    plannedtime = SolveMinimumLapTime(model = 'DynGP', iter = itr)
    doublegp_time.append(plannedtime)
    _, _, lap_time2, lap_time,col = MPCmain_forces(itr, gp_train=True,  file_name = 'optimized_gp_traj', file_dir='./GP/', return_ = True, plot_ = True, label = 'DoubleGP')
    lap_time_arr2.append(lap_time)
    lap_time_arr2_sdot.append(lap_time2)
    collision2.append(col)
    data= np.load(f'./GP/Data/traj_info_{itr}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    print("Collision: ", col)

print("--------------------")
print(f"True expected lap time {true_plannedtime}s with collision {truecol}" )
print("True lap time", true_lap_time)
print("True lap time2", true_lap_time2)
print("--------------------")
print(f"Expected Lap time with defined model {defined_plannedtime}")
print(f"Lap time with defined model {define_lap_time}s with collision {definecol}")
print(f"Lap time with defined model {define_lap_time2}s with collision {definecol}")
print("Lap time with  GP", lap_time_arr)
print("Lap time2 with  GP", lap_time_arr_sdot)
print("Collision of GP", collision)
print("--------------------")
print("Expected Lap time with iterative GP", doublegp_time)
print("Lap time with iterative GP", lap_time_arr2)
print("Lap time2 with iterative GP", lap_time_arr2_sdot)
print("Collision of iterative GP", collision2)



