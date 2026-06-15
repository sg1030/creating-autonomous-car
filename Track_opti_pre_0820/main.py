import numpy as np
import warnings
warnings.filterwarnings('ignore')

from generate_rand_track import *
from utils import *
from init import *
from MPC_main_sparse_gp_forces import *
from train_model import *
from train_model_gpflow import *
from train_model_gpflow_torch import *
from bayes_opt_track_gp import *


NUM_ITERS_TRACK = 5
NUM_ITERS = 5
BOTRIAL = 300


true_plannedtime_arr = np.zeros((NUM_ITERS_TRACK,))
true_lap_time_arr = np.zeros((NUM_ITERS_TRACK,))
defined_plannedtime_arr = np.zeros((NUM_ITERS_TRACK,))
define_lap_time_arr= np.zeros((NUM_ITERS_TRACK,))
truecol_arr= np.zeros((NUM_ITERS_TRACK,))
definecol_arr = np.zeros((NUM_ITERS_TRACK,))

lap_time_arr = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
collision = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))

doublegp_time = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
doublegp_laptime = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
collision_doublegp = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))

bo_time = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
bo_laptime = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
collision_bo = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))



# for itr_t in range(NUM_ITERS_TRACK):

itr_t = 0
failure = False
while(1):
    
    ''' Create Radom Track'''
    print(f"[ITER {itr_t}] Generating track...")
    # center_ = generate_random_closed_track()
    # center = resample_centerline(center_[:-1])
    # outer, inner = compute_offset_curves(center, TRACK_WIDTH)

    TRACK_RADIUS = np.random.randint(6,8) 
    result = create_track(TRACK_RADIUS)
    if result is False:
        print(f"[{itr_t}] Invalid track, retrying...")
        continue
    center, inner, outer = result
    
    map_dir = f"./MAP{itr_t}"
    create_directory_structure(map_dir)
    save_track_txt(center, inner, outer, map_dir)
    plot_track(center, inner, outer, os.path.join(map_dir, "track.png"))

    true_dyn = {
            "C": 1.7 - np.random.rand()*0.4,
            "B": 1.4 - np.random.rand()*0.4,
            "Iz": 0.024 - np.random.rand()*0.01,
            "wheel_friction": 1.2 - np.random.rand()*0.4}
    print(true_dyn)
    np.savez(f"true_dyn_{itr_t}.npz", true_dyn = true_dyn)

    ''' Run '''
    true_plannedtime = SolveMinimumLapTime(dir = map_dir, model = 'DynTrue', iter = 0, model_param = true_dyn)
    true_plannedtime_arr[itr_t] = true_plannedtime
    _, _, _, true_lap_time, truecol =  MPCmain_forces(-1, gp_train=False,  file_name = 'optimized_true_traj', map_dir = map_dir, file_dir='/GP/', return_ = True, plot_ = True, label = f'MAP{itr_t}_DynTrue', model_param = true_dyn, failure = failure )
    true_lap_time_arr[itr_t] = true_lap_time
    truecol_arr[itr_t] = truecol
    print("Collision with true model", truecol)

    if true_lap_time >= 30:
        failure = True
        continue
    if truecol:
        failure = True
        continue
    failure = False

    data_len = 0
    defined_plannedtime = SolveMinimumLapTime(dir = map_dir, model = 'Dyn', iter = 0)
    defined_plannedtime_arr[itr_t] = defined_plannedtime
    _, _, _, define_lap_time, definecol = MPCmain_forces(-1, gp_train=False, map_dir = map_dir, file_dir='/GP/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynTrueGP', model_param = true_dyn, failure = failure)
    define_lap_time_arr[itr_t] = define_lap_time
    definecol_arr[itr_t] = definecol
    print("Collision with defined model", definecol)

    if true_lap_time >= 30:
        failure = True
        continue
    if define_lap_time <= true_plannedtime:
        failure = True
        continue
    
    failure = False
    ''' Double GP '''
    _, _, _, nogplap_time, nogpcol = MPCmain_forces(0, gp_train=False,  map_dir = map_dir,  file_dir='/DoubleGP/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynGP')
    data= np.load(f'{map_dir}/DoubleGP/Data/traj_info_{0}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    doublegp_laptime[itr_t, 0] = nogplap_time
    collision_doublegp[itr_t, 0]  = nogpcol
    
    for itr in range(1, NUM_ITERS):
        if data_len >= 200:
            print(".....Sparse GP Training.....")
            trainMain_sparse_torch(
            iteration=itr,
            file_dir=f"{map_dir}/DoubleGP",
            inducing_method="kmeans",  # "random"도 가능
            epochs=1500,
            batch_size=1024,
            lr=1e-2,
            verbose=True)
        else:
            print(".....GP Training.....")
            trainMain(itr, file_dir = f"{map_dir}/DoubleGP")
        plannedtime = SolveMinimumLapTime(dir = map_dir, model = 'DynGP', iter = itr)
        doublegp_time[itr_t, itr] = plannedtime
        _, _, _, lap_time, col = MPCmain_forces(itr, gp_train=True,  file_name = 'optimized_gp_traj', map_dir = map_dir, file_dir='/DoubleGP/',  return_ = True, plot_ = True, label =  f'MAP{itr_t}_DoubleGP')
        doublegp_laptime[itr_t, itr] = lap_time
        data= np.load(f'{map_dir}/DoubleGP/Data/traj_info_{itr}.npz', allow_pickle=True)
        data_len += len(data['x_true'])-2
        collision_doublegp[itr_t, itr] = col

    ''' GP '''
    data_len = 0
    _, _, _, nogplap_time, nogpcol = MPCmain_forces(0, gp_train=False,  map_dir = map_dir,  file_dir='/GP/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynGP')
    data= np.load(f'{map_dir}/GP/Data/traj_info_{0}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    lap_time_arr[itr_t, 0] = nogplap_time
    collision[itr_t, 0]  = nogpcol
    
    for itr in range(1, NUM_ITERS):

        if data_len >= 200:
            print(".....Sparse GP Training.....")
            trainMain_sparse_torch(
            iteration=itr,
            file_dir=f"{map_dir}/GP",
            inducing_method="kmeans",  # "random"도 가능
            epochs=1500,
            batch_size=1024,
            lr=1e-2,
            verbose=True)
        else:
            print(".....GP Training.....")
            trainMain(itr, file_dir = f"{map_dir}/GP")
        
        traj, _, _, lap_time,col = MPCmain_forces(itr, gp_train=True,  file_name = 'optimized_traj', map_dir = map_dir,  file_dir='/GP/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynGP')
        lap_time_arr[itr_t, itr]  = lap_time
        collision[itr_t, itr] = col

        data= np.load(f'{map_dir}/GP/Data/traj_info_{itr}.npz', allow_pickle=True)
        data_len += len(data['x_true'])-2


    # ''' BO'''
    data_len = 0
    _, _, _, nogplap_time, nogpcol = MPCmain_forces(0, gp_train=False,  map_dir = map_dir,  file_dir='/Bayes/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynGP')
    data= np.load(f'{map_dir}/Bayes/Data/traj_info_{0}.npz', allow_pickle=True)
    data_len += len(data['x_true'])-2
    bo_laptime[itr_t, 0] = nogplap_time
    collision_bo[itr_t, 0]  = nogpcol

    for itr in range(1, NUM_ITERS):
        print("--------------------------------------------")
        print("Current iteration is", itr)
        if data_len >= 200:
            print(".....Sparse GP Training.....")
            trainMain_sparse_torch(
            iteration=itr,
            file_dir=f'{map_dir}/Bayes',
            inducing_method="kmeans",  # "random"도 가능
            epochs=1500,
            batch_size=1024,
            lr=1e-2,
            verbose=True)
        else:
            print(".....GP Training.....")
            trainMain(itr, file_dir = f'{map_dir}/Bayes')

        print(".....Bayesian Optimization.....")
        trackoptimization = TrackOptimization_bo(map_dir = map_dir, generate = True, gptrain = True, gpiter = itr, trial = BOTRIAL, model = 'ucb', weight = 0.1, label = f'MAP{itr_t}_ucb')
        trackoptimization.select_points()
        bo_time[itr_t, itr] = trackoptimization.best_time
        print(".....MPC.....")
        _, _, lap_time2, lap_time = MPCmain_bo(itr, gp_iter = itr, gp_train=True, map_dir = map_dir,  file_dir='/Bayes/', file_name = 'bayes_traj_best', return_ = True, plot_ = False, label = f'MAP{itr_t}_ucb')
        bo_time[itr_t, itr] = lap_time
        _, _, _, lap_time, col = MPCmain_forces(itr, gp_train=True, map_dir = map_dir, file_dir='/Bayes/', file_name = 'bayes_traj_best', return_ = True, plot_ = True, label = f'true_MAP{itr_t}_ucb')
        bo_laptime[itr_t, itr] = lap_time
        collision_bo[itr_t, itr] = col

        data= np.load(f'{map_dir}/Bayes/Data/traj_info_{itr}.npz', allow_pickle=True)
        data_len += len(data['x_true'])-2
        print("--------------------------------------------")
    
    print("--------------------")
    print("--------------------")
    print(f"MAP{itr_t}")
    print(f"True expected lap time {true_plannedtime_arr[itr_t]}s with collision {truecol_arr[itr_t]}" )
    print("True lap time", true_lap_time_arr[itr_t])
    print(" ")
    print(f"Expected Lap time with defined model {defined_plannedtime_arr[itr_t]}")
    print(f"Lap time with defined model {define_lap_time_arr[itr_t]}s with collision {definecol_arr[itr_t]}")
    print("Lap time with  GP", lap_time_arr[itr_t])
    print("Collision of GP", collision[itr_t])
    print(" ")
    print("Expected Lap time with iterative GP", doublegp_time[itr_t])
    print("Lap time with iterative GP", doublegp_laptime[itr_t])
    print("Collision of double GP", collision_doublegp[itr_t])
    print(" ")
    print("Expected Lap time with bo", bo_time[itr_t])
    print("Lap time with bo", bo_laptime[itr_t])
    print("Collision of bo", collision_bo[itr_t])

    np.savez(f"laptime_result{itr_t}.npz", 
            true_plannedtime_arr = true_plannedtime_arr[itr_t],
            true_lap_time_arr = true_lap_time_arr[itr_t],
            truecol_arr = truecol_arr[itr_t],
            defined_plannedtime_arr = defined_plannedtime_arr[itr_t],
            define_lap_time_arr = define_lap_time_arr[itr_t],
            definecol_arr = definecol_arr[itr_t],
            lap_time_arr = lap_time_arr[itr_t],
            collision = collision[itr_t],
            doublegp_time = doublegp_time[itr_t],
            doublegp_laptime = doublegp_laptime[itr_t],
            collision_doublegp = collision_doublegp[itr_t],
            bo_laptime = bo_laptime[itr_t],
            bo_time = bo_time[itr_t],
            collision_bo = collision_bo[itr_t],
            )

    itr_t = itr_t + 1
    if itr_t >= NUM_ITERS_TRACK:
        break



for itr_t in range(NUM_ITERS_TRACK):
    print("--------------------")
    print("--------------------")
    print(f"MAP{itr_t}")
    print(f"True expected lap time {true_plannedtime_arr[itr_t]}s with collision {truecol_arr}" )
    print("True lap time", true_lap_time_arr[itr_t])
    print(" ")
    print(f"Expected Lap time with defined model {defined_plannedtime_arr[itr_t]}")
    print(f"Lap time with defined model {define_lap_time_arr[itr_t]}s with collision {definecol_arr}")
    print("Lap time with  GP", lap_time_arr[itr_t])
    print("Collision of GP", collision[itr_t])
    print(" ")
    print("Expected Lap time with iterative GP", doublegp_time[itr_t])
    print("Lap time with iterative GP", doublegp_laptime[itr_t])
    print("Collision of double GP", collision_doublegp[itr_t])
    print(" ")
    print("Expected Lap time with bo", bo_time[itr_t])
    print("Lap time with bo", bo_laptime[itr_t])
    print("Collision of bo", collision_bo[itr_t])

