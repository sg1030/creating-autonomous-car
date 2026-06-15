import numpy as np
import warnings
import os
import time
warnings.filterwarnings('ignore')

from generate_rand_track import *
from utils import *
from init import *
from MPC_main_sparse_gp_forces import *
from train_model import *
from train_model_gpflow import *
from bayes_opt_track_gp import *

# Node-based 최적화 추가 (NEW!)
from bayes_opt_track_node import NodeBasedTrackOptimization


NUM_ITERS_TRACK = 1
NUM_ITERS = 5
BOTRIAL = 300  # GP-based와 Node-based 모두 동일한 trial 수 사용 (공정한 비교)

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

# Node-based BO 결과 저장용 배열 추가 (NEW!)
node_bo_time = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
node_bo_laptime = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))
collision_node_bo = np.zeros((NUM_ITERS_TRACK,NUM_ITERS))



# for itr_t in range(NUM_ITERS_TRACK):

itr_t = 0
while(1):
    
    ''' Create Radom Track'''
    print(f"[ITER {itr_t}] Generating track...")
    # center_ = generate_random_closed_track()
    # center = resample_centerline(center_[:-1])
    # outer, inner = compute_offset_curves(center, TRACK_WIDTH)

    # TRACK_RADIUS = np.random.randint(5,8) 
    # result = create_track(TRACK_RADIUS)
    # if result is False:
    #     print(f"[{itr_t}] Invalid track, retrying...")
    #     continue
    # center, inner, outer = result
    
    map_dir = f"./MAP{itr_t}"
    # create_directory_structure(map_dir)
    # save_track_txt(center, inner, outer, map_dir)
    # plot_track(center, inner, outer, os.path.join(map_dir, "track.png"))

    # # if itr_t >= NUM_ITERS_TRACK/2:
    # true_dyn = {
    #         "C": 1.7 - np.random.rand()*0.4,
    #         "B": 1.4 - np.random.rand()*0.4,
    #         "Iz": 0.024 - np.random.rand()*0.01,
    #         "wheel_friction": 1.2 - np.random.rand()*0.4}
    # print(true_dyn)
    # np.savez(f"true_dyn_{itr_t}.npz", true_dyn = true_dyn)
    # else:
    #     true_dyn = None

    ''' Run '''
    # true_plannedtime = SolveMinimumLapTime(dir = map_dir, model = 'DynTrue', iter = 0, model_param = true_dyn)
    # true_plannedtime_arr[itr_t] = true_plannedtime
    # _, _, _, true_lap_time, truecol =  MPCmain_forces(-1, gp_train=False,  file_name = 'optimized_true_traj', map_dir = map_dir, file_dir='/GP/', return_ = True, plot_ = True, label = f'MAP{itr_t}_DynTrue', model_param = true_dyn)
    # true_lap_time_arr[itr_t] = true_lap_time
    # truecol_arr[itr_t] = truecol
    # print("Collision with true model", truecol)

    # if true_lap_time >= 30:
    #     continue
    # if truecol:
    #     continue
    
    # data_len = 0
    # defined_plannedtime = SolveMinimumLapTime(dir = map_dir, model = 'Dyn', iter = 0)
    # defined_plannedtime_arr[itr_t] = defined_plannedtime
    # _, _, _, define_lap_time, definecol = MPCmain_forces(-1, gp_train=False, map_dir = map_dir, file_dir='/GP/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynTrueGP', model_param = true_dyn)
    # define_lap_time_arr[itr_t] = define_lap_time
    # definecol_arr[itr_t] = definecol
    # print("Collision with defined model", definecol)

    # if true_lap_time >= 30:
    #     continue

    # ''' BO'''
    # data_len = 0
    # _, _, _, nogplap_time, nogpcol = MPCmain_forces(0, gp_train=False,  map_dir = map_dir,  file_dir='/Bayes/', return_ = True, plot_ = True, label =  f'MAP{itr_t}_DynGP')
    # data= np.load(f'{map_dir}/Bayes/Data/traj_info_{0}.npz', allow_pickle=True)
    # data_len += len(data['x_true'])-2
    # bo_laptime[itr_t, 0] = nogplap_time
    # collision_bo[itr_t, 0]  = nogpcol

    # for itr in range(1, NUM_ITERS):
    #     print("--------------------------------------------")
    #     print("Current iteration is", itr)
    #     if data_len >= 200:
    #         print(".....Sparse GP Training.....")
    #         trainMain_sparse(itr, file_dir = f'{map_dir}/Bayes')
    #     else:
    #         print(".....GP Training.....")
    #         trainMain(itr, file_dir = f'{map_dir}/Bayes')

    #     print(".....Bayesian Optimization.....")
    #     trackoptimization = TrackOptimization_bo(map_dir = map_dir, generate = True, gptrain = True, gpiter = itr, trial = BOTRIAL, model = 'ucb', weight = 0.1, label = f'MAP{itr_t}_ucb')
    #     trackoptimization.select_points()
    #     bo_time[itr_t, itr] = trackoptimization.best_time
    #     print(".....MPC.....")
    #     _, _, lap_time2, lap_time = MPCmain_bo(itr, gp_iter = itr, gp_train=True, map_dir = map_dir,  file_dir='/Bayes/', file_name = 'bayes_traj_best', return_ = True, plot_ = False, label = f'MAP{itr_t}_ucb')
    #     bo_time[itr_t, itr] = lap_time
    #     _, _, _, lap_time, col = MPCmain_forces(itr, gp_train=True, map_dir = map_dir, file_dir='/Bayes/', file_name = 'bayes_traj_best', return_ = True, plot_ = True, label = f'true_MAP{itr_t}_ucb')
    #     bo_laptime[itr_t, itr] = lap_time
    #     collision_bo[itr_t, itr] = col

    #     data= np.load(f'{map_dir}/Bayes/Data/traj_info_{itr}.npz', allow_pickle=True)
    #     data_len += len(data['x_true'])-2
    #     print("--------------------------------------------")
    
    ''' Node-based BO (NEW!) - Wavelet BO와 동일한 조건으로 비교 '''
    print("==============================================")
    print("Starting Node-based Bayesian Optimization...")
    print("==============================================")
    
    # Node-based BO는 자체적으로 데이터를 생성하지만, 공정한 비교를 위해 동일한 iteration 구조 사용
    for itr in range(1, NUM_ITERS):
        print("--------------------------------------------")
        print(f"Node-based BO - Current iteration is {itr}")
        
        # GP 학습은 기존 Bayes 디렉토리의 데이터 활용 (동일한 학습 데이터 조건)
        try:
            # 기존 GP 데이터가 있으면 활용
            bayes_data_file = f'{map_dir}/Bayes/Data/traj_info_{itr-1}.npz'
            if os.path.exists(bayes_data_file):
                data = np.load(bayes_data_file, allow_pickle=True)
                data_len_node = len(data['x_true']) - 2
                
                if data_len_node >= 200:
                    print(".....Node-based BO: Sparse GP Training.....")
                    trainMain_sparse(itr, file_dir=f'{map_dir}/Bayes')
                else:
                    print(".....Node-based BO: GP Training.....")
                    trainMain(itr, file_dir=f'{map_dir}/Bayes')
            
            print(".....Node-based Bayesian Optimization.....")
            
            # Node-based 베이지안 최적화 실행 (동일한 trial 수와 설정 사용)
            node_optimization = NodeBasedTrackOptimization(
                map_dir=map_dir,
                max_nodes=12,          # 적당한 노드 수
                generate=True,         # 초기 데이터 생성
                gptrain=True,         # GP 학습 활용
                gp_iter=itr,          # 현재 iteration
                trial=BOTRIAL,        # Wavelet BO와 동일한 trial 수!
                model='ucb',          # 동일한 acquisition function
                weight=0.1,           # 동일한 제약조건 가중치
                label=f'MAP{itr_t}_node_ucb'
            )
            
            # 자동 랜덤 노드 생성 (사용자 입력 없이)
            print(".....Auto-generating random nodes.....")
            node_optimization.selection_mode = 'random'
            node_optimization.random_nodes = 8  # 8개 노드 사용
            node_optimization.generate_random_nodes()
            
            # 베이지안 최적화 실행
            start_time = time.time()
            node_optimization.run_bayesian_optimization()
            optimization_time = time.time() - start_time
            
            # 결과 추출 (Wavelet BO와 동일한 방식)
            result_file = f'{map_dir}/Bayes/Data/node_result_{itr}.npz'
            if os.path.exists(result_file):
                node_results = np.load(result_file, allow_pickle=True)
                if len(node_results['best_actual_time']) > 0:
                    node_bo_time[itr_t, itr] = node_results['best_actual_time'][-1]
                else:
                    node_bo_time[itr_t, itr] = 30.0  # 실패시 페널티
            else:
                node_bo_time[itr_t, itr] = 30.0
            
            print(".....Node-based BO: MPC.....")
            
            # Node 궤적으로 MPC 실행 (Wavelet과 동일하게 두 가지 평가: 모델 내부 잔차보정 + 진실 동역학)
            node_traj_file = f'{map_dir}/Traj/node_traj_best{itr}.txt'
            if os.path.exists(node_traj_file):
                # 1) 모델 내부 잔차 보정 평가 (MPCmain_bo)
                _, _, lap_time2, lap_time = MPCmain_bo(
                    itr, gp_iter=itr, gp_train=True, map_dir=map_dir, file_dir='/Bayes/',
                    file_name='node_traj_best', return_=True, plot_=True,
                    label=f'MAP{itr_t}_node_ucb')

                node_bo_time[itr_t, itr] = lap_time

                # 2) 진실 동역학 평가 (MPCmain_forces)
                _, _, _, lap_time, col = MPCmain_forces(
                    itr, gp_train=True, map_dir=map_dir, file_dir='/Bayes/',
                    file_name='node_traj_best', return_=True, plot_=True,
                    label=f'true_MAP{itr_t}_node_ucb')

                node_bo_laptime[itr_t, itr] = lap_time
                collision_node_bo[itr_t, itr] = col
                
                print(f"Node-based BO result: {lap_time:.3f}s, collision: {col}")
            else:
                print("Warning: Node trajectory file not found, using penalty")
                node_bo_laptime[itr_t, itr] = 30.0
                collision_node_bo[itr_t, itr] = 1
            
            print(f"Node-based BO optimization time: {optimization_time:.1f}s")
            
        except Exception as e:
            print(f"Error in Node-based BO iteration {itr}: {e}")
            node_bo_time[itr_t, itr] = 30.0
            node_bo_laptime[itr_t, itr] = 30.0
            collision_node_bo[itr_t, itr] = 1
            import traceback
            traceback.print_exc()
        
        print("--------------------------------------------")
    
    print("==============================================")
    print("Node-based Bayesian Optimization completed!")
    print("==============================================")
    
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
    print(" ")
    print("Expected Lap time with node-based bo", node_bo_time[itr_t])
    print("Lap time with node-based bo", node_bo_laptime[itr_t])
    print("Collision of node-based bo", collision_node_bo[itr_t])

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
            node_bo_laptime = node_bo_laptime[itr_t],
            node_bo_time = node_bo_time[itr_t],
            collision_node_bo = collision_node_bo[itr_t],
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
    print("Expected Lap time with node-based bo", node_bo_time[itr_t])
    print("Lap time with node-based bo", node_bo_laptime[itr_t])
    print("Collision of node-based bo", collision_node_bo[itr_t])

    
