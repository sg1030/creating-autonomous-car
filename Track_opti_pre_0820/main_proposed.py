import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import *
from init import *
from MPC_main_sparse_gp_forces import *
from MPC_main_bruteforce import *
from train_model import *
from bf_torch_sparse import trainMain_sparse_torch
# from train_model_gpflow import *
from brute_function import runMain_bruteforce_id
lap_time_list_gp=[]

traj, _, _, lap_time,col = MPCmain_forces(0, gp_train=False,  file_name = 'optimized_traj', map_dir = f"./MAP0",  file_dir='/GP/', return_ = True, plot_ = False, label =  f'MAP{0}_DynGP')
lap_time0=lap_time
B_prev=0
C_prev=0
mu_prev=0
Iz_prev=0
grid_division=0
grid_division_prev=0
B_lower=0
B_upper=0
C_lower=0
C_upper=0
mu_lower=0
mu_upper=0
Iz_lower=0
Iz_upper=0
for i in range(1,10):
    if i==1:
        B,C,mu,Iz,_,_=runMain_bruteforce_id(
            iteration=i,                 # uses traj_info_0.npz
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
            grid_sizes={"B": 9, "C": 9, "mu": 9, "Iz": 9},
            n_workers=12,
            chunk_size=1024,
            verbose_every=2000,
            save=True,
            plot=True,
        )
        B_lower=1.0
        B_upper=1.2
        C_lower=1.1
        C_upper=1.3
        mu_lower=0.7
        mu_upper=0.9
        Iz_lower=0.014
        Iz_upper=0.024
        if B==B_prev and C==C_prev and mu==mu_prev and Iz==Iz_prev:
            grid_division+=1
        B_prev=B
        C_prev=C
        mu_prev=mu
        Iz_prev=Iz
        
    else:
        print(B,C,mu,Iz)
        if grid_division==0:
            B,C,mu,Iz,_,_=runMain_bruteforce_id(
                iteration=i,                 # uses traj_info_0.npz
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
                grid_sizes={"B": 9, "C": 9, "mu": 9, "Iz": 9},
                n_workers=12,
                chunk_size=1024,
                verbose_every=2000,
                save=True,
                plot=True,
            )
            B_lower=1.0
            B_upper=1.2
            C_lower=1.1
            C_upper=1.3
            mu_lower=0.7
            mu_upper=0.9
            Iz_lower=0.014
            Iz_upper=0.024
        else:
            if grid_division!=grid_division_prev:
                B,C,mu,Iz,_,_=runMain_bruteforce_id(
                    iteration=i,                 # uses traj_info_0.npz
                    file_dir="./MAP0/GP",
                    map_dir="MAP0",
                    file_name="optimized_traj",
                    track_idx=0,
                    traj_type="opt_traj",
                    dt=0.1,
                    N=15,
                    bounds={
                        "B":  (B-0.1/2**(grid_division), B+0.1/2**(grid_division)),
                        "C":  (C-0.1/2**(grid_division), C+0.1/2**(grid_division)),
                        "mu": (mu-0.1/2**(grid_division), mu+0.1/2**(grid_division)),
                        "Iz": (Iz-0.005/2**(grid_division), Iz+0.005/2**(grid_division)),
                    },
                    grid_sizes={"B": 9, "C": 9, "mu": 9, "Iz": 9},
                    n_workers=12,
                    chunk_size=1024,
                    verbose_every=2000,
                    save=True,
                    plot=True,
                )
                B_lower=B-0.1/2**(grid_division)
                B_upper=B+0.1/2**(grid_division)
                C_lower=C-0.1/2**(grid_division)
                C_upper=C+0.1/2**(grid_division)
                mu_lower=mu-0.1/2**(grid_division)
                mu_upper=mu+0.1/2**(grid_division)
                Iz_lower=Iz-0.005/2**(grid_division)
                Iz_upper=Iz+0.005/2**(grid_division)
            
            else:
                B,C,mu,Iz,_,_=runMain_bruteforce_id(
                    iteration=i,                 # uses traj_info_0.npz
                    file_dir="./MAP0/GP",
                    map_dir="MAP0",
                    file_name="optimized_traj",
                    track_idx=0,
                    traj_type="opt_traj",
                    dt=0.1,
                    N=15,
                    bounds={
                        "B":  (B_lower, B_upper),
                        "C":  (C_lower, C_upper),
                        "mu": (mu_lower, mu_upper),
                        "Iz": (Iz_lower, Iz_upper),
                    },
                    grid_sizes={"B": 9, "C": 9, "mu": 9, "Iz": 9},
                    n_workers=12,
                    chunk_size=1024,
                    verbose_every=2000,
                    save=True,
                    plot=True,
                )
                
            grid_division_prev=grid_division    
        if B==B_prev and C==C_prev and mu==mu_prev and Iz==Iz_prev:
            grid_division_prev=grid_division
            grid_division+=1
        B_prev=B
        C_prev=C
        mu_prev=mu
        Iz_prev=Iz
    trainMain_sparse_torch(
    iteration=i,
    B=B,C=C,mu=mu,Iz=Iz,
    file_dir='./MAP0/GP',
    inducing_method="kmeans",  # "random"도 가능
    epochs=1500,
    batch_size=1024,
    lr=1e-2,
    verbose=True
)
    traj, _, _, lap_time,col = MPCmain_bruteforces(i,B,C,mu,Iz, gp_train=True,  file_name = 'optimized_traj', map_dir = f"./MAP0",  file_dir='/GP/', return_ = True, plot_ = False, label =  f'MAP{0}_DynGP')
    lap_time_list_gp.append(lap_time)
    print(grid_division)
    
import numpy as np
import matplotlib.pyplot as plt

# lap_time_list_gp: iteration 1~9 결과가 들어있다고 가정
lap_time_arr = np.asarray(lap_time_list_gp, dtype=float)
iters = np.arange(1, len(lap_time_arr) + 1)

# (권장) GP 학습 전 baseline lap_time(= iteration 0)을 따로 저장해두는 방식
# 예: traj0, _, _, lap_time0, col0 = MPCmain_forces(0, ...)
# 여기서는 이미 lap_time 변수가 마지막 loop 값으로 덮였을 수 있으니,
# 가능하면 아래처럼 baseline을 별도 변수로 저장해 사용하세요.
try:
    lap_time0  # noqa: F821
except NameError:
    lap_time0 = None

plt.figure()
plt.plot(iters, lap_time_arr, marker='o', linestyle='-')
plt.xlabel("Bruteforce Iteration")
plt.ylabel("Lap Time [s]")
plt.title("Lap Time vs Bruteforce Training Iteration")
plt.grid(True)

# baseline 표시 (있으면)
if lap_time0 is not None and np.isfinite(lap_time0):
    plt.axhline(lap_time0, linestyle='--')
    plt.legend(["GP-trained", f"Baseline (iter 0) = {lap_time0:.3f}s"])
else:
    plt.legend(["GP-trained"])

# 최저점 강조
best_idx = int(np.nanargmin(lap_time_arr))
best_iter = iters[best_idx]
best_time = lap_time_arr[best_idx]
plt.scatter([best_iter], [best_time], s=60)
plt.annotate(f"best: {best_time:.3f}s @ iter {best_iter}",
             xy=(best_iter, best_time),
             xytext=(best_iter, best_time),
             textcoords="offset points",
             xycoords="data",
             ha="left", va="bottom")

plt.tight_layout()
plt.show()

# 요약 출력
print(f"[Summary] iters=1..{len(lap_time_arr)}")
print(f"  min lap time : {best_time:.6f} s (iter {best_iter})")
print(f"  max lap time : {np.nanmax(lap_time_arr):.6f} s")
print(f"  last lap time: {lap_time_arr[-1]:.6f} s")

if lap_time0 is not None and np.isfinite(lap_time0):
    improvement = (lap_time0 - best_time) / lap_time0 * 100.0
    print(f"  baseline (iter0): {lap_time0:.6f} s")
    print(f"  best improvement: {improvement:.2f} %")
