import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import *
from init import *
from MPC_main_sparse_gp_forces import *
from MPC_main_bruteforce import *
from train_model import *
# from train_model_gpflow import *
from brute_function import runMain_bruteforce_id
lap_time_list_gp=[]

traj, _, _, lap_time,col = MPCmain_forces(0, gp_train=False,  file_name = 'optimized_traj', map_dir = f"./MAP0",  file_dir='/GP/', return_ = True, plot_ = False, label =  f'MAP{0}_DynGP')
lap_time0=lap_time
for i in range(1,10):
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

    traj, _, _, lap_time,col = MPCmain_bruteforces(0,B,C,mu,Iz, gp_train=False,  file_name = 'optimized_traj', map_dir = f"./MAP0",  file_dir='/GP/', return_ = True, plot_ = False, label =  f'MAP{0}_DynGP')
    lap_time_list_gp.append(lap_time)
    
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
