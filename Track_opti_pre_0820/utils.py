
import os
import numpy as np
import matplotlib.pyplot as plt

def create_directory_structure(map_dir):
    os.makedirs(map_dir, exist_ok=True)
    for sd in ['Bayes', 'GP', 'DoubleGP', 'Traj']:
        sub_path = os.path.join(map_dir, sd)
        os.makedirs(sub_path, exist_ok=True)
        if sd in ['Bayes', 'GP', 'DoubleGP']:
            for nested in ['Data', 'Figure', 'MPC', 'Traj']:
                os.makedirs(os.path.join(sub_path, nested), exist_ok=True)

def save_track_txt(center, inner, outer, save_dir):
    np.savetxt(f"./{save_dir}/Traj/centerline.txt", center, fmt="%.5f", delimiter=", ")
    np.savetxt(f"./{save_dir}/Traj/innerwall.txt", inner, fmt="%.5f", delimiter=", ")
    np.savetxt(f"./{save_dir}/Traj/outerwall.txt", outer, fmt="%.5f", delimiter=", ")

def plot_track(center, inner, outer, save_path):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.plot(center[:,0], center[:,1], 'r--', lw=1)
    ax.plot(outer[:,0], outer[:,1], 'k-', lw=2)
    ax.plot(inner[:,0], inner[:,1], 'k-', lw=2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()