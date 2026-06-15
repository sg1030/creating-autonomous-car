# torch_sparse_gp.py
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import solve_triangular

import gpytorch
from Optimization.dynamics import *
from Track.track import Track

def plot_true_predicted_variance_multioutput(
        y_true, y_mu, y_std,
        x=None, xlabel='Sample Index', ylabels=None,
        figsize=(8, 5), file_dir='./GP', varidx=0, iteration=0):

    os.makedirs(os.path.join(file_dir, "Figure"), exist_ok=True)

    l = y_true.shape[0]
    if x is None:
        x = np.arange(l)

    label = ylabels[varidx] if ylabels is not None else f"y[{varidx}]"
    y_t = y_true[:, 0]
    y_m = y_mu[:, 0]
    y_s = y_std[:, 0]

    plt.figure(10, figsize=figsize)
    plt.suptitle(f'Prediction of {label}', fontsize=14)
    gs = gridspec.GridSpec(3, 1)

    # True vs Pred
    plt.subplot(gs[:-1, :])
    plt.plot(x, y_m, '#990000', ls='-', lw=1.5, label='Predicted')
    plt.fill_between(x, y_m - 2 * y_s, y_m + 2 * y_s, alpha=0.2, color='m', label='±2σ')
    plt.plot(x, y_t, '#e68a00', ls='--', lw=1, label='True')
    plt.ylabel(label)
    plt.legend(loc='upper right')
    plt.title('True vs Predicted')

    # Residual
    plt.subplot(gs[2, :])
    error = np.abs(y_t - y_m)
    plt.plot(x, error, '#990000', lw=0.8, label='|Error|')
    plt.fill_between(x, np.maximum(0, error - 2 * y_s), error + 2 * y_s, alpha=0.2, color='m', label='±2σ')
    plt.xlabel(xlabel)
    plt.ylabel(f'Error {label}')
    plt.title('Prediction Error and Variance')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = os.path.join(file_dir, "Figure", f"learning_{label}_iter{iteration}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
 



class InducingSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, input_dim, ard=True):
        # Variational stuff
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()

        # if ard:
        #     self.covar_module = gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        #     )
        # else:
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def init_inducing_points(X_np, M=100, method="kmeans", random_state=0):
    if method == "kmeans":
        km = KMeans(n_clusters=M, random_state=random_state)
        Z = km.fit(X_np).cluster_centers_
    elif method == "random":
        idx = np.random.RandomState(random_state).choice(X_np.shape[0], size=M, replace=False)
        Z = X_np[idx, :]
    else:
        raise ValueError("method must be 'kmeans' or 'random'")
    return torch.from_numpy(Z).float()

def _rbf_kernel_numpy(X, Y, lengthscale, variance):

    ls = np.asarray(lengthscale).reshape(-1,)          # (D,) or (1,) -> (D,)
    Xs = X / ls
    Ys = Y / ls
    XX = np.sum(Xs * Xs, axis=1, keepdims=True)        # (N,1)
    YY = np.sum(Ys * Ys, axis=1, keepdims=True).T      # (1,M)
    d2 = XX + YY - 2.0 * Xs @ Ys.T                     # (N,M)
    return variance * np.exp(-0.5 * d2)

def train_svgp_single_output(
    X_np, y_np,
    inducing_method="kmeans",
    batch_size=1024, epochs=1500, lr=1e-2,
    ard=True, device=None, verbose=True
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 스케일링
    xscaler = StandardScaler().fit(X_np)
    X_scaled = xscaler.transform(X_np)

    yscaler = StandardScaler().fit(y_np)
    y_scaled = yscaler.transform(y_np)

    X = torch.from_numpy(X_scaled).float().to(device)
    y = torch.from_numpy(y_scaled).float().squeeze(-1).to(device)

    N, D = X.shape
    M = 100

    # 유도점
    Z = init_inducing_points(X_scaled, M=M, method=inducing_method).to(device)

    # 모델과 likelihood
    model = InducingSVGPModel(Z, input_dim=D, ard=ard).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    # 미니배치
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    likelihood.train()

    # Variational ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': likelihood.parameters(), 'lr': lr}
    ])

    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (ep+1) % 200 == 0:
            print(f"[Epoch {ep+1}] ELBO: {-total_loss:.4f}")

    # 평가 모드
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = likelihood(model(X))
        mean = pred.mean.cpu().numpy().reshape(-1, 1)
        var = pred.variance.cpu().numpy().reshape(-1, 1)
        std = np.sqrt(var)

    y_mu = yscaler.inverse_transform(mean)
    y_std = std * yscaler.scale_.reshape(1, -1)
    y_true = y_np 

    mse = mean_squared_error(y_true, y_mu, multioutput='raw_values')
    r2 = r2_score(y_true, y_mu, multioutput='raw_values')
    if verbose:
        print("root mean square error:", np.sqrt(mse))
        print("R2 score:", r2)


    Z_torch = model.variational_strategy.inducing_points.detach().cpu().float()
    Z_np = Z_torch.numpy()

    base = model.covar_module.base_kernel
    lengthscale = base.lengthscale.detach().cpu().numpy().reshape(-1)   # (D,) or (1,)
    variance = model.covar_module.outputscale.detach().cpu().item()     # scalar
    likelihood_var = likelihood.noise.detach().cpu().item()             # scalar
    sigma2 = float(likelihood_var)         # scalar

    X_np_scaled = X.detach().cpu().numpy()                  
    y_np_scaled = y.detach().cpu().numpy().reshape(-1, 1)   


    Kmm = _rbf_kernel_numpy(Z_np, Z_np, lengthscale, variance)   # (M, M)
    Kmm += 1e-6 * np.eye(Kmm.shape[0])                           # jitter
    Kmn = _rbf_kernel_numpy(X_np_scaled, Z_np, lengthscale, variance).T   # (M, N)

    A = Kmm + (Kmn @ Kmn.T) / sigma2
    rhs = (Kmn @ y_np_scaled) / sigma2                           # (M, 1)
    alpha = np.linalg.solve(A, rhs)                              # (M, 1)

    # (원하면 Kmm_inv도 저장)
    Kmm_inv = np.linalg.inv(Kmm)                              # (M,1)

    model_dict = {
        "Z": Z_np.astype(np.float64),                      # (M,D)
        "alpha": alpha.astype(np.float64),                 # (M,1)
        "Kmm_inv": Kmm_inv.astype(np.float64),             # (M,M)
        "lengthscale": lengthscale.astype(np.float64),     # (D,)
        "variance": float(variance),                       # scalar
        "likelihood_variance": float(likelihood_var),      # scalar
    }

    return model, likelihood, xscaler, yscaler, y_true, y_mu, y_std, model_dict

def _precompute_s_inds(track_s, s_center_len, s_query_1d):
    s_ind = np.searchsorted(track_s, s_query_1d, side="left")
    s_ind = np.where(s_ind >= s_center_len, s_ind - s_center_len, s_ind)
    s_ind = np.where(s_ind < 0, 0, s_ind)
    return s_ind.astype(int)

import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_pred_states(x_true_345, x_pred_full,
                            state_names=("vx", "vy", "omega"),
                            start=0, end=None,
                            save_path=None):
    """
    x_true_345 : (T,3)  -> data['x_true'][2:, 3:]
    x_pred_full: (T,6) or (T,3) or (T,6,1)/(T,3,1) 등
    """

    # --- normalize shapes ---
    x_true_345 = np.asarray(x_true_345)
    x_pred_full = np.asarray(x_pred_full)

    # (T,6,1) -> (T,6), (6,1) -> (6,)
    x_true_345 = np.squeeze(x_true_345)
    x_pred_full = np.squeeze(x_pred_full)

    # ensure 2D
    if x_true_345.ndim == 1:
        x_true_345 = x_true_345.reshape(-1, 1)
    if x_pred_full.ndim == 1:
        x_pred_full = x_pred_full.reshape(-1, 1)

    T = x_true_345.shape[0]
    if end is None:
        end = T
    end = min(end, T)

    # --- pick vx,vy,omega columns from prediction ---
    # case A) pred already (T,3)
    if x_pred_full.shape[1] == 3:
        x_pred_345 = x_pred_full[:T, :]

    # case B) pred is (T,6 or more)
    elif x_pred_full.shape[1] >= 6:
        x_pred_345 = x_pred_full[:T, 3:6]

    else:
        raise ValueError(
            f"x_pred_full shape is {x_pred_full.shape}. "
            f"Need (T,3) or (T,>=6). "
            f"Check how x_pred is constructed/assigned."
        )

    # --- plot ---
    t = np.arange(start, end)

    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)

    for k in range(3):
        y_true = x_true_345[start:end, k]
        y_pred = x_pred_345[start:end, k]
        resid = y_true - y_pred

        ax0 = axes[k, 0]
        ax0.plot(t, y_true, linestyle="--", linewidth=1.6, label="True")
        ax0.plot(t, y_pred, linestyle="-",  linewidth=1.8, label="Pred")
        ax0.set_ylabel(state_names[k])
        ax0.grid(True)
        if k == 0:
            ax0.set_title("True vs Pred")
            ax0.legend(loc="best")

        ax1 = axes[k, 1]
        ax1.plot(t, resid, linewidth=1.6, label="True - Pred")
        ax1.axhline(0.0, linewidth=1.0, linestyle="--")
        ax1.grid(True)
        if k == 0:
            ax1.set_title("Residual (True - Pred)")
            ax1.legend(loc="best")

    axes[-1, 0].set_xlabel("Timestep")
    axes[-1, 1].set_xlabel("Timestep")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def trainMain_sparse_torch(iteration,B,C,mu,Iz, file_dir='./GP', inducing_method="kmeans",
                           epochs=1500, batch_size=1024, lr=1e-2, verbose=True):
    dt=0.1
    N=15
    iteration1=0
    map_dir = f"./MAP0"
    file_name = 'optimized_traj'
    track_dir = map_dir + '/Traj/'
    track = Track(track_dir, track_file=f'{file_name}{iteration1}', traj_type='opt_traj')
    dynamics = VehicleDynamics(track, dt=dt, N=N)
    dynamics_true= VehicleDynamicsTrue(track, dt=dt, N=N)
    dynamics.wheel_friction = mu
    dynamics.C = C
    dynamics.B = B
    dynamics.Iz = Iz
    
    os.makedirs(os.path.join(file_dir, "Data"), exist_ok=True)
    os.makedirs(os.path.join(file_dir, "Figure"), exist_ok=True)

    state_names = ['vx', 'vy', 'omega']
    

    for VARIDX, state_name in enumerate(state_names):
        x_all, y_all = [], []
        for itr in range(iteration):
            data = np.load(os.path.join(file_dir, 'Data', f'traj_info_{itr}.npz'), allow_pickle=True)
            x_true = data['x_true'][2:, 3:]
            x_true_prev = data['x_true'][1:-1, 3:]
            x_true_prev1= data['x_true'][1:-1, :]
            x_u = data['u'][1:]
            x_pred = data['x_pred'][1:, :] 
  
            s_inds = _precompute_s_inds(track.s, len(track.s_center), np.asarray(x_true_prev)[:, 0])
            for i in range(len(x_true_prev)):
                x_pred[i, :] = np.asarray(
    dynamics.f_d_rk4_time(x_true_prev1[i], x_u[i], track.curv_center[s_inds[i]])
).reshape(-1)
            
            x_pred = x_pred[:,3:]
            plot_true_vs_pred_states(
    x_true_345=x_true,     # (T,3)
    x_pred_full=x_pred,    # (T,6)
    state_names=("vx", "vy", "omega"),
    save_path=None         # 파일 저장 원하면 "true_pred.png" 같은 경로
)
            x_input = np.hstack([x_true_prev, x_u])
            y_output = x_true[:,VARIDX] - x_pred[:,VARIDX] 

            x_all.append(x_input)
            y_all.append(y_output.reshape(-1, 1))


        X_np = np.concatenate(x_all, axis=0)
        y_np = np.concatenate(y_all, axis=0)

        # 학습
        model, likelihood, xscaler, yscaler, y_true, y_mu, y_std, bundle = train_svgp_single_output(
            X_np, y_np,
            inducing_method=inducing_method,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            ard=True,
            device=None,
            verbose=verbose
        )

        # 결과 플롯
        plot_true_predicted_variance_multioutput(
            y_true=y_true, y_mu=y_mu, y_std=y_std,
            xlabel='Sample Index',
            ylabels=['vx', 'vy', 'omega'],
            file_dir=file_dir,
            varidx=VARIDX,
            iteration=iteration-1
        )

        # 저장
        gp_train_model = "sparsegp"
        filename = os.path.join(file_dir, 'Data', f'gp_iter{iteration-1}_{state_name}.pickle')
        with open(filename, 'wb') as f:
            pickle.dump((bundle, xscaler, yscaler, gp_train_model), f)

        if verbose:
            print(f"[Saved] {filename}")


class TorchSVGPInfer:
    def __init__(self, bundle):
        self.input_dim = bundle["input_dim"]
        self.ard = bundle["ard"]
        self.M = bundle["M"]

        # 더미 유도점 초기화 후 state_dict 로드
        Z_dummy = torch.zeros(self.M, self.input_dim).float()
        self.model = InducingSVGPModel(Z_dummy, input_dim=self.input_dim, ard=self.ard)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model.load_state_dict(bundle["state_dict_model"])
        self.likelihood.load_state_dict(bundle["state_dict_likelihood"])

        self.xscaler = bundle["xscaler"]
        self.yscaler = bundle["yscaler"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.likelihood.to(self.device)
        self.model.eval()
        self.likelihood.eval()

    @torch.no_grad()
    def predict(self, X_np):
        X_scaled = self.xscaler.transform(X_np)
        X = torch.from_numpy(X_scaled).float().to(self.device)
        pred = self.likelihood(self.model(X))
        mean = pred.mean.cpu().numpy().reshape(-1, 1)
        var = pred.variance.cpu().numpy().reshape(-1, 1)
        std = np.sqrt(var)
        # 역스케일링
        y_mu = self.yscaler.inverse_transform(mean)
        y_std = std * self.yscaler.scale_.reshape(1, -1)
        return y_mu, y_std


if __name__ == "__main__":
    trainMain_sparse_torch(
        iteration=1,
        B=1,C=1.1,mu=0.7,Iz=0.02025,
        file_dir='./MAP0/GP',
        inducing_method="kmeans",  # "random"도 가능
        epochs=1500,
        batch_size=1024,
        lr=1e-2,
        verbose=True
    )