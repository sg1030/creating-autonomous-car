import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import gpflow
import tensorflow as tf
from sklearn.cluster import KMeans
from gpflow.models import SVGP
from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints
import warnings
warnings.filterwarnings('ignore')

def plot_true_predicted_variance_multioutput(
        x_true,y_true, y_mu, y_std, 
        x=None, x_scaler=None,y_scaler=None , xlabel='Sample Index', ylabels=None, 
        figsize=(8, 5), plot_title_prefix='Output', varidx = 0, iter = 0, file_dir='./GP/' ):
    """
    Plot predicted mean with confidence interval and residuals
    for each output dimension separately.
    """

    l = y_true.shape[0]

    if x is None:
        x = np.arange(l)

    ylabels = ylabels[varidx]
    # x_t = x_true*x_scaler.scale_+x_scaler.mean_
    y_t = y_true[:,0]*y_scaler.scale_+y_scaler.mean_
    y_m = y_mu[:,0]*y_scaler.scale_+y_scaler.mean_
    y_s = y_std[:,0]

    plt.figure(10, figsize=figsize)
    plt.suptitle(f'Prediction of {ylabels}', fontsize=14)
    gs = gridspec.GridSpec(3, 1)

    # --- Prediction Plot ---
    plt.subplot(gs[:-1, :])
    plt.plot(x, y_m, '#990000', ls='-', lw=1.5, label='Predicted')
    plt.fill_between(x, y_m + 2 * y_s, y_m - 2 * y_s, alpha=0.2, color='m', label='±2σ')
    plt.plot(x, y_t, '#e68a00', ls='--', lw=1, label='True')
    plt.ylabel(ylabels)
    plt.legend(loc='upper right')
    plt.title('True vs Predicted')

    # --- Residual Plot ---
    plt.subplot(gs[2, :])
    error = np.abs(y_t - y_m)
    plt.plot(x, error, '#990000', lw=0.8, label='|Error|')
    plt.fill_between(x, error+ 2 * y_s, error - 2 * y_s, alpha=0.2, color='m', label='±2σ')
    plt.xlabel(xlabel)
    plt.ylabel(f'Error {ylabels}')
    plt.title('Prediction Error and Variance')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"{file_dir}/Figure/learning{ylabels}_iter{iter}.png", dpi=300, bbox_inches='tight')
    
def trainMain_sparse(iteration, file_dir='./MAP0/GP'):
    state_names = [ 'vy', 'omega']
    
    for VARIDX in range(len(state_names)):
        filename = f'{file_dir}/Data/gp_iter{iteration-1}_{state_names[VARIDX]}.pickle'

        x_all = []
        y_all = []
        for itr in range(iteration):
            data= np.load(f'{file_dir}/Data/traj_info_{itr}.npz', allow_pickle=True)

            x_true = data['x_true'][2:, 3:]
            x_true_prev = data['x_true'][1:-1, 3:]
            x_pred = data['x_pred'][1:, 3:] 
            x_u = data['u'][1:]
            print(x_true_prev.shape)
            print(x_u.shape)

            x_input = np.hstack([x_true_prev, x_u])
            y_output = x_true[:,VARIDX] - x_pred[:,VARIDX] 

            x_all.append(x_input)
            y_all.append(y_output.reshape(-1, 1))


        x_windowed = np.concatenate(x_all, axis=0)
        y_windowed = np.concatenate(y_all, axis=0)
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(x_windowed)
        x_windowed = xscaler.transform(x_windowed)

        yscaler.fit(y_windowed)
        y_windowed  = yscaler.transform(y_windowed)


        M = 200
        idx = np.random.choice(x_windowed.shape[0], size=M, replace=False)
        Z_var = x_windowed[idx, :]
        D = x_windowed.shape[1]
        kernel = RBF(lengthscales=1, variance=1.0)
        likelihood = Gaussian()

        model = SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=InducingPoints(Z_var), 
            num_latent_gps=1
        )

        gpflow.set_trainable(model.inducing_variable, True)

        scipy_opt = gpflow.optimizers.Scipy()
        training_data = (x_windowed, y_windowed)
        loss_closure = model.training_loss_closure(data=training_data)

        scipy_opt.minimize(loss_closure, model.trainable_variables)

        # parameter 저장
        Z = model.inducing_variable.Z.numpy()
        lengthscale = model.kernel.lengthscales.numpy()
        variance = model.kernel.variance.numpy()
        likelihood_var = model.likelihood.variance.numpy()

        Kmm = model.kernel(Z) + tf.eye(M, dtype=tf.float64) * 1e-6
        Kmn = model.kernel(Z, x_windowed)
        Kmm_inv = tf.linalg.inv(Kmm)
        sigma2 = model.likelihood.variance.numpy()

        # A = Kmm + Kmn @ Kmn.T / sigma^2
        A = Kmm + tf.matmul(Kmn, Kmn, transpose_b=True) / sigma2
        A_inv = tf.linalg.inv(A)

        alpha = tf.matmul(Kmn, tf.cast(y_windowed, tf.float64)) / sigma2
        alpha = tf.matmul(A_inv, alpha)

        mean, var = model.predict_f(x_windowed)
        MSE = mean_squared_error(y_windowed, mean, multioutput='raw_values')
        R2Score = r2_score(y_windowed, mean, multioutput='raw_values')
        print('root mean square error: %s' %(np.sqrt(MSE)))
        print('R2 score: %s' %(R2Score))

        gp_train_model = "sparsegp"
        model = {
            "Z": Z,
            "alpha": alpha.numpy(),
            "Kmm_inv": Kmm_inv.numpy(),
            "lengthscale": lengthscale,
            "variance": variance,
            "likelihood_variance": likelihood_var,
        }

        with open(filename, 'wb') as f:
            pickle.dump((model, xscaler, yscaler, gp_train_model), f)

        # Plot
        plot_true_predicted_variance_multioutput(
            x_true=x_windowed,
            y_true=y_windowed,
            y_mu=mean,
            y_std=var,
            x_scaler=xscaler,
            y_scaler=yscaler,
            xlabel='Sample Index',
            ylabels=[ 'vy', 'yaw_rate'],
            plot_title_prefix='GPR Output',
            varidx = VARIDX,
            iter=iteration-1,
            file_dir=file_dir
        )


# trainMain_sparse(2)