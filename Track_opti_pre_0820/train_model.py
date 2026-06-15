import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score


def plot_true_predicted_variance_multioutput(
        y_true, y_mu, y_std, 
        x=None, xlabel='Sample Index', ylabels=None, 
        figsize=(8, 5), file_dir='./GP/Figure', varidx = 0, iter = 0 ):
    """
    Plot predicted mean with confidence interval and residuals
    for each output dimension separately.
    """

    l = y_true.shape[0]

    if x is None:
        x = np.arange(l)

    ylabels = ylabels[varidx]

    y_t = y_true[:,0]
    y_m = y_mu[:,0]
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
    # plt.show()


def trainMain(iteration, file_dir='./GP/'):
    state_names = ['vx', 'vy', 'omega']
    
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

            x_input = np.hstack([x_true_prev, x_u])
            y_output = x_true[:,VARIDX] - x_pred[:,VARIDX] 

            x_all.append(x_input)
            y_all.append(y_output.reshape(-1, 1))

        x_windowed = np.concatenate(x_all, axis=0)
        y_windowed = np.concatenate(y_all, axis=0)

        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(x_windowed)
        x_train = xscaler.transform(x_windowed)
    
        yscaler.fit(y_windowed)
        y_train = yscaler.transform(y_windowed)

        # print('GP_train_Start')
        # print(x_train.shape)
        # print(y_train.shape)
        # train GP model
        k1 = 1.0*RBF(
            length_scale=np.ones(x_train.shape[1]),
            length_scale_bounds=(1e-5, 1e5),
            )
        k2 = ConstantKernel(0.1)
        kernel = k1 + k2
        model = GaussianProcessRegressor(
            alpha=1e-2, 
            kernel=kernel, 
            normalize_y=False,
            n_restarts_optimizer=10,
            )
        # print(model.kernel)
        # print(model.kernel.get_params().keys())

        model.fit(x_train, y_train)

        gp_train_model = "fullgp"
        with open(filename, 'wb') as f:
            pickle.dump((model, xscaler, yscaler, gp_train_model), f)
        print(f'[{file_dir}Data/gp_iter{iteration-1}] is saved')


        # test GP model on training data
        y_train_mu, y_train_std = model.predict(x_train, return_std=True)
        # y_train = yscaler.inverse_transform(y_train)
        # y_train_mu = yscaler.inverse_transform(y_train_mu.reshape(-1,1))
        # y_train_std *= yscaler.scale_

        MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
        R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
        EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

        print('root mean square error: %s' %(np.sqrt(MSE)))
        # print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_train.mean()))))
        print('R2 score: %s' %(R2Score))
        # print('explained variance: %s' %(EV))

        # Plot
        plot_true_predicted_variance_multioutput(
            y_true=y_train,
            y_mu=y_train_mu.reshape(-1,1),
            y_std=y_train_std.reshape(-1,1),
            xlabel='Sample Index',
            ylabels=[ 'vx', 'vy', 'yaw_rate'],
            file_dir=file_dir,
            varidx = VARIDX,
            iter=iteration-1
        )

# trainMain(1, file_dir = "MAP0/GP")

# residuals = y_train - y_train_mu

# for i in range(y_train.shape[1]):
#     plt.figure(i + 1)
#     plt.scatter(y_train_mu[:, i], residuals[:, i], alpha=0.6)
#     plt.hlines(0, min(y_train_mu[:, i]), max(y_train_mu[:, i]), colors='r', linestyles='dashed')
#     plt.xlabel(f'Predicted Values (output {i})')
#     plt.ylabel('Residuals')
#     plt.title(f'Residual Plot - Output {i}')

# # --- Cross Validation per output ---


# for i in range(y_train.shape[1]):
#     model_i = GaussianProcessRegressor(kernel=model.kernel_, alpha=model.alpha)
#     scores = cross_val_score(model_i, x_train, y_train[:, i], cv=5, scoring='r2')
#     print(f'Output {i}: {scores}, Avg: {scores.mean():.4f}')

