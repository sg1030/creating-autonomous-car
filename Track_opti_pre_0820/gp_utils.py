import time
import numpy as np
import casadi as ca
from scipy.linalg import solve_triangular


def rbf_kernel_casadi(x1, x2, lengthscale, variance):
    """
    x1: (D,), x2: (D,)
    lengthscale: (D,), variance: scalar
    return: scalar (ca.MX)
    """
    diff = (x1 - x2) / lengthscale
    sqdist = ca.dot(diff, diff)

    return variance * ca.exp(-0.5 * sqdist)

def CasadiRBF(X, Y, length_scale, constant):
    """ RBF kernel in CasADi
    """
    sX = X.shape[0]
    sY = Y.shape[0]    
    X = X / ca.repmat(length_scale, sX , 1)
    Y = Y / ca.repmat(length_scale, sY , 1)
    dist = ca.repmat(ca.sum1(X.T**2).T,1,sY) + ca.repmat(ca.sum1(Y.T**2),sX,1) - 2*ca.mtimes(X,Y.T)
    K = constant*ca.exp(-.5 * dist)
    return K

def CasadiConstant(X, Y, constant):
    """ Constant kernel in CasADi
    """
    sX = X.shape[0]
    sY = Y.shape[0]
    K = constant*ca.DM.ones((sX, sY))
    return K

    
def loadGPModel(name, model, xscaler, yscaler):
    """ GP mean and variance as casadi.SX variable
    """
    X = model.X_train_
    x = ca.SX.sym('x', 1, X.shape[1])

    length_scale = model.kernel_.get_params()['k1__k2__length_scale'].reshape(1,-1)
    constant = model.kernel_.get_params()['k1__k1__constant_value']
    K1 = CasadiRBF(x, X, length_scale, constant)

    constant_K2 = model.kernel_.get_params()['k2__constant_value']
    K2 = CasadiConstant(x, X, constant_K2)
    K = K1 + K2
    y_mu = ca.mtimes(K, model.alpha_) + model._y_train_mean

    # variance
    L_inv = solve_triangular(model.L_.T,np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    K1_ = CasadiRBF(x, x, length_scale, constant)
    K2_ = CasadiConstant(x, x, constant_K2)
    K_ = K1_ + K2_

    y_var = ca.diag(K_) - ca.sum2(ca.mtimes(K, K_inv)*K)
    y_var = ca.fmax(y_var, 0)
    y_std = ca.sqrt(y_var)

    gpmodel = ca.Function(name, [x], [y_mu, y_std])
    return gpmodel


def loadSparseGPModel(name,
                    Z,
                    alpha,
                    Kmm_inv,
                    constant,
                    length_scale, xscaler, yscaler):

    D = Z.shape[1]
    M = Z.shape[0]
    x = ca.MX.sym('x', 1, D)

    # Compute k_star_m (1, M)
    K = CasadiRBF(x, Z, length_scale, constant)
    y_mu = ca.mtimes(K, alpha)

    K2 =  CasadiRBF(x, x, length_scale, constant)

    # predictive variance
    base_var = K - (K2 @ Kmm_inv @ K2)[0, 0]
    base_var = ca.fmax(base_var, 1e-9) 
    y_std = ca.sqrt(base_var)
    
    gpmodel = ca.Function(name, [x], [y_mu, y_std])
    
    return gpmodel
