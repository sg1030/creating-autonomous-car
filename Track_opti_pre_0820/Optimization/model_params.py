import numpy as np
from Optimization.params import *


vx_max = 7.0
vx_min = 1.0
u_a_min = 4.0
psi_dot_max = np.inf #np.pi
u_steer_max = 0.0
vy_max = 0.05

# Friction-circle (g-g) model settings — used by VehicleKinematicsGG.
# u_steer_max must be > 0 here, otherwise delta is pinned at 0 and the
# friction-circle constraint has nothing to limit (a_lat = v^2*tan(0)/L = 0).
ay_max = 3.5          # lateral accel limit [m/s^2]
u_steer_max_gg = 0.4  # steering limit for the GG model [rad]

opt_params_dyn = ModelParams_Dyn(
    vx_max= vx_max,
    vx_min= vx_min,
    vy_max= vy_max,
    vy_min= -vy_max,
    psi_dot_max= psi_dot_max,
    psi_dot_min= -psi_dot_max,
    u_a_min= -u_a_min,
    u_a_max= u_a_min,
    u_steer_max=u_steer_max,
    u_steer_min=-u_steer_max)

opt_params_kin = ModelParams_Kin(
    v_max= vx_max,
    v_min= vx_min,
    u_a_min= -u_a_min,
    u_a_max= u_a_min,
    u_steer_max=u_steer_max,
    u_steer_min=-u_steer_max)

opt_params_kin_gg = ModelParams_KinGG(
    v_max= vx_max,
    v_min= vx_min,
    u_a_min= -u_a_min,
    u_a_max= u_a_min,
    u_steer_max= u_steer_max_gg,
    u_steer_min= -u_steer_max_gg,
    ax_max= u_a_min,
    ay_max= ay_max)

opt_params_kin_delayed = ModelParams_KinDelayed(
    v_max=vx_max,
    v_min=vx_min,
    a_max=u_a_min,
    a_min=-u_a_min,
    delta_max=u_steer_max,
    delta_min=-u_steer_max,
    u_a_max=u_a_min,
    u_a_min=-u_a_min,
    u_steer_max=u_steer_max,
    u_steer_min=-u_steer_max)

mpc_prams_kin = MPCParams_kin(
    Qs= 0, 
    Qey= 1,
    Qv= 0,
    Qepsi= 1, #vy
    Qw = 0.1,
    R_a= 0.1,
    R_delta= 0.1,

    vx_max= vx_max,
    vx_min= vx_min,
    u_a_min= -u_a_min,
    u_a_max= u_a_min,
    u_steer_max=u_steer_max,
    u_steer_min=-u_steer_max)

mpc_prams_dyn = MPCParams_dyn(
    Qs= 0, 
    Qey= 5,
    Qv= 10,
    Qepsi= 0, #vy
    Qw = 0.0,
    R_a= 0.0,
    R_delta= 0.0,

    vx_max= vx_max,
    vx_min= vx_min,
    vy_max= vy_max,
    vy_min= -vy_max,
    psi_dot_max= psi_dot_max,
    psi_dot_min= -psi_dot_max,
    u_a_min= -u_a_min,
    u_a_max= u_a_min,
    u_steer_max=u_steer_max,
    u_steer_min=-u_steer_max)