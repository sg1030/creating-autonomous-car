import numpy as np
import casadi as ca
import matplotlib.pyplot as plt



class VehicleDynamics():
    def __init__(self, track, dt=0.1, N=70):

        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

        self.dt = dt
        self.t0 = 0.0
        self.M = 1 # RK4 integration steps
        self.h = self.dt/self.M
        self.N = N

        ''' Define the bicycle dynamics
            x = [s, ey, epsi, vx, vy, psidot, t]
            u = [a, delta] longitudinal acceleration, steering angle'''

        # Number of states and input
        self.n_q = 6 # ey, epsi, vx, vy, psidot, t
        self.n_u = 2 # a,delta

        # Model Prameter
        self.g = 9.81 #gravity
        self.m = 3.0 #mass
        self.wheel_friction = 1.0
        self.drag = 0.0

        self.L_f = 0.14 #wheel_dist_front
        self.L_r = 0.14 #wheel_dist_rear

        self.C = 1.3
        self.B = 1.1
        self.Df = self.wheel_friction*self.m*self.g * self.L_r / (self.L_r + self.L_f)
        self.Dr = self.wheel_friction*self.m*self.g * self.L_f / (self.L_r + self.L_f)
        self.Iz = 0.024
        
        # State
        self.sym_ey = ca.SX.sym('ey')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_vx = ca.SX.sym('vx')
        self.sym_vy = ca.SX.sym('vy')
        self.sym_psidot = ca.SX.sym('psidot')
        self.sym_t = ca.SX.sym('t')

        # Track parameters
        self.sym_u_a = ca.SX.sym('a')
        self.sym_u_s = ca.SX.sym('delta')



    def update_dynamics(self,s,u,curv):
        # Dynamic Bicycle Model
        # s = [ey, epsi, vx, vy, psidot, t]
        # u = [a, delta]
        
        # Slip angles and Pacejka tire forces
        sym_alpha_f = -ca.atan2(
            (s[3] + self.L_f * s[4]) * ca.cos(u[1]) - s[2] * ca.sin(u[1]),
            s[2] * ca.cos(u[1]) + (s[3]+ self.L_f * s[4]) * ca.sin(u[1]))
        sym_alpha_r = -ca.atan2(s[3]- self.L_r * s[4], s[2])
        sym_fyf = self.Df * ca.sin(self.C * ca.atan(self.B * sym_alpha_f))
        sym_fyr = self.Dr * ca.sin(self.C * ca.atan(self.B * sym_alpha_r))

        self.sym_ax = u[0] - curv * s[2] - sym_fyf * ca.sin(u[1]) / self.m
        self.sym_ay = (sym_fyf * ca.cos(u[1]) + sym_fyr) / self.m
        self.sym_wz = (self.L_f * sym_fyf * ca.cos(u[1]) - self.L_r * sym_fyr) / self.Iz

        self.sym_dt = 1/((s[2] * ca.cos(s[1]) - s[3] * ca.sin(s[1])) / (1 - s[0] * curv))
        self.sym_dey = (s[2] * ca.sin(s[1]) + s[3] * ca.cos(s[1]))*self.sym_dt
        self.sym_depsi = (s[4] - curv * (s[2] * ca.cos(s[1])
                                          - s[3] * ca.sin(s[1])) / (1 - s[0] * curv))*self.sym_dt

        self.sym_dvx = (self.sym_ax + s[4] * s[3])*self.sym_dt
        self.sym_dvy = (self.sym_ay - s[4] * s[2])*self.sym_dt
        self.sym_dpsidot = self.sym_wz*self.sym_dt
        
        return ca.vertcat(self.sym_dey, self.sym_depsi, self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dt)
    
    def update_dynamics_time(self,s,u,curv):
        # Dynamic Bicycle Model
        # s = [s,ey,epsi,vx,vy,psidot]
        # u = [a, delta]
        
        # Slip angles and Pacejka tire forces
        sym_alpha_f = -ca.atan2(
            (s[4] + self.L_f * s[5]) * ca.cos(u[1]) - s[3] * ca.sin(u[1]),
            s[3] * ca.cos(u[1]) + (s[4]+ self.L_f * s[5]) * ca.sin(u[1]))
        sym_alpha_r = -ca.atan2(s[4]- self.L_r * s[5], s[3])
        sym_fyf = self.Df * ca.sin(self.C * ca.atan(self.B * sym_alpha_f))
        sym_fyr = self.Dr * ca.sin(self.C * ca.atan(self.B * sym_alpha_r))

        self.sym_ax = u[0] - curv * s[3] - sym_fyf * ca.sin(u[1]) / self.m
        # self.sym_ax = u[0] - self.drag * s[2]- sym_fyf * ca.sin(u[1]) / self.m
        self.sym_ay = (sym_fyf * ca.cos(u[1]) + sym_fyr) / self.m
        self.sym_wz = (self.L_f * sym_fyf * ca.cos(u[1]) - self.L_r * sym_fyr) / self.Iz

        self.sym_ds = ((s[3] * ca.cos(s[2]) - s[4] * ca.sin(s[2])) / (1 - s[1] * curv))
        self.sym_dey = (s[3] * ca.sin(s[2]) + s[4] * ca.cos(s[2]))
        self.sym_depsi = (s[5] - curv * (s[3] * ca.cos(s[2])
                                          - s[4] * ca.sin(s[2])) / (1 - s[1] * curv))

        self.sym_dvx = (self.sym_ax + s[5] * s[4])
        self.sym_dvy = (self.sym_ay - s[5] * s[3])
        self.sym_dpsidot = self.sym_wz
        
        return ca.vertcat(self.sym_ds, self.sym_dey, self.sym_depsi, self.sym_dvx, self.sym_dvy, self.sym_dpsidot)
    

    def f_d_rk4(self, x, u, ds, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = ds/self.M
        for i in range(self.M):
            a1 = self.update_dynamics(x_p, u, curv)
            a2 = self.update_dynamics(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6 
        return x_p

    def f_d_rk4_time(self, x, u, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = self.dt/self.M
        for i in range(self.M):
            a1 = self.update_dynamics_time(x_p, u, curv)
            a2 = self.update_dynamics_time(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics_time(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics_time(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6 
        return x_p


class VehicleKinematics():
    def __init__(self, track, dt=0.1, N=70):

        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

        self.dt = dt
        self.t0 = 0.0
        self.M = 1 # RK4 integration steps
        self.h = self.dt/self.M
        self.N = N

        ''' Define the bicycle dynamics
            x = [s, ey, epsi, v, t]
            u = [a, delta] longitudinal acceleration, steering angle'''

        # Number of states and input
        self.n_q = 4 # ey, epsi,v, t
        self.n_u = 2 # a,delta

        # Model Prame
        self.drag = 0.00

        self.L = 1.0 #Length
        self.L_f = 0.5 #wheel_dist_front
        self.L_r = 0.5 #wheel_dist_rear

        # State
        self.sym_ey = ca.SX.sym('ey')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_v = ca.SX.sym('v')
        self.sym_t = ca.SX.sym('t')

        # Track parameters
        self.sym_u_a = ca.SX.sym('a')
        self.sym_u_s = ca.SX.sym('delta')
        

    def update_dynamics(self,s,u,curv):
        # Dynamic Bicycle Model in s domain
        # s = [ey, epsi, v, t]
        # u = [a, delta]
        
        self.sym_dt = 1/(s[2] * ca.cos(s[1]) / (1 - s[0]* curv))
        self.sym_dey = (s[2] * ca.sin(s[1]))*self.sym_dt
        self.sym_depsi = (s[2]*ca.tan(u[1])/(self.L))*self.sym_dt
        self.sym_dv = (u[0] - s[2]*curv)*self.sym_dt
        
        return ca.vertcat(self.sym_dey, self.sym_depsi, self.sym_dv, self.sym_dt)
    
    
    def update_dynamics_time(self,s,u,curv):
        # Dynamic Bicycle Model in time domain
        # s = [s, ey, epsi, vx]
        # u = [a, delta]

        self.sym_ds = s[3] * ca.cos(s[2]) / (1 - s[1]* curv)
        self.sym_dey = (s[3] * ca.sin(s[2]))
        self.sym_depsi = (s[3]*ca.tan(u[1])/(self.L))
        self.sym_dv = (u[0] - s[3]*curv)
        
        return ca.vertcat(self.sym_ds, self.sym_dey, self.sym_depsi, self.sym_dv)

    def f_d_rk4(self, x, u, ds, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = ds/self.M
        for i in range(self.M):
            a1 = self.update_dynamics(x_p, u, curv)
            a2 = self.update_dynamics(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6 
        return x_p



class VehicleKinematicsGG():
    """Kinematic bicycle model with friction-circle (g-g) lateral accel limit.

    State (spatial domain): x = [ey, epsi, v, t]
    Input:                  u = [a, delta]

    The vehicle's instantaneous path curvature is kappa = tan(delta)/L, so its
    lateral acceleration is a_lat = v^2 * tan(delta) / L. The friction-circle
    constraint couples longitudinal and lateral grip:

        (a / ax_max)^2 + (a_lat / ay_max)^2 <= 1

    forcing the optimizer to brake before corners instead of understeering
    ("pushing out") through them. The constraint itself is added in
    track_opt.py via friction_circle().
    """
    def __init__(self, track, dt=0.1, N=70, ax_max=1.0, ay_max=3.5):
        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

        self.dt = dt
        self.t0 = 0.0
        self.M = 1  # RK4 integration steps
        self.h = self.dt / self.M
        self.N = N

        # Number of states and input
        self.n_q = 4  # ey, epsi, v, t
        self.n_u = 2  # a, delta

        # Model parameters
        self.drag = 0.00
        self.L = 1.0    # Length
        self.L_f = 0.5  # wheel_dist_front
        self.L_r = 0.5  # wheel_dist_rear

        # Friction-circle limits
        self.ax_max = ax_max  # longitudinal accel limit [m/s^2]
        self.ay_max = ay_max  # lateral accel limit [m/s^2]

        # State
        self.sym_ey = ca.SX.sym('ey')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_v = ca.SX.sym('v')
        self.sym_t = ca.SX.sym('t')

        # Track parameters
        self.sym_u_a = ca.SX.sym('a')
        self.sym_u_s = ca.SX.sym('delta')

    def update_dynamics(self, s, u, curv):
        # Kinematic bicycle model in s domain (identical to VehicleKinematics)
        # s = [ey, epsi, v, t]
        # u = [a, delta]
        self.sym_dt = 1 / (s[2] * ca.cos(s[1]) / (1 - s[0] * curv))
        self.sym_dey = (s[2] * ca.sin(s[1])) * self.sym_dt
        self.sym_depsi = (s[2] * ca.tan(u[1]) / (self.L)) * self.sym_dt
        self.sym_dv = (u[0] - s[2] * curv) * self.sym_dt

        return ca.vertcat(self.sym_dey, self.sym_depsi, self.sym_dv, self.sym_dt)

    def lateral_accel(self, s, u):
        # a_lat = v^2 * tan(delta) / L  (centripetal accel of the bicycle model)
        return s[2] ** 2 * ca.tan(u[1]) / self.L

    def friction_circle(self, s, u):
        # (a / ax_max)^2 + (a_lat / ay_max)^2  ->  constrained <= 1 in track_opt
        a_lon = u[0]
        a_lat = self.lateral_accel(s, u)
        return (a_lon / self.ax_max) ** 2 + (a_lat / self.ay_max) ** 2

    def f_d_rk4(self, x, u, ds, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = ds / self.M
        for i in range(self.M):
            a1 = self.update_dynamics(x_p, u, curv)
            a2 = self.update_dynamics(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p



class VehicleKinematicsDelayed():
    """Kinematic bicycle model with first-order actuator lag.

    State  (spatial domain): x = [ey, epsi, v, a_actual, delta_actual, t]
    Input:                   u = [a_cmd, delta_cmd]
    Actuator lag: d(u_actual)/dt = (u_cmd - u_actual) / tau
    """
    def __init__(self, track, dt=0.1, N=70, tau_a=0.1, tau_delta=0.1):
        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

        self.dt = dt
        self.t0 = 0.0
        self.M = 1
        self.h = self.dt / self.M
        self.N = N

        self.n_q = 6  # ey, epsi, v, a_actual, delta_actual, t
        self.n_u = 2  # a_cmd, delta_cmd

        self.drag = 0.00
        self.L = 1
        self.L_f = 0.5
        self.L_r = 0.5

        self.tau_a = tau_a          # acceleration actuator time constant [s]
        self.tau_delta = tau_delta  # steering actuator time constant [s]

    def update_dynamics(self, s, u, curv):
        # s = [ey, epsi, v, a_actual, delta_actual, t]
        # u = [a_cmd, delta_cmd]
        sym_dt = 1 / (s[2] * ca.cos(s[1]) / (1 - s[0] * curv))
        sym_dey     = s[2] * ca.sin(s[1]) * sym_dt
        sym_depsi   = s[2] * ca.tan(s[4]) / self.L * sym_dt
        sym_dv      = (s[3] - s[2] * curv) * sym_dt
        sym_da      = (u[0] - s[3]) / self.tau_a * sym_dt
        sym_ddelta  = (u[1] - s[4]) / self.tau_delta * sym_dt
        return ca.vertcat(sym_dey, sym_depsi, sym_dv, sym_da, sym_ddelta, sym_dt)

    def f_d_rk4(self, x, u, ds, curv):
        x_p = x
        h = ds / self.M
        for i in range(self.M):
            a1 = self.update_dynamics(x_p, u, curv)
            a2 = self.update_dynamics(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p


class VehicleDynamicsTrue(): #Time domain
    def __init__(self, track, dt=0.1, N=70):

        self.track = track

        jac_opts = dict(enable_fd=False, enable_jacobian=False, enable_forward=False, enable_reverse=False)
        self.options = lambda fn_name: dict(jit=False, **jac_opts)

        self.dt = dt
        self.t0 = 0.0
        self.M = 1 # RK4 integration steps
        self.h = self.dt/self.M
        self.N = N

        ''' Define the bicycle dynamics
            x = [s, ey, epsi, vx, vy, psidot, t]
            u = [a, delta] longitudinal acceleration, steering angle'''

        # Number of states and input
        self.n_q = 6 # ey, epsi, vx, vy, psidot, t
        self.n_u = 2 # a,delta

        # Model Prameter
        self.g = 9.81 #gravity
        self.m = 3.0 #mass
        self.wheel_friction = 0.8
        self.drag = 0.0

        self.L_f = 0.14 #wheel_dist_front
        self.L_r = 0.14 #wheel_dist_rear

        self.C = 1.2
        self.B = 1.0
        self.Df = self.wheel_friction*self.m*self.g * self.L_r / (self.L_r + self.L_f)
        self.Dr = self.wheel_friction*self.m*self.g * self.L_f / (self.L_r + self.L_f)
        self.Iz = 0.017
        
        # State
        self.sym_ey = ca.SX.sym('ey')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_vx = ca.SX.sym('vx')
        self.sym_vy = ca.SX.sym('vy')
        self.sym_psidot = ca.SX.sym('psidot')
        self.sym_t = ca.SX.sym('t')

        # Track parameters
        self.sym_u_a = ca.SX.sym('a')
        self.sym_u_s = ca.SX.sym('delta')


    def update_dynamics(self,s,u,curv):
        # Dynamic Bicycle Model
        # s = [ey, epsi, vx, vy, psidot, t]
        # u = [a, delta]
        
        # Slip angles and Pacejka tire forces
        sym_alpha_f = -ca.atan2(
            (s[3] + self.L_f * s[4]) * ca.cos(u[1]) - s[2] * ca.sin(u[1]),
            s[2] * ca.cos(u[1]) + (s[3]+ self.L_f * s[4]) * ca.sin(u[1]))
        sym_alpha_r = -ca.atan2(s[3]- self.L_r * s[4], s[2])
        sym_fyf = self.Df * ca.sin(self.C * ca.atan(self.B * sym_alpha_f))
        sym_fyr = self.Dr * ca.sin(self.C * ca.atan(self.B * sym_alpha_r))

        self.sym_ax = u[0] - curv * s[2] - sym_fyf * ca.sin(u[1]) / self.m
        self.sym_ay = (sym_fyf * ca.cos(u[1]) + sym_fyr) / self.m
        self.sym_wz = (self.L_f * sym_fyf * ca.cos(u[1]) - self.L_r * sym_fyr) / self.Iz

        self.sym_dt = 1/((s[2] * ca.cos(s[1]) - s[3] * ca.sin(s[1])) / (1 - s[0] * curv))
        self.sym_dey = (s[2] * ca.sin(s[1]) + s[3] * ca.cos(s[1]))*self.sym_dt
        self.sym_depsi = (s[4] - curv * (s[2] * ca.cos(s[1])
                                          - s[3] * ca.sin(s[1])) / (1 - s[0] * curv))*self.sym_dt

        self.sym_dvx = (self.sym_ax + s[4] * s[3])*self.sym_dt
        self.sym_dvy = (self.sym_ay - s[4] * s[2])*self.sym_dt
        self.sym_dpsidot = self.sym_wz*self.sym_dt
        
        return ca.vertcat(self.sym_dey, self.sym_depsi, self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dt)
    
    def update_dynamics_time(self,s,u,curv):
        # Dynamic Bicycle Model
        # s = [s,ey,epsi,vx,vy,psidot]
        # u = [a, delta]
        
        # Slip angles and Pacejka tire forces
        sym_alpha_f = -ca.atan2(
            (s[4] + self.L_f * s[5]) * ca.cos(u[1]) - s[3] * ca.sin(u[1]),
            s[3] * ca.cos(u[1]) + (s[4]+ self.L_f * s[5]) * ca.sin(u[1]))
        sym_alpha_r = -ca.atan2(s[4]- self.L_r * s[5], s[3])
        sym_fyf = self.Df * ca.sin(self.C * ca.atan(self.B * sym_alpha_f))
        sym_fyr = self.Dr * ca.sin(self.C * ca.atan(self.B * sym_alpha_r))

        self.sym_ax = u[0] - curv * s[3] - sym_fyf * ca.sin(u[1]) / self.m
        # self.sym_ax = u[0] - self.drag * s[2]- sym_fyf * ca.sin(u[1]) / self.m
        self.sym_ay = (sym_fyf * ca.cos(u[1]) + sym_fyr) / self.m
        self.sym_wz = (self.L_f * sym_fyf * ca.cos(u[1]) - self.L_r * sym_fyr) / self.Iz

        self.sym_ds = ((s[3] * ca.cos(s[2]) - s[4] * ca.sin(s[2])) / (1 - s[1] * curv))
        self.sym_dey = (s[3] * ca.sin(s[2]) + s[4] * ca.cos(s[2]))
        self.sym_depsi = (s[5] - curv * (s[3] * ca.cos(s[2])
                                          - s[4] * ca.sin(s[2])) / (1 - s[1] * curv))

        self.sym_dvx = (self.sym_ax + s[5] * s[4])
        self.sym_dvy = (self.sym_ay - s[5] * s[3])
        self.sym_dpsidot = self.sym_wz
        
        return ca.vertcat(self.sym_ds, self.sym_dey, self.sym_depsi, self.sym_dvx, self.sym_dvy, self.sym_dpsidot)
    

    def f_d_rk4(self, x, u, ds, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = ds/self.M
        for i in range(self.M):
            a1 = self.update_dynamics(x_p, u, curv)
            a2 = self.update_dynamics(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6 
        return x_p

    def f_d_rk4_time(self, x, u, curv):
        '''Discrete nonlinear dynamics (RK4 approx.)'''
        x_p = x
        h = self.dt/self.M
        for i in range(self.M):
            a1 = self.update_dynamics_time(x_p, u, curv)
            a2 = self.update_dynamics_time(x_p + (h / 2) * a1, u, curv)
            a3 = self.update_dynamics_time(x_p + (h / 2) * a2, u, curv)
            a4 = self.update_dynamics_time(x_p + h * a3, u, curv)
            x_p += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6 
        return x_p