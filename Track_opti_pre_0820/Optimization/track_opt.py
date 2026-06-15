import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class MiminumTimeOptimization():
    def __init__(self, dynamics, track, control_params, model):
        
        ''' Define Parameters '''
        # Track parameters
        self.track = track
        self.track_length = track.track_length

        ''' Define Variables for optimization '''
        # Dynamics parameters
        self.dynamics = dynamics
        self.model = model

        # self.dt = dynamics.dt
        self.Nx = dynamics.n_q # number of states
        self.Nu = dynamics.n_u # number of inputs
        self.N = dynamics.N
        self.ds = self.track_length/(self.N-1)
    
        # Optimization params
        self.control_params = control_params

        # State Constraints 
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.state_ub[0] = (self.track.track_width*self.track.track_width_scale)/2
        self.state_lb[0] = -(self.track.track_width*self.track.track_width_scale)/2

        # Input Constraints
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb

        self.X = ca.SX.sym('X', (self.N + 1)*self.Nx)
        self.U = ca.SX.sym('U', self.N*self.Nu)

        self.cost = 0.0
        self.const = []

    def solve_optimization(self, x_ws=None, u_ws=None, error = None):
        
        self.cost = 0.0
        self.const = []

        if self.model == 'Kin' or self.model == 'KinGG':
            self.cost = self.X[self.Nx*(self.N) + 3]
        else:
            self.cost = self.X[self.Nx*(self.N) + 5]  # 마지막 state's time
            
            # for t in range(self.N):
            #     # self.cost += (self.X[self.Nx*(t+1) + 5] - self.X[self.Nx*(t) + 5])
            #     self.cost += 1e-3 * ca.sqrt((self.X[self.Nx*(t+1) + 3]- self.X[self.Nx*(t) + 3])**2)  # vy²
            #     self.cost += 1e-3 * ca.sqrt((self.X[self.Nx*(t+1) + 4]- self.X[self.Nx*(t) + 4])**2)  # psidot²
            #     # self.cost += 1e-3 * ca.sqrt(self.X[self.Nx*(t) + 4]**2)  # psidot²


        ## Constraint for dynamics
        for t in range(self.N):
            current_x = self.X[self.Nx*t:self.Nx*(t+1)]
            current_u = self.U[self.Nu*t:self.Nu*(t+1)]

            if error is not None:
                integrated = self.dynamics.f_d_rk4(current_x, current_u, self.ds, self.track.curv_center[t]) + error[t]
            else:
                integrated = self.dynamics.f_d_rk4(current_x, current_u, self.ds, self.track.curv_center[t])

            self.const = ca.vertcat(self.const, self.X[self.Nx*(t+1):self.Nx*(t+2)]-integrated)

        self.const = ca.vertcat(self.const, self.X[:(self.Nx-1)]-self.X[self.Nx*(self.N):-1])

        ## Friction-circle (g-g) constraint: (a/ax_max)^2 + (a_lat/ay_max)^2 <= 1
        self.n_fc = 0
        if self.model == 'KinGG':
            for t in range(self.N):
                current_x = self.X[self.Nx*t:self.Nx*(t+1)]
                current_u = self.U[self.Nu*t:self.Nu*(t+1)]
                self.const = ca.vertcat(self.const, self.dynamics.friction_circle(current_x, current_u))
            self.n_fc = self.N

        ## Constraint for state (upper bound and lower bound)
        self.lbx = list(self.state_lb)
        self.ubx = list(self.state_ub)
        self.ubx[-1] = 0.0
        for t in range(self.N):
            self.ubx +=  list(self.state_ub)
            self.lbx +=  list(self.state_lb)

        self.ubx +=  list(self.input_ub)*self.N
        self.lbx +=  list(self.input_lb)*self.N

        self.lbg_dyanmics = [0]*(self.Nx*(self.N)) +[0]*(self.Nx-1) + [0]*self.n_fc
        self.ubg_dyanmics = [0]*(self.Nx*(self.N)) +[0]*(self.Nx-1) + [1]*self.n_fc

        opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}
        nlp = {'x':ca.vertcat(self.X,self.U), 'f':self.cost, 'g':self.const }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        if x_ws is None:
            x_ws = np.ones((self.N+1, self.Nx))
            u_ws = np.zeros((self.N, self.Nu))
        u_ws = np.concatenate(u_ws)
        x_init = np.concatenate(x_ws)

        x_init = ca.vertcat(x_init, u_ws)    
        
        sol = self.solver(x0=x_init, lbx = self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)

        x = sol['x']
        g = sol['g']

        
        x_pred = np.zeros((self.N+1, self.Nx))
        u_pred = np.zeros((self.N, self.Nu))

        idx = (self.N+1)*self.Nx
        for i in range(0,self.N):
            u_pred[i, 0] = x[idx+self.Nu*i].__float__()
            u_pred[i, 1] = x[idx+self.Nu*i + 1].__float__()

        for i in range(0, self.N+1):
            for j in range(self.Nx):
                x_pred[i, j] = x[self.Nx*i + j].__float__()

        return x_pred, u_pred
