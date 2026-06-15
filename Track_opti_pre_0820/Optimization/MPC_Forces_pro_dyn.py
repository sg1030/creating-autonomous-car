import sys
sys.path.insert(0, '/home/im/Desktop/Forces') 

import numpy as np
import forcespro
import forcespro.nlp
import shutil

import casadi as ca
import sys
import os, contextlib

@contextlib.contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)




class MPC:
    def __init__(self, dynamics, track, control_params, x0, iteration=-1, dir = None, gpmodels=None, xscaler=None, yscaler=None, label = None, failure = False):
        self.dynamics = dynamics
        self.track = track
        self.control_params = control_params
        self.x0 = x0
        

        self.gpmodels = gpmodels
        self.xscaler = xscaler
        self.yscaler = yscaler
        self.use_gp = gpmodels is not None

        self.Nx = dynamics.n_q
        self.Nu = dynamics.n_u
        self.N = dynamics.N
        self.dt = dynamics.dt

        self.Q = np.diag([control_params.Qs, control_params.Qey, control_params.Qepsi, control_params.Qv])
        self.R = np.diag([control_params.R_a, control_params.R_delta])

        self.input_lb = np.array([control_params.u_a_min, control_params.u_steer_min]) 
        self.input_ub = np.array([control_params.u_a_max, control_params.u_steer_max])
        self.state_lb = np.array([-np.inf, -(self.track.track_width*(self.track.track_width_scale+0.1))/2, -np.pi, 0.0, control_params.vy_min, control_params.psi_dot_min])  
        self.state_ub = np.array([np.inf, (self.track.track_width*(self.track.track_width_scale+0.1))/2, np.pi, control_params.vx_max, control_params.vy_max, control_params.psi_dot_max])

        self.x_ws = np.tile(x0, (self.N+1, 1))
        self.u_ws = np.zeros((self.N, self.Nu))


        if gpmodels is not None:
            # self.vxgp = gpmodels['vx']
            self.vygp = gpmodels['vy']
            self.omegagp = gpmodels['omega']
            self.vxxss = gpmodels['xscaler'].scale_
            self.vxxsm = gpmodels['xscaler'].mean_
            self.vyyss = gpmodels['vyyscaler'].scale_
            self.vyysm = gpmodels['vyyscaler'].mean_
            self.wyss = gpmodels['omegayscaler'].scale_
            self.wysm = gpmodels['omegayscaler'].mean_

        self.iteration=iteration
        self.label=label
        self.dir = dir
        # if label is not None:
        #     if gpmodels is not None:
        #         print(f"solver_Path,{dir}/FORCES_MPC_GP_{label}_{iteration}")
        #         forces_solver_path = f"{dir}/FORCES_MPC_GP_{label}_{iteration}"
        #     else:
        #         forces_solver_path = f"{dir}/FORCES_MPC_{label}"
        # else:
        #     forces_solver_path = f"{dir}/FORCES_MPC"

        # if not failure and (os.path.exists(forces_solver_path) and os.path.isdir(forces_solver_path)):
        #     self.solver = forcespro.nlp.Solver.from_directory(forces_solver_path)
        # else:
        self.solver = self._setup_forces_solver()

    def _setup_forces_solver(self):
        model = forcespro.nlp.SymbolicModel(self.N)
        model.nvar = self.Nx + self.Nu
        model.neq = self.Nx
        
        # 파라미터: reference + curv
        model.npar = self.Nx + 2
            
        model.E = np.hstack([np.zeros((self.Nx, self.Nu)), np.eye(self.Nx)])

        def objective(z, p):
            u = z[:self.Nu]  
            x = z[self.Nu:]  
            
            # Progress maximization
            # cost_s = -x[0]**2  
                
            cost_ey = self.Q[1,1] * (p[1] - x[1])**2
            cost_epsi = self.Q[2,2] * (p[2] - x[2])**2
            cost_vx = self.Q[3,3] * (p[3] - x[3])**2 

            cost_a = self.R[0,0] * u[0]**2
            cost_delta = self.R[1,1] * u[1]**2
                    
            return cost_ey + cost_epsi + cost_vx + cost_a + cost_delta 
        
        model.objective = objective

        def dynamics(z, p):
            states = z[self.Nu:]
            inputs = z[:self.Nu]
            curv = p[4]  # 곡률
            
            # 표준 dynamics 예측
            x_next_nominal = self.dynamics.f_d_rk4_time(states, inputs, curv)

            return x_next_nominal


        def dynamics_with_gp(z, p):
            states = z[self.Nu:]
            inputs = z[:self.Nu]
            curv = p[4]

            x_next_nominal = self.dynamics.f_d_rk4_time(states, inputs, curv)

            vx = states[3]
            vy = states[4]
            psidot = states[5]
            a = inputs[0]
            delta = inputs[1]

            gp_input = ca.vertcat(vx, vy, psidot, a, delta)
            gp_input_scaled = (gp_input - ca.DM(self.vxxsm)) / ca.DM(self.vxxss)

            # vx_err = self.vxgp(gp_input_scaled)[0]
            vy_err = self.vygp(gp_input_scaled)[0]*ca.DM(self.vyyss) + ca.DM(self.vyysm)
            omega_err = self.omegagp(gp_input_scaled)[0]*ca.DM(self.wyss) + ca.DM(self.wysm)

            x_next = ca.vertcat(
                x_next_nominal[0],
                x_next_nominal[1],
                x_next_nominal[2],
                x_next_nominal[3], # vx_err,
                x_next_nominal[4] + vy_err,
                x_next_nominal[5] + omega_err
            )

            return x_next
        
        if self.gpmodels is not None:
            model.eq = dynamics_with_gp
        else:
            model.eq = dynamics

        model.lb = np.concatenate([self.input_lb, self.state_lb])
        model.ub = np.concatenate([self.input_ub, self.state_ub])
        model.xinitidx = range(self.Nu, self.Nu + self.Nx)

        model.bfgs_init = 2.5 * np.identity(model.nvar)

        if self.gpmodels is not None:
            if self.label is not None:
                solvername = f"FORCES_MPC_GP_{self.label}_{self.iteration}"
            else:
                solvername = f"FORCES_MPC_GP_{self.iteration}"
        else:
            if self.label is not None:
                solvername = f"FORCES_MPC_{self.label}"
            else:
                solvername = 'FORCES_MPC'

        codeoptions = forcespro.CodeOptions(solvername)
        codeoptions.maxit = 700
        codeoptions.printlevel = 0
        codeoptions.optlevel = 0
        codeoptions.cleanup = False
        codeoptions.timing = 1
        codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.solvemethod = 'PDIP_NLP'
        codeoptions.sqp_nlp.maxqps = 10
        codeoptions.sqp_nlp.reg_hessian = 1e-8
        
        # Tolerance 설정 - 매우 관대하게
        codeoptions.nlp.TolStat = 1e-2
        codeoptions.nlp.TolEq = 1e-2
        codeoptions.nlp.TolIneq = 1e-2
        codeoptions.nlp.TolComp = 1e-2

        # Variable elimination 비활성화
        codeoptions.noVariableElimination = 1

        solver_dir = os.path.join(self.dir, solvername)
        if os.path.exists(solver_dir):
            print("Remove the orignal folder")
            shutil.rmtree(solver_dir)

        with pushd(self.dir):
            solver = model.generate_solver(options=codeoptions)
        return solver
    
    def calc_ref_traj(self, x0):
        ref = np.zeros((self.N+1, self.Nx+2)) #state, input, curv, sdot
        dist_move = 0.0
        vel = x0[3]
        
        ind = np.searchsorted(self.track.s_center, x0[0])
        if ind >= len(self.track.s_center):
            ind = 0

        for i in range(0, self.N + 1):
            
            if i == 0:
                index = ind
            else:
                dist_move += abs(vel) * self.dt
                ind_move = int(round(dist_move / self.track.d_dist))
                index = min(ind + ind_move, len(self.track.s) - 1)

            ref[i,0] = self.track.s[index]

            if index >= len(self.track.s_center):
                index = index - len(self.track.s_center) 

            ref[i, 1:4] = [self.track.ey[index], self.track.epsi[index], self.track.v[index]]
            ref[i, 4] = self.track.curv_center[index]
            ref[i, 5] = self.track.sdot[index] if hasattr(self.track, 'sdot') else 0.0
            vel = self.track.v[index]

        return ref
    
    def warm_start(self, x_prev, u_prev):
        N, Nx = x_prev.shape  
        Nu = u_prev.shape[1]  

        x0 = np.zeros((N, Nx + Nu))
        for i in range(N):
            x0[i, :] = np.concatenate([u_prev[i], x_prev[i]])  

        return x0

    def solve_optimization_gp_dyn(self, state, u_prev):
        """GP가 통합된 dynamics를 사용한 MPC 최적화"""
        try:
            # if not self.use_gp:
            #     print("WARNING: GP model not available")
            #     return None, None, None
            
            # print(f"Solving MPC at state: s={state[0]:.2f}, ey={state[1]:.3f}, epsi={state[2]:.3f}, vx={state[3]:.2f}, vy={state[4]:.2f}, w={state[5]:.2f}")
            state = np.clip(state, self.state_lb, self.state_ub)
            # print(f"Solving MPC at clipped state: s={state[0]:.2f}, ey={state[1]:.3f}, epsi={state[2]:.3f}, vx={state[3]:.2f}, vy={state[4]:.2f}, w={state[5]:.2f}")

            nx = self.dynamics.n_q
            nu = self.dynamics.n_u
            ref_traj = self.calc_ref_traj(state)
            
            # State와 input 유효성 검증
            if not np.all(np.isfinite(state)):
                print(f"ERROR: Invalid state detected: {state}")
                return self.x_ws, self.u_ws, ref_traj
            
            if not np.all(np.isfinite(u_prev)):
                print(f"WARNING: Invalid u_prev, using zeros: {u_prev}")
                u_prev = np.zeros(self.Nu)
            
            # Warm start 초기화 - 안전한 초기화
            if self.x_ws is None or self.u_ws is None:
                # print("Initializing warm start variables")
                self.x_ws = np.tile(state, (self.N+1, 1))
                self.u_ws = np.zeros((self.N, self.Nu))
            
            # 다음 warm start 준비
            x_ws = np.zeros((self.N+1, self.Nx))
            u_ws = np.zeros((self.N, self.Nu))
            x_ws[0] = state
            
            # 이전 해가 있으면 시프트하여 사용
            if self.x_ws.shape[0] > self.N:
                x_ws[1:] = self.x_ws[1:self.N+1]
                u_ws[:-1] = self.u_ws[1:]
                u_ws[-1] = self.u_ws[-1]  # 마지막 제어입력 복사
            else:
                # 이전 해가 없으면 상태는 현재 상태로 복사, 제어입력은 0
                for i in range(1, self.N+1):
                    x_ws[i] = state
                u_ws[:] = 0.0
            
            # 파라미터 설정 (reference + curv)
            all_parameters = np.zeros((self.N, self.Nx + 2))
            for i in range(self.N):
                all_parameters[i, :] = ref_traj[i, :]
            
            # 파라미터 유효성 검증
            if not np.all(np.isfinite(all_parameters)):
                print("ERROR: Invalid parameters detected")
                return self.x_ws, self.u_ws, ref_traj
            
            # FORCESPro 형식으로 변환
            x0 = np.zeros((self.N, self.Nx + self.Nu))
            for i in range(self.N):
                x0[i, :] = np.concatenate([u_ws[i], x_ws[i+1]])
            
            all_parameters = all_parameters.reshape(-1, 1)
            x0 = x0.reshape(-1)

            # 초기화 유효성 검증
            if not np.all(np.isfinite(x0)):
                print("ERROR: Invalid initial guess")
                return self.x_ws, self.u_ws, ref_traj

            # 문제 설정
            problem = {
                'x0': x0,
                'xinit': state, 
                'all_parameters': all_parameters
            }

            # print("Calling FORCESPro solver...")
            # Solver 실행
            sol, exitflag, info = self.solver.solve(problem)
            
            # Exitflag 분석
            # print(f"Solver exitflag: {exitflag}")
            # if hasattr(info, 'it'):
            #     print(f"Solver iterations: {info.it}")
            # if hasattr(info, 'solvetime'):
            #     solver_hz = 1.0 / info.solvetime if info.solvetime > 0 else 0
            #     print(f"Solver time: {info.solvetime*1000:.2f}ms ({solver_hz:.1f} Hz)")

            if exitflag == 1:
                # 해 추출
                x_pred = np.zeros((self.N + 1, nx))
                u_pred = np.zeros((self.N, nu))
                x_pred[0, :] = state
                
                for i in range(self.N):
                    key1 = f'x{i+1:02d}'
                    key2 = f'x{i+1}'

                    if key1 in sol:
                        sol_i = sol[key1]
                    elif key2 in sol:
                        sol_i = sol[key2]
                    else:
                        print(f"WARNING: Solution key not found for stage {i}")
                        sol_i = np.concatenate([u_ws[i], x_ws[i+1]])

                    u_pred[i, :] = sol_i[:nu]
                    x_pred[i+1, :] = sol_i[nu:nu+nx]
                
                # 해 유효성 검증
                if np.all(np.isfinite(x_pred)) and np.all(np.isfinite(u_pred)):
                    # print(f"Optimization successful - Cost reduction achieved")
                    
                    # Warm start를 위해 저장
                    self.x_ws = x_pred.copy()
                    self.u_ws = u_pred.copy()
                    
                    return x_pred, u_pred, ref_traj
                else:
                    print("ERROR: Invalid solution detected, using previous solution")
                    return self.x_ws, self.u_ws, ref_traj
                    
            else:
                # 솔버 실패 원인 분석
                exitflag_meanings = {
                    0: "Maximum iterations reached",
                    -1: "Infeasible problem",
                    -2: "Unbounded problem", 
                    -6: "Setup error or numerical issues",
                    -7: "Memory error or other internal error",
                    -10: "User termination"
                }
                
                meaning = exitflag_meanings.get(exitflag, f"Unknown exitflag {exitflag}")
                print(f"Solver failed: {meaning}")
                
                # 실패 시 이전 해 사용 또는 안전한 제어입력 생성
                if self.x_ws is not None and self.u_ws is not None:
                    print("Using previous solution")
                    return self.x_ws.copy(), self.u_ws.copy(), ref_traj
                else:
                    print("No previous solution available, generating safe control")
                    # 안전한 기본 해 생성
                    x_safe = np.tile(state, (self.N+1, 1))
                    u_safe = np.zeros((self.N, self.Nu))
                    self.x_ws = x_safe
                    self.u_ws = u_safe
                    return x_safe, u_safe, ref_traj
                    
        except Exception as e:
            print(f"ERROR in solve_optimization_gp_dyn: {e}")
            import traceback
            traceback.print_exc()
            
            # 예외 발생 시 안전한 해 반환
            if hasattr(self, 'x_ws') and self.x_ws is not None:
                return self.x_ws.copy(), self.u_ws.copy(), self.calc_ref_traj(state)
            else:
                x_safe = np.tile(state, (self.N+1, 1))
                u_safe = np.zeros((self.N, self.Nu))
                return x_safe, u_safe, self.calc_ref_traj(state)
