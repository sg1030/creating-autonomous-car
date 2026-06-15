from dataclasses import dataclass, field 
import numpy as np


@dataclass
class ModelParams_Dyn():
    n: int = field(default=6) # dimension state space (ey,epsi,vx,vy,psidot,t)
    d: int = field(default=2) # dimension input space

    #state constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)
    vy_max: float = field(default=np.inf)
    vy_min: float = field(default=-np.inf)
    psi_dot_max: float = field(default=np.inf)
    psi_dot_min: float = field(default=-np.inf)
    t_max: float = field(default=np.inf)
    t_min: float = field(default=0.0)

    #input constraint
    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.ey_max,
                                self.e_psi_max,
                                self.vx_max,
                                self.vy_max,
                                self.psi_dot_max,
                                self.t_max])
        self.state_lb = np.array([self.ey_min,
                                self.e_psi_min,
                                self.vx_min,
                                self.vy_min,
                                self.psi_dot_min,
                                self.t_min])
        
        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

@dataclass
class ModelParams_Kin():
    n: int = field(default=4) # dimension state space (ey,epsi,v,t)
    d: int = field(default=2) # dimension input space

    #state constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    v_max: float = field(default=np.inf)
    v_min: float = field(default=-np.inf)
    t_max: float = field(default=np.inf)
    t_min: float = field(default=0.0)

    #input constraint
    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.ey_max,
                                self.e_psi_max,
                                self.v_max,
                                self.t_max])
        self.state_lb = np.array([self.ey_min,
                                self.e_psi_min,
                                self.v_min,
                                self.t_min])
        
        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

@dataclass
class ModelParams_KinGG():
    n: int = field(default=4)  # ey, epsi, v, t
    d: int = field(default=2)  # a, delta

    #state constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    v_max: float = field(default=np.inf)
    v_min: float = field(default=-np.inf)
    t_max: float = field(default=np.inf)
    t_min: float = field(default=0.0)

    #input constraint
    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    # friction-circle limits (used by VehicleKinematicsGG + track_opt)
    ax_max: float = field(default=3.0)
    ay_max: float = field(default=3.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.ey_max,
                                self.e_psi_max,
                                self.v_max,
                                self.t_max])
        self.state_lb = np.array([self.ey_min,
                                self.e_psi_min,
                                self.v_min,
                                self.t_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

@dataclass
class ModelParams_KinDelayed():
    n: int = field(default=6)  # ey, epsi, v, a_actual, delta_actual, t
    d: int = field(default=2)

    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    v_max: float = field(default=np.inf)
    v_min: float = field(default=-np.inf)
    a_max: float = field(default=3.0)
    a_min: float = field(default=-3.0)
    delta_max: float = field(default=0.5)
    delta_min: float = field(default=-0.5)
    t_max: float = field(default=np.inf)
    t_min: float = field(default=0.0)

    u_a_max: float = field(default=3.0)
    u_a_min: float = field(default=-3.0)
    u_steer_max: float = field(default=0.5)
    u_steer_min: float = field(default=-0.5)

    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf * np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf * np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf * np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf * np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf * np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf * np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.ey_max, self.e_psi_max, self.v_max,
                                   self.a_max, self.delta_max, self.t_max])
        self.state_lb = np.array([self.ey_min, self.e_psi_min, self.v_min,
                                   self.a_min, self.delta_min, self.t_min])
        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])


@dataclass
class MPCParams_kin():
    n: int = field(default=4) # dimension state space (s,ey,epsi,vx)
    d: int = field(default=2) # dimension input space
    
    # cost Pamarater
    Qs: float = field(default=5.0)
    Qey: float = field(default=5.0)
    Qv: float = field(default=5.0)
    Qepsi: float = field(default=5.0)
    Qw: float = field(default=5.0)
    R_a: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    #state constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)

    #input constraint
    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.s_max,
                                self.ey_max,
                                self.e_psi_max,
                                self.vx_max])
        self.state_lb = np.array([self.s_min,
                                self.ey_min,
                                self.e_psi_min,
                                self.vx_min])
        
        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])

@dataclass
class MPCParams_dyn():
    n: int = field(default=6) # dimension state space (s,ey,epsi,vx,vy,psidot)
    d: int = field(default=2) # dimension input space
    
    # cost Pamarater
    Qs: float = field(default=5.0)
    Qey: float = field(default=5.0)
    Qv: float = field(default=5.0)
    Qepsi: float = field(default=5.0)
    Qw: float = field(default=5.0)
    R_a: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    #state constraint
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    ey_max: float = field(default=np.inf)
    ey_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=2*np.pi)
    e_psi_min: float = field(default=-2*np.pi)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)
    vy_max: float = field(default=np.inf)
    vy_min: float = field(default=-np.inf)
    psi_dot_max: float = field(default=np.inf)
    psi_dot_min: float = field(default=-np.inf)

    #input constraint
    u_a_max: float          = field(default = 3.0)
    u_a_min: float          = field(default = -3.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.s_max,
                                self.ey_max,
                                self.e_psi_max,
                                self.vx_max,
                                self.vy_max,
                                self.psi_dot_max])
        self.state_lb = np.array([self.s_min,
                                self.ey_min,
                                self.e_psi_min,
                                self.vx_min,
                                self.vy_min,
                                self.psi_dot_min])
        
        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])


