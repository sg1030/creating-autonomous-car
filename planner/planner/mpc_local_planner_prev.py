#!/usr/bin/env python3
"""
mpc_local_planner.py  —  MPC kinematic local planner with obstacle avoidance.

State:    x = [s, ey, epsi, v]
Control:  u = [a, delta]
Dynamics: VehicleKinematics (time-domain, RK4) — matches dynamics.py exactly
Solver:   CasADi / IPOPT  (parametric NLP compiled once at startup)

ROS 2 — plug-compatible with local_planning.py:
  SUB: /scan, /vesc/odom, /global_waypoints
  PUB: /local_waypoints, /mpc_planner/markers
"""
import math
import time
import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline, PchipInterpolator

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))


def _scan_to_xy(ranges, angle_min, angle_inc):
    r_min, r_max = 0.05, 10.0
    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    return (np.where(valid, ranges * np.cos(angles), np.nan),
            np.where(valid, ranges * np.sin(angles), np.nan))


# ── Perception ────────────────────────────────────────────────────────────────

def _cluster(x, y, angle_inc):
    lambda_rad = math.radians(5.0)
    sigma = 0.06
    min_points = 5
    max_nan_gap = 10

    denom = math.sin(lambda_rad - angle_inc)
    if abs(denom) < 1e-6:
        denom = 1e-6

    clusters = []
    current = []
    prev = None
    nan_count = 0

    for i in range(len(x)):
        xi, yi = x[i], y[i]
        if not (math.isfinite(xi) and math.isfinite(yi)):
            nan_count += 1
            if nan_count > max_nan_gap:
                if len(current) >= min_points:
                    clusters.append(np.array(current))
                current = []
                prev = None
                nan_count = 0
        else:
            nan_count = 0
            if prev is None:
                current.append([xi, yi])
            else:
                r_prev = math.hypot(prev[0], prev[1])
                d_max = r_prev * math.sin(angle_inc) / denom + sigma
                jump = math.hypot(xi - prev[0], yi - prev[1])
                if jump > d_max:
                    if len(current) >= min_points:
                        clusters.append(np.array(current))
                    current = [[xi, yi]]
                else:
                    current.append([xi, yi])
            prev = (xi, yi)

    if len(current) >= min_points:
        clusters.append(np.array(current))
    return clusters


def _l_shape(clusters):
    max_obs_size = 0.60
    min_size     = 0.15
    min_edge     = 0.01
    n_angles     = 180

    thetas = np.linspace(0.0, np.pi/2 - np.pi/180, n_angles)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    obstacles = []

    for clust in clusters:
        pts = np.array(clust)
        if len(pts) < 2:
            continue

        a = pts[:, 0:1] * cos_t + pts[:, 1:2] * sin_t
        b = -pts[:, 0:1] * sin_t + pts[:, 1:2] * cos_t

        da = np.minimum(a - a.min(axis=0), a.max(axis=0) - a)
        db = np.minimum(b - b.min(axis=0), b.max(axis=0) - b)
        d  = np.minimum(da, db)

        score = np.sum(1.0 / np.maximum(d, min_edge), axis=0)
        k = int(np.argmax(score))
        c, s = cos_t[k], sin_t[k]

        ak = a[:, k]; bk = b[:, k]
        w  = float(ak.max() - ak.min())
        h  = float(bk.max() - bk.min())
        ca = 0.5 * (ak.max() + ak.min())
        cb = 0.5 * (bk.max() + bk.min())
        cx = ca * c - cb * s
        cy = ca * s + cb * c

        if max(w, h) > max_obs_size:
            continue
        raw_minor = min(w, h)
        raw_major = max(w, h)
        if raw_minor < 0.03:
            continue
        if raw_major > 0 and raw_minor < 0.08 * raw_major:
            continue

        w = max(w, min_size)
        h = max(h, min_size)
        obstacles.append((cx, cy, w, h, thetas[k]))

    return obstacles


def _tracking(obs_list, track, dt, ego, init_vel=(0.0, 0.0)):
    opp_max_lat      = 0.45   # [m] lateral gate — keeps wall clusters out
    max_misses       = 20
    sigma_pos        = 0.1
    lidar_to_base_x  = 0.27

    ex, ey_ego, eyaw = ego
    meas = None
    best_obs = None

    cands = [(cx, cy, w, h, th) for cx, cy, w, h, th in obs_list
             if cx > 0 and abs(cy) <= opp_max_lat]
    if cands:
        best_obs = min(cands, key=lambda o: math.hypot(o[0], o[1]))
        bx = best_obs[0] + lidar_to_base_x
        by = best_obs[1]
        cyaw = math.cos(eyaw); syaw = math.sin(eyaw)
        meas = np.array([ex + cyaw*bx - syaw*by,
                         ey_ego + syaw*bx + cyaw*by])

    gate_dist = 2.0
    if track is not None and meas is not None:
        pred_x = track[0][0] + track[0][2] * dt
        pred_y = track[0][1] + track[0][3] * dt
        if math.hypot(meas[0] - pred_x, meas[1] - pred_y) > gate_dist:
            meas = None

    if track is None:
        if meas is None:
            return None
        init_w = best_obs[2] if best_obs else 0.30
        init_h = best_obs[3] if best_obs else 0.20
        state = np.array([meas[0], meas[1], float(init_vel[0]), float(init_vel[1])])
        return (state, np.eye(4), 0, init_w, init_h, 0)

    state, P, misses = track[0], track[1], track[2]
    prev_w       = track[3] if len(track) > 3 else 0.30
    prev_h       = track[4] if len(track) > 4 else 0.20
    static_count = track[5] if len(track) > 5 else 0

    F = np.array([[1.0, 0.0, dt,  0.0],
                  [0.0, 1.0, 0.0, dt ],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Q = np.diag([dt, dt, 3.0*dt, 3.0*dt])
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    R = sigma_pos * np.eye(2)

    state = F @ state
    P = F @ P @ F.T + Q

    if meas is not None:
        innov_vec = meas - H @ state
        innov_mag = math.hypot(innov_vec[0], innov_vec[1])
        static_count = static_count + 1 if innov_mag < 0.04 else 0
        if static_count > 5:
            return None

    if meas is not None:
        innov = innov_vec
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ innov
        P = (np.eye(4) - K @ H) @ P

    if meas is not None and best_obs is not None:
        best_w = max(prev_w, best_obs[2])
        best_h = max(prev_h, best_obs[3])
    else:
        best_w, best_h = prev_w, prev_h

    misses = 0 if meas is not None else misses + 1
    if misses > max_misses:
        return None
    return (state, P, misses, best_w, best_h, static_count)


# ── MPC ───────────────────────────────────────────────────────────────────────

class MPCKinematic:
    """
    Kinematic bicycle MPC in Frenet frame (VehicleKinematics model).

    Decision vars : x=[s,ey,epsi,v] x (N+1),  u=[a,delta] x N,  slack x N
    Parameters    : x0(4), curv(N), vref(N), obs[s,ey,a,b](4), hw(1)
    Constraints   : IC, dynamics (RK4), obstacle (soft ellipse), soft wall
    """
    CAR_W = 0.22   # [m] vehicle width
    CAR_L = 0.33   # [m] vehicle length

    def __init__(self, N=20, dt=0.025, L=0.33,
                 Q_ey=15.0, Q_epsi=1.0, Q_v=1.0,
                 R_a=0.0, R_d=0.0, Q_ts=50.0,
                 Q_slack=800.0, L_slack=200.0,
                 v_min=0.0, v_max=6.0,
                 a_min=-1.0, a_max=1.0, delta_max=0.25):
        self.N = N; self.dt = dt; self.L = L
        self.Q_ey = Q_ey; self.Q_epsi = Q_epsi; self.Q_v = Q_v
        self.R_a = R_a; self.R_d = R_d; self.Q_ts = Q_ts
        self.Q_slack = Q_slack; self.L_slack = L_slack
        self.v_min = v_min; self.v_max = v_max
        self.a_min = a_min; self.a_max = a_max
        self.delta_max = delta_max
        # p: x0(4) + curv(N) + vref(N) + obs(4) + hw(1)
        self.n_p = 4 + N + N + 4 + 1
        self._x_ws = None; self._u_ws = None; self._sl_ws = None
        self._lam_x = None; self._lam_g = None
        self._build()

    # ── Dynamics (VehicleKinematics.update_dynamics_time) ─────────────────────

    def _kin(self, x, u, curv):
        ey, epsi, v, a, delta = x[1], x[2], x[3], u[0], u[1]
        denom = ca.fmax(1.0 - ey * curv, 0.2)
        return ca.vertcat(v * ca.cos(epsi) / denom,
                          v * ca.sin(epsi),
                          v * ca.tan(delta) / self.L,
                          a - v * curv)

    def _rk4(self, x, u, curv):
        h = self.dt
        k1 = self._kin(x, u, curv)
        k2 = self._kin(x + h/2*k1, u, curv)
        k3 = self._kin(x + h/2*k2, u, curv)
        k4 = self._kin(x + h*k3, u, curv)
        return x + h*(k1 + 2*k2 + 2*k3 + k4)/6

    # ── Build NLP once ────────────────────────────────────────────────────────

    def _build(self):
        N = self.N
        EPS = 0.01

        X  = ca.SX.sym('X', 4, N+1)
        U  = ca.SX.sym('U', 2, N)
        Sl = ca.SX.sym('Sl', N)        # obstacle slack

        P      = ca.SX.sym('P', self.n_p)
        x0_p   = P[:4]
        curv_p = P[4:4+N]
        vref_p = P[4+N:4+2*N]
        s_obs  = P[4+2*N];   ey_obs = P[4+2*N+1]
        a_obs  = P[4+2*N+2]; b_obs  = P[4+2*N+3]
        hw_p   = P[4+2*N+4]

        J = ca.SX(0)
        g_parts = []
        lbg0 = []; ubg0 = []
        self._obs_rows = []
        row = 0

        # Initial condition
        g_parts.append(X[:, 0] - x0_p)
        lbg0 += [0.]*4; ubg0 += [0.]*4; row += 4

        for k in range(N):
            xk = X[:, k]; uk = U[:, k]; sl = Sl[k]

            J += (self.Q_ey    * xk[1]**2
                  + self.Q_epsi  * xk[2]**2
                  + self.Q_v     * (xk[3] - vref_p[k])**2
                  + self.R_a     * uk[0]**2
                  + self.R_d     * uk[1]**2
                  + self.Q_slack * sl**2 + self.L_slack * sl
                  + self.Q_ts    * ca.fmax(ca.fabs(xk[1]) - (hw_p - self.CAR_W), 0)**2)

            # Dynamics equality
            g_parts.append(X[:, k+1] - self._rk4(xk, uk, curv_p[k]))
            lbg0 += [0.]*4; ubg0 += [0.]*4; row += 4

            # Obstacle: soft ellipse (sl >= 0 always, lbg set to 1 when active)
            od = ((xk[0]-s_obs)**2/(a_obs**2+EPS)
                  + (xk[1]-ey_obs)**2/(b_obs**2+EPS))
            g_parts.append(ca.vertcat(od + sl))
            lbg0 += [0.0]; ubg0 += [np.inf]
            self._obs_rows.append(row); row += 1

        # Terminal cost (5× stage weights)
        xN = X[:, N]
        J += (5.*self.Q_ey   * xN[1]**2
              + 5.*self.Q_epsi * xN[2]**2
              + 2.*self.Q_v    * (xN[3] - vref_p[-1])**2)

        w = ca.vertcat(ca.reshape(X, -1, 1),
                       ca.reshape(U, -1, 1),
                       ca.reshape(Sl, -1, 1))
        g = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in g_parts])

        lbx, ubx = [], []
        for _ in range(N+1):
            lbx += [-np.inf, -1.5, -np.pi/2, self.v_min]
            ubx += [ np.inf,  1.5,  np.pi/2, self.v_max]
        for _ in range(N):
            lbx += [self.a_min, -self.delta_max]
            ubx += [self.a_max,  self.delta_max]
        lbx += [0.] * N; ubx += [np.inf] * N

        nlp  = {'x': w, 'f': J, 'g': g, 'p': P}
        opts = {'ipopt.print_level': 0, 'print_time': 0,
                'ipopt.max_iter': 150, 'ipopt.tol': 1e-4,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-5,
                'ipopt.warm_start_mult_bound_push': 1e-5}
        self._solver = ca.nlpsol('mpc_kin', 'ipopt', nlp, opts)
        self._lbg0 = lbg0; self._ubg0 = ubg0
        self._lbx = lbx;   self._ubx = ubx

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self, x0, curv_prof, vref_prof,
              obs_s, obs_ey, obs_a, obs_b, hw, obs_active):
        N = self.N
        p_val = np.concatenate([x0, curv_prof, vref_prof,
                                 [obs_s, obs_ey, obs_a, obs_b, hw]])

        lbg = list(self._lbg0)
        for r in self._obs_rows:
            lbg[r] = 1.0 if obs_active else 0.0

        if self._x_ws is None:
            v0 = float(x0[3])
            xw = np.zeros((4, N+1))
            for k in range(N+1):
                xw[:, k] = x0.copy()
                xw[0, k] += k * self.dt * v0
            uw = np.zeros((2, N)); sw = np.zeros(N)
        else:
            xw, uw, sw = self._x_ws, self._u_ws, self._sl_ws

        w0 = np.concatenate([xw.ravel('F'), uw.ravel('F'), sw])

        try:
            t0 = time.time()
            kwargs = dict(x0=w0, lbx=self._lbx, ubx=self._ubx,
                          lbg=lbg, ubg=self._ubg0, p=p_val)
            if self._lam_x is not None:
                kwargs['lam_x0'] = self._lam_x
                kwargs['lam_g0'] = self._lam_g
            sol = self._solver(**kwargs)
            dt_solve = time.time() - t0

            wsol   = np.array(sol['x']).ravel()
            nx_tot = 4*(N+1); nu_tot = 2*N
            x_pred = wsol[:nx_tot].reshape(4, N+1, order='F')
            u_pred = wsol[nx_tot:nx_tot+nu_tot].reshape(2, N, order='F')

            self._x_ws  = np.hstack([x_pred[:, 1:], x_pred[:, -1:]])
            self._u_ws  = np.hstack([u_pred[:, 1:], u_pred[:, -1:]])
            self._sl_ws = np.zeros(N)
            self._lam_x = sol['lam_x']
            self._lam_g = sol['lam_g']
            return x_pred, u_pred, True, dt_solve

        except Exception:
            self._x_ws = self._u_ws = self._sl_ws = None
            self._lam_x = self._lam_g = None
            return None, None, False, 0.0


# ── ROS 2 Node ────────────────────────────────────────────────────────────────

class MPCLocalPlannerNode(Node):

    OBS_MARGIN       = 0.0   # [m]   safety margin added to obstacle dimensions
    OBS_TRIGGER      = 8.0   # [m]   forward distance to activate obstacle mode
    OBS_LOOKAHEAD    = 3.0   # [m]   lookahead cap sent to PP when obstacle active
    NORMAL_LOOKAHEAD = 3.0   # [m]   lookahead cap sent to PP normally (= pp_t_clip_max)
    OBS_SPEED        = 1.5   # [m/s] speed cap when obstacle is in the driving path

    def __init__(self):
        super().__init__('mpc_local_planner')
        gp = lambda n, v: self.declare_parameter(n, v).value

        self.scan_topic   = str(gp('scan_topic',   '/scan'))
        self.odom_topic   = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic = str(gp('global_topic', '/global_waypoints'))
        self.local_topic  = str(gp('local_topic',  '/local_waypoints'))

        self.track_half_w = float(gp('track_half_w', 0.3))

        N  = int(gp('mpc_N',   20))
        dt = float(gp('mpc_dt', 0.025))
        L  = float(gp('mpc_L',  0.33))
        self.mpc = MPCKinematic(
            N=N, dt=dt, L=L,
            Q_ey      = float(gp('Q_ey',     15.0)),
            Q_epsi    = float(gp('Q_epsi',    1.0)),
            Q_v       = float(gp('Q_v',       1.0)),
            R_a       = float(gp('R_a',       0.0)),
            R_d       = float(gp('R_d',       0.0)),
            Q_ts      = float(gp('Q_ts',     50.0)),
            Q_slack   = float(gp('Q_slack', 800.0)),
            L_slack   = float(gp('L_slack', 200.0)),
            v_max     = float(gp('v_max',     6.0)),
            a_min     = float(gp('a_min',    -1.0)),
            a_max     = float(gp('a_max',     1.0)),
            delta_max = float(gp('delta_max', 0.25)),
        )

        self._sx = self._sy = None
        self._vx_pchip = None
        self._s_samples = self._x_samples = self._y_samples = None
        self.s_total = 0.0

        self.ex = self.ey_world = self.eyaw = self.ev = 0.0
        self.have_pose = False
        self.ego_s = self.ego_d = 0.0

        self._track = None
        self._last_scan_t = None

        self._solve_times: list[float] = []
        self._solve_calls = 0
        self._LOG_EVERY = 50

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray, self.global_topic, self._cb_global, latched)
        self.create_subscription(Odometry,  self.odom_topic,   self._cb_odom,   10)
        self.create_subscription(LaserScan, self.scan_topic,   self._cb_scan,   10)

        self._pub_local       = self.create_publisher(WpntArray,   self.local_topic, latched)
        self._pub_marker      = self.create_publisher(MarkerArray, '/mpc_planner/markers', 5)
        self._pub_la_cap      = self.create_publisher(Float32, '/mpc_planner/lookahead_cap', 10)

        self.get_logger().info(
            f'mpc_local_planner up | horizon={N}×{dt}s={N*dt:.2f}s | '
            f'{self.global_topic} → {self.local_topic}')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _cb_global(self, msg):
        self._build_frenet_spline(list(msg.wpnts))

    def _cb_odom(self, msg):
        p = msg.pose.pose.position; q = msg.pose.pose.orientation
        self.ex, self.ey_world = p.x, p.y
        self.eyaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self.ev   = msg.twist.twist.linear.x
        self.have_pose = True

    def _cb_scan(self, msg):
        if not self.have_pose or self._sx is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = 0.05 if self._last_scan_t is None else max(now - self._last_scan_t, 1e-3)
        self._last_scan_t = now

        ranges = np.asarray(msg.ranges, dtype=float)
        x_pts, y_pts = _scan_to_xy(ranges, msg.angle_min, msg.angle_increment)
        clusters  = _cluster(x_pts, y_pts, msg.angle_increment)
        obstacles = _l_shape(clusters)
        ego = (self.ex, self.ey_world, self.eyaw)
        self._track = _tracking(obstacles, self._track, dt, ego,
                                 init_vel=(self.ev, 0.0))

        self.ego_s, self.ego_d = self.to_frenet(self.ex, self.ey_world)

        out, obs_active, obs_info = self._run_mpc()
        self._pub_local.publish(out)
        self._publish_marker(out, obs_info)
        la = Float32()
        la.data = self.OBS_LOOKAHEAD if obs_active else self.NORMAL_LOOKAHEAD
        self._pub_la_cap.publish(la)

    # ── MPC step ──────────────────────────────────────────────────────────────

    def _run_mpc(self):
        N, dt = self.mpc.N, self.mpc.dt

        s0    = self.ego_s
        ey0   = self.ego_d
        psi_path, _ = self._psi_kappa_at(s0)
        epsi0 = _wrap(self.eyaw - psi_path)
        v0    = max(abs(self.ev), 0.01)
        x0    = np.array([s0, ey0, epsi0, v0])

        curv_prof = np.array([self._kappa_at(s0 + (k+0.5)*dt*v0) for k in range(N)])
        vref_prof = np.array([self._vx_at(s0 + k*dt*v0) for k in range(N)])
        vref_prof = np.clip(vref_prof, 0.5, self.mpc.v_max)

        hw = self.track_half_w

        obs_active = False
        obs_in_path = False
        obs_wx = obs_wy = 0.0   # world-frame obstacle centre (for marker)
        obs_s = obs_ey = 0.0
        obs_a = obs_b = 0.3
        if self._track is not None:
            ox, oy = self._track[0][0], self._track[0][1]
            os, od = self.to_frenet(ox, oy)
            gap   = (os - s0) % self.s_total
            obs_w = self._track[3] if len(self._track) > 3 else 0.30
            obs_h = self._track[4] if len(self._track) > 4 else 0.20
            if 0.0 < gap < self.OBS_TRIGGER:
                obs_active = True
                obs_wx, obs_wy = ox, oy
                obs_s  = os
                obs_ey = od
                obs_a  = obs_h / 2 + MPCKinematic.CAR_L / 2 + self.OBS_MARGIN
                obs_b  = obs_w / 2 + MPCKinematic.CAR_W / 2 + self.OBS_MARGIN

                # Predict ego 1 s ahead at current speed (constant v, no steering)
                # and check whether the path enters the obstacle ellipse
                n_check = min(int(1.0 / dt) + 1, N)
                for k in range(n_check):
                    pred_s = s0 + k * dt * v0
                    ell = ((pred_s - obs_s)**2 / (obs_a**2 + 1e-9)
                         + (ey0    - obs_ey)**2 / (obs_b**2 + 1e-9))
                    if ell < 1.0:
                        obs_in_path = True
                        break

        if obs_in_path:
            vref_prof = np.minimum(vref_prof, self.OBS_SPEED)

        x_pred, u_pred, success, t_solve = self.mpc.solve(
            x0, curv_prof, vref_prof,
            obs_s, obs_ey, obs_a, obs_b, hw, obs_active)

        if success:
            self._solve_times.append(t_solve)
            if len(self._solve_times) > self._LOG_EVERY:
                self._solve_times.pop(0)
        self._solve_calls += 1
        if self._solve_calls % self._LOG_EVERY == 0 and self._solve_times:
            avg = 1e3 * sum(self._solve_times) / len(self._solve_times)
            mx  = 1e3 * max(self._solve_times)
            self.get_logger().info(f'[MPC] avg={avg:.1f}ms  max={mx:.1f}ms')

        v_cap = self.OBS_SPEED if obs_in_path else None
        obs_info = (obs_wx, obs_wy, obs_a*2, obs_b*2) if obs_active else None
        if success and x_pred is not None:
            return self._to_wpnt_array(x_pred, v_cap=v_cap), obs_active, obs_info
        self.get_logger().warn('MPC solve failed, passthrough')
        return self._passthrough(v_cap=v_cap), obs_active, obs_info

    # ── Waypoint builders ─────────────────────────────────────────────────────

    def _to_wpnt_array(self, x_pred, v_cap=None):
        out = self._empty_header()
        for k in range(x_pred.shape[1]):
            s  = float(x_pred[0, k]) % self.s_total
            ey = float(x_pred[1, k])
            xc, yc = self.to_cartesian(s, ey)
            psi, kappa = self._psi_kappa_at(s)
            vx = float(self._vx_at(s))
            if v_cap is not None:
                vx = min(vx, v_cap)
            w = Wpnt()
            w.id = k; w.s_m = s; w.d_m = ey
            w.x_m = float(xc); w.y_m = float(yc)
            w.psi_rad = float(psi); w.kappa_radpm = float(kappa)
            w.vx_mps = vx; w.ax_mps2 = 0.0
            w.d_right = 0.0; w.d_left = 0.0
            out.wpnts.append(w)
        return out

    def _passthrough(self, v_cap=None):
        N = self.mpc.N; dt = self.mpc.dt
        v0 = max(abs(self.ev), 0.5)
        out = self._empty_header()
        for k in range(N + 1):
            s = (self.ego_s + k * dt * v0) % self.s_total
            xc, yc = self.to_cartesian(s, 0.0)
            psi, kappa = self._psi_kappa_at(s)
            vx = float(self._vx_at(s))
            if v_cap is not None:
                vx = min(vx, v_cap)
            w = Wpnt()
            w.id = k; w.s_m = float(s); w.d_m = 0.0
            w.x_m = float(xc); w.y_m = float(yc)
            w.psi_rad = float(psi); w.kappa_radpm = float(kappa)
            w.vx_mps = vx; w.ax_mps2 = 0.0
            w.d_right = 0.0; w.d_left = 0.0
            out.wpnts.append(w)
        return out

    def _empty_header(self):
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        return out

    # ── Frenet spline ─────────────────────────────────────────────────────────

    def _build_frenet_spline(self, wpnts):
        if len(wpnts) < 4:
            return
        x  = np.array([w.x_m    for w in wpnts], float)
        y  = np.array([w.y_m    for w in wpnts], float)
        vx = np.array([w.vx_mps for w in wpnts], float)
        if abs(x[0]-x[-1]) > 1e-6 or abs(y[0]-y[-1]) > 1e-6:
            x  = np.append(x,  x[0])
            y  = np.append(y,  y[0])
            vx = np.append(vx, vx[0])
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        keep = np.concatenate(([True], np.diff(s) > 1e-9))
        s = s[keep]; x = x[keep]; y = y[keep]; vx = vx[keep]
        self._sx       = CubicSpline(s, x, bc_type='periodic')
        self._sy       = CubicSpline(s, y, bc_type='periodic')
        self._vx_pchip = PchipInterpolator(s, vx, extrapolate=False)
        self._s_samples = s[:-1]
        self._x_samples = x[:-1]; self._y_samples = y[:-1]
        self.s_total = float(s[-1])
        self.get_logger().info(f'Frenet spline ready (s_total={self.s_total:.2f} m)')

    def _psi_kappa_at(self, s):
        s = s % self.s_total
        dx  = float(self._sx(s, 1)); dy  = float(self._sy(s, 1))
        ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
        psi   = math.atan2(dy, dx)
        denom = (dx*dx + dy*dy)**1.5
        kappa = (dx*ddy - dy*ddx) / denom if denom > 1e-12 else 0.0
        return psi, kappa

    def _kappa_at(self, s):
        return self._psi_kappa_at(s)[1]

    def _vx_at(self, s):
        v = float(self._vx_pchip(s % self.s_total))
        return v if np.isfinite(v) else 1.5

    def to_frenet(self, x, y, n_newton=5):
        if self._sx is None:
            return 0.0, 0.0
        i = int(np.argmin((self._x_samples-x)**2 + (self._y_samples-y)**2))
        s = float(self._s_samples[i])
        for _ in range(n_newton):
            rx = float(self._sx(s)) - x; ry = float(self._sy(s)) - y
            dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
            ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
            g  = rx*dx + ry*dy
            gp = dx*dx + dy*dy + rx*ddx + ry*ddy
            if abs(gp) < 1e-12:
                break
            s = (s - g/gp) % self.s_total
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return s, 0.0
        nx, ny = -dy/nrm, dx/nrm
        d = (x - float(self._sx(s)))*nx + (y - float(self._sy(s)))*ny
        return s, d

    def to_cartesian(self, s, d):
        if self._sx is None:
            return 0.0, 0.0
        s = s % self.s_total
        x0 = float(self._sx(s)); y0 = float(self._sy(s))
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return x0, y0
        nx, ny = -dy/nrm, dx/nrm
        return x0 + d*nx, y0 + d*ny

    # ── Visualization ─────────────────────────────────────────────────────────

    def _publish_marker(self, wpnts, obs_info=None):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # ── MPC predicted path ────────────────────────────────────────────────
        path_m = Marker()
        path_m.header.frame_id = 'map'; path_m.header.stamp = now
        path_m.ns = 'mpc_path'; path_m.id = 0
        path_m.type = Marker.LINE_STRIP; path_m.action = Marker.ADD
        path_m.pose.orientation.w = 1.0
        path_m.scale.x = 0.06
        path_m.color.r, path_m.color.g, path_m.color.b, path_m.color.a = 0.2, 1.0, 0.4, 1.0
        for w in wpnts.wpnts:
            p = Point(); p.x = float(w.x_m); p.y = float(w.y_m); p.z = 0.15
            path_m.points.append(p)
        ma.markers.append(path_m)

        # ── Obstacle cylinder ─────────────────────────────────────────────────
        cyl = Marker()
        cyl.header.frame_id = 'map'; cyl.header.stamp = now
        cyl.ns = 'mpc_obstacle'; cyl.id = 0
        cyl.pose.orientation.w = 1.0
        if obs_info is not None:
            ox, oy, size_s, size_d = obs_info
            cyl.type = Marker.CYLINDER; cyl.action = Marker.ADD
            cyl.pose.position.x = float(ox)
            cyl.pose.position.y = float(oy)
            cyl.pose.position.z = 0.2
            cyl.scale.x = float(size_d)   # along ey  (track-normal)
            cyl.scale.y = float(size_s)   # along s   (track-tangent)
            cyl.scale.z = 0.4
            cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 1.0, 0.2, 0.2, 0.8
        else:
            cyl.type = Marker.CYLINDER; cyl.action = Marker.DELETE
        ma.markers.append(cyl)

        # ── Obstacle label ────────────────────────────────────────────────────
        txt = Marker()
        txt.header.frame_id = 'map'; txt.header.stamp = now
        txt.ns = 'mpc_obstacle'; txt.id = 1
        txt.pose.orientation.w = 1.0
        if obs_info is not None:
            ox, oy, _, _ = obs_info
            txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
            txt.pose.position.x = float(ox)
            txt.pose.position.y = float(oy)
            txt.pose.position.z = 0.55
            txt.scale.z = 0.18
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 1.0
            txt.text = 'OBS'
        else:
            txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.DELETE
        ma.markers.append(txt)

        self._pub_marker.publish(ma)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _wrap(angle):
    while angle >  math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MPCLocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
