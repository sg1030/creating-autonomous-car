#!/usr/bin/env python3
"""
mpc_local_planner.py  —  MPC kinematic local planner with obstacle avoidance.

State:    x = [s, ey, epsi, v]
Control:  u = [a, delta]
Dynamics: kinematic bicycle (Frenet frame, time-domain, RK4)
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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))


def _scan_to_xy(ranges, angle_min, angle_inc):
    r_min, r_max = 0.05, 10.0
    n = ranges.shape[0]
    ang = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    return (np.where(valid, ranges * np.cos(ang), np.nan),
            np.where(valid, ranges * np.sin(ang), np.nan))


# ── Perception (adapted from local_planning.py) ───────────────────────────────

def _cluster(x, y, angle_inc):
    lam = math.radians(5.0); sigma = 0.06; min_pts = 5; max_nan = 10
    denom = math.sin(lam - angle_inc)
    if abs(denom) < 1e-6:
        denom = 1e-6
    clusters, current, prev, nan_cnt = [], [], None, 0
    for xi, yi in zip(x, y):
        if not (math.isfinite(xi) and math.isfinite(yi)):
            nan_cnt += 1
            if nan_cnt > max_nan:
                if len(current) >= min_pts:
                    clusters.append(np.array(current))
                current, prev, nan_cnt = [], None, 0
        else:
            nan_cnt = 0
            if prev is None:
                current.append([xi, yi])
            else:
                d_max = math.hypot(prev[0], prev[1]) * math.sin(angle_inc) / denom + sigma
                if math.hypot(xi - prev[0], yi - prev[1]) > d_max:
                    if len(current) >= min_pts:
                        clusters.append(np.array(current))
                    current = [[xi, yi]]
                else:
                    current.append([xi, yi])
            prev = (xi, yi)
    if len(current) >= min_pts:
        clusters.append(np.array(current))
    return clusters


def _l_shape(clusters):
    max_sz, min_sz, min_e = 0.60, 0.15, 0.01
    thetas = np.linspace(0.0, np.pi/2 - np.pi/180, 180)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    obs = []
    for pts in clusters:
        pts = np.array(pts)
        if len(pts) < 2:
            continue
        a = pts[:, 0:1]*cos_t + pts[:, 1:2]*sin_t
        b = -pts[:, 0:1]*sin_t + pts[:, 1:2]*cos_t
        d = np.minimum(a - a.min(0), a.max(0) - a)
        d = np.minimum(d, np.minimum(b - b.min(0), b.max(0) - b))
        k = int(np.argmax(np.sum(1.0 / np.maximum(d, min_e), axis=0)))
        c, s = cos_t[k], sin_t[k]
        ak, bk = a[:, k], b[:, k]
        w, h = float(ak.max()-ak.min()), float(bk.max()-bk.min())
        ca_ = 0.5*(ak.max()+ak.min()); cb_ = 0.5*(bk.max()+bk.min())
        cx, cy = ca_*c - cb_*s, ca_*s + cb_*c
        if max(w, h) > max_sz or min(w, h) < 0.03:
            continue
        if max(w, h) > 0 and min(w, h) < 0.08*max(w, h):
            continue
        obs.append((cx, cy, max(w, min_sz), max(h, min_sz), thetas[k]))
    return obs


def _tracking(obs_list, track, dt, ego, init_vel=(0.0, 0.0)):
    opp_max_lat, max_miss, sig_pos = 0.7, 20, 0.1
    lidar_x = 0.27
    ex, ey_ego, eyaw = ego
    meas, best = None, None
    cands = [(cx, cy, w, h, th) for cx, cy, w, h, th in obs_list
             if cx > 0 and abs(cy) <= opp_max_lat]
    if cands:
        best = min(cands, key=lambda o: math.hypot(o[0], o[1]))
        bx, by = best[0] + lidar_x, best[1]
        c, s = math.cos(eyaw), math.sin(eyaw)
        meas = np.array([ex + c*bx - s*by, ey_ego + s*bx + c*by])
    gate = 2.0
    if track is not None and meas is not None:
        px = track[0][0] + track[0][2]*dt; py = track[0][1] + track[0][3]*dt
        if math.hypot(meas[0]-px, meas[1]-py) > gate:
            meas = None
    if track is None:
        if meas is None:
            return None
        state = np.array([meas[0], meas[1], float(init_vel[0]), float(init_vel[1])])
        return (state, np.eye(4), 0,
                best[2] if best else 0.30, best[3] if best else 0.20, 0)
    state, P, misses = track[0], track[1], track[2]
    pw = track[3] if len(track) > 3 else 0.30
    ph = track[4] if len(track) > 4 else 0.20
    sc = track[5] if len(track) > 5 else 0
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    Q = np.diag([dt, dt, 3*dt, 3*dt])
    H = np.array([[1,0,0,0],[0,1,0,0]], float)
    state = F @ state; P = F @ P @ F.T + Q
    if meas is not None:
        innov = meas - H @ state
        sc = sc + 1 if math.hypot(*innov) < 0.04 else 0
        if sc > 15:
            return None
        S = H @ P @ H.T + sig_pos*np.eye(2)
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ innov; P = (np.eye(4) - K@H) @ P
    bw = max(pw, best[2]) if (meas is not None and best) else pw
    bh = max(ph, best[3]) if (meas is not None and best) else ph
    misses = 0 if meas is not None else misses + 1
    if misses > max_miss:
        return None
    return (state, P, misses, bw, bh, sc)


# ── MPC ───────────────────────────────────────────────────────────────────────

class MPCKinematic:
    """
    Kinematic bicycle MPC in Frenet frame.

    Decision vars : x=[s,ey,epsi,v] x (N+1),  u=[a,delta] x N,  slack x N
    Parameters    : x0(4), curv(N), vref(N), obs[s,ey,a,b](4), hw(1), u_prev(2)
    Constraints   : IC, dynamics (RK4), obstacle (soft ellipse), input rates
    """
    CAR_W = 0.22   # [m] vehicle width
    CAR_L = 0.33   # [m] vehicle length

    def __init__(self, N=20, dt=0.1, L=0.33,
                 Q_ey=15.0, Q_epsi=5.0, Q_v=2.0,
                 R_a=0.05, R_d=0.3, Q_ts=50.0,
                 Q_slack=800.0, L_slack=200.0,
                 v_min=0.0, v_max=6.0,
                 a_min=-2.0, a_max=2.0, delta_max=0.4):
        self.N = N; self.dt = dt; self.L = L
        self.Q_ey = Q_ey; self.Q_epsi = Q_epsi; self.Q_v = Q_v
        self.R_a = R_a; self.R_d = R_d; self.Q_ts = Q_ts
        self.Q_slack = Q_slack; self.L_slack = L_slack
        self.v_min = v_min; self.v_max = v_max
        self.a_min = a_min; self.a_max = a_max
        self.delta_max = delta_max
        # p: x0(4)+curv(N)+vref(N)+obs(4)+hw(1)
        self.n_p = 4 + N + N + 4 + 1
        self._x_ws = None; self._u_ws = None; self._sl_ws = None
        self._lam_x = None; self._lam_g = None
        self._build()

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def _kin(self, x, u, curv):
        ey, epsi, v, a, delta = x[1], x[2], x[3], u[0], u[1]
        denom = ca.fmax(1.0 - ey * curv, 0.05)
        return ca.vertcat(v * ca.cos(epsi) / denom,
                          v * ca.sin(epsi),
                          v * ca.tan(delta) / self.L,
                          a)

    def _rk4(self, x, u, curv):
        h = self.dt
        k1 = self._kin(x, u, curv)
        k2 = self._kin(x + h/2*k1, u, curv)
        k3 = self._kin(x + h/2*k2, u, curv)
        k4 = self._kin(x + h*k3, u, curv)
        return x + h*(k1 + 2*k2 + 2*k3 + k4)/6

    # ── Build NLP once ────────────────────────────────────────────────────────

    def _build(self):
        N, dt = self.N, self.dt
        EPS = 0.01

        X  = ca.SX.sym('X', 4, N+1)   # states
        U  = ca.SX.sym('U', 2, N)      # controls
        Sl = ca.SX.sym('Sl', N)        # obstacle slack

        P = ca.SX.sym('P', self.n_p)
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

        # Initial condition (4 equality constraints)
        g_parts.append(X[:, 0] - x0_p)
        lbg0 += [0.]*4; ubg0 += [0.]*4; row += 4

        for k in range(N):
            xk = X[:, k]; uk = U[:, k]; sl = Sl[k]

            # Stage cost
            J += (self.Q_ey   * xk[1]**2
                  + self.Q_epsi * xk[2]**2
                  + self.Q_v    * (xk[3] - vref_p[k])**2
                  + self.R_a    * uk[0]**2
                  + self.R_d    * uk[1]**2
                  + self.Q_slack * sl**2 + self.L_slack * sl
                  + self.Q_ts * ca.fmax(ca.fabs(xk[1]) - (hw_p - self.CAR_W), 0)**2)

            # Dynamics (4 equality constraints)
            g_parts.append(X[:, k+1] - self._rk4(xk, uk, curv_p[k]))
            lbg0 += [0.]*4; ubg0 += [0.]*4; row += 4

            # Obstacle: (s-s_obs)²/a² + (ey-ey_obs)²/b² + slack ≥ 1 (when active)
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

        # Flatten decision vars and constraints
        w = ca.vertcat(ca.reshape(X, -1, 1),
                       ca.reshape(U, -1, 1),
                       ca.reshape(Sl, -1, 1))
        g = ca.vertcat(*[ca.reshape(gi, -1, 1) for gi in g_parts])

        # Variable bounds
        lbx, ubx = [], []
        for _ in range(N+1):
            lbx += [-np.inf, -1.5, -np.pi/2, self.v_min]
            ubx += [ np.inf,  1.5,  np.pi/2, self.v_max]
        for _ in range(N):
            lbx += [self.a_min, -self.delta_max]
            ubx += [self.a_max,  self.delta_max]
        lbx += [0.] * N; ubx += [np.inf] * N

        nlp = {'x': w, 'f': J, 'g': g, 'p': P}
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

        # Build warm-start vector
        if self._x_ws is None:
            v0 = float(x0[3])
            xw = np.zeros((4, N+1))
            for k in range(N+1):
                xw[:, k] = x0.copy()
                xw[0, k] += k * self.dt * v0
            uw = np.zeros((2, N)); sw = np.zeros(N)
        else:
            xw, uw, sw = self._x_ws, self._u_ws, self._sl_ws

        # CasADi reshape is column-major (Fortran order)
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

            wsol = np.array(sol['x']).ravel()
            nx_tot = 4*(N+1); nu_tot = 2*N
            x_pred = wsol[:nx_tot].reshape(4, N+1, order='F')
            u_pred = wsol[nx_tot:nx_tot+nu_tot].reshape(2, N, order='F')
            sl_pred = wsol[nx_tot+nu_tot:]

            # Shift warm start
            self._x_ws  = np.hstack([x_pred[:, 1:], x_pred[:, -1:]])
            self._u_ws  = np.hstack([u_pred[:, 1:], u_pred[:, -1:]])
            self._sl_ws = np.zeros(N)
            self._lam_x = sol['lam_x']
            self._lam_g = sol['lam_g']
            return x_pred, u_pred, True, dt_solve

        except Exception as e:
            self._x_ws = self._u_ws = self._sl_ws = None
            self._lam_x = self._lam_g = None
            return None, None, False, 0.0


# ── ROS 2 Node ────────────────────────────────────────────────────────────────

class MPCLocalPlannerNode(Node):

    OBS_MARGIN = 0.15   # [m] safety margin added to obstacle dimensions
    OBS_TRIGGER = 8.0   # [m] forward distance to activate MPC obstacle mode

    def __init__(self):
        super().__init__('mpc_local_planner')
        gp = lambda n, v: self.declare_parameter(n, v).value

        # Topics
        self.scan_topic   = str(gp('scan_topic',   '/scan'))
        self.odom_topic   = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic = str(gp('global_topic', '/global_waypoints'))
        self.local_topic  = str(gp('local_topic',  '/local_waypoints'))

        # Planning params
        self.track_half_w  = float(gp('track_half_w', 0.8))    # [m] fallback

        # MPC params
        N   = int(gp('mpc_N',   20))
        dt  = float(gp('mpc_dt', 0.1))
        L   = float(gp('mpc_L',  0.33))
        self.mpc = MPCKinematic(
            N=N, dt=dt, L=L,
            Q_ey    = float(gp('Q_ey',    15.0)),
            Q_epsi  = float(gp('Q_epsi',   5.0)),
            Q_v     = float(gp('Q_v',      2.0)),
            R_a     = float(gp('R_a',      0.05)),
            R_d     = float(gp('R_d',      0.3)),
            Q_ts    = float(gp('Q_ts',    50.0)),
            Q_slack = float(gp('Q_slack', 800.0)),
            L_slack = float(gp('L_slack', 200.0)),
            v_max   = float(gp('v_max',    6.0)),
            a_min   = float(gp('a_min',   -2.0)),
            a_max   = float(gp('a_max',    2.0)),
            delta_max  = float(gp('delta_max',   0.4)),
        )

        # Frenet spline state
        self._sx = self._sy = None
        self._vx_pchip = self._dl_pchip = self._dr_pchip = None
        self._s_samples = self._x_samples = self._y_samples = None
        self.s_total = 0.0

        # Ego state
        self.ex = self.ey_world = self.eyaw = self.ev = 0.0
        self.have_pose = False
        self.ego_s = self.ego_d = 0.0

        # Perception
        self._track = None
        self._last_scan_t = None

        # Solve-time profiling (rolling window of 50 calls)
        self._solve_times: list[float] = []
        self._solve_calls = 0
        self._LOG_EVERY = 50   # print stats every N solve calls


        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray, self.global_topic, self._cb_global, latched)
        self.create_subscription(Odometry,  self.odom_topic,   self._cb_odom,   10)
        self.create_subscription(LaserScan, self.scan_topic,   self._cb_scan,   10)

        self._pub_local  = self.create_publisher(WpntArray,   self.local_topic, latched)
        self._pub_marker = self.create_publisher(MarkerArray, '/mpc_planner/markers', 5)

        self.get_logger().info(
            f'mpc_local_planner up | horizon={N}×{dt}s | '
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

        # Perception
        ranges = np.asarray(msg.ranges, dtype=float)
        x_pts, y_pts = _scan_to_xy(ranges, msg.angle_min, msg.angle_increment)
        clusters = _cluster(x_pts, y_pts, msg.angle_increment)
        obstacles = _l_shape(clusters)
        ego = (self.ex, self.ey_world, self.eyaw)
        self._track = _tracking(obstacles, self._track, dt, ego,
                                 init_vel=(self.ev, 0.0))

        self.ego_s, self.ego_d = self.to_frenet(self.ex, self.ey_world)

        out = self._run_mpc()
        self._pub_local.publish(out)
        self._publish_marker(out)

    # ── MPC step ──────────────────────────────────────────────────────────────

    def _run_mpc(self):
        N, dt = self.mpc.N, self.mpc.dt

        # Initial Frenet state
        s0  = self.ego_s
        ey0 = self.ego_d
        psi_path, _ = self._psi_kappa_at(s0)
        epsi0 = _wrap(self.eyaw - psi_path)
        v0    = max(abs(self.ev), 0.01)
        x0    = np.array([s0, ey0, epsi0, v0])

        # Curvature and reference speed along horizon
        curv_prof = np.array([self._kappa_at(s0 + (k+0.5)*dt*v0) for k in range(N)])
        vref_prof = np.array([self._vx_at(s0 + k*dt*v0) for k in range(N)])
        vref_prof = np.clip(vref_prof, 0.5, self.mpc.v_max)

        hw = self._hw_at(s0)

        # Obstacle in Frenet frame
        obs_active = False
        obs_s = obs_ey = 0.0
        obs_a = obs_b = 0.3
        if self._track is not None:
            ox, oy = self._track[0][0], self._track[0][1]
            os, od = self.to_frenet(ox, oy)
            gap = (os - s0) % self.s_total
            obs_w = self._track[3] if len(self._track) > 3 else 0.30
            obs_h = self._track[4] if len(self._track) > 4 else 0.20
            if 0.0 < gap < self.OBS_TRIGGER:
                obs_active = True
                obs_s  = os
                obs_ey = od
                obs_a  = obs_h / 2 + MPCKinematic.CAR_L / 2 + self.OBS_MARGIN
                obs_b  = obs_w / 2 + MPCKinematic.CAR_W / 2 + self.OBS_MARGIN

        x_pred, u_pred, success, t_solve = self.mpc.solve(
            x0, curv_prof, vref_prof,
            obs_s, obs_ey, obs_a, obs_b, hw, obs_active)

        # --- timing stats ---
        if success:
            self._solve_times.append(t_solve)
            if len(self._solve_times) > self._LOG_EVERY:
                self._solve_times.pop(0)
        self._solve_calls += 1
        if self._solve_calls % self._LOG_EVERY == 0:
            if self._solve_times:
                avg_ms = 1e3 * sum(self._solve_times) / len(self._solve_times)
                max_ms = 1e3 * max(self._solve_times)
                min_ms = 1e3 * min(self._solve_times)
                self.get_logger().info(
                    f'[MPC timing] avg={avg_ms:.1f}ms  max={max_ms:.1f}ms  '
                    f'min={min_ms:.1f}ms  (last {len(self._solve_times)} solves)')
        # --------------------

        if success and x_pred is not None:
            return self._to_wpnt_array(x_pred)
        else:
            self.get_logger().warn('MPC solve failed, publishing passthrough')
            return self._passthrough()

    # ── Waypoint builders ─────────────────────────────────────────────────────

    def _to_wpnt_array(self, x_pred):
        """Convert MPC predicted Frenet states (N+1) directly to WpntArray."""
        out = self._empty_header()
        for k in range(x_pred.shape[1]):
            s  = float(x_pred[0, k]) % self.s_total
            ey = float(x_pred[1, k])
            xc, yc = self.to_cartesian(s, ey)
            psi, kappa = self._psi_kappa_at(s)
            w = Wpnt()
            w.id = k; w.s_m = s; w.d_m = ey
            w.x_m = float(xc); w.y_m = float(yc)
            w.psi_rad = float(psi); w.kappa_radpm = float(kappa)
            w.vx_mps = float(self._vx_at(s)); w.ax_mps2 = 0.0
            w.d_right = 0.0; w.d_left = 0.0
            out.wpnts.append(w)
        return out

    def _passthrough(self):
        """Fallback: N+1 centerline points at dt*v0 spacing ahead of ego."""
        N = self.mpc.N; dt = self.mpc.dt
        v0 = max(abs(self.ev), 0.5)
        out = self._empty_header()
        for k in range(N + 1):
            s = (self.ego_s + k * dt * v0) % self.s_total
            xc, yc = self.to_cartesian(s, 0.0)
            psi, kappa = self._psi_kappa_at(s)
            w = Wpnt()
            w.id = k; w.s_m = float(s); w.d_m = 0.0
            w.x_m = float(xc); w.y_m = float(yc)
            w.psi_rad = float(psi); w.kappa_radpm = float(kappa)
            w.vx_mps = float(self._vx_at(s)); w.ax_mps2 = 0.0
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
        dl = np.array([w.d_left  for w in wpnts], float)
        dr = np.array([w.d_right for w in wpnts], float)
        if abs(x[0]-x[-1]) > 1e-6 or abs(y[0]-y[-1]) > 1e-6:
            x=np.append(x,x[0]); y=np.append(y,y[0])
            vx=np.append(vx,vx[0]); dl=np.append(dl,dl[0]); dr=np.append(dr,dr[0])
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
        keep = np.concatenate(([True], np.diff(s) > 1e-9))
        s=s[keep]; x=x[keep]; y=y[keep]; vx=vx[keep]; dl=dl[keep]; dr=dr[keep]
        if np.all(dl < 1e-3): dl = np.full_like(dl, self.track_half_w)
        if np.all(dr < 1e-3): dr = np.full_like(dr, self.track_half_w)
        self._sx = CubicSpline(s, x, bc_type='periodic')
        self._sy = CubicSpline(s, y, bc_type='periodic')
        self._vx_pchip = PchipInterpolator(s, vx, extrapolate=False)
        self._dl_pchip = PchipInterpolator(s, dl, extrapolate=False)
        self._dr_pchip = PchipInterpolator(s, dr, extrapolate=False)
        self._s_samples = s[:-1]
        self._x_samples = x[:-1]; self._y_samples = y[:-1]
        self.s_total = float(s[-1])
        self.get_logger().info(f'Frenet spline ready (s_total={self.s_total:.2f} m)')

    def _psi_kappa_at(self, s):
        s = s % self.s_total
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
        psi = math.atan2(dy, dx)
        denom = (dx*dx + dy*dy)**1.5
        kappa = (dx*ddy - dy*ddx) / denom if denom > 1e-12 else 0.0
        return psi, kappa

    def _kappa_at(self, s):
        return self._psi_kappa_at(s)[1]

    def _vx_at(self, s):
        v = float(self._vx_pchip(s % self.s_total))
        return v if np.isfinite(v) else 1.5

    def _hw_at(self, s):
        s = s % self.s_total
        dl = float(self._dl_pchip(s))
        dr = float(self._dr_pchip(s))
        if not np.isfinite(dl): dl = self.track_half_w
        if not np.isfinite(dr): dr = self.track_half_w
        return min(dl, dr)

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

    def _publish_marker(self, wpnts):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'mpc_path'; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.4, 1.0
        for w in wpnts.wpnts:
            p = Point(); p.x = float(w.x_m); p.y = float(w.y_m); p.z = 0.15
            m.points.append(p)
        ma.markers.append(m)
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