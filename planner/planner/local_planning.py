#!/usr/bin/env python3
"""
local_planning.py - Standalone mode-selectable local planner.

Single file: perception + Frenet + planner. PP receives this node's
/local_waypoints via launch-level topic remap (PP.py unchanged).

Pipeline:
    /scan, /vesc/odom, /global_waypoints                          INPUT
        1. perception  : cluster -> l_shape_fitting -> tracking
        2. Frenet      : raceline cubic spline + to_frenet/to_cartesian
        3. mode branch :
             free          - raceline forward window
             trailing      - raceline + trailing speed cap
             spline_avoid  - left/right cubic spline candidates,
                             fall back to trailing if both infeasible
        4. publish WpntArray
    /local_waypoints                                              OUTPUT

Run:
    /usr/bin/python3 local_planning.py --ros-args -p mode:=spline_avoid
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray


# ===========================================================================
#  PERCEPTION - Geometry helpers  (provided)
# ===========================================================================

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def scan_to_xy(ranges: np.ndarray, angle_min: float, angle_inc: float):
    # --- tunable parameters ---
    r_min = 0.05    # [m] ignore returns closer than this
    r_max = 10.0    # [m] ignore returns farther than this

    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    x = np.where(valid, ranges * np.cos(angles), np.nan)
    y = np.where(valid, ranges * np.sin(angles), np.nan)
    return x, y

# ===========================================================================
#  PERCEPTION  (Using your code from perception_assignment.py)
# ===========================================================================

def cluster(x: np.ndarray, y: np.ndarray, angle_inc: float):
    """Adaptive-breakpoint segmentation of an ordered 2-D scan.

    Two consecutive points start a new cluster when their distance exceeds an
    *adaptive* threshold that grows with range (Dietmayer breakpoint detector):

        d_max = ( r * sin(angle_inc) / sin(lambda_rad - angle_inc) + 3*sigma )

    Returns a list of clusters, each an (N, 2) ndarray of points.
    """
    # --- tunable parameters ---
    # lambda_rad: smaller λ → larger D_max → fewer breakpoints.
    # At λ=5°, r=2 m: D_max ≈ 0.105 + σ.  The range jump at the corner of an
    # F1TENTH car (≈0.3 m wide) seen at range 2 m is ~0.10–0.15 m.  Using
    # λ=5° with σ=0.06 gives D_max ≈ 0.165 m which is just above that jump so
    # the two visible faces of the car stay as ONE L-shaped cluster instead of
    # splitting into two flat sub-clusters that each get rejected.
    lambda_rad = math.radians(5.0)    # reduced from 10° → larger D_max
    sigma = 0.06                      # increased from 0.03 → keeps car corners together
    min_points = 5                    # drop clusters smaller than this

    # --- variables you will use ---
    clusters = []   # list of finished clusters -> the return value
    current = []    # points of the cluster currently being built
    prev = None     # previous valid point (x, y) for the jump test

    # TODO - Adaptive-breakpoint clustering. Big picture:
    #   x[i], y[i] are in scan order, so consecutive indices are neighbouring
    #   beams. Walk through them ONCE and cut the scan into clusters.
    #
    #   1. Invalid beam (not math.isfinite): the object ends here -> flush
    #      `current` into `clusters` if it has >= min_points, then reset
    #      `current` and `prev`.
    #   2. Valid beam:
    #        - no `prev` yet  -> start a new cluster with this point.
    #        - else: for r = hypot(xi, yi) compute the adaptive threshold
    #              d_max = (r * sin(angle_inc) / sin(lambda_rad - angle_inc)
    #                       + 3*sigma) / 2
    #          and jump = distance(prev, current point).
    #          jump > d_max -> new object (flush + start new);
    #          else          -> append point to `current`.
    #        - update `prev`.
    #   3. After the loop, flush the last `current` (>= min_points).
    #   Tip: precompute the sin(lambda_rad - angle_inc) denominator once and
    #   guard it against ~0 (abs < 1e-6).
    # Thesis Eq. 2-4: D_max = r_{n-1} * sin(Δφ) / sin(λ-Δφ) + σ_r
    # KEY: uses the PREVIOUS point's range (r_{n-1}), not the current point.
    # No division by 2, no 3*sigma — matches Algorithm 1 in the thesis exactly.
    #
    # max_nan_gap: bridge up to this many consecutive NaN/missing beams without
    # breaking the cluster.  Specular reflection on the car body can drop 1–10
    # consecutive returns, splitting an L-shaped cluster into flat sub-clusters
    # that each get rejected by the shape filters.  Bridging restores the full
    # L-shape.  Genuine object edges produce a RANGE JUMP to the background wall
    # (not NaN), so they are still caught by the D_max check below.
    max_nan_gap = 10

    denom = math.sin(lambda_rad - angle_inc)
    if abs(denom) < 1e-6:
        denom = 1e-6

    nan_count = 0   # consecutive NaN beam counter

    for i in range(len(x)):
        xi, yi = x[i], y[i]
        if not (math.isfinite(xi) and math.isfinite(yi)):
            nan_count += 1
            if nan_count > max_nan_gap:
                # Exceeded bridge limit → genuine end of object
                if len(current) >= min_points:
                    clusters.append(np.array(current))
                current = []
                prev = None
                nan_count = 0
            # else: silently bridge the gap — prev unchanged, current unchanged
        else:
            nan_count = 0
            if prev is None:
                current.append([xi, yi])
            else:
                r_prev = math.hypot(prev[0], prev[1])   # r_{n-1} from thesis
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


def l_shape_fitting(clusters):
    """L-shape rectangle fit per cluster -> list of (cx, cy, w, h, theta).

    `w` x `h` = rectangle side lengths, `theta` = heading [rad], vehicle frame.
    Search-based "closeness" fitting: rotate the cluster over many candidate
    orientations and keep the one where the points hug the rectangle edges
    tightest. Drop boxes larger than max_obs_size.

    For ONE candidate angle, project every point onto the rotated axes
    (a = along the heading, b = perpendicular). The candidate rectangle is
    just the min/max box of (a, b). Each point's "closeness" is
    1 / (distance to the NEAREST of the 4 box edges); the angle that
    maximises the summed closeness

        score(theta) = sum_i  1 / max(d_i, min_edge)

    wins. From that best angle: w, h are the two (max - min) spans, the
    centre is the box midpoint rotated back into the vehicle frame, and
    theta is the best angle itself.
    """
    # --- tunable parameters ---
    # Lidar: 270° / 1080 points → angle_inc = 0.25° = 0.004363 rad (from scan msg).
    # F1TENTH car body ≈ 0.5 m × 0.3 m.  At λ=5°, σ=0.06 the ABDA D_max at
    # range 2 m ≈ 0.165 m, which is larger than the corner jump (~0.01 m geometric
    # calculation), so both visible car faces stay in ONE cluster.
    max_obs_size = 0.60   # [m] tight: car diagonal ≤ √(0.5²+0.3²)=0.58 m < 0.60 m
    min_size     = 0.15   # [m] reduced floor → box fits visible portion more closely
    min_edge     = 0.01   # [m] d_0 from Algorithm 4 (prevents 1/0 at exact edges)
    n_angles     = 180    # 0.5 deg resolution → stable orientation on L-shape fits

    # --- variables you will use ---
    thetas = np.linspace(0.0, np.pi / 2 - np.pi / 180, n_angles)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    obstacles = []        # list of (cx, cy, w, h, theta) -> the return value

    # TODO - L-shape fitting ("closeness" criterion).
    #   Each cluster is an (N, 2) array of (x, y) points in the VEHICLE
    #   frame. Process clusters with >= 2 points (skip the rest). Big picture:
    #
    #   1. Project ALL points onto EVERY candidate orientation at once.
    #      With pts shape (N, 2) and the precomputed cos_t/sin_t shape
    #      (n_angles,):
    #          a = pts[:,0:1]*cos_t + pts[:,1:2]*sin_t   # (N, n_angles)
    #          b = -pts[:,0:1]*sin_t + pts[:,1:2]*cos_t   # (N, n_angles)
    #      `a` is the coordinate along the candidate heading, `b` across it.
    #
    #   2. Candidate rectangle = the min/max box of (a, b) per angle. Each
    #      point's distance to the NEAREST of the 4 edges, per angle:
    #          da = minimum( a - a.min(0) , a.max(0) - a )   # to an a-edge
    #          db = minimum( b - b.min(0) , b.max(0) - b )   # to a  b-edge
    #          d  = minimum(da, db)                          # nearest edge
    #      (a.min(0)/a.max(0) reduce over the N points -> shape (n_angles,),
    #      they broadcast back against the (N, n_angles) arrays.)
    #
    #   3. Closeness score per angle, then the best angle:
    #          score = np.sum(1.0 / np.maximum(d, min_edge), axis=0)  # (n_angles,)
    #          k     = np.argmax(score)
    #          theta = thetas[k]   ;   c, s = cos_t[k], sin_t[k]
    #      (min_edge stops a point exactly on an edge from making 1/d blow up.)
    #
    #   4. Re-use column k (or re-project at theta) to size the box:
    #          ak = a[:, k] ; bk = b[:, k]
    #          w  = ak.max() - ak.min()      # span along the heading
    #          h  = bk.max() - bk.min()      # span across it
    #      Box centre in the ROTATED frame is the midpoint:
    #          ca = 0.5*(ak.max()+ak.min()) ;  cb = 0.5*(bk.max()+bk.min())
    #      Rotate that centre BACK into the vehicle frame (inverse of step 1):
    #          cx = ca*c - cb*s
    #          cy = ca*s + cb*c
    #
    #   5. Reject / clamp, then store:
    #          if max(w, h) > max_obs_size: skip this cluster
    #          w = max(w, min_size) ; h = max(h, min_size)
    #          obstacles.append((cx, cy, w, h, theta))
    #
    #   Tip: do steps 1-3 fully vectorised over all n_angles (no Python loop
    #   over angles) - that is what cos_t/sin_t being arrays is for. Only the
    #   outer loop over clusters needs to be a Python loop.
    for clust in clusters:
        pts = np.array(clust)
        if len(pts) < 2:
            continue

        # Step 1: project onto all candidate orientations simultaneously
        a = pts[:, 0:1] * cos_t + pts[:, 1:2] * sin_t    # (N, n_angles)
        b = -pts[:, 0:1] * sin_t + pts[:, 1:2] * cos_t   # (N, n_angles)

        # Step 2: distance of each point to its nearest box edge
        da = np.minimum(a - a.min(axis=0), a.max(axis=0) - a)
        db = np.minimum(b - b.min(axis=0), b.max(axis=0) - b)
        d  = np.minimum(da, db)

        # Step 3: closeness score -> best angle index
        score = np.sum(1.0 / np.maximum(d, min_edge), axis=0)  # (n_angles,)
        k = int(np.argmax(score))
        theta = thetas[k]
        c, s = cos_t[k], sin_t[k]

        # Step 4: box dimensions and centre
        ak = a[:, k]
        bk = b[:, k]
        w  = float(ak.max() - ak.min())
        h  = float(bk.max() - bk.min())
        ca = 0.5 * (ak.max() + ak.min())
        cb = 0.5 * (bk.max() + bk.min())
        cx = ca * c - cb * s
        cy = ca * s + cb * c

        # Step 5: reject oversized or wall-like boxes, clamp, store.
        if max(w, h) > max_obs_size:
            continue

        # Wall rejection by raw aspect ratio (before any clamping).
        # A flat wall always has raw_minor ≈ 0 (lidar hits one flat surface).
        # An L-shaped vehicle has raw_minor = shorter-face length (>> 0.03 m).
        raw_minor = min(w, h)
        raw_major = max(w, h)
        if raw_minor < 0.03:
            continue
        # Reject strongly elongated clusters (wall segments, diagonal walls).
        # Threshold 0.08: an L-shape where the second face is barely visible
        # (side face 0.04 m on a 0.45 m first face) gives ratio 0.089 > 0.08 →
        # accepted.  Flat walls always give ratio ≈ 0 regardless of orientation.
        if raw_major > 0 and raw_minor < 0.08 * raw_major:
            continue

        w = max(w, min_size)
        h = max(h, min_size)
        obstacles.append((cx, cy, w, h, theta))

    return obstacles


def tracking(obstacles, track, dt: float, ego, init_vel=(0.0, 0.0)):
    """One linear constant-velocity Kalman tracker, run in the MAP frame.

    `obstacles` = list of (cx, cy, w, h, theta) in the VEHICLE frame.
    `ego`       = (ex, ey, eyaw) ego pose in the MAP frame (from odom).
    `track`     = (state[x,y,vx,vy], P, misses) in the MAP frame, or None.

    The obstacle selection + vehicle->map transform + track bookkeeping are
    PROVIDED. The inline Kalman predict + update on the MAP-frame (state, P)
    is the part you implement (the TODO block). Running in the map frame
    makes the estimated (vx, vy) the opponent's true WORLD velocity.
    """
    # --- tunable parameters ---
    opp_max_lat = 0.7    # [m] tighter lateral gate — side walls are ~0.8 m from centre
    max_misses = 20      # coast 1 s at 20 Hz through flat-face gaps on straights
    sigma_pos = 0.1      # measurement noise [m] for x and y (R diagonal)
    lidar_to_base_x = 0.27  # [m] base_link -> laser TF (x); odom is base_link

    # --- variables you will use ---
    ex, ey, eyaw = ego
    meas = None        # [mx, my] of the opponent in the MAP frame, or None

    # TODO (A) - pick the opponent measurement, in the MAP frame:
    #   * scan `obstacles` = (cx, cy, w, h, theta) in the VEHICLE frame
    #   * keep only those ahead (cx > 0) and within opp_max_lat laterally
    #   * take the NEAREST one (smallest hypot(cx, cy))
    #   * transform it laser -> base_link (add lidar_to_base_x to the x) ->
    #     map, using ego (ex, ey, eyaw); set meas = np.array([mx, my]).
    #   Leave meas = None when there is no valid opponent.
    # Collect full obstacle info so we can grab shape dimensions of the
    # selected obstacle for the shape-memory update below.
    best_obs = None   # (cx, cy, w, h, theta) of the chosen obstacle
    cands_full = [(cx, cy, w, h, theta) for (cx, cy, w, h, theta) in obstacles
                  if cx > 0 and abs(cy) <= opp_max_lat]
    if cands_full:
        best_obs = min(cands_full, key=lambda o: math.hypot(o[0], o[1]))
        cx_v, cy_v = best_obs[0], best_obs[1]
        bx = cx_v + lidar_to_base_x   # laser -> base_link
        by = cy_v
        cyaw = math.cos(eyaw)
        syaw = math.sin(eyaw)
        mx = ex + cyaw * bx - syaw * by
        my = ey + syaw * bx + cyaw * by
        meas = np.array([mx, my])

    # --- Association gate ---
    # Reject measurement if it is too far from the predicted track position.
    # This prevents walls or unrelated clusters from hijacking the track after
    # a teleportation or sudden ego-pose jump.
    gate_dist = 2.0   # [m]  max allowed distance between prediction and meas
    if track is not None and meas is not None:
        pred_x = track[0][0] + track[0][2] * dt
        pred_y = track[0][1] + track[0][3] * dt
        if math.hypot(meas[0] - pred_x, meas[1] - pred_y) > gate_dist:
            meas = None   # too far from predicted position → likely a wall

    if track is None:
        if meas is None:
            return None
        # Initialise shape memory from first detection if dimensions are useful
        init_w = best_obs[2] if best_obs else 0.30
        init_h = best_obs[3] if best_obs else 0.20
        # Use the last known velocity so the arrow doesn't snap to zero and
        # slowly rebuild every time the track reinitialises on a straight.
        state = np.array([meas[0], meas[1], float(init_vel[0]), float(init_vel[1])])
        return (state, np.eye(4), 0, init_w, init_h, 0)

    # Unpack track — supports 3-, 5-, and 6-element tuples
    state, P, misses = track[0], track[1], track[2]
    prev_w       = track[3] if len(track) > 3 else 0.30
    prev_h       = track[4] if len(track) > 4 else 0.20
    static_count = track[5] if len(track) > 5 else 0

    # --- matrices you will build ---
    F = None   # state-transition matrix   (4x4)
    Q = None   # process-noise matrix      (4x4)
    H = None   # measurement matrix        (2x4)
    R = None   # measurement-noise matrix  (2x2)

    # TODO - Inline linear constant-velocity Kalman on (state, P):
    #   1. Build F (x += vx*dt, y += vy*dt) and process-noise Q scaled by q.
    #   2. Predict:   state = F @ state ;   P = F @ P @ F.T + Q
    #   3. If meas is not None, update with z = [x, y], H = position pick,
    #      R = r * I_2:
    #          innov = z - H @ state
    #          S     = H @ P @ H.T + R
    #          K     = P @ H.T @ inv(S)
    #          state = state + K @ innov
    #          P     = (I_4 - K @ H) @ P

    # F: constant-velocity model (thesis Eq. 3-30, A_CV)
    F = np.array([[1.0, 0.0, dt,  0.0],
                  [0.0, 1.0, 0.0, dt ],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])

    # Q: position noise = dt (thesis Eq. 3-33); velocity noise reduced from
    # 10*dt → 3*dt.  The thesis value (10*dt) was tuned for the slow DSV platform.
    # On F1TENTH at 3+ m/s the large velocity noise causes the estimated velocity
    # vector to wiggle ±0.7 m/s on straight sections.  3*dt → ±0.39 m/s, stable
    # enough on straights while still tracking direction changes in corners.
    Q = np.diag([dt, dt, 3.0 * dt, 3.0 * dt])

    # H: measure only position (thesis Eq. 3-31, H_CV)
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])

    # R: measurement noise covariance (thesis Eq. 3-33, R_CV = diag[σ_x, σ_y])
    R = sigma_pos * np.eye(2)

    # Predict
    state = F @ state
    P = F @ P @ F.T + Q

    # Static-object pruner based on INNOVATION (position residual).
    # The Kalman velocity state takes 10+ frames to converge, so checking
    # speed directly gives false positives on freshly initialised tracks.
    # Innovation = how far the object actually moved since the last prediction:
    #   Moving opponent at 3 m/s:  innov ≈ v·dt = 0.15 m/frame  >> 0.04 m ✓
    #   Static wall:                innov ≈ sensor noise = 0–0.02 m < 0.04 m ✓
    # Requires no convergence — works correctly from frame 1.
    # Only count when meas is valid; on coasting frames (meas=None, flat-face
    # filtered on straights) leave static_count unchanged.
    if meas is not None:
        innov_vec = meas - H @ state          # predicted → measured displacement
        innov_mag = math.hypot(innov_vec[0], innov_vec[1])
        static_count = static_count + 1 if innov_mag < 0.04 else 0
        if static_count > 15:
            return None   # consistently static → wall, not the opponent

    # Update (only when a valid measurement is available)
    if meas is not None:
        innov = innov_vec   # reuse already computed innovation
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ innov
        P = (np.eye(4) - K @ H) @ P
    # ---- end TODO ----

    # Shape memory — max-hold filter.
    # Dimensions can only GROW, never shrink: once we see the full L-shape
    # (both faces), that measurement locks in even on frames where some lidar
    # dots are missing or only the rear face is visible.  The shape resets
    # only when the track is fully lost (returns None above).
    if meas is not None and best_obs is not None:
        best_w = max(prev_w, best_obs[2])   # take the larger of stored vs new
        best_h = max(prev_h, best_obs[3])
    else:
        best_w, best_h = prev_w, prev_h   # no measurement → keep shape

    misses = 0 if meas is not None else misses + 1
    if misses > max_misses:
        return None
    return (state, P, misses, best_w, best_h, static_count)

def trailing(track, ego, ego_v) -> float:
    """Speed command from the MAP-frame opponent track + ego pose & speed.

    `track` = (state[x,y,vx,vy], P, misses) in the MAP frame, or None.
    `ego`   = (ex, ey, eyaw) ;  `ego_v` = ego forward speed [m/s].
    True PD on the gap: kp on the gap error, kd on the *closing* speed.
    """
    # --- tunable parameters ---
    base_speed   = 3.0    # [m/s] free-running speed (reduced for safer trailing)
    desired_gap  = 2.0    # [m] centre-to-centre desired gap (car bodies ~0.5m each)
    detect_range = 8.0    # [m] begin PD within this range
    kp           = 2.0    # P gain — gentler to avoid oscillation
    kd           = 1.5    # D gain on closing speed
    max_speed    = 6.0    # [m/s] hard cap

    # --- variables you will use ---
    speed = base_speed   # result; default = full race speed

    # TODO - Speed (PD) from the MAP-frame track + ego. Big picture:
    #   1. track is None -> no opponent: return base_speed.
    #   2. ox, oy, vx, vy = track[0]   ;   ex, ey, eyaw = ego
    #      project onto the ego heading:
    #          opp_dist = forward gap to the opponent [m]
    #          opp_vx   = opponent forward speed [m/s]
    #          closing  = opp_vx - ego_v   (= d gap / dt, the RELATIVE speed)
    #   3. opp_dist < 0 or > detect_range -> behind/far: return base_speed.
    #   4. PD law: speed = base_speed + kp*(opp_dist-desired_gap) + kd*closing
    #      then clip to [0, min(base_speed, max_speed)] (never reverse).
    if track is None:
        return base_speed

    ox, oy, vx, vy = track[0]
    ex, ey, eyaw = ego
    cyaw = math.cos(eyaw)
    syaw = math.sin(eyaw)

    dx = ox - ex
    dy = oy - ey
    opp_dist = cyaw * dx + syaw * dy     # forward gap [m]
    opp_vx   = cyaw * vx + syaw * vy    # opponent forward speed [m/s]
    closing  = opp_vx - ego_v           # d(gap)/dt — negative means closing

    if opp_dist < 0 or opp_dist > detect_range:
        return base_speed

    speed = base_speed + kp * (opp_dist - desired_gap) + kd * closing
    speed = max(0.0, min(speed, min(base_speed, max_speed)))
    return speed


# ===========================================================================
#  GEOMETRY HELPER (provided)
# ===========================================================================
def geom_psi_kappa(x: np.ndarray, y: np.ndarray):
    """Heading & signed curvature of a non-closed (x, y) sequence."""
    n = len(x)
    psi = np.zeros(n)
    kappa = np.zeros(n)
    for i in range(n):
        if i == 0:
            dx = x[1] - x[0]; dy = y[1] - y[0]
        elif i == n - 1:
            dx = x[-1] - x[-2]; dy = y[-1] - y[-2]
        else:
            dx = (x[i + 1] - x[i - 1]) * 0.5
            dy = (y[i + 1] - y[i - 1]) * 0.5
        psi[i] = math.atan2(dy, dx)
        if 0 < i < n - 1:
            ddx = x[i + 1] - 2 * x[i] + x[i - 1]
            ddy = y[i + 1] - 2 * y[i] + y[i - 1]
            denom = (dx * dx + dy * dy) ** 1.5
            kappa[i] = (dx * ddy - dy * ddx) / max(denom, 1e-9)
    return psi, kappa


# ===========================================================================
#  ROS 2 NODE
# ===========================================================================

class LocalPlanning(Node):

    def __init__(self):
        super().__init__('local_planning')

        gp = lambda name, val: self.declare_parameter(name, val).value

        # ---- mode ('free' | 'trailing' | 'spline_avoid') --------------------
        self.mode = str(gp('mode', 'spline_avoid'))

        # ---- topics ---------------------------------------------------------
        self.scan_topic   = str(gp('scan_topic',   '/scan'))
        self.odom_topic   = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic = str(gp('global_topic', '/global_waypoints'))
        self.local_topic  = str(gp('local_topic',  '/local_waypoints'))

        # ---- Frenet slice ---------------------------------------------------
        self.local_horizon = float(gp('local_horizon', 5.0))    # [m]
        self.ds_step       = float(gp('ds_step',        0.25))  # [m]
        # ego_d -> target cosine blend distance, ~ 2 * pp_lookahead
        self.s_blend       = float(gp('s_blend',        3.0))   # [m]

        # ---- avoidance (spline_avoid) ---------------------------------------
        self.d_safe        = float(gp('d_safe',       0.7))     # [m] lateral offset
        self.s_in          = float(gp('s_in',         2.0))     # [m] min approach gap
        self.s_out         = float(gp('s_out',        2.5))     # [m] peak -> raceline
        self.trigger_range = float(gp('trigger_range', 8.0))    # [m] forward trigger range
        self.margin        = float(gp('margin',       0.2))     # [m] wall/obstacle margin
        self.obs_radius    = float(gp('obs_radius',   0.4))     # [m] obstacle inflate
        self.track_half_w  = float(gp('track_half_w', 0.8))     # [m] fallback half-width
        self.a_lat_max     = float(gp('a_lat_max',    6.0))     # [m/s^2] lat accel cap
        self.vx_scale_avoid = float(gp('vx_scale_avoid', 0.5))  # avoidance vx multiplier
        # opponents whose tracked |v| exceeds this are treated as dynamic and
        # routed to trailing only (no spline avoidance).
        self.dyn_speed_thresh = float(gp('dyn_speed_thresh', 0.5))  # [m/s]

        # ---- wall clamping (per-sample clamp + PCHIP refit) -----------------
        self.clamp_to_walls = bool(gp('clamp_to_walls', True))
        self.clamp_buffer   = float(gp('clamp_buffer',  0.05))     # [m]

        # ---- Frenet state ---------------------------------------------------
        self._sx = None
        self._sy = None
        self._vx_pchip = None
        self._dl_pchip = None
        self._dr_pchip = None
        self._s_samples = None
        self._x_samples = None
        self._y_samples = None
        self.s_total = 0.0

        # ---- perception / pose state ----------------------------------------
        self.track = None
        self.last_scan_t = None
        self.ex = self.ey = self.eyaw = 0.0
        self.ev = 0.0
        self.have_pose = False
        self.ego_s = 0.0
        self.ego_d = 0.0

        # ---- avoidance commit (hysteresis) ----------------------------------
        # Hold the same spline until ego passes s_d (raceline rejoin point).
        self._avoid_state = None

        # ---- ROS interfaces -------------------------------------------------
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray, self.global_topic, self._global_wp_cb, latched)
        self.create_subscription(Odometry,  self.odom_topic,   self._odom_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic,   self._scan_cb, 10)

        self.local_pub  = self.create_publisher(WpntArray,   self.local_topic, latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/local_waypoints/markers', 10)
        self.cand_pub   = self.create_publisher(MarkerArray, '/local_planning/candidates', 5)

        self.get_logger().info(
            f'local_planning up | mode={self.mode} | horizon={self.local_horizon} m | '
            f'global={self.global_topic} -> local={self.local_topic}')

    # ================================================================== #
    # ROS callbacks
    # ================================================================== #
    def _global_wp_cb(self, msg):
        self._build_frenet_spline(list(msg.wpnts))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.ex, self.ey = p.x, p.y
        self.eyaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.ev   = msg.twist.twist.linear.x
        self.have_pose = True

    def _scan_cb(self, msg):
        if not self.have_pose or self._sx is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self.last_scan_t is None else max(now - self.last_scan_t, 1e-3)
        self.last_scan_t = now

        # 1) perception
        ranges = np.asarray(msg.ranges, dtype=float)
        x, y = scan_to_xy(ranges, msg.angle_min, msg.angle_increment)
        clusters_xy = cluster(x, y, msg.angle_increment)
        obstacles = l_shape_fitting(clusters_xy)
        ego = (self.ex, self.ey, self.eyaw)
        self.track = tracking(obstacles, self.track, dt, ego)

        # 2) ego frenet
        self.ego_s, self.ego_d = self.to_frenet(self.ex, self.ey)

        # 3) mode branch -> WpntArray
        if self.mode == 'free':
            out, used_mode = self._build_passthrough(), 'free'
        elif self.mode == 'trailing':
            out, used_mode = self._build_trailing(), 'trailing'
        elif self.mode == 'spline_avoid':
            out, used_mode = self._build_spline_avoid_or_fallback()
        else:
            self.get_logger().warn(f"unknown mode '{self.mode}' -> free")
            out, used_mode = self._build_passthrough(), 'free'

        self.local_pub.publish(out)
        self._publish_local_markers(out, used_mode)

    # ================================================================== #
    # Frenet spline
    # ================================================================== #
    def _build_frenet_spline(self, wpnts):
        if len(wpnts) < 4:
            return
        x  = np.array([w.x_m     for w in wpnts], dtype=float)
        y  = np.array([w.y_m     for w in wpnts], dtype=float)
        vx = np.array([w.vx_mps  for w in wpnts], dtype=float)
        dl = np.array([w.d_left  for w in wpnts], dtype=float)
        dr = np.array([w.d_right for w in wpnts], dtype=float)

        if abs(x[0] - x[-1]) > 1e-6 or abs(y[0] - y[-1]) > 1e-6:
            x  = np.append(x,  x[0])
            y  = np.append(y,  y[0])
            vx = np.append(vx, vx[0])
            dl = np.append(dl, dl[0])
            dr = np.append(dr, dr[0])

        ds = np.hypot(np.diff(x), np.diff(y))
        s  = np.concatenate(([0.0], np.cumsum(ds)))

        keep = np.concatenate(([True], np.diff(s) > 1e-9))
        s = s[keep]; x = x[keep]; y = y[keep]
        vx = vx[keep]; dl = dl[keep]; dr = dr[keep]

        # fallback if d_left / d_right are all zero in the CSV
        if np.all(dl < 1e-3):
            dl = np.full_like(dl, self.track_half_w)
        if np.all(dr < 1e-3):
            dr = np.full_like(dr, self.track_half_w)

        self._sx = CubicSpline(s, x, bc_type='periodic')
        self._sy = CubicSpline(s, y, bc_type='periodic')
        self._vx_pchip = PchipInterpolator(s, vx, extrapolate=False)
        self._dl_pchip = PchipInterpolator(s, dl, extrapolate=False)
        self._dr_pchip = PchipInterpolator(s, dr, extrapolate=False)

        self._s_samples = s[:-1]
        self._x_samples = x[:-1]
        self._y_samples = y[:-1]
        self.s_total = float(s[-1])
        self.get_logger().info(
            f'frenet spline built (s_total={self.s_total:.3f} m, N={len(self._s_samples)})')

    def _psi_kappa_at(self, s):
        s = s % self.s_total
        dx  = float(self._sx(s, 1)); dy  = float(self._sy(s, 1))
        ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
        psi = math.atan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        kappa = (dx * ddy - dy * ddx) / denom if denom > 1e-12 else 0.0
        return psi, kappa

    def _vx_at(self, s):
        return float(self._vx_pchip(s % self.s_total))

    def _dl_at(self, s):
        return float(self._dl_pchip(s % self.s_total))

    def _dr_at(self, s):
        return float(self._dr_pchip(s % self.s_total))

    def to_frenet(self, x, y, n_newton=5):
        if self._sx is None:
            return 0.0, 0.0
        i = int(np.argmin((self._x_samples - x) ** 2 + (self._y_samples - y) ** 2))
        s = float(self._s_samples[i])
        for _ in range(n_newton):
            rx = float(self._sx(s)) - x
            ry = float(self._sy(s)) - y
            dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
            ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
            g  = rx * dx + ry * dy
            gp = dx * dx + dy * dy + rx * ddx + ry * ddy
            if abs(gp) < 1e-12:
                break
            s = (s - g / gp) % self.s_total
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return s, 0.0
        nx, ny = -dy / nrm, dx / nrm
        d = (x - float(self._sx(s))) * nx + (y - float(self._sy(s))) * ny
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
        nx, ny = -dy / nrm, dx / nrm
        return x0 + d * nx, y0 + d * ny

    # ================================================================== #
    # Builders
    # ================================================================== #
    def _empty_header(self):
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        return out

    def _n_pts(self):
        return max(2, int(self.local_horizon / self.ds_step) + 1)

    def _make_local_wpnts(self, target_fn, v_cap=None, use_curvature_cap=False,
                          vx_scale=1.0, use_blend=True):
        """Unified builder: target d(s) + (optional) ego cosine blend + vx.

        With use_blend=True (default, raceline/trailing):
            d(s) = target_fn(s) + (ego_d - target_fn(ego_s)) * cos_alpha(s_off)
        With use_blend=False (spline_avoid): d(s) = target_fn(s) — the
        precomputed avoidance spline is published as-is, so what PP sees
        matches the candidate visualization.

        vx = min( raceline PCHIP vx,
                  v_cap (if given, scalar from trailing),
                  sqrt(a_lat_max / |kappa|) (if use_curvature_cap) ).
        """
        n = self._n_pts()
        if use_blend:
            target_at_ego = float(target_fn(self.ego_s))
            delta = self.ego_d - target_at_ego
        else:
            delta = 0.0

        # (s, d) sequence
        s_arr = np.empty(n)
        d_arr = np.empty(n)
        for k in range(n):
            s = (self.ego_s + k * self.ds_step) % self.s_total
            d_target = float(target_fn(s))
            if not use_blend:
                d = d_target
            else:
                s_off = k * self.ds_step
                if s_off >= self.s_blend:
                    d = d_target
                else:
                    alpha = 0.5 * (1.0 + math.cos(math.pi * s_off / self.s_blend))
                    d = d_target + delta * alpha
            s_arr[k] = s
            d_arr[k] = d

        # cartesian
        xs = np.empty(n)
        ys = np.empty(n)
        for k in range(n):
            xs[k], ys[k] = self.to_cartesian(s_arr[k], d_arr[k])

        # vx
        v_base = np.array([self._vx_at(s_arr[k]) for k in range(n)])
        vx = v_base.copy()
        if v_cap is not None:
            vx = np.minimum(vx, float(v_cap))
        if use_curvature_cap:
            _, kappa_g = geom_psi_kappa(xs, ys)
            v_curv = np.sqrt(self.a_lat_max / np.maximum(np.abs(kappa_g), 1e-6))
            vx = np.minimum(vx, v_curv)
        if vx_scale != 1.0:
            vx = vx * float(vx_scale)

        # build Wpnt array
        out = self._empty_header()
        for k in range(n):
            psi, kp = self._psi_kappa_at(s_arr[k])
            w = Wpnt()
            w.id          = int(k)
            w.s_m         = float(s_arr[k])
            w.d_m         = float(d_arr[k])
            w.x_m         = float(xs[k])
            w.y_m         = float(ys[k])
            w.psi_rad     = float(psi)
            w.kappa_radpm = float(kp)
            w.vx_mps      = float(vx[k])
            w.ax_mps2     = 0.0
            w.d_right     = 0.0
            w.d_left      = 0.0
            out.wpnts.append(w)
        return out

    def _avoid_d_at(self, s, st):
        """Evaluate the committed avoidance cubic spline at s (wrap-safe)."""
        s_abs = st['ego_s_init'] + (s - st['ego_s_init']) % self.s_total
        if s_abs <= st['ego_s_init']:
            return st['ego_d_init']
        if s_abs >= st['s_d']:
            return 0.0
        return float(st['cs'](s_abs))

    def _build_passthrough(self):
        """Raceline (d_target = 0) with ego cosine blend."""
        return self._make_local_wpnts(target_fn=lambda s: 0.0)

    def _build_trailing(self):
        """Passthrough + trailing PD speed cap."""
        ego = (self.ex, self.ey, self.eyaw)
        v_cap = trailing(self.track, ego, self.ev)
        return self._make_local_wpnts(target_fn=lambda s: 0.0, v_cap=v_cap)

    def _build_from_avoid_state(self, st):
        """Publish the committed avoidance spline as-is (no ego blend)."""
        return self._make_local_wpnts(
            target_fn=lambda s: self._avoid_d_at(s, st),
            use_curvature_cap=True,
            vx_scale=self.vx_scale_avoid,
            use_blend=False)

    def _build_spline_avoid_or_fallback(self):
        """Spline avoidance with hysteresis and trailing fallback.

        (A) avoidance committed -> hold until ego passes s_d
        (B) not committed -> trigger check then try a new avoidance
              not in front / off track  -> passthrough
              gap < s_in                -> trailing
              both candidates infeasible-> trailing
              otherwise                 -> commit best candidate
        """
        # (A) committed
        if self._avoid_state is not None:
            st = self._avoid_state
            ahead = (st['s_d'] - self.ego_s) % self.s_total
            if ahead > self.s_total / 2:   # passed
                self.get_logger().info(
                    f"avoidance done (s_d={st['s_d']:.2f}, ego_s={self.ego_s:.2f}) "
                    f"-> raceline")
                self._avoid_state = None
                self._clear_candidates()
            else:
                # Keep the committed candidate visible until ego passes s_d.
                return self._build_from_avoid_state(st), 'spline_avoid'

        # (B) not committed -> try new avoidance
        if self.track is None:
            self._clear_candidates()
            return self._build_passthrough(), 'free'

        ox, oy, vx_obs, vy_obs = self.track[0]
        opp_speed = math.hypot(vx_obs, vy_obs)
        s_obs, d_obs = self.to_frenet(ox, oy)
        gap = (s_obs - self.ego_s) % self.s_total

        dl_obs = self._dl_at(s_obs)
        dr_obs = self._dr_at(s_obs)
        in_front = 0.0 < gap < self.trigger_range
        on_track = -dr_obs < d_obs < dl_obs
        if not (in_front and on_track):
            self._clear_candidates()
            return self._build_passthrough(), 'free'

        # dynamic opponent -> trailing only, never commit a spline avoidance
        if opp_speed > self.dyn_speed_thresh:
            self._clear_candidates()
            return self._build_trailing(), 'trailing'

        if gap < self.s_in:
            self.get_logger().warn(
                f'gap={gap:.2f} < s_in={self.s_in:.2f} -> trailing fallback')
            self._clear_candidates()
            return self._build_trailing(), 'trailing'

        # evaluate left/right candidates
        s_obs_rel = self.ego_s + gap   # monotone (wrap-aware)
        cand_left  = self._make_avoidance_state(s_obs_rel, +self.d_safe, 'left',  d_obs)
        cand_right = self._make_avoidance_state(s_obs_rel, -self.d_safe, 'right', d_obs)

        results = []
        for st in (cand_left, cand_right):
            if st is None:
                continue
            cost = self._evaluate_state(st, ox, oy)
            results.append({'state': st, 'cost': cost})
        self._publish_candidates(results)

        feasible = [r for r in results if r['cost'] is not None]
        if not feasible:
            self.get_logger().warn(
                'both left/right candidates infeasible -> trailing fallback')
            return self._build_trailing(), 'trailing'

        best = min(feasible, key=lambda r: r['cost'])
        self._avoid_state = best['state']  # commit
        self.get_logger().info(
            f"avoidance start: '{best['state']['label']}' "
            f"(cost={best['cost']:.3f}, s_d={best['state']['s_d']:.2f})")
        return self._build_from_avoid_state(self._avoid_state), 'spline_avoid'

    # ================================================================== #
    # Avoidance state + feasibility
    # ================================================================== #
    def _make_avoidance_state(self, s_obs_rel, d_avoid, label, d_obs=0.0):
        """3-ctrl cubic spline, then push samples out of obstacle + clamp to
        walls + PCHIP refit.

        ctrl points: (ego_s_init, ego_d_init), (s_obs_rel, d_avoid),
                     (s_obs_rel + s_out, 0).
        s_d = last ctrl = commit termination check point.

        Post-processing (clamp_to_walls):
          1. dense-sample the raw cubic over [ego_s_init, s_d]
          2. push d laterally so |d - d_obs| >= sqrt(safety^2 - (s - s_obs)^2)
             on the side selected by sign(d_avoid)        (obstacle clearance)
          3. clamp each d into [-dr(s)+margin+buffer, dl(s)-margin-buffer]
             (wall clearance — final, so walls win over obstacle push if they
             collide; _evaluate_state then catches that as infeasible)
          4. refit with PCHIP (shape-preserving, no overshoot)
        """
        ego_s_init = float(self.ego_s)
        ego_d_init = float(self.ego_d)
        s_d = s_obs_rel + self.s_out
        s_ctrl = np.array([ego_s_init, s_obs_rel, s_d], dtype=float)
        d_ctrl = np.array([ego_d_init, float(d_avoid), 0.0], dtype=float)
        if not np.all(np.diff(s_ctrl) > 1e-3):
            return None
        cs_raw = CubicSpline(s_ctrl, d_ctrl, bc_type='natural')

        if self.clamp_to_walls:
            n_clamp = 40
            s_seq = np.linspace(ego_s_init, s_d, n_clamp)
            d_seq = np.asarray(cs_raw(s_seq), dtype=float)
            inset  = self.margin + self.clamp_buffer
            safety = self.obs_radius + self.margin
            side = 1.0 if d_avoid > 0 else -1.0
            for i, s in enumerate(s_seq):
                d = d_seq[i]
                # 1) push outward of obstacle within its s-influence band
                ds_obs = s - s_obs_rel
                if abs(ds_obs) < safety:
                    lat_needed = math.sqrt(safety * safety - ds_obs * ds_obs)
                    target = d_obs + side * lat_needed
                    if side > 0:
                        d = max(d, target)
                    else:
                        d = min(d, target)
                # 2) wall clamp (final authority)
                dl =  self._dl_at(s) - inset
                dr = -self._dr_at(s) + inset
                if dl < dr:                       # corridor narrower than 2*inset
                    d = 0.5 * (dl + dr)
                else:
                    d = min(max(d, dr), dl)
                d_seq[i] = d
            cs = PchipInterpolator(s_seq, d_seq, extrapolate=False)
        else:
            cs = cs_raw

        return {
            'label':       label,
            'd_avoid':     float(d_avoid),
            's_d':         float(s_d),
            'ego_s_init':  ego_s_init,
            'ego_d_init':  ego_d_init,
            'cs':          cs,
        }

    def _sample_state_full(self, st, n=60):
        """Dense sampling over [ego_s_init, s_d] for feasibility & viz."""
        s_seq = np.linspace(st['ego_s_init'], st['s_d'], n)
        d_seq = np.array([self._avoid_d_at(s, st) for s in s_seq])
        return s_seq, d_seq

    def _evaluate_state(self, st, obs_x, obs_y):
        """Wall + obstacle feasibility + cost. Returns cost or None (infeasible)."""
        s_seq, d_seq = self._sample_state_full(st)
        # 1) track width (both walls + margin)
        for s, d in zip(s_seq, d_seq):
            dl = self._dl_at(s) - self.margin
            dr = -self._dr_at(s) + self.margin
            if d > dl or d < dr:
                return None
        # 2) min distance to obstacle (must clear inflated safety distance)
        xs = np.empty_like(s_seq)
        ys = np.empty_like(s_seq)
        for k, (s, d) in enumerate(zip(s_seq, d_seq)):
            xs[k], ys[k] = self.to_cartesian(s, d)
        obs_dist = float(np.min(np.hypot(xs - obs_x, ys - obs_y)))
        safety = self.obs_radius + self.margin
        if obs_dist < safety:
            return None
        # 3) cost: 1/clearance + mean |d|
        w_obs    = 5.0
        w_offset = 1.0
        return (w_obs / max(obs_dist - safety, 0.05)
                + w_offset * float(np.mean(np.abs(d_seq))))

    # ================================================================== #
    # Visualization
    # ================================================================== #
    def _publish_local_markers(self, wpnts, mode):
        color = (0.3, 0.8, 1.0)   # sky blue (always)
        ma = MarkerArray()
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'local_waypoints_line'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.r, line.color.g, line.color.b, line.color.a = (*color, 1.0)
        # z above candidate (0.07) and global raceline so the local path
        # always draws on top in RViz.
        for w in wpnts.wpnts:
            p = Point()
            p.x, p.y, p.z = float(w.x_m), float(w.y_m), 0.20
            line.points.append(p)
        ma.markers.append(line)
        self.marker_pub.publish(ma)

    def _clear_candidates(self):
        """Publish DELETEALL to wipe leftover candidate markers."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.cand_pub.publish(ma)

    def _publish_candidates(self, results):
        """Visualize [{'state': st, 'cost': float|None}, ...]."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        feasible = [r for r in results if r['cost'] is not None]
        best_cost = min((r['cost'] for r in feasible), default=None)
        stamp = self.get_clock().now().to_msg()
        for i, r in enumerate(results):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'avoid_candidates'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            if r['cost'] is None:
                # infeasible
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.0, 0.0, 0.4
            elif best_cost is not None and r['cost'] == best_cost:
                # best
                m.scale.x = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.95
            else:
                # feasible but not best
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.55, 0.55, 0.6
            s_seq, d_seq = self._sample_state_full(r['state'])
            for s, d in zip(s_seq, d_seq):
                x, y = self.to_cartesian(s, d)
                p = Point(); p.x, p.y, p.z = float(x), float(y), 0.07
                m.points.append(p)
            # lifetime=0 -> persist in RViz until explicit DELETEALL.
            # We only clear when ego passes s_d (avoidance complete).
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            ma.markers.append(m)
        self.cand_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanning()
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
