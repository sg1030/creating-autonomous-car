#!/usr/bin/env python3
"""
trailing_node.py  —  LiDAR obstacle detection, Kalman tracking, and trailing speed control.

Sits between PP controller and drive mux as a speed-override filter:

  /vesc/high_level/ackermann_cmd_pp  ─┐
  /scan                               ├──>  TrailingNode  ──>  /vesc/high_level/ackermann_cmd
  /vesc/odom                          ┘

When trailing_enabled=true and an opponent is tracked within detect_range
AND the opponent lies within corridor_width/2 of the ego's planned path:
    steering = PP steering (pass-through, unchanged)
    speed    = PD control: pp_speed + kp*(gap - desired_gap) + kd*closing_speed

When trailing_enabled=false, no opponent detected, or opponent is off the
planned path (e.g. on a different part of the track):
    PP commands are passed through unchanged.

All tunable parameters live in ppc.yaml under the 'trailing' node name.
"""

import math
import os

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ===========================================================================
#  Geometry helpers
# ===========================================================================

def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _scan_to_xy(ranges: np.ndarray, angle_min: float, angle_inc: float):
    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= 0.05) & (ranges <= 10.0)
    x = np.where(valid, ranges * np.cos(angles), np.nan)
    y = np.where(valid, ranges * np.sin(angles), np.nan)
    return x, y


# ===========================================================================
#  Map-based LiDAR point filter
# ===========================================================================

def _load_pgm(path: str) -> np.ndarray:
    """Load a binary (P5) or ASCII (P2) PGM file. Returns uint8 array (H, W)."""
    with open(path, 'rb') as f:
        magic = f.readline().decode('ascii').strip()
        line  = f.readline().decode('ascii').strip()
        while line.startswith('#'):
            line = f.readline().decode('ascii').strip()
        w, h = map(int, line.split())
        _    = f.readline()   # maxval line — not needed (always 255 for 8-bit PGM)
        data = (np.frombuffer(f.read(), dtype=np.uint8)
                if magic == 'P5'
                else np.array(f.read().split(), dtype=np.uint8))
    return data.reshape((h, w))


def _build_map_filter(pgm_path: str, yaml_path: str, inflation_cells: int):
    """
    Load an occupancy-grid map and return a pre-inflated boolean mask.

    Returns (mask, origin_x, origin_y, resolution, height) on success,
            (None, None, None, None, None) on any error.

    mask[row, col] == True  →  the cell is a known static wall/boundary.
    """
    try:
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)

        resolution     = float(meta['resolution'])
        origin_x       = float(meta['origin'][0])
        origin_y       = float(meta['origin'][1])
        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        negate          = bool(meta.get('negate', 0))

        img = _load_pgm(pgm_path)   # (H, W) uint8

        # Convert pixel value to occupancy probability, then threshold.
        # negate=0  →  occ = (255 - px) / 255   (dark = occupied)
        # negate=1  →  occ = px / 255            (bright = occupied)
        if negate:
            occupied = img >= int(occupied_thresh * 255)
        else:
            occupied = img <= int((1.0 - occupied_thresh) * 255)

        if inflation_cells > 0:
            try:
                from scipy.ndimage import binary_dilation
                r      = inflation_cells
                struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
                occupied = binary_dilation(occupied, structure=struct)
            except ImportError:
                # Pure-numpy fallback: pad + slide
                pad    = np.pad(occupied, inflation_cells,
                                mode='constant', constant_values=False)
                h_img, w_img = occupied.shape
                result = occupied.copy()
                for dr in range(2 * inflation_cells + 1):
                    for dc in range(2 * inflation_cells + 1):
                        result |= pad[dr:dr + h_img, dc:dc + w_img]
                occupied = result

        return occupied, origin_x, origin_y, resolution, img.shape[0]

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def _filter_scan_by_map(x: np.ndarray, y: np.ndarray,
                         ex: float, ey: float, eyaw: float,
                         lidar_base_x: float,
                         map_mask: np.ndarray,
                         origin_x: float, origin_y: float,
                         resolution: float, map_height: int):
    """
    Set to nan every LiDAR point whose map-frame position falls inside an
    inflated-occupied cell. x, y are in the lidar frame.
    """
    map_h, map_w = map_mask.shape
    valid = np.isfinite(x) & np.isfinite(y)

    # lidar frame → base_link frame → map frame
    ce, se = math.cos(eyaw), math.sin(eyaw)
    xl_b = x[valid] + lidar_base_x
    mx   = np.full_like(x, np.nan)
    my   = np.full_like(y, np.nan)
    mx[valid] = ex + ce * xl_b - se * y[valid]
    my[valid] = ey + se * xl_b + ce * y[valid]

    # Integer cell indices
    col = np.zeros(len(x), dtype=np.int32)
    row = np.zeros(len(x), dtype=np.int32)
    col[valid] = ((mx[valid] - origin_x) / resolution).astype(np.int32)
    # PGM row 0 = top of image = max y in map  →  flip row
    row[valid] = (map_height - 1
                  - ((my[valid] - origin_y) / resolution).astype(np.int32))

    in_bounds = (valid
                 & (col >= 0) & (col < map_w)
                 & (row >= 0) & (row < map_h))

    # Clamp indices to (0,0) for out-of-bounds entries to allow safe indexing
    sc = np.where(in_bounds, col, 0)
    sr = np.where(in_bounds, row, 0)
    is_wall = in_bounds & map_mask[sr, sc]

    return np.where(is_wall, np.nan, x), np.where(is_wall, np.nan, y)


# ===========================================================================
#  Detection: adaptive-breakpoint clustering
# ===========================================================================

def _cluster(x: np.ndarray, y: np.ndarray, angle_inc: float,
             lambda_rad: float, sigma: float, min_points: int):
    sin_denom = math.sin(lambda_rad - angle_inc)
    if abs(sin_denom) < 1e-6:
        sin_denom = 1e-6

    clusters, current, prev = [], [], None
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        if not (math.isfinite(xi) and math.isfinite(yi)):
            if len(current) >= min_points:
                clusters.append(np.array(current))
            current, prev = [], None
        else:
            if prev is None:
                current.append([xi, yi])
            else:
                r = math.hypot(xi, yi)
                d_max = (r * math.sin(angle_inc) / sin_denom + 3 * sigma) / 2
                if math.hypot(xi - prev[0], yi - prev[1]) > d_max:
                    if len(current) >= min_points:
                        clusters.append(np.array(current))
                    current = [[xi, yi]]
                else:
                    current.append([xi, yi])
            prev = [xi, yi]

    if len(current) >= min_points:
        clusters.append(np.array(current))
    return clusters


# ===========================================================================
#  Detection: L-shape fitting (closeness criterion)
# ===========================================================================

def _l_shape_fitting(clusters, max_obs_size: float, n_angles: int):
    thetas = np.linspace(0.0, np.pi / 2 - np.pi / 180, n_angles)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    min_size, min_edge = 0.3, 0.01
    obstacles = []

    for pts in clusters:
        if len(pts) < 2:
            continue

        a = pts[:, 0:1] * cos_t + pts[:, 1:2] * sin_t     # (N, n_angles)
        b = -pts[:, 0:1] * sin_t + pts[:, 1:2] * cos_t    # (N, n_angles)

        a_min, a_max = a.min(axis=0), a.max(axis=0)
        b_min, b_max = b.min(axis=0), b.max(axis=0)

        da = np.minimum(a - a_min, a_max - a)
        db = np.minimum(b - b_min, b_max - b)
        score = np.sum(1.0 / np.maximum(np.minimum(da, db), min_edge), axis=0)

        k = int(np.argmax(score))
        c, s = cos_t[k], sin_t[k]
        ak, bk = a[:, k], b[:, k]
        w = float(ak.max() - ak.min())
        h = float(bk.max() - bk.min())

        if max(w, h) > max_obs_size:
            continue

        ca = 0.5 * (ak.max() + ak.min())
        cb = 0.5 * (bk.max() + bk.min())
        cx = float(ca * c - cb * s)
        cy = float(ca * s + cb * c)

        obstacles.append((cx, cy, max(w, min_size), max(h, min_size), thetas[k]))

    return obstacles


# ===========================================================================
#  Tracking: linear constant-velocity Kalman filter (map frame)
# ===========================================================================

def _tracking(obstacles, track, dt: float, ego,
              opp_max_lat: float, max_misses: int,
              q: float, r: float, lidar_to_base_x: float):
    ex, ey, eyaw = ego
    meas = None

    candidates = [(math.hypot(cx, cy), cx, cy)
                  for cx, cy, *_ in obstacles
                  if cx > 0 and abs(cy) <= opp_max_lat]
    if candidates:
        _, cx_n, cy_n = min(candidates, key=lambda t: t[0])
        bx = cx_n + lidar_to_base_x
        ce, se = math.cos(eyaw), math.sin(eyaw)
        meas = np.array([ex + ce * bx - se * cy_n,
                         ey + se * bx + ce * cy_n])

    if track is None:
        if meas is None:
            return None
        return (np.array([meas[0], meas[1], 0.0, 0.0]), np.eye(4), 0)

    state, P, misses = track

    F = np.array([[1, 0, dt,  0],
                  [0, 1,  0, dt],
                  [0, 0,  1,  0],
                  [0, 0,  0,  1]], dtype=float)
    Q = q * np.array([[dt**4/4, 0,       dt**3/2, 0      ],
                      [0,       dt**4/4, 0,       dt**3/2],
                      [dt**3/2, 0,       dt**2,   0      ],
                      [0,       dt**3/2, 0,       dt**2  ]], dtype=float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R_mat = r * np.eye(2)

    state = F @ state
    P = F @ P @ F.T + Q

    if meas is not None:
        innov = meas - H @ state
        S = H @ P @ H.T + R_mat
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ innov
        P = (np.eye(4) - K @ H) @ P

    misses = 0 if meas is not None else misses + 1
    if misses > max_misses:
        return None
    return (state, P, misses)


# ===========================================================================
#  Trailing: PD speed control
# ===========================================================================

def _trailing_speed(track, ego, ego_v: float, pp_speed: float,
                    desired_gap: float, detect_range: float,
                    kp: float, kd: float, max_speed: float):
    """Return a speed override, or None if PP speed should be used unchanged."""
    if track is None:
        return None

    ox, oy, vx, vy = track[0]
    ex, ey, eyaw = ego
    c, s = math.cos(eyaw), math.sin(eyaw)
    opp_dist = c * (ox - ex) + s * (oy - ey)
    opp_vx = c * vx + s * vy
    closing = opp_vx - ego_v

    if opp_dist < 0 or opp_dist > detect_range:
        return None

    speed = pp_speed + kp * (opp_dist - desired_gap) + kd * closing
    return max(0.0, min(speed, max_speed))


# ===========================================================================
#  ROS 2 node
# ===========================================================================

class TrailingNode(Node):

    DEFAULTS = {
        'trailing_enabled':    True,
        # path-corridor filter  (ego → PP lookahead direction)
        'corridor_width':       0.8,   # [m] lateral band around the ego→lookahead line
        # detection – clustering
        'lambda_deg':          15.0,
        'cluster_sigma':        0.2,
        'min_cluster_points':    10,
        # detection – L-shape fitting
        'max_obs_size':         1.0,
        'n_angles':              90,
        # tracking – Kalman filter
        'opp_max_lat':          1.0,
        'max_misses':            10,
        'kalman_q':             0.5,
        'kalman_r':            0.10,
        'lidar_to_base_x':     0.27,
        # trailing PD control
        'desired_gap':          2.0,
        'detect_range':         6.0,
        'trailing_kp':          5.0,
        'trailing_kd':          2.0,
        'trailing_max_speed':   8.0,
        # release ramp: max speed increase above ego_v when trailing ends
        'trailing_release_clip': 0.5,  # [m/s]
        # map-based static scan filter
        'use_map_filter':       True,
        'map_name':             '',
        'map_inflation_cells':   3,    # cells to inflate occupied mask (1 cell = 5 cm)
    }

    def __init__(self):
        super().__init__('trailing')

        for name, val in self.DEFAULTS.items():
            self.declare_parameter(name, val)
        p = lambda n: self.get_parameter(n).value

        self.enabled      = p('trailing_enabled')
        self.corr_half    = p('corridor_width') / 2.0
        self.lambda_rad   = math.radians(p('lambda_deg'))
        self.sigma        = p('cluster_sigma')
        self.min_pts      = p('min_cluster_points')
        self.max_obs_size = p('max_obs_size')
        self.n_angles     = int(p('n_angles'))
        self.opp_max_lat  = p('opp_max_lat')
        self.max_misses   = int(p('max_misses'))
        self.kq           = p('kalman_q')
        self.kr           = p('kalman_r')
        self.lidar_base_x = p('lidar_to_base_x')
        self.desired_gap  = p('desired_gap')
        self.detect_range = p('detect_range')
        self.kp           = p('trailing_kp')
        self.kd           = p('trailing_kd')
        self.max_speed    = p('trailing_max_speed')
        self.release_clip = p('trailing_release_clip')

        self.lookahead_pt = None   # (x, y) map-frame PP lookahead target

        self.track       = None
        self.last_scan_t = None
        self.ex = self.ey = self.eyaw = 0.0
        self.ego_v     = 0.0
        self.have_pose = False
        self.pp_cmd    = None

        # release-ramp state: prevents sudden acceleration when trailing ends
        self._trailing_was_active = False
        self._release_ramp_active = False

        # Map-based scan filter
        self.map_filter = None   # set to (mask, ox, oy, res, h) when loaded
        if p('use_map_filter'):
            map_name = p('map_name')
            if map_name:
                share_dir = get_package_share_directory('stack_master')
                map_dir   = os.path.join(share_dir, 'maps', map_name)
                pgm_path  = os.path.join(map_dir, f'{map_name}.pgm')
                yaml_path = os.path.join(map_dir, f'{map_name}.yaml')
                infl = int(p('map_inflation_cells'))
                mask, ox, oy, res, mh = _build_map_filter(pgm_path, yaml_path, infl)
                if mask is not None:
                    self.map_filter = (mask, ox, oy, res, mh)
                    self.get_logger().info(
                        f'Map filter loaded: {pgm_path} '
                        f'(inflation={infl} cells = {infl * res:.3f} m)')
                else:
                    self.get_logger().warn(
                        f'Map filter failed to load from {map_dir}; '
                        'proceeding without static filter.')
            else:
                self.get_logger().warn(
                    'use_map_filter=true but map_name is empty; '
                    'filter disabled. Pass map_name via ppc.yaml or launch arg.')

        self.create_subscription(Marker, '/pp/lookahead', self._lookahead_cb, 10)
        self.create_subscription(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd_pp',
            self._pp_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, '/scan',     self._scan_cb, 10)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.box_pub = self.create_publisher(
            MarkerArray, '/trailing/obstacles', 5)
        self.vel_pub = self.create_publisher(
            Marker, '/trailing/opponent_velocity', 5)

        status = 'ENABLED' if self.enabled else 'DISABLED'
        self.get_logger().info(f'TrailingNode ready — trailing {status}')

    # ──────────────────────────────────────────────────────────────────
    #  Callbacks
    # ──────────────────────────────────────────────────────────────────

    def _lookahead_cb(self, msg: Marker) -> None:
        self.lookahead_pt = (msg.pose.position.x, msg.pose.position.y)

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        self.ex, self.ey = pos.x, pos.y
        self.eyaw   = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self.ego_v  = msg.twist.twist.linear.x
        self.have_pose = True

    def _is_on_path(self, ox: float, oy: float) -> bool:
        """Return True if the opponent at (ox, oy) lies within corridor_half of
        the ray from the ego's current position toward the PP lookahead target,
        extended up to detect_range ahead."""
        if self.lookahead_pt is None:
            return True   # lookahead not yet received: allow trailing conservatively

        lx, ly = self.lookahead_pt
        dx, dy = lx - self.ex, ly - self.ey
        ray_len = math.hypot(dx, dy)
        if ray_len < 1e-6:
            return False

        # Unit vector in the direction ego → lookahead
        ux, uy = dx / ray_len, dy / ray_len

        # Signed distance of opponent along the ray
        t = (ox - self.ex) * ux + (oy - self.ey) * uy

        # Must be ahead of ego and within detect_range
        if t < 0.0 or t > self.detect_range:
            return False

        # Perpendicular distance from opponent to the ray
        perp = abs((ox - self.ex) * uy - (oy - self.ey) * ux)
        return perp < self.corr_half

    def _pp_cb(self, msg: AckermannDriveStamped) -> None:
        """Receive PP command and publish (possibly with trailing speed override)."""
        self.pp_cmd = msg

        if not self.enabled or not self.have_pose or self.lookahead_pt is None:
            self.drive_pub.publish(msg)
            return

        ego = (self.ex, self.ey, self.eyaw)
        speed_override = _trailing_speed(
            self.track, ego, self.ego_v,
            msg.drive.speed,
            self.desired_gap, self.detect_range,
            self.kp, self.kd, self.max_speed)

        # Only apply speed override when the tracked opponent is on our path
        if speed_override is not None and self.track is not None:
            ox, oy = float(self.track[0][0]), float(self.track[0][1])
            if not self._is_on_path(ox, oy):
                speed_override = None

        trailing_active = speed_override is not None

        # Detect trailing → non-trailing transition and arm the release ramp
        if self._trailing_was_active and not trailing_active:
            self._release_ramp_active = True
        if trailing_active:
            self._release_ramp_active = False
        self._trailing_was_active = trailing_active

        # Determine final output speed
        if trailing_active:
            out_speed = float(speed_override)
        elif self._release_ramp_active:
            # Clamp commanded speed to ego_v + release_clip to prevent sudden acceleration
            out_speed = min(msg.drive.speed, self.ego_v + self.release_clip)
            if msg.drive.speed <= self.ego_v + self.release_clip:
                self._release_ramp_active = False   # ramp complete
        else:
            out_speed = msg.drive.speed

        out = AckermannDriveStamped()
        out.header.stamp    = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id
        out.drive.steering_angle = msg.drive.steering_angle
        out.drive.speed = out_speed
        self.drive_pub.publish(out)

    def _scan_cb(self, msg: LaserScan) -> None:
        """Update detection and Kalman track from each new scan."""
        if not self.enabled or not self.have_pose:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = 0.05 if self.last_scan_t is None \
              else max(now - self.last_scan_t, 1e-3)
        self.last_scan_t = now

        ranges = np.asarray(msg.ranges, dtype=float)
        x, y   = _scan_to_xy(ranges, msg.angle_min, msg.angle_increment)

        # Remove LiDAR points that fall on known static map cells
        if self.map_filter is not None:
            mask, ox, oy, res, mh = self.map_filter
            n_before = int(np.sum(np.isfinite(x) & np.isfinite(y)))
            x, y = _filter_scan_by_map(
                x, y, self.ex, self.ey, self.eyaw,
                self.lidar_base_x, mask, ox, oy, res, mh)
            n_after = int(np.sum(np.isfinite(x) & np.isfinite(y)))
            self._filter_scan_cnt = getattr(self, '_filter_scan_cnt', 0) + 1
            if self._filter_scan_cnt % 20 == 1:
                self.get_logger().info(
                    f'[map_filter] scan #{self._filter_scan_cnt}: '
                    f'{n_before} pts → {n_after} pts '
                    f'({n_before - n_after} removed, '
                    f'{100*(n_before-n_after)/max(n_before,1):.0f}%)'
                )

        clusters  = _cluster(x, y, msg.angle_increment,
                             self.lambda_rad, self.sigma, self.min_pts)
        obstacles = _l_shape_fitting(clusters, self.max_obs_size, self.n_angles)
        self._publish_boxes(obstacles)

        ego = (self.ex, self.ey, self.eyaw)
        self.track = _tracking(obstacles, self.track, dt, ego,
                               self.opp_max_lat, self.max_misses,
                               self.kq, self.kr, self.lidar_base_x)
        self._publish_vel_arrow()

    # ──────────────────────────────────────────────────────────────────
    #  Visualisation
    # ──────────────────────────────────────────────────────────────────

    def _publish_vel_arrow(self) -> None:
        if self.track is None:
            return
        x, y, vx, vy = self.track[0]
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = 'trailing_opponent', 0
        m.type, m.action  = Marker.ARROW, Marker.ADD
        m.points = [Point(x=float(x),      y=float(y),      z=0.0),
                    Point(x=float(x + vx), y=float(y + vy), z=0.0)]
        m.scale.x, m.scale.y, m.scale.z = 0.08, 0.16, 0.20
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 0.9
        m.lifetime.nanosec = 300_000_000
        self.vel_pub.publish(m)

    def _publish_boxes(self, obstacles) -> None:
        arr   = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        ce, se = math.cos(self.eyaw), math.sin(self.eyaw)
        stamp  = self.get_clock().now().to_msg()
        for i, (cx, cy, w, h, theta) in enumerate(obstacles):
            bx = cx + self.lidar_base_x
            mx  = self.ex + ce * bx - se * cy
            my  = self.ey + se * bx + ce * cy
            mth = theta + self.eyaw

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = stamp
            m.ns, m.id        = 'trailing_obstacles', i
            m.type, m.action  = Marker.CUBE, Marker.ADD
            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.position.z = 0.0
            m.pose.orientation.z = float(math.sin(mth / 2.0))
            m.pose.orientation.w = float(math.cos(mth / 2.0))
            m.scale.x, m.scale.y, m.scale.z = float(w), float(h), 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.4, 1.0, 0.6
            m.lifetime.nanosec = 200_000_000
            arr.markers.append(m)

        self.box_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = TrailingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
