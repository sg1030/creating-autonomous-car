import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
<<<<<<< HEAD
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
=======
>>>>>>> parent of c350fdc (Initial Commit from HMCL)
from f110_msgs.msg import WpntArray

from controller.estop import EStop

PARAMS = {
    'control_rate_hz': 50.0,
    'pp_lookahead':     1.0,
    'pp_wheelbase':    0.33,
    'pp_max_steer':     0.4,
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop     = EStop(self)
        self.lookahead = p('pp_lookahead')
        self.wheelbase = p('pp_wheelbase')
        self.max_steer = p('pp_max_steer')

<<<<<<< HEAD
        self.odom       = None
        self.waypoints  = []
        self._prev_steer = 0.0
        self._lookahead_cap = self.lookahead_max   # overridden by MPC planner topic
=======
        self.scan      = None
        self.odom      = None
        self.waypoints = []
>>>>>>> parent of c350fdc (Initial Commit from HMCL)

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
<<<<<<< HEAD
        self.create_subscription(Odometry,  '/vesc/odom',       self._odom_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self._wp_cb, latched)
        self.create_subscription(Float32, '/mpc_planner/lookahead_cap',
                                 lambda m: setattr(self, '_lookahead_cap', m.data), 10)
        self.drive_pub     = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)
        self.get_logger().info('PP_L1Node ready')
=======
>>>>>>> parent of c350fdc (Initial Commit from HMCL)

        from sensor_msgs.msg import LaserScan
        self.create_subscription(LaserScan,  '/scan',             self._scan_cb, 10)
        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('PPNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

        if self.scan is not None and self.estop.is_stop_required(self.scan, self.odom):
            steer, speed = 0.0, 0.0
        else:
            steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
<<<<<<< HEAD
        # ── Step 1: ego pose ──────────────────────────────────────────
        pos = self.odom.pose.pose.position
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        ego_v = abs(self.odom.twist.twist.linear.x)

        wp_xy  = np.array([(w.x_m, w.y_m) for w in self.waypoints])
        dx, dy = wp_xy[:, 0] - pos.x, wp_xy[:, 1] - pos.y
        dists  = np.hypot(dx, dy)

        # ── Step 2: lateral error = distance to nearest waypoint ──────
        lat_err = float(dists.min())

        # ── Step 3: adaptive L1 distance ─────────────────────────────
        #   raw:   L = k · v
        #   floor: max(L_min,  √2 · lat_err)   ← grows when car is off-path
        #   ceil:  L_max
        L_raw = self.lookahead_k * ego_v
        L_lo  = max(self.lookahead_min, math.sqrt(2.0) * lat_err)
        L     = min(max(L_raw, L_lo), self._lookahead_cap)

        # ── Step 4: select the L1 target waypoint ────────────────────
        c, s    = math.cos(yaw), math.sin(yaw)
        local_x = c * dx + s * dy          # signed forward component
        ahead   = local_x > 0

        err = np.abs(dists - L)
        err[~ahead] = np.inf
        if np.all(np.isinf(err)):
            return 0.0, 0.0

        tgt_idx = int(np.argmin(err))
        self._publish_lookahead(self.waypoints[tgt_idx])

        # ── Step 5: steering (L1 / Pure Pursuit) ─────────────────────
        gx = self.waypoints[tgt_idx].x_m
        gy = self.waypoints[tgt_idx].y_m
        # transform target to vehicle frame
        lx = math.cos(-yaw) * (gx - pos.x) - math.sin(-yaw) * (gy - pos.y)
        ly = math.sin(-yaw) * (gx - pos.x) + math.cos(-yaw) * (gy - pos.y)
        L_sq = lx * lx + ly * ly
        if L_sq < 1e-6:
            return 0.0, 0.0

        # δ = atan(wheelbase · 2·ly / L²)
        steer = math.atan(self.wheelbase * 2.0 * ly / L_sq)

        # downscale at high speed
        steer = self._steer_speed_scale(steer, ego_v)
        # physical clamp
        steer = max(-self.max_steer, min(self.max_steer, steer))
        # rate limiter (prevents sudden jumps)
        steer = self._steer_rate_limit(steer)

        # ── Step 6: speed ─────────────────────────────────────────────
        speed = self._target_speed(pos, yaw, ego_v, lat_err)

        return steer, speed

    # ──────────────────────────────────────────────────────────────────
    # Speed helpers
    # ──────────────────────────────────────────────────────────────────
    def _target_speed(self, pos, yaw, ego_v, lat_err):
        """Read speed from the waypoint nearest to the time-propagated position."""
        dt  = self.speed_la
        px  = pos.x + math.cos(yaw) * ego_v * dt
        py  = pos.y + math.sin(yaw) * ego_v * dt

        wp_xy = np.array([(w.x_m, w.y_m) for w in self.waypoints])
        dists = np.hypot(wp_xy[:, 0] - px, wp_xy[:, 1] - py)
        idx   = int(np.argmin(dists))

        speed = float(self.waypoints[idx].vx_mps)
        speed=speed*0.9
        if speed <= 0.0:
            speed = 1.5
        

        return self._speed_adjust_lat_err(speed, lat_err)

    def _speed_adjust_lat_err(self, speed, lat_err):
        """Reduce speed when car is laterally off-path, especially in corners.

        Matches PP_Controller.py logic:
          lat_e_norm : 0 → 1 as lateral error grows from 0 → 0.5 m
          curv_norm  : 0 at mean |κ| ≤ 0.8, linearly → 1 at mean |κ| ≥ 1.2
          factor     = (1 - coeff) + coeff · exp(−lat_e_norm · curv_norm)
        """
        lat_e_norm = min(lat_err, 0.5) / 0.5   # [0, 1]

        kappas    = [abs(w.kappa_radpm) for w in self.waypoints]
        mean_kap  = float(np.mean(kappas)) if kappas else 0.0
        # 0 when mean_kappa ≤ 0.8 rad/m, ramps to 1 at 1.2 rad/m
        curv_norm = min(max(2.5 * mean_kap - 2.0, 0.0), 1.0)

        k      = self.lat_err_coeff
        factor = 1.0 - k + k * math.exp(-lat_e_norm * curv_norm)
        return max(speed * factor, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Steering helpers
    # ──────────────────────────────────────────────────────────────────
    def _steer_speed_scale(self, steer, speed):
        """Linearly reduce steer gain from steer_spd_start to steer_spd_end."""
        spd_range = max(0.1, self.steer_spd_end - self.steer_spd_start)
        t      = np.clip((speed - self.steer_spd_start) / spd_range, 0.0, 1.0)
        factor = 1.0 - t * self.steer_downscale
        return steer * factor

    def _steer_rate_limit(self, steer):
        """Bound steering change to ±steer_rate_limit per control step."""
        delta = steer - self._prev_steer
        if abs(delta) > self.steer_rate_lim:
            steer = self._prev_steer + math.copysign(self.steer_rate_lim, delta)
        self._prev_steer = steer
        return steer

    # ──────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────
    def _publish_lookahead(self, wp):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id        = 'pp_lookahead', 0
        m.type, m.action  = Marker.SPHERE, Marker.ADD
        m.pose.position.x = float(wp.x_m)
        m.pose.position.y = float(wp.y_m)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        self.lookahead_pub.publish(m)
=======
        # TODO: Pure Pursuit algorithm
        # inputs : self.odom, self.waypoints, self.lookahead, self.wheelbase
        # output : (steering [rad], speed [m/s])
        return 0.0, 0.0
>>>>>>> parent of c350fdc (Initial Commit from HMCL)


def main(args=None):
    rclpy.init(args=args)
    node = PPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
