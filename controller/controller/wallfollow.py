import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from controller.estop import EStop

PARAMS = {
    'control_rate_hz':           50.0,
    'gf_estop_enable':           True,   # enable/disable EStop (set False to bypass)
    'gf_speed':                   2.0,   # [m/s]  target cruise speed
    'gf_max_steer':               0.4,   # [rad]  hardware limit
    'gf_max_range':               6.0,   # [m]    clip far readings to this
    'gf_fov_deg':                90.0,   # [deg]  half-angle of forward FOV; beams outside are zeroed
    'gf_steer_kp':                1.0,   # [-]    proportional gain: steer = kp * resultant_angle
    'gf_min_speed':               0.5,   # [m/s]  speed floor — never slower than this
    'gf_speed_decay_k':           0.6,   # [-]    speed-reduction coefficient (0=constant, 1=zero at max steer)
}


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop        = EStop(self)
        self.estop_enable = bool(p('gf_estop_enable'))
        self.speed        = p('gf_speed')
        self.max_steer    = p('gf_max_steer')
        self.max_range    = p('gf_max_range')
        self.fov_rad      = float(np.deg2rad(p('gf_fov_deg')))
        self.steer_kp     = p('gf_steer_kp')
        self.min_speed    = p('gf_min_speed')
        self.speed_decay_k = p('gf_speed_decay_k')

        self.scan = None
        self.odom = None
        self._prev_time = None

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('WallFollowNode ready (Gap Follow / Disparity Extender)')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _compute_dt(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (now - self._prev_time) if self._prev_time is not None else (1.0 / 50.0)
        self._prev_time = now
        return max(dt, 1e-3)

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        steer, speed = self._compute(self._compute_dt())

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        if self.estop_enable:
            msg = self.estop.should_stop(self.scan, self.odom, msg)
        self.drive_pub.publish(msg)

    def _compute(self, dt):
        # Vector-sum algorithm:
        #   Each LiDAR beam is treated as a 2-D vector (r·cosθ, r·sinθ).
        #   The resultant of all FOV-masked beams points toward the
        #   weighted "centre of open space".  Steering is proportional
        #   to the angle of that resultant; speed reduces with steer magnitude.
        scan = self.scan
        if scan is None or len(scan.ranges) == 0:
            return 0.0, 0.0

        ranges    = np.asarray(scan.ranges, dtype=np.float32)
        angle_inc = float(scan.angle_increment)
        angles    = (float(scan.angle_min)
                     + np.arange(len(ranges), dtype=np.float32) * angle_inc)

        # --- Step 1: Preprocess ---
        # Invalid readings (NaN, Inf, below range_min) → 0.0 so they contribute
        # no vector and don't pull the resultant toward unknown space.
        valid = np.isfinite(ranges) & (ranges >= scan.range_min)
        proc  = np.where(valid, ranges, 0.0).astype(np.float32)
        proc  = np.clip(proc, 0.0, self.max_range)

        # --- Step 2: FOV mask ---
        # Zero beams outside ±fov_rad so the backward corridor never
        # attracts the resultant.
        proc[np.abs(angles) > self.fov_rad] = 0.0

        # --- Step 3: Resultant vector ---
        vx = float(np.dot(proc, np.cos(angles)))
        vy = float(np.dot(proc, np.sin(angles)))

        if vx == 0.0 and vy == 0.0:
            return 0.0, self.min_speed

        goal_angle = float(np.arctan2(vy, vx))

        # --- Step 4: Proportional steering ---
        steer = float(np.clip(self.steer_kp * goal_angle,
                                -self.max_steer, self.max_steer))

        # --- Step 5: Adaptive speed ---
        # Full speed on straights; proportionally slower on corners; never below min_speed.
        speed = max(self.min_speed,
                    self.speed * (1.0 - self.speed_decay_k
                                  * abs(steer) / max(self.max_steer, 1e-3)))

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
