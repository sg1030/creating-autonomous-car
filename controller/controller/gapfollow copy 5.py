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
    'gf_disparity_threshold':     0.15,  # [m]    min range jump to be a disparity
    'gf_bubble_radius':           0.25,  # [m]    half car-width used for disparity extension
    'gf_best_window':             10,    # [idx]  half-width of smoothing kernel for gap selection
    'gf_min_speed':               0.5,   # [m/s]  speed floor — never slower than this
    'gf_speed_decay_k':           0.6,   # [-]    speed-reduction coefficient (0=constant, 1=zero at max steer)
}


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop        = EStop(self)
        self.estop_enable = bool(p('gf_estop_enable'))
        self.speed               = p('gf_speed')
        self.max_steer           = p('gf_max_steer')
        self.max_range           = p('gf_max_range')
        self.fov_rad             = float(np.deg2rad(p('gf_fov_deg')))
        self.disparity_threshold = p('gf_disparity_threshold')
        self.bubble_radius       = p('gf_bubble_radius')
        self.best_window         = int(p('gf_best_window'))
        self.min_speed           = p('gf_min_speed')
        self.speed_decay_k       = p('gf_speed_decay_k')

        self.scan = None
        self.odom = None
        self._prev_time = None

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('GapFollowNode ready (Disparity Extender)')

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
        # Disparity Extender algorithm:
        #   1. Preprocess: NaN/Inf/below-min → max_range, then clip
        #   2. Find disparities: adjacent beams that jump > disparity_threshold
        #   3. Extend: write the nearer value into the gap side by bubble_radius arc
        #   4. Select: steer toward the argmax of a smoothed processed array
        #   5. Speed: reduce proportionally to steering magnitude
        scan = self.scan
        if scan is None or len(scan.ranges) == 0:
            return 0.0, 0.0

        ranges    = np.asarray(scan.ranges, dtype=np.float32)
        angle_inc = float(scan.angle_increment)

        # --- Step 1: Preprocess ---
        # Bad readings (NaN, Inf, below range_min) → max_range so they look open,
        # not zero, to avoid steering toward them as if they were valid gaps.
        proc = np.where(
            np.isfinite(ranges) & (ranges >= scan.range_min),
            ranges,
            self.max_range,
        ).astype(np.float32)
        proc = np.clip(proc, 0.0, self.max_range)

        # FOV mask: zero out beams outside ±fov_rad so the backward corridor
        # (where the car came from) is never a candidate for gap selection,
        # and never participates in disparity detection.
        angles = (float(scan.angle_min)
                  + np.arange(len(proc), dtype=np.float32) * angle_inc)
        fov_mask = np.abs(angles) <= self.fov_rad
        proc[~fov_mask] = 0.0

        # --- Step 2: Detect disparities ---
        diffs = np.abs(np.diff(proc))
        # Only flag disparities where BOTH neighbours are inside the FOV.
        # A jump from a valid beam to a zeroed boundary beam is artificial
        # and would cause a massive extension into the valid scan region.
        both_valid = fov_mask[:-1] & fov_mask[1:]
        disparity_idx = np.where((diffs > self.disparity_threshold) & both_valid)[0]

        # --- Step 3: Extend disparities ---
        # Cap extension to 90° to prevent very close readings from flooding the array.
        max_n = max(1, int(np.pi / 2.0 / angle_inc))
        for d in disparity_idx:
            if proc[d] < proc[d + 1]:
                # Near side at index d → extend rightward into the gap (higher indices)
                near_r = max(float(proc[d]), 1e-3)
                n = min(int(np.ceil(self.bubble_radius / near_r / angle_inc)), max_n)
                hi = min(len(proc), d + 1 + n)
                proc[d + 1:hi] = np.minimum(proc[d + 1:hi], near_r)
            else:
                # Near side at index d+1 → extend leftward into the gap (lower indices)
                near_r = max(float(proc[d + 1]), 1e-3)
                n = min(int(np.ceil(self.bubble_radius / near_r / angle_inc)), max_n)
                lo = max(0, d + 1 - n)
                proc[lo:d + 1] = np.minimum(proc[lo:d + 1], near_r)

        # --- Step 4: Find best direction ---
        # Smooth before argmax so a single far beam in a narrow corridor
        # doesn't dominate over a wide, deep gap.
        if self.best_window > 0:
            w = 2 * self.best_window + 1
            kernel = np.ones(w, dtype=np.float32) / float(w)
            smoothed = np.convolve(proc, kernel, mode='same')
            best_idx = int(np.argmax(smoothed))
        else:
            best_idx = int(np.argmax(proc))

        # --- Step 5: Steering ---
        best_angle = float(scan.angle_min) + best_idx * angle_inc
        steer = float(np.clip(best_angle, -self.max_steer, self.max_steer))

        # --- Step 6: Adaptive speed ---
        # Full speed on straights; proportionally slower on corners; never below min_speed.
        speed = max(self.min_speed,
                    self.speed * (1.0 - self.speed_decay_k * abs(steer) / max(self.max_steer, 1e-3)))

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
