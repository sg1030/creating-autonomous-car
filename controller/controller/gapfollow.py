import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

PARAMS = {
    'control_rate_hz':  50.0,
    'gf_bubble_radius':  0.3,
    'gf_speed':          2.0,
    'gf_max_steer':      0.4,
    'gf_max_range':     10.0,
    'gf_centering_gain': 0.6,
    'gf_heading_smoothing': 0.65,
    'gf_side_bias': 0.35,
    'gf_corner_steer_gain': 1.0,
    'gf_straight_steer_gain': 1.0,
    'gf_fov_deg': 30.0,
}


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.bubble_radius = p('gf_bubble_radius')
        self.speed         = p('gf_speed')
        self.max_steer     = p('gf_max_steer')
        self.max_range     = p('gf_max_range')
        self.centering_gain = p('gf_centering_gain')
        self.heading_smoothing = p('gf_heading_smoothing')
        self.side_bias = p('gf_side_bias')
        self.corner_steer_gain = p('gf_corner_steer_gain')
        self.straight_steer_gain = p('gf_straight_steer_gain')
        self.fov_rad = math.radians(p('gf_fov_deg'))

        self.scan = None
        self.odom = None
        self._prev_target_angle = 0.0

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        scan = self.scan
        if scan is None or len(scan.ranges) == 0:
            return 0.0, 0.0

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment

        front_mask = np.abs(angles) <= self.fov_rad
        if not np.any(front_mask):
            return 0.0, 0.0

        proc_ranges = np.array(ranges[front_mask], dtype=np.float32, copy=True)
        proc_angles = np.array(angles[front_mask], dtype=np.float32, copy=False)

        valid = np.isfinite(proc_ranges) & (proc_ranges >= scan.range_min)
        proc_ranges[~valid] = 0.0
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_range)

        if not np.any(proc_ranges > 0.0):
            return 0.0, 0.0

        closest_idx = int(np.argmin(np.where(proc_ranges > 0.0, proc_ranges, np.inf)))
        closest_range = float(proc_ranges[closest_idx])

        if math.isfinite(closest_range) and closest_range > 0.0:
            bubble_half_width = int(
                math.ceil(self.bubble_radius / max(closest_range * scan.angle_increment, 1e-3))
            )
            bubble_start = max(0, closest_idx - bubble_half_width)
            bubble_end = min(proc_ranges.size, closest_idx + bubble_half_width + 1)
            proc_ranges[bubble_start:bubble_end] = 0.0

        gap_mask = proc_ranges > 0.0
        if not np.any(gap_mask):
            return 0.0, 0.0

        best_start = 0
        best_end = 0
        curr_start = None
        best_score = -1.0
        preferred_side = 0.0 if abs(self._prev_target_angle) < math.radians(5.0) else math.copysign(1.0, self._prev_target_angle)

        for idx, is_open in enumerate(gap_mask):
            if is_open and curr_start is None:
                curr_start = idx
            elif not is_open and curr_start is not None:
                curr_end = idx
                segment = proc_ranges[curr_start:curr_end]
                segment_angles = proc_angles[curr_start:curr_end]
                score = float(segment.size) + 0.25 * float(np.mean(segment))
                if segment_angles.size > 0 and preferred_side != 0.0:
                    segment_center_angle = float(segment_angles[segment_angles.size // 2])
                    if abs(segment_center_angle) > math.radians(5.0):
                        segment_side = math.copysign(1.0, segment_center_angle)
                        if segment_side == preferred_side:
                            score += self.side_bias * float(segment.size)
                if score > best_score:
                    best_start, best_end, best_score = curr_start, curr_end, score
                curr_start = None

        if curr_start is not None:
            segment = proc_ranges[curr_start:]
            segment_angles = proc_angles[curr_start:]
            score = float(segment.size) + 0.25 * float(np.mean(segment))
            if segment_angles.size > 0 and preferred_side != 0.0:
                segment_center_angle = float(segment_angles[segment_angles.size // 2])
                if abs(segment_center_angle) > math.radians(5.0):
                    segment_side = math.copysign(1.0, segment_center_angle)
                    if segment_side == preferred_side:
                        score += self.side_bias * float(segment.size)
            if score > best_score:
                best_start, best_end = curr_start, proc_ranges.size

        gap_ranges = proc_ranges[best_start:best_end]
        gap_angles = proc_angles[best_start:best_end]
        if gap_ranges.size == 0:
            return 0.0, 0.0

        representative_range = max(float(np.median(gap_ranges)), self.bubble_radius, 1e-3)
        edge_margin = int(
            math.ceil(self.bubble_radius / max(representative_range * scan.angle_increment, 1e-3))
        )
        safe_start = min(edge_margin, max(gap_ranges.size - 1, 0))
        safe_end = max(safe_start + 1, gap_ranges.size - edge_margin)

        safe_gap_ranges = gap_ranges[safe_start:safe_end]
        safe_gap_angles = gap_angles[safe_start:safe_end]
        if safe_gap_ranges.size == 0:
            safe_gap_ranges = gap_ranges
            safe_gap_angles = gap_angles

        corridor_center_idx = int(safe_gap_angles.size // 2)
        corridor_center_angle = float(safe_gap_angles[corridor_center_idx])

        center_peak = float(np.max(safe_gap_ranges))
        plateau_idx = np.flatnonzero(safe_gap_ranges >= 0.9 * center_peak)
        if plateau_idx.size > 0:
            plateau_center_idx = int(plateau_idx[plateau_idx.size // 2])
        else:
            plateau_center_idx = corridor_center_idx
        plateau_angle = float(safe_gap_angles[plateau_center_idx])

        gap_span = float(abs(safe_gap_angles[-1] - safe_gap_angles[0])) if safe_gap_angles.size > 1 else 0.0
        gap_span_ratio = np.clip(gap_span / math.radians(90.0), 0.0, 1.0)
        centering_weight = float(np.clip(self.centering_gain * gap_span_ratio, 0.0, 1.0))
        target_angle = (
            (1.0 - centering_weight) * plateau_angle +
            centering_weight * corridor_center_angle
        )
        target_angle = (
            self.heading_smoothing * self._prev_target_angle +
            (1.0 - self.heading_smoothing) * target_angle
        )
        self._prev_target_angle = float(target_angle)

        turn_mix = float(np.clip(abs(target_angle) / max(math.radians(25.0), 1e-3), 0.0, 1.0))
        steer_gain = (
            (1.0 - turn_mix) * self.straight_steer_gain +
            turn_mix * self.corner_steer_gain
        )
        steer = float(np.clip(target_angle * steer_gain, -self.max_steer, self.max_steer))

        forward_mask = np.abs(proc_angles) <= math.radians(20.0)
        forward_clearance = float(np.max(proc_ranges[forward_mask])) if np.any(forward_mask) else 0.0

        steer_ratio = abs(steer) / max(self.max_steer, 1e-3)
        clearance_ratio = np.clip(forward_clearance / max(self.max_range, 1e-3), 0.0, 1.0)
        speed_scale = max(0.25, 0.45 + 0.55 * clearance_ratio - 0.45 * steer_ratio)
        speed = float(self.speed * min(1.0, speed_scale))

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
