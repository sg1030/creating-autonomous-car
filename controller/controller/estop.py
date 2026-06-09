#!/usr/bin/python3
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32

# Modified estop.py to consider ttc by min dist / velocity case and vel/acc consideration for rapid acceleration case.

class EStop:
    def __init__(self, node):
        self._node = node
        self._logger = node.get_logger()
        self._angle_deg = self._get_or_declare_param('estop_angle_deg', 30.0)
        self._min_dist = self._get_or_declare_param('estop_min_dist', 0.3)
        self._ttc_stop = self._get_or_declare_param('estop_ttc_stop', 0.1)
        self._ttc_slowdown = self._get_or_declare_param('estop_ttc_slowdown', 0.6)
        self._base_brake_decel = self._get_or_declare_param('estop_base_brake_decel', 1.5)
        self._max_brake_decel = self._get_or_declare_param('estop_max_brake_decel', 4.0)
        self._prev_t = None
        self._ttc_active_pub = node.create_publisher(Bool, '/safety/ttc_active', 10)
        self._speed_cap_pub = node.create_publisher(Float32, '/safety/ttc_speed_cap', 10)

    def _get_or_declare_param(self, name, default):
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, default)
        return self._node.get_parameter(name).value

    def _compute_dt(self):
        now = self._node.get_clock().now().nanoseconds * 1e-9
        dt = (now - self._prev_t) if self._prev_t is not None else 0.02
        self._prev_t = now
        return max(dt, 1e-3)

    def _publish_state(self, active, speed_cap):
        active_msg = Bool()
        active_msg.data = bool(active)
        self._ttc_active_pub.publish(active_msg)

        speed_cap_msg = Float32()
        speed_cap_msg.data = float(speed_cap)
        self._speed_cap_pub.publish(speed_cap_msg)

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()

        if scan is None:
            self._publish_state(False, -1.0)
            return cmd

        dt = self._compute_dt()
        requested_speed = max(float(cmd.drive.speed), 0.0)
        odom_speed = max(float(odom.twist.twist.linear.x), 0.0) if odom is not None else 0.0
        effective_speed = max(requested_speed, odom_speed)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            self._logger.warn('LaserScan received, but ranges is empty')
            self._publish_state(False, -1.0)
            return cmd

        valid = np.isfinite(ranges)
        valid &= ranges > 0.0
        if not np.any(valid):
            self._logger.warn('LaserScan has no valid ranges')
            self._publish_state(False, -1.0)
            return cmd

        abs_ranges = np.abs(ranges)
        indices = np.arange(ranges.size, dtype=np.float32)
        theta_rad = scan.angle_min + (indices * scan.angle_increment)
        theta_deg = np.rad2deg(theta_rad)

        angle_mask = np.abs(theta_deg) <= self._angle_deg
        closing_speed = effective_speed * np.cos(theta_rad)

        ttc = np.full_like(abs_ranges, np.inf, dtype=np.float32)
        valid_ttc = valid & angle_mask & (closing_speed > 1e-3)
        ttc[valid_ttc] = abs_ranges[valid_ttc] / closing_speed[valid_ttc]

        if not np.any(np.isfinite(ttc)):
            self._publish_state(False, -1.0)
            return cmd

        min_idx = int(np.argmin(ttc))
        min_ttc = float(ttc[min_idx])
        min_range = float(abs_ranges[min_idx])
        min_theta_deg = float(theta_deg[min_idx])
        min_closing_speed = float(closing_speed[min_idx])

        self._logger.warn(
            f'min_ttc={min_ttc:.3f} s, min_range={min_range:.3f} m, '
            f'min_idx={min_idx}, theta_deg={min_theta_deg:.3f}, '
            f'closing_speed={min_closing_speed:.3f} m/s'
        )

        if min_range <= self._min_dist or min_ttc <= self._ttc_stop:
            cmd.drive.speed = 0.0
            self._publish_state(True, 0.0)
            self._logger.warn(
                f'hard stop applied: speed={requested_speed:.3f}->0.000, '
                f'ttc={min_ttc:.3f}, range={min_range:.3f}'
            )
            return cmd

        if min_ttc <= self._ttc_slowdown:
            severity = (self._ttc_slowdown - min_ttc) / max(self._ttc_slowdown - self._ttc_stop, 1e-3)
            severity = float(np.clip(severity, 0.0, 1.0))
            decel = self._base_brake_decel + severity * (self._max_brake_decel - self._base_brake_decel)
            speed_cap = max(0.0, effective_speed - decel * dt)

            original_speed = float(cmd.drive.speed)
            cmd.drive.speed = min(original_speed, speed_cap)
            self._publish_state(True, cmd.drive.speed)
            self._logger.warn(
                f'slowdown applied: severity={severity:.3f}, decel={decel:.3f}, '
                f'speed={original_speed:.3f}->{cmd.drive.speed:.3f}'
            )
            return cmd

        self._publish_state(False, -1.0)
        return cmd

    def is_stop_required(self, scan: LaserScan, odom: Odometry):
        self.should_stop(scan, odom)
        return False
