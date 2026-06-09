#!/usr/bin/python3
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class EStop:
    def __init__(self, node):
        self._logger = node.get_logger()
        self._slowdown_distance = 0.5

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()

        if scan is None:
            return cmd

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            self._logger.warn('LaserScan received, but ranges is empty')
            return cmd

        valid = np.isfinite(ranges)
        valid &= ranges > 0.0
        if not np.any(valid):
            self._logger.warn('LaserScan has no valid ranges')
            return cmd

        abs_ranges = np.abs(ranges)
        total_beams = max(ranges.size - 1, 1)
        indices = np.arange(ranges.size, dtype=np.float32)
        theta_deg = -135.0 + (270.0 * indices / total_beams)
        theta_rad = np.deg2rad(theta_deg)

        projected_ranges = abs_ranges * abs(np.cos(theta_rad))
        angle_mask = np.abs(theta_deg) <= 45.0
        projected_ranges[~valid] = np.inf
        projected_ranges[~angle_mask] = np.inf

        if not np.any(np.isfinite(projected_ranges)):
            self._logger.warn('No valid LaserScan ranges in -45 to 45 deg')
            return cmd

        min_idx = int(np.argmin(projected_ranges))
        min_projected = float(projected_ranges[min_idx])
        min_range = float(abs_ranges[min_idx])
        min_theta_deg = float(theta_deg[min_idx])
        min_theta_rad = float(theta_rad[min_idx])

        self._logger.warn(
            f'min_projected={min_projected:.3f} m, min_range={min_range:.3f} m, '
            f'min_idx={min_idx}, theta_deg={min_theta_deg:.3f}, theta_rad={min_theta_rad:.3f}'
        )

        if min_range < self._slowdown_distance and min_range>0.4:
            scale = max(0.0, min_range / self._slowdown_distance)
            original_speed = float(cmd.drive.speed)
            cmd.drive.speed = original_speed*0.5
            self._logger.warn(
                f'slowdown applied: scale={scale:.3f}, '
                f'speed={original_speed:.3f}->{cmd.drive.speed:.3f}'
            )
        if min_range<0.4:
            cmd.drive.speed = 0.0
        return cmd

    def is_stop_required(self, scan: LaserScan, odom: Odometry):
        self.should_stop(scan, odom)
        return False
