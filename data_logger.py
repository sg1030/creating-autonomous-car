#!/usr/bin/env python3
"""
Driving data logger: records (s, ey, vx, vy, w, curv, a, delta) at ~50 Hz.
Run with:  python3 data_logger.py
Saves to:  ~/driving_data_YYYYMMDD_HHMMSS.csv  (on Ctrl+C or node shutdown)

Topics subscribed:
  /vesc/odom                          — state (vx, vy, w, a)
  /global_waypoints                   — centerline (s, ey, curv)
  /vesc/high_level/ackermann_cmd      — control (delta)
"""

import math
import csv
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray


class DataLogger(Node):

    def __init__(self):
        super().__init__('data_logger')

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.waypoints = []
        self.wp_xy  = None   # (N, 2) array cached for fast nearest-point search
        self.wp_psi = None   # (N,)   heading array
        self.wp_s   = None   # (N,)   arc-length array
        self.wp_k   = None   # (N,)   curvature array

        self.odom = None
        self.delta = 0.0

        self.prev_vx   = None
        self.prev_time = None

        self.records = []    # list of dicts, flushed to CSV on shutdown

        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        self.create_subscription(
            AckermannDriveStamped,
            '/vesc/high_level/ackermann_cmd',
            self._cmd_cb, 10
        )
        self.create_subscription(WpntArray, '/global_waypoints', self._wp_cb, latched)

        self.get_logger().info('DataLogger ready — waiting for /global_waypoints …')

    # ------------------------------------------------------------------ #
    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        n = len(self.waypoints)
        self.wp_xy  = np.array([[w.x_m, w.y_m]    for w in self.waypoints])
        self.wp_psi = np.array([w.psi_rad          for w in self.waypoints])
        self.wp_s   = np.array([w.s_m              for w in self.waypoints])
        self.wp_k   = np.array([w.kappa_radpm      for w in self.waypoints])
        self.get_logger().info(f'Loaded {n} global waypoints — logging started.')

    def _cmd_cb(self, msg):
        self.delta = msg.drive.steering_angle

    def _odom_cb(self, msg):
        if self.wp_xy is None:
            return

        # --- time -------------------------------------------------------
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # --- velocity ---------------------------------------------------
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        w  = msg.twist.twist.angular.z

        # longitudinal acceleration (finite difference)
        if self.prev_vx is None or self.prev_time is None:
            a = 0.0
        else:
            dt = t_sec - self.prev_time
            a  = (vx - self.prev_vx) / dt if dt > 1e-6 else 0.0
        self.prev_vx   = vx
        self.prev_time = t_sec

        # --- Frenet state from nearest global waypoint ------------------
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        diffs = self.wp_xy - np.array([px, py])
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        idx   = int(np.argmin(dists))

        s    = float(self.wp_s[idx])
        psi  = float(self.wp_psi[idx])
        curv = float(self.wp_k[idx])

        # lateral error: projection onto left-pointing normal (-sin ψ, cos ψ)
        dx = px - float(self.wp_xy[idx, 0])
        dy = py - float(self.wp_xy[idx, 1])
        ey = -math.sin(psi) * dx + math.cos(psi) * dy

        self.records.append({
            'time':  t_sec,
            's':     s,
            'ey':    ey,
            'vx':    vx,
            'vy':    vy,
            'w':     w,
            'curv':  curv,
            'a':     a,
            'delta': self.delta,
        })

    # ------------------------------------------------------------------ #
    def save(self):
        if not self.records:
            self.get_logger().warn('No data recorded — CSV not saved.')
            return

        stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.expanduser(f'~/driving_data_{stamp}.csv')
        fields   = ['time', 's', 'ey', 'vx', 'vy', 'w', 'curv', 'a', 'delta']

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)

        self.get_logger().info(
            f'Saved {len(self.records)} rows → {filename}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
