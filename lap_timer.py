#!/usr/bin/env python3
"""
lap_timer.py  —  ROS2 lap timer based on s_m progress along the track.

Subscribes to /global_waypoints (WpntArray) and /vesc/odom (Odometry).
Finds the nearest waypoint's s_m each control tick.
A lap is complete when s_m drops sharply (car crosses the start/finish line).

Usage:
    source install/setup.bash
    python3 lap_timer.py
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray


LAP_RESET_THRESHOLD = 1.0   # s_m drop larger than this [m] triggers a lap (s never decreases during normal driving)


class LapTimer(Node):

    def __init__(self):
        super().__init__('lap_timer')

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.waypoints   = []
        self.odom        = None
        self._prev_s     = None
        self._lap_start  = None
        self._lap_count  = 0
        self._lap_times  = []

        self.create_subscription(WpntArray, '/global_waypoints', self._wp_cb, latched)
        self.create_subscription(Odometry,  '/vesc/odom',        self._odom_cb, 10)
        self.create_timer(0.05, self._loop)   # 20 Hz

        self.get_logger().info('LapTimer ready — waiting for waypoints and odom...')

    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts

    def _odom_cb(self, msg):
        self.odom = msg

    def _loop(self):
        if not self.waypoints or self.odom is None:
            return

        pos = self.odom.pose.pose.position
        nearest = min(self.waypoints,
                      key=lambda w: math.hypot(w.x_m - pos.x, w.y_m - pos.y))
        s = nearest.s_m

        if self._prev_s is None:
            self._prev_s   = s
            self._lap_start = time.time()
            self.get_logger().info(f'Tracking started at s={s:.2f} m')
            return

        drop = self._prev_s - s

        if drop > LAP_RESET_THRESHOLD:
            now      = time.time()
            lap_time = now - self._lap_start
            self._lap_count  += 1
            self._lap_times.append(lap_time)
            self._lap_start   = now

            best = min(self._lap_times)
            print(f'\n{"─"*40}')
            print(f'  Lap {self._lap_count:>3}:  {lap_time:.3f} s')
            print(f'  Best  :  {best:.3f} s')
            if len(self._lap_times) > 1:
                avg = sum(self._lap_times) / len(self._lap_times)
                print(f'  Avg   :  {avg:.3f} s')
            print(f'{"─"*40}\n')

        self._prev_s = s


def main():
    rclpy.init()
    node = LapTimer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'Stopped. Total laps: {node._lap_count}')
        if node._lap_times:
            print(f'\nAll lap times: {[f"{t:.3f}" for t in node._lap_times]}')
            print(f'Best: {min(node._lap_times):.3f} s')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
