#!/usr/bin/env python3
"""
Trajectory Optimizer Node (Placeholder)

TODO: 학생들이 구현할 경로 최적화 노드
입력: maps/{map}/centerline.csv  (x_m, y_m, w_tr_right_m, w_tr_left_m)
출력: maps/{map}/global_waypoints.csv
"""

import rclpy
from rclpy.node import Node


class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        self.declare_parameter('map_name', '')
        self.declare_parameter('input_csv', 'centerline.csv')
        self.declare_parameter('output_csv', 'global_waypoints.csv')

        map_name = self.get_parameter('map_name').value
        input_csv = self.get_parameter('input_csv').value
        output_csv = self.get_parameter('output_csv').value

        self.get_logger().warn(
            f'[TrajectoryOptimizer] Not implemented yet!\n'
            f'  Map: {map_name}\n'
            f'  Input: maps/{map_name}/{input_csv}\n'
            f'  Output: maps/{map_name}/{output_csv}\n'
            f'  TODO: Implement trajectory optimization here.')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
