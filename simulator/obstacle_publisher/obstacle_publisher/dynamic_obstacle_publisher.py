#!/usr/bin/env python3
"""
Dynamic Obstacle State Publisher for F1TENTH Simulator (ROS2)

Publishes dynamic obstacle state (position, heading, size) instead of modifying the map directly.
The integrated_map_publisher in the simulator will handle map rendering.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
from visualization_msgs.msg import Marker, MarkerArray


class DynamicObstaclePublisher(Node):
    """Publish dynamic obstacle state for integrated map publisher"""

    def __init__(self):
        super().__init__('dynamic_obstacle_publisher')

        # ===== Parameters =====
        self.declare_parameter('update_rate', 20)
        self.declare_parameter('speed_scaler', 1.0)
        self.declare_parameter('constant_speed', False)
        self.declare_parameter('starting_s', 0.0)
        self.declare_parameter('obstacle_length_m', 0.65)
        self.declare_parameter('obstacle_width_m', 0.35)
        self.declare_parameter('map_name', 'f')

        self.update_rate = self.get_parameter('update_rate').value
        self.speed_scaler = self.get_parameter('speed_scaler').value
        self.constant_speed = self.get_parameter('constant_speed').value
        self.starting_s = self.get_parameter('starting_s').value
        self.obstacle_length_m = self.get_parameter('obstacle_length_m').value
        self.obstacle_width_m = self.get_parameter('obstacle_width_m').value
        self.map_name = self.get_parameter('map_name').value

        self.looptime = 1.0 / self.update_rate

        # ===== Map directory =====
        from ament_index_python.packages import get_package_share_directory
        stack_master_dir = get_package_share_directory('stack_master')
        self.map_dir = os.path.join(stack_master_dir, 'maps', self.map_name)

        # ===== Obstacle state =====
        self.current_s = self.starting_s
        self.current_speed = 0.0

        # ===== Trajectory waypoints =====
        self.waypoints = []
        self.waypoints_s_array = None
        self.max_s = None

        # ===== ROS Publishers =====
        self.state_pub = self.create_publisher(Marker, '/dynamic_obstacle_state', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dynamic_obstacle_marker', 1)

        # ===== Load trajectory =====
        if not self.load_trajectory_from_csv():
            self.get_logger().warn('[DynamicObstaclePublisher] Failed to load trajectory, using dummy')

        # ===== Timer =====
        self.timer = self.create_timer(self.looptime, self.timer_callback)

        self.get_logger().info('[DynamicObstaclePublisher] Initialized')
        self.get_logger().info(f'  - Map: {self.map_name}')
        self.get_logger().info(f'  - Speed scaler: {self.speed_scaler}')
        self.get_logger().info(f'  - Obstacle size: {self.obstacle_length_m}m x {self.obstacle_width_m}m')
        self.get_logger().info(f'  - Update rate: {self.update_rate}Hz')
        self.get_logger().info(f'  - Publishing state to /dynamic_obstacle_state')

    def load_trajectory_from_csv(self):
        """Load centerline from CSV"""
        csv_path = os.path.join(self.map_dir, 'centerline.csv')

        if not os.path.exists(csv_path):
            self.get_logger().warn(f'Centerline not found: {csv_path}')
            self.create_dummy_centerline()
            return False

        self.get_logger().info(f'Loading centerline: {csv_path}')

        waypoints_data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints_data.append(row)

        if len(waypoints_data) == 0:
            self.get_logger().error('Empty centerline CSV')
            return False

        x_array = np.array([float(row['x_m']) for row in waypoints_data])
        y_array = np.array([float(row['y_m']) for row in waypoints_data])

        dx = np.diff(x_array, prepend=x_array[0])
        dy = np.diff(y_array, prepend=y_array[0])
        ds = np.sqrt(dx**2 + dy**2)
        s_array = np.cumsum(ds)

        if 'vx_mps' in waypoints_data[0]:
            speed_array = np.array([float(row['vx_mps']) for row in waypoints_data])
        else:
            speed_array = np.ones(len(waypoints_data)) * 1.0

        if self.constant_speed:
            speed_array = np.ones(len(waypoints_data)) * self.speed_scaler
        else:
            speed_array = speed_array * self.speed_scaler

        if 'heading' in waypoints_data[0]:
            heading_array = np.array([float(row['heading']) for row in waypoints_data])
        else:
            heading_array = np.arctan2(np.diff(y_array, append=y_array[0]),
                                       np.diff(x_array, append=x_array[0]))

        self.waypoints = []
        for i in range(len(x_array)):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i]
            }
            self.waypoints.append(wpnt)

        self.waypoints_s_array = s_array
        self.max_s = s_array[-1]

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints, max_s={self.max_s:.2f}m')
        return True

    def create_dummy_centerline(self):
        """Create circular centerline for testing"""
        num_points = 100
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 5.0

        x_array = radius * np.cos(theta)
        y_array = radius * np.sin(theta)

        dx = np.diff(x_array, prepend=x_array[-1])
        dy = np.diff(y_array, prepend=y_array[-1])
        ds = np.sqrt(dx**2 + dy**2)
        s_array = np.cumsum(ds)

        speed_array = np.ones(num_points) * 2.0 * self.speed_scaler
        heading_array = theta + np.pi/2

        self.waypoints = []
        for i in range(num_points):
            wpnt = {
                'x_m': x_array[i],
                'y_m': y_array[i],
                's_m': s_array[i],
                'vx_mps': speed_array[i],
                'heading': heading_array[i]
            }
            self.waypoints.append(wpnt)

        self.waypoints_s_array = s_array
        self.max_s = s_array[-1]

        self.get_logger().info(f'Dummy centerline: {len(self.waypoints)} waypoints')

    def get_obstacle_state(self):
        """Get obstacle position, heading, speed"""
        if len(self.waypoints) == 0:
            return None, None, None, None

        idx = np.abs(self.waypoints_s_array - self.current_s).argmin()
        wpnt = self.waypoints[idx]
        return wpnt['x_m'], wpnt['y_m'], wpnt['heading'], wpnt['vx_mps']

    def publish_state(self, x_m, y_m, heading):
        """Publish obstacle state as Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = 0.0

        # Set orientation from heading
        marker.pose.orientation.z = np.sin(heading / 2.0)
        marker.pose.orientation.w = np.cos(heading / 2.0)

        # Set size
        marker.scale.x = self.obstacle_length_m
        marker.scale.y = self.obstacle_width_m
        marker.scale.z = 0.5

        # Color doesn't matter for state, but set for visibility
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.state_pub.publish(marker)

    def publish_visualization_marker(self, x_m, y_m, heading):
        """Publish visualization marker (red cube)"""
        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x_m
        marker.pose.position.y = y_m
        marker.pose.position.z = 0.0

        marker.pose.orientation.z = np.sin(heading / 2.0)
        marker.pose.orientation.w = np.cos(heading / 2.0)

        marker.scale.x = self.obstacle_length_m
        marker.scale.y = self.obstacle_width_m
        marker.scale.z = 0.5

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def timer_callback(self):
        """Main loop - publish obstacle state"""
        if len(self.waypoints) == 0:
            return

        x_m, y_m, heading, speed_mps = self.get_obstacle_state()

        if x_m is None:
            return

        # Publish state for integrated map publisher
        self.publish_state(x_m, y_m, heading)

        # Publish visualization marker
        self.publish_visualization_marker(x_m, y_m, heading)

        # Update position
        self.current_s = (self.current_s + speed_mps * self.looptime) % self.max_s


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
