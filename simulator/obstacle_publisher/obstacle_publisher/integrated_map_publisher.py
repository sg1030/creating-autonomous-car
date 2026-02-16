#!/usr/bin/env python3
"""
Integrated Map Publisher for F1TENTH Simulator (ROS2)

Subscribes to static and dynamic obstacle states, combines them with the base map,
and publishes a single unified /map topic. Adjusts publishing rate based on dynamic obstacles.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import numpy as np
import cv2
import yaml
import os
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class IntegratedMapPublisher(Node):
    """Publish unified map with static + dynamic obstacles"""

    def __init__(self):
        super().__init__('integrated_map_publisher')

        # ===== Parameters =====
        self.declare_parameter('map_name', 'f')
        self.map_name = self.get_parameter('map_name').value

        # ===== Map data =====
        from ament_index_python.packages import get_package_share_directory
        stack_master_dir = get_package_share_directory('stack_master')
        self.map_dir = os.path.join(stack_master_dir, 'maps', self.map_name)

        self.original_map = None
        self.current_map = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_height = None
        self.map_width = None

        # ===== Obstacle state =====
        self.static_obstacles = []  # List of (x, y) tuples
        self.dynamic_obstacle = None  # Marker with position, heading, scale

        # ===== ROS Publishers =====
        # Map QoS: TRANSIENT_LOCAL for latched behavior (RViz compatibility)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)

        # ===== ROS Subscribers =====
        self.static_sub = self.create_subscription(
            MarkerArray,
            '/static_obstacles',
            self.static_obstacles_cb,
            10
        )
        self.dynamic_sub = self.create_subscription(
            Marker,
            '/dynamic_obstacle_state',
            self.dynamic_obstacle_cb,
            10
        )

        # ===== Load map =====
        if not self.load_map():
            self.get_logger().error('[IntegratedMapPublisher] Failed to load map')
            raise RuntimeError("Map loading failed")

        # ===== Timer (adaptive rate) =====
        self.base_rate = 10  # Hz when no dynamic obstacle
        self.dynamic_rate = 20  # Hz when dynamic obstacle active
        self.current_rate = self.base_rate
        self.timer = self.create_timer(1.0 / self.current_rate, self.timer_callback)

        self.get_logger().info('[IntegratedMapPublisher] Initialized')
        self.get_logger().info(f'  - Map: {self.map_name}')
        self.get_logger().info(f'  - Base rate: {self.base_rate} Hz')
        self.get_logger().info(f'  - Dynamic rate: {self.dynamic_rate} Hz')

    def load_map(self):
        """Load map from PNG file"""
        map_png_path = os.path.join(self.map_dir, f'{self.map_name}.png')
        map_yaml_path = os.path.join(self.map_dir, f'{self.map_name}.yaml')

        if not os.path.exists(map_png_path) or not os.path.exists(map_yaml_path):
            self.get_logger().error(f'Map files not found: {map_png_path}')
            return False

        self.original_map = cv2.imread(map_png_path, cv2.IMREAD_GRAYSCALE)
        if self.original_map is None:
            self.get_logger().error(f'Failed to load map: {map_png_path}')
            return False

        with open(map_yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)

        self.map_resolution = map_data['resolution']
        self.map_origin_x = map_data['origin'][0]
        self.map_origin_y = map_data['origin'][1]
        self.map_height = self.original_map.shape[0]
        self.map_width = self.original_map.shape[1]

        self.current_map = self.original_map.copy()

        self.get_logger().info(f'Loaded map: {self.original_map.shape}')
        return True

    def static_obstacles_cb(self, msg: MarkerArray):
        """Callback for static obstacle positions"""
        self.static_obstacles = []
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                self.static_obstacles = []
                break
            if marker.action == Marker.ADD:
                x_m = marker.pose.position.x
                y_m = marker.pose.position.y
                self.static_obstacles.append((x_m, y_m))

    def dynamic_obstacle_cb(self, msg: Marker):
        """Callback for dynamic obstacle state"""
        if msg.action == Marker.DELETE or msg.action == Marker.DELETEALL:
            self.dynamic_obstacle = None
            self.update_rate(False)
        else:
            self.dynamic_obstacle = msg
            self.update_rate(True)

    def update_rate(self, has_dynamic):
        """Update publishing rate based on dynamic obstacle presence"""
        new_rate = self.dynamic_rate if has_dynamic else self.base_rate
        if new_rate != self.current_rate:
            self.current_rate = new_rate
            self.timer.cancel()
            self.timer = self.create_timer(1.0 / self.current_rate, self.timer_callback)
            self.get_logger().info(f'[IntegratedMapPublisher] Rate changed to {self.current_rate} Hz')

    def meters_to_pixels(self, x_m, y_m):
        """Convert meters to pixel coordinates"""
        x_px = int((x_m - self.map_origin_x) / self.map_resolution)
        y_px = int((y_m - self.map_origin_y) / self.map_resolution)
        y_px = self.map_height - y_px
        return x_px, y_px

    def add_static_obstacles(self):
        """Draw static obstacles on map (50cm circles)"""
        radius_px = int(0.25 / self.map_resolution)
        for x_m, y_m in self.static_obstacles:
            center_px = self.meters_to_pixels(x_m, y_m)
            cv2.circle(self.current_map, center_px, radius_px, 0, -1)

    def add_dynamic_obstacle(self):
        """Draw dynamic obstacle on map (rotated rectangle)"""
        if self.dynamic_obstacle is None:
            return

        x_m = self.dynamic_obstacle.pose.position.x
        y_m = self.dynamic_obstacle.pose.position.y

        # Extract heading from quaternion (z, w)
        qz = self.dynamic_obstacle.pose.orientation.z
        qw = self.dynamic_obstacle.pose.orientation.w
        heading = 2 * np.arctan2(qz, qw)

        # Get obstacle dimensions
        length_m = self.dynamic_obstacle.scale.x
        width_m = self.dynamic_obstacle.scale.y

        # Convert to pixels
        center_px = self.meters_to_pixels(x_m, y_m)
        length_px = int(length_m / self.map_resolution)
        width_px = int(width_m / self.map_resolution)

        # Create rotated rectangle
        half_length = length_px / 2.0
        half_width = width_px / 2.0

        rect_corners_local = np.array([
            [-half_width, -half_length],
            [-half_width, half_length],
            [half_width, half_length],
            [half_width, -half_length]
        ])

        heading_adjusted = heading - np.pi / 2.0
        cos_h = np.cos(heading_adjusted)
        sin_h = np.sin(heading_adjusted)
        rotation_matrix = np.array([[cos_h, sin_h], [-sin_h, cos_h]])

        rect_corners_px = (rotation_matrix @ rect_corners_local.T).T + np.array(center_px)
        rect_corners_px = rect_corners_px.astype(np.int32)

        cv2.fillPoly(self.current_map, [rect_corners_px], 0)

    def update_map(self):
        """Update map with all obstacles"""
        # Start from original
        self.current_map = self.original_map.copy()

        # Add static obstacles
        self.add_static_obstacles()

        # Add dynamic obstacle
        self.add_dynamic_obstacle()

    def publish_map(self):
        """Publish occupancy grid"""
        if self.current_map is None:
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        flipped_img = np.flipud(self.current_map)
        occupancy_data = np.zeros(self.current_map.shape, dtype=np.int8)
        occupancy_data[flipped_img < 128] = 100
        occupancy_data[flipped_img >= 128] = 0

        grid_msg.data = occupancy_data.flatten().tolist()
        self.map_pub.publish(grid_msg)

    def timer_callback(self):
        """Main loop - update and publish map"""
        self.update_map()
        self.publish_map()


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
