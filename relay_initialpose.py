#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class Relay(Node):
    def __init__(self):
        super().__init__('initialpose_relay')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/sim/initialpose', 10)
        self.sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.pub.publish, 10)
        self.get_logger().info('/initialpose → /sim/initialpose relay ready')

def main():
    rclpy.init()
    rclpy.spin(Relay())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
