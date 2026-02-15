#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from copy import deepcopy


class SimpleMuxNode(Node):

    def __init__(self):
        """
        Initialize the node, subscribe to topics, create publishers and set up member variables.
        """
        super().__init__('simple_mux')

        # Declare parameters
        self.declare_parameter('out_topic', 'low_level/ackermann_cmd_mux/output')
        self.declare_parameter('in_topic', 'high_level/ackermann_cmd_mux/input/nav_1')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('joy_max_speed', 4.0)
        self.declare_parameter('joy_max_steer', 0.4)
        self.declare_parameter('joy_freshness_threshold', 1.0)
        self.declare_parameter('servo_min', 0.15)
        self.declare_parameter('servo_max', 0.85)
        self.declare_parameter('steering_angle_to_servo_offset', 0.5)
        self.declare_parameter('steering_angle_to_servo_gain', -1.2135)

        # Get parameters
        self.out_topic = self.get_parameter('out_topic').value
        self.in_topic = self.get_parameter('in_topic').value
        self.joy_topic = self.get_parameter('joy_topic').value
        self.rate_hz = self.get_parameter('rate_hz').value
        self.max_speed = self.get_parameter('joy_max_speed').value
        self.max_steer = self.get_parameter('joy_max_steer').value
        self.joy_freshness_threshold = self.get_parameter('joy_freshness_threshold').value

        servo_min = self.get_parameter('servo_min').value
        servo_max = self.get_parameter('servo_max').value
        steering_angle_to_servo_offset = self.get_parameter('steering_angle_to_servo_offset').value
        steering_angle_to_servo_gain = self.get_parameter('steering_angle_to_servo_gain').value

        servo_max_rad = (servo_max - steering_angle_to_servo_offset) / steering_angle_to_servo_gain
        servo_min_rad = (servo_min - steering_angle_to_servo_offset) / steering_angle_to_servo_gain

        self.servo_max_abs = min(abs(servo_max_rad), abs(servo_min_rad))

        self.current_host = None

        self.human_drive = None
        self.autodrive = None
        self.zero_msg = AckermannDriveStamped()
        self.zero_msg.header.stamp = self.get_clock().now().to_msg()
        self.zero_msg.drive.steering_angle = 0.0
        self.zero_msg.drive.speed = 0.0
        self.cur_v = 0.0
        self.prev_del_v = 0.0
        self.vel_planner = 0.0

        # Create subscribers
        self.create_subscription(
            AckermannDriveStamped,
            self.in_topic,
            self.drive_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/vel_planner',
            self.planner_callback,
            10
        )

        self.create_subscription(
            Joy,
            self.joy_topic,
            self.joy_callback,
            10
        )

        # Create publishers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.out_topic,
            10
        )

        self.current_pub = self.create_publisher(
            Float64,
            '/commands/motor/current',
            10
        )

        # Create timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)

        self.get_logger().info(f'SimpleMuxNode initialized')
        self.get_logger().info(f'  Input topic: {self.in_topic}')
        self.get_logger().info(f'  Output topic: {self.out_topic}')
        self.get_logger().info(f'  Joy topic: {self.joy_topic}')
        self.get_logger().info(f'  Max speed: {self.max_speed} m/s')
        self.get_logger().info(f'  Max steering: {self.max_steer} rad')

    def check_uptodate(self, drive_msg):
        if drive_msg is None:
            return False

        current_time = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(drive_msg.header.stamp)
        time_diff = (current_time - msg_time).nanoseconds / 1e9

        if abs(time_diff) < self.joy_freshness_threshold:
            return True
        else:
            return False

    def clip_servo(self, in_drive_msg):
        drive_msg = deepcopy(in_drive_msg)

        if drive_msg.drive.steering_angle > 0 and drive_msg.drive.steering_angle > self.servo_max_abs:
            drive_msg.drive.steering_angle = self.servo_max_abs
        elif drive_msg.drive.steering_angle < 0 and drive_msg.drive.steering_angle < -self.servo_max_abs:
            drive_msg.drive.steering_angle = -self.servo_max_abs

        return drive_msg

    def timer_callback(self):
        if self.current_host is None:
            return

        if self.current_host == "autodrive" and self.check_uptodate(self.autodrive):
            drive_msg = self.clip_servo(self.autodrive)
            drive_msg.drive.steering_angle *= 1.1
            self.drive_pub.publish(drive_msg)

        elif self.current_host == "humandrive" and self.check_uptodate(self.human_drive):
            drive_msg = self.clip_servo(self.human_drive)
            self.drive_pub.publish(drive_msg)

    def planner_callback(self, msg):
        self.vel_planner = msg.data

    def odom_callback(self, msg):
        self.cur_v = msg.twist.twist.linear.x

    def joy_callback(self, msg):
        use_human_drive = msg.buttons[4] if len(msg.buttons) > 4 else False
        use_controller = msg.buttons[5] if len(msg.buttons) > 5 else False

        if use_human_drive:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.drive.steering_angle = msg.axes[3] * self.max_steer if len(msg.axes) > 3 else 0.0
            drive_msg.drive.speed = msg.axes[1] * self.max_speed if len(msg.axes) > 1 else 0.0

            self.human_drive = drive_msg
            self.current_host = "humandrive"
            self.get_logger().info(f'Human drive: speed={drive_msg.drive.speed:.2f}, steer={drive_msg.drive.steering_angle:.2f}')

        elif use_controller:
            self.current_host = "autodrive"
            self.get_logger().info('Switched to autodrive mode')

    def drive_callback(self, msg):
        self.autodrive = msg


def main(args=None):
    rclpy.init(args=args)

    simple_mux = SimpleMuxNode()

    try:
        rclpy.spin(simple_mux)
    except KeyboardInterrupt:
        pass
    finally:
        simple_mux.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
