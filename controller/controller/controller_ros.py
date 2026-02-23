#!/usr/bin/env python3
"""
controller_ros.py  –  ROS 2 wrapper for the F1TENTH path-tracking controller.

Responsibilities
----------------
* Subscribe to global waypoints (f110_msgs/WpntArray) and odometry
  (nav_msgs/Odometry).
* Convert ROS messages → plain Python types (VehicleState, Waypoint list).
* Call the pure-Python controller (PurePursuitController / StanleyController).
* Publish the resulting drive command as ackermann_msgs/AckermannDriveStamped.

All control *logic* lives in controller.py — this file only handles ROS I/O.

Topics
------
Subscribed:
  /global_waypoints  (f110_msgs/WpntArray)  – reference path
  /ego_racecar/odom  (nav_msgs/Odometry)    – vehicle pose & speed

Published:
  /drive             (ackermann_msgs/AckermannDriveStamped)

Parameters
----------
  controller_type   (str,   default 'pure_pursuit')  – 'pure_pursuit' | 'stanley'
  lookahead         (float, default 1.0)  [m]
  wheelbase         (float, default 0.33) [m]
  max_steer         (float, default 0.4)  [rad]
  speed_kp          (float, default 1.0)
  speed_ki          (float, default 0.0)
  speed_kd          (float, default 0.1)
  control_rate_hz   (float, default 50.0)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from f110_msgs.msg import WpntArray

# ROS-free control logic
from controller.controller import (
    PurePursuitController,
    StanleyController,
    LongitudinalPIDController,
    VehicleState,
    Waypoint,
)


def _euler_from_quaternion(q) -> float:
    """Extract yaw angle from a geometry_msgs/Quaternion."""
    import math
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ControllerNode(Node):

    def __init__(self) -> None:
        super().__init__('controller_node')

        # ----------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------
        self.declare_parameter('controller_type', 'pure_pursuit')
        self.declare_parameter('lookahead', 1.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steer', 0.4)
        self.declare_parameter('speed_kp', 1.0)
        self.declare_parameter('speed_ki', 0.0)
        self.declare_parameter('speed_kd', 0.1)
        self.declare_parameter('control_rate_hz', 50.0)

        ctrl_type = self.get_parameter('controller_type').value
        lookahead = self.get_parameter('lookahead').value
        wheelbase = self.get_parameter('wheelbase').value
        max_steer = self.get_parameter('max_steer').value
        kp = self.get_parameter('speed_kp').value
        ki = self.get_parameter('speed_ki').value
        kd = self.get_parameter('speed_kd').value
        rate_hz = self.get_parameter('control_rate_hz').value

        # ----------------------------------------------------------------
        # Lateral controller selection
        # ----------------------------------------------------------------
        if ctrl_type == 'stanley':
            self.lateral_ctrl = StanleyController(
                max_steering_angle=max_steer,
                wheelbase=wheelbase,
            )
        else:  # default: pure_pursuit
            self.lateral_ctrl = PurePursuitController(
                lookahead_distance=lookahead,
                wheelbase=wheelbase,
                max_steering_angle=max_steer,
            )

        # ----------------------------------------------------------------
        # Longitudinal controller
        # ----------------------------------------------------------------
        self.long_ctrl = LongitudinalPIDController(kp=kp, ki=ki, kd=kd)

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        self.waypoints: list[Waypoint] = []
        self.state: VehicleState = VehicleState()
        self._prev_stamp: float | None = None   # for dt computation

        # ----------------------------------------------------------------
        # QoS for latched waypoints publisher
        # ----------------------------------------------------------------
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # ----------------------------------------------------------------
        # Subscriptions
        # ----------------------------------------------------------------
        self.create_subscription(
            WpntArray,
            '/global_waypoints',
            self._waypoints_callback,
            latched_qos,
        )
        self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self._odom_callback,
            10,
        )

        # ----------------------------------------------------------------
        # Publisher
        # ----------------------------------------------------------------
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10,
        )

        # ----------------------------------------------------------------
        # Control timer
        # ----------------------------------------------------------------
        self.timer = self.create_timer(1.0 / rate_hz, self._control_loop)

        self.get_logger().info(
            f'ControllerNode started (type={ctrl_type}, '
            f'lookahead={lookahead:.2f} m, rate={rate_hz:.0f} Hz)'
        )

    # --------------------------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------------------------

    def _waypoints_callback(self, msg: WpntArray) -> None:
        """Convert f110_msgs/WpntArray → list of Waypoint."""
        self.waypoints = [
            Waypoint(x=w.x_m, y=w.y_m, vx=w.vx_mps, s=w.s_m)
            for w in msg.wpnts
        ]
        self.get_logger().info(
            f'Received {len(self.waypoints)} waypoints.',
            once=True,
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Convert nav_msgs/Odometry → VehicleState."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.state = VehicleState(
            x=p.x,
            y=p.y,
            yaw=_euler_from_quaternion(q),
            v=msg.twist.twist.linear.x,
        )

    # --------------------------------------------------------------------
    # Control loop (timer callback)
    # --------------------------------------------------------------------

    def _control_loop(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if not self.waypoints:
            return

        # Compute dt for the longitudinal PID
        if self._prev_stamp is None:
            dt = 0.0
        else:
            dt = now - self._prev_stamp
        self._prev_stamp = now

        # --- Lateral control ---
        cmd = self.lateral_ctrl.compute(self.state, self.waypoints)

        # --- Longitudinal control ---
        # TODO: decide whether to use the raw waypoint speed directly or
        #       run it through the PID.  Here we pass it through PID.
        accel = self.long_ctrl.compute(cmd.speed, self.state.v, dt)
        target_speed = max(0.0, self.state.v + accel * dt) if dt > 0 else cmd.speed

        # --- Publish ---
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = cmd.steering_angle
        drive_msg.drive.speed = target_speed

        self.drive_pub.publish(drive_msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
