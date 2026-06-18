#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from cartographer_ros_msgs.srv import (
    FinishTrajectory,
    StartTrajectory,
    GetTrajectoryStates,
)
from cartographer_ros_msgs.msg import StatusCode, TrajectoryStates


class SetInitialPose(Node):
    def __init__(self):
        super().__init__('set_initial_pose_node')

        # Declare and get parameters
        self.declare_parameter('config_dir', '')
        self.declare_parameter('config_base', 'localization_2d.lua')

        self.config_dir = self.get_parameter('config_dir').value
        self.config_base = self.get_parameter('config_base').value

        self.get_logger().info(f'Config directory: {self.config_dir}')
        self.get_logger().info(f'Config basename: {self.config_base}')

        # Create service clients for Cartographer
        self.finish_trajectory_client = self.create_client(
            FinishTrajectory, '/finish_trajectory')
        self.start_trajectory_client = self.create_client(
            StartTrajectory, '/start_trajectory')
        self.get_trajectory_states_client = self.create_client(
            GetTrajectoryStates, '/get_trajectory_states')

        # Wait for services to be available. Loading the pbstream can take a
        # while, so keep waiting instead of proceeding after a single timeout.
        for name, client in (
            ('/get_trajectory_states', self.get_trajectory_states_client),
            ('/finish_trajectory', self.finish_trajectory_client),
            ('/start_trajectory', self.start_trajectory_client),
        ):
            self.get_logger().info(f'Waiting for Cartographer service {name} ...')
            while not client.wait_for_service(timeout_sec=5.0):
                if not rclpy.ok():
                    return
                self.get_logger().warn(f'Still waiting for {name} ...')
        self.get_logger().info('All Cartographer services available!')

        # Guard so we ignore new clicks while a re-localization is in progress.
        self._busy = False

        # Subscribe to initial pose from RViz. RViz publishes /initialpose with
        # BEST_EFFORT reliability, so a BEST_EFFORT subscriber is required to
        # receive it (a RELIABLE subscriber gets an incompatible-QoS warning and
        # never receives any message).
        initialpose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            initialpose_qos)

        self.get_logger().info("Click the '2D Pose Estimate' button in RViz to set the robot's pose...")

    def initial_pose_callback(self, msg):
        if self._busy:
            self.get_logger().warn('Re-localization already in progress, ignoring this pose.')
            return
        self._busy = True

        self.get_logger().info(
            f'Initial position: ({msg.pose.pose.position.x:.3f}, '
            f'{msg.pose.pose.position.y:.3f}, {msg.pose.pose.position.z:.3f})')
        self.get_logger().info(
            f'Initial orientation: ({msg.pose.pose.orientation.x:.3f}, '
            f'{msg.pose.pose.orientation.y:.3f}, {msg.pose.pose.orientation.z:.3f}, '
            f'{msg.pose.pose.orientation.w:.3f})')

        # Store the pose message for later use
        self.current_pose_msg = msg

        # Query the current trajectory states so we know which trajectory is
        # active (to finish) and which is frozen (the loaded map, used as the
        # reference frame for the new initial pose).
        self.get_logger().info('Querying trajectory states...')
        future = self.get_trajectory_states_client.call_async(
            GetTrajectoryStates.Request())
        future.add_done_callback(self.get_trajectory_states_callback)

    def get_trajectory_states_callback(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Error querying trajectory states: {e}')
            self._busy = False
            return

        if result is None or result.status.code != StatusCode.OK:
            msg = result.status.message if result is not None else 'no response'
            self.get_logger().error(f'GetTrajectoryStates failed: {msg}')
            self._busy = False
            return

        ids = list(result.trajectory_states.trajectory_id)
        states = list(result.trajectory_states.trajectory_state)

        active_id = None
        frozen_id = None
        for tid, state in zip(ids, states):
            if state == TrajectoryStates.ACTIVE and active_id is None:
                active_id = tid
            if state == TrajectoryStates.FROZEN and frozen_id is None:
                frozen_id = tid

        self.get_logger().info(
            f'Trajectory states: {list(zip(ids, states))} '
            f'(active={active_id}, frozen={frozen_id})')

        # The new trajectory is placed relative to the frozen map trajectory.
        # Fall back to the lowest id (typically 0, the loaded map) if none is
        # explicitly frozen.
        self._relative_to_trajectory_id = frozen_id if frozen_id is not None else (
            min(ids) if ids else 0)

        if active_id is None:
            # Nothing active to finish; go straight to starting a new trajectory.
            self.get_logger().warn(
                'No ACTIVE trajectory found; starting a new one directly.')
            self.start_new_trajectory()
            return

        self.finish_trajectory(active_id)

    def finish_trajectory(self, trajectory_id):
        self.get_logger().info(f'Finishing trajectory {trajectory_id}')
        finish_req = FinishTrajectory.Request()
        finish_req.trajectory_id = trajectory_id
        future = self.finish_trajectory_client.call_async(finish_req)
        future.add_done_callback(self.finish_trajectory_callback)

    def finish_trajectory_callback(self, future):
        """Callback when finish trajectory is done"""
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Error finishing trajectory: {e}')
            self._busy = False
            return

        if result is None or result.status.code != StatusCode.OK:
            msg = result.status.message if result is not None else 'no response'
            self.get_logger().error(f'FinishTrajectory failed: {msg}')
            self._busy = False
            return

        self.get_logger().info('Trajectory finished successfully')
        # Schedule start_trajectory to run after a small delay (one-shot timer)
        self._start_timer = self.create_timer(0.5, self._start_trajectory_once)

    def _start_trajectory_once(self):
        """One-shot timer callback to start new trajectory"""
        self._start_timer.cancel()
        self.destroy_timer(self._start_timer)
        self.start_new_trajectory()

    def start_new_trajectory(self):
        """Start new trajectory with initial pose"""
        self.get_logger().info(
            'Starting new trajectory with initial pose '
            f'(relative_to_trajectory_id={self._relative_to_trajectory_id})')
        start_req = StartTrajectory.Request()
        start_req.configuration_directory = self.config_dir
        start_req.configuration_basename = self.config_base
        start_req.use_initial_pose = True
        start_req.initial_pose.position.x = self.current_pose_msg.pose.pose.position.x
        start_req.initial_pose.position.y = self.current_pose_msg.pose.pose.position.y
        start_req.initial_pose.position.z = self.current_pose_msg.pose.pose.position.z
        start_req.initial_pose.orientation.x = self.current_pose_msg.pose.pose.orientation.x
        start_req.initial_pose.orientation.y = self.current_pose_msg.pose.pose.orientation.y
        start_req.initial_pose.orientation.z = self.current_pose_msg.pose.pose.orientation.z
        start_req.initial_pose.orientation.w = self.current_pose_msg.pose.pose.orientation.w
        start_req.relative_to_trajectory_id = self._relative_to_trajectory_id

        future = self.start_trajectory_client.call_async(start_req)
        future.add_done_callback(self.start_trajectory_callback)

    def start_trajectory_callback(self, future):
        """Callback when start trajectory is done"""
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Error starting trajectory: {e}')
            self._busy = False
            return

        if result is None or result.status.code != StatusCode.OK:
            msg = result.status.message if result is not None else 'no response'
            self.get_logger().error(f'StartTrajectory failed: {msg}')
            self._busy = False
            return

        self.get_logger().info(f'New trajectory started with ID: {result.trajectory_id}')
        self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = SetInitialPose()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down set_initial_pose_node...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
