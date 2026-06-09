import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

PARAMS = {
    'control_rate_hz': 50.0,
    'wf_target_dist':   0.8,
    'wf_kp':            0.8,
    'wf_kd':            0.5,
    'wf_ki':            2.0,
    'wf_speed':         1.5,
    'wf_max_steer':     0.4,
    'wf_lookahead':     0.5,
    'wf_target_dist':   0.8,
}


class WallFollowNode(Node):

    def __init__(self):
        super().__init__('wall_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.target_dist = p('wf_target_dist')
        self.kp          = p('wf_kp')
        self.kd          = p('wf_kd')
        self.ki          = p('wf_ki')
        self.speed       = p('wf_speed')
        self.lookahead   = p('wf_lookahead')
        self.max_steer   = p('wf_max_steer')
        self.desired_d   = p('wf_target_dist')

        self._prev_error = 0.0
        self._integral_error = 0.0

        self.scan    = None
        self.odom    = None
        self._prev_t = None
        self._last_log_t = {}

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('WallFollowNode ready')
        self.get_logger().info(
            f'params: target_dist={self.target_dist:.2f}, kp={self.kp:.2f}, ki={self.ki:.2f}, '
            f'kd={self.kd:.2f}, speed={self.speed:.2f}, lookahead={self.lookahead:.2f}, '
            f'max_steer={self.max_steer:.2f}'
        )

    def _scan_cb(self, msg):
        self.scan = msg
        self._log_throttle('scan_cb', f'[scan_cb] received scan: n_ranges={len(msg.ranges)}', 1.0)

    def _odom_cb(self, msg):
        self.odom = msg
        vx = msg.twist.twist.linear.x
        self._log_throttle('odom_cb', f'[odom_cb] received odom: vx={vx:.3f} m/s', 1.0)

    def _loop(self):
        if self.scan is None or self.odom is None:
            self._log_throttle('[loop] waiting', '[loop] waiting for both scan and odom...', 1.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_t) if self._prev_t else 0.02
        self._prev_t = now
        self._log_throttle('[loop] dt', f'[loop] control tick: dt={dt:.4f}s', 1.0)


        steer, speed = self._compute(dt)

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)
        self._log_throttle('[loop] pub', f'[loop] publish: steer={steer:.3f}, speed={speed:.3f}', 1.0)

    def _log_throttle(self, key, message, period_s=1.0, level='info'):
        now = self.get_clock().now().nanoseconds * 1e-9
        last = self._last_log_t.get(key, None)
        if last is None or (now - last) >= period_s:
            self._last_log_t[key] = now
            logger = self.get_logger()
            if level == 'warn':
                logger.warning(message)
            else:
                logger.info(message)

    def _get_range_at_angle(self, angle_rad):
        scan = self.scan
        if scan is None:
            return None

        idx = int(round((angle_rad - scan.angle_min) / scan.angle_increment))
        if idx < 0 or idx >= len(scan.ranges):
            return None

        r = scan.ranges[idx]
        if not math.isfinite(r):
            return None
        if r < scan.range_min or r > scan.range_max:
            return None
        return r

    def _compute(self, dt):
        # TODO: Implement Wall Follow using PID control (follow the right wall)
        #
        # Goal:
        #   - Keep the vehicle at a desired distance from the right wall
        #   - Use LiDAR scan data to estimate right-wall distance/error
        #   - Compute steering using PID control
        #
        # You should return (steering, speed) from this function.
        #
        # Useful information:
        #   - self.scan.ranges             : LiDAR distance array [m]
        #   - self.scan.angle_min          : angle of first beam [rad]
        #   - self.scan.angle_increment    : angular step between beams [rad]
        #   - self.target_dist             : desired wall distance [m]
        #   - self.lookahead               : lookahead distance for projected wall error [m]
        #   - self.kp                      : proportional gain
        #   - self.ki                      : integral gain
        #   - self.kd                      : derivative gain
        #   - self._prev_error             : previous wall-distance error (for D term)
        #   - self._integral_error         : accumulated error (for I term)
        #   - dt                           : control timestep [s]
        #
        # Suggested approach (right-wall geometry):
        #   - Pick two LiDAR beams on the right side
        #   - Estimate wall angle (alpha) from the two ranges
        #   - Estimate current perpendicular distance to the wall
        #   - Project the distance error forward using lookahead
        #   - error = target_dist - projected_right_wall_distance
        #   - integral_error += error * dt
        #   - derivative = (error - prev_error) / dt
        #   - steering = kp * error + ki * integral_error + kd * derivative
        #   - Clamp steering to max steering angle if needed
        #
        # Output:
        #   - steering [rad]
        #   - speed [m/s]

        dt = max(dt, 1e-3)

        # Right-wall geometry using two beams:
        # b: right-perpendicular beam (-90 deg), a: front-right beam (-55 deg)
        b_angle = -math.pi / 2.0
        theta = math.radians(60.0)
        a_angle = b_angle + theta

        a = self._get_range_at_angle(a_angle)
        b = self._get_range_at_angle(b_angle)

        if a is None or b is None:
            self._log_throttle(
                '[compute] missing range',
                '[compute] invalid LiDAR sample at right beams; commanding safe stop',
                1.0,
                level='warn',
            )
            return 0.0, 0.0

        alpha = math.atan2((a * math.cos(theta) - b), (a * math.sin(theta)))
        d_curr = b * math.cos(alpha)
        d_proj = d_curr + self.lookahead * math.sin(alpha)

        error = self.target_dist - d_proj
        self._integral_error += error * dt
        self._integral_error = float(np.clip(self._integral_error, -1.0, 1.0))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        steer = self.kp * error + self.ki * self._integral_error + self.kd * derivative
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        steer_ratio = abs(steer) / max(self.max_steer, 1e-3)
        speed_scale = max(0.35, 1.0 - 0.3 * steer_ratio)
        speed = self.speed * speed_scale

        self._log_throttle(
            '[compute] geom',
            f'[compute] a={a:.3f}, b={b:.3f}, alpha={alpha:.3f}, d_curr={d_curr:.3f}, d_proj={d_proj:.3f}',
            1.0,
        )
        self._log_throttle(
            '[compute] pid',
            f'[compute] err={error:.3f}, I={self._integral_error:.3f}, D={derivative:.3f}, '
            f'steer={steer:.3f}, speed={speed:.3f}',
            1.0,
        )

        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
