#!/usr/bin/env python3
"""
PP_L1.py  —  L1-guidance Pure Pursuit controller

Improvements over basic PP.py (inspired by PP_Controller.py / ForzaETH stack):

  1. Adaptive L1 lookahead with lateral-error floor
       L = clip(k·v,  [max(L_min, √2·|d|),  L_max])
       When the car drifts off the path, the lookahead floor grows so PP
       steers back more aggressively.

  2. Separate speed lookahead
       Target speed is read from the waypoint nearest to the position
       propagated (speed_lookahead seconds) ahead, not from the steering
       target.  Gives smoother, more anticipatory speed commands.

  3. Lateral-error speed reduction
       In corners, speed is reduced proportionally to lateral offset and
       path curvature: v *= (1 - k + k·exp(-lat_e_norm · curv_norm))

  4. High-speed steer downscaling
       Steer gain is linearly reduced from steer_spd_start to steer_spd_end
       to prevent over-steering at speed.

  5. Steer rate limiting
       Steering angle change per control step is bounded to steer_rate_limit
       to eliminate sudden jumps.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

PARAMS = {
    'control_rate_hz': 50.0,
    'pp_wheelbase':    0.33,    # [m]
    'pp_max_steer':    0.4,     # [rad]

    # ── Adaptive L1 lookahead ──────────────────────────────────────────
    'pp_lookahead':    0.5,     # [m]   absolute minimum lookahead
    'pp_t_clip_max':   3.0,     # [m]   absolute maximum lookahead
    'lookahead_k':     0.4,     # [s]   L = k·v  (before clamping)

    # ── Speed ─────────────────────────────────────────────────────────
    'speed_lookahead': 0.2,     # [s]   propagate position by this to read speed
    'lat_speed_gain':  1.0,     # [≥0]  speed reduction per [m] of lateral error (0=off)
    'delta_speed_gain': 2.0,    # [≥0]  speed reduction per [rad] of steering angle (0=off)

    # ── Steering ──────────────────────────────────────────────────────
    'steer_spd_start': 3.0,     # [m/s] speed at which steer downscaling begins
    'steer_spd_end':   6.0,     # [m/s] speed at which downscaling saturates
    'steer_downscale': 0.3,     # [0–1] fraction to remove at steer_spd_end
    'steer_rate_limit': 0.4,    # [rad] max |Δsteer| per control step
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.wheelbase       = p('pp_wheelbase')
        self.max_steer       = p('pp_max_steer')
        self.lookahead_min   = p('pp_lookahead')
        self.lookahead_max   = p('pp_t_clip_max')
        self.lookahead_k     = p('lookahead_k')
        self.speed_la        = p('speed_lookahead')
        self.lat_speed_gain   = p('lat_speed_gain')
        self.delta_speed_gain = p('delta_speed_gain')
        self.steer_spd_start  = p('steer_spd_start')
        self.steer_spd_end   = p('steer_spd_end')
        self.steer_downscale = p('steer_downscale')
        self.steer_rate_lim  = p('steer_rate_limit')

        self.odom       = None
        self.waypoints  = []
        self._prev_steer = 0.0
        self._lookahead_cap = self.lookahead_max   # overridden by MPC planner topic

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Odometry,  '/vesc/odom',       self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._wp_cb, latched)
        self.create_subscription(Float32, '/mpc_planner/lookahead_cap',
                                 lambda m: setattr(self, '_lookahead_cap', m.data), 10)
        self.drive_pub     = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)
        self.get_logger().info('PP_L1Node ready')

    # ──────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ──────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self,  msg): self.waypoints = msg.wpnts

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return
        steer, speed = self._compute()
        msg = AckermannDriveStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    # Main compute
    # ──────────────────────────────────────────────────────────────────
    def _compute(self):
        # ── Step 1: ego pose ──────────────────────────────────────────
        pos = self.odom.pose.pose.position
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        ego_v = abs(self.odom.twist.twist.linear.x)

        wp_xy  = np.array([(w.x_m, w.y_m) for w in self.waypoints])
        dx, dy = wp_xy[:, 0] - pos.x, wp_xy[:, 1] - pos.y
        dists  = np.hypot(dx, dy)

        # ── Step 2: lateral error = distance to nearest waypoint ──────
        nearest_idx = int(np.argmin(dists))
        lat_err = float(dists[nearest_idx])

        # ── Step 3: adaptive L1 distance ─────────────────────────────
        #   raw:   L = k · v
        #   floor: max(L_min,  √2 · lat_err)   ← grows when car is off-path
        #   ceil:  L_max
        L_raw = self.lookahead_k * ego_v
        L_lo  = max(self.lookahead_min, math.sqrt(2.0) * lat_err)
        L     = min(max(L_raw, L_lo), self._lookahead_cap)

        # ── Step 4: select lookahead waypoint by walking forward from nearest ──
        # Walk forward along the track index order (handles circular tracks correctly).
        # This prevents jumping to geometrically close but track-order-wrong waypoints.
        n = len(self.waypoints)
        tgt_idx = nearest_idx
        for i in range(1, n):
            idx = (nearest_idx + i) % n
            d = math.hypot(wp_xy[idx, 0] - pos.x, wp_xy[idx, 1] - pos.y)
            if d >= L:
                tgt_idx = idx
                break

        self._publish_lookahead(self.waypoints[tgt_idx])

        # ── Step 5: steering (L1 / Pure Pursuit) ─────────────────────
        gx = self.waypoints[tgt_idx].x_m
        gy = self.waypoints[tgt_idx].y_m
        # transform target to vehicle frame
        lx = math.cos(-yaw) * (gx - pos.x) - math.sin(-yaw) * (gy - pos.y)
        ly = math.sin(-yaw) * (gx - pos.x) + math.cos(-yaw) * (gy - pos.y)
        L_sq = lx * lx + ly * ly
        if L_sq < 1e-6:
            return 0.0, 0.0

        # δ = atan(wheelbase · 2·ly / L²)
        steer = math.atan(self.wheelbase * 2.0 * ly / L_sq)

        # downscale at high speed
        steer = self._steer_speed_scale(steer, ego_v)
        # physical clamp
        steer = max(-self.max_steer, min(self.max_steer, steer))
        # rate limiter (prevents sudden jumps)
        steer = self._steer_rate_limit(steer)

        # ── Step 6: speed ─────────────────────────────────────────────
        speed = self._target_speed(nearest_idx, ego_v, lat_err)
        speed = speed / (1.0 + self.delta_speed_gain * abs(steer))

        return steer, speed

    # ──────────────────────────────────────────────────────────────────
    # Speed helpers
    # ──────────────────────────────────────────────────────────────────
    def _target_speed(self, nearest_idx, ego_v, lat_err):
        """Read speed from a waypoint ahead of nearest_idx by speed_lookahead seconds."""
        n = len(self.waypoints)
        lookahead_dist = ego_v * self.speed_la
        acc = 0.0
        idx = nearest_idx
        for i in range(1, n):
            next_idx = (nearest_idx + i) % n
            prev_idx = (nearest_idx + i - 1) % n
            acc += math.hypot(
                self.waypoints[next_idx].x_m - self.waypoints[prev_idx].x_m,
                self.waypoints[next_idx].y_m - self.waypoints[prev_idx].y_m,
            )
            idx = next_idx
            if acc >= lookahead_dist:
                break

        speed = float(self.waypoints[idx].vx_mps)
        if speed <= 0.0:
            speed = 1.5

        return self._speed_adjust_lat_err(speed, lat_err)

    def _speed_adjust_lat_err(self, speed, lat_err):
        factor = 1.0 / (1.0 + self.lat_speed_gain * lat_err)
        return max(speed * factor, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Steering helpers
    # ──────────────────────────────────────────────────────────────────
    def _steer_speed_scale(self, steer, speed):
        """Linearly reduce steer gain from steer_spd_start to steer_spd_end."""
        spd_range = max(0.1, self.steer_spd_end - self.steer_spd_start)
        t      = np.clip((speed - self.steer_spd_start) / spd_range, 0.0, 1.0)
        factor = 1.0 - t * self.steer_downscale
        return steer * factor

    def _steer_rate_limit(self, steer):
        """Bound steering change to ±steer_rate_limit per control step."""
        delta = steer - self._prev_steer
        if abs(delta) > self.steer_rate_lim:
            steer = self._prev_steer + math.copysign(self.steer_rate_lim, delta)
        self._prev_steer = steer
        return steer

    # ──────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────
    def _publish_lookahead(self, wp):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id        = 'pp_lookahead', 0
        m.type, m.action  = Marker.SPHERE, Marker.ADD
        m.pose.position.x = float(wp.x_m)
        m.pose.position.y = float(wp.y_m)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        self.lookahead_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = PPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
