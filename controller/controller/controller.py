"""
controller.py  –  ROS-free control logic for F1TENTH racing.

This module contains pure Python classes for lateral and longitudinal
control. No ROS imports — all inputs/outputs are plain Python types
so the logic can be unit-tested without a ROS environment.

Coordinate convention:
  x  – forward (vehicle body frame)
  y  – left
  yaw – counter-clockwise from x-axis, in radians

Typical usage (called from controller_ros.py):
    ctrl = PurePursuitController(lookahead=1.0, wheelbase=0.33)
    steer, speed = ctrl.compute(state, waypoints)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers (shared between controller.py and controller_ros.py)
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Current ego-vehicle state in the global (map) frame."""
    x: float = 0.0          # [m]
    y: float = 0.0          # [m]
    yaw: float = 0.0        # [rad]
    v: float = 0.0          # longitudinal speed [m/s]


@dataclass
class Waypoint:
    """A single reference waypoint."""
    x: float = 0.0          # [m]  (map frame)
    y: float = 0.0          # [m]
    vx: float = 0.0         # desired speed [m/s]
    s: float = 0.0          # arc-length along the path [m]


@dataclass
class ControlCommand:
    """Output of any controller: steering angle and target speed."""
    steering_angle: float = 0.0   # [rad], positive = left
    speed: float = 0.0            # [m/s]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Pure Pursuit lateral controller
# ---------------------------------------------------------------------------

class PurePursuitController:
    """
    Classic Pure Pursuit algorithm for path tracking.

    Reference: Coulter, R. C. (1992). "Implementation of the Pure Pursuit
    Path Tracking Algorithm." Carnegie Mellon University, CMU-RI-TR-92-01.

    Args:
        lookahead_distance: Fixed look-ahead distance [m].
            Set to None to use adaptive look-ahead (speed-based).
        wheelbase: Distance between front and rear axles [m].
        min_lookahead: Minimum look-ahead when adaptive mode is active [m].
        max_lookahead: Maximum look-ahead when adaptive mode is active [m].
        lookahead_gain: Gain on speed for adaptive look-ahead.
        max_steering_angle: Saturation limit [rad].
    """

    def __init__(
        self,
        lookahead_distance: Optional[float] = 1.0,
        wheelbase: float = 0.33,
        min_lookahead: float = 0.5,
        max_lookahead: float = 3.0,
        lookahead_gain: float = 0.5,
        max_steering_angle: float = 0.4,
    ) -> None:
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_gain = lookahead_gain
        self.max_steering_angle = max_steering_angle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        state: VehicleState,
        waypoints: List[Waypoint],
    ) -> ControlCommand:
        """
        Compute steering angle and pass-through target speed.

        Args:
            state:     Current vehicle state (global frame).
            waypoints: Ordered list of global-frame waypoints.

        Returns:
            ControlCommand with steering_angle [rad] and speed [m/s].
        """
        if not waypoints:
            return ControlCommand()

        ld = self._get_lookahead(state.v)
        target = self._find_target_point(state, waypoints, ld)

        if target is None:
            # Fall back to the last waypoint
            target = waypoints[-1]

        steering = self._lateral_control(state, target, ld)
        speed = target.vx

        return ControlCommand(
            steering_angle=float(
                max(-self.max_steering_angle,
                    min(self.max_steering_angle, steering))
            ),
            speed=speed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lookahead(self, speed: float) -> float:
        if self.lookahead_distance is not None:
            return self.lookahead_distance
        # Adaptive: ld = gain * v  (clamped)
        return float(
            max(self.min_lookahead,
                min(self.max_lookahead, self.lookahead_gain * abs(speed)))
        )

    def _find_target_point(
        self,
        state: VehicleState,
        waypoints: List[Waypoint],
        ld: float,
    ) -> Optional[Waypoint]:
        """
        Find the first waypoint that is at least `ld` metres ahead of the
        vehicle in the forward direction.
        """
        # TODO: replace the linear scan with an efficient nearest-neighbour
        #       search (e.g. KD-tree) for long waypoint arrays.
        best: Optional[Waypoint] = None
        best_dist = float('inf')

        cos_yaw = math.cos(state.yaw)
        sin_yaw = math.sin(state.yaw)

        for wp in waypoints:
            dx = wp.x - state.x
            dy = wp.y - state.y
            dist = math.sqrt(dx * dx + dy * dy)

            # Only consider points ahead of the vehicle
            forward_proj = cos_yaw * dx + sin_yaw * dy
            if forward_proj < 0.0:
                continue

            if dist >= ld and dist < best_dist:
                best_dist = dist
                best = wp

        return best

    def _lateral_control(
        self,
        state: VehicleState,
        target: Waypoint,
        ld: float,
    ) -> float:
        """
        Compute the Pure Pursuit steering angle.

            delta = atan(2 * L * sin(alpha) / ld)

        where alpha is the angle from the vehicle heading to the target.
        """
        dx = target.x - state.x
        dy = target.y - state.y

        # Transform target point into the vehicle body frame
        alpha = math.atan2(dy, dx) - state.yaw
        alpha = normalize_angle(alpha)

        # Pure pursuit formula
        steering = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        return steering


# ---------------------------------------------------------------------------
# Stanley lateral controller  (alternative)
# ---------------------------------------------------------------------------

class StanleyController:
    """
    Stanley lateral controller (front-axle method).

    Reference: Hoffmann et al. (2007). "Autonomous Automobile Trajectory
    Tracking for Off-Road Driving." American Control Conference.

    Args:
        k: Cross-track error gain.
        k_soft: Softening constant to avoid division by zero at low speed.
        max_steering_angle: Saturation limit [rad].
        wheelbase: Distance between axles [m].
    """

    def __init__(
        self,
        k: float = 0.5,
        k_soft: float = 1.0,
        max_steering_angle: float = 0.4,
        wheelbase: float = 0.33,
    ) -> None:
        self.k = k
        self.k_soft = k_soft
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase

    def compute(
        self,
        state: VehicleState,
        waypoints: List[Waypoint],
    ) -> ControlCommand:
        """
        Compute Stanley steering angle.

        Args:
            state:     Current vehicle state.
            waypoints: Reference path (global frame).

        Returns:
            ControlCommand with steering and speed.
        """
        if not waypoints:
            return ControlCommand()

        nearest_idx = self._nearest_waypoint(state, waypoints)
        nearest = waypoints[nearest_idx]

        # Heading error
        if nearest_idx + 1 < len(waypoints):
            next_wp = waypoints[nearest_idx + 1]
        else:
            next_wp = waypoints[nearest_idx]

        path_heading = math.atan2(
            next_wp.y - nearest.y,
            next_wp.x - nearest.x,
        )
        heading_error = normalize_angle(path_heading - state.yaw)

        # Cross-track error at the front axle
        front_x = state.x + self.wheelbase * math.cos(state.yaw)
        front_y = state.y + self.wheelbase * math.sin(state.yaw)

        dx = front_x - nearest.x
        dy = front_y - nearest.y

        # Sign: positive cte = vehicle is to the left of path
        cross_track_err = (
            math.cos(path_heading + math.pi / 2.0) * dx
            + math.sin(path_heading + math.pi / 2.0) * dy
        )

        stanley_term = math.atan2(
            self.k * cross_track_err,
            self.k_soft + abs(state.v),
        )

        steering = heading_error + stanley_term
        steering = float(
            max(-self.max_steering_angle,
                min(self.max_steering_angle, steering))
        )

        return ControlCommand(steering_angle=steering, speed=nearest.vx)

    # ------------------------------------------------------------------

    def _nearest_waypoint(
        self,
        state: VehicleState,
        waypoints: List[Waypoint],
    ) -> int:
        """Return the index of the waypoint closest to the vehicle."""
        min_dist = float('inf')
        idx = 0
        for i, wp in enumerate(waypoints):
            d = euclidean_distance(state.x, state.y, wp.x, wp.y)
            if d < min_dist:
                min_dist = d
                idx = i
        return idx


# ---------------------------------------------------------------------------
# Longitudinal PID speed controller
# ---------------------------------------------------------------------------

class LongitudinalPIDController:
    """
    Simple PID controller for longitudinal speed tracking.

    The output is an acceleration command [m/s²] that can be mapped to
    throttle/brake by the ROS wrapper.

    Args:
        kp, ki, kd: PID gains.
        max_accel:  Acceleration saturation [m/s²].
        max_decel:  Deceleration saturation [m/s²] (positive value).
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.1,
        max_accel: float = 5.0,
        max_decel: float = 5.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_accel = max_accel
        self.max_decel = max_decel

        self._integral: float = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        """Reset integrator and derivative state."""
        self._integral = 0.0
        self._prev_error = None

    def compute(self, v_ref: float, v_meas: float, dt: float) -> float:
        """
        Compute acceleration command.

        Args:
            v_ref:  Desired speed [m/s].
            v_meas: Measured speed [m/s].
            dt:     Time step [s].

        Returns:
            Acceleration command [m/s²] (positive = accelerate).
        """
        if dt <= 0.0:
            return 0.0

        error = v_ref - v_meas
        self._integral += error * dt

        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        accel = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(-self.max_decel, min(self.max_accel, accel)))
