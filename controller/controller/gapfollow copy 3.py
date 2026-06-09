import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from controller.estop import EStop

PARAMS = {
    'control_rate_hz':  50.0,
    'gf_bubble_radius':  0.3,
    'gf_bubble_speed_gain': 0.10,
    'gf_speed':          2.0,
    'gf_max_steer':      0.4,
    'gf_max_range':     10.0,
    'gf_obstacle_dilation_radius': 0.20,
    'gf_min_gap_distance': 1.2,
    'gf_target_hysteresis_gain': 0.35,
    'gf_direction_hold_enable': True,
    'gf_direction_hold_steps': 8,
    'gf_debug_viz':     True,
}

FTG_TUNING = {
    'front_half_fov_deg':       90.0,
    'best_point_window':         5,
    'range_epsilon':             1e-3,
    'front_clearance_deg':      15.0,
    'steer_speed_reduction':     0.3,
    'min_steer_speed_scale':     0.35,
    'min_clearance_speed_scale': 0.30,
    'clearance_time_sec':        1.5,
    'min_clearance_m':           1.0,
    'min_gap_size_beams':        5,
    'discontinuity_threshold_m': 0.50,
    'center_point_weight':       0.70,
    'farthest_point_weight':     0.30,
    'direction_hold_trigger_deg': 8.0,
    'direction_hold_neutral_deg': 3.0,
    'direction_hold_release_clearance_m': 1.8,
}

DEBUG_VIZ = {
    'marker_topic':           '/gap_follow/debug_markers',
    'processed_scan_topic':   '/scan_proc/markers',
    'all_gaps_topic':         '/all_gaps/markers',
    'best_gap_topic':         '/best_gap/markers',
    'best_point_topic':       '/best_points/marker',
    'free_space_stride':       4,
    'gap_stride':              2,
    'point_scale_m':           0.05,
    'highlight_scale_m':       0.12,
    'line_scale_m':            0.03,
    'text_scale_m':            0.20,
}


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop         = EStop(self)
        self.bubble_radius = p('gf_bubble_radius')
        self.bubble_speed_gain = p('gf_bubble_speed_gain')
        self.speed         = p('gf_speed')
        self.max_steer     = p('gf_max_steer')
        self.max_range     = p('gf_max_range')
        self.obstacle_dilation_radius = p('gf_obstacle_dilation_radius')
        self.min_gap_distance = p('gf_min_gap_distance')
        self.target_hysteresis_gain = p('gf_target_hysteresis_gain')
        self.direction_hold_enable = bool(p('gf_direction_hold_enable'))
        self.direction_hold_steps = int(p('gf_direction_hold_steps'))
        self.debug_viz     = bool(p('gf_debug_viz'))

        self.front_half_fov = math.radians(FTG_TUNING['front_half_fov_deg'])
        self.best_point_window = FTG_TUNING['best_point_window']
        self.range_epsilon = FTG_TUNING['range_epsilon']
        self.front_clearance_angle = math.radians(FTG_TUNING['front_clearance_deg'])
        self.steer_speed_reduction = FTG_TUNING['steer_speed_reduction']
        self.min_steer_speed_scale = FTG_TUNING['min_steer_speed_scale']
        self.min_clearance_speed_scale = FTG_TUNING['min_clearance_speed_scale']
        self.clearance_time_sec = FTG_TUNING['clearance_time_sec']
        self.min_clearance_m = FTG_TUNING['min_clearance_m']
        self.min_gap_size_beams = FTG_TUNING['min_gap_size_beams']
        self.discontinuity_threshold = FTG_TUNING['discontinuity_threshold_m']
        self.center_point_weight = FTG_TUNING['center_point_weight']
        self.farthest_point_weight = FTG_TUNING['farthest_point_weight']
        self.direction_hold_trigger_angle = math.radians(FTG_TUNING['direction_hold_trigger_deg'])
        self.direction_hold_neutral_angle = math.radians(FTG_TUNING['direction_hold_neutral_deg'])
        self.direction_hold_release_clearance = FTG_TUNING['direction_hold_release_clearance_m']

        self.debug_free_space_stride = DEBUG_VIZ['free_space_stride']
        self.debug_gap_stride = DEBUG_VIZ['gap_stride']
        self.debug_point_scale = DEBUG_VIZ['point_scale_m']
        self.debug_highlight_scale = DEBUG_VIZ['highlight_scale_m']
        self.debug_line_scale = DEBUG_VIZ['line_scale_m']
        self.debug_text_scale = DEBUG_VIZ['text_scale_m']

        self.scan = None
        self.odom = None
        self.prev_target_idx = None
        self.prev_steer = 0.0
        self.direction_hold_sign = 0
        self.direction_hold_count = 0

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.debug_pub = self.create_publisher(MarkerArray, DEBUG_VIZ['marker_topic'], 10)
        self.proc_scan_pub = self.create_publisher(MarkerArray, DEBUG_VIZ['processed_scan_topic'], 10)
        self.all_gaps_pub = self.create_publisher(MarkerArray, DEBUG_VIZ['all_gaps_topic'], 10)
        self.best_gap_pub = self.create_publisher(MarkerArray, DEBUG_VIZ['best_gap_topic'], 10)
        self.best_point_pub = self.create_publisher(Marker, DEBUG_VIZ['best_point_topic'], 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _debug_header(self):
        stamp = self.get_clock().now().to_msg()
        frame_id = 'base_link'

        if self.scan is not None:
            stamp = self.scan.header.stamp
            if self.scan.header.frame_id:
                frame_id = self.scan.header.frame_id

        return stamp, frame_id

    def _make_point(self, x, y, z=0.0):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def _polar_to_point(self, distance, angle, z=0.0):
        return self._make_point(distance * math.cos(angle), distance * math.sin(angle), z)

    def _make_marker(self, marker_id, marker_type, stamp, frame_id, ns='gap_follow_debug'):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _delete_all_marker(self, stamp, frame_id):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.action = Marker.DELETEALL
        return marker

    def _publish_marker_array(self, publisher, markers):
        stamp, frame_id = self._debug_header()
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_marker(stamp, frame_id))
        marker_array.markers.extend(markers)
        publisher.publish(marker_array)

    def _clear_debug_markers(self):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()
        self._publish_marker_array(self.debug_pub, [])
        self._publish_marker_array(self.proc_scan_pub, [])
        self._publish_marker_array(self.all_gaps_pub, [])
        self._publish_marker_array(self.best_gap_pub, [])
        self.best_point_pub.publish(self._delete_all_marker(stamp, frame_id))

    def _reset_guidance_state(self):
        self.prev_target_idx = None
        self.prev_steer = 0.0
        self.direction_hold_sign = 0
        self.direction_hold_count = 0

    def _indices_to_points(self, ranges, angles, indices):
        points = []
        for idx in indices:
            if idx < 0 or idx >= len(ranges):
                continue
            distance = float(ranges[idx])
            if distance <= self.range_epsilon:
                continue
            angle = float(angles[idx])
            points.append(self._polar_to_point(distance, angle))
        return points

    def _publish_processed_scan_markers(self, free_space, angles):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()
        markers = []

        if free_space is not None and angles is not None:
            free_indices = np.flatnonzero(free_space > self.range_epsilon)[::self.debug_free_space_stride]
            free_points = self._indices_to_points(free_space, angles, free_indices)
            if free_points:
                marker = self._make_marker(0, Marker.POINTS, stamp, frame_id, ns='scan_proc')
                marker.scale.x = self.debug_point_scale
                marker.scale.y = self.debug_point_scale
                marker.color.r = 1.00
                marker.color.g = 1.00
                marker.color.b = 1.00
                marker.color.a = 0.90
                marker.points = free_points
                markers.append(marker)

        self._publish_marker_array(self.proc_scan_pub, markers)

    def _publish_best_gap_markers(self, free_space, angles, gap_start, gap_end, target_idx):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()
        markers = []

        if free_space is not None and angles is not None and gap_start is not None and gap_end is not None:
            gap_indices = np.arange(gap_start, gap_end + 1, self.debug_gap_stride, dtype=int)
            gap_points = self._indices_to_points(free_space, angles, gap_indices)
            if gap_points:
                marker = self._make_marker(0, Marker.POINTS, stamp, frame_id, ns='best_gap')
                marker.scale.x = self.debug_point_scale * 1.3
                marker.scale.y = self.debug_point_scale * 1.3
                marker.color.r = 1.00
                marker.color.g = 1.00
                marker.color.b = 0.00
                marker.color.a = 0.95
                marker.points = gap_points
                markers.append(marker)

        if (
            free_space is not None and
            angles is not None and
            target_idx is not None and
            0 <= target_idx < len(free_space) and
            free_space[target_idx] > self.range_epsilon
        ):
            target_point = self._polar_to_point(float(free_space[target_idx]), float(angles[target_idx]))

            line = self._make_marker(1, Marker.LINE_STRIP, stamp, frame_id, ns='best_gap')
            line.scale.x = self.debug_line_scale
            line.color.r = 0.10
            line.color.g = 0.80
            line.color.b = 1.00
            line.color.a = 0.95
            line.points = [self._make_point(0.0, 0.0, 0.0), target_point]
            markers.append(line)

            marker = self._make_marker(2, Marker.SPHERE, stamp, frame_id, ns='best_gap')
            marker.scale.x = self.debug_highlight_scale
            marker.scale.y = self.debug_highlight_scale
            marker.scale.z = self.debug_highlight_scale
            marker.color.r = 0.10
            marker.color.g = 0.45
            marker.color.b = 1.00
            marker.color.a = 0.95
            marker.pose.position = target_point
            markers.append(marker)

        self._publish_marker_array(self.best_gap_pub, markers)

    def _publish_all_gap_markers(self, free_space, angles, candidate_gaps):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()
        markers = []

        if free_space is not None and angles is not None:
            for marker_id, (gap_start, gap_end) in enumerate(candidate_gaps):
                gap_indices = np.arange(gap_start, gap_end + 1, self.debug_gap_stride, dtype=int)
                gap_points = self._indices_to_points(free_space, angles, gap_indices)
                if not gap_points:
                    continue

                marker = self._make_marker(marker_id, Marker.POINTS, stamp, frame_id, ns='all_gaps')
                marker.scale.x = self.debug_point_scale * 1.15
                marker.scale.y = self.debug_point_scale * 1.15
                marker.color.r = 1.00
                marker.color.g = 1.00
                marker.color.b = 1.00
                marker.color.a = 0.80
                marker.points = gap_points
                markers.append(marker)

        self._publish_marker_array(self.all_gaps_pub, markers)

    def _publish_best_point_marker(self, free_space, angles, target_idx):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()

        if (
            free_space is None or
            angles is None or
            target_idx is None or
            target_idx < 0 or
            target_idx >= len(free_space) or
            free_space[target_idx] <= self.range_epsilon
        ):
            self.best_point_pub.publish(self._delete_all_marker(stamp, frame_id))
            return

        marker = self._make_marker(0, Marker.SPHERE, stamp, frame_id, ns='best_point')
        marker.scale.x = self.debug_highlight_scale
        marker.scale.y = self.debug_highlight_scale
        marker.scale.z = self.debug_highlight_scale
        marker.color.r = 0.00
        marker.color.g = 1.00
        marker.color.b = 1.00
        marker.color.a = 0.95
        marker.pose.position = self._polar_to_point(float(free_space[target_idx]), float(angles[target_idx]))
        self.best_point_pub.publish(marker)

    def _publish_reference_debug_topics(self, free_space, angles, candidate_gaps, gap_start, gap_end, target_idx):
        if not self.debug_viz:
            return

        self._publish_processed_scan_markers(free_space, angles)
        self._publish_all_gap_markers(free_space, angles, candidate_gaps)
        self._publish_best_gap_markers(free_space, angles, gap_start, gap_end, target_idx)
        self._publish_best_point_marker(free_space, angles, target_idx)

    def _publish_debug_markers(self, ranges, free_space, angles, candidate_gaps, closest_idx, target_idx, gap_start, gap_end, bubble_start, bubble_end, steer, speed):
        if not self.debug_viz:
            return

        stamp, frame_id = self._debug_header()
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.stamp = stamp
        delete_marker.header.frame_id = frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        free_indices = np.flatnonzero(free_space > self.range_epsilon)[::self.debug_free_space_stride]
        free_points = self._indices_to_points(free_space, angles, free_indices)
        if free_points:
            marker = self._make_marker(0, Marker.POINTS, stamp, frame_id)
            marker.scale.x = self.debug_point_scale
            marker.scale.y = self.debug_point_scale
            marker.color.r = 0.10
            marker.color.g = 0.90
            marker.color.b = 0.20
            marker.color.a = 0.85
            marker.points = free_points
            marker_array.markers.append(marker)

        for gap_marker_id, (candidate_start, candidate_end) in enumerate(candidate_gaps, start=10):
            gap_indices = np.arange(candidate_start, candidate_end + 1, self.debug_gap_stride, dtype=int)
            gap_points = self._indices_to_points(free_space, angles, gap_indices)
            if not gap_points:
                continue

            marker = self._make_marker(gap_marker_id, Marker.POINTS, stamp, frame_id, ns='gap_follow_candidates')
            marker.scale.x = self.debug_point_scale * 1.1
            marker.scale.y = self.debug_point_scale * 1.1
            marker.color.r = 1.00
            marker.color.g = 1.00
            marker.color.b = 1.00
            marker.color.a = 0.55
            marker.points = gap_points
            marker_array.markers.append(marker)

        if bubble_start is not None and bubble_end is not None:
            bubble_indices = np.arange(bubble_start, bubble_end + 1, self.debug_gap_stride, dtype=int)
            bubble_points = self._indices_to_points(ranges, angles, bubble_indices)
            if bubble_points:
                marker = self._make_marker(1, Marker.POINTS, stamp, frame_id)
                marker.scale.x = self.debug_point_scale
                marker.scale.y = self.debug_point_scale
                marker.color.r = 1.00
                marker.color.g = 0.00
                marker.color.b = 0.60
                marker.color.a = 0.85
                marker.points = bubble_points
                marker_array.markers.append(marker)

        if gap_start is not None and gap_end is not None:
            gap_indices = np.arange(gap_start, gap_end + 1, self.debug_gap_stride, dtype=int)
            gap_points = self._indices_to_points(free_space, angles, gap_indices)
            if gap_points:
                marker = self._make_marker(2, Marker.POINTS, stamp, frame_id)
                marker.scale.x = self.debug_point_scale * 1.3
                marker.scale.y = self.debug_point_scale * 1.3
                marker.color.r = 1.00
                marker.color.g = 0.85
                marker.color.b = 0.10
                marker.color.a = 0.95
                marker.points = gap_points
                marker_array.markers.append(marker)

        if closest_idx is not None and ranges[closest_idx] > self.range_epsilon:
            marker = self._make_marker(3, Marker.SPHERE, stamp, frame_id)
            marker.scale.x = self.debug_highlight_scale
            marker.scale.y = self.debug_highlight_scale
            marker.scale.z = self.debug_highlight_scale
            marker.color.r = 1.00
            marker.color.g = 0.15
            marker.color.b = 0.15
            marker.color.a = 0.95
            marker.pose.position = self._polar_to_point(float(ranges[closest_idx]), float(angles[closest_idx]))
            marker_array.markers.append(marker)

        if target_idx is not None and free_space[target_idx] > self.range_epsilon:
            target_point = self._polar_to_point(float(free_space[target_idx]), float(angles[target_idx]))

            marker = self._make_marker(4, Marker.SPHERE, stamp, frame_id)
            marker.scale.x = self.debug_highlight_scale
            marker.scale.y = self.debug_highlight_scale
            marker.scale.z = self.debug_highlight_scale
            marker.color.r = 0.10
            marker.color.g = 0.45
            marker.color.b = 1.00
            marker.color.a = 0.95
            marker.pose.position = target_point
            marker_array.markers.append(marker)

            line = self._make_marker(5, Marker.LINE_STRIP, stamp, frame_id)
            line.scale.x = self.debug_line_scale
            line.color.r = 0.10
            line.color.g = 0.80
            line.color.b = 1.00
            line.color.a = 0.95
            line.points = [self._make_point(0.0, 0.0, 0.0), target_point]
            marker_array.markers.append(line)

        text = self._make_marker(6, Marker.TEXT_VIEW_FACING, stamp, frame_id)
        text.scale.z = self.debug_text_scale
        text.color.r = 1.00
        text.color.g = 1.00
        text.color.b = 1.00
        text.color.a = 0.95
        text.pose.position = self._make_point(0.0, 0.0, 0.35)
        hold_text = 'L' if self.direction_hold_sign > 0 else 'R' if self.direction_hold_sign < 0 else '-'
        text.text = f'steer={steer:.2f} rad | speed={speed:.2f} m/s | hold={hold_text}'
        marker_array.markers.append(text)

        self.debug_pub.publish(marker_array)

    def _preprocess_ranges(self):
        ranges = np.asarray(self.scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return None, None

        sensor_max = float(self.scan.range_max)
        if not math.isfinite(sensor_max) or sensor_max <= 0.0:
            sensor_max = self.max_range

        processed = np.nan_to_num(ranges, nan=sensor_max, posinf=sensor_max, neginf=0.0)
        processed = np.clip(processed, 0.0, min(sensor_max, self.max_range))

        min_valid_range = max(float(self.scan.range_min), self.range_epsilon)
        processed[processed < min_valid_range] = 0.0

        angles = self.scan.angle_min + np.arange(processed.size, dtype=np.float32) * self.scan.angle_increment
        processed[np.abs(angles) > self.front_half_fov] = 0.0
        return processed, angles

    def _compute_dynamic_bubble_radius(self, current_speed):
        return self.bubble_radius + self.bubble_speed_gain * max(current_speed, 0.0)

    def _apply_obstacle_dilation(self, free_space, ranges, angle_increment, dilation_radius):
        if dilation_radius <= self.range_epsilon:
            return free_space

        dilated = free_space.copy()
        diffs = np.abs(np.diff(ranges))
        jump_indices = np.flatnonzero(diffs > self.discontinuity_threshold)

        if jump_indices.size == 0:
            return dilated

        beam_angle = max(abs(angle_increment), self.range_epsilon)

        for idx in jump_indices:
            candidates = []
            left_range = float(ranges[idx])
            right_range = float(ranges[idx + 1])

            if left_range > self.range_epsilon:
                candidates.append((left_range, int(idx)))
            if right_range > self.range_epsilon:
                candidates.append((right_range, int(idx + 1)))
            if not candidates:
                continue

            obstacle_dist, obstacle_idx = min(candidates, key=lambda item: item[0])
            dilation_angle = math.atan2(dilation_radius, max(obstacle_dist, self.range_epsilon))
            dilation_beams = max(1, int(math.ceil(dilation_angle / beam_angle)))
            start = max(0, obstacle_idx - dilation_beams)
            end = min(len(dilated) - 1, obstacle_idx + dilation_beams)
            dilated[start:end + 1] = 0.0

        return dilated

    def _find_candidate_gaps(self, free_space):
        candidate_gaps = []
        gap_start = None

        for i, value in enumerate(free_space):
            if value > self.range_epsilon:
                if gap_start is None:
                    gap_start = i
            elif gap_start is not None:
                gap_end = i - 1
                gap_len = gap_end - gap_start + 1
                if gap_len >= self.min_gap_size_beams:
                    candidate_gaps.append((gap_start, gap_end))
                gap_start = None

        if gap_start is not None:
            gap_end = len(free_space) - 1
            gap_len = gap_end - gap_start + 1
            if gap_len >= self.min_gap_size_beams:
                candidate_gaps.append((gap_start, gap_end))

        return candidate_gaps

    def _find_max_gap(self, free_space, angles, candidate_gaps=None):
        if candidate_gaps is None:
            candidate_gaps = self._find_candidate_gaps(free_space)

        if not candidate_gaps:
            return None, None

        def gap_score(gap):
            start, end = gap
            gap_len = end - start + 1
            center_idx = (start + end) // 2
            center_angle = abs(float(angles[center_idx]))
            mean_range = float(np.mean(free_space[start:end + 1]))
            # Prefer longer gaps, then straighter centerlines, then more open gaps.
            return (gap_len, -center_angle, mean_range)

        best_start, best_end = max(candidate_gaps, key=gap_score)
        return best_start, best_end

    def _find_best_point(self, start_idx, end_idx, free_space, angles):
        gap = free_space[start_idx:end_idx + 1]
        if gap.size == 0:
            return start_idx

        window = min(self.best_point_window, gap.size)
        if window % 2 == 0:
            window -= 1

        if window >= 3:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            smoothed_gap = np.convolve(gap, kernel, mode='same')
        else:
            smoothed_gap = gap

        center_local = (gap.size - 1) // 2
        farthest_local = int(np.argmax(smoothed_gap))
        target_local = int(round(
            self.center_point_weight * center_local +
            self.farthest_point_weight * farthest_local
        ))

        if self.prev_target_idx is not None:
            alpha = float(np.clip(self.target_hysteresis_gain, 0.0, 0.95))
            prev_local = int(np.clip(self.prev_target_idx - start_idx, 0, gap.size - 1))
            target_local = int(round((1.0 - alpha) * target_local + alpha * prev_local))

        if self.direction_hold_enable and self.direction_hold_count > 0 and self.direction_hold_sign != 0:
            gap_angles = angles[start_idx:end_idx + 1]
            local_indices = np.arange(gap.size, dtype=int)
            same_side_mask = (
                (np.sign(gap_angles) == self.direction_hold_sign) |
                (np.abs(gap_angles) <= self.direction_hold_neutral_angle)
            )
            same_side_indices = local_indices[same_side_mask]
            if same_side_indices.size > 0:
                nearest_idx = int(np.argmin(np.abs(same_side_indices - target_local)))
                target_local = int(same_side_indices[nearest_idx])
                self.direction_hold_count = max(0, self.direction_hold_count - 1)
            else:
                self.direction_hold_sign = 0
                self.direction_hold_count = 0

        return start_idx + target_local

    def _update_direction_hold(self, target_angle, front_clearance):
        if not self.direction_hold_enable:
            self.direction_hold_sign = 0
            self.direction_hold_count = 0
            return

        if front_clearance >= self.direction_hold_release_clearance:
            self.direction_hold_sign = 0
            self.direction_hold_count = 0
            return

        if abs(target_angle) >= self.direction_hold_trigger_angle:
            target_sign = int(np.sign(target_angle))
            if target_sign != 0:
                self.direction_hold_sign = target_sign
                self.direction_hold_count = self.direction_hold_steps
                return

        if self.direction_hold_count > 0:
            self.direction_hold_count -= 1
            if self.direction_hold_count == 0:
                self.direction_hold_sign = 0

    def _loop(self):
        if self.scan is None or self.odom is None:
            self._reset_guidance_state()
            self._clear_debug_markers()
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        # Follow-the-Gap (FTG) using LiDAR free-space selection
        #
        # Goal:
        #   - Use LiDAR scan data to find a collision-free gap
        #   - Select a safe target direction inside the best gap
        #   - Convert the target direction into a steering command
        #
        # You should return (steering, speed) from this function.
        #
        # Useful information:
        #   - self.scan.ranges             : LiDAR distance array [m]
        #   - self.scan.angle_min          : angle of first beam [rad]
        #   - self.scan.angle_increment    : angular step between beams [rad]
        #   - self.bubble_radius           : safety bubble radius around closest obstacle [m]
        #   - self.max_range               : cap LiDAR ranges to suppress outliers [m]
        #   - self.max_steer               : steering clamp [rad]
        #   - self.speed                   : nominal driving speed [m/s]
        #   - self.odom                    : current vehicle motion (optional for speed adjustment)
        #
        # Suggested approach (FTG pipeline):
        #   - Preprocess ranges:
        #       * replace NaN/Inf/invalid values
        #       * clip ranges to [0, self.max_range]
        #       * optionally focus on a front field-of-view
        #   - Find the closest obstacle beam
        #   - Create a safety bubble around that obstacle (zero-out nearby beams)
        #   - Find the longest contiguous non-zero gap
        #   - Choose the best target beam in the gap
        #       * e.g., farthest beam or weighted by distance and heading
        #   - Convert target beam index to steering angle
        #   - Clamp steering to +/- self.max_steer
        #   - Optionally reduce speed when |steering| is large or obstacle is close
        #
        # Output:
        #   - steering [rad]
        #   - speed [m/s]
        ranges, angles = self._preprocess_ranges()
        if ranges is None:
            self._reset_guidance_state()
            self._clear_debug_markers()
            return 0.0, 0.0

        current_speed = abs(float(self.odom.twist.twist.linear.x)) if self.odom is not None else self.speed
        valid_mask = ranges > self.range_epsilon
        if not np.any(valid_mask):
            self._reset_guidance_state()
            self._publish_reference_debug_topics(
                free_space=np.zeros_like(ranges),
                angles=angles,
                candidate_gaps=[],
                gap_start=None,
                gap_end=None,
                target_idx=None,
            )
            self._publish_debug_markers(
                ranges=ranges,
                free_space=np.zeros_like(ranges),
                angles=angles,
                candidate_gaps=[],
                closest_idx=None,
                target_idx=None,
                gap_start=None,
                gap_end=None,
                bubble_start=None,
                bubble_end=None,
                steer=0.0,
                speed=0.0,
            )
            return 0.0, 0.0

        closest_idx = int(np.argmin(np.where(valid_mask, ranges, np.inf)))
        closest_dist = float(ranges[closest_idx])

        dynamic_bubble_radius = self._compute_dynamic_bubble_radius(current_speed)
        bubble_angle = math.atan2(dynamic_bubble_radius, max(closest_dist, self.range_epsilon))
        bubble_size = max(1, int(math.ceil(bubble_angle / max(abs(self.scan.angle_increment), self.range_epsilon))))

        free_space = ranges.copy()
        bubble_start = max(0, closest_idx - bubble_size)
        bubble_end = min(len(free_space) - 1, closest_idx + bubble_size)
        free_space[bubble_start:bubble_end + 1] = 0.0
        free_space = self._apply_obstacle_dilation(
            free_space=free_space,
            ranges=ranges,
            angle_increment=self.scan.angle_increment,
            dilation_radius=self.obstacle_dilation_radius,
        )

        gap_free_space = free_space.copy()
        gap_free_space[gap_free_space < self.min_gap_distance] = 0.0

        candidate_gaps = self._find_candidate_gaps(gap_free_space)
        gap_start, gap_end = self._find_max_gap(gap_free_space, angles, candidate_gaps)
        if gap_start is None or gap_end is None:
            self._publish_reference_debug_topics(
                free_space=gap_free_space,
                angles=angles,
                candidate_gaps=candidate_gaps,
                gap_start=None,
                gap_end=None,
                target_idx=None,
            )
            self._publish_debug_markers(
                ranges=ranges,
                free_space=gap_free_space,
                angles=angles,
                candidate_gaps=candidate_gaps,
                closest_idx=closest_idx,
                target_idx=None,
                gap_start=None,
                gap_end=None,
                bubble_start=bubble_start,
                bubble_end=bubble_end,
                steer=0.0,
                speed=0.0,
            )
            return 0.0, 0.0

        best_idx = self._find_best_point(gap_start, gap_end, gap_free_space, angles)
        target_angle = float(angles[best_idx])
        steer = float(np.clip(target_angle, -self.max_steer, self.max_steer))

        front_mask = (np.abs(angles) <= self.front_clearance_angle) & (ranges > self.range_epsilon)
        if np.any(front_mask):
            front_clearance = float(np.percentile(ranges[front_mask], 25))
        else:
            front_clearance = 0.0

        steer_ratio = abs(steer) / max(self.max_steer, self.range_epsilon)
        steer_scale = max(self.min_steer_speed_scale, 1.0 - self.steer_speed_reduction * steer_ratio)

        desired_clearance = max(self.min_clearance_m, max(self.speed, current_speed) * self.clearance_time_sec)
        clearance_scale = float(np.clip(front_clearance / desired_clearance, self.min_clearance_speed_scale, 1.0))

        speed = float(self.speed * min(steer_scale, clearance_scale))
        self.prev_target_idx = best_idx
        self.prev_steer = steer
        self._update_direction_hold(target_angle, front_clearance)
        self._publish_reference_debug_topics(
            free_space=gap_free_space,
            angles=angles,
            candidate_gaps=candidate_gaps,
            gap_start=gap_start,
            gap_end=gap_end,
            target_idx=best_idx,
        )
        self._publish_debug_markers(
            ranges=ranges,
            free_space=gap_free_space,
            angles=angles,
            candidate_gaps=candidate_gaps,
            closest_idx=closest_idx,
            target_idx=best_idx,
            gap_start=gap_start,
            gap_end=gap_end,
            bubble_start=bubble_start,
            bubble_end=bubble_end,
            steer=steer,
            speed=speed,
        )
        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
