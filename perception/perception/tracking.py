"""
tracking.py  –  ROS-free obstacle tracking for F1TENTH.

This module maintains a set of tracked obstacles over time by associating
new detections with existing tracks.  No ROS imports.

Pipeline (per frame)
--------------------
1. Receive a list of DetectedObstacle from the detector.
2. Associate detections with existing tracks (nearest-neighbour).
3. Update matched tracks with new measurements (Kalman filter).
4. Mark unmatched tracks as "coasted" (predicted-only).
5. Delete tracks that have been missing too long.
6. Create new tracks for unmatched detections.
7. Return a list of TrackedObstacle with stable IDs, velocities, etc.

Typical usage (called from tracking_ros.py):
    tracker = ObstacleTracker(max_coasting=5, association_threshold=1.0)
    tracked = tracker.update(detections, dt)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from perception.detect import DetectedObstacle


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrackedObstacle:
    """
    A tracked obstacle with a stable ID, estimated state, and history.
    """
    track_id: int = -1           # persistent ID across frames

    # Estimated position (map frame or sensor frame, depending on caller)
    cx: float = 0.0              # [m]
    cy: float = 0.0              # [m]

    # Estimated velocity
    vx: float = 0.0              # [m/s]
    vy: float = 0.0              # [m/s]

    # Bounding-box size (inherited from last associated detection)
    width: float = 0.1           # [m]
    height: float = 0.1          # [m]

    # Track confidence
    hits: int = 1                # number of times matched
    coasted: int = 0             # number of frames without a match

    # Latest raw detection (may be None if coasting)
    latest_detection: Optional[DetectedObstacle] = None


# ---------------------------------------------------------------------------
# Internal track object with Kalman state
# ---------------------------------------------------------------------------

class _Track:
    """
    Internal representation of a single Kalman-filtered track.

    State vector: [cx, cy, vx, vy]
    Measurement:  [cx, cy]
    """

    # Simple constant-velocity model matrices (set at class level)
    _next_id: int = 0

    def __init__(self, det: DetectedObstacle) -> None:
        self.track_id: int = _Track._next_id
        _Track._next_id += 1

        # --- Kalman state: [cx, cy, vx, vy] ---
        self._x = np.array([det.cx, det.cy, 0.0, 0.0], dtype=np.float64)

        # Covariance
        self._P = np.diag([1.0, 1.0, 10.0, 10.0])

        # Process noise
        self._Q = np.diag([0.1, 0.1, 1.0, 1.0])

        # Measurement noise
        self._R = np.diag([0.5, 0.5])

        # Measurement matrix H: maps state → measurement
        self._H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)

        self.hits: int = 1
        self.coasted: int = 0
        self.latest_det: DetectedObstacle = det

    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Kalman predict step with constant-velocity motion model."""
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=np.float64)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q

    def update(self, det: DetectedObstacle) -> None:
        """Kalman update step with a new detection."""
        z = np.array([det.cx, det.cy], dtype=np.float64)
        y = z - self._H @ self._x                   # innovation
        S = self._H @ self._P @ self._H.T + self._R  # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = self._x + K @ y
        I = np.eye(len(self._x))
        self._P = (I - K @ self._H) @ self._P

        self.hits += 1
        self.coasted = 0
        self.latest_det = det

    # ------------------------------------------------------------------

    @property
    def cx(self) -> float:
        return float(self._x[0])

    @property
    def cy(self) -> float:
        return float(self._x[1])

    @property
    def vx(self) -> float:
        return float(self._x[2])

    @property
    def vy(self) -> float:
        return float(self._x[3])

    def to_tracked_obstacle(self) -> TrackedObstacle:
        return TrackedObstacle(
            track_id=self.track_id,
            cx=self.cx,
            cy=self.cy,
            vx=self.vx,
            vy=self.vy,
            width=self.latest_det.width,
            height=self.latest_det.height,
            hits=self.hits,
            coasted=self.coasted,
            latest_detection=self.latest_det,
        )


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class ObstacleTracker:
    """
    Nearest-neighbour + Kalman filter multi-object tracker.

    Args:
        max_coasting:         Delete a track after this many consecutive
                              frames without an associated detection.
        association_threshold: Maximum distance [m] to associate a detection
                              with an existing track.
        min_hits:             A track is only reported in the output after
                              it has been matched at least this many times.
                              (Prevents spurious one-frame detections.)
    """

    def __init__(
        self,
        max_coasting: int = 5,
        association_threshold: float = 1.0,
        min_hits: int = 1,
    ) -> None:
        self.max_coasting = max_coasting
        self.assoc_th = association_threshold
        self.min_hits = min_hits

        self._tracks: Dict[int, _Track] = {}   # track_id → _Track

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[DetectedObstacle],
        dt: float = 0.05,
    ) -> List[TrackedObstacle]:
        """
        Update all tracks with a new set of detections.

        Args:
            detections: Output of the detector for the current frame.
            dt:         Time elapsed since the last call [s].

        Returns:
            List of TrackedObstacle with stable IDs and velocity estimates.
        """
        # 1. Predict all existing tracks
        for track in self._tracks.values():
            track.predict(dt)

        # 2. Associate detections ↔ tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # 3. Update matched tracks
        for track_id, det_idx in matched:
            self._tracks[track_id].update(detections[det_idx])

        # 4. Coast unmatched tracks
        for track_id in unmatched_tracks:
            self._tracks[track_id].coasted += 1

        # 5. Delete stale tracks
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.coasted <= self.max_coasting
        }

        # 6. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = _Track(detections[det_idx])
            self._tracks[new_track.track_id] = new_track

        # 7. Return confirmed tracks
        return [
            t.to_tracked_obstacle()
            for t in self._tracks.values()
            if t.hits >= self.min_hits
        ]

    def reset(self) -> None:
        """Clear all tracks (e.g. on localization reset)."""
        self._tracks.clear()

    # ------------------------------------------------------------------
    # Nearest-neighbour association
    # ------------------------------------------------------------------

    def _associate(
        self,
        detections: List[DetectedObstacle],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy nearest-neighbour assignment.

        Returns:
            matched:          [(track_id, det_idx), ...]
            unmatched_dets:   [det_idx, ...]  – new detections
            unmatched_tracks: [track_id, ...] – tracks without a detection
        """
        track_ids = list(self._tracks.keys())

        if not track_ids or not detections:
            return [], list(range(len(detections))), track_ids

        # Build cost matrix: rows = tracks, cols = detections
        cost = np.full((len(track_ids), len(detections)), fill_value=np.inf)
        for r, tid in enumerate(track_ids):
            t = self._tracks[tid]
            for c, det in enumerate(detections):
                d = math.sqrt((t.cx - det.cx) ** 2 + (t.cy - det.cy) ** 2)
                cost[r, c] = d

        matched: List[Tuple[int, int]] = []
        used_rows: set = set()
        used_cols: set = set()

        # Greedy: repeatedly pick the minimum cost pair
        while True:
            if cost.size == 0:
                break
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            r, c = int(idx[0]), int(idx[1])
            if cost[r, c] > self.assoc_th:
                break
            if r not in used_rows and c not in used_cols:
                matched.append((track_ids[r], c))
                used_rows.add(r)
                used_cols.add(c)
            cost[r, :] = np.inf
            cost[:, c] = np.inf

        unmatched_tracks = [track_ids[r] for r in range(len(track_ids))
                            if r not in used_rows]
        unmatched_dets = [c for c in range(len(detections))
                          if c not in used_cols]

        return matched, unmatched_dets, unmatched_tracks
