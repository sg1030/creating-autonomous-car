"""
detect.py  –  ROS-free LiDAR obstacle detection for F1TENTH.

This module clusters raw 2-D LiDAR scan data into obstacle candidates.
No ROS imports — all inputs/outputs are plain Python / NumPy types so the
logic can be developed and unit-tested without a ROS environment.

Pipeline
--------
1. Convert polar scan (ranges, angles) → Cartesian point cloud.
2. Filter out invalid ranges and known static regions (optional).
3. Cluster adjacent points (simple jump-distance or DBSCAN).
4. Fit a bounding-box / centroid to each cluster.
5. Return a list of DetectedObstacle objects.

Typical usage (called from detect_ros.py):
    detector = LidarDetector(cluster_threshold=0.3, min_points=3)
    obstacles = detector.detect(ranges, angle_min, angle_increment)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    x: float = 0.0   # [m] in sensor frame
    y: float = 0.0   # [m] in sensor frame


@dataclass
class DetectedObstacle:
    """
    Axis-aligned bounding box of a detected obstacle cluster,
    expressed in the LiDAR sensor frame.
    """
    # Centroid
    cx: float = 0.0          # [m]
    cy: float = 0.0          # [m]

    # Bounding-box half-extents
    width: float = 0.0       # [m] (x-extent)
    height: float = 0.0      # [m] (y-extent)

    # Number of scan points belonging to this cluster
    num_points: int = 0

    # Raw cluster points (kept for downstream use)
    points: List[Point2D] = field(default_factory=list)

    # Unique id within a single scan frame (assigned by detector)
    id: int = -1


# ---------------------------------------------------------------------------
# LiDAR obstacle detector
# ---------------------------------------------------------------------------

class LidarDetector:
    """
    Simple jump-distance cluster-based 2-D LiDAR detector.

    A new cluster starts whenever the Euclidean distance between two
    consecutive valid scan points exceeds `cluster_threshold`.  Clusters
    with fewer than `min_points` are discarded.

    Args:
        cluster_threshold: Gap distance that splits clusters [m].
        min_points:        Minimum cluster size (points).
        max_range:         Ignore ranges above this distance [m].
        min_range:         Ignore ranges below this distance (dead zone) [m].
    """

    def __init__(
        self,
        cluster_threshold: float = 0.3,
        min_points: int = 3,
        max_range: float = 10.0,
        min_range: float = 0.05,
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.min_points = min_points
        self.max_range = max_range
        self.min_range = min_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[DetectedObstacle]:
        """
        Detect obstacles from a single 2-D LiDAR scan.

        Args:
            ranges:          Array of range measurements [m], shape (N,).
            angle_min:       Angle of the first ray [rad].
            angle_increment: Angular step between rays [rad].

        Returns:
            List of DetectedObstacle, one per cluster found.
        """
        points = self._scan_to_cartesian(ranges, angle_min, angle_increment)
        clusters = self._cluster(points)
        obstacles = [self._cluster_to_obstacle(cl, idx)
                     for idx, cl in enumerate(clusters)]
        return obstacles

    # ------------------------------------------------------------------
    # Step 1 – polar → Cartesian
    # ------------------------------------------------------------------

    def _scan_to_cartesian(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[Point2D]:
        """Filter invalid ranges and convert to (x, y) in sensor frame."""
        valid = []
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            if r < self.min_range or r > self.max_range:
                continue
            angle = angle_min + i * angle_increment
            valid.append(Point2D(
                x=r * math.cos(angle),
                y=r * math.sin(angle),
            ))
        return valid

    # ------------------------------------------------------------------
    # Step 2 – jump-distance clustering
    # ------------------------------------------------------------------

    def _cluster(self, points: List[Point2D]) -> List[List[Point2D]]:
        """
        Split the ordered point list into clusters by jump distance.

        Two consecutive points are in the same cluster if and only if
        their Euclidean distance is less than `cluster_threshold`.
        """
        if not points:
            return []

        clusters: List[List[Point2D]] = []
        current: List[Point2D] = [points[0]]

        for prev, curr in zip(points[:-1], points[1:]):
            dist = math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)
            if dist < self.cluster_threshold:
                current.append(curr)
            else:
                if len(current) >= self.min_points:
                    clusters.append(current)
                current = [curr]

        if len(current) >= self.min_points:
            clusters.append(current)

        return clusters

    # ------------------------------------------------------------------
    # Step 3 – cluster → bounding box
    # ------------------------------------------------------------------

    def _cluster_to_obstacle(
        self,
        cluster: List[Point2D],
        idx: int,
    ) -> DetectedObstacle:
        """Compute centroid and AABB from a list of 2-D points."""
        xs = [p.x for p in cluster]
        ys = [p.y for p in cluster]

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return DetectedObstacle(
            cx=cx,
            cy=cy,
            width=max(x_max - x_min, 0.05),    # at least 5 cm
            height=max(y_max - y_min, 0.05),
            num_points=len(cluster),
            points=cluster,
            id=idx,
        )


# ---------------------------------------------------------------------------
# Optional: DBSCAN-based detector (more robust, requires scikit-learn)
# ---------------------------------------------------------------------------

class DBSCANDetector:
    """
    Obstacle detector using DBSCAN clustering (scikit-learn).

    Useful when the scan is not ordered or when you need better handling
    of non-convex obstacle shapes.

    Args:
        eps:        DBSCAN neighbourhood radius [m].
        min_samples: Minimum number of points to form a core point.
        max_range:  Ignore ranges above this distance [m].
        min_range:  Ignore ranges below this distance [m].
    """

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 3,
        max_range: float = 10.0,
        min_range: float = 0.05,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.max_range = max_range
        self.min_range = min_range

    def detect(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
    ) -> List[DetectedObstacle]:
        """Same interface as LidarDetector.detect()."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError(
                'scikit-learn is required for DBSCANDetector. '
                'Install with: pip install scikit-learn'
            )

        # Convert to Cartesian
        points = []
        for i, r in enumerate(ranges):
            if not math.isfinite(r) or r < self.min_range or r > self.max_range:
                continue
            angle = angle_min + i * angle_increment
            points.append([r * math.cos(angle), r * math.sin(angle)])

        if not points:
            return []

        xy = np.array(points)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xy)

        obstacles = []
        for label in set(labels):
            if label == -1:          # noise
                continue
            mask = labels == label
            cluster_xy = xy[mask]
            cx, cy = cluster_xy[:, 0].mean(), cluster_xy[:, 1].mean()
            w = cluster_xy[:, 0].max() - cluster_xy[:, 0].min()
            h = cluster_xy[:, 1].max() - cluster_xy[:, 1].min()
            cluster_pts = [Point2D(x=p[0], y=p[1]) for p in cluster_xy]
            obstacles.append(DetectedObstacle(
                cx=float(cx), cy=float(cy),
                width=max(float(w), 0.05),
                height=max(float(h), 0.05),
                num_points=int(mask.sum()),
                points=cluster_pts,
                id=int(label),
            ))

        return obstacles
