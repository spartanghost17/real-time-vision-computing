"""
SENTINEL — Multi-Object Tracker
ByteTrack-style tracker with speed estimation and counting lines.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection

logger = logging.getLogger("sentinel.cv.tracker")


@dataclass
class Track:
    """Persistent object track across frames."""
    track_id: int
    class_name: str
    class_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)  # centroid history
    timestamps: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    bbox: List[float] = field(default_factory=list)
    age: int = 0          # frames since creation
    hits: int = 0         # frames with matched detection
    misses: int = 0       # consecutive frames without match
    speed_kmh: float = 0.0
    is_stopped: bool = False

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None


@dataclass
class CountingLine:
    """
    Virtual line for counting objects crossing.
    Defined by two points (x1, y1) -> (x2, y2).
    """
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    direction: str = "both"  # "up", "down", "both"
    count_in: int = 0
    count_out: int = 0
    _crossed_ids: Set[int] = field(default_factory=set)


class SentinelTracker:
    """
    Multi-object tracker using Hungarian algorithm for association.
    Inspired by ByteTrack's two-stage matching approach.

    Features:
      - Persistent track IDs across frames
      - Speed estimation from pixel displacement + calibration
      - Counting lines for flow measurement
      - Stopped vehicle detection
    """

    def __init__(
        self,
        max_age: int = 30,           # frames before track deletion
        min_hits: int = 3,           # hits before track is confirmed
        iou_threshold: float = 0.3,  # min IoU for association
        pixels_per_meter: float = 8.0,  # calibration: px/m (adjust per camera)
        stopped_threshold_kmh: float = 2.0,
        stopped_frames: int = 90,    # ~3 seconds at 30fps
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.pixels_per_meter = pixels_per_meter
        self.stopped_threshold_kmh = stopped_threshold_kmh
        self.stopped_frames = stopped_frames

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.counting_lines: List[CountingLine] = []
        self._stopped_duration: Dict[int, int] = defaultdict(int)  # track_id -> stopped frame count

    def add_counting_line(self, name: str, p1: Tuple[float, float], p2: Tuple[float, float], direction: str = "both"):
        """Add a virtual counting line."""
        self.counting_lines.append(CountingLine(name=name, p1=p1, p2=p2, direction=direction))

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: Current frame detections from SentinelDetector
            timestamp: Frame timestamp in seconds

        Returns:
            List of confirmed (active) tracks with updated state
        """
        # Age all existing tracks
        for track in self.tracks.values():
            track.age += 1

        if not detections:
            self._handle_misses(list(self.tracks.keys()))
            return self._get_confirmed_tracks()

        if not self.tracks:
            # First frame — initialize all detections as new tracks
            for det in detections:
                self._create_track(det, timestamp)
            return self._get_confirmed_tracks()

        # ---- Hungarian matching via IoU ----
        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]

        cost_matrix = np.zeros((len(track_list), len(detections)))
        for i, track in enumerate(track_list):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou  # Hungarian minimizes cost

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < (1.0 - self.iou_threshold):
                tid = track_ids[row]
                det = detections[col]
                self._update_track(tid, det, timestamp)
                matched_tracks.add(tid)
                matched_dets.add(col)

        # Handle unmatched tracks (increment miss counter)
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_tracks]
        self._handle_misses(unmatched_track_ids)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, timestamp)

        # Prune dead tracks
        dead_ids = [tid for tid, t in self.tracks.items() if t.misses > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]
            self._stopped_duration.pop(tid, None)

        return self._get_confirmed_tracks()

    def _create_track(self, det: Detection, timestamp: float):
        """Initialize a new track from a detection."""
        track = Track(
            track_id=self.next_id,
            class_name=det.class_name,
            class_id=det.class_id,
            positions=[tuple(det.centroid)],
            timestamps=[timestamp],
            confidences=[det.confidence],
            bbox=det.bbox,
            hits=1,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track_id: int, det: Detection, timestamp: float):
        """Update existing track with matched detection."""
        track = self.tracks[track_id]
        track.positions.append(tuple(det.centroid))
        track.timestamps.append(timestamp)
        track.confidences.append(det.confidence)
        track.bbox = det.bbox
        track.hits += 1
        track.misses = 0

        # Estimate speed from last two positions
        if len(track.positions) >= 2 and len(track.timestamps) >= 2:
            p1 = np.array(track.positions[-2])
            p2 = np.array(track.positions[-1])
            dt = track.timestamps[-1] - track.timestamps[-2]
            if dt > 0:
                pixel_dist = np.linalg.norm(p2 - p1)
                meters = pixel_dist / self.pixels_per_meter
                speed_ms = meters / dt
                track.speed_kmh = speed_ms * 3.6
            else:
                track.speed_kmh = 0.0

        # Stopped vehicle detection
        if track.speed_kmh < self.stopped_threshold_kmh:
            self._stopped_duration[track_id] += 1
        else:
            self._stopped_duration[track_id] = 0

        track.is_stopped = self._stopped_duration[track_id] >= self.stopped_frames

        # Check counting line crossings
        if len(track.positions) >= 2:
            self._check_line_crossings(track)

        # Keep position history bounded (last 300 frames ~10s at 30fps)
        max_history = 300
        if len(track.positions) > max_history:
            track.positions = track.positions[-max_history:]
            track.timestamps = track.timestamps[-max_history:]
            track.confidences = track.confidences[-max_history:]

    def _handle_misses(self, track_ids: List[int]):
        for tid in track_ids:
            if tid in self.tracks:
                self.tracks[tid].misses += 1

    def _get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]

    def _check_line_crossings(self, track: Track):
        """Check if a track's last movement crossed any counting line."""
        p_prev = track.positions[-2]
        p_curr = track.positions[-1]

        for line in self.counting_lines:
            if track.track_id in line._crossed_ids:
                continue  # already counted

            if self._segments_intersect(p_prev, p_curr, line.p1, line.p2):
                # Determine direction via cross product
                dx = p_curr[0] - p_prev[0]
                dy = p_curr[1] - p_prev[1]
                lx = line.p2[0] - line.p1[0]
                ly = line.p2[1] - line.p1[1]
                cross = dx * ly - dy * lx

                if cross > 0:
                    line.count_in += 1
                else:
                    line.count_out += 1

                line._crossed_ids.add(track.track_id)
                logger.info(
                    f"Track {track.track_id} ({track.class_name}) crossed "
                    f"'{line.name}' | IN: {line.count_in} OUT: {line.count_out}"
                )

    @staticmethod
    def _compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _segments_intersect(
        p1: Tuple[float, float], p2: Tuple[float, float],
        p3: Tuple[float, float], p4: Tuple[float, float],
    ) -> bool:
        """Check if line segment p1-p2 intersects p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
