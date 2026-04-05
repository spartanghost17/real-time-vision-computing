"""
SENTINEL — CV Detection Module
YOLOv8 object detection on video frames with confidence filtering.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("sentinel.cv.detector")


@dataclass
class Detection:
    """Single object detection from a video frame."""
    bbox: List[float]          # [x1, y1, x2, y2] pixel coords
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None  # Assigned later by tracker
    speed_kmh: Optional[float] = None
    centroid: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.centroid:
            x1, y1, x2, y2 = self.bbox
            self.centroid = [(x1 + x2) / 2, (y1 + y2) / 2]


class SentinelDetector:
    """
    Wraps YOLOv8 for SENTINEL's vehicle/pedestrian detection.

    Filters to relevant COCO classes:
      - 2: car, 3: motorcycle, 5: bus, 7: truck (vehicles)
      - 0: person (pedestrians)
      - 1: bicycle
    """

    # COCO class IDs we care about
    VEHICLE_CLASSES = {2, 3, 5, 7}
    PERSON_CLASSES = {0}
    BICYCLE_CLASSES = {1}
    TARGET_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES | BICYCLE_CLASSES

    CLASS_LABELS = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck"
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # nano for speed; swap to yolov8s/m for accuracy
        confidence_threshold: float = 0.35,
        device: str = "cpu",  # "cuda:0" if GPU available
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device

        logger.info(f"Loading YOLOv8 model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3)

        Returns:
            List of Detection objects for relevant classes above threshold.
        """
        t0 = time.perf_counter()

        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            classes=list(self.TARGET_CLASSES),
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.CLASS_LABELS.get(cls_id, f"class_{cls_id}"),
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Detection: {len(detections)} objects in {elapsed_ms:.1f}ms")

        return detections

    def classify_detection(self, det: Detection) -> str:
        """Categorize detection into SENTINEL's entity taxonomy."""
        if det.class_id in self.VEHICLE_CLASSES:
            return "VEHICLE"
        elif det.class_id in self.PERSON_CLASSES:
            return "PEDESTRIAN"
        elif det.class_id in self.BICYCLE_CLASSES:
            return "CYCLIST"
        return "UNKNOWN"
