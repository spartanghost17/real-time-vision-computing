"""
SENTINEL — Kafka Producer
Emits structured CV detection events to Kafka topics.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from confluent_kafka import Producer

from .tracker import Track, CountingLine

logger = logging.getLogger("sentinel.cv.producer")


# ── Kafka topics ──────────────────────────────────────────────────
TOPIC_DETECTIONS = "sentinel.cv.detections"       # Per-frame track updates
TOPIC_ALERTS = "sentinel.cv.alerts"               # Anomaly alerts (stopped vehicle, etc.)
TOPIC_COUNTS = "sentinel.cv.counts"               # Counting line events
TOPIC_ANALYTICS = "sentinel.cv.analytics"         # Aggregated per-frame stats


class SentinelProducer:
    """
    Produces structured JSON messages to Kafka for downstream
    consumption by Spark Structured Streaming.

    Message schemas are designed for easy Spark DataFrame ingestion:
      - Flat JSON (no deep nesting)
      - Consistent timestamp format (ISO 8601)
      - camera_id as partition key for ordering guarantees
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        camera_id: str = "cam_001",
        camera_lat: float = 40.7128,
        camera_lon: float = -74.0060,
    ):
        self.camera_id = camera_id
        self.camera_lat = camera_lat
        self.camera_lon = camera_lon

        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": f"sentinel-cv-{camera_id}",
            "acks": "1",
            "linger.ms": 5,
            "batch.num.messages": 100,
        })

        logger.info(f"Kafka producer initialized for camera {camera_id}")

    def _delivery_callback(self, err, msg):
        if err:
            logger.error(f"Kafka delivery failed: {err}")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def emit_tracks(self, tracks: List[Track], frame_number: int):
        """
        Emit one message per active track to the detections topic.

        This is the PRIMARY data feed that Spark consumes.
        Each message represents a single object's state at this frame.
        """
        timestamp = self._now_iso()

        for track in tracks:
            msg = {
                "event_type": "track_update",
                "camera_id": self.camera_id,
                "camera_lat": self.camera_lat,
                "camera_lon": self.camera_lon,
                "timestamp": timestamp,
                "frame_number": frame_number,
                "track_id": track.track_id,
                "class_name": track.class_name,
                "class_id": track.class_id,
                "bbox_x1": track.bbox[0],
                "bbox_y1": track.bbox[1],
                "bbox_x2": track.bbox[2],
                "bbox_y2": track.bbox[3],
                "centroid_x": track.last_position[0] if track.last_position else 0,
                "centroid_y": track.last_position[1] if track.last_position else 0,
                "confidence": track.confidences[-1] if track.confidences else 0,
                "speed_kmh": round(track.speed_kmh, 2),
                "is_stopped": track.is_stopped,
                "track_age": track.age,
                "track_hits": track.hits,
            }

            self.producer.produce(
                topic=TOPIC_DETECTIONS,
                key=self.camera_id,
                value=json.dumps(msg),
                callback=self._delivery_callback,
            )

    def emit_alert(self, track: Track, alert_type: str, details: Optional[Dict] = None):
        """
        Emit anomaly alert to the alerts topic.
        Consumed by Spark for cross-stream correlation.
        """
        msg = {
            "event_type": "alert",
            "alert_type": alert_type,  # "stopped_vehicle", "wrong_way", "speeding"
            "camera_id": self.camera_id,
            "camera_lat": self.camera_lat,
            "camera_lon": self.camera_lon,
            "timestamp": self._now_iso(),
            "track_id": track.track_id,
            "class_name": track.class_name,
            "speed_kmh": round(track.speed_kmh, 2),
            "bbox_x1": track.bbox[0],
            "bbox_y1": track.bbox[1],
            "bbox_x2": track.bbox[2],
            "bbox_y2": track.bbox[3],
            "details": details or {},
        }

        self.producer.produce(
            topic=TOPIC_ALERTS,
            key=self.camera_id,
            value=json.dumps(msg),
            callback=self._delivery_callback,
        )

        logger.warning(f"ALERT [{alert_type}] Track {track.track_id} @ {self.camera_id}")

    def emit_counting_update(self, line: CountingLine):
        """Emit counting line state to the counts topic."""
        msg = {
            "event_type": "counting_update",
            "camera_id": self.camera_id,
            "timestamp": self._now_iso(),
            "line_name": line.name,
            "count_in": line.count_in,
            "count_out": line.count_out,
            "net_flow": line.count_in - line.count_out,
        }

        self.producer.produce(
            topic=TOPIC_COUNTS,
            key=self.camera_id,
            value=json.dumps(msg),
            callback=self._delivery_callback,
        )

    def emit_frame_analytics(
        self,
        frame_number: int,
        total_vehicles: int,
        total_pedestrians: int,
        avg_speed_kmh: float,
        stopped_count: int,
        fps: float,
    ):
        """
        Emit per-frame aggregate analytics.
        Spark uses this for windowed trend analysis.
        """
        msg = {
            "event_type": "frame_analytics",
            "camera_id": self.camera_id,
            "camera_lat": self.camera_lat,
            "camera_lon": self.camera_lon,
            "timestamp": self._now_iso(),
            "frame_number": frame_number,
            "total_vehicles": total_vehicles,
            "total_pedestrians": total_pedestrians,
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "stopped_count": stopped_count,
            "inference_fps": round(fps, 1),
        }

        self.producer.produce(
            topic=TOPIC_ANALYTICS,
            key=self.camera_id,
            value=json.dumps(msg),
            callback=self._delivery_callback,
        )

    def flush(self):
        """Flush all pending messages. Call periodically or at shutdown."""
        self.producer.flush(timeout=5)
