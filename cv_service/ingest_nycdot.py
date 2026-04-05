"""
SENTINEL — NYC DOT Camera Ingestion
Polls JPEG snapshots from webcams.nyctmc.org at ~1 FPS.

NYC DOT cameras serve static JPEG images (352x240) that refresh every ~1 second.
No RTSP/HLS — we poll the image URL and decode each response as a frame.

API:
  GET https://webcams.nyctmc.org/api/cameras          → JSON array of all cameras
  GET https://webcams.nyctmc.org/api/cameras/{id}/image → current JPEG snapshot
"""

import io
import time
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger("sentinel.cv.ingest")

NYCTMC_BASE = "https://webcams.nyctmc.org/api/cameras"


@dataclass
class CameraConfig:
    """NYC DOT camera configuration."""
    camera_id: str          # SENTINEL internal ID (e.g. "cam_nyc_042")
    nyctmc_uuid: str        # NYCTMC UUID for the image endpoint
    name: str               # Human-readable location
    lat: float
    lon: float


# ── Pre-configured NYC cameras for the demo ──────────────────────
NYC_CAMERAS = [
    CameraConfig(
        camera_id="cam_nyc_042",
        nyctmc_uuid="907af141-b289-47d3-9f41-edc7667cff7e",
        name="5th Ave @ 42nd St",
        lat=40.7527, lon=-73.9822,
    ),
    CameraConfig(
        camera_id="cam_nyc_088",
        nyctmc_uuid="a8f2d065-c266-4378-ac43-3a6b1440e7aa",
        name="Queensboro Bridge North",
        lat=40.7570, lon=-73.9561,
    ),
    CameraConfig(
        camera_id="cam_nyc_times",
        nyctmc_uuid="eafc65f5-6ff9-4203-905f-3995b9fbc9eb",
        name="Times Square @ 46th St",
        lat=40.7580, lon=-73.9855,
    ),
    CameraConfig(
        camera_id="cam_nyc_fdr",
        nyctmc_uuid="50bdd399-1289-480d-8a05-97c05a785ab9",
        name="FDR Drive @ Houston St",
        lat=40.7205, lon=-73.9754,
    ),
    CameraConfig(
        camera_id="cam_nyc_bqe",
        nyctmc_uuid="8d2b3ae9-da68-4d37-8ae2-d3bc014f827b",
        name="BQE @ Bedford Ave / S 5th St",
        lat=40.710983, lon=-73.963168,
    ),
    CameraConfig(
        camera_id="cam_nyc_atlantic",
        nyctmc_uuid="0bbea8bd-10f1-4126-b3c5-9e9432eab749",
        name="Atlantic Ave @ Pennsylvania Ave",
        lat=40.675787, lon=-73.896898,
    ),
]


class NYCDOTPoller:
    """
    Polls NYC DOT JPEG camera endpoints for frames.

    The NYCTMC API serves a fresh JPEG snapshot on every request.
    No authentication required. Resolution is 352x240.
    Rate: ~1 request/second is respectful and matches the camera refresh rate.
    """

    def __init__(self, camera: CameraConfig, poll_interval: float = 1.0, timeout: float = 5.0):
        self.camera = camera
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.image_url = f"{NYCTMC_BASE}/{camera.nyctmc_uuid}/image"

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "SENTINEL-CV/1.0 (University Research Project)",
        })

        self._consecutive_failures = 0
        self._max_failures = 10

        logger.info(
            f"NYC DOT poller initialized: {camera.camera_id} "
            f"({camera.name}) @ {self.image_url}"
        )

    def fetch_frame(self) -> Optional[np.ndarray]:
        """
        Fetch a single JPEG snapshot and decode to OpenCV BGR array.

        Returns:
            np.ndarray (H, W, 3) BGR image, or None on failure.
        """
        try:
            resp = self._session.get(
                self.image_url,
                timeout=self.timeout,
                params={"t": int(time.time() * 1000)},  # cache-bust
            )
            resp.raise_for_status()

            # Decode JPEG bytes → numpy array → BGR
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning(f"Failed to decode JPEG from {self.camera.camera_id}")
                self._consecutive_failures += 1
                return None

            self._consecutive_failures = 0
            return frame

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout polling {self.camera.camera_id}")
            self._consecutive_failures += 1
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error polling {self.camera.camera_id}: {e}")
            self._consecutive_failures += 1
            return None

    @property
    def is_healthy(self) -> bool:
        return self._consecutive_failures < self._max_failures

    def close(self):
        self._session.close()


def discover_cameras() -> list:
    """
    Fetch the full NYC DOT camera registry from the API.
    Returns a list of camera metadata dicts.

    Useful for dynamically discovering available cameras.
    """
    try:
        resp = requests.get(NYCTMC_BASE, timeout=10)
        resp.raise_for_status()
        cameras = resp.json()
        online = [c for c in cameras if c.get("isOnline") == "true"]
        logger.info(f"Discovered {len(online)} online cameras out of {len(cameras)} total")
        return online
    except Exception as e:
        logger.error(f"Failed to discover cameras: {e}")
        return []
