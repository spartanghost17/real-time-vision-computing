"""
SENTINEL — CV Pipeline Main
Orchestrates video ingestion → detection → tracking → Kafka emission.

Supports two ingestion modes:
  1. NYC DOT polling (default) — JPEG snapshots from webcams.nyctmc.org
  2. Local video file — for offline demo/testing

Usage:
    python -m cv_service.pipeline --mode nycdot --camera cam_nyc_042
    python -m cv_service.pipeline --mode nycdot --camera all
    python -m cv_service.pipeline --mode video --source ./demo/traffic.mp4
"""

import os, sys, time, signal, logging, argparse, threading
from typing import Optional

import cv2
import numpy as np

from .detector import SentinelDetector
from .tracker import SentinelTracker
from .producer import SentinelProducer
from .ingest_nycdot import NYCDOTPoller, NYC_CAMERAS, CameraConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.cv.pipeline")


class SentinelCVPipeline:
    def __init__(self, camera_config, kafka_bootstrap="localhost:9092",
                 model_path="yolov8n.pt", confidence=0.35, show_preview=False,
                 poll_interval=1.0, speed_limit_kmh=60.0):
        self.camera = camera_config
        self.show_preview = show_preview
        self.poll_interval = poll_interval
        self.speed_limit_kmh = speed_limit_kmh
        self.running = False

        self.detector = SentinelDetector(model_path=model_path, confidence_threshold=confidence)
        self.tracker = SentinelTracker(pixels_per_meter=4.0, stopped_frames=30)
        self.producer = SentinelProducer(
            bootstrap_servers=kafka_bootstrap,
            camera_id=camera_config.camera_id,
            camera_lat=camera_config.lat, camera_lon=camera_config.lon,
        )
        self.poller = NYCDOTPoller(camera_config, poll_interval=poll_interval)
        self._alerted_stopped = set()
        self._alerted_speeding = set()
        self._counting_lines_configured = False
        logger.info(f"Pipeline ready: {camera_config.camera_id} ({camera_config.name})")

    def _setup_counting_lines(self, w, h):
        if self._counting_lines_configured:
            return
        self.tracker.add_counting_line("main_crossing", (0, int(h*0.6)), (w, int(h*0.6)))
        self._counting_lines_configured = True

    def _process_frame(self, frame, frame_count, timestamp):
        h, w = frame.shape[:2]
        self._setup_counting_lines(w, h)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, timestamp)
        vehicles = [t for t in tracks if t.class_id in self.detector.VEHICLE_CLASSES]
        pedestrians = [t for t in tracks if t.class_id in self.detector.PERSON_CLASSES]

        for track in vehicles:
            if track.is_stopped and track.track_id not in self._alerted_stopped:
                self.producer.emit_alert(track, "stopped_vehicle",
                    {"stopped_frames": self.tracker._stopped_duration[track.track_id]})
                self._alerted_stopped.add(track.track_id)
            elif not track.is_stopped:
                self._alerted_stopped.discard(track.track_id)
            if track.speed_kmh > self.speed_limit_kmh and track.track_id not in self._alerted_speeding:
                self.producer.emit_alert(track, "speeding",
                    {"speed_kmh": track.speed_kmh, "limit_kmh": self.speed_limit_kmh})
                self._alerted_speeding.add(track.track_id)

        self.producer.emit_tracks(tracks, frame_count)
        for line in self.tracker.counting_lines:
            self.producer.emit_counting_update(line)

        avg_speed = sum(t.speed_kmh for t in vehicles) / len(vehicles) if vehicles else 0.0
        stopped_count = sum(1 for t in vehicles if t.is_stopped)

        return tracks, vehicles, pedestrians, avg_speed, stopped_count

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        self.running = True
        frame_count = 0
        t_start = time.perf_counter()
        logger.info(f"Starting NYC DOT pipeline: {self.camera.camera_id}")

        try:
            while self.running and self.poller.is_healthy:
                frame = self.poller.fetch_frame()
                if frame is None:
                    time.sleep(self.poll_interval)
                    continue

                frame_count += 1
                tracks, vehicles, peds, avg_speed, stopped = self._process_frame(
                    frame, frame_count, time.time())

                elapsed = time.perf_counter() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                self.producer.emit_frame_analytics(
                    frame_count, len(vehicles), len(peds), avg_speed, stopped, fps)

                if frame_count % 5 == 0:
                    self.producer.flush()

                if self.show_preview:
                    self._draw_debug(frame, tracks)
                    cv2.imshow(f"SENTINEL — {self.camera.camera_id}", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_count % 30 == 0:
                    logger.info(f"[{self.camera.camera_id}] F:{frame_count} V:{len(vehicles)} P:{len(peds)} Stop:{stopped}")

                time.sleep(self.poll_interval)
        finally:
            self.producer.flush()
            self.poller.close()
            if self.show_preview:
                cv2.destroyAllWindows()

    def run_video(self, source):
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        source_arg = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(source_arg)
        if not cap.isOpened():
            logger.error(f"Cannot open: {source}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(fps / 10.0))
        self.running = True
        raw = proc = 0
        t_start = time.perf_counter()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                raw += 1
                if raw % skip != 0:
                    continue
                proc += 1
                tracks, vehicles, peds, avg_speed, stopped = self._process_frame(
                    frame, proc, time.time())
                elapsed = time.perf_counter() - t_start
                self.producer.emit_frame_analytics(
                    proc, len(vehicles), len(peds), avg_speed, stopped,
                    proc / elapsed if elapsed > 0 else 0)
                if proc % 10 == 0:
                    self.producer.flush()
                if self.show_preview:
                    self._draw_debug(frame, tracks)
                    cv2.imshow("SENTINEL CV", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.producer.flush()
            cap.release()

    def stop(self):
        logger.info("Shutdown signal received")
        self.running = False

    def _draw_debug(self, frame, tracks):
        for line in self.tracker.counting_lines:
            cv2.line(frame, (int(line.p1[0]),int(line.p1[1])),
                     (int(line.p2[0]),int(line.p2[1])), (0,255,255), 2)
        for track in tracks:
            x1,y1,x2,y2 = [int(v) for v in track.bbox]
            color = (0,0,255) if track.is_stopped else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{track.track_id} {track.class_name} {track.speed_kmh:.0f}km/h",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def run_multi_camera(cameras, kafka, model, conf):
    threads, pipelines = [], []
    for cam in cameras:
        p = SentinelCVPipeline(cam, kafka, model, conf)
        pipelines.append(p)
        t = threading.Thread(target=p.run, name=cam.camera_id, daemon=True)
        threads.append(t)
        t.start()
        logger.info(f"Thread started: {cam.camera_id}")
    def shutdown(*_):
        for p in pipelines: p.stop()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    for t in threads: t.join()


def main():
    parser = argparse.ArgumentParser(description="SENTINEL CV Pipeline")
    parser.add_argument("--mode", choices=["nycdot","video"], default="nycdot")
    parser.add_argument("--camera", default=os.getenv("CAMERA_ID", "cam_nyc_042"))
    parser.add_argument("--source", default="0")
    parser.add_argument("--kafka", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "nycdot":
        if args.camera == "all":
            run_multi_camera(NYC_CAMERAS, args.kafka, args.model, args.confidence)
        else:
            cam = next((c for c in NYC_CAMERAS if c.camera_id == args.camera), None)
            if not cam:
                logger.error(f"Unknown camera. Available: {', '.join(c.camera_id for c in NYC_CAMERAS)}")
                sys.exit(1)
            SentinelCVPipeline(cam, args.kafka, args.model, args.confidence,
                              args.preview, args.poll_interval).run()
    else:
        cam = CameraConfig(os.getenv("CAMERA_ID","cam_local"), "", "Local Video",
                          float(os.getenv("CAMERA_LAT","40.7128")),
                          float(os.getenv("CAMERA_LON","-74.0060")))
        SentinelCVPipeline(cam, args.kafka, args.model, args.confidence, args.preview).run_video(args.source)

if __name__ == "__main__":
    main()
