"""
SENTINEL — API Server
FastAPI + WebSocket gateway between Spark/Kafka and the React frontend.

Endpoints:
  GET  /health              — Health check
  GET  /api/cameras         — List active camera feeds
  GET  /api/analytics       — Latest windowed analytics
  WS   /ws/live             — Real-time detection stream
  WS   /ws/alerts           — Real-time alert stream

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel.api")


# ── Kafka consumer config ────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"
TOPICS = {
    "detections": "sentinel.cv.detections",
    "alerts": "sentinel.cv.alerts",
    "analytics": "sentinel.cv.analytics",
    "counts": "sentinel.cv.counts",
}


# ── WebSocket connection manager ─────────────────────────────────
class ConnectionManager:
    """Manages WebSocket connections for real-time data push."""

    def __init__(self):
        self.live_connections: Set[WebSocket] = set()
        self.alert_connections: Set[WebSocket] = set()

    async def connect_live(self, ws: WebSocket):
        await ws.accept()
        self.live_connections.add(ws)
        logger.info(f"Live client connected ({len(self.live_connections)} total)")

    async def connect_alerts(self, ws: WebSocket):
        await ws.accept()
        self.alert_connections.add(ws)
        logger.info(f"Alert client connected ({len(self.alert_connections)} total)")

    def disconnect(self, ws: WebSocket):
        self.live_connections.discard(ws)
        self.alert_connections.discard(ws)

    async def broadcast_live(self, message: str):
        dead = set()
        for ws in self.live_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.live_connections -= dead

    async def broadcast_alert(self, message: str):
        dead = set()
        for ws in self.alert_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.alert_connections -= dead


manager = ConnectionManager()


# ── Kafka consumer background tasks ──────────────────────────────
async def kafka_to_websocket():
    """
    Background task: consume from Kafka topics and push to
    connected WebSocket clients.
    """
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "sentinel-api-gateway",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })

    consumer.subscribe([
        TOPICS["detections"],
        TOPICS["alerts"],
        TOPICS["analytics"],
    ])

    logger.info("Kafka consumer started — bridging to WebSocket clients")

    try:
        while True:
            msg = consumer.poll(timeout=0.1)

            if msg is None:
                await asyncio.sleep(0.01)
                continue

            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka error: {msg.error()}")
                continue

            topic = msg.topic()
            value = msg.value().decode("utf-8")

            if topic == TOPICS["alerts"]:
                await manager.broadcast_alert(value)
            else:
                await manager.broadcast_live(value)

    except Exception as e:
        logger.error(f"Kafka consumer error: {e}")
    finally:
        consumer.close()


# ── App lifecycle ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Kafka consumer on app startup."""
    task = asyncio.create_task(kafka_to_websocket())
    logger.info("SENTINEL API server started")
    yield
    task.cancel()
    logger.info("SENTINEL API server stopped")


# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="SENTINEL API",
    description="Real-time traffic intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "operational",
        "service": "sentinel-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "websocket_clients": {
            "live": len(manager.live_connections),
            "alerts": len(manager.alert_connections),
        },
    }


@app.get("/api/cameras")
async def list_cameras():
    """List configured camera feeds."""
    return {
        "cameras": [
            {
                "camera_id": "cam_001",
                "name": "Broadway & 42nd St",
                "lat": 40.7580,
                "lon": -73.9855,
                "status": "active",
                "source_type": "DOT_RTSP",
            },
            {
                "camera_id": "cam_002",
                "name": "FDR Drive & Houston",
                "lat": 40.7205,
                "lon": -73.9754,
                "status": "active",
                "source_type": "DOT_RTSP",
            },
        ]
    }


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Real-time detection + analytics stream.
    Frontend connects here for the live map + dashboard.
    """
    await manager.connect_live(ws)
    try:
        while True:
            # Keep connection alive; data pushed by kafka_to_websocket()
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    Alert-only stream for the notification panel.
    Only receives stopped_vehicle, speeding, etc.
    """
    await manager.connect_alerts(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)