#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SENTINEL — Create Kafka Topics
# Run this after starting Kafka with docker compose up kafka
# ─────────────────────────────────────────────────────────────

set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9092}"

echo "╔═══════════════════════════════════════════╗"
echo "║  SENTINEL — Kafka Topic Setup             ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "Bootstrap server: $BOOTSTRAP"
echo ""

TOPICS=(
    "sentinel.cv.detections:3"    # Partitioned by camera_id for parallelism
    "sentinel.cv.alerts:1"        # Single partition — low volume, ordering matters
    "sentinel.cv.counts:1"        # Counting line updates
    "sentinel.cv.analytics:3"     # Per-frame analytics
)

for topic_config in "${TOPICS[@]}"; do
    IFS=':' read -r topic partitions <<< "$topic_config"
    echo "Creating topic: $topic (partitions=$partitions)"
    docker exec sentinel-kafka kafka-topics.sh \
        --bootstrap-server "$BOOTSTRAP" \
        --create --if-not-exists \
        --topic "$topic" \
        --partitions "$partitions" \
        --replication-factor 1
done

echo ""
echo "Current topics:"
docker exec sentinel-kafka kafka-topics.sh \
    --bootstrap-server "$BOOTSTRAP" --list

echo ""
echo "Done."
