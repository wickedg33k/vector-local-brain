#!/bin/bash
# (Re)create the vector-curious container with all its mounts.
# Run as the (non-root) user that owns the data dirs below.
#   VECTOR_FACES_DIR/current_people.json  <- written here after each capture
#   ROBOT_LOCK                             <- shared robot-control flock (faces/enroll.py, music/play.py)
# A plain `docker restart vector-curious` keeps these; only re-run this to recreate.
# NOTE: a (re)start triggers an immediate curious cycle.
set -euo pipefail
BASE="${VECTOR_BASE:-$HOME}"
VAULT_VECTOR_DIR="${VAULT_VECTOR_DIR:-$BASE/obsidian-brain/20-agents/vector}"
HA_WWW_DIR="${HA_WWW_DIR:-$BASE/homeassistant/www/vector}"   # optional: sentry snapshots for Home Assistant
ROBOT_LOCK="${ROBOT_LOCK:-$BASE/vector-pod/data/robot_control.lock}"
touch "$ROBOT_LOCK"
docker rm -f vector-curious >/dev/null 2>&1 || true
docker run -d \
  --name vector-curious \
  --network host \
  --restart unless-stopped \
  -e VECTOR_OWNER_NAME="${VECTOR_OWNER_NAME:-friend}" \
  -e VECTOR_DISALLOWED_NAMES="${VECTOR_DISALLOWED_NAMES:-}" \
  -v "$BASE/vector-curious:/app" \
  -v "$HA_WWW_DIR:/mnt/ha-www" \
  -v "$VAULT_VECTOR_DIR:/mnt/vector-agent-memory" \
  -v "$BASE/vector-brain:/mnt/vector-brain" \
  -v "$BASE/vector-logs:/mnt/vector-logs:ro" \
  -v "$BASE/vector-pod/data/vector-music:/mnt/vector-music:ro" \
  -v "$BASE/vector-pod/data/anki_vector:/root/.anki_vector:ro" \
  -v "$BASE/vector-faces:/mnt/vector-faces" \
  -v "$ROBOT_LOCK:/mnt/robot_control.lock" \
  vector-curious:latest
