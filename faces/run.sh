#!/bin/bash
# (Re)build and (re)start the vector-faces container. Run as the non-root
# user that owns this directory. `run.sh --no-build` restarts only.
set -euo pipefail
BASE="${VECTOR_BASE:-$HOME}"
FACES_DIR="${FACES_DIR:-$BASE/vector-faces}"
ROBOT_LOCK="${ROBOT_LOCK:-$BASE/vector-pod/data/robot_control.lock}"
cd "$FACES_DIR"
[ "${1:-}" = "--no-build" ] || docker build -q -t vector-faces:latest .
docker rm -f vector-faces >/dev/null 2>&1 || true
docker run -d \
  --name vector-faces \
  --network host \
  --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e HOME=/root \
  -e TZ="${TZ:-UTC}" \
  -e FACES_BIND=127.0.0.1 \
  -e FACES_PORT=8096 \
  -v "$FACES_DIR:/app" \
  -v "$BASE/vector-brain:/mnt/vector-brain:ro" \
  -v "$BASE/vector-pod/data/anki_vector:/root/.anki_vector:ro" \
  -v "$ROBOT_LOCK:$ROBOT_LOCK" \
  vector-faces:latest >/dev/null
sleep 3
curl -s -m 5 http://127.0.0.1:8096/health; echo
