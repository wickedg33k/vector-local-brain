#!/bin/bash
# Dumps the tail of vector-curious's docker logs to a plain file so the
# read-only vector-dashboard container can show it without a docker socket
# mount. Runs on the HOST (gpu-host) as youruser via cron, once a minute.
# Does NOT touch, restart, or exec into the vector-curious container --
# read-only `docker logs`.
set -euo pipefail
OUT="$HOME/vector-dashboard/cache/curious.log"
TMP="${OUT}.tmp"
mkdir -p "$(dirname "$OUT")"
docker logs --tail 500 vector-curious > "$TMP" 2>&1 || true
mv "$TMP" "$OUT"
