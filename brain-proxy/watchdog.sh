#!/bin/bash
# Runs every minute via cron. Restarts the proxy if it isn't answering.
BASE="${VECTOR_BRAIN_DIR:-$HOME/vector-brain}"
if ! curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:11500/v1/models | grep -qE '^[0-9]+$'; then
    echo "$(date -Is) watchdog: not responding, restarting" >> "$BASE/watchdog.log"
    "$BASE/start.sh"
    exit 0
fi
# also make sure the pidfile process is actually alive (not a stale port holder)
PIDFILE="$BASE/brain.pid"
if [ -f "$PIDFILE" ] && ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date -Is) watchdog: pidfile stale, restarting" >> "$BASE/watchdog.log"
    rm -f "$PIDFILE"
    "$BASE/start.sh"
fi
