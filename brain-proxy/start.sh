#!/bin/bash
# Start the vector-brain proxy if it isn't already running.
set -u
BASE="${VECTOR_BRAIN_DIR:-$HOME/vector-brain}"
PIDFILE="$BASE/brain.pid"
LOG="$BASE/start.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date -Is) already running (pid $(cat "$PIDFILE"))" >> "$LOG"
    exit 0
fi

cd "$BASE" || exit 1
nohup python3 "$BASE/server.py" >> "$BASE/brain.log" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date -Is) started (pid $(cat "$PIDFILE"))" >> "$LOG"
