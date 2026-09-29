#!/bin/bash
BASE="${VECTOR_BRAIN_DIR:-$HOME/vector-brain}"
PIDFILE="$BASE/brain.pid"
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "$(date -Is) stopped (pid $PID)" >> "$BASE/start.log"
    fi
    rm -f "$PIDFILE"
else
    echo "no pidfile"
fi
