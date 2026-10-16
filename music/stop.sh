#!/bin/bash
# stop.sh -- kill the running vector-music play.py worker, if any.
set -u

PID_PATH="/run/vector-music.pid"
PID_PATH_FALLBACK="/tmp/vector-music.pid"
LOG="/opt/vector-music/music.log"
DANCE_FLAG="/opt/vector-music/dance.flag"  # stray flag shouldn't auto-start dancing on the next play

rm -f "$DANCE_FLAG"

pidfile=""
if [ -f "$PID_PATH" ]; then
    pidfile="$PID_PATH"
elif [ -f "$PID_PATH_FALLBACK" ]; then
    pidfile="$PID_PATH_FALLBACK"
fi

if [ -z "$pidfile" ]; then
    echo "no pidfile found, nothing to stop"
    exit 0
fi

pid=$(cat "$pidfile" 2>/dev/null | tr -d '[:space:]')
if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "no running worker (pid=$pid stale), cleaning up pidfile"
    rm -f "$pidfile"
    exit 0
fi

echo "$(date -Iseconds) stop.sh: killing worker pid=$pid" >> "$LOG"
kill -TERM "$pid" 2>/dev/null
for i in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.3
done
if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null
fi
rm -f "$pidfile"
echo "stopped pid=$pid"
