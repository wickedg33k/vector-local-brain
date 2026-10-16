#!/bin/sh
# duck_watcher.sh -- runs on the GPU HOST (not inside any container).
# Tails wire-pod's (vector-pod) docker logs and mirrors voice-capture
# start/end into $HOME/vector-pod/data/vector-music/duck.flag, which is
# bind-mounted into vector-pod at /opt/vector-music/duck.flag -- the same
# file play.py's new _duck_watcher thread polls to drop/restore Vector's
# master volume so his mic can hear the owner over the music.
#
# "Stream type: OPUS"   -> a voice capture just started (wake word heard)
# "Transcribed text:"   -> that capture just ended (whatever the result)
#
# Added 2026-10-15 (hermes) after the owner's dance-party session where his
# replies kept transcribing as a garbled single "the" -- the song was
# drowning his voice in Vector's own mic. Deliberately dumb/line-based:
# no process to manage inside vector-pod, no new bind mount, no restart.
FLAG="${DUCK_FLAG:-${VECTOR_BASE:-$HOME}/vector-pod/data/vector-music/duck.flag}"
rm -f "$FLAG"
docker logs -f --since 0s vector-pod 2>&1 | while IFS= read -r line; do
  case "$line" in
    *"Stream type: OPUS"*)
      touch "$FLAG"
      ;;
    *"Transcribed text:"*)
      # small grace period so trailing mic buffer/STT finalization isn't
      # still happening the instant we unduck
      ( sleep 1; rm -f "$FLAG" ) &
      ;;
  esac
done
