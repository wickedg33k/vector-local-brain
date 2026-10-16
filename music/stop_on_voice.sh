#!/bin/sh
# stop_on_voice.sh -- runs on the GPU HOST (not inside any container).
#
# Replaces the earlier "duck" approach (added 2026-10-15, same day): lowering
# Vector's master volume mid-stream was live-tested and found to kill the
# external audio stream outright (robot's own journal showed
# sdk.audiostream_ended at ~500ms every time). the owner's actual complaint was
# that he couldn't interrupt the music by voice at all ("the party's over"
# didn't match anything, "stop the music" got transcribed as garbage because
# the music was drowning the mic). the owner prefers control over continuity, so
# instead of trying to duck the volume, this just STOPS the music outright
# the moment a voice capture starts -- using the exact same stop.sh the owner's
# "stop the music" custom intent already uses, so there's only one stop path
# to reason about.
#
# Tails wire-pod's (vector-pod) docker logs for two line shapes:
#   "Custom Intent Matched: play_music_custom" / "...dance_music_custom"
#       -> records when a song/dance actually started (wall clock).
#   "Stream type: OPUS"
#       -> a voice capture just began (wake word heard). If this happens
#          MORE than GRACE_SECONDS after the last recorded music start, stop
#          the music so the owner is heard clearly. The grace window exists so
#          this doesn't fire on wire-pod's own near-immediate conversation-
#          mode auto-listen right after the custom intent's ack, which is not
#          the owner trying to say anything.
#
# `docker exec vector-pod /opt/vector-music/stop.sh` is idempotent/harmless
# when nothing is playing (confirmed: it just logs "no running worker,
# stale" and exits 0), so this intentionally does NOT try to track whether
# the song already ended naturally -- every post-grace OPUS stream just
# calls stop.sh, and stop.sh itself is the single source of truth for
# whether there's anything to actually kill.
GRACE_SECONDS=5
LOG="${STOP_ON_VOICE_LOG:-${VECTOR_BASE:-$HOME}/vector-pod/stop_on_voice.log}"
music_started_at=0

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" >> "$LOG"
}

log "stop_on_voice.sh starting"
docker logs -f --since 0s vector-pod 2>&1 | while IFS= read -r line; do
  case "$line" in
    *"Custom Intent Matched: play_music_custom"*|*"Custom Intent Matched: dance_music_custom"*)
      music_started_at=$(date +%s)
      log "music start detected: $line"
      ;;
    *"Stream type: OPUS"*)
      now=$(date +%s)
      elapsed=$((now - music_started_at))
      if [ "$music_started_at" -gt 0 ] && [ "$elapsed" -ge "$GRACE_SECONDS" ]; then
        log "voice capture ${elapsed}s after music start -- stopping music"
        docker exec vector-pod /opt/vector-music/stop.sh >> "$LOG" 2>&1
      else
        log "voice capture ${elapsed}s after music start -- within grace window, ignoring"
      fi
      ;;
  esac
done
