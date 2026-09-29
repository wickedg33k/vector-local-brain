#!/usr/bin/env bash
# deploy_live10.sh -- flip the live Vector wire-pod container to
# wire-pod-local:live-10-py (2026-09-29, Hermes).
#
# BACKGROUND: three fixes landed on top of live-9-py's source:
#
#  1. LLM-first command reliability (classify.go, matchIntentSend.go):
#     app-sourced doIntent/AppIntent calls are dropped by the robot's
#     UserIntentComponent more often than not (reproduced 13:32Z 2026-09-29,
#     robot docked/idle: "UserIntentComponent.Update.PendingIntentNotCleared.
#     ForceClear" -> "@behavior.voice_command.dropped explore_start App").
#     Voice-sourced intents (a real keyphrase CORE match) always work. Fix:
#     before falling through to the LLM chat path, ProcessTextAll now makes
#     one fast (<=3s, temp 0, /no_think, direct-to-Ollama, non-streaming)
#     classification call against the same AllowedLLMIntents whitelist
#     doIntent already enforces. A confident match is sent as the voice
#     stream's OWN intent result -- exactly like a keyphrase CORE match --
#     and the LLM chat turn (and therefore doIntent) is skipped entirely for
#     that turn. NONE/timeout/error falls straight through to the existing
#     SOCIAL/LLM path, unchanged; pendingintent.go's deferred-doIntent
#     handling is untouched and stays as a secondary fallback.
#
#  2. Crash fix (kgsim_cmds.go): GetActionsFromString panicked
#     ("index out of range [1] with length 1") on a single-segment
#     {{intent_name}} tag with no "||" separator (real LLM output:
#     "{{intent_play_popawheelie}}"). Now guarded -- a missing param is
#     treated as empty, never panics, matches CmdParamToAction's existing
#     log-and-ignore behavior for any unrecognized command name.
#
#  3. More CORE explore keyphrases (intent-data/en-US.json): added "free
#     roam", "go roam", "roam around", "wander", "go wander", "go play" to
#     intent_explore_start (go explore/explore mode/explorer mode were
#     already present from earlier rounds). Deliberately did NOT re-add
#     "look around" or "back" -- those were removed on purpose in live-9.
#
# Separately (NOT part of the image -- runtime config / other host state):
#  - apiConfig.json: appended verbatim "COMMON SENSE RULES" persona text to
#    knowledge.openai_prompt. Backed up as apiConfig.json.bak-commonsense.
#    knowledge.model/llm_first/conversation_mode left untouched
#    (qwen3:32b / true / true).
#  - vector-brain proxy (~/vector-brain/server.py, separate host
#    process, NOT in this image): now sends keep_alive on every request it
#    forwards to Ollama, so qwen3:32b stays resident between turns instead
#    of cold-loading (14-18s) after ~idle. Backed up as
#    server.py.bak-keepalive, already restarted+verified (`ollama ps` shows
#    qwen3:32b UNTIL=Forever).
#
# Tests: new unit tests in classify_test.go (parseClassifyResponse,
# buildClassifyPrompt, stripThinkTags) and kgsim_cmds_test.go (the exact
# crash repro + regression coverage for the existing two-segment
# {{cmd||param}} form) -- go vet clean, full pkg/wirepod/ttr suite green
# (see live10-vet-test.log).
#
# DEPLOY TIMING: only run this when the pod log shows no requests in the
# last ~2 minutes (the owner not mid-conversation) -- check first:
#   ssh youruser@GPU_HOST_IP 'docker logs --since 2m vector-pod 2>&1 | tail -30'
#
# POST-DEPLOY: watch the pod log for "wire-pod started successfully" and a
# jdocs handshake for YOUR_ESN, then for real "Transcribed text" lines on
# the next few real requests with no panics/EOF/DeadlineExceeded/Canceled.
# If ANY request fails to transcribe, or a panic appears, roll back
# immediately:
#   ./deploy_live10.sh --rollback
#
# Usage:
#   ./deploy_live10.sh            # deploy live-10-py (does NOT touch apiConfig.json)
#   ./deploy_live10.sh --rollback # IMAGE SWAP ONLY -- does NOT touch apiConfig.json
#
# Recreates `vector-pod` with the SAME mounts/env/network flags as the
# current live container. If those mounts/env ever change, update BOTH this
# script and CUTOVER.md's "step 9" reference command together.

set -euo pipefail

VECTOR_POD_DIR="${VECTOR_POD_DIR:-$HOME/vector-pod}"
APICONFIG="$VECTOR_POD_DIR/data/chipper/apiConfig.json"
CONTAINER_NAME="vector-pod"
ESN="${VECTOR_ESN:-YOUR_ESN}"
NEW_IMAGE="wire-pod-local:live-10-py"
PREV_IMAGE="wire-pod-local:live-9-py"  # the LAST KNOWN GOOD image -- confirm live-9 was actually verified before relying on this
JDOCS_WAIT_TIMEOUT_SEC=180  # robot Wi-Fi drops ~once/minute; be patient (target restart <=30s)
TS="$(date +%Y%m%d%H%M%S)"

log() { echo "[deploy_live10] $*"; }

recreate_container() {
  local image_tag="$1"
  log "Removing any existing '$CONTAINER_NAME' / staging containers..."
  docker rm -f vector-pod-staging "$CONTAINER_NAME" >/dev/null 2>&1 || true

  log "Starting '$CONTAINER_NAME' from image $image_tag ..."
  docker run -d --name "$CONTAINER_NAME" \
    --network host --restart unless-stopped \
    -e WIREPOD_DATA_DIR=/data \
    -e DISABLE_MDNS=false \
    -v "$VECTOR_POD_DIR/data:/data" \
    -v "$VECTOR_POD_DIR/data/vector-bridge:/etc/vector-bridge" \
    -v "$VECTOR_POD_DIR/data/vector-ask:/opt/vector-ask" \
    -v "$VECTOR_POD_DIR/data/vector-music:/opt/vector-music" \
    -v "$VECTOR_POD_DIR/data/vector-wake:/opt/vector-wake" \
    -v "$VECTOR_POD_DIR/data/anki_vector:/root/.anki_vector" \
    "$image_tag"
}

wait_for_jdocs() {
  log "Waiting up to ${JDOCS_WAIT_TIMEOUT_SEC}s for \"Successfully got jdocs from $ESN\" ..."
  local waited=0
  while (( waited < JDOCS_WAIT_TIMEOUT_SEC )); do
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Successfully got jdocs from $ESN"; then
      log "Robot reconnected (jdocs handshake seen) after ~${waited}s."
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  log "WARNING: did not see the jdocs handshake within ${JDOCS_WAIT_TIMEOUT_SEC}s."
  log "This does NOT necessarily mean it failed -- check 'docker logs $CONTAINER_NAME' and"
  log "consider power-cycling the robot to force mDNS rediscovery (per CUTOVER.md step 11)."
  return 1
}

print_status() {
  log "----- status -----"
  docker ps --filter "name=$CONTAINER_NAME" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  log "Last 15 log lines:"
  docker logs "$CONTAINER_NAME" 2>&1 | tail -15
  log "get_config knowledge section:"
  curl -s http://127.0.0.1:8080/api/get_config 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -A2 -i "conversation" || \
    log "(get_config check skipped/failed -- non-fatal, verify manually if needed)"
  log "-------------------"
}

deploy() {
  log "=== Deploying $NEW_IMAGE (LLM-first classifier + crash fix + explore keyphrases) ==="

  if [[ -z "$(docker images -q "$NEW_IMAGE" 2>/dev/null)" ]]; then
    log "ERROR: image $NEW_IMAGE not found locally. Build it first (see CUTOVER.md)."
    exit 1
  fi

  log "NOT touching apiConfig.json this round via this script (its commonsense-rules"
  log "append already happened separately and is backed up as apiConfig.json.bak-commonsense)."

  recreate_container "$NEW_IMAGE"
  wait_for_jdocs || true
  print_status

  log "Deploy complete. If anything looks wrong, run: $0 --rollback"
  log "Previous image tag: $PREV_IMAGE (still present locally)."
}

rollback() {
  log "=== Rolling back to $PREV_IMAGE (IMAGE SWAP ONLY) ==="
  log "Rollback does NOT touch apiConfig.json AT ALL -- only swaps the container image."

  if [[ -z "$(docker images -q "$PREV_IMAGE" 2>/dev/null)" ]]; then
    log "ERROR: rollback image $PREV_IMAGE not found locally. Cannot roll back automatically."
    exit 1
  fi

  recreate_container "$PREV_IMAGE"
  wait_for_jdocs || true
  print_status

  log "Rollback complete (image only -- apiConfig.json untouched)."
}

case "${1:-}" in
  --rollback)
    rollback
    ;;
  "" )
    deploy
    ;;
  * )
    echo "Usage: $0 [--rollback]" >&2
    exit 2
    ;;
esac
