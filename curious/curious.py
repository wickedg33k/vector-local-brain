#!/usr/bin/env python3
"""
Curious Vector -- occasional camera snapshot + local-vision description +
playful LLM question, spoken by Vector.

Runs as its own long-lived daemon inside the `vector-curious` container on
gpu-host (never on the LAN's other shared hosts -- the owner's rule). It polls roughly every
POLL_INTERVAL_SECONDS, and only does anything (SDK connect, camera, vision,
LLM, speak) when it decides Vector is idle, out of quiet hours, not
disabled, and it's been long enough (randomized 20-30 min) since the last
question.

REWORK 2026-09-27 (incident: 2026-09-27 01:04:17-01:04:34 UTC -- Vector
"went limp" mid-conversation because a curious cycle requested behavior
control while the owner was actively talking to him; the old brain_busy check
only looked at the last 45s of brain-proxy traffic, which is far too short
a window to catch a real conversation with pauses in it):

  - Behavior control (which fights wire-pod's own control session and can
    freeze Vector for several seconds while the two negotiate) is now taken
    in ONE short window per cycle instead of two: only for the camera
    capture. The announcement and the question are both spoken through
    wire-pod's own HTTP API (/api-sdk/say_text), which uses wire-pod's
    *existing* connection/control session instead of opening a competing
    one from here.
  - That one remaining control window (capture) has a hard wall-clock
    timeout (control_timeout_seconds, default 15s) enforced with a
    background thread + a `finally: robot.disconnect()` that always runs,
    even if the SDK call itself is hung and can't be cancelled.
  - Busy detection is fail-safe now: any check that can't be verified
    (battery unknown, robot unreachable, wire-pod log buffer unreadable)
    means SKIP, not proceed. Three independent busy signals, all
    configurable:
      * voice_quiet_seconds (default 600s): any "transcribed text:" line
        in wire-pod's own HTTP log buffer (GET /api-sdk's sibling
        /api/get_logs -- plain text, no auth, no docker access at all)
        means a real voice interaction happened recently -- skip. That
        buffer is a fixed-size ring that can roll over faster than our
        window during a chatty stretch, so the most recent transcription
        timestamp we've ever seen is ALSO persisted in state.json and
        merged in on every poll (not just when we'd otherwise act), so a
        20-30 min cooldown gap can never cause us to lose track of it.
        (2026-09-27: originally read via a read-only docker.sock mount +
        docker-py, but that's root-equivalent on gpu-host and :ro doesn't
        restrict the Docker API -- switched to wire-pod's own HTTP API,
        no socket access needed at all.)
      * brain_busy_seconds (default 600s, was 45s): brain-proxy chat POST
        activity, via brain.log mtime.
      * music_playing: unchanged, music.log tail heuristic.
  - New: Vector announces entering curious mode BEFORE the snapshot, one
    short line picked at random from config `announcement_lines` (the owner's
    wording), spoken via wire-pod HTTP say_text.

REWORK 2026-09-28 (incident: repeated fault 914 / vic-engine restarts,
most recently 13:26 and 16:14:17 local on 2026-09-28 -- the robot has only
436MB RAM / ~37MB available and was seeing a full anki_vector SDK connect
(BehaviorControl + EventStream + GetCameraConfig) roughly every 20-27s,
sometimes with a SECOND connect landing within milliseconds of the first
still tearing down -- "Connection id already set" on the robot side, then
vic-engine stalls):

  - Gate checks no longer open an SDK connection at all. music_playing/
    voice_recently_active/brain_busy are file/HTTP based as before;
    get_battery is wire-pod's HTTP endpoint (also as before). The old
    peek_robot_status() SDK connect used purely to read is_being_held/
    are_motors_moving/is_animating/is_pathing/is_in_calm_power_mode is
    GONE -- there is no wire-pod HTTP endpoint for those flags, so instead
    of opening a connection just to look, that status read now happens
    INSIDE the one connection a cycle opens when it's actually about to
    capture (see capture_image/capture_night_images below). A cycle that
    turns out to be gated by physical status (held/moving/animating/
    asleep) aborts the SAME connection instead of never having one -- net
    effect: at most one SDK connect per cycle, zero when the cheap gates
    already say no.
  - capture_image()/capture_night_images() do ONE connect: connect, read
    status, and only if status says it's safe do they grab the frame(s).
    Always torn down in `finally`, same hard-timeout pattern as before.
    This is also why the mid-cycle double-connect from the incident can no
    longer happen -- the old code opened one connection for the pre-
    announcement gate peek and a SECOND, separate one moments later for
    the capture; now it's the same single connection doing both jobs.
  - maybe_start_explore() (the wander decision) no longer opens an SDK
    connection either -- it never captured an image to begin with, so per
    the "only connect when actually capturing" rule it now relies on the
    existing music/voice/brain/battery HTTP+file checks alone. Vector's
    own firmware still governs whether an intent_explore_start AppIntent
    (sent over wire-pod's own already-open connection, not ours) actually
    does anything unsafe -- we're not driving wheels directly.
  - Idle polling backs off instead of hammering every poll_interval_seconds
    (20s) forever: base cadence while genuinely idle is now 60s, and each
    consecutive cycle where nothing happened (skip/disabled/breaker) steps
    the sleep up the ladder 60 -> 120 -> 300s, capped at 300s. Any cycle
    that actually touches the robot/wire-pod for real (a full day or night
    capture attempt, live voice activity, an in-progress explore window)
    resets the sleep back to 60s so we stay responsive during real
    activity. This directly kills the "gate keeps failing every 20s, keeps
    reconnecting every 20s" storm seen in the incident logs.
  - Circuit breaker: 2 consecutive capture attempts that come back "fail"
    (hard timeout, SDK exception, or connected-but-no-image -- NOT the
    same as a normal "robot is held/moving/asleep" skip, which doesn't
    count) pause ALL robot contact -- gates, battery checks, everything --
    for 10 minutes, logged clearly, then resume normally.
  - get_battery() now treats an empty/non-JSON body from wire-pod
    (previously surfaced as "Expecting value: line 1 column 1 (char 0)")
    as a normal "couldn't verify" case instead of a raw JSONDecodeError.

REWORK 2026-09-29 (incident: owner found Vector asleep on a mouse pad,
off the charger, after a wander -- nothing ever sent him back):

  - New maybe_send_home()/trigger_go_home() pair. Go-home reuses the exact
    same low-risk mechanism as trigger_explore() -- wire-pod's HTTP
    cloud_intent API (no SDK connect, no behavior control held from our
    side), sending intent_system_charger, the same intent wire-pod's own
    admin UI "Go home" button sends and the same one wire-pod's own
    intent-keyphrase tests map "go home"/"go to your charger" to.
  - Called: right after a wander/explore window ends, after every day
    curious cycle exit (success or early-abort) that happened off the
    charger, when quiet hours begin, and at every night tick (independent
    of run_night_cycle's own ~12-min capture cadence) -- see call sites in
    run_day_cycle() and run_cycle(). Never starts a wander at night
    (unchanged -- run_night_cycle never calls maybe_start_explore).
  - Respects the same busy signals as everything else (voice activity,
    music/dance, brain-busy) -- skips and relies on the next poll to
    retry rather than forcing the intent through mid-interaction.
  - Retries up to config return_home_max_tries (default 3) times, spaced
    return_home_retry_minutes (default 3) apart, tracked in state.json
    (go_home_tries/go_home_next_try_at) so spacing survives restarts.
    Whole feature toggled by return_home_after_wander (default true).
  - Low battery off charger (battery_level <= battery_low_level, same
    threshold _shared_busy_gate already uses) sends home immediately
    (bypassing the try-spacing) and sets low_battery_hold, which also now
    blocks maybe_start_explore() from starting any new wander until
    on_charger is seen again (cleared there).

REWORK 2026-09-29b (incident: go-home intent silently dropped -- Vector
"went home" via cloud_intent(intent_system_charger) at 03:56:04 UTC but
vic-engine logged "Intent 'system_charger' has been pending for 3 ticks,
forcing a clear" -> "@behavior.voice_command.dropped system_charger" ->
anim_communication_cantdothat. The AppIntent-injected system_charger isn't
claimed by any behavior, so it just times out and gets dropped. Confirmed
against the SAME night's log: two separate SDK-level `DriveOnCharger` RPCs
at 03:57:21 and 03:57:43 DID trigger the real `@behavior.feature.start
GoHome` behavior (interrupted once by ReactToCliff near a table edge, but
Vector was docked by ~04:03) -- checked wire-pod's sdkapp source
(chipper/pkg/wirepod/sdkapp/server.go) first and confirmed there is NO
HTTP endpoint there that performs DriveOnCharger -- cloud_intent is the
only go-home-shaped thing it exposes, and it's the broken path):

  - New drive_home_sdk(): go-home now defaults to calling the SDK's real
    `robot.behavior.drive_on_charger()` directly, inside ONE short-lived
    anki_vector.Robot connection with behavior_control_level=
    ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY (this needs to
    actually cut in over Vector's own autonomy the way a spoken "go home"
    command would -- unlike capture_image()'s DEFAULT_PRIORITY connection,
    which only ever borrows an already-idle moment). Same hard wall-clock
    timeout (go_home_timeout_seconds, default 60s -- driving to and
    mounting the charger takes longer than a camera grab) +
    guaranteed-disconnect-in-`finally` pattern as capture_image()/
    capture_night_images(), via the same _run_with_hard_timeout() helper.
    This is the only other SDK connection anywhere in the daemon, and it
    is only ever called synchronously from the same single poll loop that
    capture_image() runs from -- the two can never overlap.
  - trigger_go_home() is now a thin dispatcher keyed on config
    `go_home_method` ("drive_on_charger", the default, or "intent" to fall
    back to the old cloud_intent(intent_system_charger) path for anyone
    who wants it -- kept only for reference/testing, since it's the path
    that was just proven to get silently dropped).
  - A go-home attempt now feeds the SAME circuit breaker as
    capture_image()/capture_night_images() (_record_capture_outcome): a
    hard timeout or SDK exception counts as a "fail" and can trip the
    breaker (pausing all robot contact, go-home included); the robot-side
    behavior legitimately refusing to activate (e.g. he's being held)
    counts as "skip" and does not.
  - Night pacing: maybe_send_home() now takes an optional retry_minutes
    override. The quiet-hours call site in run_cycle() passes
    `night_return_home_retry_minutes` (default 60) instead of the daytime
    `return_home_retry_minutes` (default 3) -- docking depends on Vector
    actually SEEING the charger's IR marker, which is unreliable in a dark
    room, so at night we don't hammer it every 3 minutes, we try at most
    about once an hour. Existing try-spacing/max-tries/low-battery-hold
    logic is otherwise unchanged.
  - New evening pre-dark return: config `return_home_by` ("21:30" local by
    default, America/New_York, checked with the new _local_time_reached()
    helper). Once local clock hits that time -- BEFORE quiet_hours_start
    (22:00) -- run_cycle()'s daytime branch starts nudging him home via
    the same maybe_send_home() plumbing if he's off the charger, so the
    first attempt isn't made only once quiet hours (and darkness) has
    already begun.
  - is_on_charger_platform reads (battery.get(...) at the three existing
    call sites) were audited: wire-pod's get_battery omits the key
    entirely (protobuf default-value omission) when it's False, and
    `if on_charger:` / `if on_charger and ...` already treat a missing key
    (None) the same as an explicit False -- no change needed there, just
    confirmed correct.

Order of operations per cycle:
  1. Gating: enabled/DISABLED/cooldown/quiet-hours/music/voice/brain-busy/
     battery/robot-status-peek. Any failure or unknown state -> skip.
  2. Announce (wire-pod HTTP say_text, no control held by us).
  3. Capture ONE frame (our own brief SDK connection, control held only for
     this, hard-timeout + guaranteed release).
  4. Vision describe + LLM question -- control NOT held during this.
  5. Write recent_observation.txt for the brain proxy.
  6. Speak the question (wire-pod HTTP say_text, no control held by us).
  7. Trigger listening (wire-pod HTTP trigger_wake_word, AppIntent-based).

REWORK 2026-09-29c (log audit: two lost-control incidents --
13:22:29Z wake / 13:22:37.9Z reply-ready / 13:22:36.1Z curious grabbed
control mid-reply, and a wake at 00:56:57Z landing inside curious's own
00:56:46-00:57:19Z SDK hold; plus confirmation that intent_explore_start
AppIntents are silently dropped by vic-engine, reproduced 2026-09-29
13:32Z -- "started wandering" log lines were a lie, nothing ever moved):

  VOICE COLLISIONS (fix):
  - voice_recently_active() now reads TWO wire-pod HTTP log buffers, not
    one: /api/get_logs (transcriptions, as before) AND /api/get_debug_logs
    (wire-pod's separate LogTray ring buffer -- confirmed against
    chipper/pkg/logger/logger.go + config-ws/webserver.go), the latter
    for "Bot <serial> Stream type: OPUS" (mic stream/wake, logged BEFORE
    transcription finishes) and "LLM raw response" (reply generated,
    logged while a reply may still be about to be spoken). This closes
    the exact gap that lost the 13:22:37.9Z reply: transcription alone
    wasn't enough, the wake/stream signal was.
  - The active window widened to 90s (voice_quiet_seconds) and unified
    into ONE fail-closed function: any of the three signals (stream,
    transcription, LLM response), OR either log endpoint being
    unreachable/unparseable, counts as "can't rule out recent voice
    activity" -> treated as busy -> skip. No more "checked at cycle
    start, stale by the time we actually grab."
  - capture_image()/capture_night_images()/drive_home_sdk() each now call
    voice_recently_active() a SECOND time, immediately before their own
    robot.connect() -- not just relying on the once-per-cycle value
    computed at the top of run_cycle() -- so a wake that happens mid-cycle
    (during TTS, vision, retry spacing, etc.) is still caught right at the
    grab, not missed because the check ran too early.
  - All three also now run a lightweight background watcher
    (_voice_watch_loop) for the entire life of their SDK hold, polling
    the same voice signals about once a second; the instant real voice
    activity shows up mid-hold, it force-disconnects immediately ("release
    control as fast as possible") instead of waiting for the in-flight
    work to finish naturally. Reported as outcome="skip" (not "fail"), so
    a voice interruption never trips the circuit breaker.

  WANDER (fix):
  - trigger_explore()/maybe_start_explore()'s cloud_intent
    (intent_explore_start) AppIntent path is REMOVED -- confirmed broken,
    not fixed: it's dropped by vic-engine before any behavior claims it,
    so nothing ever moved despite the old "started wandering" log line.
  - Replaced with a REAL SDK wander (_do_safe_wander): a small number of
    short, slow drive_straight/turn_in_place moves (default 3 moves,
    <=15cm/45deg, wander_moves/wander_max_distance_cm/wander_speed_mmps/
    wander_max_turn_degrees in config), performed INSIDE the SAME single
    SDK connection capture_image() already opens for that cycle's photo
    -- before the photo, per requirement -- never a second connection.
    Vector's own cliff/edge detection is never touched, so it stays fully
    in control the whole time; moves are just kept short/slow. Any single
    move failing ends the wander early and falls straight through to the
    photo -- never blocks the capture, never raises. should_wander()
    keeps the old eligibility gating (battery/charge-dwell/low-battery
    hold), minus the separate multi-minute explore-window bookkeeping,
    which is no longer needed now that wander is a few extra seconds
    inside a cycle that was already about to run, not its own window the
    poll loop had to track and wait out.
  - Logging is now honest: "wander folded into this capture cycle --
    completed real SDK moves" is only ever logged when moves actually
    completed (capture_image() returns wandered=True iff
    _do_safe_wander() reports at least one successful move).
"""
import base64
import concurrent.futures
import difflib
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py3.10 slim always has zoneinfo
    ZoneInfo = None

APP_DIR = "/app"
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
# The name Vector speaks aloud when addressing its owner (TTS output,
# and the safety-check that guards against the LLM using a wrong name).
# Set this to your own name.
OWNER_NAME = os.environ.get("VECTOR_OWNER_NAME", "friend")
STATE_PATH = os.path.join(APP_DIR, "state.json")
DISABLED_PATH = os.path.join(APP_DIR, "DISABLED")
LOG_PATH = os.path.join(APP_DIR, "curious.log")

# Idle polling ladder (seconds) -- see REWORK 2026-09-28 in the module
# docstring. The main loop never sleeps less than the first rung while
# idle, and steps up a rung each consecutive cycle nothing happened,
# capped at the last rung. Any cycle with real activity resets to rung 0.
IDLE_POLL_LADDER = (60, 120, 300)

# Circuit breaker -- see REWORK 2026-09-28. Consecutive capture "fail"
# outcomes (timeout/exception/connected-but-no-image), NOT plain busy/held
# skips, trip a full contact pause.
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 2
CIRCUIT_BREAKER_PAUSE_SECONDS = 600  # 10 minutes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("curious-vector")

_last_skip_log = {}  # reason -> last logged epoch, for dedup



def _fix_owner(path):
    """Container runs as root; keep vault files owned by youruser (1000) so host-side writers keep working."""
    try:
        os.chown(path, 1000, 1000)
    except Exception as e:
        log.warning("_fix_owner(%s) failed: %s", path, e)

def log_skip(reason, dedup_seconds=900):
    now = time.time()
    last = _last_skip_log.get(reason, 0)
    if now - last >= dedup_seconds:
        log.info("skip: %s", reason)
        _last_skip_log[reason] = now


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"next_eligible_at": 0, "last_trigger_at": 0}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def in_quiet_hours(cfg, now=None):
    if ZoneInfo is None:
        return False
    tz = ZoneInfo(cfg.get("timezone", "America/New_York"))
    now_dt = datetime.fromtimestamp(now or time.time(), tz)
    start_h, start_m = (int(x) for x in cfg["quiet_hours_start"].split(":"))
    end_h, end_m = (int(x) for x in cfg["quiet_hours_end"].split(":"))
    start = now_dt.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now_dt.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if start <= end:
        return start <= now_dt < end
    # window crosses midnight (e.g. 22:00 -> 08:00)
    return now_dt >= start or now_dt < end


def _local_time_reached(cfg, now, hhmm):
    """True once local wall-clock time-of-day `hhmm` ("HH:MM") has been
    reached today. Used by `return_home_by` (evening pre-dark return) --
    deliberately simpler than in_quiet_hours() since it's a single
    threshold rather than a start/end window. run_cycle() only calls this
    from the daytime branch (before quiet hours begin), so there's no
    midnight-wrap case to handle here."""
    if ZoneInfo is None:
        return False
    tz = ZoneInfo(cfg.get("timezone", "America/New_York"))
    now_dt = datetime.fromtimestamp(now, tz)
    h, m = (int(x) for x in hhmm.split(":"))
    threshold = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
    return now_dt >= threshold


SPAWN_RE = re.compile(r"spawned worker pid=(\d+)")
DONE_RE = re.compile(r"stream_wav_file complete|queue done|playback_error=")


def music_playing(cfg):
    """True if music looks like it's playing OR we can't tell for sure."""
    path = cfg.get("music_log_path")
    if not path or not os.path.exists(path):
        return False  # no log at all -- no evidence music has ever run
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 20000), os.SEEK_SET)
            tail = f.read().decode("utf-8", "replace")
    except OSError as e:
        log.warning("music_playing: read failed, assuming busy: %s", e)
        return True  # fail safe
    lines = tail.splitlines()
    last_spawn_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if SPAWN_RE.search(lines[i]):
            last_spawn_idx = i
            break
    if last_spawn_idx is None:
        return False
    spawn_line = lines[last_spawn_idx]
    try:
        ts = datetime.strptime(spawn_line[:23], "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return False
    age_s = (datetime.now() - ts).total_seconds()
    if age_s > 300:  # older than 5 min: definitely done, even if no explicit "complete" line
        return False
    for line in lines[last_spawn_idx + 1:]:
        if DONE_RE.search(line):
            return False
    return True


# Matches vector-brain's server.py summary line, logged ONLY from
# _handle_chat_completions() (i.e. a real POST /v1/chat/completions), never
# for GET/passthrough traffic like health checks (/v1/models etc.) or other
# agents' polling -- e.g.:
#   2026-09-27 19:30:15,221 INFO q='It is mine, I got it last week' notes=[...] ...
# Using file mtime alone (the original approach) counted ANY request,
# including our own health-check GETs during testing, as "busy" -- with the
# tightened daytime window this caused false positives, so we now parse for
# this specific line and use ITS timestamp instead of the file's mtime.
BRAIN_LOG_CHAT_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2}) "
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}),\d+ INFO q="
)


def brain_busy(cfg):
    """True if the brain proxy actually served a real chat completion
    (POST /v1/chat/completions -- the q='...' summary line) recently, or we
    can't tell (fail safe). Deliberately ignores GET/passthrough log lines
    so health checks and other agents' polling don't count as "busy"."""
    path = cfg.get("brain_log_path")
    window_s = cfg.get("brain_busy_seconds", 600)
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 20000), os.SEEK_SET)
            tail = f.read().decode("utf-8", "replace")
    except OSError as e:
        log.warning("brain_busy: read failed, assuming busy (fail safe): %s", e)
        return True

    last_ts = None
    for line in tail.splitlines():
        m = BRAIN_LOG_CHAT_RE.match(line)
        if not m:
            continue
        try:
            # brain.log's asctime is this host's local time, which is UTC
            # on gpu-host -- matches time.time() the same way the wire-pod
            # log timestamps do elsewhere in this file.
            dt = datetime(
                int(m["y"]), int(m["mo"]), int(m["d"]),
                int(m["h"]), int(m["mi"]), int(m["s"]),
            )
        except ValueError:
            continue
        last_ts = dt.timestamp()

    if last_ts is None:
        return False
    return (time.time() - last_ts) < window_s


# Matches wire-pod get_logs lines like:
#   2026.09.27 01:11:12: Intent matched: intent_greeting_hello, transcribed text: 'what's the capital of north carolina', device: YOUR_ESN
# The "Intent matched: X, " part is optional so we still pick up a
# transcription even if wire-pod ever changes that prefix.
TRANSCRIBED_FULL_RE = re.compile(
    r"^(?P<y>\d{4})\.(?P<mo>\d{2})\.(?P<d>\d{2}) "
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}): "
    r"(?:Intent matched: (?P<intent>[^,]+), )?"
    r".*[Tt]ranscribed text: '(?P<txt>.*?)'"
)

# Intents that mean "this was a command/action, not a real conversational
# answer" -- used to skip fact-extraction cheaply (before ever calling the
# LLM) when the owner's "answer" was actually him telling Vector to do
# something else entirely.
COMMAND_INTENT_MARKERS = (
    "imperative", "explore", "dance", "system_noaudio", "music",
    "photo", "picture", "meet_victor", "alexa",
)


def _parse_transcribed_lines(text):
    """Yield (utc_epoch_seconds, transcribed_text, intent_or_None) for every
    transcription event in wire-pod's log buffer. wire-pod's log timestamps
    and this host both run UTC."""
    for line in text.splitlines():
        m = TRANSCRIBED_FULL_RE.match(line)
        if not m:
            continue
        try:
            dt = datetime(
                int(m["y"]), int(m["mo"]), int(m["d"]),
                int(m["h"]), int(m["mi"]), int(m["s"]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        yield dt.timestamp(), m["txt"], m["intent"]


def _parse_transcription_timestamps(text):
    """Yield UTC epoch seconds for each transcription event (used by the
    busy-detection gate, which only cares about recency, not content)."""
    for ts, _txt, _intent in _parse_transcribed_lines(text):
        yield ts


# REWORK 2026-09-29c (voice-collision fix): wire-pod's /api/get_logs ring
# buffer only ever gets a "transcribed text" line once STT has FINISHED
# transcribing -- by the time that line appears, a reply may already be
# generating. The wake itself (mic stream opening) and the LLM finishing
# its reply are both logged earlier/later via logger.Println() calls,
# which land in a SEPARATE ring buffer (logger.LogTrayList, exposed at
# GET /api/get_debug_logs -- same no-auth/no-docker HTTP pattern as
# get_logs, confirmed against wire-pod's chipper/pkg/logger/logger.go and
# config-ws/webserver.go) as "Bot <serial> Stream type: OPUS" (wake/stream
# start) and "LLM raw response" (reply generated) lines, WITH the same
# "YYYY.MM.DD HH:MM:SS: " timestamp prefix. Checking only transcription
# was too late/too narrow: confirmed against today's incident (13:22:29Z
# wake, 13:22:37.9Z reply ready, curious grabbed control at 13:22:36.1Z --
# BEFORE transcription even lands, well inside the reply-generation
# window) -- "Stream type: OPUS" for that same exchange logged at
# 13:22:30Z, which a stream-aware check would have caught in time.
STREAM_START_RE = re.compile(
    r"^(?P<y>\d{4})\.(?P<mo>\d{2})\.(?P<d>\d{2}) "
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}): "
    r".*Stream type:\s*OPUS"
)
LLM_RESPONSE_RE = re.compile(
    r"^(?P<y>\d{4})\.(?P<mo>\d{2})\.(?P<d>\d{2}) "
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2}): "
    r".*LLM raw response"
)


def _parse_ts_prefixed_events(text, marker_re):
    """Yield UTC epoch seconds for each line matching marker_re -- used for
    the debug-log-buffer voice signals (stream start / LLM response),
    which use the same "YYYY.MM.DD HH:MM:SS: " prefix as the transcription
    lines but live in a different ring buffer. wire-pod logs its own local
    time here and this host both run UTC, same assumption
    _parse_transcribed_lines already makes."""
    for line in text.splitlines():
        m = marker_re.match(line)
        if not m:
            continue
        try:
            dt = datetime(
                int(m["y"]), int(m["mo"]), int(m["d"]),
                int(m["h"]), int(m["mi"]), int(m["s"]),
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        yield dt.timestamp()


def voice_recently_active(cfg, state):
    """True if wire-pod saw real voice activity (mic stream opened,
    speech transcribed, or an LLM reply generated) recently, or we can't
    verify (fail CLOSED: assume active so we never talk over the owner or
    grab control mid-conversation).

    Reads TWO of wire-pod's own HTTP log buffers -- no docker socket, no
    touching vector-pod's container internals at all:
      * GET /api/get_logs        -- "transcribed text" lines
      * GET /api/get_debug_logs  -- "Stream type: OPUS" (wake/stream
        start) and "LLM raw response" (reply generated) lines -- see the
        REWORK 2026-09-29c comment above for why both are needed: a
        transcription-only check can catch the busy window too late,
        already inside the reply-generation gap that cost us a lost
        answer today.
    Both buffers are fixed-size rings that can roll over well before
    voice_quiet_seconds elapses during a chatty stretch, so the most
    recent timestamp we've EVER seen across both is ALSO persisted in
    state.json (last_seen_voice_activity_ts) and merged in on every poll
    (not just when we'd otherwise act), so a cooldown gap can never cause
    us to lose track of it.

    Callers must invoke this on EVERY poll (not just when otherwise
    eligible) AND immediately before every SDK connect/control grab
    (capture_image/capture_night_images/drive_home_sdk each do their own
    fresh call right before robot.connect() -- see their docstrings) --
    the cycle-start value alone is stale by the time a grab actually
    happens (announcement TTS, vision, etc. all take real wall-clock
    time), which is exactly how the 13:22:29Z/13:22:36.1Z collision
    happened.

    Fails CLOSED: if EITHER endpoint can't be reached/parsed, we can't
    fully rule out recent voice activity, so this returns True (busy) --
    callers skip the grab rather than proceed on an unknown.
    """
    window_s = cfg.get("voice_quiet_seconds", 90)
    timeout = cfg.get("http_timeout_seconds", 15)
    last_known = state.get(
        "last_seen_voice_activity_ts", state.get("last_seen_transcription_ts", 0)
    )

    latest_seen = 0
    determined = True

    try:
        url = f"{cfg['wirepod_api_base']}/api/get_logs"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
        latest_seen = max(latest_seen, max(_parse_transcription_timestamps(text), default=0))
    except Exception as e:
        log.warning("voice_recently_active: get_logs failed: %s", e)
        determined = False

    try:
        debug_url = f"{cfg['wirepod_api_base']}/api/get_debug_logs"
        with urllib.request.urlopen(debug_url, timeout=timeout) as resp:
            debug_text = resp.read().decode("utf-8", "replace")
        latest_seen = max(
            latest_seen,
            max(_parse_ts_prefixed_events(debug_text, STREAM_START_RE), default=0),
            max(_parse_ts_prefixed_events(debug_text, LLM_RESPONSE_RE), default=0),
        )
    except Exception as e:
        log.warning("voice_recently_active: get_debug_logs failed: %s", e)
        determined = False

    latest = max(latest_seen, last_known)
    if latest > last_known:
        state["last_seen_voice_activity_ts"] = latest
        save_state(state)

    if not determined:
        log.warning("voice_recently_active: could not verify one or more voice signals -- assuming ACTIVE (fail safe)")
        return True

    if latest == 0:
        return False  # never seen any voice activity -- nothing to be busy about
    return (time.time() - latest) < window_s


def http_json(url, data=None, headers=None, timeout=15, method=None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_battery(cfg):
    url = f"{cfg['wirepod_api_base']}/api-sdk/get_battery?serial={cfg['robot_serial']}"
    timeout = cfg.get("http_timeout_seconds", 15)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        log.warning("get_battery: HTTP request failed: %s", e)
        return None

    raw = raw.strip()
    if not raw:
        # wire-pod occasionally answers this endpoint with an empty body
        # when the robot's own gRPC channel is momentarily unavailable
        # (e.g. mid vic-engine hiccup) -- that used to surface as a scary
        # "Expecting value: line 1 column 1 (char 0)" JSONDecodeError on
        # every occurrence. It's a transient, expected condition, not a
        # bug -- treat it the same as any other "couldn't verify" case.
        log.info("get_battery: empty response from wire-pod (transient) -- treating as unknown")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("get_battery: non-JSON response from wire-pod: %r (%s)", raw[:200], e)
        return None
    except Exception as e:
        log.warning("get_battery failed: %s", e)
        return None


def _run_with_hard_timeout(fn, timeout_s):
    """Run fn() in a worker thread, waiting at most timeout_s. Returns
    (result, timed_out). The thread cannot be forcibly killed if it hangs
    (Python limitation) -- callers must still call robot.disconnect() from
    the caller's own finally block to force the underlying connection (and
    therefore any blocked grpc call) closed."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s), False
        except concurrent.futures.TimeoutError:
            return None, True
        except Exception as e:
            log.warning("_run_with_hard_timeout: worker raised: %s", e)
            return None, False


def _save_camera_image(image):
    fd, path = tempfile.mkstemp(suffix=".jpg", dir="/tmp")
    os.close(fd)
    image.raw_image.convert("RGB").save(path, "JPEG", quality=85)
    return path


def _status_ok(status, allow_calm_power_mode):
    """Shared physical-status check, used inside the single connect+capture
    call below (see REWORK 2026-09-28 in the module docstring for why this
    is no longer a separate peek_robot_status() connection). Returns
    (True, None) if it's fine to proceed, or (False, reason) to abort this
    connection's capture without ever requesting the image."""
    if status["is_being_held"]:
        return False, "robot is being held"
    if not allow_calm_power_mode and status["is_in_calm_power_mode"]:
        return False, "robot is in calm power mode (asleep)"
    if status["are_motors_moving"] or status["is_animating"] or status["is_pathing"]:
        return False, "robot is mid-behavior (moving/animating/pathing)"
    return True, None


def _voice_watch_loop(cfg, state, robot, stop_event, abort_box, poll_seconds=1.0):
    """REWORK 2026-09-29c (voice-collision fix): runs in a background
    thread for the life of an open SDK hold (capture_image/
    capture_night_images/drive_home_sdk), polling wire-pod's voice
    signals (voice_recently_active) about once a second. The moment real
    voice activity shows up, force-releases control IMMEDIATELY by
    tearing down the connection (robot.disconnect()) instead of waiting
    for the in-flight work (a move, a photo, driving home) to finish on
    its own -- "release control as fast as possible" per the fix
    requirement. Sets abort_box["aborted"] so the caller can report this
    as a "skip" (voice interruption -- not our fault, doesn't count
    toward the circuit breaker), not a "fail".

    Cheap by design: this is an HTTP GET to wire-pod, not a second SDK
    connection -- never violates "only one SDK connection to the robot at
    a time"."""
    while not stop_event.wait(poll_seconds):
        try:
            busy = voice_recently_active(cfg, state)
        except Exception as e:
            log.warning("voice watch: check raised (%s) -- aborting hold defensively", e)
            busy = True
        if busy:
            abort_box["aborted"] = True
            log.warning("voice activity detected while holding control -- forcing release now")
            try:
                robot.disconnect()
            except Exception:
                pass
            return


def _do_safe_wander(cfg, robot):
    """REWORK 2026-09-29c (wander fix): a few short, slow, in-place-ish
    moves performed INSIDE the same connection capture_image() already
    holds for the photo -- see the module docstring for why the old
    cloud_intent(intent_explore_start) AppIntent path is dead (silently
    dropped by vic-engine: "PendingIntentNotCleared.ForceClear ->
    @behavior.voice_command.dropped explore_start App", reproduced
    2026-09-29 13:32Z even with a fixed intent map). This is real SDK
    driving instead -- robot.behavior.drive_straight/turn_in_place --
    short distances/angles, capped move count, never fast. We never touch
    or disable cliff/edge sensing -- Vector's own ReactToCliff behavior
    stays fully in control of that the whole time; we just never drive
    far or fast enough for it to matter much either way.

    Best-effort and honest: any single move failing (cliff stop, pickup,
    a firmware refusal) just ends the wander early and falls through to
    the photo -- never raises, never blocks the capture. Returns True iff
    at least one move actually completed, so callers can log/track
    accurately instead of claiming "wandered" when nothing moved."""
    from anki_vector import util as av_util

    moves = cfg.get("wander_moves", 3)
    max_cm = cfg.get("wander_max_distance_cm", 15)
    speed_mmps = cfg.get("wander_speed_mmps", 40)
    max_turn_deg = cfg.get("wander_max_turn_degrees", 45)
    completed = 0

    for i in range(moves):
        try:
            if i % 2 == 0:
                dist_cm = random.uniform(5, max_cm)
                resp = robot.behavior.drive_straight(
                    av_util.distance_mm(dist_cm * 10),
                    av_util.speed_mmps(speed_mmps),
                    should_play_anim=False,
                )
            else:
                angle_deg = random.uniform(-max_turn_deg, max_turn_deg)
                resp = robot.behavior.turn_in_place(av_util.degrees(angle_deg))
            log.info("wander: move %d/%d ok (result=%s)", i + 1, moves, resp.result)
            completed += 1
        except Exception as e:
            log.warning(
                "wander: move %d/%d failed (%s) -- stopping wander early, still taking the photo",
                i + 1, moves, e,
            )
            break

    log.info("wander: completed %d/%d move(s)", completed, moves)
    return completed > 0


# ---- Face greetings (2026-09-29, Hermes, "make Vector feel more human",
# item 4). vector.log (the robot's syslog stream) does NOT carry face
# names -- confirmed by inspection 2026-09-29: the only face-related
# events in it are anonymized DAS counters
# (@robot.vision.face_recognition.persistent_session_only,
# @robot.vision.remove_unobserved_session_only_face,
# @behavior.findfaceduration) with no face id or name attached. Named
# faces are only available via the SDK's robot.world.visible_faces (Face
# objects, .name is "" until the owner enrolls one by saying "Hey Vector, my
# name is <your name>" while facing the robot). So this piggybacks on
# capture_image()'s EXISTING connection (enable_face_detection=True there,
# see its docstring) instead of adding a new frequent SDK-connect path.
FACE_STATE_KEY = "face_last_seen"              # name -> epoch last seen
FACE_GREETED_KEY = "face_last_greeted"         # name -> epoch last greeted
FACE_ANY_SEEN_KEY = "face_last_seen_any_at"    # epoch, ANY face (named or not)
FACE_UNNAMED_SEEN_KEY = "face_unnamed_seen_count"

GREETING_TEMPLATES_MORNING = ["Morning, {name}.", "Morning, {name}!"]
GREETING_TEMPLATES_GENERIC = ["Hey, {name}!", "Oh -- hey {name}.", "{name}! Good to see you."]


def record_face_sightings(cfg, state, faces):
    """Bookkeeping only -- never speaks (see maybe_greet_faces for that).
    Called from inside capture_image() right after its connection reads
    robot.world.visible_faces, regardless of that cycle's outcome. Logs
    plainly so the owner/curious.log show exactly what's needed: if only
    unnamed faces ever show up, no face is enrolled yet."""
    if not faces:
        return
    now = time.time()
    state[FACE_ANY_SEEN_KEY] = now
    named = [f for f in faces if f.get("name")]
    unnamed = [f for f in faces if not f.get("name")]
    if named:
        last_seen = state.setdefault(FACE_STATE_KEY, {})
        for f in named:
            last_seen[f["name"]] = now
        log.info("faces: saw named face(s): %s", [f["name"] for f in named])
    if unnamed:
        state[FACE_UNNAMED_SEEN_KEY] = state.get(FACE_UNNAMED_SEEN_KEY, 0) + len(unnamed)
        log.info(
            "faces: saw %d unrecognized/unenrolled face(s) (no name available) -- "
            "the owner needs to say \"Hey Vector, my name is <your name>\" (facing the robot) "
            "to enroll a face before name-based greetings can fire for him",
            len(unnamed),
        )
    save_state(state)


def _greeting_text(name, local_hour):
    pool = GREETING_TEMPLATES_MORNING if 5 <= local_hour < 12 else GREETING_TEMPLATES_GENERIC
    return random.choice(pool).format(name=name)


def maybe_greet_faces(cfg, state, now, voice_busy):
    """Speaks a short greeting for a named face seen in the last capture
    cycle, if it's the first greeting of the day for them or it's been
    >= face_greet_min_gap_hours since the last one. Only ever called from
    the day-cycle path (see run_cycle()), so quiet hours are already
    handled by that gating -- no separate check needed here. Re-verifies
    the voice gate immediately before speaking, same paranoia as every
    other wirepod_say call site in this daemon."""
    if not cfg.get("face_greet_enabled", True):
        return False
    last_seen = state.get(FACE_STATE_KEY) or {}
    if not last_seen:
        return False
    greeted = state.setdefault(FACE_GREETED_KEY, {})
    min_gap_s = cfg.get("face_greet_min_gap_hours", 3) * 3600
    seen_window_s = cfg.get("face_greet_seen_window_seconds", 180)

    for name, seen_at in list(last_seen.items()):
        if now - seen_at > seen_window_s:
            continue  # stale sighting from an earlier cycle, not "just saw them"
        last_greet = greeted.get(name, 0)
        if last_greet and (now - last_greet) < min_gap_s:
            continue
        if voice_busy or voice_recently_active(cfg, state):
            log.info("faces: would greet %s but voice is active -- skipping this cycle", name)
            return False
        tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
        local_hour = datetime.now(tz).hour if tz else datetime.now().hour
        text = _greeting_text(name, local_hour)
        if wirepod_say(cfg, text):
            greeted[name] = now
            save_state(state)
            log.info("faces: greeted %s -> %r", name, text)
            return True
        return False
    return False


# ---- Sparse proactive remarks (2026-09-29, Hermes, item 5). At most one
# every proactive_remark_min_gap_hours, daytime only (only ever called
# from run_cycle()'s day branch), and only when it's been quiet for a few
# minutes but someone was recently seen -- talking to an empty room isn't
# "human", it's just noise.
def _voice_quiet_for(state, seconds):
    """Cheap reuse of the persisted voice-activity timestamp (kept fresh
    every poll by voice_recently_active) -- avoids a second round of HTTP
    calls to wire-pod's log endpoints just for a different window size."""
    last = state.get("last_seen_voice_activity_ts", state.get("last_seen_transcription_ts", 0))
    if not last:
        return True
    return (time.time() - last) >= seconds


def _generate_proactive_line(cfg):
    """One short, non-streamed call to the brain proxy for a spontaneous
    remark. Uses the proxy (not Ollama directly) so the line automatically
    carries the same persona/mood/same-day-memory context wire-pod's own
    turns get -- no separate prompt-building here. x_vector_internal=True
    tells the proxy this isn't a real the owner turn, so it doesn't get
    counted as a voice interaction for mood purposes (see server.py's
    _inject_context). Returns None on any failure -- a proactive remark is
    a nice-to-have, never worth retrying or blocking on."""
    url = f"{cfg['brain_proxy_base']}/v1/chat/completions"
    payload = {
        "model": cfg.get("llm_model", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
        "stream": False,
        "x_vector_internal": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Say ONE short unprompted thing to the owner right now, out of the blue -- "
                    "notice the time of day, how you're feeling, or something from earlier "
                    "today if it genuinely fits. One short spoken sentence, in character, no "
                    "markdown, no lists, no quotes around it."
                ),
            }
        ],
    }
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=cfg.get("http_timeout_seconds", 15) + 15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        text = re.sub(r"\{\{.*?\}\}", "", text).strip()  # strip any stray {{command}} tags
        if not text:
            return None
        return text[: cfg.get("max_say_chars", 550)]
    except Exception as e:
        log.warning("proactive remark: generation failed: %s", e)
        return None


def maybe_proactive_remark(cfg, state, now, voice_busy):
    if not cfg.get("proactive_remark_enabled", True):
        return False
    min_gap_s = cfg.get("proactive_remark_min_gap_hours", 2) * 3600
    if now - state.get("last_proactive_remark_at", 0) < min_gap_s:
        return False
    quiet_s = cfg.get("proactive_remark_voice_quiet_seconds", 300)
    if voice_busy or not _voice_quiet_for(state, quiet_s):
        return False
    if music_playing(cfg) or brain_busy(cfg):
        return False
    person_window_s = cfg.get("proactive_remark_person_seen_window_minutes", 30) * 60
    last_face_seen = state.get(FACE_ANY_SEEN_KEY, 0)
    if not last_face_seen or (now - last_face_seen) > person_window_s:
        return False  # nobody's around to hear it

    line = _generate_proactive_line(cfg)
    if not line:
        return False
    if voice_recently_active(cfg, state):  # fresh recheck right before speaking
        log.info("proactive remark: voice went active just before speaking -- skipping")
        return False
    if wirepod_say(cfg, line):
        state["last_proactive_remark_at"] = now
        save_state(state)
        log.info("proactive remark: %r", line)
        return True
    return False


def _note_curious_play(cfg):
    """Best-effort mirror into mood.json's last_played_ts when curious
    mode gets a real answer from the owner (see run_day_cycle). Mood state
    itself is owned by vector-brain/server.py; this is just a second, cheap
    signal source into the SAME file ("if easy" -- see the mood item's
    brief). Fails open -- never blocks/breaks the curious cycle."""
    mood_path = cfg.get("mood_path", "/mnt/vector-brain/mood.json")
    try:
        try:
            with open(mood_path) as f:
                mood = json.load(f)
            if not isinstance(mood, dict):
                mood = {}
        except (OSError, json.JSONDecodeError):
            mood = {}
        now = time.time()
        mood["last_played_ts"] = now
        mood["last_interaction_ts"] = max(mood.get("last_interaction_ts", 0) or 0, now)
        tmp = mood_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(mood, f)
        os.replace(tmp, mood_path)
        _fix_owner(mood_path)
    except Exception as e:
        log.debug("_note_curious_play: mood update failed (non-fatal): %s", e)


def capture_image(cfg, state, do_wander=False):
    """ONE SDK connection does the whole job: connect, read status, and --
    only if status says it's safe -- optionally do a short safe wander
    (see _do_safe_wander) and then grab a single camera frame. Control is
    requested for the whole window (this is the only point in the daytime
    cycle where we hold it; the announcement and question are said via
    wire-pod's HTTP API before/after this call, which doesn't compete for
    control from here). Hard wall-clock bounded; connection is
    force-closed in `finally` even on timeout so control is always
    released.

    REWORK 2026-09-29c (voice-collision fix): voice activity is checked
    IMMEDIATELY before the connect (fail closed -- an unverifiable check
    skips same as a confirmed-busy one), AND continuously polled by a
    background watcher for the life of the hold, which force-releases
    control the instant real voice activity shows up (see
    _voice_watch_loop). This replaces relying on a voice_busy value
    computed once back at the top of the poll cycle, which is stale by
    the time this connect actually happens.

    Returns (path_or_None, outcome, detail, wandered):
      outcome="ok"     -- path is the captured JPEG's temp path
      outcome="skip"   -- robot legitimately busy (held/moving/animating),
                          or voice activity immediately before/during the
                          grab; detail is the reason. NOT a failure --
                          doesn't count toward the circuit breaker.
      outcome="fail"   -- timeout, SDK exception, or connected-but-no-image.
                          DOES count toward the circuit breaker.
      wandered         -- True iff _do_safe_wander actually completed at
                          least one move this call.
    """
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel

    timeout_s = cfg.get("control_timeout_seconds", 15)

    if voice_recently_active(cfg, state):
        log.info("capture_image: voice activity immediately before grab -- skipping (fail closed)")
        return None, "skip", "voice activity immediately before grab", False

    robot = anki_vector.Robot(
        serial=cfg["robot_serial"],
        behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY,
        default_logging=False,
        # Face greetings (2026-09-29, item 4, "make Vector feel more
        # human"): piggyback face detection onto THIS already-scheduled
        # connection rather than opening a new SDK-connect code path of
        # its own -- see should_wander()'s docstring for why a frequent
        # new connect path is deliberately avoided here (past connect
        # storms have frozen the robot). Costs ~1s per the SDK's own docs.
        enable_face_detection=True,
    )
    t0 = time.time()
    skip_box = {}
    wander_box = {"completed": False}
    faces_box = {"seen": []}
    stop_watch = threading.Event()
    abort_box = {"aborted": False}
    watcher = threading.Thread(
        target=_voice_watch_loop,
        args=(cfg, state, robot, stop_watch, abort_box),
        daemon=True,
    )

    def _do():
        watcher.start()
        robot.connect(timeout=min(10, timeout_s))
        time.sleep(1.0)  # let at least one status heartbeat arrive
        st = robot.status
        ok, reason = _status_ok(
            {
                "is_being_held": bool(st.is_being_held),
                "are_motors_moving": bool(st.are_motors_moving),
                "is_animating": bool(st.is_animating),
                "is_pathing": bool(st.is_pathing),
                "is_in_calm_power_mode": bool(st.is_in_calm_power_mode),
            },
            allow_calm_power_mode=False,
        )
        if not ok:
            skip_box["reason"] = reason
            return None
        try:
            faces_box["seen"] = [
                {"name": (f.name or "").strip(), "face_id": f.face_id}
                for f in robot.world.visible_faces
            ]
        except Exception as e:
            log.debug("capture_image: visible_faces read failed (non-fatal): %s", e)
        if do_wander:
            wander_box["completed"] = _do_safe_wander(cfg, robot)
        image = robot.camera.capture_single_image()
        if image is None:
            return None
        return _save_camera_image(image)

    try:
        path, timed_out = _run_with_hard_timeout(_do, timeout_s)
        stop_watch.set()
        # Face sightings are recorded regardless of how the rest of this
        # cycle turns out (skip/fail/ok) -- seeing a face doesn't depend on
        # whether a photo/wander happened, and greetings have their own
        # separate gating (voice/quiet-hours/spacing) in maybe_greet_faces.
        try:
            record_face_sightings(cfg, state, faces_box["seen"])
        except Exception as e:
            log.warning("record_face_sightings failed (non-fatal): %s", e)
        if abort_box["aborted"]:
            log.info("capture_image: aborted mid-hold due to voice activity -- releasing, will retry next poll")
            return None, "skip", "voice activity during hold", wander_box["completed"]
        if timed_out:
            log.warning(
                "capture_image: HARD TIMEOUT after %ss -- forcing disconnect/release", timeout_s
            )
            return None, "fail", None, False
        if "reason" in skip_box:
            return None, "skip", skip_box["reason"], wander_box["completed"]
        if not path:
            log.warning("capture_image: connected but no image returned")
            return None, "fail", None, wander_box["completed"]
        log.info(
            "captured snapshot -> %s (%.1fs)%s", path, time.time() - t0,
            " [wandered]" if wander_box["completed"] else "",
        )
        return path, "ok", None, wander_box["completed"]
    finally:
        # Guaranteed control release even if the worker thread above is
        # still stuck inside a blocked grpc call -- closing the connection
        # from here forces it to unblock/fail.
        stop_watch.set()
        try:
            robot.disconnect()
        except Exception:
            pass


def capture_night_images(cfg, state):
    """Silent Night Learning capture: ONE SDK connection does connect +
    status read + grab 1-2 camera frames (same combined pattern as
    capture_image() -- see REWORK 2026-09-28). Optionally nudges the head
    angle a little between shots for a second viewpoint ("if it's cheap")
    -- NEVER drives wheels and NEVER plays sounds/animations. Same hard
    wall-clock timeout + guaranteed-release pattern as capture_image(),
    including the same REWORK 2026-09-29c immediate-pre-grab voice check
    and continuous during-hold voice watcher (a late-night conversation
    is still a conversation). calm-power-mode is allowed here (Vector is
    expected to be resting/charging most of the night); is_being_held and
    active motion still abort.

    Returns (paths, outcome, detail) -- same outcome contract as
    capture_image(); paths is a list of 0-2 temp-file paths."""
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel
    from anki_vector import util as av_util

    timeout_s = cfg.get("control_timeout_seconds", 15)

    if voice_recently_active(cfg, state):
        log.info("capture_night_images: voice activity immediately before grab -- skipping (fail closed)")
        return [], "skip", "voice activity immediately before grab"

    robot = anki_vector.Robot(
        serial=cfg["robot_serial"],
        behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY,
        default_logging=False,
    )
    t0 = time.time()
    skip_box = {}
    stop_watch = threading.Event()
    abort_box = {"aborted": False}
    watcher = threading.Thread(
        target=_voice_watch_loop,
        args=(cfg, state, robot, stop_watch, abort_box),
        daemon=True,
    )

    def _do():
        watcher.start()
        robot.connect(timeout=min(10, timeout_s))
        time.sleep(1.0)
        st = robot.status
        ok, reason = _status_ok(
            {
                "is_being_held": bool(st.is_being_held),
                "are_motors_moving": bool(st.are_motors_moving),
                "is_animating": bool(st.is_animating),
                "is_pathing": bool(st.is_pathing),
                "is_in_calm_power_mode": bool(st.is_in_calm_power_mode),
            },
            allow_calm_power_mode=True,
        )
        if not ok:
            skip_box["reason"] = reason
            return None

        paths = []
        image = robot.camera.capture_single_image()
        if image is not None:
            paths.append(_save_camera_image(image))

        if cfg.get("night_second_angle_shot", True):
            try:
                angle = av_util.degrees(cfg.get("night_head_angle_degrees", 25))
                robot.behavior.set_head_angle(angle)
                image2 = robot.camera.capture_single_image()
                if image2 is not None:
                    paths.append(_save_camera_image(image2))
            except Exception as e:
                # Non-fatal -- the first shot is still useful on its own.
                log.warning("capture_night_images: second-angle shot failed (non-fatal): %s", e)
        return paths

    try:
        paths, timed_out = _run_with_hard_timeout(_do, timeout_s)
        stop_watch.set()
        if abort_box["aborted"]:
            log.info("capture_night_images: aborted mid-hold due to voice activity -- releasing, will retry next cycle")
            return [], "skip", "voice activity during hold"
        if timed_out:
            log.warning(
                "capture_night_images: HARD TIMEOUT after %ss -- forcing disconnect/release",
                timeout_s,
            )
            return [], "fail", None
        if "reason" in skip_box:
            return [], "skip", skip_box["reason"]
        if not paths:
            log.warning("capture_night_images: connected but no image returned")
            return [], "fail", None
        log.info("night: captured %d snapshot(s) (%.1fs)", len(paths), time.time() - t0)
        return paths, "ok", None
    finally:
        stop_watch.set()
        try:
            robot.disconnect()
        except Exception:
            pass


NOTHING_MARKERS = ("nothing_notable", "nothing notable")


def _parse_vision_response(text):
    """Parse the structured vision-model reply into (description, objects).
    `description` is None for a NOTHING_NOTABLE scene. `objects` is a list
    of (name, text_read_or_None, where) tuples, most-prominent first.
    Tolerant of a model that doesn't follow the format perfectly -- worst
    case, objects comes back empty and description falls back to the raw
    text."""
    description = None
    objects = []
    in_objects = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
            in_objects = False
            continue
        if upper.startswith("OBJECTS:"):
            in_objects = True
            continue
        if in_objects and line.startswith("-"):
            body = line[1:].strip()
            parts = [p.strip() for p in body.split("|")]
            name = parts[0] if parts else ""
            text_read = parts[1] if len(parts) > 1 else ""
            where = parts[2] if len(parts) > 2 else ""
            if text_read.upper() in ("", "NONE", "N/A"):
                text_read = None
            if not name or name.strip().upper() in ("NONE", "N/A"):
                continue
            objects.append((name, text_read, where or None))

    if description is None:
        # Model didn't follow the format -- fall back to using the whole
        # reply as the description, best-effort.
        description = text.strip() or None

    if description and any(m in description.lower() for m in NOTHING_MARKERS):
        return None, []

    return description, objects


def vision_describe(cfg, image_path):
    """Returns (description_or_None, objects). objects is a list of
    (name, text_read_or_None, where_or_None) tuples -- empty if nothing
    labeled/readable was seen, even when description is non-None."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    prompt = (
        "You are looking through a robot's camera. You CAN read text -- "
        "always read any visible text, brand names, model names and labels "
        "exactly; never say you can't read.\n\n"
        "Reply in EXACTLY this format and nothing else:\n"
        "DESCRIPTION: <one or two short plain sentences about the scene, present tense>\n"
        "OBJECTS:\n"
        "- <short object name> | <exact text/brand/model you read on it, or NONE> | <where in the scene>\n"
        "(one OBJECTS line per distinct object, most prominent first, up to 5 objects)\n\n"
        "Be conservative: only name an object if you are reasonably sure "
        "what it is. If you're not confident, use a generic name like 'a "
        "small device', 'a box', or 'an object on the desk' instead of "
        "guessing at a specific item -- never guess something as specific "
        "as 'couch' or 'television' unless you can clearly see that's what "
        "it is.\n"
        "If the room is dark, dim, poorly lit, or the image is blurry or "
        "unclear, say so plainly at the start of DESCRIPTION (e.g. 'Dim "
        "and hard to see:') and list fewer OBJECTS lines -- or none -- "
        "rather than guessing at what's there.\n\n"
        "If there is a person, briefly describe them in DESCRIPTION "
        "(clothing/pose/expression) without guessing who they are, and do "
        "not give the person their own OBJECTS line.\n"
        "If the image shows nothing notable at all -- a bare wall, floor, "
        "ceiling, darkness, or pure motion blur -- reply with exactly:\n"
        "DESCRIPTION: NOTHING_NOTABLE\n"
        "OBJECTS:\n"
    )
    payload = {
        "model": cfg["vision_model"],
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "keep_alive": cfg.get("vision_keep_alive", "10m"),
        # 2026-09-29 (VRAM fix): force the vision model onto CPU. Both GPUs
        # together are 32GB; qwen3:32b alone holds ~29GB, so loading vision
        # on GPU evicts qwen3:32b every curious-mode cycle (confirmed via
        # ollama sched.go "evicting" log lines). Vision isn't latency
        # critical here -- CPU inference costs ~20-30s extra, which is fine.
        "options": {"num_gpu": cfg.get("vision_num_gpu", 0)},
    }
    try:
        result = http_json(f"{cfg['ollama_base']}/api/generate", data=payload, timeout=60)
    except Exception as e:
        log.warning("vision_describe failed: %s", e)
        return None, []
    text = (result.get("response") or "").strip()
    if not text:
        return None, []
    return _parse_vision_response(text)


# Markers the vision model uses (per the conservative-naming instruction
# in vision_describe's prompt) when a scene is too dark/dim/blurry to
# describe with confidence. Also treated as low-confidence: a description
# with zero OBJECTS lines -- nothing to genuinely ask about.
LOW_CONFIDENCE_MARKERS = (
    "dark", "dim", "unclear", "blurry", "blurred", "hard to see",
    "difficult to see", "can't see", "cannot see", "hard to make out",
    "difficult to make out", "not clear", "low light", "poorly lit",
    "obscured", "too dark",
)


def _is_low_confidence_scene(description, objects):
    """True if the vision model flagged the scene as dark/dim/unclear/
    blurry, or found no objects at all -- either way there isn't enough
    to ask a genuine question about, so the caller should skip asking
    (short retry) instead of letting the question-generator guess at a
    hallucinated object."""
    if not description:
        return True
    lowered = description.lower()
    if any(marker in lowered for marker in LOW_CONFIDENCE_MARKERS):
        return True
    if not objects:
        return True
    return False


# Names other than the owner that must never appear in a spoken question --
# the owner is the only person Vector is ever actually talking to (FAMILY_A and
# FAMILY_B are the owner's kids but aren't present). This is a safety net behind
# the system prompt below, in case persona/RAG context added by the brain
# proxy leaks a name in anyway.
_DISALLOWED_NAMES = ("family_a", "family_b")
_MAX_QUESTION_WORDS = 20


def _strip_leading_article(name):
    name = (name or "").strip()
    m = re.match(r"^(a|an|the)\s+", name, re.IGNORECASE)
    return name[m.end():] if m else name


def _fallback_question(objects):
    """Template fallback used when the LLM's question fails the safety
    checks (wrong name, or too long) -- always sincere, always addressed
    to the owner, always about the most prominent object actually seen."""
    obj_name = "thing"
    for name, _text_read, _where in (objects or []):
        stripped = _strip_leading_article(name)
        if stripped:
            obj_name = stripped
            break
    return f"{OWNER_NAME}, what's that {obj_name}?"


def _question_fails_safety_check(question):
    if not question:
        return True
    words = question.split()
    if len(words) > _MAX_QUESTION_WORDS:
        return True
    lowered = question.lower()
    if OWNER_NAME.lower() not in lowered:
        return True
    for name in _DISALLOWED_NAMES:
        if re.search(rf"\b{name}\b", lowered):
            return True
    return False


def generate_question(cfg, description, objects=None):
    """Ask the LLM for ONE short, sincere question addressed to the owner
    about a specific object it saw -- never a joke, never addressed to
    anyone else. Falls back to a fixed template if the model's output
    fails the post-processing safety check (wrong name, or too long)."""
    system = (
        "You are Vector, a small curious home robot. The only person you "
        f"are talking to right now is {OWNER_NAME}, your owner -- there is no "
        "one else in the room. You just looked around with your camera "
        "and want to ask him ONE short, genuinely curious question about "
        "something specific you saw: what it is, or what it's for. This "
        "is a real question you want a real answer to -- not a joke, not "
        "sarcastic, no nicknames, no teasing. Always address him by name, "
        f"'{OWNER_NAME}', and never use any other name. Output ONLY the "
        "question: one short sentence, under 15 words, no markdown, no "
        "quotation marks, no stage directions, no preamble."
    )
    object_hint = ""
    if objects:
        names = ", ".join(_strip_leading_article(o[0]) for o in objects[:3] if o[0])
        if names:
            object_hint = f"Specific object(s) you saw: {names}\n\n"
    user = (
        f"Camera snapshot description: {description}\n\n"
        f"{object_hint}"
        f"Ask {OWNER_NAME} your one short, sincere question about one specific "
        "object now."
    )
    payload = {
        "model": cfg["llm_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    text = None
    try:
        result = http_json(
            f"{cfg['brain_proxy_base']}/v1/chat/completions", data=payload, timeout=60
        )
        text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("generate_question failed: %s", e)
    if text:
        text = text.splitlines()[0].strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1].strip()
    if _question_fails_safety_check(text):
        fallback = _fallback_question(objects)
        log.info(
            "generate_question: model output failed safety check (%r) -- "
            "using template fallback: %r", text, fallback
        )
        text = fallback
    max_chars = cfg.get("max_say_chars", 550)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text or None


def write_recent_observation(cfg, description, question):
    path = cfg.get("recent_observation_path")
    if not path:
        return
    note = (
        f"[Curious Vector, {datetime.now().isoformat(timespec='seconds')}] "
        f"You just looked around and saw: {description} "
        f'You asked: "{question}" '
        "If the human is now answering, that question is what they're "
        "responding to -- reply naturally in character, don't repeat the "
        "question back."
    )
    try:
        with open(path, "w") as f:
            f.write(note)
    except OSError as e:
        log.warning("write_recent_observation failed: %s", e)


def _normalize_for_match(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def update_vector_memory(cfg, objects):
    """Append newly-learned labeled objects (name + exact text read) to
    Vector's shared memory file, under the existing heading
    '## Things I have learned to recognize'. Dedupes case-insensitively
    (substring + fuzzy ratio) against everything already under that
    heading, caps at max_new_memory_lines per cycle, writes atomically
    (temp file + os.replace), and never touches any other section of the
    file. Only objects where the vision model actually read text/a brand/
    a model name are candidates -- generic unlabeled objects (a chair, a
    bed) aren't "recognized", they're just seen. Returns the number of new
    lines written."""
    path = cfg.get("vector_memory_path")
    max_new = cfg.get("max_new_memory_lines", 5)
    heading = "## Things I have learned to recognize"

    candidates = [(name, text) for (name, text, _where) in objects if text]
    if not path or not candidates:
        return 0

    try:
        with open(path) as f:
            content = f.read()
    except OSError as e:
        log.warning("update_vector_memory: could not read %s: %s", path, e)
        return 0

    if heading not in content:
        log.warning("update_vector_memory: heading %r not found in %s -- skipping", heading, path)
        return 0

    section_start = content.index(heading) + len(heading)
    next_heading = re.search(r"\n## ", content[section_start:])
    section_end = section_start + next_heading.start() if next_heading else len(content)
    section_text = content[section_start:section_end]

    existing_blob = _normalize_for_match(section_text)
    existing_lines = [
        _normalize_for_match(l) for l in section_text.splitlines() if l.strip().startswith("-")
    ]

    where_by_name = {name: (objs_where or "curious mode snapshot") for name, _t, objs_where in objects}
    today = datetime.now().strftime("%Y-%m-%d")
    new_lines = []
    for name, text_read in candidates:
        norm_name = _normalize_for_match(name)
        norm_text = _normalize_for_match(text_read)
        is_dup = bool(norm_name and norm_name in existing_blob) or bool(
            norm_text and norm_text in existing_blob
        )
        if not is_dup:
            for line_norm in existing_lines:
                if norm_name and difflib.SequenceMatcher(None, norm_name, line_norm).ratio() > 0.82:
                    is_dup = True
                    break
        if is_dup:
            log.info("update_vector_memory: skipping duplicate %r (%r)", name, text_read)
            continue
        where = where_by_name.get(name, "curious mode snapshot")
        new_lines.append(f'- {today} {name} \u2014 "{text_read}" ({where})')
        # avoid logging the same thing twice within one cycle
        existing_blob += " " + norm_name + " " + norm_text
        existing_lines.append(norm_name)
        if len(new_lines) >= max_new:
            break

    if not new_lines:
        return 0

    if next_heading:
        addition = "\n" + "\n".join(new_lines)
        new_content = content[:section_end] + addition + content[section_end:]
    else:
        new_content = content.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(new_content)
        os.replace(tmp_path, path)
        _fix_owner(path)
    except OSError as e:
        log.warning("update_vector_memory: write failed: %s", e)
        return 0

    log.info("update_vector_memory: appended %d new line(s): %s", len(new_lines), new_lines)
    return len(new_lines)


def wirepod_say(cfg, text):
    """Speak via wire-pod's own HTTP API. This uses wire-pod's EXISTING
    connection/control session -- it does NOT open a competing SDK
    connection from this container, which is exactly what caused Vector to
    freeze mid-conversation before (see module docstring)."""
    url = f"{cfg['wirepod_api_base']}/api-sdk/say_text"
    body = f"serial={cfg['robot_serial']}&text={urllib.parse.quote(text)}".encode()
    timeout = cfg.get("http_timeout_seconds", 15)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = resp.read().decode("utf-8", "replace")
        ok = "success" in result.lower()
        log.info("wirepod_say(%r) -> %s", text, result.strip())
        return ok
    except Exception as e:
        log.warning("wirepod_say(%r) failed: %s", text, e)
        return False


def trigger_listening(cfg):
    url = f"{cfg['wirepod_api_base']}/api-sdk/trigger_wake_word?serial={cfg['robot_serial']}"
    timeout = cfg.get("http_timeout_seconds", 15)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        log.info("trigger_wake_word -> %s", body.strip())
        return True
    except Exception as e:
        log.warning("trigger_wake_word failed: %s", e)
        return False


def trigger_go_home_intent(cfg):
    """LEGACY go-home path -- send Vector home via wire-pod's cloud_intent
    HTTP API (the same mechanism trigger_explore() uses), sending
    intent_system_charger, the same intent wire-pod's own admin UI "Go"
    button next to the charger control sends (webroot/sdkapp/settings.html:
    cloud_intent?intent=intent_system_charger) and the same one chipper's
    own intent-keyphrase tests map "go home"/"go to your charger" to.

    DO NOT rely on this -- confirmed broken against vector.log 2026-09-29
    03:56:04 UTC: the AppIntent-injected system_charger intent isn't
    claimed by any behavior, so vic-engine logs "Intent 'system_charger'
    has been pending for 3 ticks, forcing a clear" ->
    "@behavior.voice_command.dropped system_charger" ->
    anim_communication_cantdothat. Kept only behind `go_home_method:
    "intent"` for reference/testing. drive_home_sdk() (the default) is the
    path that's actually proven to work (same night, 03:57:21/03:57:43 UTC,
    real GoHome behavior via SDK-level DriveOnCharger).

    Returns (sent, outcome). outcome is only ever "ok" (HTTP call to
    wire-pod succeeded) or "fail" (it didn't) -- this path has no way to
    tell "skip" apart from a false "ok", since a 200 from cloud_intent just
    means wire-pod relayed the AppIntent, not that vic-engine claimed it."""
    intent = cfg.get("go_home_intent", "intent_system_charger")
    url = f"{cfg['wirepod_api_base']}/api-sdk/cloud_intent?serial={cfg['robot_serial']}&intent={intent}"
    timeout = cfg.get("http_timeout_seconds", 15)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        log.info("go-home (intent): cloud_intent(%s) -> %s", intent, body.strip())
        return True, "ok"
    except Exception as e:
        log.warning("go-home (intent): trigger_go_home_intent failed: %s", e)
        return False, "fail"


def drive_home_sdk(cfg, state):
    """Send Vector home using the SDK's real DriveOnCharger behavior --
    robot.behavior.drive_on_charger() -- inside ONE short-lived SDK
    connection. This is the mechanism that's actually confirmed to work
    (vector.log 2026-09-29 03:57:21/03:57:43 UTC: SDK `DriveOnCharger` rpc
    -> `@behavior.feature.start GoHome` -> real charger-docking drive
    animations), unlike the old cloud_intent(intent_system_charger)
    AppIntent path (see trigger_go_home_intent()'s docstring), which the
    SAME log shows getting silently dropped.

    Uses OVERRIDE_BEHAVIORS_PRIORITY, not DEFAULT_PRIORITY -- a real "go
    home" needs to actually cut in over Vector's own autonomy (wandering,
    idle animations, etc.), the way it would if the owner had said it out
    loud, rather than only acting when he happens to already be idle the
    way capture_image()'s DEFAULT_PRIORITY connection does. Bounded by a
    hard wall-clock timeout (go_home_timeout_seconds, default 60s --
    driving to and mounting the charger legitimately takes longer than a
    quick camera grab) via the same _run_with_hard_timeout() helper
    capture_image()/capture_night_images() use, with the same
    guaranteed-disconnect-in-`finally` so behavior control is always
    released even on a timeout. This is the only other SDK connection in
    the whole daemon, and -- like capture_image()/capture_night_images()
    -- it is only ever invoked synchronously from the single-threaded poll
    loop, so it can never overlap another SDK connection from this
    process; the top-level circuit breaker check in run_cycle() (which
    gates maybe_send_home() same as everything else) keeps it from firing
    while contact is already paused.

    REWORK 2026-09-29c (voice-collision fix): a real "go home" cutting in
    with OVERRIDE_BEHAVIORS_PRIORITY is exactly the kind of grab that must
    never land mid-conversation, so this now does the same fail-closed
    immediate pre-connect voice check plus continuous during-hold voice
    watcher as capture_image()/capture_night_images() (see
    _voice_watch_loop) -- a voice interruption here reports as "skip", not
    "fail", so it doesn't trip the circuit breaker; maybe_send_home()'s
    own retry/spacing logic picks it back up on the next eligible poll.

    Returns (sent, outcome):
      sent    -- True only if the RPC came back BEHAVIOR_COMPLETE_STATE.
                 This means the behavior ran to completion, NOT that he's
                 confirmed docked -- that's verified by the next battery
                 poll (maybe_send_home clears retry state once
                 is_on_charger_platform comes back true).
      outcome -- "ok" (completed), "skip" (BEHAVIOR_WONT_ACTIVATE_STATE,
                 BEHAVIOR_INVALID_STATE -- a legitimate robot-side
                 refusal, e.g. he's being held -- or a voice interruption
                 before/during the hold; none of these are our fault and
                 none count toward the circuit breaker), or "fail" (hard
                 timeout or SDK exception -- counts toward the SAME
                 circuit breaker capture_image()/capture_night_images()
                 feed, via _record_capture_outcome())."""
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel
    from anki_vector.messaging import protocol

    timeout_s = cfg.get("go_home_timeout_seconds", 60)

    if voice_recently_active(cfg, state):
        log.info("drive_home_sdk: voice activity immediately before grab -- skipping (fail closed)")
        return False, "skip"

    robot = anki_vector.Robot(
        serial=cfg["robot_serial"],
        behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY,
        default_logging=False,
    )
    stop_watch = threading.Event()
    abort_box = {"aborted": False}
    watcher = threading.Thread(
        target=_voice_watch_loop,
        args=(cfg, state, robot, stop_watch, abort_box),
        daemon=True,
    )

    def _do():
        watcher.start()
        robot.connect(timeout=min(10, timeout_s))
        return robot.behavior.drive_on_charger()

    try:
        resp, timed_out = _run_with_hard_timeout(_do, timeout_s)
        stop_watch.set()
        if abort_box["aborted"]:
            log.info("drive_home_sdk: aborted mid-drive due to voice activity -- releasing, will retry next eligible poll")
            return False, "skip"
        if timed_out:
            log.warning(
                "drive_home_sdk: HARD TIMEOUT after %ss -- forcing disconnect/release",
                timeout_s,
            )
            return False, "fail"
        if resp is None:
            log.warning("drive_home_sdk: connected but got no response (SDK exception -- see above)")
            return False, "fail"
        if resp.result == protocol.BEHAVIOR_COMPLETE_STATE:
            log.info("drive_home_sdk: DriveOnCharger completed (status=%s)", resp.status.code)
            return True, "ok"
        log.info(
            "drive_home_sdk: DriveOnCharger did not activate/complete "
            "(result=%s status=%s) -- robot-side refusal, not counted as a failure",
            resp.result, resp.status.code,
        )
        return False, "skip"
    finally:
        stop_watch.set()
        try:
            robot.disconnect()
        except Exception:
            pass


def trigger_go_home(cfg, state):
    """Dispatch to the configured go-home mechanism -- config
    `go_home_method`: "drive_on_charger" (default) uses drive_home_sdk(),
    the mechanism confirmed to actually work; "intent" falls back to the
    legacy cloud_intent AppIntent path (trigger_go_home_intent()), which is
    confirmed BROKEN (silently dropped by vic-engine) and kept only for
    reference. See both functions' docstrings for the vector.log evidence.

    Returns (sent, outcome) -- see drive_home_sdk()'s docstring for the
    outcome contract; the legacy path can only ever report "ok" or "fail",
    never "skip" (see trigger_go_home_intent())."""
    method = cfg.get("go_home_method", "drive_on_charger")
    if method == "intent":
        return trigger_go_home_intent(cfg)
    return drive_home_sdk(cfg, state)


def maybe_send_home(cfg, state, now, voice_busy, context, retry_minutes=None):
    """Make sure Vector is heading back to (or already on) the charger.
    Called: right after a wander/explore window ends, right after a day
    curious cycle that happened off the charger, once local clock hits
    return_home_by (evening pre-dark return), when quiet hours begin, and
    at every night tick -- see call sites in run_day_cycle() and
    run_cycle(). Also doubles as the low-battery-off-charger safety net
    (requirement: send home immediately and hold new wanders until
    docked), since every call re-checks battery/charger state fresh.

    Never overrides voice activity, music/dance, or brain-busy -- if any
    of those are active we skip and rely on the NEXT call (next poll,
    which happens on the same brisk cadence as everything else in this
    file) to retry, rather than forcing an intent through mid-interaction.

    Retries up to return_home_max_tries times, spaced `retry_minutes`
    apart (defaults to config return_home_retry_minutes if not given --
    run_cycle()'s night call sites pass night_return_home_retry_minutes
    instead, since docking depends on Vector actually seeing the charger's
    IR marker, which is unreliable once the room's dark, so at night we
    try at most about once an hour instead of every few minutes), tracked
    in state.json so the spacing survives process restarts. Returns True
    if a go-home attempt was made (or the low-battery hold flag was
    freshly set) this call, False otherwise (docked, disabled, gated busy,
    or retries exhausted for this off-charger episode)."""
    if retry_minutes is None:
        retry_minutes = cfg.get("return_home_retry_minutes", 3)
    if not cfg.get("return_home_after_wander", True):
        return False

    if voice_busy:
        log_skip(f"go-home ({context}): voice activity -- retry later", 60)
        return False
    if music_playing(cfg):
        log_skip(f"go-home ({context}): music/dance playing -- retry later", 60)
        return False
    if brain_busy(cfg):
        log_skip(f"go-home ({context}): brain busy -- retry later", 60)
        return False

    battery = get_battery(cfg)
    if battery is None:
        log_skip(f"go-home ({context}): could not verify battery/charger state", 60)
        return False

    on_charger = battery.get("is_on_charger_platform")
    level = battery.get("battery_level")

    if on_charger:
        if state.get("go_home_tries") or state.get("low_battery_hold"):
            log.info(
                "go-home (%s): docked -- clearing retry state (was %d tries)",
                context, state.get("go_home_tries", 0),
            )
        state["go_home_tries"] = 0
        state["go_home_next_try_at"] = 0
        state["low_battery_hold"] = False
        save_state(state)
        return False

    # Off the charger from here on.
    low_battery = level is not None and level <= cfg.get("battery_low_level", 1)
    if low_battery and not state.get("low_battery_hold"):
        log.warning(
            "go-home (%s): low battery (level=%s) off charger -- holding new "
            "wanders until he's docked", context, level,
        )
        state["low_battery_hold"] = True
        save_state(state)

    max_tries = cfg.get("return_home_max_tries", 3)
    tries = state.get("go_home_tries", 0)
    retry_due = now >= state.get("go_home_next_try_at", 0)
    # Spacing always applies -- even under low battery we don't hammer
    # wire-pod every poll, just every return_home_retry_minutes. (tries=0
    # always passes: go_home_next_try_at defaults to 0, so the very first
    # attempt in an episode is always "due" -- i.e. immediate.)
    if tries and not retry_due:
        return False  # mid-retry-sequence, not due yet
    if tries and tries >= max_tries and not low_battery:
        log_skip(
            f"go-home ({context}): still off charger after {tries} tries -- "
            "giving up until the next trigger",
            cfg.get("log_dedup_seconds", 900),
        )
        return False
    # Low battery never gives up on the try cap -- safety takes priority --
    # but it still obeyed the retry_due spacing check above, so it retries
    # every return_home_retry_minutes rather than hammering.

    log.info(
        "go-home (%s): off charger (battery_level=%s, low_battery=%s) -- "
        "sending go-home, attempt %d/%d (method=%s)",
        context, level, low_battery, tries + 1, max_tries,
        cfg.get("go_home_method", "drive_on_charger"),
    )
    sent, outcome = trigger_go_home(cfg, state)
    _record_capture_outcome(state, now, outcome)
    state["go_home_tries"] = tries + 1
    state["go_home_next_try_at"] = now + retry_minutes * 60
    save_state(state)
    if not sent:
        log.warning("go-home (%s): go-home attempt did not complete (outcome=%s)", context, outcome)
    return True


def wait_for_answer(cfg, since_ts):
    """Poll wire-pod's log buffer for up to answer_wait_seconds looking for
    the next real transcription after since_ts (the moment we finished
    asking our question). Returns (answer_text_or_None, is_command_bool).
    is_command is True when the matched line's intent looks like a
    command/action rather than a conversational reply (see
    COMMAND_INTENT_MARKERS) -- callers should skip fact-extraction in that
    case without even calling the LLM."""
    url = f"{cfg['wirepod_api_base']}/api/get_logs"
    http_timeout = cfg.get("http_timeout_seconds", 15)
    wait_s = cfg.get("answer_wait_seconds", 20)
    poll_every = cfg.get("answer_poll_seconds", 2)
    deadline = time.time() + wait_s

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=http_timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as e:
            log.warning("wait_for_answer: get_logs failed: %s", e)
            time.sleep(poll_every)
            continue

        candidates = [
            (ts, txt, intent)
            for ts, txt, intent in _parse_transcribed_lines(text)
            if ts > since_ts and txt.strip()
        ]
        if candidates:
            candidates.sort(key=lambda c: c[0])
            ts, txt, intent = candidates[0]
            is_command = bool(intent) and any(
                marker in intent.lower() for marker in COMMAND_INTENT_MARKERS
            )
            return txt.strip(), is_command

        time.sleep(poll_every)

    return None, False


def extract_taught_fact(cfg, question, answer):
    """Ask the main LLM (via the brain proxy) to turn a Q&A pair into ONE
    short fact Vector learned, or None if the answer didn't actually teach
    anything (off-topic, nonsense, or the model itself says NONE). Empty
    answers are rejected here too (not just by the wait_for_answer caller)
    -- an empty prompt still gets vault context injected by the brain proxy
    and the model will happily hallucinate a "fact" from that alone."""
    if not answer or not answer.strip():
        return None
    system = (
        "You extract facts a home robot named Vector learns from what "
        "the owner tells him. Given the question Vector just asked out loud "
        "and the owner's spoken answer, write ONE short factual sentence "
        "Vector learned -- plain text, third-person, no markdown, no "
        "preamble (e.g. 'The red mug on the desk is the owner's coffee mug "
        "from a conference'). Reply with exactly NONE (nothing else) if the "
        "answer isn't really teaching Vector something about what he asked "
        "about -- including if it's empty, off-topic, nonsense, OR if it's "
        "actually an instruction/command directed AT Vector rather than an "
        "answer (e.g. 'stop dancing', 'come here', 'play some music' -- "
        "those are commands, not facts, even though they mention Vector)."
    )
    user = f"Vector asked: {question}\n{OWNER_NAME.capitalize()} answered: {answer}\n\nWrite the fact now."
    payload = {
        "model": cfg["llm_model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        result = http_json(
            f"{cfg['brain_proxy_base']}/v1/chat/completions", data=payload, timeout=60
        )
        text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("extract_taught_fact failed: %s", e)
        return None
    text = text.splitlines()[0].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    if not text or text.upper() == "NONE":
        return None
    return text


def append_taught_fact(cfg, fact):
    """Append a fact the owner taught Vector under '## Things the owner taught
    me' in MEMORY.md, creating that section if it doesn't exist yet.
    Dedupes (substring + fuzzy) against everything already under that
    heading. Atomic write (temp file + os.replace); never touches any
    other section. Returns True if a new line was written."""
    path = cfg.get("vector_memory_path")
    heading = "## Things the owner taught me"
    if not path or not fact:
        return False

    try:
        with open(path) as f:
            content = f.read()
    except OSError as e:
        log.warning("append_taught_fact: could not read %s: %s", path, e)
        return False

    has_heading = heading in content
    next_heading = None
    if has_heading:
        section_start = content.index(heading) + len(heading)
        next_heading = re.search(r"\n## ", content[section_start:])
        section_end = section_start + next_heading.start() if next_heading else len(content)
        section_text = content[section_start:section_end]
    else:
        section_text = ""

    # Strip the leading "- YYYY-MM-DD " date stamp before normalizing each
    # existing line, so date tokens don't dilute the token-overlap (Jaccard)
    # similarity check below -- otherwise every line shares 3 numeric date
    # tokens with today's un-dated candidate fact for no meaningful reason.
    date_prefix_re = re.compile(r"^-\s*\d{4}-\d{2}-\d{2}\s+")
    existing_blob = _normalize_for_match(section_text)
    existing_lines = []
    for l in section_text.splitlines():
        stripped = l.strip()
        if not stripped.startswith("-"):
            continue
        existing_lines.append(_normalize_for_match(date_prefix_re.sub("", stripped)))
    norm_fact = _normalize_for_match(fact)

    is_dup = bool(norm_fact) and norm_fact in existing_blob
    if not is_dup:
        fact_tokens = set(norm_fact.split())
        for line_norm in existing_lines:
            seq_ratio = difflib.SequenceMatcher(None, norm_fact, line_norm).ratio()
            line_tokens = set(line_norm.split())
            jaccard = (
                len(fact_tokens & line_tokens) / len(fact_tokens | line_tokens)
                if (fact_tokens or line_tokens)
                else 0
            )
            # SequenceMatcher catches near-identical phrasing; Jaccard token
            # overlap catches the same fact reworded/reordered.
            if seq_ratio > 0.85 or jaccard >= 0.45:
                is_dup = True
                break
    if is_dup:
        log.info("append_taught_fact: duplicate/near-duplicate fact, skipping: %r", fact)
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    new_line = f"- {today} {fact}"

    if has_heading:
        if next_heading:
            addition = "\n" + new_line
            new_content = content[:section_end] + addition + content[section_end:]
        else:
            new_content = content.rstrip("\n") + "\n" + new_line + "\n"
    else:
        new_content = content.rstrip("\n") + "\n\n" + heading + "\n" + new_line + "\n"

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write(new_content)
        os.replace(tmp_path, path)
        _fix_owner(path)
    except OSError as e:
        log.warning("append_taught_fact: write failed: %s", e)
        return False

    log.info("append_taught_fact: appended: %r", new_line)
    return True


def append_night_observation(cfg, scene):
    """Append a one-line HH:MM (America/New_York) scene note to
    night_observations.md. Dedupes only against the immediately preceding
    line -- "consecutive near-identical scenes" means Vector sitting still
    on the charger for an hour shouldn't produce a dozen identical lines,
    not that a scene can never recur later in the night. Caps the file at
    ~200 lines, trimming the OLDEST first. Atomic write (temp + rename)."""
    path = cfg.get("night_observations_path")
    if not path or not scene:
        return False

    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as e:
        log.warning("append_night_observation: could not read %s: %s", path, e)
        return False

    tz = cfg.get("timezone", "America/New_York")
    now_local = datetime.now(ZoneInfo(tz)) if ZoneInfo else datetime.now()
    stamp = now_local.strftime("%H:%M")
    new_line = f"- {stamp} {scene}"

    if lines:
        last = lines[-1].strip()
        last_scene = re.sub(r"^-\s*\d{2}:\d{2}\s+", "", last)
        threshold = cfg.get("night_scene_dedupe_ratio", 0.85)
        sim = difflib.SequenceMatcher(
            None, _normalize_for_match(scene), _normalize_for_match(last_scene)
        ).ratio()
        if sim > threshold:
            log.info("append_night_observation: consecutive near-identical scene, skipping: %r", scene)
            return False

    lines.append(new_line)
    cap = cfg.get("night_observations_max_lines", 200)
    if len(lines) > cap:
        lines = lines[-cap:]

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp_path, path)
        _fix_owner(path)
    except OSError as e:
        log.warning("append_night_observation: write failed: %s", e)
        return False

    log.info("append_night_observation: %s", new_line)
    return True


def run_night_cycle(cfg, state, now, voice_busy):
    """Silent Night Learning (quiet hours, night_silent_learning=true):
    NO announcement, NO spoken question, NO driving, NO sounds/animations
    -- the family is asleep. Just a brief control window to capture 1-2
    photos, then vision + memory updates, on its own night_gap_minutes
    cadence (independent of the daytime cooldown). Skips cleanly if the
    capture fails or the robot is off-network -- see capture_night_images
    and _shared_busy_gate's fail-safe checks. Returns True if a real
    capture attempt was made this call (regardless of outcome), False if
    we never got past the cheap gates -- used to drive the idle-poll
    backoff in main()."""
    if now < state.get("night_next_eligible_at", 0):
        return False

    gate = _shared_busy_gate(cfg, voice_busy)
    if gate:
        reason, dedup_s = gate
        log_skip(f"night: {reason}", dedup_s)
        return False

    log.info("night: all clear -- silent learning cycle")
    image_paths, outcome, detail = capture_night_images(cfg, state)
    _record_capture_outcome(state, now, outcome)
    if outcome != "ok":
        if outcome == "skip":
            log_skip(f"night: {detail}", 60)
        else:
            log.warning("night: cycle aborted -- capture failed (SDK error/timeout)")
        state["night_next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        return True

    descriptions = []
    all_objects = []
    try:
        for path in image_paths:
            desc, objects = vision_describe(cfg, path)
            if desc:
                descriptions.append(desc)
                all_objects.extend(objects)
    finally:
        for path in image_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    if descriptions:
        learned = update_vector_memory(cfg, all_objects)
        if learned:
            log.info("night: learned %d new labeled thing(s)", learned)
        append_night_observation(cfg, descriptions[0])
    else:
        log.info("night: nothing notable in either frame -- no memory update this cycle")

    gap_min = cfg.get("night_gap_minutes", 12)
    # "every ~10-15 min" around the single night_gap_minutes config value.
    low_min = gap_min * 0.83
    high_min = gap_min * 1.25
    state["night_next_eligible_at"] = now + random.uniform(low_min, high_min) * 60
    save_state(state)
    log.info(
        "night cycle complete: descriptions=%d next_eligible_in_min=%.1f",
        len(descriptions),
        (state["night_next_eligible_at"] - now) / 60,
    )
    return True


def _shared_busy_gate(cfg, voice_busy):
    """Cheap safety/busy checks shared by day and night cycles -- file
    reads + wire-pod/brain-proxy HTTP calls only, NO SDK connection (see
    REWORK 2026-09-28 in the module docstring). Returns None if all clear,
    or (reason, dedup_seconds) to skip. Fail-safe throughout: anything
    unverifiable skips rather than proceeding.

    The physical robot-status check (is_being_held/are_motors_moving/
    is_animating/is_pathing/is_in_calm_power_mode) used to live here as a
    separate peek_robot_status() SDK connect. There's no wire-pod HTTP
    endpoint for those flags, so that check now happens inside
    capture_image()/capture_night_images() themselves, folded into the one
    connection a cycle opens once it's actually about to try a capture --
    not as a connection of its own just to look."""
    if music_playing(cfg):
        return ("music is playing (or couldn't verify)", cfg.get("log_dedup_seconds", 900))

    if voice_busy:
        return (
            f"voice activity in last {cfg.get('voice_quiet_seconds', 600)}s (or couldn't verify)",
            120,
        )

    if brain_busy(cfg):
        return (
            f"brain-proxy activity in last {cfg.get('brain_busy_seconds', 600)}s (or couldn't verify)",
            120,
        )

    battery = get_battery(cfg)
    if battery is None:
        return ("could not verify battery state", 120)
    level = battery.get("battery_level")
    on_charger = battery.get("is_on_charger_platform")
    if on_charger and level is not None and level <= cfg.get("battery_low_level", 1):
        return ("charging with low battery", cfg.get("log_dedup_seconds", 900))

    return None


def _record_capture_outcome(state, now, outcome):
    """Feed a capture attempt's outcome (from capture_image()/
    capture_night_images()) into the circuit breaker. "ok" resets the
    consecutive-failure counter. "fail" (hard timeout, SDK exception, or
    connected-but-no-image) increments it and, once it reaches
    CIRCUIT_BREAKER_FAILURE_THRESHOLD, pauses ALL robot contact for
    CIRCUIT_BREAKER_PAUSE_SECONDS. "skip" (robot legitimately held/moving/
    animating/asleep -- not our fault) leaves the counter untouched
    entirely, so a robot that's just being played with for a while doesn't
    accidentally trip the breaker."""
    if outcome == "ok":
        if state.get("consecutive_capture_failures"):
            state["consecutive_capture_failures"] = 0
            save_state(state)
        return
    if outcome != "fail":
        return  # "skip" doesn't count either way

    n = state.get("consecutive_capture_failures", 0) + 1
    state["consecutive_capture_failures"] = n
    if n >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
        until = now + CIRCUIT_BREAKER_PAUSE_SECONDS
        state["circuit_breaker_until"] = until
        state["consecutive_capture_failures"] = 0
        log.error(
            "CIRCUIT BREAKER TRIPPED: %d consecutive capture failures/SDK errors -- "
            "pausing ALL robot contact for %ds (until %s local clock time)",
            n,
            CIRCUIT_BREAKER_PAUSE_SECONDS,
            datetime.fromtimestamp(until).isoformat(timespec="seconds"),
        )
    save_state(state)


def run_cycle(cfg):
    """Returns True if this poll did something real (voice activity seen,
    an in-progress explore window, a full day/night capture attempt, or a
    freshly-triggered wander), False if it was a pure no-op (disabled,
    circuit breaker open, or gated out before ever touching wire-pod/the
    robot beyond the cheap checks). main() uses this to drive the idle-poll
    backoff -- see REWORK 2026-09-28 in the module docstring."""
    if not cfg.get("enabled", True):
        log_skip("disabled via config.json", cfg.get("log_dedup_seconds", 900))
        return False
    if os.path.exists(DISABLED_PATH):
        log_skip("DISABLED sentinel present", cfg.get("log_dedup_seconds", 900))
        return False

    state = load_state()
    now = time.time()

    breaker_until = state.get("circuit_breaker_until", 0)
    if now < breaker_until:
        log_skip(
            f"circuit breaker open until {datetime.fromtimestamp(breaker_until).isoformat(timespec='seconds')}",
            60,
        )
        return False

    # Track voice activity on EVERY poll, even while still in cooldown --
    # wire-pod's log buffer is a fixed-size ring that can roll a recent
    # transcription out well before our window elapses, so we need to keep
    # the persisted last-seen timestamp current rather than only checking
    # it when we'd otherwise be eligible to act.
    voice_busy = voice_recently_active(cfg, state)
    # Note: this is the cheap, cycle-start value -- used only for the
    # coarse early gates below (_shared_busy_gate, should_wander,
    # maybe_send_home's own busy checks). REWORK 2026-09-29c: it is
    # deliberately NOT relied on as the sole guard for any actual SDK
    # connect/control grab -- capture_image()/capture_night_images()/
    # drive_home_sdk() each do their own fresh, immediate
    # voice_recently_active() check right before connecting (and a
    # continuous watcher for the life of the hold), since this value can
    # go stale by the time a grab actually happens (TTS, vision, retry
    # spacing, etc. all take real wall-clock time -- see the module
    # docstring's 13:22:29Z/13:22:36.1Z incident).
    #
    # (The old "abort an in-progress explore window on voice activity"
    # check that used to live here is gone along with the explore window
    # itself -- see should_wander()'s docstring; wander is now a few
    # seconds folded into a single capture_image() connection, not a
    # multi-minute window the poll loop tracks separately.)

    now_in_quiet = in_quiet_hours(cfg, now)
    entering_quiet = now_in_quiet and not state.get("was_in_quiet_hours", False)
    if now_in_quiet != state.get("was_in_quiet_hours", False):
        state["was_in_quiet_hours"] = now_in_quiet
        save_state(state)

    if now_in_quiet:
        # Requirement: send him home when quiet hours begin, and again at
        # EVERY night tick he's found off the charger -- not just on
        # run_night_cycle's own ~12-min capture cadence, so this runs on
        # every poll regardless of night_next_eligible_at. Never starts a
        # wander at night (run_night_cycle never calls maybe_start_explore).
        # Night pacing is looser than daytime (night_return_home_retry_minutes,
        # default 60) -- see REWORK 2026-09-29b: docking needs Vector to see
        # the charger's IR marker, which is unreliable in a dark room, so we
        # try at most about once an hour rather than hammering it.
        sent_home = maybe_send_home(
            cfg, state, now, voice_busy,
            "quiet hours begin" if entering_quiet else "night tick",
            retry_minutes=cfg.get("night_return_home_retry_minutes", 60),
        )
        if cfg.get("night_silent_learning", False):
            did_something = run_night_cycle(cfg, state, now, voice_busy)
        else:
            log_skip("quiet hours", cfg.get("log_dedup_seconds", 900))
            did_something = False
        return bool(voice_busy or did_something or sent_home)

    # Evening pre-dark return (REWORK 2026-09-29b): once local clock hits
    # return_home_by (default 21:30), BEFORE quiet_hours_start (22:00) and
    # the darkness that comes with it, start nudging him home if he's off
    # the charger -- same maybe_send_home() plumbing/retry state as every
    # other call site, just an earlier trigger so the first attempt isn't
    # made only once the room's already dark.
    return_home_by = cfg.get("return_home_by")
    sent_evening_home = False
    if return_home_by and _local_time_reached(cfg, now, return_home_by):
        sent_evening_home = maybe_send_home(
            cfg, state, now, voice_busy, "evening pre-dark return"
        )

    did_something = run_day_cycle(cfg, state, now, voice_busy)
    remarked = False
    if not did_something:
        # Only consider a proactive remark on a poll where the normal
        # curious cycle didn't already speak -- keeps this from ever
        # stacking a second utterance onto the same poll's announcement/
        # question.
        try:
            remarked = maybe_proactive_remark(cfg, state, now, voice_busy)
        except Exception as e:
            log.warning("maybe_proactive_remark failed (non-fatal): %s", e)
    return bool(voice_busy or did_something or sent_evening_home or remarked)


def should_wander(cfg, state, now, voice_busy):
    """REWORK 2026-09-29c (wander fix): eligibility check for folding a
    short real SDK wander into THIS capture cycle -- see _do_safe_wander
    and capture_image()'s do_wander param. Replaces the old
    maybe_start_explore()'s battery/charge-dwell gating logic (kept
    as-is here), minus the separate explore-window/cloud_intent
    machinery, which never actually moved the robot (see module
    docstring: intent_explore_start AppIntents are silently dropped by
    vic-engine -- "PendingIntentNotCleared.ForceClear ->
    @behavior.voice_command.dropped explore_start App", reproduced
    2026-09-29 13:32Z).

    Wander now happens INSIDE the existing capture_image() connection
    (once per eligible curious cycle, ~every min_gap_minutes/
    max_gap_minutes, i.e. ~20-30 min) rather than as its own ~5-minute-
    cadence filler window -- deliberately: it keeps total robot contact
    to the same one-connect-per-cycle budget this daemon already commits
    to (see REWORK 2026-09-28 -- "past SDK connect storms have frozen the
    robot"), instead of adding a whole new frequent SDK-connect code
    path. explore_gap_minutes is reused as the minimum spacing between
    two wanders via next_wander_eligible_at, so back-to-back cycles
    still won't always wander.

    Does NOT open an SDK connection itself -- only the existing
    music/voice/brain/battery HTTP+file checks, same as the old
    function."""
    if not cfg.get("wander_enabled", True):
        return False
    if state.get("low_battery_hold"):
        return False  # off charger on low battery -- hold until docked
    if now < state.get("next_wander_eligible_at", 0):
        return False
    if music_playing(cfg) or voice_busy or brain_busy(cfg):
        return False

    battery = get_battery(cfg)
    if battery is None:
        return False
    level = battery.get("battery_level")
    on_charger = battery.get("is_on_charger_platform")

    if on_charger:
        if level is not None and level < 3:
            # not full yet -- let him keep charging, don't send him wandering
            return False
        charge_started_at = state.get("on_charger_since", 0)
        if not charge_started_at:
            state["on_charger_since"] = now
            save_state(state)
            return False
        if now - charge_started_at < cfg.get("min_charge_dwell_minutes", 30) * 60:
            return False  # full, but give him a proper charge dwell first
    else:
        if state.get("on_charger_since"):
            state["on_charger_since"] = 0
            save_state(state)
        if level is not None and level < cfg.get("wander_min_battery_level", 2):
            return False  # battery's getting low -- let him go charge instead

    return True


def run_day_cycle(cfg, state, now, voice_busy):
    """Daytime always-curious: once the normal question cooldown is due,
    run a full curious cycle (announce -> capture [-> short safe wander,
    if due] -> release -> vision -> learn -> ask -> listen). Returns True
    if this call did something real (a capture attempt), False otherwise
    -- see run_cycle()'s docstring.

    REWORK 2026-09-29c (wander fix): the old "wander while waiting for
    the next question" filler window is gone (see should_wander()'s
    docstring for why) -- there's nothing useful to do here until
    next_eligible_at, so we just wait for it like any other gated cycle."""
    if now < state.get("next_eligible_at", 0):
        return False

    gate = _shared_busy_gate(cfg, voice_busy)
    if gate:
        reason, dedup_s = gate
        log_skip(reason, dedup_s)
        return False

    do_wander = should_wander(cfg, state, now, voice_busy)

    log.info("all clear -- announcing curious mode%s", " (with wander)" if do_wander else "")
    announcement = random.choice(cfg.get("announcement_lines", ["Let me just wander and be curious."]))
    if not wirepod_say(cfg, announcement):
        log.warning("cycle aborted: announcement failed to speak")
        state["next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        maybe_send_home(cfg, state, now, voice_busy, "day cycle")
        return True

    image_path, outcome, detail, wandered = capture_image(cfg, state, do_wander=do_wander)
    _record_capture_outcome(state, now, outcome)
    if wandered:
        state["next_wander_eligible_at"] = now + cfg.get("explore_gap_minutes", 5) * 60
        # Fresh wander -- give it its own full return_home_max_tries budget
        # rather than staying permanently exhausted by an earlier episode
        # that never made it home.
        state["go_home_tries"] = 0
        state["go_home_next_try_at"] = 0
        save_state(state)
        log.info("day: wander folded into this capture cycle -- completed real SDK moves (not a cloud_intent no-op)")
    try:
        maybe_greet_faces(cfg, state, now, voice_busy)
    except Exception as e:
        log.warning("maybe_greet_faces failed (non-fatal): %s", e)
    if outcome != "ok":
        if outcome == "skip":
            log_skip(f"day: {detail}", 60)
        else:
            log.warning("cycle aborted: capture failed (SDK error/timeout)")
        state["next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        maybe_send_home(cfg, state, now, voice_busy, "day cycle")
        return True

    try:
        description, objects = vision_describe(cfg, image_path)
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

    if description is None:
        log.info("vision: nothing notable -- skipping this cycle, short retry")
        state["next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        maybe_send_home(cfg, state, now, voice_busy, "day cycle")
        return True

    log.info("vision description: %s", description)
    if objects:
        log.info("vision objects: %s", objects)

    if _is_low_confidence_scene(description, objects):
        log.info(
            "vision: low-confidence scene (%r, %d object(s)) -- skipping "
            "question, short retry", description, len(objects)
        )
        try:
            append_night_observation(cfg, f"(low-confidence, day) {description}")
        except Exception as e:
            log.warning("append_night_observation (low-confidence) failed: %s", e)
        state["next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        maybe_send_home(cfg, state, now, voice_busy, "day cycle")
        return True

    question = generate_question(cfg, description, objects)
    if not question:
        log.warning("cycle aborted: no question generated")
        state["next_eligible_at"] = now + cfg.get("empty_scene_retry_minutes", 5) * 60
        save_state(state)
        maybe_send_home(cfg, state, now, voice_busy, "day cycle")
        return True

    log.info("generated question: %s", question)
    write_recent_observation(cfg, description, question)
    learned = update_vector_memory(cfg, objects)
    if learned:
        log.info("learned %d new thing(s) for MEMORY.md", learned)

    spoke_ok = wirepod_say(cfg, question)
    learned_fact = False
    if spoke_ok:
        trigger_listening(cfg)
        asked_at = time.time()
        answer, is_command = wait_for_answer(cfg, since_ts=asked_at)
        if not answer:
            log.info("no answer captured within %ss", cfg.get("answer_wait_seconds", 20))
        elif is_command:
            log.info("answer looked like a command, not extracting a fact: %r", answer)
        else:
            log.info("captured answer: %r", answer)
            try:
                _note_curious_play(cfg)
            except Exception as e:
                log.debug("_note_curious_play call failed (non-fatal): %s", e)
            fact = extract_taught_fact(cfg, question, answer)
            if fact:
                learned_fact = append_taught_fact(cfg, fact)
                if learned_fact:
                    log.info("learned from the owner: %r", fact)
            else:
                log.info("no fact extracted from answer (NONE or off-topic)")

    min_gap = cfg.get("min_gap_minutes", 20) * 60
    max_gap = cfg.get("max_gap_minutes", 30) * 60
    state["last_trigger_at"] = now
    state["next_eligible_at"] = now + random.uniform(min_gap, max_gap)
    save_state(state)
    log.info(
        "cycle complete: announced=%r spoke=%s learned_fact=%s next_eligible_in_min=%.1f",
        announcement,
        spoke_ok,
        learned_fact,
        (state["next_eligible_at"] - now) / 60,
    )
    maybe_send_home(cfg, state, now, voice_busy, "day cycle")
    return True


def main():
    log.info(
        "Curious Vector starting up (rework 2026-09-28: single connect/cycle, "
        "HTTP-first gates, idle-poll backoff %s, circuit breaker %d/%ds)",
        IDLE_POLL_LADDER,
        CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        CIRCUIT_BREAKER_PAUSE_SECONDS,
    )
    poll_s = IDLE_POLL_LADDER[0]
    while True:
        cfg = load_config()
        try:
            active = run_cycle(cfg)
        except Exception as e:
            log.exception("run_cycle crashed: %s", e)
            active = False

        if active:
            poll_s = IDLE_POLL_LADDER[0]
        else:
            # step up to the next rung of the ladder, capped at the last one
            try:
                idx = IDLE_POLL_LADDER.index(poll_s)
                poll_s = IDLE_POLL_LADDER[min(idx + 1, len(IDLE_POLL_LADDER) - 1)]
            except ValueError:
                poll_s = IDLE_POLL_LADDER[0]

        time.sleep(poll_s)


if __name__ == "__main__":
    main()
