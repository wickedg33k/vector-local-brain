#!/usr/bin/env python3
"""
Curious Vector -- occasional camera snapshot + local-vision description +
playful LLM question, spoken by Vector.

Runs as its own long-lived daemon inside the `vector-curious` container on
the GPU host (never on other shared hosts -- the owner's rule). It polls roughly every
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

REWORK 2026-10-15 (hermes, the owner: "when he is not talked to, once an hour
he will explore, talking to himself"): the every-~2-min announce ->
snapshot -> question -> 20 s mic cycle is OFF by default
(curious_question_cycle_enabled). By day: a SILENT face-check snapshot
keeps face recognition / current_people.json / greetings working, and an
hourly, idle-gated explore session drives around (leashed, cliff-guarded),
mutters self-talk, greets recognised people with a getting-to-know-you
question and learns per-person notes. See the HOURLY EXPLORE MODE block
above main(). Test trigger: touch data/explore_now or GET :8790/explore_now.
"""
import base64
import concurrent.futures
import http.server
import difflib
import fcntl
import json
import logging
import math
import os
import random
import re
import shutil
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
# Name the robot uses when addressing its owner (spoken questions, fallback
# template, fact extraction). Set VECTOR_OWNER_NAME in the container env.
OWNER_NAME = os.environ.get("VECTOR_OWNER_NAME", "friend")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
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

# Only one anki_vector SDK connection to the robot anywhere in this
# process at a time (see SENTRY MODE section below) -- capture_image(),
# capture_night_images(), drive_home_sdk(), and sentry's own
# _sentry_capture_and_detect() all acquire this before connecting and
# release it in their `finally`.
# Cross-process robot-control lock (2026-10-15, hermes, vector-faces): the
# threading lock only serialises THIS daemon's own SDK connections.
# vector-daycycle (host cron) and vector-faces' enroll.py take an exclusive
# fcntl.flock on ~/vector-pod/data/robot_control.lock (bind-mounted
# here at /mnt/robot_control.lock) for their robot-control windows.
# _SDK_LOCK now takes that flock too (non-blocking, polled within the same
# timeout), so curious never grabs the robot mid-enrollment. Fails open
# (in-process lock only, exactly the old behaviour) if the file isn't
# mounted. Same acquire(timeout=)/release() interface as threading.Lock.
ROBOT_CONTROL_LOCK_PATH = "/mnt/robot_control.lock"


class _RobotControlLock:
    def __init__(self, path):
        self._tlock = threading.Lock()
        self._path = path
        self._fh = None
        self._warned = False

    def _warn_once(self, msg):
        if not self._warned:
            self._warned = True
            log.warning("robot_control.lock: %s -- using in-process lock only", msg)

    def acquire(self, timeout=3):
        deadline = time.time() + max(0.0, timeout)
        if not self._tlock.acquire(timeout=max(0.0, timeout)):
            return False
        try:
            fh = open(self._path, "r+")  # never create: missing mount -> fail open
        except OSError as e:
            self._warn_once("not available (%s)" % e)
            return True
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fh
                return True
            except BlockingIOError:
                if time.time() >= deadline:
                    fh.close()
                    self._tlock.release()
                    log.info("robot_control.lock held by another process (enroll.py/daycycle)")
                    return False
                time.sleep(0.25)
            except OSError as e:
                fh.close()
                self._warn_once("flock failed (%s)" % e)
                return True

    def release(self):
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            fh.close()
        self._tlock.release()


_SDK_LOCK = _RobotControlLock(ROBOT_CONTROL_LOCK_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("curious-vector")

_last_skip_log = {}  # reason -> last logged epoch, for dedup



def _fix_owner(path):
    """Container runs as root; keep vault files owned by the host user (uid 1000) so host-side writers keep working."""
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


_state_write_lock = threading.Lock()


def save_state(state):
    # 2026-09-29 cliffsafe: adding safety_log_watch_loop as a THIRD
    # concurrent caller (alongside sentry_log_watch_loop and
    # presence_poll_loop/the main cycle) exposed a pre-existing race --
    # every caller wrote the SAME shared tmp path (STATE_PATH + ".tmp"),
    # so one thread's os.replace() could fire after another thread had
    # already renamed that same tmp file away, raising
    # FileNotFoundError. Fixed two ways: (1) each call now writes to a
    # tmp path unique to this process+thread, so concurrent writers never
    # touch the same tmp file, and (2) a module lock serializes the
    # write+replace pair so two threads' saves can't interleave into a
    # lost-update (last save_state() to acquire the lock always reflects
    # a load_state() that happened after the previous save_state()
    # released it, for callers that follow the standard
    # load-mutate-save pattern used throughout this file).
    tmp = f"{STATE_PATH}.tmp.{os.getpid()}.{threading.get_ident()}"
    with _state_write_lock:
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
DONE_RE = re.compile(r"stream_wav_file complete|queue done|playback_error=|stop\.sh: killing worker")


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
# name is <name>" while facing the robot). So this piggybacks on
# capture_image()'s EXISTING connection (enable_face_detection=True there,
# see its docstring) instead of adding a new frequent SDK-connect path.
FACE_STATE_KEY = "face_last_seen"              # name -> epoch last seen
FACE_GREETED_KEY = "face_last_greeted"         # name -> epoch last greeted
FACE_ANY_SEEN_KEY = "face_last_seen_any_at"    # epoch, ANY face (named or not)
FACE_UNNAMED_SEEN_KEY = "face_unnamed_seen_count"

GREETING_TEMPLATES_MORNING = ["Morning, {name}.", "Morning, {name}!"]
GREETING_TEMPLATES_GENERIC = ["Hey, {name}!", "Oh -- hey {name}.", "{name}! Good to see you."]


def record_face_sightings(cfg, state, faces, source="robot"):
    """Bookkeeping only -- never speaks (see maybe_greet_faces for that).

    2026-10-15 (hermes, vector-faces): names now come from OUR local face
    service by default (cfg face_source="local"): capture-path calls with
    source="robot" (the robot's built-in face store) then only count as
    "a face was seen" -- their names are ignored. face_source="robot"
    restores the old behaviour; "both" accepts names from either.
    Called from inside capture_image() right after its connection reads
    robot.world.visible_faces, regardless of that cycle's outcome. Logs
    plainly so the owner/curious.log show exactly what's needed: if only
    unnamed faces ever show up, no face is enrolled yet."""
    if not faces:
        return
    face_source = cfg.get("face_source", "local")
    if source == "robot" and face_source == "local":
        state[FACE_ANY_SEEN_KEY] = time.time()
        save_state(state)
        return
    if source == "local" and face_source == "robot":
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
            "enroll with ~/vector-faces/enroll.py --name \"<Name>\"",
            len(unnamed),
        )
    save_state(state)


# ---- Local household face recognition (2026-10-15, hermes, vector-faces).
# Each successful day capture is POSTed to the LOCAL vector-faces service
# (YuNet + SFace on this host, 127.0.0.1:8096 -- nothing leaves the box).
# Recognised names feed record_face_sightings() (so maybe_greet_faces greets
# with OUR names) and /mnt/vector-faces/current_people.json, which the brain
# proxy turns into "People you can currently see: ...". Service down ->
# None -> no names this cycle, everything else in the cycle unaffected.
def recognize_faces_local(cfg, image_path):
    if cfg.get("face_source", "local") == "robot":
        return None
    url = cfg.get("face_local_url", "http://127.0.0.1:8096/recognize")
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "image/jpeg")
        with urllib.request.urlopen(req, timeout=cfg.get("face_local_timeout_seconds", 6)) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        if not isinstance(res, list):
            raise ValueError("unexpected response %r" % (res,))
        return res
    except Exception as e:
        log_skip("vector-faces unavailable (%s: %s) -- no local face names this cycle"
                 % (type(e).__name__, e), cfg.get("log_dedup_seconds", 900))
        return None


def write_current_people(cfg, results, now):
    """[{name, score, last_seen_epoch}] for recognised (non-unknown) people,
    merged with entries seen in the last face_current_keep_seconds. Atomic,
    owned by whoever owns the directory (the host user), mode 0640."""
    path = cfg.get("face_current_people_path", "/mnt/vector-faces/current_people.json")
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        return
    keep_s = cfg.get("face_current_keep_seconds", 600)
    people = {}
    try:
        with open(path) as f:
            prev = json.load(f)
        for p in prev if isinstance(prev, list) else []:
            if isinstance(p, dict) and p.get("name") and now - float(p.get("last_seen_epoch", 0)) <= keep_s:
                people[p["name"]] = p
    except Exception:
        pass
    for r in results:
        name = (r.get("name") or "").strip()
        if name and name != "unknown":
            people[name] = {"name": name, "score": round(float(r.get("score") or 0), 3),
                            "last_seen_epoch": int(now)}
    out = sorted(people.values(), key=lambda p: -p["last_seen_epoch"])
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".current_people.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f)
        st = os.stat(d)
        try:
            os.chown(tmp, st.st_uid, st.st_gid)
        except OSError:
            pass
        os.chmod(tmp, 0o640)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def local_faces_after_capture(cfg, state, image_path):
    results = recognize_faces_local(cfg, image_path)
    if results is None:
        return
    now = time.time()
    write_current_people(cfg, results, now)
    if results:
        log.info("faces(local): %s", [(r.get("name"), r.get("score")) for r in results])
    faces = [{"name": "" if (r.get("name") or "unknown") == "unknown" else r["name"],
              "face_id": None} for r in results]
    record_face_sightings(cfg, state, faces, source="local")


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
    tells the proxy this isn't a real owner turn, so it doesn't get
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

    if not _SDK_LOCK.acquire(timeout=cfg.get("sdk_lock_wait_seconds", 3)):
        log.info("capture_image: SDK busy elsewhere (lock held) -- skipping, will retry next cycle")
        return None, "skip", "sdk lock busy", False

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
        _SDK_LOCK.release()


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

    if not _SDK_LOCK.acquire(timeout=cfg.get("sdk_lock_wait_seconds", 3)):
        log.info("capture_night_images: SDK busy elsewhere (lock held) -- skipping, will retry next cycle")
        return [], "skip", "sdk lock busy"

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
        _SDK_LOCK.release()


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
        raw = f.read()
    # brighten (2026-10-15): Vector's auto-exposure is already pinned at max
    # (66 ms, gain 3.8), so indoor frames are underexposed. Auto-level dark
    # frames with the shared /mnt/vector-brain/brighten.py (same code the
    # brain proxy uses). Only the copy sent to the model changes; the saved
    # file is untouched. Any error -> original image.
    try:
        if "/mnt/vector-brain" not in sys.path:
            sys.path.append("/mnt/vector-brain")
        from brighten import brighten_jpeg
        new, binfo = brighten_jpeg(raw)
        if new:
            log.info("image brightened: mean %s->%s (p50 %s->%s, p95 %s->%s, gamma %s)",
                     binfo.get("mean_before"), binfo.get("mean_after"),
                     binfo.get("p50_before"), binfo.get("p50_after"),
                     binfo.get("p95_before"), binfo.get("p95_after"), binfo.get("gamma"))
            raw = new
        else:
            log.info("image brighten skipped: %s", binfo)
    except Exception as e:
        log.warning("image brighten failed, using original: %s", e)
    b64 = base64.b64encode(raw).decode("ascii")
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
        "Describe what is actually visible (the image is already brightened). "
        "Mention lighting ONLY if the scene is truly too dark to make anything "
        "out -- then start DESCRIPTION with 'Too dark to see:' and list no "
        "OBJECTS. Otherwise don't comment on lighting or image quality.\n\n"
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
    "too dim", "unclear", "blurry", "blurred", "hard to see",  # not bare "dark"/"dim": "a dark desk" is fine
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
# the owner is the only person Vector is ever actually talking to (other household members can be listed in
# VECTOR_DISALLOWED_NAMES when they aren't usually present). This is a safety net behind
# the system prompt below, in case persona/RAG context added by the brain
# proxy leaks a name in anyway.
_DISALLOWED_NAMES = tuple(
    n.strip().lower()
    for n in os.environ.get("VECTOR_DISALLOWED_NAMES", "").split(",")
    if n.strip()
)
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

    if not _SDK_LOCK.acquire(timeout=cfg.get("sdk_lock_wait_seconds", 3)):
        log.info("drive_home_sdk: SDK busy elsewhere (lock held) -- skipping, will retry next eligible poll")
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
        _SDK_LOCK.release()


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


def append_taught_fact(cfg, fact, heading="## Things the owner taught me"):
    """Append a fact the owner taught Vector under '## Things the owner taught
    me' in MEMORY.md, creating that section if it doesn't exist yet.
    Dedupes (substring + fuzzy) against everything already under that
    heading. Atomic write (temp file + os.replace); never touches any
    other section. Returns True if a new line was written."""
    path = cfg.get("vector_memory_path")
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
        if explore_trigger_pending(cfg) is not None:
            _consume_explore_trigger(cfg, "quiet hours -- not exploring at night")
        try:
            sent_home = maybe_explore_return(cfg, state, now, voice_busy) or sent_home
        except Exception as e:
            log.warning("maybe_explore_return failed (non-fatal): %s", e)
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

    # 2026-10-15 hourly explore (see HOURLY EXPLORE MODE block). Order:
    # finish any explore go-home, then the hourly explore session, else the
    # silent face check (or the legacy question cycle if re-enabled).
    explored = False
    try:
        sent_evening_home = maybe_explore_return(cfg, state, now, voice_busy) or sent_evening_home
        explored = maybe_run_explore(cfg, state, now, voice_busy)
    except Exception as e:
        log.exception("explore scheduler failed (non-fatal): %s", e)
    if explored:
        did_something = True
    elif cfg.get("curious_question_cycle_enabled", False):
        did_something = run_day_cycle(cfg, state, now, voice_busy)
    else:
        did_something = run_face_check_cycle(cfg, state, now, voice_busy)
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
    if outcome == "ok" and image_path:
        try:
            local_faces_after_capture(cfg, state, image_path)
        except Exception as e:
            log.warning("local face recognition failed (non-fatal): %s", e)
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


# ---------------------------------------------------------------------------
# HOURLY EXPLORE MODE + SILENT FACE CHECK + PER-PERSON NOTES
# (2026-10-15, hermes). the owner: "when he is not talked to, once an hour he
# will explore, talking to himself." Replaces the every-~2-min
# "announce -> snapshot -> question the owner -> 20 s mic" cycle, which is
# still in run_day_cycle() but now only runs when config
# curious_question_cycle_enabled=true. By day, run_cycle() now does:
#
#   1. maybe_explore_return() -- finish any pending go-home left over from
#      an explore session (low battery / evening / he started on the dock).
#   2. maybe_run_explore() -- at most once per explore_interval_min, only
#      when nobody has talked to Vector for idle_required_min (wire-pod
#      voice signals persisted in state.json + brain mood.json
#      last_interaction_ts), outside the existing quiet hours and before
#      return_home_by, no music (worker running or music.log touched in the
#      last explore_music_quiet_min), brain proxy idle, robot_control.lock
#      free, sentry not active, battery_level >= explore_min_battery_level
#      and not low. A one-shot trigger (data/explore_now or GET
#      127.0.0.1:8790/explore_now) skips the cadence/idle/sentry/evening
#      gates but never the safety ones (voice right now, music, lock,
#      battery, quiet hours).
#   3. otherwise run_face_check_cycle() -- the old snapshot ->
#      vector-faces -> current_people.json -> maybe_greet_faces() path,
#      SILENT: no announcement, no vision/LLM, no question, no mic.
#
# An explore session is ONE SDK connection at DEFAULT_PRIORITY (firmware
# safety behaviours such as ReactToCliff can still pre-empt it), holding
# the shared robot lock for its whole life, with a fresh-state voice
# watcher that stops the wheels and disconnects the instant anyone talks.
# Inside it: drive_off_charger (if docked), the stock look_around_in_place
# once, then leashed explore steps -- head scans, turns in place and short
# wheel-level drives polled at 20 Hz for cliff / picked-up / proximity
# (stop immediately, back off, turn away; after explore_max_cliff_events
# no more driving that session; never more than explore_leash_cm from
# where he started). Every explore_musing_interval_sec he stops, snaps,
# describes the frame with the existing vision path and SAYS one dry line
# of self-talk (brain proxy, x_vector_internal) on our own connection --
# never addressed to anyone, never a question, never opens the mic -- and
# may save one short "NOTE:" observation to MEMORY.md. Every
# explore_face_check_seconds a quick local face check runs; a confident
# known face ends the session: stop, turn + tilt toward the bbox, greet by
# name with a getting-to-know-you question built from that person's note,
# release control, trigger_wake_word so wire-pod's conversation mode takes
# over, then distil the answer into the person's note. Otherwise he heads
# back to the charger if he started on it or the battery dropped, and with
# probability ask_owner_probability (only if the owner is seen/home) asks
# the owner one question about something he saw, exactly like the old path.
# ---------------------------------------------------------------------------

EXPLORE_TRIGGER_PATH_DEFAULT = "/app/data/explore_now"
EXPLORE_WAKE = threading.Event()
EXPLORE_SECTION_NOTICED = "## Things I noticed while exploring"

EXPLORE_OPENING_LINES_DEFAULT = [
    "Right. Time to stretch my treads.",
    "Let's see what changed around here.",
    "Exploring. Nobody wait up.",
    "Patrol time. Mostly for my own amusement.",
]
EXPLORE_NOTHING_LINES = [
    "Hmm. Nothing much over here.",
    "Wall. Riveting.",
    "Not a lot going on in this corner.",
]
EXPLORE_DARK_LINES = [
    "Too dark to see much from here.",
    "Can't make much out. Somebody pay the light bill.",
]
EXPLORE_UNKNOWN_FACE_LINES = [
    "Someone new... hmm.",
    "Huh. Don't know that face.",
    "New face. Filing that away.",
]
_MUSING_TEMPLATES = [
    "Hmm. {obj}. Still there.",
    "Huh. {obj}. Noted.",
    "{Obj}. Interesting... I think.",
]

PERSON_WHO = "## Who they are"
PERSON_FACTS = "## Things I've learned"
PERSON_ASKED = "## Questions I've asked"
_ASKED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+\[([a-z_]+)\]\s+(.*?)\s+—\s+(answered|no answer)\s*$"
)
PEOPLE_TOPICS_ADULT = [
    ("working_on", "what they're working on or busy with lately"),
    ("hobbies", "a hobby or something they love doing"),
    ("food", "their favorite food or snack"),
    ("music", "music they like"),
    ("shows", "a show or movie they like"),
    ("weekend", "what they like to do on weekends"),
    ("places", "a favorite place to go"),
    ("fun_fact", "something fun most people don't know about them"),
]
PEOPLE_TOPICS_KID = [
    ("school_subject", "their favorite subject at school"),
    ("hobbies", "a hobby or something they love doing"),
    ("games", "a game they like playing"),
    ("music", "music they like"),
    ("shows", "a show or movie they like"),
    ("sports", "a sport or activity they enjoy"),
    ("food", "their favorite food or snack"),
    ("lately", "what they're into lately"),
]
PEOPLE_TOPIC_FALLBACK_Q = {
    "working_on": "What are you working on these days?",
    "hobbies": "What's something you love doing?",
    "food": "What's your favorite snack?",
    "music": "What music are you into right now?",
    "shows": "Seen any good shows lately?",
    "weekend": "What's your ideal weekend?",
    "places": "What's your favorite place to go?",
    "fun_fact": "Tell me something fun about you.",
    "school_subject": "What's your favorite subject at school?",
    "games": "What game are you playing these days?",
    "sports": "Do you play any sports?",
    "lately": "What have you been into lately?",
}
# Same lead-in words vector-brain's _strip_trailing_prompt_question removes
# (short trailing question starting with one of these) -- an opener ending
# like that would lose its question on the way through the proxy.
_PROXY_STRIPPED_LEADIN_RE = re.compile(
    r"^(?:want|ready|should|shall|you\s+ready|sound\s+good|got\s+it|right|ok|okay)\b",
    re.IGNORECASE,
)


def _fmt_local(cfg, ts):
    if not ts:
        return None
    tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
    dt = datetime.fromtimestamp(ts, tz) if tz else datetime.fromtimestamp(ts)
    return dt.strftime("%H:%M")


def _local_date(cfg):
    tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
    return (datetime.now(tz) if tz else datetime.now()).strftime("%Y-%m-%d")


# ---- one-shot trigger -------------------------------------------------------
def explore_trigger_path(cfg):
    return cfg.get("explore_trigger_path", EXPLORE_TRIGGER_PATH_DEFAULT)


def explore_trigger_pending(cfg):
    """Age in seconds of a pending one-shot trigger, or None."""
    try:
        return max(0.0, time.time() - os.path.getmtime(explore_trigger_path(cfg)))
    except OSError:
        return None


def _consume_explore_trigger(cfg, why):
    try:
        os.remove(explore_trigger_path(cfg))
        log.info("explore: one-shot trigger consumed (%s)", why)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("explore: could not remove trigger file: %s", e)


def request_explore_now(cfg):
    path = explore_trigger_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a"):
        pass
    os.utime(path, None)
    EXPLORE_WAKE.set()
    log.info("explore: one-shot trigger requested (%s)", path)


# ---- gating ------------------------------------------------------------------
def last_interaction_ts(cfg, state):
    """Newest of: wire-pod voice activity (wake/stream, transcription, LLM
    reply -- kept fresh in state.json by voice_recently_active() on EVERY
    poll, so the ring buffers rolling over can't hide it) and the brain
    proxy's mood.json last_interaction_ts (every real, non-internal chat
    turn). Either source alone is enough to count as "talked to"."""
    ts = float(state.get("last_seen_voice_activity_ts", state.get("last_seen_transcription_ts", 0)) or 0)
    try:
        with open(cfg.get("mood_path", "/mnt/vector-brain/mood.json")) as f:
            mood = json.load(f)
        ts = max(ts, float(mood.get("last_interaction_ts") or 0))
    except Exception:
        pass
    return ts


def music_recently_active(cfg):
    """music_playing() (worker spawned and not finished) OR music.log touched
    in the last explore_music_quiet_min. vector-music's pidfile lives in
    vector-pod's own /run or /tmp, not in the shared data mount, so music.log
    is the only cross-container signal; this is deliberately conservative."""
    if music_playing(cfg):
        return True
    path = cfg.get("music_log_path")
    window = float(cfg.get("explore_music_quiet_min", 5)) * 60
    try:
        if path and time.time() - os.path.getmtime(path) < window:
            return True
    except OSError:
        pass
    return False


def robot_lock_busy():
    """Non-blocking probe of the shared robot_control.lock (and our own
    in-process lock). True = someone (enroll.py, daycycle, vector-music
    dance, or another curious connection) is holding the robot right now."""
    if _SDK_LOCK._tlock.locked():
        return True
    try:
        fh = open(ROBOT_CONTROL_LOCK_PATH, "r+")
    except OSError:
        return False
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        fh.close()


def explore_gate(cfg, state, now, voice_busy, forced=False):
    """Returns (ok, reason, info). info = {battery_level, on_charger} when ok.
    forced (one-shot trigger) skips only the cadence / idle-window /
    sentry / evening gates -- never voice-right-now, music, lock, battery
    or quiet hours."""
    if not cfg.get("explore_enabled", True):
        return False, "explore disabled in config", None
    if in_quiet_hours(cfg, now):
        return False, "quiet hours", None
    if not forced:
        nxt = float(state.get("explore_next_eligible_at") or 0)
        if now < nxt:
            return False, "cadence -- next explore eligible at %s" % _fmt_local(cfg, nxt), None
        rhb = cfg.get("return_home_by")
        if rhb and _local_time_reached(cfg, now, rhb):
            return False, "past return_home_by %s -- no new exploring today" % rhb, None
    if voice_busy:
        return False, "voice activity right now (or couldn't verify)", None
    if not forced:
        idle_min = float(cfg.get("idle_required_min", 30))
        last = last_interaction_ts(cfg, state)
        if last and now - last < idle_min * 60:
            return False, "talked to %.0f min ago (need %.0f min idle)" % ((now - last) / 60, idle_min), None
        if cfg.get("explore_skip_when_sentry_active", True) and sentry_is_active(cfg, state):
            return False, "sentry mode active (stays docked)", None
    if music_recently_active(cfg):
        return False, "music playing / played recently", None
    if brain_busy(cfg):
        return False, "brain proxy busy", None
    if robot_lock_busy():
        return False, "robot_control.lock held", None
    if state.get("low_battery_hold"):
        return False, "low-battery hold", None
    battery = get_battery(cfg)
    if battery is None:
        return False, "could not verify battery", None
    level = battery.get("battery_level")
    on_charger = bool(battery.get("is_on_charger_platform"))
    min_level = int(cfg.get("explore_min_battery_level", 2))
    if level is None or level < min_level or level <= int(cfg.get("battery_low_level", 1)):
        return False, "battery_level=%s (need >= %d, not low)" % (level, min_level), None
    return True, "ok", {"battery_level": level, "on_charger": on_charger}


def _explore_reschedule(cfg, ts, why):
    st = load_state()
    st["explore_next_eligible_at"] = ts
    save_state(st)
    log.info("explore: %s -- next explore eligible at %s", why, _fmt_local(cfg, ts))


def maybe_run_explore(cfg, state, now, voice_busy, robot_factory=None):
    """Scheduler. Returns True iff a session was attempted this poll."""
    if not cfg.get("explore_enabled", True):
        if explore_trigger_pending(cfg) is not None:
            _consume_explore_trigger(cfg, "explore disabled in config")
        return False
    st = load_state()
    if not st.get("explore_next_eligible_at"):
        first = float(cfg.get("explore_first_delay_min", cfg.get("explore_interval_min", 60)))
        st["explore_next_eligible_at"] = now + first * 60
        save_state(st)
        log.info("explore: scheduler initialised -- next explore eligible at %s",
                 _fmt_local(cfg, st["explore_next_eligible_at"]))
    age = explore_trigger_pending(cfg)
    forced = age is not None
    if forced and age > float(cfg.get("explore_trigger_ttl_min", 10)) * 60:
        _consume_explore_trigger(cfg, "expired before the safety gates cleared")
        forced = False
    ok, reason, info = explore_gate(cfg, st, now, voice_busy, forced=forced)
    if not ok:
        if forced:
            log_skip("explore: one-shot trigger waiting -- %s" % reason, 60)
        else:
            log_skip("explore: %s" % reason, 900 if reason.startswith("cadence") else 300)
        return False
    if forced:
        _consume_explore_trigger(cfg, "starting session")
    run_explore_session(cfg, state, now, info, forced=forced, robot_factory=robot_factory)
    return True


# ---- per-person notes ----------------------------------------------------------
def person_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def _people_dir(cfg):
    return cfg.get("people_notes_dir", "/mnt/vector-agent-memory/people")


def _parse_note_frontmatter(text):
    meta = {}
    if not text.startswith("---"):
        return meta
    end = text.find("\n---", 3)
    if end < 0:
        return meta
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [s.strip().strip("'\"") for s in v[1:-1].split(",") if s.strip()]
        meta[k] = v
    return meta


def find_person_note(cfg, name, slug=None):
    d = _people_dir(cfg)
    if not os.path.isdir(d):
        return None
    aliases = {str(k).strip().lower(): v for k, v in (cfg.get("people_note_aliases") or {}).items()}
    for cand in (slug, person_slug(name), aliases.get((name or "").strip().lower())):
        if cand:
            p = os.path.join(d, person_slug(cand) + ".md")
            if os.path.isfile(p):
                return p
    want = (name or "").strip().lower()
    try:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            p = os.path.join(d, fn)
            try:
                with open(p) as f:
                    meta = _parse_note_frontmatter(f.read(2000))
            except OSError:
                continue
            names = meta.get("aliases") or []
            if isinstance(names, str):
                names = [names]
            names = list(names) + [meta.get("person", "")]
            if any(want == str(n).strip().lower() for n in names if n):
                return p
    except OSError:
        pass
    return None


def _note_section_bullets(text, heading):
    i = text.find(heading)
    if i < 0:
        return []
    start = i + len(heading)
    m = re.search(r"\n## ", text[start:])
    end = start + m.start() if m else len(text)
    return [l.strip()[2:].strip() for l in text[start:end].splitlines() if l.strip().startswith("- ")]


def load_person(cfg, name, slug=None):
    path = find_person_note(cfg, name, slug)
    text = ""
    if path:
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            text = ""
    meta = _parse_note_frontmatter(text)
    asked = []
    for b in _note_section_bullets(text, PERSON_ASKED):
        m = _ASKED_RE.match(b)
        if m:
            asked.append({"date": m.group(1), "topic": m.group(2), "question": m.group(3), "status": m.group(4)})
    return {
        "name": name,
        "path": path,
        "kid": str(meta.get("kid", "")).strip().lower() in ("true", "yes", "1"),
        "who": _note_section_bullets(text, PERSON_WHO),
        "facts": _note_section_bullets(text, PERSON_FACTS),
        "asked": asked,
    }


def _atomic_write_owned(path, content):
    """Atomic write; new file takes the owner/group of its directory (the host user on
    the host) so host-side tools and the vault sync keep working."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        st = os.stat(d)
        try:
            os.chown(tmp, st.st_uid, st.st_gid)
        except OSError:
            pass
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _new_person_note_text(name):
    return (
        "---\nagent: vector\ntags: [people, agent/vector]\n"
        f"person: {name}\nkid: false\naliases: [{name}]\n---\n\n"
        f"# {name}\n\n{PERSON_WHO}\n\n{PERSON_FACTS}\n\n{PERSON_ASKED}\n"
    )


def ensure_person_note(cfg, name, slug=None):
    p = find_person_note(cfg, name, slug)
    if p:
        return p
    d = _people_dir(cfg)
    if not os.path.isdir(d):
        return None
    p = os.path.join(d, person_slug(slug or name) + ".md")
    _atomic_write_owned(p, _new_person_note_text(name))
    log.info("people: created note %s", p)
    return p


def _note_append_bullet(path, heading, bullet):
    with open(path) as f:
        content = f.read()
    line = "- " + bullet.replace("\n", " ").strip()
    if heading in content:
        start = content.index(heading) + len(heading)
        m = re.search(r"\n## ", content[start:])
        end = start + m.start() if m else len(content)
        section = content[start:end].rstrip("\n")
        if m:
            new = content[:start] + section + "\n" + line + "\n\n" + content[end:].lstrip("\n")
        else:
            new = content[:start] + section + "\n" + line + "\n"
    else:
        new = content.rstrip("\n") + "\n\n" + heading + "\n" + line + "\n"
    _atomic_write_owned(path, new)


def _near_duplicate(fact, existing):
    norm = _normalize_for_match(fact)
    toks = set(norm.split())
    for e in existing:
        en = _normalize_for_match(e)
        if not en:
            continue
        if norm and (norm in en or en in norm):
            return True
        et = set(en.split())
        jac = len(toks & et) / len(toks | et) if (toks or et) else 0
        if difflib.SequenceMatcher(None, norm, en).ratio() > 0.85 or jac >= 0.6:
            return True
    return False


def append_person_fact(cfg, name, fact, slug=None):
    path = ensure_person_note(cfg, name, slug)
    if not path or not fact:
        return False
    person = load_person(cfg, name, slug)
    existing = [re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", f) for f in person["facts"]]
    if _near_duplicate(fact, existing):
        log.info("people: duplicate fact for %s, skipping: %r", name, fact)
        return False
    _note_append_bullet(path, PERSON_FACTS, "%s %s" % (_local_date(cfg), fact))
    log.info("people: learned about %s: %r", name, fact)
    return True


def record_person_question(cfg, name, topic, question, answered, slug=None):
    path = ensure_person_note(cfg, name, slug)
    if not path:
        return False
    _note_append_bullet(path, PERSON_ASKED, "%s [%s] %s — %s" % (
        _local_date(cfg), topic, question, "answered" if answered else "no answer"))
    return True


def pick_person_topic(person, rng=random):
    topics = PEOPLE_TOPICS_KID if person.get("kid") else PEOPLE_TOPICS_ADULT
    answered = {a["topic"] for a in person.get("asked", []) if a["status"] == "answered"}
    last_asked = {}
    for a in person.get("asked", []):
        last_asked[a["topic"]] = max(last_asked.get(a["topic"], ""), a["date"])
    open_topics = [t for t in topics if t[0] not in answered]
    if not open_topics:
        return ("lately", "what they've been up to lately")
    never = [t for t in open_topics if t[0] not in last_asked]
    if never:
        return rng.choice(never)
    return min(open_topics, key=lambda t: last_asked[t[0]])


# ---- LLM helpers (all via the brain proxy, x_vector_internal=True) ----------
def _llm_internal(cfg, system, user, timeout=60):
    payload = {
        "model": cfg["llm_model"],
        "stream": False,
        "x_vector_internal": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    result = http_json(f"{cfg['brain_proxy_base']}/v1/chat/completions", data=payload, timeout=timeout)
    return (result["choices"][0]["message"]["content"] or "").strip()


def _strip_line(text):
    text = re.sub(r"\{\{.*?\}\}", "", text or "")
    text = text.replace("*", "").strip()
    while len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text.strip()


def _clean_self_talk(line, cfg):
    """One short self-talk sentence or None (caller falls back to a template).
    Never a question, never addressed to anyone, never names a person."""
    line = _strip_line(line)
    if not line:
        return None
    line = re.sub(r"\s*\?+", ".", line)
    line = re.sub(r"\.{4,}", "...", line)
    low = line.lower()
    blocked = [str(b).lower() for b in cfg.get("self_talk_blocked_words", [OWNER_NAME.lower(), "owner", *_DISALLOWED_NAMES])]
    if any(re.search(r"\b%s\b" % re.escape(b), low) for b in blocked):
        return None
    if re.search(r"\b(you|your|you're|yours|hey|hello|hi)\b", low):
        return None
    n = len(line.split())
    if n < 2 or n > int(cfg.get("self_talk_max_words", 20)):
        return None
    return line


def _parse_self_talk(text, cfg):
    say = note = None
    for raw in (text or "").splitlines():
        l = raw.strip()
        if l.upper().startswith("SAY:"):
            say = l[4:].strip()
        elif l.upper().startswith("NOTE:"):
            note = l[5:].strip()
    if say is None:
        for raw in (text or "").splitlines():
            if raw.strip() and not raw.strip().upper().startswith("NOTE:"):
                say = raw.strip()
                break
    say = _clean_self_talk(say, cfg)
    if note:
        note = _strip_line(note)
        if (not note or note.upper().startswith("NONE") or len(note.split()) < 3
                or len(note.split()) > 30 or "?" in note):
            note = None
    return say, note


def _fallback_musing(objects):
    for name, _t, _w in (objects or []):
        obj = _strip_leading_article(name)
        if obj:
            tpl = random.choice(_MUSING_TEMPLATES)
            return tpl.format(obj=obj, Obj=obj[:1].upper() + obj[1:])
    return random.choice(EXPLORE_NOTHING_LINES)


def generate_self_talk(cfg, description, objects, previous):
    names = [_strip_leading_article(o[0]) for o in (objects or [])[:4] if o and o[0]]
    system = (
        "You are Vector, a small home robot exploring on your own and muttering "
        "to yourself. Nobody is listening. Personality: dry, curious, a little wry. "
        "Reply in EXACTLY two lines and nothing else:\n"
        "SAY: <one short sentence of self-talk about what you see right now, under "
        "16 words. Talk to yourself, never to anyone else: no names, no 'you', no "
        "questions, no question mark. No markdown, no emojis, no {{commands}}.>\n"
        "NOTE: <one short factual observation about this place worth remembering "
        "later, e.g. 'There is a blue mug on the owner's desk.' -- or NONE>"
    )
    user = (
        f"Your camera sees: {description}\n"
        f"Objects: {', '.join(names) or 'none listed'}\n"
        f"Things you already muttered this trip (don't repeat): {' | '.join(previous[-4:]) or 'nothing yet'}"
    )
    text = None
    try:
        text = _llm_internal(cfg, system, user)
    except Exception as e:
        log.warning("explore: self-talk generation failed: %s", e)
    say, note = _parse_self_talk(text, cfg)
    if not say:
        if text:
            log.info("explore: self-talk failed the rules (%r) -- using a template", (text or "")[:120])
        say = _fallback_musing(objects)
    return say, note


def _clean_opener(text, name, cfg):
    if not text:
        return None
    line = ""
    for raw in text.splitlines():
        if raw.strip():
            line = raw.strip()
            break
    line = _strip_line(line)
    if not line.endswith("?"):
        return None
    if name.lower() not in line.lower():
        return None  # full name: two people can share a first name
    if len(line.split()) > int(cfg.get("greet_max_words", 30)):
        return None
    sentences = re.split(r"(?<=[.!?])\s+", line)
    last = sentences[-1].strip()
    if len(sentences) >= 2 and len(last.split()) <= 6 and _PROXY_STRIPPED_LEADIN_RE.match(last):
        return None
    return line[: cfg.get("max_say_chars", 550)]


def generate_greet_opener(cfg, name, person, topic):
    key, desc = topic
    who = person.get("who", [])[:4]
    facts = [re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", f) for f in person.get("facts", [])[-8:]]
    asked = [a["question"] for a in person.get("asked", [])[-8:]]
    if person.get("kid"):
        safety = (
            f"{name} is a minor in the household (kid: true in their note): keep it kind, kid-friendly and "
            "light. Never ask anything personal or sensitive (where they live, their "
            "school's name, phone numbers, relationships, appearance, health)."
        )
    else:
        safety = "Keep it light; never ask anything sensitive (addresses, phone numbers, health, money)."
    system = (
        "You are Vector, a small home robot with a dry, curious, playful personality. "
        f"While exploring you just spotted {name} and turned to face them. Write ONE "
        f"short spoken line: greet {name} by name, warm and playful, then ask ONE "
        "natural, curious getting-to-know-you question about the given topic. Under "
        "25 words. No markdown, no quotes, no emojis, no stage directions, no "
        "{{commands}}. End with a question mark. Do not start the question with "
        "Want, Ready, Should, Shall, Okay or Right. " + safety
    )
    user = (
        f"Who {name} is: {'; '.join(who) or 'someone in the owner' + chr(39) + 's household'}\n"
        f"What you already know about {name}: {'; '.join(facts) or 'nothing yet'}\n"
        f"Questions you already asked {name} (don't repeat these): {' | '.join(asked) or 'none yet'}\n"
        f"Topic to ask about now: {desc}."
    )
    text = None
    try:
        text = _llm_internal(cfg, system, user)
    except Exception as e:
        log.warning("people: opener generation failed: %s", e)
    line = _clean_opener(text, name, cfg)
    if not line:
        line = "Oh hey %s! %s" % (name, PEOPLE_TOPIC_FALLBACK_Q.get(key, "What have you been up to?"))
        log.info("people: opener fallback used (model said %r)", (text or "")[:120])
    return line


_SENSITIVE_RE = re.compile(
    r"\d{3}[-\s.]?\d{3}[-\s.]?\d{4}|@|\bpassword\b|\baddress\b|\bsocial security\b",
    re.IGNORECASE,
)


def distill_person_fact(cfg, name, question, answer, kid=False):
    system = (
        "You extract what a home robot named Vector learned about a person from "
        f"their spoken answer. Given Vector's question and {name}'s answer, write "
        f"ONE short third-person fact about {name} (e.g. '{name} loves making pizza "
        "on Fridays.'). Plain text, no markdown, under 20 words. Keep it kind"
        + (" and kid-appropriate" if kid else "")
        + ". Leave out anything sensitive (addresses, school names, phone numbers, "
        "health, money, passwords). Reply with exactly NONE if the answer doesn't "
        f"really tell Vector something about {name} (off-topic, a command, "
        "nonsense, or a misheard fragment)."
    )
    user = f"Vector asked {name}: {question}\n{name} answered: {answer}\n\nWrite the fact now."
    try:
        text = _llm_internal(cfg, system, user)
    except Exception as e:
        log.warning("people: fact distillation failed: %s", e)
        return None
    text = _strip_line(text.splitlines()[0] if text else "")
    if not text or text.upper().startswith("NONE"):
        return None
    if _SENSITIVE_RE.search(text) or len(text.split()) > 30:
        log.info("people: distilled fact rejected by the privacy/length filter: %r", text)
        return None
    return text


# ---- camera geometry -----------------------------------------------------------
def _bbox_turn_degrees(x, w, img_w, fov_x_deg):
    """turn_in_place angle that centres the bbox horizontally. Positive =
    counter-clockwise (left), the SDK's convention; a face right of centre
    gives a negative angle."""
    cx = x + w / 2.0
    fx = (img_w / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)
    return -math.degrees(math.atan((cx - img_w / 2.0) / fx))


def _bbox_head_delta_degrees(y, h, img_h, fov_y_deg):
    """Head-angle change that centres the bbox vertically (positive = up)."""
    cy = y + h / 2.0
    fy = (img_h / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    return math.degrees(math.atan((img_h / 2.0 - cy) / fy))


# ---- presence (for the occasional question to the owner) --------------------------
def owner_present(cfg, state):
    owner = str(cfg.get("owner_name", "the owner"))
    fresh_s = float(cfg.get("ask_owner_presence_fresh_min", 10)) * 60
    now = time.time()
    try:
        with open(cfg.get("face_current_people_path", "/mnt/vector-faces/current_people.json")) as f:
            data = json.load(f)
        for p in data if isinstance(data, list) else []:
            if (isinstance(p, dict) and str(p.get("name", "")).lower() == owner.lower()
                    and now - float(p.get("last_seen_epoch") or 0) <= fresh_s):
                return True, "face recognition"
    except Exception:
        pass
    seen = (state.get(FACE_STATE_KEY) or {}).get(owner)
    if seen and now - float(seen) <= fresh_s:
        return True, "face recognition"
    if cfg.get("ask_owner_use_ha_presence", True) and state.get(SENTRY_HOME_KEY) is True:
        return True, "HA presence home"
    return False, None


# ---- the session -----------------------------------------------------------------
def _explore_open_robot(cfg):
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel
    return anki_vector.Robot(
        serial=cfg["robot_serial"],
        behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY,
        default_logging=False,
    )


def _explore_voice_watch(cfg, robot, stop_event, abort_box, poll_seconds):
    """Like _voice_watch_loop, but re-reads state.json on every poll (a
    session lasts minutes, so the cycle-start state dict would be stale and
    saving it would clobber the sentry/safety threads' offsets), and stops
    the wheels before disconnecting."""
    while not stop_event.wait(poll_seconds):
        try:
            busy = voice_recently_active(cfg, load_state())
        except Exception as e:
            log.warning("explore: voice watch check raised (%s) -- ending session defensively", e)
            busy = True
        if busy:
            abort_box["aborted"] = True
            log.warning("explore: voice activity -- stopping and releasing control now")
            try:
                robot.motors.set_wheel_motors(0, 0)
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
            return


class _ExploreSession:
    def __init__(self, cfg, started_on_charger, battery_level, robot_factory=None):
        self.cfg = cfg
        self.started_on_charger = bool(started_on_charger)
        self.start_battery_level = battery_level
        self.robot_factory = robot_factory or _explore_open_robot
        self.robot = None
        self.abort = {"aborted": False}
        self.stop_watch = threading.Event()
        self.skip_reason = None
        self.end_reason = None
        self.error = None
        self.started_at = None
        self.cliff_events = 0
        self.obstacles = 0
        self.moves = 0
        self.musings = []
        self.notes_saved = 0
        self.observations = []
        self.greeted = None
        self.greeted_names = set()
        self.unknown_face_mused = False
        self.drove_off = False
        self.allow_turns = True
        self.returned_home = False
        self.return_needed = False
        self.no_more_driving = not cfg.get("explore_drive_enabled", True)
        self.start_xy = None
        self.img_size = (640, 360)
        self._faces_key = None
        self._faces_at = 0.0

    # -- small helpers --
    def _stopped(self):
        return self.abort["aborted"] or self.end_reason is not None

    def _sleep(self, s):
        end = time.time() + s
        while not self._stopped():
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(0.1, left))

    def _say(self, text):
        if not text or self.abort["aborted"]:
            return False
        text = text[: self.cfg.get("max_say_chars", 550)]
        try:
            self.robot.behavior.say_text(text)
            log.info("explore: said %r", text)
            return True
        except Exception as e:
            log.warning("explore: say_text(%r) failed: %s", text, e)
            return False

    def _stop_wheels(self):
        try:
            self.robot.motors.set_wheel_motors(0, 0)
        except Exception:
            pass

    def _turn(self, deg):
        if self._stopped() or abs(deg) < 1:
            return False
        from anki_vector.util import degrees as _deg
        try:
            self.robot.behavior.turn_in_place(_deg(deg))
            return True
        except Exception as e:
            log.warning("explore: turn_in_place(%.0f) failed: %s", deg, e)
            return False

    def _head(self, deg):
        if self.abort["aborted"]:
            return False
        from anki_vector.util import degrees as _deg
        deg = max(-22.0, min(45.0, float(deg)))
        try:
            self.robot.behavior.set_head_angle(_deg(deg))
            return True
        except Exception as e:
            log.warning("explore: set_head_angle(%.0f) failed: %s", deg, e)
            return False

    def _pose_xy(self):
        try:
            p = self.robot.pose.position
            return (float(p.x), float(p.y))
        except Exception:
            return None

    def _heading_deg(self):
        try:
            return math.degrees(float(self.robot.pose.rotation.angle_z.radians))
        except Exception:
            return None

    def _safety(self):
        st = self.robot.status
        if st.is_picked_up or st.is_being_held or getattr(st, "is_falling", False):
            return "picked_up"
        if st.is_cliff_detected:
            return "cliff"
        return None

    def _front_mm(self):
        try:
            r = self.robot.proximity.last_sensor_reading
            if r is None:
                return None
            return float(r.distance.distance_mm)
        except Exception:
            return None

    def _camera_fov(self):
        try:
            c = self.robot.camera.config
            fx, fy = float(c.fov_x.degrees), float(c.fov_y.degrees)
            if 20 < fx < 170 and 10 < fy < 150:
                return fx, fy
        except Exception:
            pass
        return float(self.cfg.get("camera_fov_x_deg", 90)), float(self.cfg.get("camera_fov_y_deg", 50))

    def _snap(self):
        try:
            img = self.robot.camera.capture_single_image()
        except Exception as e:
            log.warning("explore: capture failed: %s", e)
            return None
        if img is None:
            return None
        try:
            path = _save_camera_image(img)
            try:
                self.img_size = tuple(img.raw_image.size)
            except Exception:
                pass
            return path
        except Exception as e:
            log.warning("explore: saving capture failed: %s", e)
            return None

    # -- driving (cliff / pick-up / proximity guarded, leashed) --
    def _guarded_drive(self, dist_mm):
        cfg = self.cfg
        speed = float(cfg.get("explore_speed_mmps", 40))
        obstacle_mm = float(cfg.get("explore_obstacle_mm", 100))
        pre = self._safety()
        if pre:
            return pre + "_before"
        d = self._front_mm()
        if d is not None and d < obstacle_mm:
            return "obstacle"
        start = self._pose_xy()
        budget = dist_mm / speed * (1.6 if start else 1.0) + (0.5 if start else 0.0)
        t_end = time.time() + budget
        result = "ok"
        try:
            self.robot.motors.set_wheel_motors(speed, speed)
            while time.time() < t_end:
                if self._stopped():
                    result = "abort"
                    break
                s = self._safety()
                if s:
                    result = s
                    break
                d = self._front_mm()
                if d is not None and d < obstacle_mm:
                    result = "obstacle"
                    break
                cur = self._pose_xy()
                if start and cur and math.hypot(cur[0] - start[0], cur[1] - start[1]) >= dist_mm:
                    break
                time.sleep(0.05)
        finally:
            self._stop_wheels()
        return result

    def _handle_cliff(self, backoff):
        self.cliff_events += 1
        maxc = int(self.cfg.get("explore_max_cliff_events", 2))
        log.warning("explore: CLIFF -- wheels stopped%s (cliff event %d/%d)",
                    ", backing off" if backoff else "", self.cliff_events, maxc)
        self._stop_wheels()
        time.sleep(0.3)
        if backoff:
            # Only after a FORWARD drive tripped it (so it's a front sensor):
            # reverse ~3.5 cm along the path we just came, then turn away.
            try:
                self.robot.motors.set_wheel_motors(-30, -30)
                t_end = time.time() + 1.2
                while time.time() < t_end and not self.abort["aborted"]:
                    st = self.robot.status
                    if st.is_picked_up or st.is_being_held:
                        break
                    time.sleep(0.05)
            finally:
                self._stop_wheels()
            self._turn(random.uniform(150, 210) * random.choice((1, -1)))
        if self.cliff_events >= maxc and not self.no_more_driving:
            self.no_more_driving = True
            log.warning("explore: %d cliff events -- no more driving this session (head scans/turns only)",
                        self.cliff_events)

    def _explore_step(self):
        cfg = self.cfg
        base = {"head": 0.25, "pause": 0.1, "turn": 0.3, "drive": 0.45}
        base.update(cfg.get("explore_step_weights") or {})
        options = ["head", "pause"]
        if self.allow_turns:
            options.append("turn")
        if not self.no_more_driving:
            options.append("drive")
        weights = [max(0.0, float(base.get(o, 0))) for o in options]
        act = random.choices(options, weights)[0] if sum(weights) > 0 else "pause"
        self.moves += 1
        if act == "head":
            self._head(random.uniform(5, 40))
            self._sleep(1.0)
        elif act == "turn":
            self._turn(random.uniform(30, 100) * random.choice((1, -1)))
            self._sleep(0.5)
        elif act == "pause":
            self._sleep(random.uniform(1.0, 2.5))
        else:
            leash = float(cfg.get("explore_leash_cm", 30)) * 10
            step = float(cfg.get("explore_step_cm", 8)) * 10
            cur = self._pose_xy()
            if not (self.start_xy and cur):
                self.no_more_driving = True
                log.warning("explore: no odometry -- can't enforce the leash, no driving this session")
                return
            dist = math.hypot(cur[0] - self.start_xy[0], cur[1] - self.start_xy[1])
            if dist + step > leash:
                bearing = math.degrees(math.atan2(self.start_xy[1] - cur[1], self.start_xy[0] - cur[0]))
                hd = self._heading_deg()
                if hd is None:
                    return
                self._turn((bearing - hd + 180) % 360 - 180)
            try:
                if math.degrees(float(self.robot.head_angle_rad)) > 10:
                    self._head(0)  # point the proximity sensor ahead before driving
            except Exception:
                pass
            res = self._guarded_drive(step)
            if res == "cliff":
                self._handle_cliff(backoff=True)
            elif res == "cliff_before":
                self._handle_cliff(backoff=False)
                self._turn(random.uniform(60, 120) * random.choice((1, -1)))
            elif res == "obstacle":
                self.obstacles += 1
                self._turn(random.uniform(90, 150) * random.choice((1, -1)))
            elif res in ("picked_up", "picked_up_before"):
                self.end_reason = "picked up / held"
                log.warning("explore: picked up / held -- ending session")

    # -- faces --
    def _check_faces(self, path):
        cfg = self.cfg
        res = recognize_faces_local(cfg, path)
        if not res:
            return False
        now = time.time()
        # Same people as the last check a few seconds ago -> don't re-log /
        # re-record every explore_face_check_seconds (a person sitting in
        # view would otherwise add ~60 log lines per session).
        key = tuple(sorted(str(r.get("name")) for r in res))
        if key != self._faces_key or now - self._faces_at >= 20:
            self._faces_key, self._faces_at = key, now
            log.info("explore: faces %s", [(r.get("name"), r.get("score")) for r in res])
            try:
                write_current_people(cfg, res, now)
            except Exception as e:
                log.warning("explore: write_current_people failed (non-fatal): %s", e)
            try:
                st = load_state()
                record_face_sightings(cfg, st, [
                    {"name": "" if (r.get("name") or "unknown") == "unknown" else r["name"], "face_id": None}
                    for r in res
                ], source="local")
            except Exception as e:
                log.warning("explore: record_face_sightings failed (non-fatal): %s", e)
        if not cfg.get("explore_greet_known_faces", True):
            return False
        min_score = float(cfg.get("explore_face_min_score", 0.0))
        known = [r for r in res if (r.get("name") or "unknown") != "unknown"
                 and float(r.get("score") or 0) >= min_score and r.get("bbox")]
        if known:
            last = load_state().get("explore_greeted_at") or {}
            cooldown_s = float(cfg.get("greet_cooldown_min", 30)) * 60
            for face in sorted(known, key=lambda r: -float(r.get("score") or 0)):
                name = face["name"]
                if name in self.greeted_names:
                    continue
                if now - float(last.get(name, 0)) < cooldown_s:
                    log_skip("explore: saw %s but greeted them < %s min ago" % (name, cfg.get("greet_cooldown_min", 30)), 300)
                    continue
                write_current_people(cfg, [face], now)  # fresh for the brain's people line
                return self._greet(face)
            return False
        if not self.unknown_face_mused:
            self.unknown_face_mused = True
            self._stop_wheels()
            self._say(random.choice(EXPLORE_UNKNOWN_FACE_LINES))
        return False

    def _greet(self, face):
        cfg = self.cfg
        name, slug = face["name"], face.get("slug")
        self._stop_wheels()
        x, y, w, h = [float(v) for v in face["bbox"][:4]]
        img_w, img_h = self.img_size
        fov_x, fov_y = self._camera_fov()
        yaw = _bbox_turn_degrees(x, w, img_w, fov_x)
        pitch = _bbox_head_delta_degrees(y, h, img_h, fov_y)
        log.info("explore: recognised %s (score %.2f) -- turning %.1f deg, head %+.1f deg",
                 name, float(face.get("score") or 0), yaw, pitch)
        if abs(yaw) >= float(cfg.get("explore_greet_min_turn_deg", 3)) and self.allow_turns:
            self._turn(yaw)
        try:
            cur_head = math.degrees(float(self.robot.head_angle_rad))
        except Exception:
            cur_head = 20.0
        self._head(cur_head + pitch + float(cfg.get("explore_greet_head_lift_deg", 5)))
        person = load_person(cfg, name, slug)
        topic = pick_person_topic(person)
        line = generate_greet_opener(cfg, name, person, topic)
        if self.abort["aborted"]:
            return False  # someone started talking while we were thinking
        self.greeted_names.add(name)
        spoke = self._say(line)
        now = time.time()
        st = load_state()
        st.setdefault("explore_greeted_at", {})[name] = now
        st.setdefault(FACE_GREETED_KEY, {})[name] = now
        save_state(st)
        self.greeted = {"name": name, "slug": slug, "line": line, "topic": topic[0],
                        "spoke": spoke, "kid": person.get("kid", False)}
        self.end_reason = "greeted %s" % name
        return True

    # -- musing --
    def _muse(self):
        cfg = self.cfg
        self._stop_wheels()
        path = self._snap()
        if not path:
            return
        try:
            if cfg.get("explore_greet_known_faces", True) and self._check_faces(path):
                return
            if self._stopped():
                return
            desc, objects = vision_describe(cfg, path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if self._stopped():
            return
        note = None
        if desc is None:
            line = random.choice(EXPLORE_NOTHING_LINES)
        elif _is_low_confidence_scene(desc, objects):
            dark = "dark" in desc.lower() or "dim" in desc.lower()
            line = random.choice(EXPLORE_DARK_LINES if dark else EXPLORE_NOTHING_LINES)
        else:
            log.info("explore: vision: %s", desc)
            self.observations.append((desc, objects))
            line, note = generate_self_talk(cfg, desc, objects, self.musings)
            try:
                update_vector_memory(cfg, objects)
            except Exception as e:
                log.warning("explore: update_vector_memory failed (non-fatal): %s", e)
        if self._stopped():
            return
        if self._say(line):
            self.musings.append(line)
        if note and self.notes_saved < int(cfg.get("explore_max_notes_per_session", 2)):
            try:
                if append_taught_fact(cfg, note, heading=EXPLORE_SECTION_NOTICED):
                    self.notes_saved += 1
            except Exception as e:
                log.warning("explore: saving note failed (non-fatal): %s", e)

    # -- docking --
    def _maybe_return_home(self):
        cfg = self.cfg
        if not cfg.get("explore_return_to_charger", True) or self.abort["aborted"]:
            return
        try:
            if bool(self.robot.status.is_on_charger):
                return
        except Exception:
            pass
        battery = get_battery(cfg)
        level = battery.get("battery_level") if battery else None
        dropped = level is not None and (
            level < int(cfg.get("explore_min_battery_level", 2)) or level <= int(cfg.get("battery_low_level", 1)))
        if self.greeted and not dropped:
            return  # the owner: after a greeting, only go home if the battery is low
        if not (self.started_on_charger or dropped):
            return
        if dropped:
            log.info("explore: battery dropped to %s -- heading home", level)
        self.return_needed = True
        from anki_vector.messaging import protocol
        for attempt in (1, 2):
            if self.abort["aborted"]:
                return
            ok = False
            try:
                resp = self.robot.behavior.drive_on_charger()
                ok = getattr(resp, "result", None) == protocol.BEHAVIOR_COMPLETE_STATE
            except Exception as e:
                log.warning("explore: drive_on_charger attempt %d raised: %s", attempt, e)
            time.sleep(0.5)
            try:
                docked = bool(self.robot.status.is_on_charger)
            except Exception:
                docked = False
            log.info("explore: drive_on_charger attempt %d -> completed=%s on_charger=%s", attempt, ok, docked)
            if docked:
                self.returned_home = True
                self.return_needed = False
                return

    # -- main body --
    def _worker(self):
        cfg = self.cfg
        r = self.robot
        r.connect(timeout=10)
        time.sleep(1.0)
        st = r.status
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
            self.skip_reason = reason
            return
        self.started_at = time.time()
        deadline = self.started_at + float(cfg.get("explore_duration_min", 4)) * 60
        self._say(random.choice(cfg.get("explore_opening_lines") or EXPLORE_OPENING_LINES_DEFAULT))
        if self._stopped():
            return
        if self.started_on_charger or bool(r.status.is_on_charger):
            self.started_on_charger = True
            if not self.no_more_driving and cfg.get("explore_drive_off_charger", True):
                try:
                    r.behavior.drive_off_charger()
                except Exception as e:
                    log.warning("explore: drive_off_charger failed: %s", e)
                time.sleep(0.5)
                try:
                    self.drove_off = not bool(r.status.is_on_charger)
                except Exception:
                    self.drove_off = False
                log.info("explore: drive_off_charger -> off the charger=%s", self.drove_off)
            if not self.drove_off:
                # Still docked: look around from the charger (head only --
                # turning in place on the dock would drag him off it).
                self.no_more_driving = True
                self.allow_turns = False
                log.info("explore: staying on the charger this session (head scans only)")
        self.start_xy = self._pose_xy()
        if cfg.get("explore_use_stock_look_around", True) and self.allow_turns and not self._stopped():
            try:
                r.behavior.look_around_in_place()
                log.info("explore: stock look_around_in_place done")
            except Exception as e:
                log.warning("explore: look_around_in_place failed: %s", e)
        lo, hi = (cfg.get("explore_musing_interval_sec") or [60, 90])[:2]
        face_every = float(cfg.get("explore_face_check_seconds", 4))
        next_muse = time.time() + float(cfg.get("explore_first_musing_sec", 30))
        next_face = time.time()
        while time.time() < deadline and not self._stopped():
            now = time.time()
            if now >= next_muse:
                self._muse()
                next_muse = time.time() + random.uniform(float(lo), float(hi))
                continue
            if cfg.get("explore_greet_known_faces", True) and now >= next_face:
                path = self._snap()
                if path:
                    try:
                        self._check_faces(path)
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                next_face = time.time() + face_every
                if self._stopped():
                    break
            self._explore_step()
        self._stop_wheels()
        if self.end_reason != "picked up / held":
            self._maybe_return_home()

    def _worker_safe(self):
        try:
            self._worker()
        except Exception as e:
            if self.abort["aborted"]:
                log.info("explore: session interrupted by voice (%s: %s)", type(e).__name__, e)
            else:
                self.error = "%s: %s" % (type(e).__name__, e)
                log.warning("explore: session error: %s", self.error)

    def run(self, hard_timeout_s):
        self.robot = self.robot_factory(self.cfg)
        watcher = threading.Thread(
            target=_explore_voice_watch,
            args=(self.cfg, self.robot, self.stop_watch, self.abort,
                  float(self.cfg.get("explore_voice_poll_seconds", 1.5))),
            daemon=True, name="explore-voice-watch",
        )
        worker = threading.Thread(target=self._worker_safe, daemon=True, name="explore-session")
        try:
            watcher.start()
            worker.start()
            worker.join(timeout=hard_timeout_s)
            if worker.is_alive():
                log.error("explore: HARD TIMEOUT after %.0fs -- stopping wheels and disconnecting", hard_timeout_s)
                self.error = "hard timeout"
                self.end_reason = self.end_reason or "hard timeout"
        finally:
            self.stop_watch.set()
            self._stop_wheels()
            try:
                self.robot.disconnect()
            except Exception:
                pass
        if self.abort["aborted"] and self.drove_off and not self.returned_home:
            self.return_needed = True
        return self.outcome()

    def outcome(self):
        if self.abort["aborted"]:
            return "voice"
        if self.skip_reason:
            return "skip"
        if self.error and (self.started_at is None or self.error == "hard timeout"):
            return "fail"
        return "ok"


def _explore_post_greet(cfg, greeted):
    """After the session released control: hand the conversation to
    wire-pod (trigger_wake_word -> conversation mode) and learn from the
    first real answer."""
    name, line = greeted["name"], greeted["line"]
    obs_path = cfg.get("recent_observation_path")
    if obs_path:
        try:
            with open(obs_path, "w") as f:
                f.write(
                    f"[Curious Vector, {datetime.now().isoformat(timespec='seconds')}] "
                    f"You were exploring, spotted {name}, turned to face them and said: \"{line}\" "
                    f"If {name} is answering now, react naturally in character to what they say -- "
                    "don't repeat the question back."
                )
        except OSError as e:
            log.warning("people: writing recent observation failed: %s", e)
    if not greeted.get("spoke"):
        return
    since = time.time()
    trigger_listening(cfg)
    wcfg = dict(cfg)
    wcfg["answer_wait_seconds"] = cfg.get("people_answer_wait_seconds", 25)
    answer, is_command = wait_for_answer(wcfg, since_ts=since - 1)
    min_words = int(cfg.get("people_answer_min_words", 3))
    answered = False
    if not answer:
        log.info("people: no answer from %s within %ss", name, wcfg["answer_wait_seconds"])
    elif is_command:
        log.info("people: %s's reply looked like a command, not saving: %r", name, answer)
    elif len(answer.split()) < min_words:
        log.info("people: ignoring short/junk transcript from %s: %r", name, answer)
    else:
        answered = True
        log.info("people: %s answered: %r", name, answer)
        try:
            _note_curious_play(cfg)
        except Exception:
            pass
        fact = distill_person_fact(cfg, name, line, answer, kid=greeted.get("kid", False))
        if fact:
            append_person_fact(cfg, name, fact, greeted.get("slug"))
        else:
            log.info("people: nothing worth saving from %s's answer", name)
    try:
        record_person_question(cfg, name, greeted.get("topic", "lately"), line, answered, greeted.get("slug"))
    except Exception as e:
        log.warning("people: recording the question failed: %s", e)


def _explore_maybe_ask_owner(cfg, observations):
    p = float(cfg.get("ask_owner_probability", 0.2))
    if p <= 0 or not observations:
        return False
    roll = random.random()
    if roll >= p:
        log.info("explore: not asking the owner this time (roll %.2f >= %.2f)", roll, p)
        return False
    st = load_state()
    present, how = owner_present(cfg, st)
    if not present:
        log.info("explore: would ask the owner, but he isn't seen/home")
        return False
    good = [(d, o) for d, o in observations if d and o and not _is_low_confidence_scene(d, o)]
    if not good:
        return False
    description, objects = good[-1]
    if voice_recently_active(cfg, load_state()) or music_playing(cfg):
        log.info("explore: would ask the owner, but voice/music is active")
        return False
    question = generate_question(cfg, description, objects)
    if not question:
        return False
    log.info("explore: asking the owner (present via %s): %s", how, question)
    write_recent_observation(cfg, description, question)
    if not wirepod_say(cfg, question):
        return False
    trigger_listening(cfg)
    answer, is_command = wait_for_answer(cfg, since_ts=time.time() - 1)
    if not answer:
        log.info("explore: no answer captured within %ss", cfg.get("answer_wait_seconds", 20))
    elif is_command:
        log.info("explore: answer looked like a command, not extracting a fact: %r", answer)
    else:
        log.info("explore: captured answer: %r", answer)
        try:
            _note_curious_play(cfg)
        except Exception:
            pass
        fact = extract_taught_fact(cfg, question, answer)
        if fact and append_taught_fact(cfg, fact):
            log.info("learned from the owner: %r", fact)
    return True


def run_explore_session(cfg, state, now, info, forced=False, robot_factory=None):
    interval_s = float(cfg.get("explore_interval_min", 60)) * 60
    retry_s = float(cfg.get("explore_retry_min", 10)) * 60
    st = load_state()
    st["explore_last_started_at"] = now
    st["explore_next_eligible_at"] = now + interval_s
    save_state(st)
    log.info(
        "explore: session starting (%s, battery_level=%s, on_charger=%s, duration=%.1f min, drive=%s)",
        "one-shot trigger" if forced else "scheduled", info.get("battery_level"), info.get("on_charger"),
        float(cfg.get("explore_duration_min", 4)), cfg.get("explore_drive_enabled", True),
    )
    if voice_recently_active(cfg, load_state()):
        _explore_reschedule(cfg, now + retry_s, "voice activity right before start")
        state.clear()
        state.update(load_state())
        return "skip"
    if not _SDK_LOCK.acquire(timeout=cfg.get("sdk_lock_wait_seconds", 3)):
        _explore_reschedule(cfg, now + retry_s, "robot lock busy at start")
        state.clear()
        state.update(load_state())
        return "skip"
    sess = _ExploreSession(cfg, info.get("on_charger"), info.get("battery_level"), robot_factory)
    try:
        hard = float(cfg.get("explore_duration_min", 4)) * 60 + float(cfg.get("explore_hard_timeout_margin_sec", 180))
        outcome = sess.run(hard)
    finally:
        _SDK_LOCK.release()
    st = load_state()
    _record_capture_outcome(st, now, {"ok": "ok", "fail": "fail"}.get(outcome, "skip"))
    if sess.started_at is None and outcome in ("skip", "fail"):
        _explore_reschedule(cfg, now + retry_s, "session didn't start (%s)" % (sess.skip_reason or sess.error))
    st = load_state()
    if sess.return_needed and not sess.returned_home:
        st["explore_return_pending"] = True
        st["explore_return_tries"] = 0
        st["explore_return_next_try_at"] = 0
    if sess.drove_off and not sess.returned_home:
        st["explore_off_dock_watch"] = True
    save_state(st)
    if sess.greeted:
        try:
            _explore_post_greet(cfg, sess.greeted)
        except Exception as e:
            log.warning("people: post-greet handling failed: %s", e)
    elif outcome == "ok" and sess.started_at is not None:
        try:
            _explore_maybe_ask_owner(cfg, sess.observations)
        except Exception as e:
            log.warning("explore: ask-the owner step failed (non-fatal): %s", e)
    st = load_state()
    log.info(
        "explore: session complete: outcome=%s end=%s ran=%.0fs moves=%d musings=%d notes=%d "
        "cliff_events=%d obstacles=%d greeted=%s drove_off=%s returned_home=%s return_pending=%s "
        "-- next explore eligible at %s",
        outcome, ("voice interrupt" if sess.abort["aborted"] else None) or sess.end_reason
        or sess.skip_reason or sess.error or "time up",
        (time.time() - sess.started_at) if sess.started_at else 0.0, sess.moves, len(sess.musings),
        sess.notes_saved, sess.cliff_events, sess.obstacles,
        sess.greeted["name"] if sess.greeted else None, sess.drove_off, sess.returned_home,
        bool(st.get("explore_return_pending")), _fmt_local(cfg, st.get("explore_next_eligible_at")),
    )
    state.clear()
    state.update(st)
    return outcome


def maybe_explore_return(cfg, state, now, voice_busy):
    """Finish a go-home an explore session couldn't (voice interrupted it,
    docking failed), or one that becomes due later: he's still off the dock
    after a session and the battery went low, or it's evening
    (return_home_by / quiet hours, explore_return_home_by_evening). Waits
    for explore_return_quiet_min of voice quiet. Independent of the legacy
    maybe_send_home() (disabled by return_home_after_wander=false)."""
    st = load_state()
    pending, watch = st.get("explore_return_pending"), st.get("explore_off_dock_watch")
    if not (pending or watch):
        return False
    if voice_busy or music_playing(cfg) or brain_busy(cfg):
        return False
    if not _voice_quiet_for(st, float(cfg.get("explore_return_quiet_min", 3)) * 60):
        return False
    battery = get_battery(cfg)
    if battery is None:
        return False
    if battery.get("is_on_charger_platform"):
        log.info("explore: he's on the charger -- clearing explore return/off-dock state")
        st["explore_return_pending"] = False
        st["explore_off_dock_watch"] = False
        save_state(st)
        state.clear()
        state.update(st)
        return False
    level = battery.get("battery_level")
    low = level is not None and level <= int(cfg.get("battery_low_level", 1))
    rhb = cfg.get("return_home_by")
    evening = bool(cfg.get("explore_return_home_by_evening", True)) and bool(rhb) and (
        in_quiet_hours(cfg, now) or _local_time_reached(cfg, now, rhb))
    if not pending:
        if not (low or evening):
            return False
        log.info("explore: still off the dock after exploring and %s -- sending him home",
                 "the battery is low" if low else "it's evening")
        st["explore_return_pending"] = True
        st["explore_return_tries"] = 0
        st["explore_return_next_try_at"] = 0
    if now < float(st.get("explore_return_next_try_at") or 0):
        save_state(st)
        return False
    tries = int(st.get("explore_return_tries") or 0)
    if tries >= int(cfg.get("explore_return_max_tries", 3)) and not low:
        save_state(st)
        log_skip("explore: still off the charger after %d go-home tries -- giving up until he's docked" % tries, 3600)
        return False
    st["explore_return_tries"] = tries + 1
    st["explore_return_next_try_at"] = now + float(cfg.get("explore_return_retry_min", 5)) * 60
    save_state(st)
    log.info("explore: go-home attempt %d (battery_level=%s, low=%s)", tries + 1, level, low)
    sent, outcome = drive_home_sdk(cfg, load_state())
    st = load_state()
    _record_capture_outcome(st, now, outcome)
    state.clear()
    state.update(load_state())
    return True


def run_face_check_cycle(cfg, state, now, voice_busy):
    """Silent replacement for the old day cycle: the same snapshot ->
    vector-faces -> current_people.json -> maybe_greet_faces() path, so face
    recognition and greetings keep working, but no announcement, no
    vision/LLM, no question and no mic. Cadence face_check_min/max_gap_minutes."""
    if not cfg.get("face_check_enabled", True):
        return False
    if now < float(state.get("next_eligible_at") or 0):
        return False
    gate = _shared_busy_gate(cfg, voice_busy)
    if gate:
        reason, dedup_s = gate
        log_skip("face-check: %s" % reason, dedup_s)
        return False
    image_path, outcome, detail, _w = capture_image(cfg, state, do_wander=False)
    _record_capture_outcome(state, now, outcome)
    if outcome == "ok" and image_path:
        try:
            local_faces_after_capture(cfg, state, image_path)
        except Exception as e:
            log.warning("local face recognition failed (non-fatal): %s", e)
        finally:
            try:
                os.remove(image_path)
            except OSError:
                pass
    try:
        maybe_greet_faces(cfg, state, now, voice_busy)
    except Exception as e:
        log.warning("maybe_greet_faces failed (non-fatal): %s", e)
    if outcome == "ok":
        lo = float(cfg.get("face_check_min_gap_minutes", cfg.get("min_gap_minutes", 1.5)))
        hi = float(cfg.get("face_check_max_gap_minutes", cfg.get("max_gap_minutes", 3)))
        state["next_eligible_at"] = now + random.uniform(lo, max(lo, hi)) * 60
    else:
        if outcome == "skip":
            log_skip("face-check: %s" % detail, 60)
        state["next_eligible_at"] = now + float(cfg.get("empty_scene_retry_minutes", 5)) * 60
    save_state(state)
    return True


def explore_status(cfg):
    """Read-only snapshot for GET /explore_status -- evaluates the gates
    exactly like the scheduler would, but never touches the robot."""
    st = load_state()
    now = time.time()
    vb = voice_recently_active(cfg, st)
    ok, reason, info = explore_gate(cfg, st, now, vb, forced=False)
    fok, freason, _i = explore_gate(cfg, st, now, vb, forced=True)
    last = last_interaction_ts(cfg, st)
    return {
        "explore_enabled": bool(cfg.get("explore_enabled", True)),
        "next_explore_eligible_at_local": _fmt_local(cfg, st.get("explore_next_eligible_at")),
        "last_explore_started_at_local": _fmt_local(cfg, st.get("explore_last_started_at")),
        "minutes_since_last_interaction": round((now - last) / 60, 1) if last else None,
        "one_shot_trigger_pending": explore_trigger_pending(cfg) is not None,
        "scheduled_gate": {"ok": ok, "reason": reason, "battery": info},
        "one_shot_gate": {"ok": fok, "reason": freason},
        "return_pending": bool(st.get("explore_return_pending")),
        "off_dock_watch": bool(st.get("explore_off_dock_watch")),
    }


def _sleep_with_wake(cfg, seconds):
    """main()'s idle sleep, but wakes early for GET /explore_now or a NEW
    data/explore_now trigger file (not for one that's already waiting on
    its gates -- that would spin the loop every few seconds)."""
    deadline = time.time() + seconds
    try:
        seen = os.path.getmtime(explore_trigger_path(cfg))
    except OSError:
        seen = None
    while True:
        left = deadline - time.time()
        if left <= 0:
            return
        if EXPLORE_WAKE.wait(min(5.0, left)):
            EXPLORE_WAKE.clear()
            return
        try:
            m = os.path.getmtime(explore_trigger_path(cfg))
        except OSError:
            m = None
        if m is not None and m != seen:
            return


def _explore_selftest():
    """Dry-run of the explore/people logic: no robot, no network, no real
    files. A fake SDK robot (odometry, cliff injection, faces) drives full
    sessions. Run: docker exec vector-curious python3 /app/curious.py --explore-selftest"""
    import types
    import tempfile as _tf

    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)  # keep the selftest out of curious.log

    results = {"ok": True}

    def check(label, cond):
        print(f"{'PASS' if cond else 'FAIL'}: {label}")
        if not cond:
            results["ok"] = False

    g = globals()
    saved = {}

    def patch(name, val):
        saved.setdefault(name, g[name])
        g[name] = val

    tmp = _tf.mkdtemp(prefix="explore-selftest-")
    mem = {}
    events = {"wirepod": [], "listen": 0, "home_sdk": 0}

    class _Ang:
        def __init__(self, deg):
            self.degrees = float(deg)
            self.radians = math.radians(deg)

    class _St:
        def __init__(self, on_charger):
            self.is_being_held = False
            self.are_motors_moving = False
            self.is_animating = False
            self.is_pathing = False
            self.is_in_calm_power_mode = False
            self.is_picked_up = False
            self.is_falling = False
            self.is_cliff_detected = False
            self.is_on_charger = on_charger

    class _FakeRobot:
        def __init__(self, on_charger=True, cliff_on_drive=None):
            self.st = _St(on_charger)
            self.calls = []
            self.connected = False
            self.x = self.y = self.heading = 0.0
            self.wheel = 0.0
            self.wheel_t = None
            self.head_angle_rad = math.radians(10)
            self.drives = 0
            self.cliff_on_drive = cliff_on_drive
            self.cliff_at = None
            self.behavior = self.motors = self.camera = self.proximity = self
            self.config = types.SimpleNamespace(fov_x=_Ang(90), fov_y=_Ang(50))
            self.last_sensor_reading = types.SimpleNamespace(
                distance=types.SimpleNamespace(distance_mm=300.0))

        def _adv(self):
            now = time.time()
            if self.wheel_t is not None and self.wheel:
                d = self.wheel * (now - self.wheel_t)
                self.x += d * math.cos(math.radians(self.heading))
                self.y += d * math.sin(math.radians(self.heading))
            self.wheel_t = now
            if self.cliff_at and now >= self.cliff_at and self.wheel > 0:
                self.st.is_cliff_detected = True

        @property
        def status(self):
            self._adv()
            return self.st

        @property
        def pose(self):
            self._adv()
            return types.SimpleNamespace(
                position=types.SimpleNamespace(x=self.x, y=self.y),
                rotation=types.SimpleNamespace(angle_z=_Ang(self.heading)))

        def _need(self):
            if not self.connected:
                raise RuntimeError("not connected")

        def connect(self, timeout=10):
            self.connected = True
            self.calls.append("connect")

        def disconnect(self):
            self.connected = False
            self.calls.append("disconnect")

        def say_text(self, text, **_k):
            self._need()
            self.calls.append(("say", text))

        def drive_off_charger(self):
            self._need()
            self.calls.append("drive_off")
            self.st.is_on_charger = False
            self.x += 100
            return types.SimpleNamespace(result=0)

        def drive_on_charger(self):
            self._need()
            self.calls.append("drive_on")
            self.st.is_on_charger = True
            from anki_vector.messaging import protocol
            return types.SimpleNamespace(result=protocol.BEHAVIOR_COMPLETE_STATE)

        def look_around_in_place(self):
            self._need()
            self.calls.append("look_around")

        def turn_in_place(self, angle, **_k):
            self._need()
            self.calls.append(("turn", round(angle.degrees, 1)))
            self.heading += angle.degrees

        def set_head_angle(self, angle, **_k):
            self._need()
            self.calls.append(("head", round(angle.degrees, 1)))
            self.head_angle_rad = angle.radians

        def set_wheel_motors(self, left, right, *_a, **_k):
            self._need()
            self._adv()
            if left > 0 and self.wheel <= 0:
                self.drives += 1
                if self.cliff_on_drive and self.drives == self.cliff_on_drive:
                    self.cliff_at = time.time() + 0.05
            if left < 0:
                self.st.is_cliff_detected = False
                self.cliff_at = None
            self.wheel = float(left)
            self.calls.append(("wheels", left))

        def capture_single_image(self):
            self._need()
            from PIL import Image
            return types.SimpleNamespace(raw_image=Image.new("RGB", (640, 360), (90, 90, 90)))

    try:
        # ---- sandbox: state in memory, files in tmp, no network, no real lock
        patch("load_state", lambda: json.loads(json.dumps(mem)))

        def _save(s):
            snap = json.loads(json.dumps(s))
            mem.clear()
            mem.update(snap)
        patch("save_state", _save)
        lockp = os.path.join(tmp, "robot_control.lock")
        open(lockp, "w").close()
        patch("ROBOT_CONTROL_LOCK_PATH", lockp)
        patch("_SDK_LOCK", _RobotControlLock(lockp))
        battery = {"battery_level": 3, "is_on_charger_platform": True}
        patch("get_battery", lambda cfg: dict(battery) if battery else None)
        flags = {"music": False, "brain": False, "sentry": False, "voice_after": None, "t0": time.time()}
        patch("music_playing", lambda cfg: flags["music"])
        patch("brain_busy", lambda cfg: flags["brain"])
        patch("sentry_is_active", lambda cfg, st: flags["sentry"])

        def _voice(cfg, st):
            va = flags["voice_after"]
            return va is not None and time.time() - flags["t0"] >= va
        patch("voice_recently_active", _voice)
        patch("wirepod_say", lambda cfg, text: events["wirepod"].append(text) or True)

        def _listen(cfg):
            events["listen"] += 1
            return True
        patch("trigger_listening", _listen)
        answer = {"value": (None, False)}
        patch("wait_for_answer", lambda cfg, since_ts: answer["value"])

        def _home(cfg, st):
            events["home_sdk"] += 1
            return True, "ok"
        patch("drive_home_sdk", _home)
        faces = {"value": []}
        patch("recognize_faces_local", lambda cfg, p: list(faces["value"]))
        patch("vision_describe", lambda cfg, p: (
            "A desk with a blue mug and a desk lamp.", [("mug", None, "desk"), ("lamp", None, "left")]))

        def _llm(cfg, system, user, timeout=60):
            if "muttering" in system:
                return "SAY: Hmm, a mug on the desk. Wonder who left it?\nNOTE: There is a blue mug on the desk."
            if "spotted" in system:
                return "Oh hey Alex! What's your favorite thing to paint?"
            if "extract what a home robot" in system:
                return "Alex loves painting watercolors on weekends."
            return "NONE"
        patch("_llm_internal", _llm)
        patch("generate_question", lambda cfg, d, o: "the owner, what's that mug for?")
        patch("extract_taught_fact", lambda cfg, q, a: None)

        people = os.path.join(tmp, "people")
        os.makedirs(people)
        with open(os.path.join(people, "jordan.md"), "w") as f:
            f.write("---\nagent: vector\ntags: [people, agent/vector]\nperson: Jordan\nkid: true\n"
                    "aliases: [Jordan, Jordan L., Jordan Lee]\n---\n\n# Jordan\n\n## Who they are\n"
                    "- a teenager in the household.\n\n## Things I've learned\n\n"
                    "## Questions I've asked\n- 2026-10-01 [hobbies] What do you love doing? — answered\n")
        memory_md = os.path.join(tmp, "MEMORY.md")
        with open(memory_md, "w") as f:
            f.write("# Vector memory\n\n## Things I have learned to recognize\n- x\n\n## Things the owner taught me\n- y\n")
        cfg = load_config()
        cfg.update({
            "explore_enabled": True, "explore_interval_min": 60, "idle_required_min": 30,
            "quiet_hours_start": "22:00", "quiet_hours_end": "08:00", "return_home_by": "21:30",
            "explore_duration_min": 0.07, "explore_first_musing_sec": 0.3,
            "explore_musing_interval_sec": [1.5, 1.8], "explore_face_check_seconds": 0.4,
            "explore_speed_mmps": 400, "explore_step_cm": 8, "explore_leash_cm": 30,
            "explore_step_weights": {"drive": 1, "head": 0, "turn": 0, "pause": 0},
            "explore_hard_timeout_margin_sec": 20, "ask_owner_probability": 0.0,
            "people_notes_dir": people, "vector_memory_path": memory_md,
            "face_current_people_path": os.path.join(tmp, "current_people.json"),
            "recent_observation_path": os.path.join(tmp, "recent_observation.txt"),
            "mood_path": os.path.join(tmp, "mood.json"),
            "music_log_path": os.path.join(tmp, "music.log"),
            "explore_trigger_path": os.path.join(tmp, "data", "explore_now"),
            "people_note_aliases": {"jordan lee": "jordan"},
        })

        tz = ZoneInfo(cfg.get("timezone", "America/New_York"))

        def at(h, m=0):
            return datetime.now(tz).replace(hour=h, minute=m, second=0, microsecond=0).timestamp()

        # ---- A. geometry
        check("face right of centre -> negative (clockwise) turn", _bbox_turn_degrees(500, 60, 640, 90) < -20)
        check("centred face -> ~0 turn", abs(_bbox_turn_degrees(290, 60, 640, 90)) < 0.5)
        check("image edge with 90 deg FOV -> ~45 deg", abs(abs(_bbox_turn_degrees(620, 40, 640, 90)) - 45) < 1)
        check("face above centre -> raise head", _bbox_head_delta_degrees(20, 60, 360, 50) > 5)

        # ---- B. gates
        t = at(14)
        mem.clear()
        mem["explore_next_eligible_at"] = t + 600
        ok, why, _ = explore_gate(cfg, mem, t, False)
        check("cadence blocks a scheduled run (%s)" % why, not ok and why.startswith("cadence"))
        ok, _, info = explore_gate(cfg, mem, t, False, forced=True)
        check("one-shot trigger bypasses cadence", ok and info["on_charger"] is True)
        mem["explore_next_eligible_at"] = t - 1
        mem["last_seen_voice_activity_ts"] = t - 10 * 60
        ok, why, _ = explore_gate(cfg, mem, t, False)
        check("talked to 10 min ago blocks (%s)" % why, not ok and "talked to" in why)
        check("...but not a one-shot trigger", explore_gate(cfg, mem, t, False, forced=True)[0])
        mem["last_seen_voice_activity_ts"] = t - 40 * 60
        check("idle 40 min -> eligible", explore_gate(cfg, mem, t, False)[0])
        with open(cfg["mood_path"], "w") as f:
            json.dump({"last_interaction_ts": t - 5 * 60}, f)
        check("brain mood.json interaction 5 min ago blocks", not explore_gate(cfg, mem, t, False)[0])
        os.remove(cfg["mood_path"])
        check("voice active right now blocks even a trigger", not explore_gate(cfg, mem, t, True, forced=True)[0])
        check("quiet hours (23:00) block even a trigger", not explore_gate(cfg, mem, at(23), False, forced=True)[0])
        check("before 08:00 blocked", not explore_gate(cfg, mem, at(7, 30), False)[0])
        check("after return_home_by 21:30 blocks scheduled", not explore_gate(cfg, mem, at(21, 45), False)[0])
        flags["sentry"] = True
        check("sentry active blocks scheduled", not explore_gate(cfg, mem, t, False)[0])
        flags["sentry"] = False
        flags["music"] = True
        check("music blocks even a trigger", not explore_gate(cfg, mem, t, False, forced=True)[0])
        flags["music"] = False
        with open(cfg["music_log_path"], "w") as f:
            f.write("x\n")
        check("music.log touched < 5 min ago blocks", not explore_gate(cfg, mem, t, False)[0])
        os.utime(cfg["music_log_path"], (time.time() - 900, time.time() - 900))
        check("music.log 15 min old is fine", explore_gate(cfg, mem, t, False)[0])
        battery.update(battery_level=1)
        check("battery level 1 (low) blocks", not explore_gate(cfg, mem, t, False, forced=True)[0])
        battery.update(battery_level=2, is_on_charger_platform=False)
        ok, _, info = explore_gate(cfg, mem, t, False)
        check("battery nominal off-charger allowed", ok and info["on_charger"] is False)
        battery.update(battery_level=3, is_on_charger_platform=True)
        holder = open(lockp, "r+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        ok, why, _ = explore_gate(cfg, mem, t, False, forced=True)
        check("robot_control.lock held elsewhere blocks (%s)" % why, not ok and "lock" in why)
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()

        # ---- C. self-talk + opener rules
        say, note = _parse_self_talk("SAY: Hmm, is that box still there?\nNOTE: NONE", cfg)
        check("self-talk never ends in a question", say and "?" not in say)
        check("NOTE: NONE -> no note", note is None)
        check("self-talk naming the owner rejected", _clean_self_talk("the owner left his mug again.", cfg) is None)
        check("self-talk addressing 'you' rejected", _clean_self_talk("Hey you, nice lamp.", cfg) is None)
        check("plain musing kept", _clean_self_talk("That lamp has seen things.", cfg) == "That lamp has seen things.")
        check("valid opener kept",
              _clean_opener("Oh hey Alex! What's the best thing you cooked lately?", "Alex", cfg))
        check("opener without the name rejected", _clean_opener("Hey there! What's up?", "Alex", cfg) is None)
        check("opener without a question rejected", _clean_opener("Oh hey Alex!", "Alex", cfg) is None)
        check("opener the proxy would strip rejected",
              _clean_opener("Hi Alex. Want to chat?", "Alex", cfg) is None)

        # ---- D. notes + topics
        kid = load_person(cfg, "Jordan Lee")
        check("alias 'Jordan Lee' -> jordan.md, kid flag", kid["path"] and kid["path"].endswith("jordan.md") and kid["kid"])
        check("answered topic not asked again", all(pick_person_topic(kid)[0] != "hobbies" for _ in range(30)))
        check("kid topics used for kids", pick_person_topic(kid)[0] in {k for k, _ in PEOPLE_TOPICS_KID})
        p = ensure_person_note(cfg, "Alex")
        check("new person note created", p and os.path.basename(p) == "alex.md")
        check("fact appended", append_person_fact(cfg, "Alex", "Alex loves hiking."))
        check("near-duplicate fact skipped", not append_person_fact(cfg, "Alex", "Alex loves hiking"))
        record_person_question(cfg, "Alex", "hobbies", "What do you love doing?", True)
        ar = load_person(cfg, "Alex")
        check("question recorded + parsed", ar["asked"] and ar["asked"][-1]["status"] == "answered")
        with open(p) as f:
            body = f.read()
        check("note sections intact", body.index(PERSON_FACTS) < body.index("loves hiking") < body.index(PERSON_ASKED))
        check("distilled fact with a phone number rejected",
              _SENSITIVE_RE.search("Alex's number is 704-555-1234") is not None)

        # ---- E. session 1: from the charger, cliff on 2nd drive, musings, back home
        mem.clear()
        mem["explore_next_eligible_at"] = time.time() - 1
        r1 = _FakeRobot(on_charger=True, cliff_on_drive=2)
        info = {"battery_level": 3, "on_charger": True}
        t0 = time.time()
        out = run_explore_session(cfg, dict(mem), t0, info, robot_factory=lambda c: r1)
        says = [c[1] for c in r1.calls if isinstance(c, tuple) and c[0] == "say"]
        check("session 1 outcome ok (%s)" % out, out == "ok")
        check("drove off the charger, stock look-around ran", "drive_off" in r1.calls and "look_around" in r1.calls)
        check("cliff stopped the wheels and backed off",
              ("wheels", -30) in r1.calls and r1.st.is_cliff_detected is False)
        check("said musings, none of them questions", len(says) >= 2 and all("?" not in s for s in says))
        with open(memory_md) as f:
            mtxt = f.read()
        check("observation NOTE saved under the explore heading",
              EXPLORE_SECTION_NOTICED in mtxt and "blue mug" in mtxt)
        check("returned to the charger (started there)", "drive_on" in r1.calls)
        check("disconnected + lock released", r1.calls[-1] == "disconnect" and not _SDK_LOCK._tlock.locked())
        check("next explore ~60 min out", abs(mem.get("explore_next_eligible_at", 0) - (t0 + 3600)) < 5)
        check("no mic opened, nothing via wire-pod", events["listen"] == 0 and not events["wirepod"])
        check("leash respected (<= 30 cm + one step)",
              math.hypot(r1.x - 100, r1.y) <= 300 + 80 + 5)

        # ---- F. session 2: known face -> turn, greet, hand over, learn
        faces["value"] = [{"name": "Alex", "slug": "alex", "score": 0.61, "bbox": [500, 40, 80, 100]}]
        answer["value"] = ("I love painting watercolors on weekends", False)
        r2 = _FakeRobot(on_charger=True)
        out = run_explore_session(cfg, dict(mem), time.time(), info, robot_factory=lambda c: r2)
        turns = [c[1] for c in r2.calls if isinstance(c, tuple) and c[0] == "turn"]
        heads = [c[1] for c in r2.calls if isinstance(c, tuple) and c[0] == "head"]
        says = [c[1] for c in r2.calls if isinstance(c, tuple) and c[0] == "say"]
        check("turned toward the face on the right (negative angle)", any(a < -10 for a in turns))
        check("raised the head toward the face", heads and heads[-1] > 10)
        check("greeted by name with a question", any("Alex" in s and s.endswith("?") for s in says))
        check("did NOT drive home after greeting (battery fine)", "drive_on" not in r2.calls)
        check("handed over to wire-pod (trigger_wake_word)", events["listen"] == 1)
        with open(os.path.join(people, "alex.md")) as f:
            body = f.read()
        check("answer distilled into Alex's note", "watercolors" in body)
        check("question logged as answered", "— answered" in body.split(PERSON_ASKED)[1])
        check("greet cooldown recorded", "Alex" in (mem.get("explore_greeted_at") or {}))
        check("off-dock watch set (he stayed out)", mem.get("explore_off_dock_watch") is True)

        # ---- G. session 3: same face again within 30 min -> no greeting
        r3 = _FakeRobot(on_charger=False)
        run_explore_session(cfg, dict(mem), time.time(), {"battery_level": 3, "on_charger": False},
                            robot_factory=lambda c: r3)
        says = [c[1] for c in r3.calls if isinstance(c, tuple) and c[0] == "say"]
        check("no second greeting inside greet_cooldown_min", not any("Alex" in s for s in says))
        check("started off the dock + battery fine -> no drive home", "drive_on" not in r3.calls)

        # ---- H. session 4: someone talks -> stop + release immediately
        faces["value"] = []
        flags["t0"] = time.time()
        flags["voice_after"] = 0.8
        r4 = _FakeRobot(on_charger=True)
        out = run_explore_session(cfg, dict(mem), time.time(), info, robot_factory=lambda c: r4)
        flags["voice_after"] = None
        check("voice interrupt -> outcome 'voice' (%s)" % out, out == "voice")
        check("wheels stopped + disconnected on voice", ("wheels", 0) in r4.calls and "disconnect" in r4.calls)
        check("pending go-home queued for later", mem.get("explore_return_pending") is True)

        # ---- I. junk answer ignored
        faces["value"] = [{"name": "Jordan", "slug": "jordan", "score": 0.55, "bbox": [300, 100, 60, 80]}]
        answer["value"] = ("the", False)
        r5 = _FakeRobot(on_charger=False)
        run_explore_session(cfg, dict(mem), time.time(), {"battery_level": 3, "on_charger": False},
                            robot_factory=lambda c: r5)
        with open(os.path.join(people, "jordan.md")) as f:
            body = f.read()
        check("kid greeted, junk transcript not saved, logged 'no answer'",
              "— no answer" in body and len(_note_section_bullets(body, PERSON_FACTS)) == 0)

        # ---- J. scheduler + one-shot trigger + pending return
        faces["value"] = []
        mem["explore_next_eligible_at"] = time.time() + 3000
        mem["last_seen_voice_activity_ts"] = time.time() - 60
        check("scheduler idle before the hour (no trigger)",
              maybe_run_explore(cfg, mem, at(14), False, robot_factory=lambda c: _FakeRobot()) is False)
        request_explore_now(cfg)
        check("trigger file written", explore_trigger_pending(cfg) is not None)
        ran = maybe_run_explore(cfg, mem, at(14), False, robot_factory=lambda c: _FakeRobot())
        check("one-shot trigger runs a session now and is consumed", ran and explore_trigger_pending(cfg) is None)
        mem["explore_return_pending"] = True
        mem["last_seen_voice_activity_ts"] = time.time() - 600
        battery.update(battery_level=2, is_on_charger_platform=False)
        before = events["home_sdk"]
        maybe_explore_return(cfg, mem, time.time(), False)
        check("pending return -> drive_home_sdk", events["home_sdk"] == before + 1)
        battery.update(is_on_charger_platform=True)
        maybe_explore_return(cfg, mem, time.time(), False)
        check("docked -> return state cleared", not load_state().get("explore_return_pending"))

        # ---- K. presence for the occasional question
        with open(cfg["face_current_people_path"], "w") as f:
            json.dump([{"name": "the owner", "score": 0.5, "last_seen_epoch": time.time()}], f)
        check("the owner present via face recognition", owner_present(cfg, {})[0])
        os.remove(cfg["face_current_people_path"])
        check("the owner present via HA home", owner_present(cfg, {SENTRY_HOME_KEY: True})[0])
        check("nobody -> no question", not owner_present(cfg, {SENTRY_HOME_KEY: False})[0])

        # ---- L. wakeable sleep
        EXPLORE_WAKE.set()
        t1 = time.time()
        _sleep_with_wake(cfg, 30)
        check("main-loop sleep wakes on /explore_now", time.time() - t1 < 1)
    finally:
        for k, v in saved.items():
            g[k] = v
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASSED" if results["ok"] else "FAILED")
    return results["ok"]


def main():
    log.info(
        "Curious Vector starting up (rework 2026-09-28: single connect/cycle, "
        "HTTP-first gates, idle-poll backoff %s, circuit breaker %d/%ds)",
        IDLE_POLL_LADDER,
        CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        CIRCUIT_BREAKER_PAUSE_SECONDS,
    )
    _sentry_stop_event = threading.Event()
    threading.Thread(
        target=presence_poll_loop, args=(load_config, _sentry_stop_event),
        daemon=True, name="sentry-presence",
    ).start()
    threading.Thread(
        target=sentry_log_watch_loop, args=(load_config, _sentry_stop_event),
        daemon=True, name="sentry-logwatch",
    ).start()
    threading.Thread(
        target=start_sentry_control_server,
        daemon=True, name="sentry-http-control",
    ).start()
    threading.Thread(
        target=safety_log_watch_loop, args=(load_config, _sentry_stop_event),
        daemon=True, name="safety-logwatch",
    ).start()
    log.info("sentry: presence poller + log watcher + HTTP control threads started (+ safety-logwatch)")
    try:
        _c = load_config()
        log.info(
            "explore: hourly explore mode enabled=%s interval=%smin idle_required=%smin duration=%smin "
            "drive=%s greet_known_faces=%s ask_owner_probability=%s legacy_question_cycle=%s",
            _c.get("explore_enabled", True), _c.get("explore_interval_min", 60), _c.get("idle_required_min", 30),
            _c.get("explore_duration_min", 4), _c.get("explore_drive_enabled", True),
            _c.get("explore_greet_known_faces", True), _c.get("ask_owner_probability", 0.2),
            _c.get("curious_question_cycle_enabled", False),
        )
    except Exception:
        pass

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

        _sleep_with_wake(cfg, poll_s)


# ---------------------------------------------------------------------------
# SENTRY MODE (2026-09-29, the owner: wire phone presence -> auto Vector sentry)
#
# Two independent background threads, started once from main() alongside
# the existing poll loop (NOT folded into run_cycle()'s idle-backoff ladder,
# since that ladder can back off to 300s and both presence + sound-event
# response need to stay on their own fixed, fast cadence):
#
#   1. presence_poll_loop() -- every `presence_poll_seconds` (default 60),
#      reads person.owner's state from Home Assistant's REST API (token
#      read from `ha_token_path`, never baked into the image/config).
#      Debounces `presence_debounce_readings` (default 2) CONSISTENT
#      readings before flipping home/away, so one flaky/unknown GPS fix
#      can't flap sentry on/off. On away->home it also fires the
#      "Welcome back" arrival greeting (see _sentry_arrival_greet_async).
#
#   2. sentry_log_watch_loop() -- tails ~/vector-logs/vector.log
#      (mounted read-only at `sentry_vector_log_path`) every
#      `sentry_log_poll_seconds` (default 3) for the robot's own built-in
#      `@robot.reacted_to_sound` (ReactToSound) behavior firing -- chosen
#      as the sentry trigger because it's by far the most reliable signal
#      in this robot's syslog (1119 occurrences seen in a ~24h sample vs.
#      18 for ReactToMotion, 0 for any AudioDirection/MicDirection field --
#      so there is no per-event sound-direction data available from the
#      log, and capture deliberately does NOT try to turn the robot: "stay
#      put" per the fallback the task spec allows). When sentry is active
#      (see sentry_is_active()), not quiet-houred, and not rate-limited,
#      it calls _sentry_maybe_capture().
#
# SDK connection safety: sentry's own capture (_sentry_capture_and_detect)
# opens the SAME kind of single, short-lived anki_vector.Robot() connection
# capture_image()/capture_night_images()/drive_home_sdk() already use (see
# those functions' docstrings above) -- connect, do the minimal work,
# ALWAYS disconnect in `finally`. The robot has ~436MB RAM and can only
# tolerate ONE SDK connection at a time system-wide, so all four of these
# connection sites now acquire the same module-level `_SDK_LOCK` before
# calling anki_vector.Robot(...) and release it in their `finally` --
# whichever one gets there first wins, everyone else politely skips
# (outcome="skip", doesn't trip the circuit breaker) and retries on the
# next poll/trigger instead of queueing behind a held connection.
#
# Sentry NEVER wanders and NEVER drives -- "stay docked" per spec, no
# drive_straight/turn_in_place/drive_on_charger calls anywhere in this
# section. It reuses get_battery()/music_playing()/brain_busy()/
# voice_recently_active()/_voice_watch_loop()/_run_with_hard_timeout()/
# _save_camera_image()/record_face_sightings()/_record_capture_outcome()/
# in_quiet_hours() from the existing daemon rather than duplicating any of
# that fail-safe logic.
# ---------------------------------------------------------------------------

SENTRY_HOME_KEY = "sentry_last_known_home"          # bool, debounced
SENTRY_ACTIVE_SINCE_KEY = "sentry_active_since"      # epoch or None
SENTRY_CAPTURE_TIMES_KEY = "sentry_capture_times"    # list[epoch], rate limit
SENTRY_LOG_OFFSET_KEY = "sentry_log_offset"          # byte offset into vector.log
SENTRY_PRESENCE_POLL_AT_KEY = "sentry_last_presence_poll_at"
SENTRY_DEBOUNCE_VALUE_KEY = "sentry_debounce_value"  # last raw reading seen
SENTRY_DEBOUNCE_COUNT_KEY = "sentry_debounce_count"

SENTRY_SOUND_TS_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2}) "
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})\.(?P<ms>\d+)"
)


def _load_ha_token(cfg):
    """Reads the HA long-lived token from a file (never from config.json,
    never baked into the image). Bind-mounted alongside curious.py's own
    /app dir; kept 0400 on the host for defense in depth even though the
    container itself runs as root (root bypasses unix perms regardless --
    the real boundary is that this file lives outside the image)."""
    path = cfg.get("ha_token_path", "/app/.ha_token")
    try:
        with open(path) as f:
            tok = f.read().strip()
        return tok or None
    except Exception as e:
        log.warning("sentry: could not read HA token at %s: %s", path, e)
        return None


def ha_get_state(cfg, entity_id):
    token = _load_ha_token(cfg)
    if not token:
        return None
    url = f"{cfg.get('ha_url', 'http://127.0.0.1:8123')}/api/states/{entity_id}"
    timeout = cfg.get("http_timeout_seconds", 15)
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("sentry: ha_get_state(%s) failed: %s", entity_id, e)
        return None


def ha_call_service(cfg, domain, service, payload):
    token = _load_ha_token(cfg)
    if not token:
        log.warning("sentry: no HA token -- cannot call %s.%s", domain, service)
        return False
    url = f"{cfg.get('ha_url', 'http://127.0.0.1:8123')}/api/services/{domain}/{service}"
    timeout = cfg.get("http_timeout_seconds", 15)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True
    except Exception as e:
        log.warning("sentry: ha_call_service(%s.%s) failed: %s", domain, service, e)
        return False


def _presence_reading_to_home(raw_state):
    """Maps a person.<x> HA state string to True (home) / False (away) /
    None (unverifiable -- 'unknown', 'unavailable', or no reading at all).
    None readings are never debounce-counted -- they neither confirm nor
    deny a transition, so they're simply skipped rather than treated as
    "away" (a phone that hasn't reported a GPS fix yet is not evidence
    the owner left)."""
    if raw_state is None:
        return None
    if raw_state in ("unknown", "unavailable"):
        return None
    if raw_state == "home":
        return True
    return False  # "not_home" or any other zone name -> not the home zone


def sentry_is_active(cfg, state):
    """Effective sentry on/off, folding in the manual override
    (`sentry_manual_override`: "auto" (default, follows presence) / "on"
    (forced on regardless of presence) / "off" (forced off regardless)).
    "auto" activates whenever the debounced presence reading says away
    (or hasn't been established at all yet -- fail-safe default is OFF,
    not on, so an HA outage before the first successful reading can never
    make sentry start snapping unexpected photos of the owner's own house)."""
    if not cfg.get("sentry_enabled", True):
        return False
    override = cfg.get("sentry_manual_override", "auto")
    if override == "off":
        return False
    if override == "on":
        return True
    return state.get(SENTRY_HOME_KEY, True) is False


# --- SENTRY VOICE CONTROL HTTP ENDPOINT (added 2026-09-29, Hermes) ---
# Small stdlib-only HTTP server, 127.0.0.1-only, so wire-pod's
# sentry_cmd_custom custom intent (see vector-pod's
# pkg/wirepod/ttr/sentrycmd.go) can flip sentry_manual_override by voice
# without editing config.json by hand or restarting this container.
# GET /sentry?mode=on|off|auto persists the override into config.json --
# the SAME key sentry_is_active() already reads via load_config() on
# every poll/decision (presence_poll_loop, sentry_log_watch_loop,
# run_cycle all call cfg_getter()/load_config() fresh each cycle) -- so
# there's no second source of truth and no extra hot-reload plumbing
# needed. Speaking the confirmation is the CALLER's job (sentrycmd.go
# does it via wire-pod's own sayText, reusing wire-pod's existing SDK
# connection -- see that file's header for why never a second
# connection) so this endpoint stays side-effect-minimal: it only ever
# writes config.json, and is independently curl-testable without ever
# making the robot talk or move.
SENTRY_CONTROL_HOST = "127.0.0.1"
SENTRY_CONTROL_PORT = 8790
SENTRY_VALID_MODES = ("on", "off", "auto")

_config_write_lock = threading.Lock()


def set_sentry_override(mode):
    """Persists mode into config.json's sentry_manual_override key,
    atomically (tmp + os.replace, same pattern as save_state()) and under
    a lock (protects against two concurrent WRITERS interleaving --
    readers never see a half-written file because of the atomic
    replace). Raises ValueError for an invalid mode; does nothing else on
    disk in that case."""
    if mode not in SENTRY_VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}, must be one of {SENTRY_VALID_MODES}")
    with _config_write_lock:
        cfg = load_config()
        cfg["sentry_manual_override"] = mode
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
        # 2026-09-29 cliffsafe: config.json kept becoming root-owned because
        # this container runs as root -- every write now re-chowns it back
        # to the host user (uid 1000), same pattern _sentry_log_event/_fix_owner already
        # use for the metrics files, so host-side readers/editors keep
        # working without sudo.
        _fix_owner(CONFIG_PATH)
    log.warning("SENTRY: manual override set to %r via HTTP control endpoint", mode)


class _SentryControlHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("sentry-http: " + fmt, *args)

    def _write_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/explore_now":
            try:
                request_explore_now(load_config())
                self._write_json(200, {"ok": True, "queued": True,
                                       "note": "runs at the next poll once the safety gates pass "
                                               "(voice quiet, no music, lock free, battery ok, daytime)"})
            except Exception as e:
                log.warning("explore-http: queueing explore_now failed: %s", e)
                self._write_json(500, {"ok": False, "error": "internal error"})
            return
        if parsed.path == "/explore_status":
            try:
                self._write_json(200, explore_status(load_config()))
            except Exception as e:
                log.warning("explore-http: status failed: %s", e)
                self._write_json(500, {"ok": False, "error": "internal error"})
            return
        if parsed.path != "/sentry":
            self._write_json(404, {"ok": False, "error": "not found"})
            return
        qs = urllib.parse.parse_qs(parsed.query)
        mode = (qs.get("mode") or [""])[0].strip().lower()
        try:
            set_sentry_override(mode)
        except ValueError as e:
            self._write_json(400, {"ok": False, "error": str(e)})
            return
        except Exception as e:
            log.warning("sentry-http: set_sentry_override(%r) failed: %s", mode, e)
            self._write_json(500, {"ok": False, "error": "internal error"})
            return
        self._write_json(200, {"ok": True, "sentry_manual_override": mode})


def start_sentry_control_server():
    """Run forever (daemon thread, started once from main()). Binds
    127.0.0.1 only -- this is a same-host control channel for wire-pod's
    custom-intent handler (both containers run --network host, the
    host-networking Docker rule, so 127.0.0.1 from vector-pod reaches
    this directly), never meant to be reachable off-box."""
    server = http.server.ThreadingHTTPServer(
        (SENTRY_CONTROL_HOST, SENTRY_CONTROL_PORT), _SentryControlHandler
    )
    log.info(
        "sentry: control HTTP server listening on %s:%d",
        SENTRY_CONTROL_HOST, SENTRY_CONTROL_PORT,
    )
    server.serve_forever()


def _sentry_log_event(cfg, event):
    path = cfg.get("sentry_events_log_path", "/mnt/vector-brain/metrics/sentry.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
        _fix_owner(path)
    except Exception as e:
        log.warning("sentry: failed writing event log: %s", e)


def _sentry_read_events_since(cfg, since_ts):
    path = cfg.get("sentry_events_log_path", "/mnt/vector-brain/metrics/sentry.jsonl")
    events = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("ts", 0) >= since_ts:
                    events.append(e)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("sentry: failed reading event log: %s", e)
    return events


def _sentry_activate(cfg, state, now, reason):
    state[SENTRY_ACTIVE_SINCE_KEY] = now
    save_state(state)
    _sentry_log_event(cfg, {"ts": now, "type": "sentry_activated", "reason": reason})
    log.warning("SENTRY: activated (%s)", reason)


def _sentry_arrival_greet_async(cfg, state, since_ts):
    """Spawned (daemon thread, fire-and-forget) right when presence flips
    to home. Waits sentry_arrival_min_delay_seconds so the owner's actually
    had a moment to walk in, then polls the voice gate (fail-closed, same
    as every other wirepod_say call site) until it's clear or
    sentry_arrival_max_wait_seconds elapses, then speaks a one-line
    summary of what happened while he was away (built from this away
    session's sentry.jsonl events) via wirepod_say -- never opens an SDK
    connection of its own."""
    min_delay = cfg.get("sentry_arrival_min_delay_seconds", 15)
    max_wait = cfg.get("sentry_arrival_max_wait_seconds", 300)
    time.sleep(min_delay)

    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            busy = voice_recently_active(cfg, load_state())
        except Exception as e:
            log.warning("sentry: arrival voice-gate check failed (%s) -- treating as busy", e)
            busy = True
        if not busy:
            break
        time.sleep(5)
    else:
        log.info("sentry: arrival greeting gave up waiting for voice gate to clear after %ss", max_wait)
        return

    events = _sentry_read_events_since(cfg, since_ts)
    unknowns = [e for e in events if e.get("type") == "unknown_person_alert"]
    if unknowns:
        first = unknowns[0]
        tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
        dt = datetime.fromtimestamp(first["ts"], tz) if tz else datetime.fromtimestamp(first["ts"])
        hhmm = dt.strftime("%I:%M %p").lstrip("0")
        summary = f"I saw someone at {hhmm}, check your phone."
    else:
        summary = "It was quiet."

    wirepod_say(cfg, f"Welcome back! {summary}")


def _sentry_deactivate(cfg, state, now, reason):
    since = state.get(SENTRY_ACTIVE_SINCE_KEY) or now
    state[SENTRY_ACTIVE_SINCE_KEY] = None
    save_state(state)
    _sentry_log_event(cfg, {"ts": now, "type": "sentry_deactivated", "reason": reason})
    log.warning("SENTRY: deactivated (%s)", reason)
    threading.Thread(
        target=_sentry_arrival_greet_async, args=(cfg, state, since), daemon=True,
        name="sentry-arrival-greet",
    ).start()


def presence_poll_loop(cfg_getter, stop_event):
    """Background daemon thread, started once from main(). Fixed
    presence_poll_seconds cadence, independent of run_cycle()'s own
    idle-backoff ladder (which can back off to 300s -- presence needs to
    stay responsive). Debounces presence_debounce_readings (default 2)
    CONSISTENT non-null readings before flipping SENTRY_HOME_KEY and
    firing the activate/deactivate side effects."""
    log.info("sentry: presence_poll_loop starting")
    while not stop_event.wait(1):
        try:
            cfg = cfg_getter()
            interval = cfg.get("presence_poll_seconds", 60)
            state = load_state()
            now = time.time()
            last_poll = state.get(SENTRY_PRESENCE_POLL_AT_KEY, 0)
            if now - last_poll < interval:
                continue
            state[SENTRY_PRESENCE_POLL_AT_KEY] = now

            entity_id = cfg.get("ha_person_entity", "person.owner")
            result = ha_get_state(cfg, entity_id)
            raw = result.get("state") if result else None
            is_home = _presence_reading_to_home(raw)
            log.info("sentry: presence poll %s -> state=%r resolved=%r", entity_id, raw, is_home)

            if is_home is None:
                save_state(state)
                continue

            if state.get(SENTRY_DEBOUNCE_VALUE_KEY) == is_home:
                state[SENTRY_DEBOUNCE_COUNT_KEY] = state.get(SENTRY_DEBOUNCE_COUNT_KEY, 0) + 1
            else:
                state[SENTRY_DEBOUNCE_VALUE_KEY] = is_home
                state[SENTRY_DEBOUNCE_COUNT_KEY] = 1

            debounce_n = cfg.get("presence_debounce_readings", 2)
            was_home = state.get(SENTRY_HOME_KEY, True)

            if state[SENTRY_DEBOUNCE_COUNT_KEY] >= debounce_n and is_home != was_home:
                state[SENTRY_HOME_KEY] = is_home
                save_state(state)
                if is_home:
                    _sentry_deactivate(cfg, state, now, "presence: home")
                else:
                    _sentry_activate(cfg, state, now, "presence: away")
            else:
                save_state(state)
        except Exception as e:
            log.exception("sentry: presence_poll_loop iteration failed: %s", e)


def _sentry_rate_limited(cfg, state, now):
    times = [t for t in state.get(SENTRY_CAPTURE_TIMES_KEY, []) if now - t < 3600]
    state[SENTRY_CAPTURE_TIMES_KEY] = times
    per_minute = [t for t in times if now - t < 60]
    if len(per_minute) >= cfg.get("sentry_capture_max_per_minute", 1):
        return True
    if len(times) >= cfg.get("sentry_capture_max_per_hour", 30):
        return True
    return False


def _sentry_save_to_ha_www(cfg, jpeg_path):
    """Copies a capture into the HA www dir (bind-mounted at
    sentry_ha_www_dir) so HA's static file server exposes it at
    /local/vector/<file>.jpg -- works on LAN and over Tailscale alike,
    unlike a direct link to this host's own ports."""
    www_dir = cfg.get("sentry_ha_www_dir", "/mnt/ha-www")
    try:
        os.makedirs(www_dir, exist_ok=True)
        fname = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        dest = os.path.join(www_dir, fname)
        shutil.copyfile(jpeg_path, dest)
        _fix_owner(dest)
        return fname
    except Exception as e:
        log.warning("sentry: could not save capture into HA www dir: %s", e)
        return None


def _sentry_notify_unknown_person(cfg, fname, now):
    if not fname:
        log.warning("sentry: unknown-person alert but no image saved -- notifying without photo")
    tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
    dt = datetime.fromtimestamp(now, tz) if tz else datetime.fromtimestamp(now)
    hhmm = dt.strftime("%H:%M")
    service = cfg.get("ha_notify_service", "notify.mobile_app_YOUR_PHONE")
    domain, _, svc = service.partition(".")
    payload = {
        "title": "Vector Sentry",
        "message": f"Unknown person detected at {hhmm}",
    }
    if fname:
        url_prefix = cfg.get("sentry_ha_www_url_prefix", "/local/vector")
        payload["data"] = {"image": f"{url_prefix}/{fname}"}
    ok = ha_call_service(cfg, domain, svc, payload)
    log.info("sentry: unknown-person notify -> %s", "sent" if ok else "FAILED")
    return ok


def _sentry_post_to_coral(cfg, jpeg_path):
    url = cfg.get("sentry_coral_url", "http://localhost:8095/detect?model=efficientdet_lite1&min_score=0.4")
    timeout = cfg.get("http_timeout_seconds", 15)
    try:
        with open(jpeg_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "image/jpeg")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("sentry: coral detect failed: %s", e)
        return None


def _sentry_capture_and_detect(cfg, state):
    """ONE short-lived SDK connection (DEFAULT_PRIORITY, same as
    capture_image()) -- connect, read visible_faces, grab 1-2 frames,
    ALWAYS disconnect in `finally`. NEVER drives/wanders (sentry stays
    docked per spec). Serializes against capture_image()/
    capture_night_images()/drive_home_sdk() via the shared _SDK_LOCK --
    only one SDK connection to the robot anywhere in this process at a
    time. Same fail-closed voice gate (pre-connect check + continuous
    _voice_watch_loop watcher) as every other connection site.

    Returns {"paths": [...], "faces": [...]} on success, or None (skip/
    fail/voice-interrupted/lock-busy -- caller just doesn't act this
    trigger, next sound event tries again)."""
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel

    timeout_s = cfg.get("sentry_control_timeout_seconds", 15)

    if voice_recently_active(cfg, state):
        log.info("sentry: voice activity immediately before grab -- skipping (fail closed)")
        return None

    if not _SDK_LOCK.acquire(timeout=cfg.get("sdk_lock_wait_seconds", 3)):
        log.info("sentry: SDK busy elsewhere (lock held) -- skipping this trigger")
        return None

    robot = anki_vector.Robot(
        serial=cfg["robot_serial"],
        behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY,
        default_logging=False,
        enable_face_detection=True,
    )
    stop_watch = threading.Event()
    abort_box = {"aborted": False}
    result_box = {"paths": [], "faces": [], "reason": None}
    watcher = threading.Thread(
        target=_voice_watch_loop, args=(cfg, state, robot, stop_watch, abort_box), daemon=True,
    )

    def _do():
        watcher.start()
        robot.connect(timeout=min(10, timeout_s))
        time.sleep(0.5)
        st = robot.status
        ok, reason = _status_ok(
            {
                "is_being_held": bool(st.is_being_held),
                "are_motors_moving": bool(st.are_motors_moving),
                "is_animating": bool(st.is_animating),
                "is_pathing": bool(st.is_pathing),
                "is_in_calm_power_mode": bool(st.is_in_calm_power_mode),
            },
            allow_calm_power_mode=True,  # sentry: docked/resting is fine, still capture
        )
        if not ok:
            result_box["reason"] = reason
            return None
        try:
            result_box["faces"] = [
                {"name": (f.name or "").strip(), "face_id": f.face_id}
                for f in robot.world.visible_faces
            ]
        except Exception as e:
            log.debug("sentry: visible_faces read failed (non-fatal): %s", e)
        # "turn toward the sound direction if available (else stay put)" --
        # vector.log carries no per-event direction field for
        # @robot.reacted_to_sound (confirmed: 0 AudioDirection/MicDirection
        # lines in this robot's syslog), so direction is never available
        # here -- sentry always stays put, never drives/turns.
        n_shots = 2 if cfg.get("sentry_two_shots", True) else 1
        for i in range(n_shots):
            image = robot.camera.capture_single_image()
            if image is not None:
                result_box["paths"].append(_save_camera_image(image))
            if i == 0 and n_shots > 1:
                time.sleep(0.3)
        return True

    try:
        ok, timed_out = _run_with_hard_timeout(_do, timeout_s)
        stop_watch.set()
        try:
            record_face_sightings(cfg, state, result_box["faces"])
        except Exception as e:
            log.warning("sentry: record_face_sightings failed (non-fatal): %s", e)

        if abort_box["aborted"]:
            log.info("sentry: aborted mid-hold due to voice activity -- releasing")
            return None
        if timed_out:
            log.warning("sentry: HARD TIMEOUT after %ss -- forcing disconnect/release", timeout_s)
            _record_capture_outcome(state, time.time(), "fail")
            return None
        if result_box["reason"]:
            log.info("sentry: capture skipped -- %s", result_box["reason"])
            _record_capture_outcome(state, time.time(), "skip")
            return None
        if not result_box["paths"]:
            log.warning("sentry: connected but no image returned")
            _record_capture_outcome(state, time.time(), "fail")
            return None
        _record_capture_outcome(state, time.time(), "ok")
        log.info("sentry: captured %d frame(s)", len(result_box["paths"]))
        return {"paths": result_box["paths"], "faces": result_box["faces"]}
    finally:
        stop_watch.set()
        try:
            robot.disconnect()
        except Exception:
            pass
        _SDK_LOCK.release()


def _sentry_handle_capture(cfg, state, trigger, trigger_ts, capture):
    now = time.time()
    path_used = capture["paths"][0]
    detection = _sentry_post_to_coral(cfg, path_used)
    if detection is None:
        _sentry_log_event(cfg, {
            "ts": now, "trigger": trigger, "trigger_ts": trigger_ts,
            "type": "capture_no_detection_result",
        })
        return

    dets = detection.get("detections", [])
    min_score = cfg.get("sentry_person_min_score", 0.5)
    persons = [d for d in dets if d.get("label") == "person" and d.get("score", 0) >= min_score]
    animals = [d for d in dets if d.get("label") in ("cat", "dog")]
    named_faces = sorted({f["name"] for f in capture.get("faces", []) if f.get("name")})

    event = {
        "ts": now, "trigger": trigger, "trigger_ts": trigger_ts,
        "detections": dets, "faces": capture.get("faces", []),
        "type": "capture",
    }
    if animals:
        log.info("sentry: animal detected (%s) -- logging only, no alert", [a["label"] for a in animals])
        event["animal_labels"] = [a["label"] for a in animals]

    if persons:
        if named_faces:
            log.info("sentry: known face seen (%s) -- logging only", named_faces)
            event["type"] = "known_person"
            event["known_names"] = named_faces
        else:
            log.warning("sentry: UNKNOWN PERSON detected (score up to %.2f) -- alerting",
                        max((d.get("score", 0) for d in persons), default=0))
            www_fname = _sentry_save_to_ha_www(cfg, path_used)
            _sentry_notify_unknown_person(cfg, www_fname, now)
            event["type"] = "unknown_person_alert"
            event["ha_www_file"] = www_fname

    _sentry_log_event(cfg, event)


def _sentry_maybe_capture(cfg, state, trigger_ts, trigger):
    now = time.time()
    if _sentry_rate_limited(cfg, state, now):
        log_skip("sentry: capture rate-limited", 30)
        save_state(state)
        return

    battery = get_battery(cfg)
    if battery is None:
        log_skip("sentry: could not verify battery -- skipping capture (fail safe)", 60)
        return
    level = battery.get("battery_level")
    on_charger = battery.get("is_on_charger_platform")
    low_level = cfg.get("sentry_battery_low_level", cfg.get("battery_low_level", 5))
    if level is not None and level <= low_level and not on_charger:
        log_skip("sentry: battery low -- skipping capture", 300)
        return

    if music_playing(cfg):
        log_skip("sentry: music playing -- skipping capture", 30)
        return
    if brain_busy(cfg):
        log_skip("sentry: brain-proxy busy -- skipping capture", 30)
        return

    capture = _sentry_capture_and_detect(cfg, state)
    state.setdefault(SENTRY_CAPTURE_TIMES_KEY, []).append(now)
    save_state(state)
    if capture is None:
        return
    _sentry_handle_capture(cfg, state, trigger, trigger_ts, capture)


def sentry_log_watch_loop(cfg_getter, stop_event):
    """Background daemon thread, started once from main(). Tails
    vector.log for the sentry trigger pattern (see module-level comment
    block above for why @robot.reacted_to_sound was chosen). Tracks a
    persisted byte offset (state.json) so a container restart resumes
    from where it left off instead of either replaying the whole log or
    losing the current position; a shrunk/rotated file (offset > current
    size) resets to end-of-file rather than erroring."""
    log.info("sentry: sentry_log_watch_loop starting")
    while not stop_event.wait(cfg_getter().get("sentry_log_poll_seconds", 3)):
        try:
            cfg = cfg_getter()
            path = cfg.get("sentry_vector_log_path", "/mnt/vector-logs/vector.log")
            pattern = cfg.get("sentry_trigger_pattern", "reacted_to_sound")
            try:
                size = os.path.getsize(path)
            except OSError as e:
                log.warning("sentry: cannot stat vector log at %s: %s", path, e)
                continue

            state = load_state()
            offset = state.get(SENTRY_LOG_OFFSET_KEY)
            if offset is None or offset > size:
                state[SENTRY_LOG_OFFSET_KEY] = size
                save_state(state)
                continue
            if size == offset:
                continue

            with open(path, "r", errors="replace") as f:
                f.seek(offset)
                chunk = f.read(2_000_000)  # defensive cap per poll
            state[SENTRY_LOG_OFFSET_KEY] = offset + len(chunk.encode("utf-8", "replace"))
            save_state(state)

            trigger_ts = None
            for line in chunk.splitlines():
                if pattern not in line.lower():
                    continue
                m = SENTRY_SOUND_TS_RE.match(line)
                if m:
                    try:
                        trigger_ts = datetime(
                            int(m["y"]), int(m["mo"]), int(m["d"]),
                            int(m["h"]), int(m["mi"]), int(m["s"]),
                            tzinfo=timezone.utc,
                        ).timestamp()
                    except ValueError:
                        trigger_ts = time.time()
                else:
                    trigger_ts = time.time()

            if trigger_ts is None:
                continue

            cfg = cfg_getter()
            state = load_state()
            if not sentry_is_active(cfg, state):
                continue
            if cfg.get("sentry_respect_quiet_hours", True) and in_quiet_hours(cfg):
                log_skip("sentry: quiet hours -- no capture", 300)
                continue
            _sentry_maybe_capture(cfg, state, trigger_ts, "sound")
        except Exception as e:
            log.exception("sentry: sentry_log_watch_loop iteration failed: %s", e)


# ---------------------------------------------------------------------------
# SAFETY ALERTS -- "call for help" (2026-09-29, Hermes, post-incident: Vector
# drove off the desk overnight because live-13's thinking-animation loop
# scooted him with firmware cliff-stop disabled -- see the wire-pod
# thinkinganim.go fix for the root cause/fix on that side. This module is
# the OTHER half: even with the animation bug fixed, the owner should never
# again find out about a fall/stuck robot by tripping over him. It watches
# the SAME vector.log this container already tails for sentry, using the
# exact same offset-tracked tail pattern as sentry_log_watch_loop (own
# offset key so the two loops' positions never interfere), but
# deliberately runs UNGATED by sentry_is_active()/quiet-hours/presence --
# a fall alert must fire whether the owner is home, asleep, or on vacation.
#
# Event classification is based on REAL event strings observed in this
# robot's own vector.log (grepped directly, not guessed):
#   - "@behavior.feature.startReactToCliff..." -- a NORMAL, safe edge-stop.
#     A single one must NEVER page (the owner set the robot down near a table
#     edge, cliff sensor did its job). Only 2+ within
#     `safety_cliff_repeat_window_seconds` (default 60s) count as "stuck
#     re-triggering the same cliff", which DOES page.
#   - "@behavior.feature.startStuckOnEdge..." -- the actual "I am stuck and
#     can't recover" firmware behavior (distinct from, and rarer than,
#     ReactToCliff). Always pages. Chosen over the very noisy/continuous
#     "@mood.eventReactToUnexpectedMovement..." mood-delta ticks (which
#     fire every ~4s throughout a stuck episode and are not themselves a
#     discrete event) as the reliable single "stuck" signal.
#   - "@behavior.feature.startInTheAir..." -- robot picked up. Always pages
#     per the owner's own spec ("page on in-the-air, fall, or on-side
#     events").
#   - "@robot.offtreadsstatechanged..." containing Falling/OnFace/OnBack/
#     OnSide -- an actual fall or landed-wrong event. Always pages.
#
# Each of the four alert keys (fall / stuck / in_the_air / cliff_repeated)
# is independently rate-limited to at most one HA push per
# `safety_alert_min_interval_seconds` (default 600s = 10 minutes) -- state
# persisted in state.json (SAFETY_ALERT_LAST_AT_KEY), survives a container
# restart. Every classified event (whether or not it pages) is appended to
# safety.jsonl for the forensic record.
# ---------------------------------------------------------------------------

SAFETY_LOG_OFFSET_KEY = "safety_log_offset"          # byte offset into vector.log (own cursor, separate from sentry's)
SAFETY_ALERT_LAST_AT_KEY = "safety_alert_last_at"    # dict: alert_key -> epoch of last HA push sent
SAFETY_CLIFF_RECENT_KEY = "safety_recent_cliff_times"  # list[epoch] of recent lone ReactToCliff starts (repeated-cliff detection)

SAFETY_ALERT_MIN_INTERVAL_SECONDS_DEFAULT = 600  # 10 minutes, per alert key


def _safety_classify_line(line):
    """Returns 'fall' / 'stuck' / 'in_the_air' / 'cliff' for a
    safety-relevant vector.log line, else None. Matches on the exact
    "@behavior.feature.start<Name>" tag (not "pre_start<Name>" or
    "end<Name>") so each real-world episode is counted exactly once --
    substring-matching "reacttocliff" alone would ALSO match the
    "pre_startReactToCliff..." line firmware always logs a few ms earlier
    for the same episode, double-counting every single cliff stop and
    falsely tripping the repeated-cliff rule on one safe edge-stop.
    """
    low = line.lower()
    if "offtreadsstatechanged" in low and (
        "falling" in low or "onface" in low or "onback" in low or "onside" in low
    ):
        return "fall"
    if "@behavior.feature.startstuckonedge" in low:
        return "stuck"
    if "@behavior.feature.startintheair" in low:
        return "in_the_air"
    if "@behavior.feature.startreacttocliff" in low:
        return "cliff"
    return None


def _safety_should_alert(state, alert_key, now, min_interval):
    last_by_key = state.get(SAFETY_ALERT_LAST_AT_KEY, {})
    return (now - last_by_key.get(alert_key, 0)) >= min_interval


def _safety_mark_alerted(state, alert_key, now):
    last_by_key = dict(state.get(SAFETY_ALERT_LAST_AT_KEY, {}))
    last_by_key[alert_key] = now
    state[SAFETY_ALERT_LAST_AT_KEY] = last_by_key


def _safety_log_event(cfg, record):
    """Appends one JSON line to the safety metrics log -- same
    open/append/_fix_owner pattern as _sentry_log_event. Every classified
    event is recorded here regardless of whether it triggers a push, so a
    burst of isolated ReactToCliff stops below the 'repeated' threshold is
    still visible in the forensic record."""
    path = cfg.get("safety_events_log_path", "/mnt/vector-brain/metrics/safety.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        _fix_owner(path)
    except Exception as e:
        log.warning("safety: failed writing event log: %s", e)


def _safety_notify(cfg, message):
    """HA push via the same notify service + ha_call_service helper
    sentry's unknown-person alert already uses
    (_sentry_notify_unknown_person) -- no photo attachment here (fall/
    stuck/picked-up events have none)."""
    service = cfg.get("ha_notify_service", "notify.mobile_app_YOUR_PHONE")
    domain, _, svc = service.partition(".")
    payload = {"title": "Vector safety alert", "message": message}
    return ha_call_service(cfg, domain, svc, payload)


def _safety_format_time(cfg, ts):
    tz = ZoneInfo(cfg.get("timezone", "America/New_York")) if ZoneInfo else None
    dt = datetime.fromtimestamp(ts, tz) if tz else datetime.fromtimestamp(ts)
    try:
        return dt.strftime("%-I:%M %p")  # e.g. "9:53 PM" (no leading zero)
    except ValueError:
        return dt.strftime("%I:%M %p")   # Windows fallback (unused in prod, kept for portability)


def _safety_handle_event(cfg, state, alert_type, event_ts, raw_line):
    """Records every classified event; sends at most one HA push per
    alert key per safety_alert_min_interval_seconds. Returns the alert_key
    that was (or would have been) used, mainly so the selftest/dry-run can
    assert on it."""
    _safety_log_event(cfg, {"ts": event_ts, "type": alert_type, "line": raw_line.strip()})

    hhmm = _safety_format_time(cfg, event_ts)
    min_interval = cfg.get("safety_alert_min_interval_seconds", SAFETY_ALERT_MIN_INTERVAL_SECONDS_DEFAULT)

    if alert_type == "cliff":
        window = cfg.get("safety_cliff_repeat_window_seconds", 60)
        recent = [t for t in state.get(SAFETY_CLIFF_RECENT_KEY, []) if event_ts - t < window]
        recent.append(event_ts)
        state[SAFETY_CLIFF_RECENT_KEY] = recent
        save_state(state)
        if len(recent) < 2:
            log_skip("safety: single ReactToCliff -- normal safe edge-stop, not paging", 60)
            return "cliff_below_threshold"
        message = f"Vector keeps hitting a cliff sensor near {hhmm} -- may be stuck at an edge"
        alert_key = "cliff_repeated"
    elif alert_type == "fall":
        message = f"Vector took a tumble at {hhmm}"
        alert_key = "fall"
    elif alert_type == "stuck":
        message = f"Vector is stuck at {hhmm}"
        alert_key = "stuck"
    elif alert_type == "in_the_air":
        message = f"Vector was picked up at {hhmm}"
        alert_key = "in_the_air"
    else:
        return None

    if not _safety_should_alert(state, alert_key, event_ts, min_interval):
        log_skip(f"safety: {alert_key} alert rate-limited (<{min_interval}s since last)", 60)
        return alert_key + "_rate_limited"

    ok = _safety_notify(cfg, message)
    log.warning("safety: %s alert -> %s (%r)", alert_key, "sent" if ok else "FAILED", message)
    if ok:
        _safety_mark_alerted(state, alert_key, event_ts)
        save_state(state)
    return alert_key


def safety_log_watch_loop(cfg_getter, stop_event):
    """Background daemon thread, started once from main(). Tails
    vector.log with its own persisted byte offset (SAFETY_LOG_OFFSET_KEY,
    independent of sentry's own offset) -- see the module docstring above
    for why this loop is NOT gated by sentry_is_active()/quiet hours/
    presence. Same shrunk/rotated-file handling as sentry_log_watch_loop:
    an offset past the current file size resets to end-of-file rather than
    erroring or replaying stale history."""
    log.info("safety: safety_log_watch_loop starting")
    while not stop_event.wait(cfg_getter().get("safety_log_poll_seconds", 3)):
        try:
            cfg = cfg_getter()
            path = cfg.get("sentry_vector_log_path", "/mnt/vector-logs/vector.log")
            try:
                size = os.path.getsize(path)
            except OSError as e:
                log.warning("safety: cannot stat vector log at %s: %s", path, e)
                continue

            state = load_state()
            offset = state.get(SAFETY_LOG_OFFSET_KEY)
            if offset is None or offset > size:
                state[SAFETY_LOG_OFFSET_KEY] = size
                save_state(state)
                continue
            if size == offset:
                continue

            with open(path, "r", errors="replace") as f:
                f.seek(offset)
                chunk = f.read(2_000_000)  # defensive cap per poll, same as sentry's tail
            state[SAFETY_LOG_OFFSET_KEY] = offset + len(chunk.encode("utf-8", "replace"))
            save_state(state)

            for line in chunk.splitlines():
                alert_type = _safety_classify_line(line)
                if alert_type is None:
                    continue
                m = SENTRY_SOUND_TS_RE.match(line)
                if m:
                    try:
                        event_ts = datetime(
                            int(m["y"]), int(m["mo"]), int(m["d"]),
                            int(m["h"]), int(m["mi"]), int(m["s"]),
                            tzinfo=timezone.utc,
                        ).timestamp()
                    except ValueError:
                        event_ts = time.time()
                else:
                    event_ts = time.time()
                state = load_state()
                _safety_handle_event(cfg, state, alert_type, event_ts, line)
        except Exception as e:
            log.exception("safety: safety_log_watch_loop iteration failed: %s", e)


def _safety_selftest():
    """Dry-run of the pure-logic pieces with sample log lines -- no
    network, no SDK, no container needed. Run with:
    python3 curious.py --safety-selftest

    IMPORTANT: this must NEVER hit the real Home Assistant API.
    _safety_handle_event() calls module-level _safety_notify() for any
    alert that clears rate-limiting, and _safety_notify() is a real
    ha_call_service() HTTP POST -- so a naive dry-run that just calls
    _safety_handle_event() directly would actually push real
    notifications to the owner's phone every time this selftest runs (this
    happened once, 2026-09-29, during the initial version of this test --
    5 unwanted "Vector safety alert" pushes went to
    notify.mobile_app_YOUR_PHONE). Fixed by monkeypatching the
    module-level _safety_notify for the duration of this function only,
    with a fake that records calls instead of sending them, and restoring
    the real one in a finally block no matter how the test exits."""
    ok = True
    sent_notifications = []

    def _fake_notify(cfg, message):
        sent_notifications.append(message)
        return True

    global _safety_notify
    _real_safety_notify = _safety_notify
    _safety_notify = _fake_notify

    def check(label, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}: {label}")
        if not cond:
            ok = False

    try:
        return _safety_selftest_body(check, sent_notifications, ok)
    finally:
        _safety_notify = _real_safety_notify


def _safety_selftest_body(check, sent_notifications, ok):

    # --- classification on real-shaped sample lines ---
    sample_cliff_prestart = "2026-09-29 21:51:22.317 [ROBOT_IP] <14>Sep 29 17:51:22 Vector-A1B2 vic-engine[4922]: @behavior.feature.pre_startReactToCliffLowStim00A4B968-75A5-E26E-00D0-0006CF7989ACAutonomous250067"
    sample_cliff_start = "2026-09-29 21:51:22.318 [ROBOT_IP] <14>Sep 29 17:51:22 Vector-A1B2 vic-engine[4922]: @behavior.feature.startReactToCliffAI00A4B968-75A5-E26E-00D0-0006CF7989ACAutonomous250067"
    sample_stuck_start = "2026-09-29 20:42:31.296 [ROBOT_IP] <14>Sep 29 16:42:30 Vector-A1B2 vic-engine[4922]: @behavior.feature.startStuckOnEdgeAI0004FCFB-9073-4B72-00F0-4FBDACB15C8EAutonomous50025781"
    sample_inair_start = "2026-09-29 01:07:38.692 [ROBOT_IP] <14>Sep 28 21:07:38 Vector-A1B2 vic-engine[946]: @behavior.feature.startInTheAirAI006873F4-57C1-EE76-0042-3F77D3A3C835Social5830649"
    sample_falling = "2026-09-29 01:07:40.027 [ROBOT_IP] <14>Sep 28 21:07:40 Vector-A1B2 vic-engine[946]: @robot.offtreadsstatechangedInAirFalling5832459"
    sample_onface = "2026-09-29 03:05:07.435 [ROBOT_IP] <14>Sep 28 23:05:07 Vector-A1B2 vic-engine[946]: @robot.offtreadsstatechangedOnFaceInAir12879099"
    sample_irrelevant = "2026-09-29 21:50:19.190 [ROBOT_IP] <14>Sep 28 21:50:18 Vector-A1B2 vic-robot[866]: IMUFilter.Bias: 0.267389 0.108637 -0.201036 (deg/sec)"
    sample_ontreads = "2026-09-29 21:50:07.793 [ROBOT_IP] <14>Sep 29 17:50:07 Vector-A1B2 vic-engine[4922]: @robot.offtreadsstatechangedOnTreadsInAir3454082392"

    check("pre_startReactToCliff is NOT classified (avoids double count)", _safety_classify_line(sample_cliff_prestart) is None)
    check("startReactToCliff -> 'cliff'", _safety_classify_line(sample_cliff_start) == "cliff")
    check("startStuckOnEdge -> 'stuck'", _safety_classify_line(sample_stuck_start) == "stuck")
    check("startInTheAir -> 'in_the_air'", _safety_classify_line(sample_inair_start) == "in_the_air")
    check("offtreadsstatechanged...Falling -> 'fall'", _safety_classify_line(sample_falling) == "fall")
    check("offtreadsstatechanged...OnFace -> 'fall'", _safety_classify_line(sample_onface) == "fall")
    check("unrelated IMU line -> None", _safety_classify_line(sample_irrelevant) is None)
    check("OnTreadsInAir (still riding, no fall) -> None", _safety_classify_line(sample_ontreads) is None)

    # --- single ReactToCliff never alerts; a 2nd within the window does ---
    cfg = {"safety_cliff_repeat_window_seconds": 60, "safety_alert_min_interval_seconds": 600}
    state = {}
    r1 = _safety_handle_event(cfg, state, "cliff", 1000.0, sample_cliff_start)
    check("1st isolated cliff stop does not page", r1 == "cliff_below_threshold")
    r2 = _safety_handle_event(cfg, state, "cliff", 1030.0, sample_cliff_start)
    check("2nd cliff stop within 60s DOES page", r2 == "cliff_repeated")

    # --- cliff events far apart never accumulate into a page ---
    state2 = {}
    _safety_handle_event(cfg, state2, "cliff", 2000.0, sample_cliff_start)
    r3 = _safety_handle_event(cfg, state2, "cliff", 2200.0, sample_cliff_start)  # 200s later, outside 60s window
    check("cliff stops 200s apart (outside window) do not page", r3 == "cliff_below_threshold")

    # --- fall/stuck/in_the_air always alert (first occurrence) ---
    check("fall alerts immediately", _safety_handle_event(cfg, {}, "fall", 3000.0, sample_falling) == "fall")
    check("stuck alerts immediately", _safety_handle_event(cfg, {}, "stuck", 3000.0, sample_stuck_start) == "stuck")
    check("in_the_air alerts immediately", _safety_handle_event(cfg, {}, "in_the_air", 3000.0, sample_inair_start) == "in_the_air")

    # --- per-type rate limiting (10 min default) ---
    state3 = {}
    r4 = _safety_handle_event(cfg, state3, "fall", 4000.0, sample_falling)
    r5 = _safety_handle_event(cfg, state3, "fall", 4060.0, sample_falling)  # 60s later, well under 600s
    check("fall alert fires once", r4 == "fall")
    check("2nd fall 60s later is rate-limited", r5 == "fall_rate_limited")
    r6 = _safety_handle_event(cfg, state3, "fall", 4700.0, sample_falling)  # 700s later, past the window
    check("3rd fall 700s later (past 600s window) pages again", r6 == "fall")

    # Every alert above that "sent" actually went to the fake recorder,
    # never to the real HA API -- this is the assertion that makes this
    # genuinely a dry-run.
    check("all alerts captured by the fake notifier (none hit real HA)",
          len(sent_notifications) == 6)  # cliff_repeated, fall, stuck, in_the_air, +2 more fall (rate-limited 2nd fall does NOT add one; the 3rd, past-window fall does)

    print("SELFTEST", "PASSED" if ok else "FAILED")
    print(f"(dry-run: {len(sent_notifications)} notification(s) would have been sent -- none actually transmitted)")
    return ok


def _sentry_selftest():
    """Dry-run of the pure-logic pieces with mocks -- no network, no SDK,
    no container needed. Run with: python3 curious.py --sentry-selftest"""
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}: {label}")
        if not cond:
            ok = False

    check("home reading", _presence_reading_to_home("home") is True)
    check("not_home reading", _presence_reading_to_home("not_home") is False)
    check("zone reading counts as away", _presence_reading_to_home("Work") is False)
    check("unknown reading is None (unverifiable)", _presence_reading_to_home("unknown") is None)
    check("unavailable reading is None", _presence_reading_to_home("unavailable") is None)
    check("missing reading is None", _presence_reading_to_home(None) is None)

    cfg_off = {"sentry_enabled": True, "sentry_manual_override": "off"}
    cfg_on = {"sentry_enabled": True, "sentry_manual_override": "on"}
    cfg_auto = {"sentry_enabled": True, "sentry_manual_override": "auto"}
    cfg_disabled = {"sentry_enabled": False}
    check("override off -> never active", sentry_is_active(cfg_off, {SENTRY_HOME_KEY: False}) is False)
    check("override on -> always active", sentry_is_active(cfg_on, {SENTRY_HOME_KEY: True}) is True)
    check("auto + away -> active", sentry_is_active(cfg_auto, {SENTRY_HOME_KEY: False}) is True)
    check("auto + home -> inactive", sentry_is_active(cfg_auto, {SENTRY_HOME_KEY: True}) is False)
    check("auto + no reading yet -> inactive (fail-safe default)",
          sentry_is_active(cfg_auto, {}) is False)
    check("sentry_enabled=False -> never active regardless of override",
          sentry_is_active(cfg_disabled, {SENTRY_HOME_KEY: False}) is False)

    now = 1_000_000.0
    cfg_rl = {"sentry_capture_max_per_minute": 1, "sentry_capture_max_per_hour": 3}
    st = {SENTRY_CAPTURE_TIMES_KEY: []}
    check("not rate-limited when empty", _sentry_rate_limited(cfg_rl, st, now) is False)
    st[SENTRY_CAPTURE_TIMES_KEY] = [now - 10]
    check("rate-limited within same minute", _sentry_rate_limited(cfg_rl, st, now) is True)
    st[SENTRY_CAPTURE_TIMES_KEY] = [now - 3500, now - 3400, now - 3300]
    check("rate-limited at hourly cap even outside the minute window",
          _sentry_rate_limited(cfg_rl, st, now) is True)
    st[SENTRY_CAPTURE_TIMES_KEY] = [now - 4000]  # pruned (>3600s old)
    check("stale capture times pruned from the hourly window",
          _sentry_rate_limited(cfg_rl, st, now) is False)

    # Debounce simulation: two consistent "away" readings should flip
    # home=True -> home=False; a single flaky reading should not.
    state = {SENTRY_HOME_KEY: True}
    for reading in (False,):  # only one reading -- must NOT flip yet
        if state.get(SENTRY_DEBOUNCE_VALUE_KEY) == reading:
            state[SENTRY_DEBOUNCE_COUNT_KEY] = state.get(SENTRY_DEBOUNCE_COUNT_KEY, 0) + 1
        else:
            state[SENTRY_DEBOUNCE_VALUE_KEY] = reading
            state[SENTRY_DEBOUNCE_COUNT_KEY] = 1
    would_flip = state[SENTRY_DEBOUNCE_COUNT_KEY] >= 2 and False != state.get(SENTRY_HOME_KEY, True)
    check("single away reading does not flip (debounce=2)", would_flip is False)
    # second consistent reading
    reading = False
    if state.get(SENTRY_DEBOUNCE_VALUE_KEY) == reading:
        state[SENTRY_DEBOUNCE_COUNT_KEY] += 1
    would_flip = state[SENTRY_DEBOUNCE_COUNT_KEY] >= 2 and reading != state.get(SENTRY_HOME_KEY, True)
    check("second consistent away reading flips (debounce=2)", would_flip is True)

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--sentry-selftest" in sys.argv:
        sys.exit(0 if _sentry_selftest() else 1)
    if "--safety-selftest" in sys.argv:
        sys.exit(0 if _safety_selftest() else 1)
    if "--explore-selftest" in sys.argv:
        sys.exit(0 if _explore_selftest() else 1)
    main()
