#!/usr/bin/env python3
"""
play.py -- "play music" custom intent for Vector, via wire-pod + yt-dlp.

Usage (called by wire-pod custom intent exec, or manually):
    play.py <query words...>                 # normal: detach, download, stream to robot
    play.py --no-robot <query words...>       # offline test: download+convert only, print format
    play.py --max-seconds N <query words...>  # cap playback duration (also caps download)
    play.py --mode {control,nocontrol,reserve,wander} <query...>  # behavior-control strategy
    play.py --dance <query words...>           # force dance-while-streaming (mode=control only)
    play.py --worker <query...>                # internal: actual worker body (do not call directly)

DANCE MODE (added 2026-09-28, the owner's "dance party" request):
"Hey Vector, dance" doesn't work DURING a song because the built-in
dance-to-the-beat behavior needs behavior control + the mic, both of which
this script already holds/occupies in `control` mode. So dancing has to be
driven from inside play.py itself, on a background thread alongside the
blocking `robot.audio.stream_wav_file()` call in the main thread.

This works because every SDK sync wrapper method (stream_wav_file,
play_animation_trigger, set_head_angle, set_lift_height, turn_in_place, ...)
is really `asyncio.run_coroutine_threadsafe(coro, conn.loop).result()` under
the hood (see connection.py's `on_connection_thread` decorator / Connection.
run_coroutine) -- there's ONE shared asyncio event loop for the whole
connection, running on its own thread. Calling one of these methods from any
Python thread just submits a coroutine onto that shared loop and blocks
*that calling thread* (not the loop, not the other callers) until it
finishes. Since stream_wav_file awaits network responses in a loop, it
yields control regularly, so a second thread's animation/motion coroutines
genuinely interleave on the same loop instead of queuing up behind it. This
is the exact same pattern `_stream_wander` already uses below (a background
thread streams audio while the main thread drives wheels) -- dance mode just
flips which side of that split is on the main thread.

Dance moves (`_dance_worker`, reworked 2026-10-15 after the owner's "stutters
and raises his arm" feedback on the old hand-rolled nod/bob/turn loop): a
real "DanceBeatGetIn" to enter dance stance, then the robot's OWN stock
anim_dancebeat_* clips (the same ones "Hey Vector, dance" uses) played
back-to-back with robot.anim.play_animation() -- no manual time.sleep()
between calls, since play_animation() already blocks until that clip
finishes, so one call ending IS the next one's gap-free start. No more raw
set_head_angle/set_lift_height/turn_in_place commands (that hand-rolled lift
bob was the literal "raises his arm" complaint); any lift motion now only
happens if the specific anim clip authored it. Which clips get picked leans
on a lightweight local BPM estimate (`estimate_bpm`, numpy-only -- no
librosa/scipy in this venv and the brief says not to add heavy deps) of the
already-downloaded WAV: faster songs favor the small/quick clips
(eyebeat/headnod), slower songs favor the bigger head+lift+body clips. Both
anim_list (needed for play_animation by name) and anim_trigger_list (needed
for the GetIn/GetOut trigger bookends) are pre-loaded once, while the
connection is still idle, before the stream starts -- see
_resolve_dance_anim_pool and the 2026-09-28 test note on _dance_worker below
for why a metadata lookup during an active stream is not safe to risk again.
"DanceBeatGetOut" runs on a clean end-of-song exit (best-effort; a stop.sh
SIGTERM just kills the whole process/thread, so GetOut is skipped in that
case -- known cosmetic limit, not worth the complexity of a signal handler
for a pose reset).

Voice routing: "play the song X and dance" reaches this script's normal
play_music_custom path (wire-pod matches "play the song" first), and
`resolve_dance_and_query()` below detects the trailing "and dance" itself,
strips it from the search query, and turns dance mode on. A standalone
"dance party" / "dance to the music" / "let's dance" phrase is routed here
via a NEW dance_music_custom custom intent (see customIntents.json) and,
having nothing left to search for, falls back to a random pick from
DANCE_PARTY_FALLBACK_SONGS.

Mid-song "dance" (said WHILE a song is already streaming) is NOT wired to
voice on purpose: wire-pod matches custom intents by plain substring across
ALL configured utterances, checked before any built-in/system intent, with
no way to scope a match to "only when music is playing". A bare "dance"
utterance in customIntents.json would therefore permanently shadow the
built-in "Hey Vector, dance" behavior even when nothing is playing -- exactly
the breakage this feature is supposed to avoid, not add. Also, in practice
the mic may not even open for a bare "dance" while streaming holds control.
Instead there's a manual/future hook: `_dance_flag_watcher` polls
DANCE_FLAG_PATH (/opt/vector-music/dance.flag) about once a second while
streaming; if that file appears mid-song it's consumed and dancing starts
without interrupting the song. Nothing currently touches that file -- it's
there for `touch /opt/vector-music/dance.flag` testing or a future, more
careful voice wiring (e.g. a wake-word-scoped intent) -- documented instead
of implemented, per the brief.

IMPORTANT (found while reading wire-pod source, pkg/wirepod/ttr/matchIntentSend.go):
wire-pod's customIntentHandler runs the exec command with Cmd.Run() -- which
BLOCKS until the process exits -- before it sends the intent back to the
robot (IntentPass). If play.py did the full download+stream in the
foreground, wire-pod (and the robot's voice session) would hang for up to
--max-seconds/4 minutes on every "play music" request. To avoid that, the
default (non---no-robot, non---worker) invocation immediately double-forks
a detached background worker and returns in well under a second, so
wire-pod's exec call completes fast and the robot gets its ack intent
right away while the download/stream continues in the background.

MODES (added 2026-09-25, GL explore-while-playing experiment):
`anki_vector.Robot(...)` as a context manager requests behavior control on
enter (DEFAULT_PRIORITY by default), which freezes Vector's own idle/explore
behaviors for the duration -- he just stands there while music plays. To let
him keep roaming while music streams, `AudioComponent.stream_wav_file` is
decorated `@on_connection_thread(requires_control=True)`, which is a
CLIENT-SIDE check only (see anki_vector/connection.py on_connection_thread's
log_handler: `if requires_control and not control.is_set(): ...raise
VectorControlException`) -- there's no server-side gate baked into the
gRPC call itself (`ExternalAudioStreamPlayback`). So:
  - `nocontrol`: connect with `behavior_control_level=None` (no control ever
    requested) and call the same request-building internals
    (`AudioComponent._open_file` / `_request_handler`) directly against
    `robot.audio.grpc_interface.ExternalAudioStreamPlayback(...)`, run via
    `robot.conn.run_coroutine(...)`, bypassing the decorator's client-side
    gate entirely.
  - `reserve`: same bypass, but connect with
    `behavior_control_level=ControlPriorityLevel.RESERVE_CONTROL` (per the
    SDK docstring, RESERVE_CONTROL is used to disable idle behaviors -- kept
    here for comparison, expected to still freeze him).
  - `wander`: take full DEFAULT_PRIORITY control (freezes built-in behaviors
    on purpose) and drive our own simple roam loop with `robot.motors` +
    `robot.behavior.turn_in_place` while `robot.audio.stream_wav_file` runs
    in a background thread. Only used as a fallback if nocontrol/reserve
    don't actually keep him moving, since it fights DEFAULT_PRIORITY control
    rather than avoiding it.

MY MUSIC (added 2026-09-26): "play my liked songs" / "play my <name>
playlist" / "play my <name> mix" resolve tracks via ytmusicapi against
the owner's YouTube Music account (see ytm_common.py / ytm_auth.py / README.md)
and queue them sequentially through this same download-by-id -> convert ->
stream pipeline, one videoId at a time. If ytmusicapi isn't authenticated yet
(no oauth.json) or nothing resolves (no match / empty playlist), this falls
back to the normal single-song public search below and logs why. Everything
else ("play the song X", "play some music") is unchanged.
"""
import argparse
import fcntl
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import wave

VECTOR_MUSIC_DIR = "/opt/vector-music"
VENV_PY = f"{VECTOR_MUSIC_DIR}/.venv/bin/python3"
LOG_PATH = f"{VECTOR_MUSIC_DIR}/music.log"
PID_PATH = "/run/vector-music.pid"
PID_PATH_FALLBACK = "/tmp/vector-music.pid"

DEFAULT_MAX_SECONDS = 240  # 4 minutes cap per brief
SDK_TARGET_RATE = 16000    # SDK accepts 8000-16025 Hz mono 16-bit; 16000 is safe middle
DEFAULT_VOLUME = 100       # 0-100; tiny speaker, run it at max (the owner 2026-09-25)

# Behavior-control strategy for streaming. See MODES docstring above.
DEFAULT_MODE = "control"
VALID_MODES = ("control", "nocontrol", "reserve", "wander")

# Dance mode (see DANCE MODE docstring above). Only meaningful with mode="control"
# -- that's the mode that actually holds DEFAULT_PRIORITY behavior control, which
# animation/motion calls require.
DANCE_MOVE_CADENCE_S = 2.5  # rough seconds between dance moves (brief: ~2-3s)
DANCE_FLAG_PATH = "/opt/vector-music/dance.flag"  # manual/future mid-song dance trigger

# Music ducking (added 2026-10-15, the owner's dance-party session -- see
# _duck_watcher below). Touched/removed by duck_watcher.sh, a host-side
# script (NOT inside this container) that tails `docker logs -f vector-pod`
# for "Stream type: OPUS" (voice capture starting) / "Transcribed text:"
# (capture ended) and mirrors that into this flag file, which lives in the
# same bind-mounted /opt/vector-music dir play.py already uses.
DUCK_FLAG_PATH = "/opt/vector-music/duck.flag"
DUCK_POLL_S = 0.3

# Cross-process robot-control lock (added 2026-10-15, hermes/vector-faces --
# see curious.py's _SDK_LOCK in vector-curious, same day). vector-curious,
# vector-faces' enroll.py, and vector-daycycle (host cron) all take an
# exclusive fcntl.flock on ~/vector-pod/data/robot_control.lock
# before connecting the anki_vector SDK to the robot. play.py was NOT part
# of that change, so vector-curious's periodic capture_image() SDK
# connections could (and on 2026-09-28, did -- see "dance move failed:
# connection has been closed" in music.log) steal behavior control out from
# under a running dance session. This mirrors the SAME lock class so play.py
# participates too. Inside THIS container the file is reachable at
# /data/robot_control.lock (the existing /data bind mount is the parent of
# the /opt/vector-music bind mount -- no new mount or restart needed). Fails
# open (no-op) if the file isn't mounted, matching curious.py's behavior.
ROBOT_CONTROL_LOCK_PATH = "/data/robot_control.lock"


class _RobotControlLock:
    """Non-blocking, poll-with-timeout exclusive flock wrapper. Same
    interface and implementation as vector-curious/curious.py's
    _RobotControlLock (deliberately kept near-identical so the two scripts'
    locking behavior can't drift) -- see that file's header comment for the
    full cross-process rationale. Unlike curious.py's version, there's no
    in-process threading.Lock layer here: play.py's stream_to_robot runs
    once per worker process, so there's no second same-process caller to
    serialize against."""

    def __init__(self, path):
        self._path = path
        self._fh = None
        self._warned = False

    def _warn_once(self, msg):
        if not self._warned:
            self._warned = True
            log.warning("robot_control.lock: %s -- proceeding without it", msg)

    def acquire(self, timeout=5):
        deadline = time.time() + max(0.0, timeout)
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
                    log.info("robot_control.lock held by another process (vector-curious/enroll.py/daycycle)")
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


_SDK_LOCK = _RobotControlLock(ROBOT_CONTROL_LOCK_PATH)

# Real dance-behavior clips (same anims the stock "Hey Vector, dance" behavior
# uses). Bookends played once each, while the connection is idle, before/after
# the audio stream -- see the 2026-09-28 test note on _dance_worker for why
# these (and any other anim-list/anim-trigger-list metadata lookup) must never
# happen concurrently with the audio stream.
DANCE_GET_IN_TRIGGER = "DanceBeatGetIn"
DANCE_GET_OUT_TRIGGER = "DanceBeatGetOut"

# Reworked 2026-10-15 (the owner: "he stutters moves and raises his arm" on the
# old hand-rolled loop). The live loop (_dance_worker) now plays the robot's
# OWN anim_dancebeat_* clips (via _resolve_dance_anim_pool) instead of manual
# head/lift/turn commands -- no more raw lift commands at all; a clip only
# moves the lift if it was authored to. DANCE_HEAD_NOD_ANGLES_DEG/
# DANCE_TURN_DEG survive only as the ultra-fallback used if the robot
# somehow reports zero anim_dancebeat_* clips (firmware oddity) -- see
# _dance_worker. Angles kept well inside the SDK's clamped safe range
# (MIN/MAX_HEAD_ANGLE -22..45 deg); the fallback turn is a small in-place
# pivot only -- no driving forward, safe on a desk.
DANCE_HEAD_NOD_ANGLES_DEG = (-10.0, 30.0)
DANCE_TURN_DEG = 20.0

# Clip-name substrings that mark an anim_dancebeat_* entry as NOT a loopable
# beat move (bookends, behavior-state/filler clips) -- see
# _resolve_dance_anim_pool.
DANCEBEAT_EXCLUDE_SUBSTR = (
    "getin", "getout", "getready", "idle", "listening", "quit",
    "cantdothat", "eyehold",
)

# Hardcoded fallback (added 2026-10-15 after a live miss: "dance party
# dancing queen" -- ListAnimations hit DEADLINE_EXCEEDED while resolving
# the anim pool, anim_list came back empty, and the dance loop silently
# fell through to the ultra-fallback head-nod-only move; the owner saw
# "normal head movement", no actual dancing). These are the exact 30
# anim_dancebeat_* clip names fetched directly from this robot earlier the
# same day (silent, read-only ListAnimations probe, nothing played).
# _resolve_dance_anim_pool always starts from this set now, so dancing no
# longer depends on a live ListAnimations call succeeding at all -- see
# its docstring, and _dance_worker's use of protocol.Animation(name=...)
# to play a move without needing anim_list populated either.
KNOWN_DANCEBEAT_ANIMS = (
    "anim_dancebeat_cantdothat_01",
    "anim_dancebeat_eyebeat_01",
    "anim_dancebeat_eyebeat_left_01",
    "anim_dancebeat_eyebeat_right_01",
    "anim_dancebeat_eyehold_01",
    "anim_dancebeat_getin_01",
    "anim_dancebeat_getin_02",
    "anim_dancebeat_getout_01",
    "anim_dancebeat_getout_02",
    "anim_dancebeat_getready_01",
    "anim_dancebeat_headlift_01",
    "anim_dancebeat_headliftbody_back_01",
    "anim_dancebeat_headliftbody_fwd_01",
    "anim_dancebeat_headliftbody_left_large_01",
    "anim_dancebeat_headliftbody_left_med_01",
    "anim_dancebeat_headliftbody_left_small_01",
    "anim_dancebeat_headliftbody_right_large_01",
    "anim_dancebeat_headliftbody_right_med_01",
    "anim_dancebeat_headliftbody_right_small_01",
    "anim_dancebeat_headnod_01",
    "anim_dancebeat_headnod_down_01",
    "anim_dancebeat_idle_01",
    "anim_dancebeat_listening_01",
    "anim_dancebeat_listening_02",
    "anim_dancebeat_listening_03",
    "anim_dancebeat_pivot_left_01",
    "anim_dancebeat_pivot_right_01",
    "anim_dancebeat_quit_01",
    "anim_dancebeat_scoot_left_01",
    "anim_dancebeat_scoot_right_01",
)

# Cap on how long _resolve_dance_anim_pool will wait for a LIVE anim_list
# fetch (well under the SDK's own internal 10s ListAnimations RPC timeout)
# before giving up and using KNOWN_DANCEBEAT_ANIMS only. Only ever waited
# out while the connection is idle, before the stream starts.
ANIM_LIST_FETCH_TIMEOUT_S = 3.0

# BPM used when estimate_bpm() can't get a confident read (too short/quiet/
# degenerate signal) -- a reasonable mid-tempo-pop default.
DEFAULT_BPM = 115.0

# "dance party" / bare "dance" with no song named -> pick one of these.
DANCE_PARTY_FALLBACK_SONGS = [
    "uptown funk bruno mars",
    "walking on sunshine katrina and the waves",
    "happy pharrell williams",
    "cant stop the feeling justin timberlake",
    "i gotta feeling black eyed peas",
]
# Phrases that mean "just dance, no specific song" once trigger-prefixes and
# a trailing "and dance"/"dance party"/"dance" suffix have been stripped.
DANCE_ONLY_PHRASES = {
    "", "dance", "dance party", "lets dance", "let's dance",
    "time to dance", "dance to the music", "dance to music",
    # "and dance" (added 2026-10-15): the post-strip_trigger() leftover of
    # "play and dance" ("play" is a TRIGGER_PREFIXES entry, so it's already
    # gone by the time resolve_dance_and_query() sees this) -- without this,
    # the generic " dance" trailing-suffix strip below turns it into the
    # bogus one-word query "and". Caught here, before that stripping runs.
    "and dance",
}

# Leading "dance to <song>" / "let's dance to <song>" phrasings (added
# 2026-10-15 after a live miss: Vosk emits the apostrophe'd "let's dance to
# dancing queen by abba", which dance_music_custom's utterance list now
# matches -- see customIntents.json -- but with no "play ..." prefix for
# strip_trigger() to peel off, resolve_dance_and_query() has to strip these
# itself to find the song name. Longest-first so a shorter prefix never
# steals part of a longer one that was actually present (not that it
# matters for str.startswith -- these don't share a common leading
# substring -- but kept consistent with TRIGGER_PREFIXES' convention).
DANCE_LEADING_PREFIXES = [
    "let's dance to",
    "lets dance to",
    "dance with me to",
    "play and dance to",
    "dance to",
    "play and dance",
]

# Leading trigger words to strip so we're left with the actual song/artist query.
TRIGGER_PREFIXES = [
    "play me a song",
    "play me the song",
    "play me some music",
    "play me",
    "play the song",
    "play a song",
    "play some music",
    "play some",
    "play song",
    "play music",
    "play",
]

sys.path.insert(0, VECTOR_MUSIC_DIR)
from ytm_common import (  # noqa: E402
    MY_MUSIC_QUEUE_LIMIT,
    classify_my_music_request,
    resolve_my_music_ids,
)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger("vector-music")


def strip_trigger(text: str) -> str:
    t = text.strip().lower()
    # try longest prefixes first (already ordered longest->shortest above)
    for prefix in TRIGGER_PREFIXES:
        if t.startswith(prefix):
            rest = t[len(prefix):].strip()
            # common connector leftovers
            for lead in ("some ", "the song ", "the ", "of "):
                if rest.startswith(lead):
                    rest = rest[len(lead):].strip()
            return rest
    return t


def resolve_dance_and_query(raw_query: str, query: str) -> tuple[str, bool]:
    """Decide whether dance mode should be enabled from the raw utterance,
    and return the query text to actually search for.

    `raw_query` is the full speechText wire-pod passed us; `query` is that
    text after `strip_trigger()` already peeled off a "play ..." prefix (or
    is unchanged if this came in via dance_music_custom, which has no such
    prefix). Handles three shapes:
      - "play the song X and dance"    -> dance=True,  query="X"
      - "dance party" / "let's dance"  -> dance=True,  query=<random pick>
      - anything else                  -> dance=False, query unchanged
    """
    raw_l = raw_query.strip().lower()
    if "dance" not in raw_l:
        return query, False

    q = query.strip()
    ql = q.lower()

    # Check for a dance-only phrase BEFORE suffix-stripping -- "let's dance"
    # must match here, not get mangled into "let's" by the " dance" suffix
    # strip below.
    if ql in DANCE_ONLY_PHRASES:
        return random.choice(DANCE_PARTY_FALLBACK_SONGS), True

    # Leading "dance to <song>" / "let's dance to <song>" -- dance_music_custom
    # passes the raw speechText with no "play ..." prefix, so without this,
    # the whole sentence (not just the song name) would become the search
    # query. Checked before the trailing-suffix block below since these are
    # leading, not trailing, phrasings -- see DANCE_LEADING_PREFIXES.
    for prefix in DANCE_LEADING_PREFIXES:
        if ql.startswith(prefix):
            remainder = q[len(prefix):].strip()
            return (remainder if remainder else random.choice(DANCE_PARTY_FALLBACK_SONGS)), True

    for suffix in (" and dance", " dance party", " and dance party", " dance"):
        if ql.endswith(suffix):
            q = q[: -len(suffix)].strip()
            ql = q.lower()
            break

    if not q or ql in DANCE_ONLY_PHRASES:
        q = random.choice(DANCE_PARTY_FALLBACK_SONGS)

    return q, True


def pid_path():
    try:
        d = os.path.dirname(PID_PATH)
        if os.access(d, os.W_OK):
            return PID_PATH
    except OSError:
        pass
    return PID_PATH_FALLBACK


def write_pid(pid: int):
    try:
        with open(pid_path(), "w") as f:
            f.write(str(pid))
    except OSError as e:
        log.warning("could not write pidfile: %s", e)


def download_audio(query: str, workdir: str, max_seconds: int) -> str | None:
    """Search + download best audio for `query` via yt-dlp. Returns path to
    downloaded source file, or None if nothing was found."""
    import yt_dlp

    out_template = os.path.join(workdir, "src.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "default_search": "ytsearch1",
        # keep downloads short: skip anything absurdly long (2x cap, sanity bound)
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration < {max(max_seconds * 3, 600)}"
        ),
    }
    search_query = f"ytsearch1:{query} audio"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=True)
        except yt_dlp.utils.DownloadError as e:
            log.error("yt-dlp download error for %r: %s", query, e)
            return None
    if not info:
        return None
    entries = info.get("entries") if "entries" in info else [info]
    if not entries or not entries[0]:
        return None
    # find whatever file yt-dlp actually wrote
    for fname in os.listdir(workdir):
        if fname.startswith("src."):
            return os.path.join(workdir, fname)
    return None


def download_audio_by_id(video_id: str, workdir: str, max_seconds: int) -> str | None:
    """Like download_audio, but for a known YouTube videoId (from ytmusicapi)
    instead of a text search -- used by the 'my music' queue path so we play
    exactly the track ytmusicapi resolved, not a re-search of its title."""
    import yt_dlp

    out_template = os.path.join(workdir, "src.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration < {max(max_seconds * 3, 600)}"
        ),
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            log.error("yt-dlp download error for videoId=%s: %s", video_id, e)
            return None
    if not info:
        return None
    for fname in os.listdir(workdir):
        if fname.startswith("src."):
            return os.path.join(workdir, fname)
    return None


def convert_to_wav(src_path: str, dst_path: str, max_seconds: int) -> bool:
    """ffmpeg-convert to mono 16-bit PCM WAV at SDK_TARGET_RATE, capped to
    max_seconds. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", str(SDK_TARGET_RATE),
        "-sample_fmt", "s16",
        "-af", "loudnorm=I=-12:LRA=7:TP=-1",
        "-f", "wav",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed: %s", result.stderr[-2000:])
        return False
    return True


def wav_info(path: str):
    with wave.open(path, "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sampwidth_bits": w.getsampwidth() * 8,
            "framerate": w.getframerate(),
            "duration_s": round(w.getnframes() / float(w.getframerate()), 2),
        }


def estimate_bpm(wav_path: str) -> float:
    """Lightweight BPM estimate from the already-downloaded local WAV, so
    dance pacing/anim choice can roughly match the song's tempo. numpy-only
    (no librosa/scipy/aubio in this venv, and the brief says not to add
    heavy deps unless already available) -- a classic onset-envelope
    autocorrelation: a short-time energy envelope, its rectified first
    difference as an onset-strength signal, then autocorrelate that
    envelope over the lag range for 60-180 BPM and take the strongest lag.
    This is a coarse "fast vs slow" read for picking which anim_dancebeat_*
    clips to favor (see _dance_worker) -- NOT a frame-accurate beat grid;
    none of Vector's stock anim clips expose a per-clip duration we could
    sync against anyway. Falls back to DEFAULT_BPM on anything too short,
    too quiet, or degenerate. Pure local-file analysis -- no robot I/O."""
    try:
        import numpy as np
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        if samples.size < rate * 2:  # need at least ~2s of audio
            return DEFAULT_BPM

        hop = max(1, int(rate * 0.05))  # ~50ms hops
        n_hops = samples.size // hop
        if n_hops < 20:
            return DEFAULT_BPM
        trimmed = samples[: n_hops * hop].reshape(n_hops, hop).astype(np.float64)
        env = np.sqrt(np.mean(trimmed ** 2, axis=1))

        onset = np.diff(env)
        onset[onset < 0] = 0.0
        if onset.std() < 1e-6:
            return DEFAULT_BPM
        onset = onset - onset.mean()

        hop_s = hop / rate
        min_lag = int(round(60.0 / 180.0 / hop_s))  # 180 BPM upper bound
        max_lag = int(round(60.0 / 60.0 / hop_s))   # 60 BPM lower bound
        max_lag = min(max_lag, onset.size - 1)
        if min_lag < 1 or max_lag <= min_lag:
            return DEFAULT_BPM

        best_lag, best_score = None, -1.0
        for lag in range(min_lag, max_lag + 1):
            score = float(np.dot(onset[:-lag], onset[lag:]))
            if score > best_score:
                best_score, best_lag = score, lag
        if best_lag is None or best_score <= 0:
            return DEFAULT_BPM

        bpm = 60.0 / (best_lag * hop_s)
        # fold into a sane dance-tempo range (half/double as needed --
        # autocorrelation on a short clip often locks onto a sub-harmonic)
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        return float(bpm)
    except Exception as e:  # noqa: BLE001
        log.warning("bpm estimate failed: %s -- using default %.0f", e, DEFAULT_BPM)
        return DEFAULT_BPM


def say_text(message: str):
    """Best-effort robot TTS, used for the 'no result' case."""
    try:
        import anki_vector
        serial = os.environ.get("ROBOT_SERIAL", "").strip()
        with anki_vector.Robot(serial=serial or None) as robot:
            robot.behavior.say_text(message)
    except Exception as e:
        log.error("say_text failed: %s", e)


async def _stream_bypassing_control(robot, filename, volume):
    """Mirrors AudioComponent.stream_wav_file's body (anki_vector/audio.py)
    but skips the @on_connection_thread(requires_control=True) CLIENT-SIDE
    gate, so it can run over a connection made with
    behavior_control_level=None (nocontrol) or RESERVE_CONTROL (reserve) --
    i.e. without ever taking DEFAULT_PRIORITY behavior control, which is what
    freezes Vector's own explore/idle behaviors. Must be run via
    robot.conn.run_coroutine(...) so the asyncio.Event()s below are created
    on the connection thread's own event loop (they are not thread-safe)."""
    import asyncio
    from anki_vector.exceptions import VectorExternalAudioPlaybackException

    audio = robot.audio
    if audio._is_active_event is None:
        audio._is_active_event = asyncio.Event()
    if audio._is_active_event.is_set():
        raise VectorExternalAudioPlaybackException("Cannot start audio when another sound is playing")
    if volume < 0 or volume > 100:
        raise VectorExternalAudioPlaybackException("Volume must be between 0 and 100")

    reader, params = audio._open_file(filename)
    playback_error = None
    audio._is_active_event.set()
    if audio._done_event is None:
        audio._done_event = asyncio.Event()

    try:
        async for response in audio.grpc_interface.ExternalAudioStreamPlayback(
            audio._request_handler(reader, params, volume)
        ):
            response_type = response.WhichOneof("audio_response_type")
            if response_type == "audio_stream_playback_complete":
                playback_error = None
            elif response_type in ("audio_stream_buffer_overrun", "audio_stream_playback_failyer"):
                playback_error = response_type
            audio._done_event.set()
    except (asyncio.CancelledError,):
        log.debug("bypass audio stream future was cancelled")
    finally:
        audio._is_active_event = None
        audio._done_event = None

    return playback_error


def _stream_no_control(wav_path: str, volume: int, control_level, serial: str):
    """Connect with the given behavior_control_level (None for nocontrol, or
    ControlPriorityLevel.RESERVE_CONTROL for reserve) and stream audio via
    the control-bypassing path above."""
    import anki_vector

    with anki_vector.Robot(serial=serial or None, behavior_control_level=control_level) as robot:
        fut = robot.conn.run_coroutine(_stream_bypassing_control(robot, wav_path, volume))
        err = fut.result()
        log.info("bypass-control stream finished, playback_error=%s", err)
        if err is not None:
            raise RuntimeError(f"audio playback error: {err}")


def _stream_wander(wav_path: str, volume: int, max_seconds: int, serial: str):
    """Take full DEFAULT_PRIORITY control (this freezes Vector's own
    behaviors, same as `control` mode) but drive our own roam loop on top of
    it while streaming audio in a background thread, so he still moves
    around instead of standing still. Fallback if nocontrol/reserve fail to
    keep him moving."""
    import random
    import threading

    import anki_vector
    from anki_vector.util import degrees

    with anki_vector.Robot(serial=serial or None) as robot:
        if robot.status.is_on_charger:
            raise RuntimeError("wander mode refused: robot is on charger")

        stream_exc = {}

        def _play():
            try:
                robot.audio.stream_wav_file(wav_path, volume)
            except Exception as e:  # noqa: BLE001
                stream_exc["e"] = e

        player = threading.Thread(target=_play, daemon=True)
        player.start()

        deadline = time.time() + max_seconds
        try:
            while time.time() < deadline and player.is_alive():
                if robot.status.is_picked_up or robot.status.is_cliff_detected:
                    robot.motors.set_wheel_motors(0, 0)
                    time.sleep(0.3)
                    continue

                dist_mm = None
                try:
                    reading = robot.proximity.last_sensor_reading
                    if reading is not None:
                        dist_mm = reading.distance.distance_mm
                except Exception:  # noqa: BLE001
                    dist_mm = None

                if dist_mm is not None and dist_mm < 120:
                    robot.motors.set_wheel_motors(0, 0)
                    robot.motors.set_wheel_motors(-60, -60)
                    time.sleep(0.6)
                    robot.motors.set_wheel_motors(0, 0)
                    turn_deg = random.uniform(60, 150) * random.choice((1, -1))
                    robot.behavior.turn_in_place(degrees(turn_deg))
                else:
                    robot.motors.set_wheel_motors(60, 60)
                time.sleep(0.2)
        finally:
            robot.motors.set_wheel_motors(0, 0)

        player.join(timeout=5)
        if "e" in stream_exc:
            raise stream_exc["e"]


def _dance_bookend(robot, trigger: str):
    """Play a DanceBeat* anim TRIGGER synchronously. MUST only be called
    while the connection is otherwise idle (before the audio stream starts,
    or after it has finished) -- see 2026-09-28 live-robot test note below.
    """
    try:
        robot.anim.play_animation_trigger(trigger)
    except Exception as e:  # noqa: BLE001
        log.warning("dance: %s failed: %s", trigger, e)


def _resolve_dance_anim_pool(robot):
    """Build the quick/big anim_dancebeat_* move pools. Always starts from
    KNOWN_DANCEBEAT_ANIMS (see its comment -- a hardcoded, previously
    live-verified clip list) so dancing never depends on ListAnimations
    succeeding. Opportunistically also tries a real robot.anim.anim_list
    fetch, capped at ANIM_LIST_FETCH_TIMEOUT_S (well under the SDK's own
    10s internal RPC timeout) in case this robot's firmware has clips
    beyond the hardcoded set -- on timeout/failure that's just skipped,
    not fatal, and critically we never touch the `robot.anim.anim_list`
    property itself in the failure path (that property's own fallback
    would try ANOTHER blocking, no-timeout load if the dict is still
    empty -- exactly the DEADLINE_EXCEEDED hang this rework exists to
    avoid). Only ever called while the connection is idle, before the
    stream starts -- same rule as _dance_bookend. Returns (quick_pool,
    big_pool) of plain anim-name strings; _dance_worker wraps each into a
    protocol.Animation object itself so play_animation() never needs
    anim_list populated to actually play a move (see KNOWN_DANCEBEAT_ANIMS
    note) -- this pool is just which names exist, not how they get played."""
    names = set(KNOWN_DANCEBEAT_ANIMS)
    try:
        fut = robot.anim.load_animation_list()
        if hasattr(fut, "result"):
            fut.result(timeout=ANIM_LIST_FETCH_TIMEOUT_S)
        discovered = [n for n in robot.anim.anim_list if n.startswith("anim_dancebeat_")]
        names |= set(discovered)
        log.info("dance: live anim_list fetch ok, %d anim_dancebeat_* clips seen", len(discovered))
    except Exception as e:  # noqa: BLE001
        log.warning(
            "dance: live anim_list fetch failed/timed out (%s) -- using "
            "hardcoded clip list only", e,
        )
    loopable = [n for n in names if not any(x in n for x in DANCEBEAT_EXCLUDE_SUBSTR)]
    quick = [n for n in loopable if "eyebeat" in n or "headnod" in n]
    big = [n for n in loopable if n not in quick]
    log.info(
        "dance: anim pool quick=%d big=%d (%d total candidate clips)",
        len(quick), len(big), len(names),
    )
    return quick, big


def _dance_worker(robot, stop_event: "threading.Event", quick_pool, big_pool, bpm: float):
    """Runs on a background daemon thread alongside the main thread's
    blocking robot.audio.stream_wav_file() call (see DANCE MODE docstring
    at the top of this file for why concurrent SDK calls generally
    interleave safely on the one shared connection event loop).

    Reworked 2026-10-15 (the owner: "he stutters moves and raises his arm").
    The old version here called set_head_angle/set_lift_height/
    turn_in_place directly with a fixed DANCE_MOVE_CADENCE_S sleep between
    moves -- the sleep IS the stutter (a dead gap between each move with
    nothing happening), and the lift-bob move was literally the "raises
    his arm" complaint. This version instead plays the robot's own stock
    anim_dancebeat_* clips back-to-back with robot.anim.play_animation():
    that call already blocks until the clip finishes playing (same
    blocking-RPC pattern _dance_bookend already relies on), so simply not
    sleeping between calls IS "back to back, no gaps" -- one clip ending
    is the very next call starting.

    quick_pool/big_pool come from _resolve_dance_anim_pool, resolved once
    while the connection was idle, before the stream started -- see
    stream_to_robot. NO anim-list/anim-trigger-list metadata lookup happens
    in this loop (both are already cached): the 2026-09-28 test note below
    is why that matters.

    2026-09-28 live-robot test note: the first version of this loop called
    robot.anim.play_animation_trigger(...) directly here, concurrently with
    the audio stream, with NEITHER list pre-loaded. The trigger needed a
    ListAnimations metadata lookup the first time (to resolve trigger name
    -> anim name), and that lookup got starved by the audio stream's own
    steady traffic on the same connection until it hit DEADLINE_EXCEEDED --
    and the whole connection then wedged: every later call failed with
    "connection has been closed", and robot.audio.stream_wav_file() itself
    never returned, hanging the worker until it was SIGTERM'd by hand via
    stop.sh. Pre-loading anim_list/anim_trigger_list while idle (see
    stream_to_robot) avoids that exact failure mode for play_animation()
    calls in THIS loop; it does not relax the general rule in this file
    about not calling audio-path methods (e.g. set_master_volume) mid-
    stream -- that is a separate, also-confirmed-live way to kill the
    stream (see _duck_watcher's 2026-10-15 note).

    quick_pool is favored over big_pool roughly in proportion to tempo
    (faster songs -> more of the small eyebeat/headnod clips; slower songs
    -> more of the head+lift+body clips) -- a coarse heuristic, not a real
    beat grid (see estimate_bpm). Falls back to a single safe head-nod-only
    move (no lift, no turn) if the robot reports zero usable
    anim_dancebeat_* clips at all. Lift is reset to neutral by the caller
    before this loop starts and after it exits (see stream_to_robot), not
    here -- this loop itself never issues a raw lift command."""
    from anki_vector.util import degrees
    from anki_vector.messaging import protocol

    quick_weight = 0.75 if bpm >= 110 else (0.3 if bpm < 90 else 0.5)
    pool = quick_pool + big_pool

    def _pick():
        if quick_pool and big_pool:
            return random.choice(quick_pool) if random.random() < quick_weight else random.choice(big_pool)
        if pool:
            return random.choice(pool)
        return None

    while not stop_event.is_set():
        name = _pick()
        try:
            if name:
                # Pass a protocol.Animation object, NOT a plain string --
                # play_animation()'s `if not isinstance(anim,
                # protocol.Animation): ... _ensure_loaded() ...` check (see
                # anki_vector/animation.py) only does the anim_list dict
                # lookup for the string form. A pre-built Animation object
                # skips that branch entirely, so this call never depends on
                # anim_list being populated (added 2026-10-15 alongside
                # KNOWN_DANCEBEAT_ANIMS -- same live-miss fix).
                robot.anim.play_animation(protocol.Animation(name=name))
                log.info("dance move %s ok", name)
            else:
                # ultra-fallback: no anim_dancebeat_* clips at all -- a
                # single safe head move, no lift, no turn, then a short
                # wait so this doesn't spin-loop
                robot.behavior.set_head_angle(degrees(random.choice(DANCE_HEAD_NOD_ANGLES_DEG)))
                log.info("dance move (fallback head-nod) ok")
                stop_event.wait(0.6)
        except Exception as e:  # noqa: BLE001
            log.warning("dance move %s failed: %s", name, e)
            # brief pause so a persistent failure doesn't spin-loop RPCs
            stop_event.wait(0.3)


def _stream_watchdog(max_seconds: int, stream_done: "threading.Event",
                      watchdog_stop: "threading.Event", grace_s: int = 20):
    """Safety net for dance mode only (see 2026-09-28 test note above): if
    robot.audio.stream_wav_file() hasn't returned within max_seconds +
    grace_s -- e.g. a wedged connection from a concurrent RPC gone bad --
    force the whole worker process to exit rather than leave Vector
    silently stuck until a human runs stop.sh by hand. os._exit() drops the
    connection at the socket level, which the robot treats as a disconnect
    and releases behavior control on its own."""
    deadline = time.time() + max_seconds + grace_s
    while time.time() < deadline:
        if stream_done.is_set() or watchdog_stop.is_set():
            return
        time.sleep(1.0)
    if not stream_done.is_set():
        log.error(
            "dance watchdog: stream_wav_file did not return within %ss, force-exiting worker",
            max_seconds + grace_s,
        )
        os._exit(1)


def _dance_flag_watcher(robot, dance_state: dict, watcher_stop: "threading.Event"):
    """Polls DANCE_FLAG_PATH about once a second for the duration of the
    stream. NOTHING currently sets this flag (see DANCE MODE docstring --
    wiring a bare "dance" voice intent to it would shadow the built-in
    "Hey Vector, dance" behavior). It's here so dancing can be turned on
    mid-song by `touch /opt/vector-music/dance.flag` (manual test) or a
    future, more careful trigger, without restarting the song. No
    DanceBeatGetIn bookend runs for this path -- the connection is busy
    streaming at exactly the moment the flag fires, and that's the one
    situation _dance_bookend must not be called in (see 2026-09-28 test
    note on _dance_worker); it goes straight into the anim-clip loop
    (quick_pool/big_pool/bpm were already resolved at song start, before
    the stream began -- see dance_state in stream_to_robot)."""
    while not watcher_stop.is_set():
        try:
            if os.path.exists(DANCE_FLAG_PATH) and not dance_state.get("active"):
                os.remove(DANCE_FLAG_PATH)
                dance_state["active"] = True
                d_stop = threading.Event()
                dance_state["stop_event"] = d_stop
                t = threading.Thread(
                    target=_dance_worker,
                    args=(
                        robot, d_stop,
                        dance_state.get("quick_pool", []),
                        dance_state.get("big_pool", []),
                        dance_state.get("bpm", DEFAULT_BPM),
                    ),
                    daemon=True,
                )
                t.start()
                dance_state["thread"] = t
                log.info("dance: started via flag file mid-song")
        except OSError as e:
            log.warning("dance flag watcher: %s", e)
        watcher_stop.wait(1.0)


def _duck_watcher(robot, stop_event: "threading.Event"):
    """Polls DUCK_FLAG_PATH (touched/removed by duck_watcher.sh, a host-side
    script that tails wire-pod's docker logs for "Stream type: OPUS" /
    "Transcribed text:") and drops/restores Vector's MASTER volume so the
    robot's own mic can hear the owner over the music during a voice capture.
    Added 2026-10-15 after a live dance-party session where the owner's replies
    kept transcribing as a single garbled "the" -- the song was drowning his
    voice. robot.audio.set_master_volume is a separate control plane from
    the per-stream `volume` passed to stream_wav_file, and (like the dance
    moves in _dance_worker) can be called from this thread while
    stream_wav_file blocks the main thread -- same event-loop-interleaving
    trick documented at the top of this file. Known simplification: restores
    to a fixed MEDIUM_HIGH rather than whatever level was in effect before
    ducking -- the SDK exposes no get_master_volume call, and this is a
    single-user home robot that doesn't otherwise touch its own volume."""
    from anki_vector.audio import RobotVolumeLevel

    # DISABLED 2026-10-15 (live-tested, same session): calling
    # robot.audio.set_master_volume() WHILE robot.audio.stream_wav_file()
    # has an active ExternalAudioStreamPlayback in flight kills that stream
    # almost immediately. Confirmed on-robot: with this call live, the
    # robot's OWN event log showed `sdk.audiostream_ended` at ~500ms every
    # time duck.flag was set (vs. a full, uninterrupted stream without it,
    # e.g. the 76s "let's have a dance party" session earlier the same
    # day). Unlike the dance moves in _dance_worker (motion/animation --
    # a genuinely separate subsystem from audio), master volume appears to
    # sit on the SAME server-side path as the external audio stream, so
    # changing it server-side tears the stream down. Left this thread
    # running (poll + log only) so the host-side duck_watcher.sh plumbing
    # and DUCK_FLAG_PATH wiring are already in place; flip real ducking
    # back on here ONLY after a mechanism that doesn't touch master volume
    # mid-stream is found and verified live (e.g. lowering `volume` for
    # the NEXT song rather than live, or an actual pause/resume of the
    # stream rather than a volume change).
    ducked = False
    while not stop_event.is_set():
        try:
            flagged = os.path.exists(DUCK_FLAG_PATH)
            if flagged and not ducked:
                ducked = True
                log.info("duck: voice capture detected (ducking DISABLED -- see 2026-10-15 note above)")
            elif not flagged and ducked:
                ducked = False
                log.info("duck: voice capture ended")
        except Exception as e:  # noqa: BLE001
            log.warning("duck watcher: %s", e)
        stop_event.wait(DUCK_POLL_S)


def stream_to_robot(wav_path: str, volume: int, mode: str, max_seconds: int, dance: bool = False):
    import anki_vector
    from anki_vector.util import degrees

    serial = os.environ.get("ROBOT_SERIAL", "").strip()

    if mode == "control":
        from anki_vector.audio import RobotVolumeLevel

        if not _SDK_LOCK.acquire(timeout=5):
            log.warning(
                "robot_control.lock held by another process (vector-curious/"
                "enroll.py/daycycle) -- skipping this play request rather "
                "than fighting it for behavior control"
            )
            return
        try:
            with anki_vector.Robot(serial=serial or None) as robot:
                # stale flag from a previous crashed run shouldn't auto-start dancing
                try:
                    os.remove(DANCE_FLAG_PATH)
                except OSError:
                    pass
                try:
                    os.remove(DUCK_FLAG_PATH)
                except OSError:
                    pass

                dance_state = {"active": dance}
                if dance:
                    # connection is idle here -- safe to resolve the anim
                    # pool and play the real anim clip (see 2026-09-28 test
                    # note on _dance_worker, and _resolve_dance_anim_pool)
                    quick_pool, big_pool = _resolve_dance_anim_pool(robot)
                    bpm = estimate_bpm(wav_path)
                    log.info("dance: estimated tempo ~%.0f BPM", bpm)
                    dance_state["quick_pool"] = quick_pool
                    dance_state["big_pool"] = big_pool
                    dance_state["bpm"] = bpm
                    _dance_bookend(robot, DANCE_GET_IN_TRIGGER)
                    # No raw lift commands in the loop itself (the owner:
                    # "raises his arm") -- start from a known-neutral lift
                    # so a clip that doesn't touch the lift track doesn't
                    # inherit a raised arm from a prior session. Idle here,
                    # same safety as the bookend call above.
                    try:
                        robot.behavior.set_lift_height(0.0)
                    except Exception as e:  # noqa: BLE001
                        log.warning("dance: lift reset before dancing failed: %s", e)
                    d_stop = threading.Event()
                    dance_state["stop_event"] = d_stop
                    dancer = threading.Thread(
                        target=_dance_worker,
                        args=(robot, d_stop, quick_pool, big_pool, bpm),
                        daemon=True,
                    )
                    dancer.start()
                    dance_state["thread"] = dancer
                    log.info("dance: started at song start")

                watcher_stop = threading.Event()
                watcher = threading.Thread(
                    target=_dance_flag_watcher, args=(robot, dance_state, watcher_stop), daemon=True
                )
                watcher.start()

                # Music ducking (added 2026-10-15) -- see _duck_watcher.
                duck_stop = threading.Event()
                duck_watcher_thread = threading.Thread(
                    target=_duck_watcher, args=(robot, duck_stop), daemon=True
                )
                duck_watcher_thread.start()

                # dance-only safety net: force-exit if the stream hangs (see
                # 2026-09-28 test note) instead of needing stop.sh by hand
                stream_done = threading.Event()
                watchdog_stop = threading.Event()
                if dance:
                    watchdog = threading.Thread(
                        target=_stream_watchdog, args=(max_seconds, stream_done, watchdog_stop), daemon=True
                    )
                    watchdog.start()

                try:
                    robot.audio.stream_wav_file(wav_path, volume)
                finally:
                    stream_done.set()
                    watchdog_stop.set()
                    watcher_stop.set()
                    duck_stop.set()
                    # belt-and-suspenders: never leave the robot stuck quiet
                    # if the host watcher's flag-clear raced with song end
                    try:
                        robot.audio.set_master_volume(RobotVolumeLevel.MEDIUM_HIGH)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        os.remove(DUCK_FLAG_PATH)
                    except OSError:
                        pass
                    d_stop = dance_state.get("stop_event")
                    if d_stop is not None:
                        d_stop.set()
                    dancer_thread = dance_state.get("thread")
                    if dancer_thread is not None:
                        dancer_thread.join(timeout=5)
                    watcher.join(timeout=2)
                    duck_watcher_thread.join(timeout=2)
                    if dance_state.get("active"):
                        # connection is idle again -- safe to reset pose + play
                        # the real anim clip
                        try:
                            robot.behavior.set_head_angle(degrees(0.0))
                            robot.behavior.set_lift_height(0.0)
                        except Exception as e:  # noqa: BLE001
                            log.warning("dance: pose reset failed: %s", e)
                        _dance_bookend(robot, DANCE_GET_OUT_TRIGGER)
        finally:
            _SDK_LOCK.release()
        return

    if mode == "nocontrol":
        _stream_no_control(wav_path, volume, None, serial)
        return

    if mode == "reserve":
        from anki_vector.connection import ControlPriorityLevel
        _stream_no_control(wav_path, volume, ControlPriorityLevel.RESERVE_CONTROL, serial)
        return

    if mode == "wander":
        _stream_wander(wav_path, volume, max_seconds, serial)
        return

    raise ValueError(f"unknown mode: {mode!r}")


def do_work_queue(video_ids, max_seconds, no_robot, volume, mode, label, dance=False):
    """Play a queue of known videoIds sequentially through the same
    download -> convert -> stream pipeline as a single song. One worker
    process handles the whole queue, so stop.sh (which SIGTERMs this
    process) kills the entire queue, not just the current track.

    dance=True re-enters dance stance (DanceBeatGetIn/GetOut) once per
    track, since each track is its own stream_to_robot() call/connection --
    a little choppier between songs than a single continuous dance, but
    correct and simple; the "my music" queue path isn't this feature's
    primary use case."""
    log.info("my-music queue start label=%r count=%d", label, len(video_ids))
    played = 0
    for i, vid in enumerate(video_ids):
        with tempfile.TemporaryDirectory(prefix="vector-music-q-") as workdir:
            src = download_audio_by_id(vid, workdir, max_seconds)
            if not src:
                log.warning("queue[%d/%d]: no source for videoId=%s, skipping", i + 1, len(video_ids), vid)
                continue
            wav_path = os.path.join(workdir, "out.wav")
            if not convert_to_wav(src, wav_path, max_seconds):
                log.warning("queue[%d/%d]: convert failed for videoId=%s, skipping", i + 1, len(video_ids), vid)
                continue
            info = wav_info(wav_path)
            log.info("queue[%d/%d] videoId=%s wav info: %s", i + 1, len(video_ids), vid, info)
            if no_robot:
                keep_path = f"{VECTOR_MUSIC_DIR}/last_offline_test.wav"
                subprocess.run(["cp", wav_path, keep_path], check=False)
                played += 1
                continue
            try:
                stream_to_robot(wav_path, volume, mode, max_seconds, dance=dance)
            except Exception as e:  # noqa: BLE001
                log.error("queue[%d/%d]: stream failed for videoId=%s: %s", i + 1, len(video_ids), vid, e)
                continue
            played += 1
    log.info("my-music queue done label=%r played=%d/%d", label, played, len(video_ids))
    if played == 0:
        if not no_robot:
            say_text("Couldn't play that.")
        print("NO_RESULT")
        return 1
    print(f"OK queue played={played}/{len(video_ids)}")
    return 0


def do_work(query: str, max_seconds: int, no_robot: bool, volume: int, mode: str,
            my_kind: str | None = None, my_name: str | None = None, dance: bool = False):
    log.info(
        "query=%r max_seconds=%s no_robot=%s mode=%s my_kind=%s my_name=%r dance=%s",
        query, max_seconds, no_robot, mode, my_kind, my_name, dance,
    )

    if my_kind:
        video_ids = resolve_my_music_ids(my_kind, my_name, limit=MY_MUSIC_QUEUE_LIMIT)
        if video_ids:
            label = f"{my_kind}:{my_name or ''}"
            return do_work_queue(video_ids, max_seconds, no_robot, volume, mode, label, dance=dance)
        log.warning(
            "my-music request kind=%s name=%r produced no tracks (no auth / no match / empty) "
            "-- falling back to public search for query=%r",
            my_kind, my_name, query,
        )
        # fall through to the normal single-song public-search path below,
        # using the already-stripped `query` text as-is.

    with tempfile.TemporaryDirectory(prefix="vector-music-") as workdir:
        src = download_audio(query, workdir, max_seconds)
        if not src:
            log.warning("no result for query=%r", query)
            if not no_robot:
                say_text("Couldn't find that.")
            print("NO_RESULT")
            return 1
        wav_path = os.path.join(workdir, "out.wav")
        if not convert_to_wav(src, wav_path, max_seconds):
            if not no_robot:
                say_text("Couldn't find that.")
            print("CONVERT_FAILED")
            return 1
        info = wav_info(wav_path)
        log.info("converted wav info: %s", info)
        if no_robot:
            print(
                f"OK duration={info['duration_s']}s rate={info['framerate']}Hz "
                f"bits={info['sampwidth_bits']} channels={info['channels']} "
                f"file={wav_path}"
            )
            # copy out of the tempdir so caller can inspect it after we exit
            keep_path = "/opt/vector-music/last_offline_test.wav"
            subprocess.run(["cp", wav_path, keep_path], check=False)
            print(f"saved={keep_path}")
            return 0
        stream_to_robot(wav_path, volume, mode, max_seconds, dance=dance)
        log.info("stream_wav_file complete for query=%r mode=%s dance=%s", query, mode, dance)
        return 0


def daemonize_and_run(query: str, max_seconds: int, volume: int, mode: str,
                       my_kind: str | None = None, my_name: str | None = None, dance: bool = False):
    """Double-fork so the parent (wire-pod's exec child) returns almost
    immediately, decoupling the actual download/stream from wire-pod's
    blocking Cmd.Run()."""
    args = [
        VENV_PY, os.path.abspath(__file__),
        "--worker",
        "--max-seconds", str(max_seconds),
        "--volume", str(volume),
        "--mode", mode,
    ]
    if my_kind:
        args += ["--my-kind", my_kind, "--my-name", my_name or ""]
    if dance:
        args.append("--dance")
    args.append(query)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from wire-pod's process group
        cwd=VECTOR_MUSIC_DIR,
    )
    write_pid(proc.pid)
    log.info("spawned worker pid=%s for query=%r mode=%s my_kind=%s my_name=%r dance=%s",
              proc.pid, query, mode, my_kind, my_name, dance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    ap.add_argument("--volume", type=int, default=DEFAULT_VOLUME)
    ap.add_argument("--mode", choices=VALID_MODES, default=DEFAULT_MODE,
                     help="behavior-control strategy while streaming (default: %(default)s)")
    ap.add_argument("--dance", action="store_true",
                     help="dance while streaming (mode=control only; also auto-detected from "
                          "the utterance at the top-level invocation -- see resolve_dance_and_query)")
    ap.add_argument("--my-kind", choices=("liked", "playlist", "mix"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--my-name", default="", help=argparse.SUPPRESS)
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    raw_query = " ".join(args.query).strip().strip('"')

    # Only re-classify at the top-level (non-worker) invocation -- the worker
    # invocation already gets --my-kind/--my-name passed through explicitly
    # so it doesn't need to re-parse wire-pod's speechText.
    my_kind, my_name = (args.my_kind, args.my_name or None)
    if not args.worker and my_kind is None:
        my_kind, my_name = classify_my_music_request(raw_query)

    query = strip_trigger(raw_query)
    if not query:
        query = raw_query  # nothing left after stripping trigger words

    # Same rule as my_kind/my_name above: only auto-detect "...and dance" /
    # "dance party" at the top-level invocation. The forked --worker
    # invocation gets --dance passed through explicitly if it was set here.
    dance = args.dance
    if not args.worker:
        query, auto_dance = resolve_dance_and_query(raw_query, query)
        dance = dance or auto_dance

    max_seconds = max(5, min(args.max_seconds, DEFAULT_MAX_SECONDS))
    volume = max(0, min(args.volume, 100))
    mode = args.mode

    if args.no_robot or args.worker:
        # foreground: either an offline test, or we ARE the detached worker
        write_pid(os.getpid()) if args.worker else None
        rc = do_work(query, max_seconds, no_robot=args.no_robot, volume=volume, mode=mode,
                     my_kind=my_kind, my_name=my_name, dance=dance)
        sys.exit(rc)
    else:
        # normal wire-pod invocation: detach immediately, exit fast so
        # wire-pod's blocking exec returns and the robot gets its ack intent
        daemonize_and_run(query, max_seconds, volume, mode, my_kind=my_kind, my_name=my_name, dance=dance)
        sys.exit(0)


if __name__ == "__main__":
    main()
