#!/usr/bin/env python3
"""
Vector's brain proxy.

Sits between wire-pod (on the GPU host) and Ollama (127.0.0.1:11434 on this host,
gpu-host) and injects Obsidian-vault context into every chat completion
call. OpenAI-compatible: implements /v1/chat/completions (streaming SSE and
non-streaming) and passes everything else straight through to Ollama.

Fails open: any error in retrieval/indexing forwards the original request
to Ollama unmodified rather than blocking Vector.
"""
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - not expected on this host's Python
    ZoneInfo = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import Index, static_block, retrieval_block, VAULT_ROOT  # noqa: E402
# brighten (2026-10-15): software auto-levels for Vector's underexposed camera
# frames (see brighten.py). Optional: needs Pillow+numpy (user site-packages);
# if unavailable, image turns are forwarded unmodified.
try:
    from brighten import brighten_data_uri  # noqa: E402
except Exception as _e:  # pragma: no cover
    brighten_data_uri = None
    _BRIGHTEN_IMPORT_ERR = "%s: %s" % (type(_e).__name__, _e)
else:
    _BRIGHTEN_IMPORT_ERR = None
BRIGHTEN_DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brighten-test")

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = int(os.environ.get("VECTOR_BRAIN_PORT", "11500"))
OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# Keep-alive fix (2026-09-29, live-10, Hermes): OLLAMA_KEEP_ALIVE=-1 is set
# at the systemd level (ollama.service.d/override.conf) but Ollama's
# OpenAI-compatible /v1/chat/completions endpoint does not honor that
# server-wide default per request -- confirmed via `ollama ps` showing a
# finite TTL ("9 minutes from now") instead of "Forever" for a model only
# ever loaded through this endpoint. Setting keep_alive explicitly on every
# request this proxy sends keeps qwen3:32b resident between turns (a cold
# load costs ~14-18s and reads as the robot being broken). "24h" rather
# than -1 here so a stuck/leaked proxy process can't pin the model forever
# by accident; the systemd -1 default still applies to anything that
# doesn't go through this proxy.
OLLAMA_KEEP_ALIVE_VALUE = os.environ.get("OLLAMA_KEEP_ALIVE_VALUE", "24h")
# Context-length fix (2026-09-29, ctx-vram, Hermes): qwen3:32b was
# resident at num_ctx=32768 (~29GB of the 32GB total across both GPUs),
# leaving no room for the vision model and forcing it onto CPU (~64s/turn).
# The real prompts here (persona + ~3000 tok RAG budget + <=8 turns of
# same-day memory + a short question) never approach 32768 tokens.
# IMPORTANT: this can't be fixed with a per-request options.num_ctx on this
# code path -- confirmed empirically that Ollama 0.34.2's OpenAI-compatible
# /v1/chat/completions endpoint silently IGNORES num_ctx (nested in
# "options" or top-level) and always reloads the model at the server-wide
# OLLAMA_CONTEXT_LENGTH default (32768) regardless of what's sent. That's
# also why classify.go (wire-pod's fast intent classifier, same endpoint,
# same qwen3:32b) could never be brought in line via a request-field change.
# Fix applied instead at the Ollama model level: the qwen3:32b tag itself
# was recreated with `PARAMETER num_ctx 8192` baked into its Modelfile
# (original preserved as qwen3:32b-orig32k-bak-20260929-ctx for rollback --
# `ollama cp qwen3:32b-orig32k-bak-20260929-ctx qwen3:32b` to revert). This
# makes EVERY caller of "qwen3:32b" -- this proxy, diary.py, curious.py,
# AND classify.go -- load at num_ctx=8192 (~22GB) with zero code changes
# and no risk of two callers fighting over context length. Same trick was
# applied to qwen2.5vl:3b (PARAMETER num_ctx 4096; original backed up as
# qwen2.5vl:3b-orig-bak-20260929-ctx).
LOG_PATH = os.environ.get("VECTOR_BRAIN_LOG_PATH", os.path.expanduser("~/vector-brain/brain.log"))
REINDEX_POLL_SECONDS = 60

# Curious Vector (gpu-host, vector-curious container) drops a short note
# here right after it asks a spoken question, so that if the owner answers,
# this proxy can tell the main LLM what was just asked. Fails open: missing
# or stale (>TTL) file means no-op, never blocks a normal chat completion.
RECENT_OBSERVATION_PATH = os.environ.get("VECTOR_RECENT_OBSERVATION_PATH", os.path.expanduser("~/vector-brain/recent_observation.txt"))
RECENT_OBSERVATION_TTL_SECONDS = 120

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("brain")

def _read_recent_observation():
    """Return the curious-vector observation note if it exists and is
    still fresh, else None. Fails open on any error."""
    try:
        mtime = os.path.getmtime(RECENT_OBSERVATION_PATH)
    except OSError:
        return None
    if time.time() - mtime > RECENT_OBSERVATION_TTL_SECONDS:
        return None
    try:
        with open(RECENT_OBSERVATION_PATH) as f:
            text = f.read().strip()
    except OSError:
        return None
    return text or None


# ---- Time-of-day / battery mood context (added 2026-09-26, day-cycle eyes
# + mood-aware LLM project). Backed up as server.py.bak-20260926-mood before
# this edit -- applied on top of curious-vector's recent_observation change,
# not over an older copy. Injects one system line with the local time
# bucket + cached battery level so the LLM's {{playAnimationWI||...}} choice
# and tone match reality (sleepy late at night / on low battery). Fails
# open everywhere: any error here must never block a chat completion.
TIMEZONE_NAME = os.environ.get("VECTOR_TIMEZONE", "America/New_York")
WIREPOD_SDK_BASE = os.environ.get("WIREPOD_URL", "http://127.0.0.1:8080")
ROBOT_SERIAL = os.environ.get("VECTOR_ESN", "YOUR_ESN")
BATTERY_CACHE_TTL_SECONDS = 120
BATTERY_FETCH_TIMEOUT_SECONDS = 3

_battery_cache = {"level": None, "ts": 0.0}
_battery_cache_lock = threading.Lock()


def _time_of_day_bucket(dt):
    h = dt.hour
    if h >= 22 or h < 5:
        return "late night"
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    return "evening"


def _get_local_time():
    """Returns (datetime, bucket_str). Fails open to naive local time if
    zoneinfo/tzdata isn't available on this host."""
    try:
        now = datetime.now(ZoneInfo(TIMEZONE_NAME)) if ZoneInfo is not None else datetime.now()
    except Exception:
        now = datetime.now()
    return now, _time_of_day_bucket(now)


def _get_battery_level():
    """Returns an int battery_level (1=low .. 3=full) or None. Cached up to
    BATTERY_CACHE_TTL_SECONDS; on fetch failure falls open to the last
    cached value (or None) rather than raising."""
    now = time.time()
    with _battery_cache_lock:
        cached, ts = _battery_cache["level"], _battery_cache["ts"]
        if cached is not None and (now - ts) < BATTERY_CACHE_TTL_SECONDS:
            return cached
    try:
        url = f"{WIREPOD_SDK_BASE}/api-sdk/get_battery?serial={ROBOT_SERIAL}"
        with urllib.request.urlopen(url, timeout=BATTERY_FETCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        level = data.get("battery_level")
        if isinstance(level, int):
            with _battery_cache_lock:
                _battery_cache["level"] = level
                _battery_cache["ts"] = now
            return level
    except Exception as e:
        log.debug("battery fetch failed (fail-open): %s", e)
    with _battery_cache_lock:
        return _battery_cache["level"]


# ---- Mood state (added 2026-09-29, Hermes, "make Vector feel more
# human", item 3). A small persisted JSON -- energy/social/last-interaction/
# last-played -- updated from real voice turns below (_note_voice_turn,
# called from _inject_context on every chat-completion request that has a
# real user message) and, best-effort, from curious-vector's day cycle
# (see curious.py's own write to this same file after a spoken Q&A). Feeds
# a ONE-LINE mood summary that REPLACES the old bespoke "sleepy" clause in
# _mood_context_line below (split from the time/battery facts as part of
# the 2026-09-29 speed pass, see _inject_context) -- the time/battery facts stay (still
# useful/true), but the editorializing about how Vector feels now comes
# from this unified mood model instead of just late-night/low-battery.
# Fails open everywhere: a missing/corrupt mood.json is treated as "never
# interacted", never raises.
MOOD_PATH = os.environ.get("VECTOR_MOOD_PATH", os.path.expanduser("~/vector-brain/mood.json"))
MOOD_PLAYED_RECENT_SECONDS = 900          # "the owner just played with you"
MOOD_LONELY_AFTER_SECONDS = 5 * 3600      # "nobody's talked to you in Xh"
MOOD_BORED_AFTER_SECONDS = 2 * 3600
_mood_lock = threading.Lock()

_DEFAULT_MOOD = {
    "energy": "steady",
    "social": "content",
    "last_interaction_ts": 0.0,
    "last_played_ts": 0.0,
}

# Reuse the same playful-command detector the escalation heuristic below
# already defines (dance/trick/explore/pet/etc.) to decide whether a turn
# counts as "played with", not just "talked to". Defined lazily via a
# forward reference since _COMMAND_KEYWORDS is declared further down this
# file; see _note_voice_turn.


def _load_mood():
    try:
        with open(MOOD_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(_DEFAULT_MOOD)
        merged = dict(_DEFAULT_MOOD)
        merged.update(data)
        return merged
    except Exception:
        return dict(_DEFAULT_MOOD)


def _save_mood(mood):
    try:
        os.makedirs(os.path.dirname(MOOD_PATH), exist_ok=True)
        tmp = MOOD_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(mood, f)
        os.replace(tmp, MOOD_PATH)
    except Exception as e:
        log.debug("mood save failed (fail-open): %s", e)


def _note_voice_turn(user_text):
    """Called once per real chat-completion turn (see _inject_context).
    Updates last_interaction_ts always, and last_played_ts when the turn
    looks like a playful robot-command (dance/trick/explore/pet/etc.).
    Never raises -- mood tracking must never block a chat completion."""
    try:
        text = (user_text or "").strip()
        if not text:
            return
        with _mood_lock:
            mood = _load_mood()
            now = time.time()
            mood["last_interaction_ts"] = now
            if _COMMAND_KEYWORDS.search(text):
                mood["last_played_ts"] = now
            _save_mood(mood)
    except Exception as e:
        log.debug("mood update failed (fail-open): %s", e)


def _mood_summary(now_ts, bucket, battery_level):
    """Returns (mood_line, is_sleepy) -- mood_line is the one-line
    tone-setting summary for the prompt; is_sleepy also gates the
    yawn-y-tone instruction, same as the old late-night/low-battery-only
    check did."""
    mood = _load_mood()
    last_interaction = mood.get("last_interaction_ts") or 0
    last_played = mood.get("last_played_ts") or 0
    is_late_night = bucket == "late night"
    is_low_battery = battery_level == 1

    if last_played and (now_ts - last_played) < MOOD_PLAYED_RECENT_SECONDS:
        desc = "happy and a little wound up -- the owner just played with you"
    elif is_late_night or is_low_battery:
        desc = "sleepy and winding down"
    elif not last_interaction:
        desc = "a bit lonely -- nobody's talked to you yet today"
    else:
        gap = now_ts - last_interaction
        if gap >= MOOD_LONELY_AFTER_SECONDS:
            hrs = int(gap // 3600)
            desc = f"a little lonely -- it's been about {hrs}h since anyone talked to you"
        elif gap >= MOOD_BORED_AFTER_SECONDS:
            desc = "a bit bored, could use some company"
        else:
            desc = "content and alert"

    is_sleepy = is_late_night or is_low_battery
    return f"You feel: {desc}.", is_sleepy


# Speed pass (2026-09-29, Hermes, "cut voice latency"): this used to be
# ONE function returning ONE line that mixed the exact clock time/battery
# (changes basically every request) with the mood summary (changes at
# most a few times a day). Mixed together, that whole line had to sit in
# the per-question dynamic tier, forcing the mood text to be reprocessed
# every turn even though it's almost always unchanged. Split in two so
# _inject_context can put _mood_context_line() in the SLOW tier (with
# same-day memory) and _time_battery_line() in the DYNAMIC tier (with RAG
# hits) -- see _inject_context's docstring for the full tier ordering.
def _mood_context_line():
    """Slow-changing tier: the mood summary + (when sleepy) the tone
    instruction. Returns None on any error (fail-open)."""
    try:
        now, bucket = _get_local_time()
        battery_level = _get_battery_level()
        mood_line, is_sleepy = _mood_summary(time.time(), bucket, battery_level)
        parts = [mood_line]
        if is_sleepy:
            parts.append(
                "Use a slower, yawn-y, winding-down tone (you may mention being sleepy). "
                "Do NOT add sad animations just because it's late -- only use sad when the "
                "content is actually sad."
            )
        return " ".join(parts)
    except Exception as e:
        log.debug("mood context build failed (fail-open): %s", e)
        return None


def _time_battery_line():
    """Dynamic tier: the exact clock time + battery level facts. Returns
    None on any error (fail-open)."""
    try:
        now, bucket = _get_local_time()
        battery_level = _get_battery_level()
        time_str = now.strftime("%I:%M %p").lstrip("0") or now.strftime("%I:%M %p")
        parts = [f"Current time: {now.strftime('%A, %B')} {now.day}, {now.year}, {time_str} ({TIMEZONE_NAME}), time-of-day: {bucket}."]
        if battery_level is not None:
            parts.append(f"Battery level: {battery_level}/3 (1=low, 3=full).")
        return " ".join(parts)
    except Exception as e:
        log.debug("time/battery line build failed (fail-open): %s", e)
        return None


# ---- Household faces (2026-10-15, hermes, vector-faces). vector-curious
# runs each daytime capture through the LOCAL face service (YuNet+SFace on
# this host) and writes CURRENT_PEOPLE_PATH = [{name, score, last_seen_epoch}].
# Names seen within CURRENT_PEOPLE_TTL_SECONDS go into the DYNAMIC tier;
# FACES_RULE is constant text appended to the STATIC tier (byte-identical
# every turn, so it doesn't hurt the Ollama prompt cache). Fails open.
CURRENT_PEOPLE_PATH = os.environ.get("VECTOR_CURRENT_PEOPLE_PATH", os.path.expanduser("~/vector-faces/current_people.json"))
CURRENT_PEOPLE_TTL_SECONDS = 120
FACES_RULE = (
    "Faces: you only know who you can see from a 'People you can currently see:' line "
    "(your own face recognition). Use those names naturally, without announcing that "
    "you recognized them. Never claim to see or recognize someone who isn't listed there."
)


def _current_people_line():
    try:
        with open(CURRENT_PEOPLE_PATH) as f:
            data = json.load(f)
        now = time.time()
        names = []
        for p in data if isinstance(data, list) else []:
            if not isinstance(p, dict):
                continue
            n = str(p.get("name") or "").strip()
            age = now - float(p.get("last_seen_epoch") or 0)
            if n and n.lower() != "unknown" and -5 <= age <= CURRENT_PEOPLE_TTL_SECONDS and n not in names:
                names.append(n)
        if not names:
            return None
        if len(names) == 1:
            joined = names[0]
        else:
            joined = ", ".join(names[:-1]) + " and " + names[-1]
        return "People you can currently see: %s." % joined
    except Exception:
        return None


# ---- Per-person notes (2026-10-15, hermes, vector-curious explore/greet).
# vector-curious keeps one note per household member under
# 20-agents/vector/people/<slug>.md ("## Who they are" / "## Things I've
# learned" / "## Questions I've asked"). When current_people.json has a
# fresh (same TTL as the people line) recognised person, a compact summary
# of that person's note (<= PEOPLE_NOTE_MAX_LINES bullets in total) goes
# into the DYNAMIC tier so the conversation can use it. LOCAL ONLY: when
# it's injected, the uncertainty escalation sends Hermes the bare question
# without the local draft (see _people_ctx_injected). Fails open.
PEOPLE_NOTES_DIR = os.path.join(VAULT_ROOT, "20-agents/vector/people")
PEOPLE_NOTE_MAX_LINES = 10
PEOPLE_NOTE_MAX_PEOPLE = 2


def _fresh_people_names():
    try:
        with open(CURRENT_PEOPLE_PATH) as f:
            data = json.load(f)
        now = time.time()
        names = []
        for p in data if isinstance(data, list) else []:
            if not isinstance(p, dict):
                continue
            n = str(p.get("name") or "").strip()
            age = now - float(p.get("last_seen_epoch") or 0)
            if n and n.lower() != "unknown" and -5 <= age <= CURRENT_PEOPLE_TTL_SECONDS and n not in names:
                names.append(n)
        return names
    except Exception:
        return []


def _person_note_path(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    path = os.path.join(PEOPLE_NOTES_DIR, slug + ".md")
    if slug and os.path.isfile(path):
        return path
    want = name.strip().lower()
    try:
        for fn in sorted(os.listdir(PEOPLE_NOTES_DIR)):
            if not fn.endswith(".md"):
                continue
            p = os.path.join(PEOPLE_NOTES_DIR, fn)
            with open(p, "r", errors="ignore") as f:
                head = f.read(2000)
            m = re.search(r"^aliases:\s*\[(.*?)\]\s*$", head, re.MULTILINE)
            aliases = [a.strip().strip("'\"").lower() for a in (m.group(1).split(",") if m else [])]
            m2 = re.search(r"^person:\s*(.+?)\s*$", head, re.MULTILINE)
            if m2:
                aliases.append(m2.group(1).strip().lower())
            if want in aliases:
                return p
    except OSError:
        pass
    return None


def _note_bullets(text, heading):
    i = text.find(heading)
    if i < 0:
        return []
    start = i + len(heading)
    m = re.search(r"\n## ", text[start:])
    end = start + m.start() if m else len(text)
    return [l.strip()[2:].strip() for l in text[start:end].splitlines() if l.strip().startswith("- ")]


def _people_notes_block():
    names = _fresh_people_names()[:PEOPLE_NOTE_MAX_PEOPLE]
    if not names:
        return None
    try:
        budget = PEOPLE_NOTE_MAX_LINES
        parts = []
        for name in names:
            if budget <= 0:
                break
            path = _person_note_path(name)
            if not path:
                continue
            with open(path, "r", errors="ignore") as f:
                text = f.read()
            who = _note_bullets(text, "## Who they are")[:3]
            facts = list(reversed(_note_bullets(text, "## Things I've learned")))  # newest first
            lines = (who + facts)[:budget]
            if not lines:
                continue
            budget -= len(lines)
            parts.append("What you know about %s (your own notes):\n%s"
                         % (name, "\n".join("- " + l for l in lines)))
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


# ---- Same-day conversational memory (added 2026-09-29, Hermes, item 1).
# Pulls the last few turns of TODAY's conversation from the same vault log
# vector_memory_logger.py already writes (conversations_YYYY-MM.md), so
# the owner gets continuity ("still tired from the gaming last night?")
# without a second logging system. The log has no per-line date, only
# HH:MM -- _todays_conversation_lines() detects the most recent day
# rollover by walking backward from the newest line and stopping the
# instant a walked-back timestamp is LATER than the one that followed it
# (that jump means we've crossed back over midnight into yesterday).
# Fails open everywhere: any read/parse error just means no same-day
# context gets added, never blocks a chat completion.
SAME_DAY_MEMORY_MAX_TURNS = 8
SAME_DAY_MEMORY_MAX_CHARS_PER_MSG = 220
# Speaker label the conversation logger writes for the owner's turns in the
# vault's daily log ("- HH:MM (NY time) **Owner:** ..."). Configurable so it
# can match whatever your own logger writes.
OWNER_LOG_LABEL = os.environ.get("VECTOR_OWNER_LOG_LABEL", "Owner")
_CONVERSATION_LINE_RE = re.compile(
    r"^-\s*(\d{2}:\d{2})\s*\(NY time\)\s*\*\*" + re.escape(OWNER_LOG_LABEL) + r":\*\*\s*(.*?)\s*"
    r"\u2192\s*\*\*Vector:\*\*\s*(.*)$"
)
_INTENT_SUFFIX_RE = re.compile(r"_\(intent:.*?\)_\s*$")


def _todays_conversation_lines(now):
    try:
        fname = f"conversations_{now.strftime('%Y-%m')}.md"
        path = os.path.join(VAULT_ROOT, "20-agents/vector", fname)
        with open(path, "r", errors="ignore") as f:
            text = f.read()
    except OSError:
        return []
    lines = [l for l in text.splitlines() if l.startswith("- ") and ("**%s:**" % OWNER_LOG_LABEL) in l]
    todays = []
    prev_minutes = now.hour * 60 + now.minute
    for line in reversed(lines):
        m = re.match(r"^-\s*(\d{2}):(\d{2})\s*\(NY time\)", line)
        if not m:
            continue
        mins = int(m.group(1)) * 60 + int(m.group(2))
        if mins > prev_minutes + 2:  # a later time than what follows it -> day rolled over
            break
        todays.append(line)
        prev_minutes = mins
        if len(todays) >= SAME_DAY_MEMORY_MAX_TURNS:
            break
    todays.reverse()
    return todays


def _compact_turn_line(raw_line):
    line = _INTENT_SUFFIX_RE.sub("", raw_line).strip()
    m = _CONVERSATION_LINE_RE.match(line)
    if not m:
        return None
    ts, owner_said, vector_said = m.groups()
    owner_said = owner_said.strip()[:SAME_DAY_MEMORY_MAX_CHARS_PER_MSG]
    if not owner_said:
        return None
    vector_said = vector_said.strip()
    if not vector_said or "no response captured" in vector_said.lower():
        return f'{ts} the owner said: "{owner_said}" (you did not get a reply in).'
    vector_said = vector_said[:SAME_DAY_MEMORY_MAX_CHARS_PER_MSG]
    return f'{ts} the owner said: "{owner_said}" -- you replied: "{vector_said}"'


def _today_conversation_context():
    """Returns the same-day-memory system-message text, or None. Fails
    open (see module docstring above)."""
    try:
        now, _ = _get_local_time()
        raw_lines = _todays_conversation_lines(now)
        compact = [c for c in (_compact_turn_line(l) for l in raw_lines) if c]
        if not compact:
            return None
        header = (
            "Earlier today, you and the owner already talked (most recent last -- "
            "use this for continuity, e.g. following up naturally; do not "
            "recite it back verbatim):"
        )
        return header + "\n" + "\n".join(compact)
    except Exception as e:
        log.debug("today-conversation context build failed (fail-open): %s", e)
        return None


# ---- Hermes escalation (added 2026-09-27 -- "his LLM is lacking"). When the
# local model is asked something it's likely to botch (current-events/factual
# lookups) or its own answer reads as an "I don't know", the proxy hands the
# question to a bigger remote assistant ("Hermes" here, via an HTTP
# ask receiver -- see VECTOR_ASK_URL) and speaks Hermes' answer instead. Fails open at every step: any
# error/timeout falls back to the local model's own answer, never to silence.
# Configurable via env vars so this can be tuned/disabled without a code
# change:
#   HERMES_ESCALATION_ENABLED           "1"/"0" (default "1")
#   HERMES_ASK_TIMEOUT_SECONDS          seconds to wait on Hermes (default 20)
#   HERMES_ESCALATION_MIN_INTERVAL_SECONDS  rate limit between calls (default 30)
#   HERMES_ESCALATION_EXTRA_KEYWORDS    comma-separated extra trigger words
VECTOR_ENV_PATH = os.environ.get("VECTOR_ENV_PATH", os.path.expanduser("~/vector-pod/data/vector-bridge/vector.env"))
HERMES_ASK_URL_DEFAULT = os.environ.get("VECTOR_ESCALATION_URL_DEFAULT", "http://ASSISTANT_HOST_IP:8791/ask")

HERMES_ESCALATION_ENABLED = os.environ.get("HERMES_ESCALATION_ENABLED", "1") != "0"
HERMES_ASK_TIMEOUT_SECONDS = float(os.environ.get("HERMES_ASK_TIMEOUT_SECONDS", "20"))
HERMES_ESCALATION_MIN_INTERVAL_SECONDS = float(
    os.environ.get("HERMES_ESCALATION_MIN_INTERVAL_SECONDS", "30")
)

_BASE_TOPIC_KEYWORDS = [
    "current", "latest", "news", "weather", "forecast", "score", "capital",
    "president", "prime minister", "population", "price", "stock", "define",
    "definition", "meaning of", "how many", "how much", "explain",
    "difference between", "recipe", "convert", "history of", "when did",
    "when was", "who is", "who was", "who was the", "what is the",
    "what are the", "what's the", "election", "release date", "exchange rate",
    # Collab-mode additions (2026-09-29, Hermes): sports results and
    # the owner's calendar/email/schedule are live/external-info categories
    # per the collab-mode spec -- always escalate, never answerable from
    # a static vault snapshot.
    "who won", "final score", "calendar", "email", "schedule", "meeting today",
    "appointment",
]
_extra_kw = [w.strip() for w in os.environ.get("HERMES_ESCALATION_EXTRA_KEYWORDS", "").split(",") if w.strip()]
_ESCALATION_TOPIC_KEYWORDS = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in (_BASE_TOPIC_KEYWORDS + _extra_kw)) + r")\b",
    re.IGNORECASE,
)
_ESCALATION_QUESTION_WORDS = re.compile(r"\b(who|what|when|where|why|how|which|whose)\b", re.IGNORECASE)

_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey( there)?|good (morning|afternoon|evening|night)|how are you|"
    r"thank(s| you)|i love you|bye|goodbye|what'?s your name|who are you\??\s*$|"
    r"sing (a|me a) song|tell me a joke|good (boy|robot)|you'?re (funny|silly|cute))",
    re.IGNORECASE,
)
_COMMAND_KEYWORDS = re.compile(
    r"\b(explore|charger|dock|dance|do a trick|change your eye|eye color|"
    r"volume up|volume down|turn (left|right|around)|take a (photo|picture|selfie)|"
    r"come here|follow me|wake up|go to sleep|freeze|stop moving|pet you|"
    r"what time is it|set a timer|set an alarm)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_PATTERNS = re.compile(
    r"\b(i don'?t know|i do not know|i'?m not sure|i am not sure|not certain|"
    r"no idea|can'?t answer|cannot answer|unable to answer|"
    r"don'?t have (that|this) information|not familiar with|beyond my knowledge|"
    r"i'?m unsure)\b",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r'[.!?](?:["\'\)\]]*)(?:\s|$)')
# Speed pass (2026-09-29, Hermes): lowered from 200 -- see the
# comment at the buffering loop in _relay_stream_with_uncertainty_check
# for why a long run-on first "sentence" (commas, no early '.'/'!'/'?')
# used to cost extra seconds of pure decode wait before anything
# reached wire-pod. Still generous for a real short sentence.
FIRST_SENTENCE_MAX_CHARS = 120

_escalation_lock = threading.Lock()
_last_escalation_ts = 0.0

# ---- Metrics (added 2026-09-29, Hermes, collab-mode item 4). Every real
# voice turn's outcome is appended to METRICS_PATH as one JSON line:
# {ts, q, route, latency_ms}. route is one of "command" (robot command,
# never considered for escalation), "local" (answered by the local model,
# whether or not it was ever an escalation candidate), "hermes" (Hermes
# answered), "hermes_timeout" (escalation was attempted but Hermes never
# replied in time, so a local/generic fallback spoke instead). Fails open
# everywhere -- a metrics write must never affect what the robot says.
METRICS_PATH = os.environ.get("VECTOR_METRICS_PATH", os.path.expanduser("~/vector-brain/metrics/turns.jsonl"))


def _log_turn_metrics(route, question, latency_ms, extra=None):
    q = (question or "").strip()
    if not q:
        return
    try:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "q": q[:300],
            "route": route,
            "latency_ms": int(latency_ms),
        }
        # Speed-pass instrumentation (2026-09-29, Hermes): optional
        # per-turn prefill/cache numbers pulled from Ollama's
        # stream_options.include_usage trailer -- see
        # _forward_and_capture_stream / _extract_sse_usage. Only written
        # when present so old records and turns that never streamed
        # (image turns, escalation fallbacks) stay exactly the old shape.
        if extra:
            for k in ("ttft_ms", "prompt_tokens", "cached_tokens", "completion_tokens"):
                v = extra.get(k)
                if v is not None:
                    rec[k] = v
        with open(METRICS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        log.debug("metrics log failed (fail-open): %s", e)


def _compute_metrics_summary():
    """Reads METRICS_PATH and returns counts for today (local TIMEZONE_NAME
    calendar day) and the trailing 7 days, plus % local for each window.
    Fails open to an empty/zeroed summary rather than raising -- this
    backs a GET endpoint, not the voice path, but the same fail-open
    discipline applies throughout this file."""
    try:
        local_now = datetime.now(ZoneInfo(TIMEZONE_NAME)) if ZoneInfo is not None else datetime.now()
    except Exception:
        local_now = datetime.now()
    today_str = local_now.strftime("%Y-%m-%d")
    seven_days_ago_epoch = time.time() - 7 * 86400

    counts_today, counts_7d = {}, {}
    total_today = total_7d = 0
    try:
        with open(METRICS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec_dt = datetime.fromisoformat(rec.get("ts", ""))
                except Exception:
                    continue
                route = rec.get("route", "unknown")
                try:
                    rec_local = rec_dt.astimezone(ZoneInfo(TIMEZONE_NAME)) if ZoneInfo is not None else rec_dt
                except Exception:
                    rec_local = rec_dt
                if rec_local.strftime("%Y-%m-%d") == today_str:
                    counts_today[route] = counts_today.get(route, 0) + 1
                    total_today += 1
                if rec_dt.timestamp() >= seven_days_ago_epoch:
                    counts_7d[route] = counts_7d.get(route, 0) + 1
                    total_7d += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        log.error("metrics summary read failed: %s", e)

    def pct_local(counts, total):
        return round(100.0 * counts.get("local", 0) / total, 1) if total else 0.0

    return {
        "today": {
            "date": today_str,
            "total": total_today,
            "by_route": counts_today,
            "pct_local": pct_local(counts_today, total_today),
        },
        "last_7_days": {
            "total": total_7d,
            "by_route": counts_7d,
            "pct_local": pct_local(counts_7d, total_7d),
        },
    }


# ---- Learn-back (added 2026-09-29, Hermes, collab-mode item 3). After a
# successful Hermes escalation, the Q/A is appended to the same
# lessons.jsonl the training pipeline already reads (training/lessons.jsonl,
# {"prompt", "ideal_answer"} schema -- matches the existing 152 entries) and
# to a new vault note (20-agents/vector/lessons/from-hermes.md), which the
# existing VECTOR_GLOB ("20-agents/vector/**/*.md") already picks up on the
# proxy's 60s reindex poll -- no rag.py change needed for RAG to find it.
# Time-sensitive answers (weather/news/scores/prices/elections) are
# deliberately NOT learned -- they'd be stale and wrong the next time the
# same question is asked, so the proxy would rather escalate again than
# parrot yesterday's score. Fails open everywhere: a learn-back failure
# must never affect the answer already sent to Vector.
LESSONS_JSONL_PATH = os.environ.get("VECTOR_LESSONS_PATH", os.path.expanduser("~/vector-brain/training/lessons.jsonl"))
VAULT_LESSONS_FROM_HERMES_PATH = os.path.join(VAULT_ROOT, "20-agents/vector/lessons/from-hermes.md")
_TIME_SENSITIVE_KEYWORDS = re.compile(
    r"\b(weather|forecast|score|who won|news|current|latest|price|stock|"
    r"exchange rate|election|today|tonight|tomorrow|right now)\b",
    re.IGNORECASE,
)
_learn_lock = threading.Lock()


def _is_time_sensitive(question):
    return bool(_TIME_SENSITIVE_KEYWORDS.search((question or "").strip()))


def _learn_from_hermes(question, answer):
    """Appends a Hermes-answered Q/A pair to lessons.jsonl + the vault note,
    deduped on the normalized question text, skipping time-sensitive Q/A.
    Never raises -- called after the response to Vector has already been
    sent, so any failure here is purely a missed learning opportunity, not
    a user-facing error."""
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return
    if _is_time_sensitive(q):
        log.info("learn-back skipped (time-sensitive): q=%r", q)
        return
    norm_q = q.lower().strip().rstrip("?.! ")
    try:
        with _learn_lock:
            existing_prompts = set()
            try:
                with open(LESSONS_JSONL_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        p = str(rec.get("prompt", "")).lower().strip().rstrip("?.! ")
                        if p:
                            existing_prompts.add(p)
            except FileNotFoundError:
                pass

            if norm_q in existing_prompts:
                log.info("learn-back skipped (dedup, lessons.jsonl already has it): q=%r", q)
                return

            os.makedirs(os.path.dirname(LESSONS_JSONL_PATH), exist_ok=True)
            with open(LESSONS_JSONL_PATH, "a") as f:
                f.write(json.dumps({"prompt": q, "ideal_answer": a}) + "\n")

            os.makedirs(os.path.dirname(VAULT_LESSONS_FROM_HERMES_PATH), exist_ok=True)
            existing_vault_text = ""
            try:
                with open(VAULT_LESSONS_FROM_HERMES_PATH) as f:
                    existing_vault_text = f.read()
            except FileNotFoundError:
                pass
            if not existing_vault_text:
                existing_vault_text = (
                    "# Lessons from Hermes\n\n"
                    "Q/A pairs learned from Hermes escalations (collab mode). "
                    "Non-time-sensitive only -- see server.py's _learn_from_hermes.\n\n"
                )
            if f"**Q:** {q}" not in existing_vault_text:
                ts = datetime.now(timezone.utc).isoformat()
                entry = f"## {ts}\n**Q:** {q}\n**A:** {a}\n\n"
                with open(VAULT_LESSONS_FROM_HERMES_PATH, "w") as f:
                    f.write(existing_vault_text + entry)
            log.info("learn-back saved: q=%r", q)
    except Exception as e:
        log.debug("learn-back failed (fail-open): %s", e)


def _load_env_file(path):
    """Tiny KEY=VALUE loader (same format as ask_hermes.py's load_env).
    Never logs values -- callers must not print the returned dict."""
    env = {}
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env


def classify_turn(question):
    """Collab-mode decision function (2026-09-29, Hermes item 1 "smarter
    escalation decision"). Three-way route classification used both for
    the escalation heuristic below AND for /metrics/summary bucketing:
      "command" -- robot command (explore/dance/dock/etc) -- never escalate
      "local"   -- chit-chat, greeting, or nothing marks it as needing
                   live/external info or encyclopedic lookup -- local model
                   answers, no escalation
      "hermes"  -- looks like live/external info (news/weather/sports/
                   calendar/email) or a real factual/knowledge question --
                   escalation candidate (still subject to the "known
                   locally" RAG-similarity skip and the rate limiter
                   upstream of this function).
    Order matters: command check first (an explore/dance command should
    never be miscategorized as a question), then chit-chat, then topic
    keywords, then a generic question-word+"?" fallback for anything else
    that reads like a real question. Pure regex/string ops -- no LLM call,
    so this costs microseconds and adds no latency to the request path."""
    q = (question or "").strip()
    if not q or len(q) < 4:
        return "local"
    if _COMMAND_KEYWORDS.search(q):
        return "command"
    if _CHITCHAT_PATTERNS.search(q):
        return "local"
    if _ESCALATION_TOPIC_KEYWORDS.search(q):
        return "hermes"
    if "?" in q and _ESCALATION_QUESTION_WORDS.search(q) and len(q) > 12:
        return "hermes"
    return "local"


def _is_escalation_candidate(question):
    """Heuristic: does this look like a factual/knowledge/current-events
    question the local model is likely to botch? Excludes chit-chat and
    robot-command phrasing so we only escalate real questions. Thin
    wrapper over classify_turn() -- kept as its own function since it's
    the call site used throughout the escalation path below."""
    return classify_turn(question) == "hermes"


def _reserve_escalation_slot():
    """Rate limiter: at most one Hermes escalation call per
    HERMES_ESCALATION_MIN_INTERVAL_SECONDS. Returns True iff this call
    claimed the slot (and therefore owns making the call)."""
    global _last_escalation_ts
    with _escalation_lock:
        now = time.time()
        if now - _last_escalation_ts < HERMES_ESCALATION_MIN_INTERVAL_SECONDS:
            return False
        _last_escalation_ts = now
        return True


def _ask_hermes(question, timeout, local_draft=None):
    """Calls the vector-ask receiver directly -- the same API
    ask_hermes.py (the wire-pod custom-intent handler) uses. Returns the
    reply string, or None on any failure/timeout/empty reply. Never logs
    the token.

    local_draft (collab-mode, 2026-09-29, Hermes item 5): whatever partial
    local-model text we have for this turn, if any, so the question
    injected into Hermes' session is unambiguous ("the owner asked Vector: X.
    Vector's local draft: Y.") rather than a bare string. None is a valid,
    expected value on the preemptive-escalation path, where the local
    fallback generation runs in parallel and hasn't produced anything yet
    by the time this call goes out -- Hermes is told "local draft: none"
    in that case rather than waiting on it and losing the parallelism."""
    env = _load_env_file(VECTOR_ENV_PATH)
    url = env.get("VECTOR_ASK_URL", HERMES_ASK_URL_DEFAULT)
    token = env.get("VECTOR_ASK_TOKEN", "")
    if not token:
        log.error("hermes escalation: VECTOR_ASK_TOKEN missing from %s", VECTOR_ENV_PATH)
        return None
    req_payload = {"question": question, "esn": ROBOT_SERIAL}
    if local_draft:
        req_payload["local_draft"] = local_draft[:500]
    body = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Vector-Ask-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        reply = (data.get("reply") or "").strip()
        return reply or None
    except Exception as e:
        log.warning("hermes escalation: ask receiver failed/timed out: %s", e)
        return None


def _call_ollama_nonstream(payload, timeout=60):
    """Runs the same chat-completion request against Ollama directly,
    non-streaming, for use as a fast local fallback while Hermes is being
    asked. Returns the answer text, or None on failure."""
    p = dict(payload)
    p["stream"] = False
    # Keep-alive fix (2026-09-29, live-10, Hermes) -- see the matching
    # comment in _handle_chat_completions for why this is needed per-request.
    p.setdefault("keep_alive", OLLAMA_KEEP_ALIVE_VALUE)
    body = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_BASE + "/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.error("hermes escalation: local fallback generation failed: %s", e)
        return None


def _sse_chunk(content, finish_reason=None):
    obj = {
        "id": "vector-brain-hermes",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vector-brain-hermes",
        "choices": [{
            "index": 0,
            "delta": ({"content": content} if content else {}),
            "finish_reason": finish_reason,
        }],
    }
    return ("data: " + json.dumps(obj) + "\n\n").encode("utf-8")


def _sse_done():
    return b"data: [DONE]\n\n"


# ---- Forced camera trigger (2026-10-15, hermes, "forcephoto") ----
# The main model often answered "what am I holding?" from conversation
# history ("you're holding your phone") instead of emitting wire-pod's
# {{getImage||front}} command, so Vector never took a NEW photo. For a
# clear vision question in the LATEST user message (which has no image
# yet) we skip the LLM entirely and return the command ourselves.
# Exact syntax verified in vector-pod kgsim_cmds.go: GetActionsFromString
# splits on "{{" / "}}" / "||"; command "getImage", params "front, lookingUp".
# Kept deliberately tight -- open-ended forms ("what is this/that/it") must
# be the whole utterance so "what is that supposed to mean" never fires.
VISION_TRIGGER_REPLY = "Let me take a look. {{getImage||front}}"
_VT_TAIL = r"(?:[\s,]+(?:vector|hector|buddy|bud|man|dude|right\s+now|now|please))*\s*[?.!]*\s*$"
_VISION_TRIGGER_RE = re.compile(
    r"(?:"
    r"\bwhat\s+am\s+i\s+holding\b"
    r"|\bwhat\s+do\s+i\s+have\s+in\s+my\s+hands?\b"
    r"|\bwhat(?:'s|s|\s+is)\s+in\s+my\s+hands?\b"
    r"|\bwhat\s+(?:do|can)\s+you\s+see\b"
    r"|\blook\s+at\s+(?:this|that|me)\b"
    r"|\bwhat(?:'s|s|\s+is)\s+in\s+front\s+of\s+you\b"
    r"|\bcan\s+you\s+see\s+(?:this|that|me|it|what|anything|something|my)\b"
    r"|\bcan\s+you\s+see" + _VT_TAIL +
    r"|\bscan\s+(?:this|it)\b"
    r"|\bidentify\s+(?:this|it)\b"
    r"|\btake\s+a\s+(?:photo|picture|pic)\b"
    r"|^\s*(?:(?:hey\s+)?vector[\s,]+)?(?:so\s+|okay\s+|ok\s+)?what(?:'s|s|\s+is)\s+(?:this|that|it)"
    r"(?:\s+(?:thing|object|item))?" + _VT_TAIL +
    r")",
    re.IGNORECASE,
)


def _msg_text(content):
    """Text of a message content (str or list of parts); images ignored."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def _msg_has_image(content):
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content)


# visionprompt (2026-10-15): questions about something held/shown get the
# "You're holding ..." framing (only if a held object is actually visible).
_HOLDING_Q_RE = re.compile(
    r"\b(hold|holding|held|in my hands?|showing|show(?:ed)? you|what(?:'s| is) this|what is that|what's that)\b",
    re.I)


def _is_vision_question(text):
    t = (text or "").replace("\u2019", "'").replace("\u2018", "'").strip()
    return bool(t) and bool(_VISION_TRIGGER_RE.search(t))


# ---- Capture loop guard (2026-10-15, hermes, "loopguard") ----
# Live bug ~19:57 UTC: gemma3:4b put {{getImage||front/lookingUp}} in its
# image-turn answers ("Let me get a better angle. {{getImage||lookingUp}}"),
# wire-pod captured again -> another image turn -> loop every ~10s. The text
# model also emitted it for STT noise (q='the'). Default-deny: a capture
# marker survives ONLY on a text turn where the user explicitly asked to
# look/see/photo and the previous user turn was not itself a photo. (Clear
# vision questions never reach the LLM at all -- see _send_forced_getimage.)
_EXPLICIT_LOOK_RE = re.compile(
    r"\b(?:look|looking|see|seeing|photo|photos|picture|pictures|pic|camera|snapshot|"
    r"holding|scan|identify)\b", re.IGNORECASE)
_GI_TAG_RE = re.compile(r"^\s*getImage\b", re.IGNORECASE)
_NVR_TAG_RE = re.compile(r"^\s*newVoiceRequest\b", re.IGNORECASE)


class _CaptureMarkerFilter:
    """Streaming-safe remover of {{getImage||...}} markers, plus a
    {{newVoiceRequest||...}} marker directly attached after one. Text from
    an unclosed "{{" is held back until its "}}" arrives (bounded), so a
    marker split across SSE chunks is still caught."""
    MAX_HOLD = 200

    def __init__(self, reason):
        self.reason = reason
        self.buf = ""
        self.after_gi = False
        self.stripped = []

    def feed(self, text):
        self.buf += text or ""
        return self._drain(False)

    def flush(self):
        return self._drain(True)

    def text(self, text):
        return self.feed(text) + self.flush()

    def _drain(self, final):
        out = []
        while self.buf:
            if self.after_gi:
                st = self.buf.lstrip()
                if not st:
                    if final:
                        out.append(self.buf)
                        self.buf = ""
                    break
                if st.startswith("{{"):
                    j = st.find("}}")
                    if j < 0 and not final and len(st) < self.MAX_HOLD:
                        break
                    tag = st[2:j] if j >= 0 else st[2:]
                    if _NVR_TAG_RE.match(tag):
                        self.stripped.append(st[:j + 2] if j >= 0 else st)
                        self.buf = st[j + 2:] if j >= 0 else ""
                elif st == "{" and not final:
                    break
                self.after_gi = False
                continue
            i = self.buf.find("{{")
            if i < 0:
                if self.buf.endswith("{") and not final:
                    out.append(self.buf[:-1])
                    self.buf = "{"
                else:
                    out.append(self.buf)
                    self.buf = ""
                break
            if i > 0:
                out.append(self.buf[:i])
                self.buf = self.buf[i:]
                continue
            j = self.buf.find("}}")
            if j < 0:
                if not final and len(self.buf) < self.MAX_HOLD:
                    break
                if _GI_TAG_RE.match(self.buf[2:]):
                    self.stripped.append(self.buf)
                else:
                    out.append(self.buf)
                self.buf = ""
                break
            whole = self.buf[:j + 2]
            self.buf = self.buf[j + 2:]
            if _GI_TAG_RE.match(whole[2:-2]):
                self.stripped.append(whole)
                self.after_gi = True
            else:
                out.append(whole)
        return "".join(out)


def _sanitize_completion_json(data, flt):
    obj = json.loads(data)
    for ch in obj.get("choices") or []:
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            msg["content"] = flt.text(msg["content"])
    return json.dumps(obj).encode("utf-8")


def _sanitize_trailing_prompt_question_json(data):
    """Non-stream counterpart of the trailing-prompt-question strip
    applied to streamed replies in _forward_and_capture_stream (see
    _strip_trailing_prompt_question's docstring for the rule). A
    non-streamed completion is already fully generated before Ollama
    sends anything, so there's no latency cost to buffering it here --
    unlike the streamed path, nothing needs a tail-holdback trick.
    Never raises -- any parse failure returns `data` unchanged."""
    try:
        obj = json.loads(data)
        for ch in obj.get("choices") or []:
            msg = ch.get("message") or {}
            if isinstance(msg.get("content"), str):
                msg["content"] = _strip_trailing_prompt_question(msg["content"])
        return json.dumps(obj).encode("utf-8")
    except Exception as e:
        log.error("non-stream trailing-question strip failed, sending as-is: %s", e)
        return data


def _extract_sse_delta_content(line):
    """Best-effort parse of one raw upstream SSE line -> delta content
    string, or "" if it's not a content chunk (control lines, [DONE],
    malformed JSON, etc). Never raises."""
    try:
        s = line.decode("utf-8").strip()
        if not s.startswith("data:"):
            return ""
        s = s[len("data:"):].strip()
        if s == "[DONE]" or not s:
            return ""
        obj = json.loads(s)
        choices = obj.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("delta", {}).get("content") or ""
    except Exception:
        return ""


def _extract_sse_usage(line):
    """Best-effort parse of one raw upstream SSE line -> its OpenAI-style
    'usage' dict (present only on the trailing chunk emitted when the
    request set stream_options.include_usage -- see the speed-pass
    instrumentation in _handle_chat_completions), else None. Never
    raises. That trailer chunk has choices: [] (no delta/content), which
    is exactly why it needs to be swallowed rather than relayed -- wire-pod
    never asked for it and isn't expecting a chunk with no choices."""
    try:
        s = line.decode("utf-8").strip()
        if not s.startswith("data:"):
            return None
        s = s[len("data:"):].strip()
        if s == "[DONE]" or not s:
            return None
        obj = json.loads(s)
        return obj.get("usage")
    except Exception:
        return None


# ---- Trailing prompt-question safety net (2026-10-15, "fix vector and
# the damn ready") ----
# the owner's persona was ending replies with a short follow-up question
# ("Want me to revive them?", "Ready?") even though conversation mode
# already keeps the mic open automatically -- no prompt is ever needed.
# The primary fix is the persona prompt itself (apiConfig.json knowledge.
# openai_prompt, BUDDY RULES). This is the backend safety net in case the
# model still slips: strip a short (<=6 word) trailing question starting
# with one of the stock lead-in words, as long as real content remains
# before it, and never touch a trailing {{...}} command marker.
_TRAIL_HOLDBACK_CHARS = 80  # ~2x the longest matchable trailing question
_TRAILING_PROMPT_LEADIN_RE = re.compile(
    r"^(?:want|ready|should|shall|you\s+ready|sound\s+good|got\s+it|right|ok|okay)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TRAILING_MARKERS_RE = re.compile(r"(\s*\{\{[^{}]*\}\}\s*)+$")


def _strip_trailing_prompt_question(text):
    """Strips a short trailing prompt-question from the END of text, e.g.
    "...the real answer. Want me to revive them?" -> "...the real
    answer." Pure string function (no I/O) so it's safe to call from both
    the non-streamed and streamed response paths. Rules (per the
    2026-10-15 brief):
      - only a FINAL sentence ending in "?"
      - <=6 words
      - starts with a stock lead-in word (want/ready/should/shall/you
        ready/sound good/got it/right/ok/okay)
      - only strips if earlier text remains (never produces an empty or
        marker-only reply)
      - a trailing {{...}} command marker is peeled off first and always
        reattached unchanged -- never touched.
    """
    if not text:
        return text
    m = _TRAILING_MARKERS_RE.search(text)
    if m:
        markers_suffix = text[m.start():]
        body = text[:m.start()]
    else:
        markers_suffix = ""
        body = text
    body_rstripped = body.rstrip()
    if not body_rstripped.endswith("?"):
        return text
    sentences = _SENTENCE_SPLIT_RE.split(body_rstripped)
    if len(sentences) < 2:
        return text  # the whole reply is just this one sentence -- keep it
    candidate = sentences[-1].strip()
    words = candidate.split()
    if not words or len(words) > 6:
        return text
    if not _TRAILING_PROMPT_LEADIN_RE.match(candidate):
        return text
    new_body = body_rstripped[: len(body_rstripped) - len(candidate)].rstrip()
    if not new_body:
        return text  # defensive -- shouldn't happen given the len(sentences)<2 check above
    if markers_suffix:
        return new_body + " " + markers_suffix.strip()
    return new_body



index = Index()


def _reindex_loop():
    # Build the index once at startup, then poll every 60s.
    try:
        index.refresh(force=True)
    except Exception as e:
        log.error("initial index build failed: %s", e)
    while True:
        time.sleep(REINDEX_POLL_SECONDS)
        try:
            changed = index.refresh()
            if changed:
                log.info("re-index: vault change detected, index updated")
        except Exception as e:
            log.error("re-index poll failed: %s", e)


class Handler(BaseHTTPRequestHandler):
    server_version = "VectorBrain/1.0"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.client_address[0], fmt % args)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def _forward_headers(self):
        h = {}
        for k, v in self.headers.items():
            if k.lower() not in HOP_BY_HOP and k.lower() != "host":
                h[k] = v
        return h

    def _proxy_passthrough(self, method):
        """Generic pass-through for anything that isn't /v1/chat/completions."""
        body = self._read_body()
        url = OLLAMA_BASE + self.path
        req = urllib.request.Request(url, data=body or None, method=method,
                                      headers=self._forward_headers())
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP:
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read() or b"")
        except Exception as e:
            log.error("passthrough error for %s %s: %s", method, self.path, e)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        if self.path.rstrip("/") == "/metrics/summary":
            self._handle_metrics_summary()
            return
        self._proxy_passthrough("GET")

    def _handle_metrics_summary(self):
        try:
            summary = _compute_metrics_summary()
            body = json.dumps(summary).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            log.error("metrics summary handler failed: %s", e)
            body = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_DELETE(self):
        self._proxy_passthrough("DELETE")

    def do_PUT(self):
        self._proxy_passthrough("PUT")

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._proxy_passthrough("POST")
            return
        self._handle_chat_completions()

    def _inject_context(self, payload):
        """Mutates payload['messages'] in place, inserting up to THREE
        separate system messages right after wire-pod's own leading
        system message(s), in a fixed STATIC -> SLOW -> DYNAMIC order.
        Returns (included_labels, retrieval_ms).

        Speed pass (2026-09-29, Hermes, "cut voice latency"): this used to
        build ONE combined system message mixing always-on notes (static)
        with per-question RAG hits, curious-vector's observation, same-day
        memory and the exact clock/battery line (all dynamic) into a
        single string. Because that string differs on nearly every
        request -- RAG hits alone guarantee it -- Ollama's prompt/KV cache
        could never reuse ANY of it, including the always-on notes that
        are actually byte-identical turn to turn. Verified empirically on
        this host (see bench notes in the return report): an identical
        single system message gets prompt_eval_cached_count ~= its full
        token count on a repeat call (near-free re-prefill), but appending
        anything to that same string -- even a few words at the very end
        -- drops the cache hit to almost nothing. So the fix has to be
        separate messages, ordered least- to most-variable:
          1. STATIC  -- always-on vault notes (static_block). Byte-
             identical across requests unless the vault file changes.
          2. SLOW    -- mood line + same-day conversation memory. Changes
             at most once per turn, and is identical across any two Ollama
             calls that happen to fire within the same turn (e.g. the
             local-fallback race in _handle_escalated_chat).
          3. DYNAMIC -- per-question RAG hits + curious-vector's recent
             observation + the exact clock/battery line. Different on
             essentially every request; kept last so it never breaks the
             shared prefix of the two tiers ahead of it.
        Empty tiers are simply skipped -- this can still insert 0, 1, 2 or
        3 messages depending on what's available for a given turn.
        """
        messages = payload.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return [], 0
        latest_user_text = user_msgs[-1].get("content", "")
        if isinstance(latest_user_text, list):
            # some clients send content as a list of parts; flatten text parts
            latest_user_text = " ".join(
                p.get("text", "") for p in latest_user_text if isinstance(p, dict)
            )

        static_text, static_included = static_block(index)
        static_text = (static_text + "\n\n" + FACES_RULE) if static_text else FACES_RULE
        retrieval_text, retrieval_included, retrieval_ms = retrieval_block(index, latest_user_text)

        # x_vector_internal=True (curious-vector's proactive-remark
        # generator, see curious.py's _generate_proactive_line) is Vector
        # asking its own brain for a line, not a real the owner voice turn --
        # skip mood tracking for it so a quiet room doesn't look "social"
        # just because the proactive-remark generator itself ran. RAG/
        # mood/same-day context is still injected below either way, since
        # that's exactly what makes the generated line good.
        if not payload.get("x_vector_internal"):
            _note_voice_turn(latest_user_text)

        # ---- SLOW tier: mood + same-day memory ----
        slow_parts = []
        included = list(static_included)
        mood_line = _mood_context_line()
        if mood_line:
            slow_parts.append(mood_line)
            included.append("mood context")
        today_line = _today_conversation_context()
        if today_line:
            slow_parts.append(today_line)
            included.append("today's conversation so far")
        slow_text = "\n\n".join(slow_parts) if slow_parts else None

        # ---- DYNAMIC tier: RAG hits + curious observation + clock/battery ----
        dynamic_parts = []
        if retrieval_text:
            dynamic_parts.append(retrieval_text)
            included += retrieval_included
        recent_obs = _read_recent_observation()
        if recent_obs:
            dynamic_parts.append(f"[curious-vector recent observation]\n{recent_obs}")
            included.append("curious-vector recent observation")
        people_line = _current_people_line()
        if people_line:
            dynamic_parts.append(people_line)
            included.append("people in view")
            people_notes = _people_notes_block()
            if people_notes:
                dynamic_parts.append(people_notes)
                included.append("people notes")
                self._people_ctx_injected = True
        tb_line = _time_battery_line()
        if tb_line:
            dynamic_parts.append(tb_line)
            included.append("time/battery context")
        dynamic_text = "\n\n".join(dynamic_parts) if dynamic_parts else None

        if not (static_text or slow_text or dynamic_text):
            return [], retrieval_ms

        # Insert right after the leading run of system messages, so wire-pod's
        # own system prompt + {{command}} instructions stay first/intact.
        # Each tier becomes its own system message, inserted in
        # STATIC -> SLOW -> DYNAMIC order (see docstring above).
        insert_at = 0
        while insert_at < len(messages) and messages[insert_at].get("role") == "system":
            insert_at += 1
        for text in (static_text, slow_text, dynamic_text):
            if text:
                messages.insert(insert_at, {"role": "system", "content": text})
                insert_at += 1
        payload["messages"] = messages
        return included, retrieval_ms

    def _handle_chat_completions(self):
        t0 = time.time()
        raw_body = self._read_body()
        included, retrieval_ms = [], 0
        question = ""
        self._metrics_route_logged = False  # collab-mode: guards against double metrics writes
        self._gi_filter = None  # loopguard: per-request, handler is reused on keep-alive
        self._people_ctx_injected = False  # people notes: keep local drafts away from Hermes when set

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            log.warning("could not parse request JSON, forwarding raw: %s", e)
            payload = None

        body_to_send = raw_body

        # Forced camera trigger (2026-10-15, forcephoto) -- see
        # _VISION_TRIGGER_RE. Only when the LATEST message is the user's, it
        # carries no image, and its text is a clear vision question.
        latest_has_image = False
        prev_user_image = False  # loopguard
        guard_q = ""
        if payload is not None:
            try:
                _all = payload.get("messages", []) or []
                _lu = max((i for i, mm in enumerate(_all) if mm.get("role") == "user"), default=-1)
                if _lu >= 0:
                    latest_has_image = _msg_has_image(_all[_lu].get("content"))
                    _qtext = _msg_text(_all[_lu].get("content"))
                    guard_q = _qtext
                    for mm in reversed(_all[:_lu]):
                        if mm.get("role") == "user":
                            prev_user_image = (_msg_has_image(mm.get("content"))
                                               or "[earlier photo]" in _msg_text(mm.get("content")))
                            break
                    if (not latest_has_image and _lu == len(_all) - 1
                            and not payload.get("x_vector_internal")
                            and _is_vision_question(_qtext)):
                        self._send_forced_getimage(payload, _qtext, t0)
                        return
            except Exception as e:
                log.error("vision trigger check failed, continuing normally: %s", e)

        if payload is not None:
            try:
                user_msgs = [m for m in payload.get("messages", []) if m.get("role") == "user"]
                question = str(user_msgs[-1].get("content", "") if user_msgs else "")[:300]
                # forcephoto: an image turn gets NO injected memory/RAG/
                # same-day-conversation context -- that text is exactly the
                # history that made Vector say "your phone" for a new photo.
                if latest_has_image:
                    included, retrieval_ms = [], 0
                else:
                    included, retrieval_ms = self._inject_context(payload)
                body_to_send = json.dumps(payload).encode("utf-8")
            except Exception as e:
                # Fail open: forward the original, unmodified body.
                log.error("context injection failed, forwarding unmodified: %s", e)
                body_to_send = raw_body
                included, retrieval_ms = [], 0

        # Vision support (2026-09-26): wire-pod's {{getImage}} sends a photo as
        # multimodal content; the main text model rejects it (400 content type nil).
        # Normalize null contents and route image turns to the local vision model.
        has_image = False
        if payload is not None:
            try:
                # 2026-10-15 (hermes): only the LATEST user message decides routing.
                # wire-pod replays remembered chats, so an old photo in history used to
                # flag EVERY later turn as an image turn -> qwen2.5vl:3b, which never
                # emits {{getImage}} ("I will take a photo" ... and nothing). Older
                # images are flattened to text so the text model accepts the history.
                _msgs = payload.get("messages", [])
                _last_user = max((i for i, mm in enumerate(_msgs) if mm.get("role") == "user"), default=-1)
                for _i, m in enumerate(_msgs):
                    c = m.get("content")
                    if c is None:
                        m["content"] = ""
                    elif isinstance(c, list):
                        if any(isinstance(x, dict) and x.get("type") == "image_url" for x in c):
                            if _i == _last_user:
                                has_image = True
                            else:
                                _txt = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text").strip()
                                m["content"] = (_txt + " [earlier photo]").strip()
                if has_image:
                    payload["model"] = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")
                    # 2026-09-29 (ctx-vram fix): vision back on GPU. qwen3:32b
                    # now loads at num_ctx=8192 (~22GB, see the context-length
                    # fix comment near OLLAMA_KEEP_ALIVE_VALUE above) instead of
                    # 32768 (~29GB), freeing ~10GB across the two 16GB cards --
                    # comfortably enough for qwen2.5vl:3b (4.1GB weights, and
                    # its own num_ctx is capped at 4096 via a Modelfile bake, see
                    # same comment). -1 = let Ollama's scheduler offload every
                    # layer it can fit rather than forcing a specific count.
                    # Set VISION_NUM_GPU=0 to force CPU again (e.g. if qwen3
                    # ever grows back past what's free) without a code change.
                    payload.setdefault("options", {})
                    payload["options"]["num_gpu"] = int(os.environ.get("VISION_NUM_GPU", "-1"))
                    # the owner 2026-09-27: he must be able to READ. Tell the vision model to
                    # read labels/brands/text exactly and never claim it can't read.
                    payload.setdefault("messages", []).insert(0, {"role": "system", "content":
                        # visionprompt (2026-10-15): "always read brands" pushed gemma
                        # into inventing them ("Samsung Galaxy S23" on a desk photo).
                        "You are looking through your own camera. You CAN read: when text, brand names, "
                        "model names or labels are clearly legible, read them exactly and mention them. "
                        "Never say you can't read, but never invent text that isn't legible."})
                    # forcephoto (2026-10-15): strip earlier chat turns so the
                    # vision model describes THIS photo, not what history says
                    # ("you're holding your phone"). Keep every system prompt
                    # (incl. the "You CAN read" one above) + only the latest
                    # user message. wire-pod's image message is image-only, so
                    # carry the question over from the most recent earlier user
                    # text turn as a text part.
                    _msgs = payload["messages"]
                    _lu = max((i for i, mm in enumerate(_msgs) if mm.get("role") == "user"), default=-1)
                    _img_msg = dict(_msgs[_lu])
                    _parts = [p for p in (_img_msg.get("content") or []) if isinstance(p, dict)]
                    _vq = _msg_text(_parts)
                    if not _vq:
                        for mm in reversed(_msgs[:_lu]):
                            if mm.get("role") == "user":
                                _vq = _msg_text(mm.get("content")).replace("[earlier photo]", "").strip()
                                if _vq:
                                    break
                        _vq = _vq or "What do you see?"
                        _parts = [{"type": "text", "text": _vq}] + [pp for pp in _parts if pp.get("type") != "text"]
                    _img_msg["content"] = _parts
                    # brighten (2026-10-15): Vector's camera runs auto-exposure
                    # already pinned at max (66 ms, gain 3.8), so indoor frames
                    # arrive at mean luma ~55-85. Auto-level dark frames before
                    # the vision model sees them. Own try: must never break routing.
                    if os.environ.get("BRIGHTEN_IMAGES", "1") != "0":
                        try:
                            if brighten_data_uri is None:
                                log.warning("image brighten unavailable: %s", _BRIGHTEN_IMPORT_ERR)
                            else:
                                for _pp in _parts:
                                    if _pp.get("type") != "image_url":
                                        continue
                                    _iu = _pp.get("image_url")
                                    _u = _iu.get("url") if isinstance(_iu, dict) else _iu
                                    _new, _binfo = brighten_data_uri(_u)
                                    if not _new:
                                        log.info("image brighten skipped: %s", _binfo)
                                        continue
                                    if isinstance(_iu, dict):
                                        _iu["url"] = _new
                                    else:
                                        _pp["image_url"] = _new
                                    log.info("image brightened: mean %s->%s (p50 %s->%s, p95 %s->%s, gamma %s, gain %s)",
                                             _binfo.get("mean_before"), _binfo.get("mean_after"),
                                             _binfo.get("p50_before"), _binfo.get("p50_after"),
                                             _binfo.get("p95_before"), _binfo.get("p95_after"),
                                             _binfo.get("gamma"), _binfo.get("gain"))
                                    try:  # keep the latest pair for eyeballing
                                        import base64 as _b64
                                        os.makedirs(BRIGHTEN_DEBUG_DIR, exist_ok=True)
                                        for _nm, _du in (("last_before.jpg", _u), ("last_after.jpg", _new)):
                                            with open(os.path.join(BRIGHTEN_DEBUG_DIR, _nm), "wb") as _fh:
                                                _fh.write(_b64.b64decode(_du.split(",", 1)[1]))
                                    except Exception:
                                        pass
                        except Exception as _be:
                            log.error("image brighten failed, sending original: %s", _be)
                    _sys = [mm for mm in _msgs if mm.get("role") == "system"]
                    _dropped = len(_msgs) - len(_sys) - 1
                    # visionprompt (2026-10-15): the perspective fix always told
                    # gemma the person was "showing or holding something", so it
                    # invented held objects ("You're holding a pair of black
                    # glasses" for a face, a "Galaxy S23" for a desk). Now the
                    # "You're holding ..." framing is only offered when the
                    # question is about holding/showing, and even then only if a
                    # hand holding an object is clearly visible.
                    _holding_q = bool(_HOLDING_Q_RE.search(_vq or ""))
                    if _holding_q:
                        _persp = (
                            "The person talking to you (usually the owner) is asking about something they "
                            "may be holding or showing you. ONLY if you can clearly see a hand holding or "
                            "showing an object, refer to them as 'you' (e.g. 'You're holding a ...'). If no "
                            "hand or held object is clearly visible, say you can't see anything in their "
                            "hand, then plainly describe what you do see (e.g. 'I see a desk with ...'). ")
                    else:
                        _persp = (
                            "Plainly describe what is actually visible, from your own point of view "
                            "(e.g. 'I see a desk with ...'). Do not assume anyone is holding or showing "
                            "you anything. ")
                    _sys.append({"role": "system", "content":
                        "A NEW photo from your camera is attached to the user's message. Answer the "
                        "question using ONLY what is visible in this photo. Do not rely on earlier "
                        "conversation and do not use the getImage command again. "
                        "This photo is from your own camera: it is your eyes, so you never appear in it "
                        "and you have no hands. " + _persp +
                        "Any person in the photo is the person talking to you: call them 'you' "
                        "(e.g. 'I see you with ...'), never 'myself' or 'me'. "
                        "Never say 'I am holding' or 'in my hand'. "
                        "Describe only what is clearly visible. If unsure, say what it looks like and "
                        "that you're not sure. Never guess a brand or model unless the text is legible. "
                        "Never invent labels or codes. If there's no readable text, don't mention any. "
                        "Keep it to 1-2 short spoken sentences, no markdown."})
                    payload["messages"] = _sys + [_img_msg]
                    log.info("image turn detected -> routing to %s (history stripped: %d earlier msgs dropped, q=%r, holding_q=%s)",
                             payload["model"], _dropped, _vq[:120], _holding_q)
                body_to_send = json.dumps(payload).encode("utf-8")
            except Exception as e:
                log.error("vision normalization failed, forwarding as-is: %s", e)
        # qwen3 dense models (e.g. qwen3:32b) are hybrid "thinking" models: via the OpenAI
        # endpoint they burn the whole budget on reasoning and return empty content.
        # Use qwen3's soft switch (/no_think) + Ollama's reasoning_effort to disable it.
        if payload is not None:
            try:
                mdl = str(payload.get("model", ""))
                if mdl.startswith("qwen3") and "instruct" not in mdl and "vl" not in mdl:
                    msgs = payload.get("messages", [])
                    for m in reversed(msgs):
                        if m.get("role") == "user":
                            c = m.get("content")
                            if isinstance(c, str) and "/no_think" not in c:
                                m["content"] = c + " /no_think"
                            break
                    payload["reasoning_effort"] = "none"
                    body_to_send = json.dumps(payload).encode("utf-8")
            except Exception as e:
                log.error("no_think injection failed: %s", e)
        # Keep-alive fix (2026-09-29, live-10, Hermes): see
        # OLLAMA_KEEP_ALIVE_VALUE's definition above for why this must be
        # set per-request rather than relying on the systemd env default.
        # Applied unconditionally (any model, any turn) after the qwen3
        # /no_think block above so it can't be skipped by that block's own
        # early exceptions.
        if payload is not None:
            try:
                payload.setdefault("keep_alive", OLLAMA_KEEP_ALIVE_VALUE)
                body_to_send = json.dumps(payload).encode("utf-8")
            except Exception as e:
                log.error("keep_alive injection failed: %s", e)
        is_stream = bool(payload.get("stream")) if payload else False

        # Speed-pass instrumentation (2026-09-29, Hermes, "cut voice
        # latency"): ask Ollama for its OpenAI-compatible usage trailer
        # (prompt_tokens / cached_tokens / completion_tokens) on every
        # streamed request, so turns.jsonl can show real prefill/cache-hit
        # numbers per turn instead of only wall-clock latency. wire-pod
        # never asked for this chunk and isn't expecting it -- it's
        # captured then swallowed before relaying to the real client, see
        # _forward_and_capture_stream / _extract_sse_usage below. Fails
        # open: any error here just means this turn's metrics are missing
        # the usage fields, it never blocks the request.
        if payload is not None and is_stream:
            try:
                so = payload.get("stream_options")
                if not isinstance(so, dict):
                    so = {}
                so["include_usage"] = True
                payload["stream_options"] = so
                body_to_send = json.dumps(payload).encode("utf-8")
            except Exception as e:
                log.error("stream_options injection failed: %s", e)

        # loopguard (2026-10-15): decide whether this turn may emit a camera
        # capture marker -- see _CaptureMarkerFilter.
        if payload is not None:
            try:
                _gq = (guard_q or "").strip()
                if has_image or latest_has_image:
                    _reason = "image-turn"
                elif prev_user_image:
                    _reason = "after-image-turn"
                elif _is_vision_question(_gq) or _EXPLICIT_LOOK_RE.search(_gq):
                    _reason = None
                elif len(_gq.split()) <= 2:
                    _reason = "noise"
                else:
                    _reason = "no-explicit-ask"
                if _reason:
                    self._gi_filter = _CaptureMarkerFilter(_reason)
            except Exception as e:
                log.error("loopguard setup failed (defaulting to strip): %s", e)
                self._gi_filter = _CaptureMarkerFilter("guard-error")

        # Hermes escalation -- preemptive path (added 2026-09-27). Only for
        # real streamed text turns (never image turns, never non-streamed
        # internal calls). Excludes chit-chat/commands via heuristic. Rate
        # limited to <=1 escalation per HERMES_ESCALATION_MIN_INTERVAL_SECONDS
        # so a chatty session can't hammer the Hermes relay.
        # Don't escalate when Vector's own notes/lessons already cover the question
        # (the owner 2026-09-28: "is his qwen brain down cause he is asking you").
        _local_best = 0.0
        for _lbl in (included or []):
            _m = re.search(r"sim=([0-9.]+)", str(_lbl))
            if _m and "20-agents/vector/" in str(_lbl):
                _local_best = max(_local_best, float(_m.group(1)))
        _known_locally = _local_best >= float(os.environ.get("HERMES_ESCALATION_LOCAL_SIM", "0.55"))
        if _known_locally and HERMES_ESCALATION_ENABLED and _is_escalation_candidate(question):
            log.info("HERMES-ESCALATION skipped (known locally, sim=%.2f) q=%r", _local_best, question)
        if (HERMES_ESCALATION_ENABLED and payload is not None and is_stream
                and not has_image and not _known_locally and _is_escalation_candidate(question)):
            if _reserve_escalation_slot():
                log.info("HERMES-ESCALATION preemptive q=%r", question)
                self._handle_escalated_chat(payload, question, t0)
                self._log_gi_guard(question)
                return
            else:
                log.info("HERMES-ESCALATION candidate skipped (rate-limited) q=%r", question)

        url = OLLAMA_BASE + "/v1/chat/completions"
        headers = self._forward_headers()
        headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body_to_send, method="POST", headers=headers)

        try:
            upstream = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            total_ms = int((time.time() - t0) * 1000)
            log.info("q=%r notes=%s retrieval_ms=%d total_ms=%d upstream_status=%d",
                      question, included, retrieval_ms, total_ms, e.code)
            self.send_response(e.code)
            self.send_header("X-Vector-Brain-Context", str(len(included)))
            self.end_headers()
            self.wfile.write(e.read() or b"")
            return
        except Exception as e:
            log.error("upstream call failed: %s", e)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # _pre_body: non-streamed body, fully buffered and sanitized
        # before headers go out. Originally loopguard-only; extended
        # 2026-10-15 (Hermes, "fix vector and the damn ready") to
        # ALWAYS buffer+fix up a non-stream reply, since a non-streamed
        # completion is already fully generated upstream before
        # anything is sent -- buffering it here costs nothing extra.
        _pre_body = None
        if not is_stream:
            try:
                _pre_body = upstream.read()
                if self._gi_filter is not None:
                    try:
                        _pre_body = _sanitize_completion_json(_pre_body, self._gi_filter)
                    except Exception as e:
                        log.error("loopguard non-stream sanitize failed, sending as-is: %s", e)
                try:
                    _pre_body = _sanitize_trailing_prompt_question_json(_pre_body)
                except Exception as e:
                    log.error("non-stream trailing-question strip failed, sending as-is: %s", e)
            except Exception as e:
                log.error("non-stream body read failed: %s", e)
                _pre_body = b""

        self.send_response(upstream.status)
        for k, v in upstream.getheaders():
            if k.lower() not in HOP_BY_HOP and not (_pre_body is not None and k.lower() == "content-length"):
                self.send_header(k, v)
        if _pre_body is not None:
            self.send_header("Content-Length", str(len(_pre_body)))
        self.send_header("X-Vector-Brain-Context", str(len(included)))
        self.end_headers()

        check_uncertainty = (
            HERMES_ESCALATION_ENABLED and is_stream and not has_image
            and payload is not None and len(question.strip()) >= 4
            and not _CHITCHAT_PATTERNS.search(question)
            and not _COMMAND_KEYWORDS.search(question)
        )

        stream_stats = {}
        try:
            if is_stream and check_uncertainty:
                self._relay_stream_with_uncertainty_check(upstream, question, t0)
            elif is_stream:
                stream_stats = self._forward_and_capture_stream(upstream, t0)
            elif _pre_body is not None:
                self.wfile.write(_pre_body)
            else:
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass
        self._log_gi_guard(question)

        total_ms = int((time.time() - t0) * 1000)
        log.info("q=%r notes=%s retrieval_ms=%d total_ms=%d stream=%s ttft_ms=%s prompt_tokens=%s cached_tokens=%s",
                  question, included, retrieval_ms, total_ms, is_stream,
                  stream_stats.get("ttft_ms"), stream_stats.get("prompt_tokens"), stream_stats.get("cached_tokens"))

        # Collab-mode metrics (item 4): only fires for turns that fell
        # through to plain local passthrough without ever going through
        # _handle_escalated_chat or _relay_stream_with_uncertainty_check
        # (both of which log their own route + return before reaching
        # here). Skips image turns and internal (x_vector_internal)
        # proactive-remark generation -- neither is a real the owner voice
        # turn worth counting in the local/hermes scoreboard.
        if (not getattr(self, "_metrics_route_logged", False) and question
                and not has_image and not (payload is not None and payload.get("x_vector_internal"))):
            route = "command" if _COMMAND_KEYWORDS.search(question) else "local"
            self._log_metrics_once(route, question, t0, extra=stream_stats)

    def _gi_filter_lines(self, line):
        """loopguard: run one raw upstream SSE line through this turn's
        capture-marker filter. Returns a list of (bytes, content) entries."""
        content = _extract_sse_delta_content(line)
        f = getattr(self, "_gi_filter", None)
        if f is None:
            return [(line, content)]
        if content:
            out = f.feed(content)
            if out == content:
                return [(line, content)]
            if not out:
                return []
            return [(_sse_chunk(out)[:-1], out)]
        if f.buf and line.strip().startswith(b"data:"):
            rem = f.flush()
            res = [(_sse_chunk(rem), rem)] if rem else []
            res.append((line, ""))
            return res
        return [(line, content)]

    def _gi_filter_text(self, text):
        f = getattr(self, "_gi_filter", None)
        return f.text(text) if (f is not None and text) else text

    def _log_gi_guard(self, question):
        f = getattr(self, "_gi_filter", None)
        if f is not None and f.stripped:
            log.info("loopguard: stripped %s (reason=%s) q=%r", f.stripped, f.reason, str(question)[:120])

    def _send_forced_getimage(self, payload, qtext, t0):
        """forcephoto (2026-10-15): answer a vision question WITHOUT the LLM
        by returning wire-pod's camera command, so Vector always takes a
        fresh photo. Honors stream=true (OpenAI SSE chunks + [DONE])."""
        log.info("vision trigger -> forced getImage for q=%r", (qtext or "")[:200])
        try:
            if not payload.get("x_vector_internal"):
                _note_voice_turn(qtext)
        except Exception:
            pass
        content = VISION_TRIGGER_REPLY
        try:
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Vector-Brain-Context", "0")
                self.send_header("X-Vector-Brain-Route", "vision-trigger")
                self.end_headers()
                self.wfile.write(_sse_chunk(content))
                self.wfile.write(_sse_chunk("", finish_reason="stop"))
                self.wfile.write(_sse_done())
                self.wfile.flush()
            else:
                body = json.dumps({
                    "id": "vector-brain-vision-trigger",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(payload.get("model") or "vector-brain"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Vector-Brain-Context", "0")
                self.send_header("X-Vector-Brain-Route", "vision-trigger")
                self.end_headers()
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        log.info("vision trigger done total_ms=%d", int((time.time() - t0) * 1000))

    def _log_metrics_once(self, route, question, t0, extra=None):
        if getattr(self, "_metrics_route_logged", False):
            return
        self._metrics_route_logged = True
        latency_ms = int((time.time() - t0) * 1000)
        _log_turn_metrics(route, question, latency_ms, extra=extra)

    def _handle_escalated_chat(self, payload, question, t0):
        """Handles a chat-completions request that was preemptively picked
        for Hermes escalation. Sends an early filler chunk immediately (so
        wire-pod's SSE stream reader has something to speak while it waits
        -- see kgsim.go's punctuation-triggered speech, which means it
        speaks as soon as this chunk lands), then races a local fallback
        generation against the ask-Hermes receiver call, and streams back
        whichever answer is usable. Always finishes the SSE stream itself,
        even on total failure, so wire-pod never hangs."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Vector-Brain-Context", "0")
        self.send_header("X-Vector-Brain-Escalation", "hermes-preemptive")
        self.end_headers()

        try:
            self.wfile.write(_sse_chunk("{{playAnimationWI||thinking}} Let me check with Hermes. "))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        local_result = {}

        def _local_worker():
            local_result["text"] = _call_ollama_nonstream(payload)

        local_thread = threading.Thread(target=_local_worker, daemon=True)
        local_thread.start()

        hermes_reply = _ask_hermes(question, HERMES_ASK_TIMEOUT_SECONDS)
        local_thread.join(timeout=HERMES_ASK_TIMEOUT_SECONDS)

        if hermes_reply:
            final_text = hermes_reply
            source = "hermes"
        elif local_result.get("text"):
            final_text = local_result["text"]
            source = "local-fallback"
        else:
            final_text = "Hermes didn't get back to me in time, and I'm drawing a blank too."
            source = "generic-fallback"

        final_text = self._gi_filter_text(final_text)  # loopguard
        final_text = _strip_trailing_prompt_question(final_text)
        try:
            self.wfile.write(_sse_chunk(final_text, finish_reason="stop"))
            self.wfile.write(_sse_done())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

        total_ms = int((time.time() - t0) * 1000)
        log.info("HERMES-ESCALATION q=%r source=%s hermes_replied=%s total_ms=%d",
                  question, source, bool(hermes_reply), total_ms)

        # Collab-mode (2026-09-29, Hermes): metrics + learn-back. "hermes"
        # only when Hermes itself actually answered; both fallback sources
        # ("local-fallback" and "generic-fallback") mean escalation was
        # attempted but Hermes never replied in time, so they both count
        # as hermes_timeout in the scoreboard, not "local" -- the point of
        # the metric is to show how often Hermes is unreachable/slow, and
        # a local-fallback still means the robot spoke, but not with the
        # escalation working as intended.
        route = "hermes" if hermes_reply else "hermes_timeout"
        self._log_metrics_once(route, question, t0)
        if hermes_reply:
            threading.Thread(
                target=_learn_from_hermes, args=(question, hermes_reply), daemon=True
            ).start()

    def _forward_and_capture_stream(self, upstream, t0, t_first_content=None):
        """Reads the rest of the SSE lines from `upstream` and forwards
        them to the client unchanged EXCEPT (a) the trailing usage-only
        chunk that our own stream_options.include_usage=True request (see
        _handle_chat_completions) causes Ollama to add -- that chunk has
        choices: [] and wire-pod never asked for it, so it's captured for
        metrics and swallowed rather than relayed, and (b) a short
        trailing prompt-question (2026-10-15 safety net, see
        _strip_trailing_prompt_question) -- see below for how that's done
        without re-introducing the "wait for the whole reply before
        speaking" latency this codebase has repeatedly fixed (same
        concern _relay_stream_with_uncertainty_check's header calls out).
        Returns a dict with ttft_ms / prompt_tokens / cached_tokens /
        completion_tokens (any may be None if never seen). t_first_content,
        if already known (e.g. the uncertainty-check path already
        buffered a first sentence and knows when its first token
        arrived), is passed in so it's preserved rather than recomputed
        from whatever's left.

        Trailing-strip mechanics: instead of buffering the WHOLE reply
        (which would delay speech until the entire thing finished
        generating), only a small bounded TAIL of already-received
        content is held back -- enough to cover the longest possible
        matched trailing question (~40 chars for 6 words), doubled for
        margin. Everything older is flushed immediately, same as before.
        Only once the stream truly ends do we check whether the full
        accumulated text's ending matches the trailing-question pattern;
        if so, the still-held-back tail is trimmed (or dropped entirely)
        before the final flush. Non-content lines (blank SSE separators,
        the finish_reason chunk, [DONE]) are never touched by the trim --
        only content-bearing chunks are shortened/dropped."""
        usage = None
        full_text_parts = []
        pending = []  # list of [line_bytes, content_str], oldest first
        pending_len = 0

        while True:
            line = upstream.readline()
            if not line:
                break
            usage_here = _extract_sse_usage(line)
            if usage_here is not None:
                usage = usage_here
                continue  # swallow -- see docstring above
            for line, content in self._gi_filter_lines(line):  # loopguard
                if content:
                    if t_first_content is None:
                        t_first_content = time.time()
                    full_text_parts.append(content)
                pending.append([line, content])
                pending_len += len(content)
            while pending_len > _TRAIL_HOLDBACK_CHARS and len(pending) > 1:
                old_line, old_content = pending.pop(0)
                self.wfile.write(old_line)
                pending_len -= len(old_content)
            self.wfile.flush()

        _gf = getattr(self, "_gi_filter", None)  # loopguard: stream ended mid-hold
        if _gf is not None and _gf.buf:
            _rem = _gf.flush()
            if _rem:
                full_text_parts.append(_rem)
                pending.append([_sse_chunk(_rem), _rem])

        full_text = "".join(full_text_parts)
        stripped = _strip_trailing_prompt_question(full_text)
        removed_len = len(full_text) - len(stripped)
        if removed_len > 0:
            idx = len(pending) - 1
            while removed_len > 0 and idx >= 0:
                line_bytes, content_str = pending[idx]
                if not content_str:
                    idx -= 1
                    continue  # control/blank line -- skip over, keep as-is
                if len(content_str) <= removed_len:
                    removed_len -= len(content_str)
                    del pending[idx]
                else:
                    keep = content_str[: len(content_str) - removed_len]
                    pending[idx] = [_sse_chunk(keep), keep]
                    removed_len = 0
                idx -= 1

        for line_bytes, _content_str in pending:
            self.wfile.write(line_bytes)
        self.wfile.flush()

        details = (usage or {}).get("prompt_tokens_details") or {}
        return {
            "ttft_ms": int((t_first_content - t0) * 1000) if t_first_content else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
        }

    def _relay_stream_with_uncertainty_check(self, upstream, question, t0):
        """Relays the local model's stream to wire-pod, but buffers only
        up to the FIRST sentence (the same boundary wire-pod itself waits
        for before speaking -- see kgsim.go's punctuation split -- so this
        adds no perceptible latency for the common case). If that first
        sentence reads as an "I don't know"-style hedge, aborts the local
        answer and escalates to Hermes instead; otherwise flushes the
        buffered sentence and continues plain byte-for-byte passthrough
        for the remainder, unchanged from the non-escalating path."""
        buf_parts = []
        pending_lines = []
        first_sentence_done = False
        t_first_content = None

        while not first_sentence_done:
            line = upstream.readline()
            if not line:
                break
            if _extract_sse_usage(line) is not None:
                continue  # swallow -- see _forward_and_capture_stream's docstring
            pending_lines.append(line)
            content = _extract_sse_delta_content(line)
            if content:
                if t_first_content is None:
                    t_first_content = time.time()
                buf_parts.append(content)
                joined = "".join(buf_parts)
                # Speed pass (2026-09-29, Hermes): two tweaks, both aimed at
                # the same measured problem -- a long, comma-heavy first
                # "sentence" with no early '.', '!' or '?' used to make
                # Vector wait for the full FIRST_SENTENCE_MAX_CHARS safety
                # cap (was 200) before anything reached wire-pod, adding
                # up to ~1-2s of pure decode wait on this hardware's
                # ~18-20 tok/s. (1) The cap itself is lower now -- still
                # generous for a real short sentence, much cheaper on a
                # run-on one. (2) The uncertainty check now also runs on
                # every increment, not just once the buffer is "done" --
                # hedge phrases ("I don't know" / "I'm not sure" / etc.)
                # are always short and near the START of a sentence when
                # the model produces them, so this lets a genuine
                # escalation fire as soon as the phrase appears instead of
                # waiting for the rest of the sentence to generate first.
                if _SENTENCE_END_RE.search(joined) or len(joined) > FIRST_SENTENCE_MAX_CHARS:
                    first_sentence_done = True
                elif _UNCERTAINTY_PATTERNS.search(joined):
                    first_sentence_done = True

        first_sentence_text = "".join(buf_parts)
        ttft_ms = int((t_first_content - t0) * 1000) if t_first_content else None

        if first_sentence_text and _UNCERTAINTY_PATTERNS.search(first_sentence_text) \
                and _reserve_escalation_slot():
            log.info("HERMES-ESCALATION uncertainty-triggered q=%r first_sentence=%r",
                      question, first_sentence_text[:120])
            try:
                upstream.close()
            except Exception:
                pass
            self.wfile.write(_sse_chunk("{{playAnimationWI||thinking}} Let me check with Hermes. "))
            self.wfile.flush()
            # local_draft=first_sentence_text (collab-mode item 5): unlike
            # the preemptive path, we already have the local model's own
            # (hedging) first sentence here, so Hermes gets it as context
            # instead of "local draft: none".
            # people notes (2026-10-15): if this turn carried a person's
            # private note summary, the local draft may echo it -- keep it
            # local and send Hermes the bare question only.
            _draft = None if getattr(self, "_people_ctx_injected", False) else first_sentence_text
            hermes_reply = _ask_hermes(question, HERMES_ASK_TIMEOUT_SECONDS, local_draft=_draft)
            final_text = hermes_reply or first_sentence_text.strip() or "I'm not sure, sorry."
            final_text = self._gi_filter_text(final_text)  # loopguard
            final_text = _strip_trailing_prompt_question(final_text)
            self.wfile.write(_sse_chunk(final_text, finish_reason="stop"))
            self.wfile.write(_sse_done())
            self.wfile.flush()
            total_ms = int((time.time() - t0) * 1000)
            log.info("HERMES-ESCALATION uncertainty q=%r source=%s total_ms=%d",
                      question, "hermes" if hermes_reply else "local-first-sentence", total_ms)
            self._log_metrics_once("hermes" if hermes_reply else "hermes_timeout", question, t0,
                                    extra={"ttft_ms": ttft_ms})
            if hermes_reply:
                threading.Thread(
                    target=_learn_from_hermes, args=(question, hermes_reply), daemon=True
                ).start()
            return

        # Not uncertain (or rate-limited past its window): flush what we
        # buffered, then continue passthrough (with the same usage/ttft
        # capture as the non-escalating path) for the remainder. This is a
        # genuine "answered locally" outcome for the scoreboard.
        for line in pending_lines:
            for _b, _c in self._gi_filter_lines(line):  # loopguard
                self.wfile.write(_b)
        self.wfile.flush()
        tail_stats = self._forward_and_capture_stream(upstream, t0, t_first_content=t_first_content)
        if tail_stats.get("ttft_ms") is None and ttft_ms is not None:
            tail_stats["ttft_ms"] = ttft_ms
        self._log_metrics_once("local", question, t0, extra=tail_stats)


def main():
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    t = threading.Thread(target=_reindex_loop, daemon=True)
    t.start()
    log.info("vector-brain proxy listening on %s:%d -> %s", LISTEN_HOST, LISTEN_PORT, OLLAMA_BASE)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
