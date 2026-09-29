#!/usr/bin/env python3
"""
Vector's brain proxy.

Sits between wire-pod and Ollama (127.0.0.1:11434 on this host,
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
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - not expected on this host's Python
    ZoneInfo = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import Index, build_context_block, VAULT_ROOT  # noqa: E402

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 11500
OLLAMA_BASE = "http://127.0.0.1:11434"
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
TIMEZONE_NAME = "America/New_York"
WIREPOD_SDK_BASE = "http://127.0.0.1:8080"
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
# _time_battery_context_line below -- the time/battery facts stay (still
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


def _time_battery_context_line():
    """Builds the one-line system-message addition. Returns None on any
    error (fail-open) so it never blocks a chat completion."""
    try:
        now, bucket = _get_local_time()
        battery_level = _get_battery_level()
        time_str = now.strftime("%I:%M %p").lstrip("0") or now.strftime("%I:%M %p")
        parts = [f"Current time: {now.strftime('%A')} {time_str} ({TIMEZONE_NAME}), time-of-day: {bucket}."]
        if battery_level is not None:
            parts.append(f"Battery level: {battery_level}/3 (1=low, 3=full).")
        mood_line, is_sleepy = _mood_summary(time.time(), bucket, battery_level)
        parts.append(mood_line)
        if is_sleepy:
            parts.append(
                "Use a slower, yawn-y, winding-down tone (you may mention being sleepy). "
                "Do NOT add sad animations just because it's late -- only use sad when the "
                "content is actually sad."
            )
        return " ".join(parts)
    except Exception as e:
        log.debug("time/battery context build failed (fail-open): %s", e)
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
_CONVERSATION_LINE_RE = re.compile(
    r"^-\s*(\d{2}:\d{2})\s*\(NY time\)\s*\*\*the owner:\*\*\s*(.*?)\s*"
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
    lines = [l for l in text.splitlines() if l.startswith("- ") and "**the owner:**" in l]
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
# question to Hermes (a bigger remote LLM assistant, via a small HTTP
# receiver on a separate host) and speaks Hermes' answer instead. Fails open at every step: any
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

_escalation_lock = threading.Lock()
_last_escalation_ts = 0.0


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


def _is_escalation_candidate(question):
    """Heuristic: does this look like a factual/knowledge/current-events
    question the local model is likely to botch? Excludes chit-chat and
    robot-command phrasing so we only escalate real questions."""
    q = (question or "").strip()
    if not q or len(q) < 4:
        return False
    if _CHITCHAT_PATTERNS.search(q):
        return False
    if _COMMAND_KEYWORDS.search(q):
        return False
    if _ESCALATION_TOPIC_KEYWORDS.search(q):
        return True
    if "?" in q and _ESCALATION_QUESTION_WORDS.search(q) and len(q) > 12:
        return True
    return False


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


def _ask_hermes(question, timeout):
    """Calls the vector-ask receiver directly -- the same API
    ask_hermes.py (the wire-pod custom-intent handler) uses. Returns the
    reply string, or None on any failure/timeout/empty reply. Never logs
    the token."""
    env = _load_env_file(VECTOR_ENV_PATH)
    url = env.get("VECTOR_ASK_URL", HERMES_ASK_URL_DEFAULT)
    token = env.get("VECTOR_ASK_TOKEN", "")
    if not token:
        log.error("hermes escalation: VECTOR_ASK_TOKEN missing from %s", VECTOR_ENV_PATH)
        return None
    body = json.dumps({"question": question, "esn": ROBOT_SERIAL}).encode("utf-8")
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
        self._proxy_passthrough("GET")

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
        """Mutates payload['messages'] in place. Returns (included_labels, retrieval_ms)."""
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

        context_text, included, retrieval_ms = build_context_block(index, latest_user_text)

        # x_vector_internal=True (curious-vector's proactive-remark
        # generator, see curious.py's _generate_proactive_line) is Vector
        # asking its own brain for a line, not a real the owner voice turn --
        # skip mood tracking for it so a quiet room doesn't look "social"
        # just because the proactive-remark generator itself ran. RAG/
        # mood/same-day context is still injected below either way, since
        # that's exactly what makes the generated line good.
        if not payload.get("x_vector_internal"):
            _note_voice_turn(latest_user_text)

        recent_obs = _read_recent_observation()
        if recent_obs:
            obs_block = f"[curious-vector recent observation]\n{recent_obs}"
            context_text = f"{context_text}\n\n{obs_block}" if context_text else obs_block
            included = list(included) + ["curious-vector recent observation"]

        today_line = _today_conversation_context()
        if today_line:
            context_text = f"{context_text}\n\n{today_line}" if context_text else today_line
            included = list(included) + ["today's conversation so far"]

        tb_line = _time_battery_context_line()
        if tb_line:
            context_text = f"{context_text}\n\n{tb_line}" if context_text else tb_line
            included = list(included) + ["time/battery mood context"]

        if not context_text:
            return [], retrieval_ms

        # Insert right after the leading run of system messages, so wire-pod's
        # own system prompt + {{command}} instructions stay first/intact.
        insert_at = 0
        while insert_at < len(messages) and messages[insert_at].get("role") == "system":
            insert_at += 1
        messages.insert(insert_at, {"role": "system", "content": context_text})
        payload["messages"] = messages
        return included, retrieval_ms

    def _handle_chat_completions(self):
        t0 = time.time()
        raw_body = self._read_body()
        included, retrieval_ms = [], 0
        question = ""

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            log.warning("could not parse request JSON, forwarding raw: %s", e)
            payload = None

        body_to_send = raw_body
        if payload is not None:
            try:
                user_msgs = [m for m in payload.get("messages", []) if m.get("role") == "user"]
                question = str(user_msgs[-1].get("content", "") if user_msgs else "")[:300]
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
                for m in payload.get("messages", []):
                    c = m.get("content")
                    if c is None:
                        m["content"] = ""
                    elif isinstance(c, list):
                        if any(isinstance(x, dict) and x.get("type") == "image_url" for x in c):
                            has_image = True
                if has_image:
                    payload["model"] = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")
                    # 2026-09-29 (VRAM fix): force vision onto CPU. qwen3:32b
                    # alone uses ~29GB of the 32GB total across both GPUs, so
                    # loading vision on GPU evicts qwen3:32b (confirmed via
                    # ollama sched.go "evicting" log lines). Not latency
                    # critical on this path either -- ~20-30s CPU cost is fine.
                    payload.setdefault("options", {})
                    payload["options"]["num_gpu"] = int(os.environ.get("VISION_NUM_GPU", "0"))
                    # the owner 2026-09-27: he must be able to READ. Tell the vision model to
                    # read labels/brands/text exactly and never claim it can't read.
                    payload.setdefault("messages", []).insert(0, {"role": "system", "content":
                        "You are looking through your own camera. You CAN read. Always read any visible "
                        "text, brand names, model names and labels exactly and mention them in your answer. "
                        "Never say you can't read."})
                    log.info("image turn detected -> routing to %s", payload["model"])
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

        self.send_response(upstream.status)
        for k, v in upstream.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("X-Vector-Brain-Context", str(len(included)))
        self.end_headers()

        check_uncertainty = (
            HERMES_ESCALATION_ENABLED and is_stream and not has_image
            and payload is not None and len(question.strip()) >= 4
            and not _CHITCHAT_PATTERNS.search(question)
            and not _COMMAND_KEYWORDS.search(question)
        )

        try:
            if is_stream and check_uncertainty:
                self._relay_stream_with_uncertainty_check(upstream, question, t0)
            elif is_stream:
                while True:
                    line = upstream.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
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

        total_ms = int((time.time() - t0) * 1000)
        log.info("q=%r notes=%s retrieval_ms=%d total_ms=%d stream=%s",
                  question, included, retrieval_ms, total_ms, is_stream)

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

        try:
            self.wfile.write(_sse_chunk(final_text, finish_reason="stop"))
            self.wfile.write(_sse_done())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

        total_ms = int((time.time() - t0) * 1000)
        log.info("HERMES-ESCALATION q=%r source=%s hermes_replied=%s total_ms=%d",
                  question, source, bool(hermes_reply), total_ms)

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

        while not first_sentence_done:
            line = upstream.readline()
            if not line:
                break
            pending_lines.append(line)
            content = _extract_sse_delta_content(line)
            if content:
                buf_parts.append(content)
                joined = "".join(buf_parts)
                if _SENTENCE_END_RE.search(joined) or len(joined) > 200:
                    first_sentence_done = True

        first_sentence_text = "".join(buf_parts)

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
            hermes_reply = _ask_hermes(question, HERMES_ASK_TIMEOUT_SECONDS)
            final_text = hermes_reply or first_sentence_text.strip() or "I'm not sure, sorry."
            self.wfile.write(_sse_chunk(final_text, finish_reason="stop"))
            self.wfile.write(_sse_done())
            self.wfile.flush()
            total_ms = int((time.time() - t0) * 1000)
            log.info("HERMES-ESCALATION uncertainty q=%r source=%s total_ms=%d",
                      question, "hermes" if hermes_reply else "local-first-sentence", total_ms)
            return

        # Not uncertain (or rate-limited past its window): flush what we
        # buffered, then continue plain passthrough, unchanged.
        for line in pending_lines:
            self.wfile.write(line)
        self.wfile.flush()
        while True:
            line = upstream.readline()
            if not line:
                break
            self.wfile.write(line)
            self.wfile.flush()


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
