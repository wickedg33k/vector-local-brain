"""
Vector Dashboard v1 — read-only local web app showing what the owner's Anki
Vector robot is doing, hearing, seeing and learning.

All data sources are read from :ro bind mounts under /data (except the
snapshots dir, which is the one documented exception). No writes ever
happen to any mount from this process except serving files back over HTTP.
"""
import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths (all read-only bind mounts, see docker run / Dockerfile notes)
# ---------------------------------------------------------------------------
HEALTH_LOG = Path("/data/health/health.log")
VECTOR_AGENT_DIR = Path("/data/vector-agent")
MEMORY_MD = VECTOR_AGENT_DIR / "MEMORY.md"
NIGHT_OBS_MD = VECTOR_AGENT_DIR / "night_observations.md"
TRAINING_DIR = Path("/data/training")
LESSONS_JSONL = TRAINING_DIR / "lessons.jsonl"
EVAL_LOG = TRAINING_DIR / "eval.log"
VECTOR_LOG = Path("/data/logs/vector.log")
CURIOUS_LOG = Path("/data/cache/curious.log")
SNAPSHOTS_DIR = Path("/data/snapshots")

WIREPOD_BATTERY_URL = "http://localhost:8080/api-sdk/get_battery"
VECTOR_SERIAL = os.environ.get("VECTOR_SERIAL", "YOUR_ESN")

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_tail_lines(path: Path, max_lines: int = 400, max_bytes: int = 2_000_000):
    """Read up to `max_bytes` from the end of a (possibly huge) file and
    return up to `max_lines` decoded lines. Never loads the whole file."""
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop partial first line
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _read_text(path: Path, max_bytes: int = 500_000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _latest_conversations_file() -> Optional[Path]:
    files = sorted(glob.glob(str(VECTOR_AGENT_DIR / "conversations_*.md")))
    if not files:
        return None
    return Path(files[-1])


# ---------------------------------------------------------------------------
# Section parsers — each is defensive: a bad/missing file returns an empty
# result instead of raising, so one broken source never blanks the page.
# ---------------------------------------------------------------------------

HEALTH_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"loss=(?P<loss>\S+) rtt=(?P<rtt>\S+) pod=(?P<pod>\S+) brain=(?P<brain>\S+) "
    r"voice_reqs_5m=(?P<voice>\S+) container=(?P<container>\S+)\s*$"
)


def parse_health():
    lines = _safe_tail_lines(HEALTH_LOG, max_lines=60, max_bytes=20_000)
    history = []
    for line in lines:
        m = HEALTH_RE.match(line.strip())
        if not m:
            continue
        history.append(m.groupdict())
    latest = history[-1] if history else None

    online = False
    age_seconds = None
    if latest:
        try:
            ts = datetime.strptime(latest["ts"], "%Y-%m-%d %H:%M:%S")
            age_seconds = (datetime.now() - ts).total_seconds()
            online = age_seconds < 180 and latest["pod"] == "200"
        except Exception:
            pass

    return {
        "latest": latest,
        "online": online,
        "age_seconds": age_seconds,
        "history": history[-20:],
    }


_battery_cache = {"data": None, "ts": 0.0, "error": None}
BATTERY_MIN_INTERVAL = 60.0  # robot is fragile -- never poll more than 1x/min


def get_battery_cached():
    now = time.time()
    if _battery_cache["data"] is not None and (now - _battery_cache["ts"]) < BATTERY_MIN_INTERVAL:
        return _battery_cache["data"], _battery_cache["error"], False
    try:
        resp = httpx.get(
            WIREPOD_BATTERY_URL, params={"serial": VECTOR_SERIAL}, timeout=5.0
        )
        resp.raise_for_status()
        raw = resp.json()
        # NOTE: wire-pod omits false booleans entirely -- missing means False.
        data = {
            "battery_level": raw.get("battery_level"),
            "battery_volts": raw.get("battery_volts"),
            "is_charging": raw.get("is_charging", False),
            "is_on_charger_platform": raw.get("is_on_charger_platform", False),
            "cube_battery": raw.get("cube_battery"),
            "status_code": (raw.get("status") or {}).get("code"),
        }
        _battery_cache.update(data=data, ts=now, error=None)
        return data, None, True
    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not fatal
        err = str(exc)
        _battery_cache["error"] = err
        # Keep serving last-known-good data if we have it.
        return _battery_cache["data"], err, False


CONV_RE = re.compile(
    r"^-\s*(?P<time>\d{2}:\d{2})\s*\(NY time\)\s*\*\*the owner:\*\*\s*(?P<the owner>.*?)\s*"
    r"→\s*\*\*Vector:\*\*\s*(?P<vector>.*?)\s*"
    r"_\(intent:\s*(?P<intent>[^;)]*)(?:;\s*animations:\s*(?P<animations>[^)]*))?\)_\s*$"
)


def parse_conversations(limit: int = 25):
    f = _latest_conversations_file()
    if not f:
        return {"file": None, "entries": []}
    lines = _safe_tail_lines(f, max_lines=2000, max_bytes=800_000)
    entries = []
    for line in lines:
        line = line.strip()
        if not line.startswith("- "):
            continue
        m = CONV_RE.match(line)
        if not m:
            # keep raw so nothing silently disappears
            entries.append({
                "time": None, "the owner": None, "vector": line,
                "intent": None, "animations": None, "no_response": False,
                "raw": True,
            })
            continue
        vector_text = m.group("vector")
        entries.append({
            "time": m.group("time"),
            "the owner": m.group("the owner"),
            "vector": vector_text,
            "intent": m.group("intent"),
            "animations": m.group("animations"),
            "no_response": "no response captured" in vector_text.lower(),
            "raw": False,
        })
    entries.reverse()  # latest first
    return {"file": f.name, "entries": entries[:limit]}


def parse_learned_recognize():
    text = _read_text(MEMORY_MD)
    m = re.search(
        r"##\s*Things I have learned to recognize\s*\n(.*?)(?=\n##\s|\Z)",
        text, re.S,
    )
    if not m:
        return []
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip().startswith("-")]
    return lines


def parse_corrections():
    text = _read_text(MEMORY_MD)
    m = re.search(
        r"##\s*Corrections from Hermes\s*\n(.*?)(?=\n##\s|\Z)",
        text, re.S,
    )
    if not m:
        return []
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip().startswith("-")]
    return lines


def parse_night_observations(limit: int = 15):
    lines = _safe_tail_lines(NIGHT_OBS_MD, max_lines=limit, max_bytes=200_000)
    return [l.strip() for l in lines if l.strip().startswith("-")]


CURIOUS_PATTERNS = ("vision description:", "vision objects:", "generated question:")


def parse_curious_log(limit: int = 20):
    lines = _safe_tail_lines(CURIOUS_LOG, max_lines=4000, max_bytes=1_000_000)
    out = []
    for line in lines:
        low = line.lower()
        if any(p in low for p in CURIOUS_PATTERNS):
            out.append(line.strip())
    out.reverse()
    return out[:limit]


LESSON_TOPIC_DIR = VECTOR_AGENT_DIR / "lessons"


def parse_learning():
    lessons = []
    count = 0
    if LESSONS_JSONL.exists():
        lines = _safe_tail_lines(LESSONS_JSONL, max_lines=200_000, max_bytes=5_000_000)
        count = len(lines)
        for line in lines[-5:]:
            try:
                lessons.append(json.loads(line))
            except Exception:
                continue
        lessons.reverse()

    topics = []
    if LESSON_TOPIC_DIR.exists():
        try:
            topics = sorted(
                p.stem.replace("-", " ")
                for p in LESSON_TOPIC_DIR.glob("*.md")
            )
        except Exception:
            topics = []

    return {
        "lesson_count": count,
        "latest_lessons": lessons,
        "topics": topics,
        "corrections": parse_corrections(),
    }


SCORE_RE = re.compile(
    r"^(?P<ts>\S+)\s+SCORE\s+(?P<round>round\d+):\s*(?P<num>\d+)/(?P<den>\d+)\s*(?P<notes>.*)$"
)


def parse_eval_scores(limit: int = 12):
    lines = _safe_tail_lines(EVAL_LOG, max_lines=5000, max_bytes=1_000_000)
    scores = []
    for line in lines:
        m = SCORE_RE.match(line.strip())
        if not m:
            continue
        scores.append({
            "ts": m.group("ts"),
            "round": m.group("round"),
            "num": int(m.group("num")),
            "den": int(m.group("den")),
            "notes": m.group("notes").strip(),
        })
    scores.reverse()
    return scores[:limit]


FAULT_PATTERNS = (
    "handle fault code",
    "fault-code-handler",
    "restart anki-robot",
    "@behavior.voice_command.dropped",
)


def parse_faults(limit: int = 20):
    # vector.log is a large syslog stream -- only ever read the tail.
    lines = _safe_tail_lines(VECTOR_LOG, max_lines=20000, max_bytes=3_000_000)
    out = []
    for line in lines:
        low = line.lower()
        if any(p in low for p in FAULT_PATTERNS):
            out.append(line.strip())
    out.reverse()
    return out[:limit]


def list_snapshots(limit: int = 24):
    if not SNAPSHOTS_DIR.exists():
        return []
    try:
        files = [
            p for p in SNAPSHOTS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    except Exception:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:limit]:
        sidecar = p.with_suffix(".json")
        meta = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except Exception:
                meta = {}
        out.append({
            "file": p.name,
            "mtime": p.stat().st_mtime,
            "description": meta.get("description"),
            "objects": meta.get("objects"),
        })
    return out


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Vector Dashboard")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/now")
def api_now():
    health = parse_health()
    battery, battery_err, fresh = get_battery_cached()
    return JSONResponse({
        "server_time": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "battery": battery,
        "battery_error": battery_err,
        "battery_fresh": fresh,
    })


@app.get("/api/health")
def api_health():
    return JSONResponse(parse_health())


@app.get("/api/battery")
def api_battery():
    battery, err, fresh = get_battery_cached()
    return JSONResponse({"battery": battery, "error": err, "fresh": fresh})


@app.get("/api/conversations")
def api_conversations():
    return JSONResponse(parse_conversations())


@app.get("/api/saw")
def api_saw():
    return JSONResponse({
        "night_observations": parse_night_observations(),
        "learned_to_recognize": parse_learned_recognize(),
        "curious_log": parse_curious_log(),
    })


@app.get("/api/learning")
def api_learning():
    return JSONResponse(parse_learning())


@app.get("/api/eval")
def api_eval():
    return JSONResponse({"scores": parse_eval_scores()})


@app.get("/api/faults")
def api_faults():
    return JSONResponse({"faults": parse_faults()})


@app.get("/api/snapshots")
def api_snapshots():
    return JSONResponse({"snapshots": list_snapshots()})


@app.get("/snapshots/{filename}")
def get_snapshot(filename: str):
    # Defensive path handling -- filename only, no traversal.
    safe_name = os.path.basename(filename)
    path = SNAPSHOTS_DIR / safe_name
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/healthz")
def healthz():
    return {"ok": True}
