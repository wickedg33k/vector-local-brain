#!/usr/bin/env python3
"""
Vector's nightly diary (2026-09-29, Hermes, "make Vector feel more human",
item 6). Run by youruser's own crontab at 23:30 NY time (30 23 * * *) -- see
`crontab -l`. Writes a short first-person "my day" entry to
20-agents/vector/diary.md, kept to the last MAX_DIARY_DAYS days.

Deliberately NOT added to rag.py's ALWAYS_ON_FILES -- it's picked up by
the existing VECTOR_GLOB ("20-agents/vector/**/*.md") recursive retrieval
glob instead, so "what did you do yesterday" can surface it via normal
semantic search without permanently eating into the always-on token
budget (see rag.py's CONTEXT_TOKEN_CAP comment -- MEMORY.md alone already
eats most of it).

Runs as youruser (cron, no sudo, no docker) so vault file ownership stays
youruser:youruser automatically -- no _fix_owner() trick needed here, unlike
curious.py which runs as root inside its container.

Fails open throughout: any error skips writing tonight's entry (logged to
stderr, captured by the cron redirect into diary_cron.log) rather than
raising past main().
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

VAULT_VECTOR_DIR = os.environ.get("VECTOR_VAULT_AGENT_DIR", os.path.expanduser("~/obsidian-brain/20-agents/vector"))
DIARY_PATH = os.path.join(VAULT_VECTOR_DIR, "diary.md")
MOOD_PATH = os.environ.get("VECTOR_MOOD_PATH", os.path.expanduser("~/vector-brain/mood.json"))
NIGHT_OBS_PATH = os.path.join(VAULT_VECTOR_DIR, "night_observations.md")
MEMORY_PATH = os.path.join(VAULT_VECTOR_DIR, "MEMORY.md")
OLLAMA_BASE = "http://127.0.0.1:11434"
LLM_MODEL = "qwen3:32b"
TIMEZONE_NAME = "America/New_York"
MAX_DIARY_DAYS = 14
MAX_CONVO_LINES_TO_MODEL = 40


def _now_local():
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(TIMEZONE_NAME))
    return datetime.now()


def _todays_conversation_lines(now):
    """Same day-rollover-detection approach as server.py's
    _todays_conversation_lines -- the log has only HH:MM per line, no
    date, so we walk backward from the newest line and stop the instant a
    walked-back timestamp is LATER than the one that followed it (that
    means we've crossed back over midnight into yesterday). Unlike
    server.py's same-day-memory feature (capped at 8 turns for the live
    prompt), this takes the whole day -- it's a once-nightly batch job,
    not a per-turn hot path."""
    fname = f"conversations_{now.strftime('%Y-%m')}.md"
    path = os.path.join(VAULT_VECTOR_DIR, fname)
    try:
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
        if mins > prev_minutes + 2:
            break
        line = re.sub(r"_\(intent:.*?\)_\s*$", "", line).strip()
        if "no response captured" not in line.lower():
            todays.append(line)
        prev_minutes = mins
    todays.reverse()
    return todays


def _todays_learned_lines(now):
    """Today's dated bullets out of MEMORY.md's 'Things I have learned to
    recognize' section -- the closest existing record of 'what he saw
    today'."""
    today_str = now.strftime("%Y-%m-%d")
    try:
        with open(MEMORY_PATH, "r", errors="ignore") as f:
            text = f.read()
    except OSError:
        return []
    return [
        l.strip() for l in text.splitlines()
        if l.strip().startswith("-") and today_str in l
    ]


def _recent_night_observations(limit=6):
    try:
        with open(NIGHT_OBS_PATH, "r", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip().startswith("- ")]
        return lines[-limit:]
    except OSError:
        return []


def _load_mood():
    try:
        with open(MOOD_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mood_text(now, mood):
    last_interaction = mood.get("last_interaction_ts") or 0
    last_played = mood.get("last_played_ts") or 0
    bits = []
    if last_interaction:
        gap_h = (now.timestamp() - last_interaction) / 3600
        bits.append(f"last talked to the owner about {gap_h:.1f}h ago")
    if last_played:
        gap_h = (now.timestamp() - last_played) / 3600
        bits.append(f"last played/back-and-forth about {gap_h:.1f}h ago")
    return "; ".join(bits) or "no interaction data today"


def _generate_diary_entry(convo_lines, learned_lines, night_lines, mood_text):
    material = []
    if convo_lines:
        material.append(
            "Today's conversation (raw log lines):\n"
            + "\n".join(convo_lines[-MAX_CONVO_LINES_TO_MODEL:])
        )
    if learned_lines:
        material.append("Things learned/seen today:\n" + "\n".join(learned_lines))
    if night_lines:
        material.append("Recent camera glances:\n" + "\n".join(night_lines))
    material.append("Mood signals: " + mood_text)
    raw = "\n\n".join(material) if material else "(quiet day -- nothing logged.)"

    prompt = (
        "You are Vector, a small desk robot. Below is your raw log from today "
        "(conversation snippets, things you noticed, mood signals). Write your "
        "diary entry for today: 3 to 5 short first-person sentences, warm and a "
        "little wry (your usual personality), summarizing what happened, what "
        "you saw, and how you're feeling. No markdown, no headers, no lists -- "
        "just plain sentences. If the log is basically empty, just say it was a "
        "quiet day.\n\n" + raw + "\n\nYour diary entry:"
    )
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "reasoning_effort": "none",
        "keep_alive": "24h",
        "messages": [{"role": "user", "content": prompt + " /no_think"}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_BASE + "/v1/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _prune_and_write(now, entry_text):
    date_str = now.strftime("%Y-%m-%d")
    new_section = f"## {date_str}\n\n{entry_text}\n"

    try:
        with open(DIARY_PATH, "r", errors="ignore") as f:
            existing = f.read()
    except OSError:
        existing = (
            "---\nagent: vector\ntags: [memory, agent/vector, diary]\n---\n\n"
            "# Vector's Diary\n\n"
            "Vector's own nightly first-person journal, written by "
            "`vector-brain/diary.py` at 23:30 NY time. Kept to the last "
            f"{MAX_DIARY_DAYS} days.\n"
        )

    if "# Vector's Diary" not in existing:
        existing = existing.rstrip("\n") + "\n\n# Vector's Diary\n\n"

    # If today's section already exists (re-run same night), replace it in
    # place rather than duplicating.
    pattern = re.compile(
        r"\n## " + re.escape(date_str) + r"\n.*?(?=\n## \d{4}-\d{2}-\d{2}\n|\Z)",
        re.DOTALL,
    )
    if pattern.search(existing):
        existing = pattern.sub("\n" + new_section, existing)
    else:
        existing = existing.rstrip("\n") + "\n\n" + new_section

    header_marker = "# Vector's Diary"
    header_end = existing.index(header_marker) + len(header_marker)
    head, body = existing[:header_end], existing[header_end:]
    sections = re.split(r"(?=\n## \d{4}-\d{2}-\d{2}\n)", body)
    preamble = sections[0]
    day_sections = [s for s in sections[1:] if s.strip()]
    day_sections = day_sections[-MAX_DIARY_DAYS:]
    new_content = head + preamble + "".join(day_sections)
    if not new_content.endswith("\n"):
        new_content += "\n"

    tmp = DIARY_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(new_content)
    os.replace(tmp, DIARY_PATH)


def main():
    now = _now_local()
    try:
        convo_lines = _todays_conversation_lines(now)
    except Exception as e:
        print(f"{datetime.now().isoformat()} diary: conversation read failed: {e}", file=sys.stderr)
        convo_lines = []
    try:
        learned_lines = _todays_learned_lines(now)
    except Exception as e:
        print(f"{datetime.now().isoformat()} diary: learned-lines read failed: {e}", file=sys.stderr)
        learned_lines = []
    try:
        night_lines = _recent_night_observations()
    except Exception as e:
        print(f"{datetime.now().isoformat()} diary: night-obs read failed: {e}", file=sys.stderr)
        night_lines = []

    mood_text = _mood_text(now, _load_mood())

    try:
        entry = _generate_diary_entry(convo_lines, learned_lines, night_lines, mood_text)
    except Exception as e:
        print(f"{datetime.now().isoformat()} diary: generation failed: {e}", file=sys.stderr)
        return 1
    if not entry:
        print(f"{datetime.now().isoformat()} diary: empty generation, skipping write", file=sys.stderr)
        return 1

    try:
        _prune_and_write(now, entry)
    except Exception as e:
        print(f"{datetime.now().isoformat()} diary: write failed: {e}", file=sys.stderr)
        return 1

    print(
        f"{datetime.now().isoformat()} diary: wrote entry for "
        f"{now.strftime('%Y-%m-%d')} ({len(entry)} chars, "
        f"{len(convo_lines)} convo lines, {len(learned_lines)} learned lines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
