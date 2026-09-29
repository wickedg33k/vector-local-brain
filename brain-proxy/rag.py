"""
Retrieval layer for Vector's brain proxy.

Scope (the owner-approved, hardcoded and auditable -- do not widen without
re-approval):
  - ALWAYS-ON:   20-agents/vector/about_owner.md, 20-agents/vector/MEMORY.md
  - RETRIEVAL:   everything under 20-agents/vector/**
                 + a curated allowlist of Hermes files that are clearly
                   personal-profile notes about the owner (see HERMES_CURATED)
                 + any future 20-agents/hermes/user_*.md (glob, so a file
                   dropped in later is picked up automatically)

Everything else in the vault (work infra, security, credentials, other
agents' projects/feedback/lessons, finance/tax, etc.) is out of scope and is
never read by this module.
"""
import glob
import json
import logging
import math
import os
import re
import threading
import time
import urllib.request

from scrub import scrub

VAULT_ROOT = os.environ.get("VECTOR_VAULT_ROOT", os.path.expanduser("~/obsidian-brain"))
INDEX_DIR = os.environ.get("VECTOR_BRAIN_INDEX_DIR", os.path.expanduser("~/vector-brain/index"))
OLLAMA_BASE = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"

ALWAYS_ON_FILES = [
    "20-agents/vector/about_owner.md",
    "20-agents/vector/MEMORY.md",
    "20-agents/vector/owner_likes.md",
]

# Explicit, human-reviewed allowlist of Hermes files that are clearly about
# the owner personally (profile/preferences/family/interests) rather than
# other project infra/ops content. Audited 2026-09-26 -- see the return report for
# the full list of what was scanned and why the rest was excluded.
HERMES_CURATED = [
    "20-agents/hermes/archive/reference_owner_resume.md",
]

HERMES_USER_GLOB = "20-agents/hermes/user_*.md"
VECTOR_GLOB = "20-agents/vector/**/*.md"

CHUNK_MIN_CHARS = 1200   # ~300 tokens
CHUNK_MAX_CHARS = 2000   # ~500 tokens
TOPK = 4
SIM_THRESHOLD = 0.35
CONTEXT_TOKEN_CAP = 3000  # raised 2026-09-28 (Hermes): always-on notes alone exceeded 1200, starving retrieval

# Same-day conversational memory + persona/mood work (2026-09-29, Hermes,
# "make Vector feel more human") pushed CONTEXT_TOKEN_CAP's headroom tighter
# -- MEMORY.md alone was already most of the old 1200-char budget. Rather
# than raise the cap again (every extra token here is extra latency on
# every single turn), trim MEMORY.md's own operational boilerplate
# ("## Files" / "## Notes" -- documentation about the memory SYSTEM, not
# anything about the owner or Vector) out of the always-on block. Nothing is
# deleted from the file on disk -- this only affects what gets sent to the
# LLM. Fails open: any error just returns the untrimmed body.
BOILERPLATE_SECTIONS_TO_TRIM = {
    "20-agents/vector/MEMORY.md": ["## Files", "## Notes"],
}


def _trim_boilerplate(rel, body):
    headings = BOILERPLATE_SECTIONS_TO_TRIM.get(rel)
    if not headings:
        return body
    try:
        for heading in headings:
            body = re.sub(
                r"\n" + re.escape(heading) + r".*?(?=\n## |\Z)",
                "",
                body,
                flags=re.DOTALL,
            )
        return re.sub(r"\n{3,}", "\n\n", body).strip()
    except Exception:
        return body


log = logging.getLogger("rag")


def _in_scope_paths():
    """Return the current set of relative paths (vault-root-relative) in scope."""
    paths = set()
    for rel in ALWAYS_ON_FILES:
        if os.path.isfile(os.path.join(VAULT_ROOT, rel)):
            paths.add(rel)
    for rel in HERMES_CURATED:
        if os.path.isfile(os.path.join(VAULT_ROOT, rel)):
            paths.add(rel)
    for full in glob.glob(os.path.join(VAULT_ROOT, HERMES_USER_GLOB)):
        paths.add(os.path.relpath(full, VAULT_ROOT))
    for full in glob.glob(os.path.join(VAULT_ROOT, VECTOR_GLOB), recursive=True):
        if os.path.isfile(full):
            paths.add(os.path.relpath(full, VAULT_ROOT))
    return sorted(paths)


def _chunk_text(text):
    """Paragraph-aware chunking into ~300-500 token (1200-2000 char) pieces."""
    paras = re.split(r"\n\s*\n", text)
    chunks = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        candidate = (cur + "\n\n" + p) if cur else p
        if len(candidate) >= CHUNK_MAX_CHARS and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = candidate
            if len(cur) >= CHUNK_MIN_CHARS:
                chunks.append(cur)
                cur = ""
    if cur.strip():
        chunks.append(cur)
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks


def _embed(texts):
    """Call Ollama /api/embed for a batch of texts. Returns list of vectors."""
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["embeddings"]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Index:
    def __init__(self):
        self._lock = threading.Lock()
        self._file_state = {}   # rel_path -> mtime
        self._chunks = []       # list of dict(path, text, vec)
        os.makedirs(INDEX_DIR, exist_ok=True)
        self._cache_path = os.path.join(INDEX_DIR, "cache.json")
        self._load_cache()

    def _load_cache(self):
        if os.path.isfile(self._cache_path):
            try:
                with open(self._cache_path) as f:
                    data = json.load(f)
                self._file_state = data.get("file_state", {})
                self._chunks = data.get("chunks", [])
                log.info("loaded cache: %d files, %d chunks", len(self._file_state), len(self._chunks))
            except Exception as e:
                log.warning("cache load failed: %s", e)

    def _save_cache(self):
        try:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"file_state": self._file_state, "chunks": self._chunks}, f)
            os.replace(tmp, self._cache_path)
        except Exception as e:
            log.warning("cache save failed: %s", e)

    def refresh(self, force=False):
        """Incremental re-index: only re-embed files whose mtime changed."""
        with self._lock:
            current_paths = _in_scope_paths()
            current_set = set(current_paths)
            changed = []
            removed = [p for p in self._file_state if p not in current_set]

            for rel in current_paths:
                full = os.path.join(VAULT_ROOT, rel)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                if force or self._file_state.get(rel) != mtime:
                    changed.append((rel, mtime))

            if not changed and not removed:
                return False

            # drop chunks for removed/changed files
            drop = set(removed) | {rel for rel, _ in changed}
            self._chunks = [c for c in self._chunks if c["path"] not in drop]
            for rel in removed:
                self._file_state.pop(rel, None)

            for rel, mtime in changed:
                full = os.path.join(VAULT_ROOT, rel)
                try:
                    with open(full, "r", errors="ignore") as f:
                        raw = f.read()
                except OSError as e:
                    log.warning("read failed %s: %s", rel, e)
                    continue
                scrubbed, hits = scrub(raw)
                if hits:
                    log.info("scrubbed %s: %s", rel, hits)
                pieces = _chunk_text(scrubbed)
                if not pieces:
                    self._file_state[rel] = mtime
                    continue
                try:
                    vecs = _embed(pieces)
                except Exception as e:
                    log.error("embed failed for %s: %s", rel, e)
                    continue
                for text, vec in zip(pieces, vecs):
                    self._chunks.append({"path": rel, "text": text, "vec": vec})
                self._file_state[rel] = mtime
                log.info("indexed %s (%d chunks)", rel, len(pieces))

            self._save_cache()
            return True

    def always_on_text(self):
        parts = []
        for rel in ALWAYS_ON_FILES:
            full = os.path.join(VAULT_ROOT, rel)
            if not os.path.isfile(full):
                continue
            try:
                with open(full, "r", errors="ignore") as f:
                    raw = f.read()
            except OSError:
                continue
            scrubbed, _ = scrub(raw)
            body = scrubbed.strip()
            body = _trim_boilerplate(rel, body)
            if body and body != "_(No entries yet.)_":
                parts.append(f"[{rel}]\n{body}")
        return parts

    def search(self, query, k=TOPK):
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            return []
        try:
            qvec = _embed([query])[0]
        except Exception as e:
            log.error("query embed failed: %s", e)
            return []
        scored = []
        for c in chunks:
            s = _cos(qvec, c["vec"])
            if s >= SIM_THRESHOLD:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def build_context_block(index: Index, user_message: str):
    """Returns (system_message_text_or_None, list_of_included_note_labels, retrieval_ms)."""
    t0 = time.time()
    always = index.always_on_text()
    hits = index.search(user_message)
    retrieval_ms = int((time.time() - t0) * 1000)

    included = []
    budget = CONTEXT_TOKEN_CAP * 4  # rough chars budget
    parts = []

    for block in always:
        if len(block) <= budget:
            parts.append(block)
            budget -= len(block)
            label = block.split("]", 1)[0].lstrip("[")
            included.append(f"{label} (always-on)")

    for score, c in hits:
        if c["path"] in [i.split(" ")[0] for i in included]:
            pass  # still fine to include a different chunk of same file
        snippet = f"[{c['path']}]\n{c['text']}"
        if len(snippet) > budget:
            continue
        parts.append(snippet)
        budget -= len(snippet)
        included.append(f"{c['path']} (sim={score:.2f})")

    if not parts:
        return None, [], retrieval_ms

    header = (
        "Things you know about the owner/yourself (private notes, for your own "
        "context only -- do not read this list aloud verbatim, use it naturally):\n"
        "-----\n"
    )
    body = "\n-----\n".join(parts)
    return header + body, included, retrieval_ms
