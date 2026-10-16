"""
Retrieval layer for Vector's brain proxy.

Scope (owner-approved, hardcoded and auditable -- do not widen without
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
OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "nomic-embed-text"

ALWAYS_ON_FILES = [
    "20-agents/vector/about_owner.md",
    "20-agents/vector/MEMORY.md",
    "20-agents/vector/owner_likes.md",
]

# Explicit, human-reviewed allowlist of Hermes files that are clearly about
# the owner personally (profile/preferences/family/interests) rather than
# the team/infra/ops content. Audited 2026-09-26 -- see the return report for
# the full list of what was scanned and why the rest was excluded.
HERMES_CURATED = [
    "20-agents/hermes/archive/reference_owner_resume.md",
]

HERMES_USER_GLOB = "20-agents/hermes/user_*.md"
VECTOR_GLOB = "20-agents/vector/**/*.md"

CHUNK_MIN_CHARS = 1200   # ~300 tokens -- target floor when paragraph-splitting an oversized section
CHUNK_MAX_CHARS = 2000   # ~500 tokens -- ceiling; sections bigger than this get paragraph-split
TOPK = 4
SIM_THRESHOLD = 0.35

# ---- Chunk-quality fix (2026-09-30, Hermes, "fix retrieval quality").
# Root cause (see training/eval.log rounds 68/72): the old chunker
# (_chunk_text, now _split_oversized_body below) only did paragraph-aware
# merging up to CHUNK_MIN_CHARS/CHUNK_MAX_CHARS with no awareness of
# markdown structure. Hobby/lesson files grew to 15-20 "## " sections
# over many training rounds, and a single 1200-2000 char chunk would span
# 2-4 unrelated sections (e.g. a "places worth knowing" section holding one
# specific fact, merged with neighboring sections on related techniques).
# The query embedding for a narrow question about that one fact then had
# to compete against a chunk whose vector was an average over several
# topics, so it didn't win top-K, and qwen3 confabulated an answer instead
# of saying "don't know."
#
# Fix: chunk on markdown structure FIRST (# / ## headings), not on raw
# character count. Every "## " (and top-level "# ") section becomes its own
# chunk, with "<file title> -- <section heading>: " prepended so the
# embedding and the text shown to the model both carry which topic/section
# this is from even after retrieval strips it from its file. Only two
# fallbacks still use character counts:
#   SECTION_MERGE_CHARS -- a section this small (e.g. a heading with one
#     short bullet) is merged into a neighbor instead of standing alone as
#     a near-empty, low-signal chunk.
#   CHUNK_MAX_CHARS -- a section bigger than this (e.g. files with no "##"
#     headings at all, like night_observations.md) still gets the old
#     paragraph-aware split so no single chunk balloons past the retrieval
#     budget.
SECTION_MERGE_CHARS = 200

# ---- Latency work (2026-09-29, speed pass, Hermes). Two separate budgets
# instead of one shared CONTEXT_TOKEN_CAP:
#   ALWAYS_ON_TOKEN_CAP  -- the static-notes tier. This text is now short
#     (~1100-1200 tokens after the MEMORY.md trim below) and is meant to be
#     BYTE-IDENTICAL across every request so Ollama's prompt/KV cache can
#     reuse it turn to turn (see build_static_block). The cap here is just
#     a safety ceiling in case the vault files grow -- it should rarely bind.
#   RETRIEVAL_TOKEN_CAP  -- the per-question RAG-hit tier. This is
#     inherently different on every turn (different question -> different
#     nearest chunks), so it's kept in its OWN system message, positioned
#     AFTER the static/slow tiers (see server.py's _inject_context), so a
#     changing RAG block never breaks the cache for the static prefix ahead
#     of it. Lowered from the old combined CONTEXT_TOKEN_CAP=3000 (Hermes,
#     2026-09-28) now that always-on no longer eats into the same budget --
#     3000 was sized to cover both tiers together and was starving
#     retrieval; 1500 is plenty for TOPK=4 chunks and cuts prefill on any
#     cache-miss turn by ~1500 tokens vs the old combined cap.
ALWAYS_ON_TOKEN_CAP = 2200
RETRIEVAL_TOKEN_CAP = 1500

# Kept for anything that still imports the old name.
CONTEXT_TOKEN_CAP = RETRIEVAL_TOKEN_CAP

# Same-day conversational memory + persona/mood work (2026-09-29, Hermes,
# "make Vector feel more human") pushed the old shared budget's headroom
# tighter -- MEMORY.md alone was already most of the old 1200-char budget.
# Speed pass (2026-09-29, Hermes, "cut voice latency"): rather than just
# trim boilerplate ("## Files"/"## Notes" -- documentation about the memory
# SYSTEM, not anything about the owner or Vector), also move the two biggest,
# most day-to-day-noisy sections -- the raw desk-object recognition log and
# the full Hermes-corrections log -- OUT of always-on. Nothing is deleted
# from the file on disk, and both sections stay fully in scope for RAG
# (MEMORY.md is still indexed whole via VECTOR_GLOB), so a question that
# actually needs a specific correction still finds it by similarity search
# -- it just isn't re-sent on EVERY turn regardless of relevance. This cuts
# MEMORY.md's always-on footprint from ~6.3KB to ~1.7KB (about 1150 fewer
# always-on tokens), which both shrinks the byte-identical cacheable prefix
# (faster to reprocess on any cache-miss turn) and frees real budget for
# RAG. "## What I learned recently" is kept always-on on purpose -- it's
# short and it's the direct answer to "what did you learn last night",
# a question the owner asks often enough that it shouldn't depend on retrieval
# similarity scoring correctly every time. Fails open: any error here just
# returns the untrimmed body.
BOILERPLATE_SECTIONS_TO_TRIM = {
    "20-agents/vector/MEMORY.md": [
        "## Files",
        "## Notes",
        "## Things I have learned to recognize",
        "## Corrections from Hermes",
    ],
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


def _clean_title_from_filename(rel):
    """Fallback title for files with no top-level '# ' heading -- derive a
    human-readable title from the filename (night_observations.md ->
    'Night Observations')."""
    base = os.path.splitext(os.path.basename(rel))[0]
    words = [w for w in re.split(r"[-_]+", base) if w]
    return " ".join(w if w.isupper() else w.capitalize() for w in words) or rel


def _clean_title(title):
    """Strip the repetitive 'Lesson: ... (from Hermes)' wrapper Hermes'
    lesson files all share, so the prepended chunk prefix reads like
    'Palworld -- Fun Pal facts: ...' instead of 'Lesson: Palworld (from
    Hermes) -- Fun Pal facts: ...'. Harmless no-op on titles that don't
    match (about_owner.md, MEMORY.md, etc. keep their own H1 as-is)."""
    if not title:
        return title
    t = re.sub(r"^Lesson:\s*", "", title.strip())
    t = re.sub(r"\s*\(from Hermes\)\s*$", "", t)
    return t.strip()


def _split_by_heading(text):
    """Split a markdown file's text into (title, [(heading, body), ...]).

    - A leading YAML frontmatter block ('---\\n...\\n---\\n') is stripped
      before parsing, so its keys never leak into a chunk.
    - If the very first line is a top-level '# ' heading, it becomes the
      file's title (not a section of its own).
    - Everything else is split on '# '/'## ' heading lines. Content before
      the first heading (or the whole file, if it has no headings) comes
      back as one section with heading=''.
    """
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)
    text = text.lstrip("\n")

    title = None
    h1 = re.match(r"^#[ \t]+(.+?)\s*$", text, re.MULTILINE)
    if h1 and h1.start() == 0:
        title = h1.group(1).strip()
        text = text[h1.end():]

    heading_re = re.compile(r"^#{1,2}[ \t]+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    sections = []
    if not matches:
        body = text.strip()
        if body:
            sections.append(("", body))
        return title, sections

    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append(("", pre))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    return title, sections


def _merge_small_sections(sections):
    """Fold sections smaller than SECTION_MERGE_CHARS into a neighbor so a
    stray one-liner heading doesn't become its own near-empty, low-signal
    chunk. Merges backward into the previous kept section when possible,
    otherwise forward into the next one (e.g. a tiny first section)."""
    merged = []
    for heading, body in sections:
        if merged and len(body) < SECTION_MERGE_CHARS:
            prev_heading, prev_body = merged[-1]
            addition = (f"({heading}) " if heading else "") + body
            merged[-1] = (prev_heading, (prev_body + "\n\n" + addition) if prev_body else addition)
        else:
            merged.append((heading, body))

    if len(merged) >= 2 and len(merged[0][1]) < SECTION_MERGE_CHARS:
        h0, b0 = merged[0]
        h1, b1 = merged[1]
        addition = (f"({h0}) " if h0 else "") + b0
        merged[1] = (h1, (addition + "\n\n" + b1) if b1 else addition)
        merged = merged[1:]

    return merged


def _split_oversized_body(body):
    """Paragraph-aware split of one section's body into <=CHUNK_MAX_CHARS
    pieces (>=CHUNK_MIN_CHARS where possible). Used both for sections that
    grew too large on their own, and (via the no-heading fallback in
    _chunk_markdown) for whole files that have no '#'/'##' structure at
    all. This is the same greedy paragraph-merge the old flat chunker used
    -- kept as-is, just scoped to one section instead of the whole file.

    Falls back to splitting on single newlines when the body has no
    blank-line paragraph breaks at all (e.g. one-bullet-per-line logs like
    conversations_*.md/night_observations.md) -- otherwise re.split on
    blank lines returns the entire body as a single "paragraph" and the
    section never actually splits."""
    if len(body) <= CHUNK_MAX_CHARS:
        return [body]
    paras = re.split(r"\n\s*\n", body)
    if len(paras) <= 1:
        paras = body.split("\n")
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
    if not chunks:
        chunks = [body]
    return chunks


def _chunk_markdown(rel, text):
    """Heading-first chunking (2026-09-30 fix -- see the comment above
    SECTION_MERGE_CHARS). Splits on '#'/'##' headings so each chunk is one
    coherent section instead of a character-count window that can span
    several unrelated topics. Each chunk is prefixed with
    '<file title> -- <section heading>: ' for embedding + display context.
    """
    raw_title, sections = _split_by_heading(text)
    if not sections:
        return []
    title = _clean_title(raw_title) or _clean_title_from_filename(rel)
    sections = _merge_small_sections(sections)

    pieces = []
    for heading, body in sections:
        prefix = f"{title} -- {heading}: " if heading else f"{title}: "
        for part in _split_oversized_body(body):
            pieces.append(prefix + part)
    return pieces


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
                pieces = _chunk_markdown(rel, scrubbed)
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


# ---- Speed pass (2026-09-29, Hermes): build_context_block used to return
# ONE combined system-message string mixing always-on notes (static) with
# per-question RAG hits (dynamic) in a single blob. That meant the dynamic
# part broke the shared prefix for EVERY turn, even though the always-on
# part alone is byte-identical turn to turn. Split into two independently
# budgeted, independently built blocks so server.py can send them as two
# separate system messages in a fixed STATIC-then-DYNAMIC order -- see
# _inject_context in server.py for the full static -> slow -> dynamic
# message ordering this enables. build_context_block() is kept as a thin
# compatibility wrapper (nothing outside this repo imports it, but cheap
# to keep) that just concatenates the two new blocks the old way.

def static_block(index):
    """Always-on tier only. Returns (text_or_None, included_labels).
    Deterministic and independent of the question -- this is the part
    meant to be byte-identical across requests for KV-cache reuse."""
    always = index.always_on_text()
    budget = ALWAYS_ON_TOKEN_CAP * 4
    included = []
    kept = []
    for block in always:
        if len(block) <= budget:
            kept.append(block)
            budget -= len(block)
            label = block.split("]", 1)[0].lstrip("[")
            included.append(f"{label} (always-on)")
    if not kept:
        return None, []
    header = (
        "Things you always know about the owner/yourself (private notes, for "
        "your own context only -- do not read this list aloud verbatim, "
        "use it naturally):\n-----\n"
    )
    return header + "\n-----\n".join(kept), included


def retrieval_block(index, user_message, budget_tokens=None):
    """Per-question RAG-hit tier only. Returns (text_or_None,
    included_labels, retrieval_ms). Nothing here is byte-stable across
    requests -- it's meant to sit AFTER the static/slow tiers in the
    prompt so it never breaks their shared prefix."""
    t0 = time.time()
    hits = index.search(user_message)
    retrieval_ms = int((time.time() - t0) * 1000)

    budget = (budget_tokens if budget_tokens is not None else RETRIEVAL_TOKEN_CAP) * 4
    included = []
    parts = []
    for score, c in hits:
        snippet = f"[{c['path']}]\n{c['text']}"
        if len(snippet) > budget:
            continue
        parts.append(snippet)
        budget -= len(snippet)
        included.append(f"{c['path']} (sim={score:.2f})")

    if not parts:
        return None, [], retrieval_ms

    header = (
        "Things that might help answer this specific question (private "
        "notes, for your own context only -- do not read this list aloud "
        "verbatim, use it naturally):\n-----\n"
    )
    return header + "\n-----\n".join(parts), included, retrieval_ms


def build_context_block(index: Index, user_message: str):
    """Compatibility wrapper: old callers get the old combined-string
    shape back. New code (server.py) should call static_block() and
    retrieval_block() separately instead -- see the comment above."""
    s_text, s_included = static_block(index)
    r_text, r_included, retrieval_ms = retrieval_block(index, user_message)
    parts = [t for t in (s_text, r_text) if t]
    if not parts:
        return None, [], retrieval_ms
    return "\n\n".join(parts), (s_included + r_included), retrieval_ms
