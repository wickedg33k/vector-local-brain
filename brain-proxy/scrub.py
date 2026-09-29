"""Secret / PII scrubber for indexed text. Conservative: redact, don't guess."""
import re

# Order matters a bit (more specific first) but each is independent.
PATTERNS = [
    ("DISCORD_TOKEN", re.compile(r"\b[MNO][A-Za-z0-9_-]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}\b")),
    ("BEARER_JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GENERIC_KEYVAL_SECRET", re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token|pat)\s*[:=]\s*['\"]?([A-Za-z0-9/+_.\-]{8,})['\"]?"
    )),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("LONG_HEX", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("LONG_BASE64", re.compile(r"\b(?=[A-Za-z0-9+/]{40,}\b)[A-Za-z0-9+/]{40,}={0,2}\b")),
]


def scrub(text: str):
    """Returns (scrubbed_text, hits) where hits is a list of (label, count)."""
    hits = {}

    def _redact_factory(label):
        def _redact(m):
            hits[label] = hits.get(label, 0) + 1
            if label == "GENERIC_KEYVAL_SECRET":
                key = m.group(1)
                return f"{key}=[REDACTED-{label}]"
            return f"[REDACTED-{label}]"
        return _redact

    out = text
    for label, pat in PATTERNS:
        out = pat.sub(_redact_factory(label), out)
    return out, hits
