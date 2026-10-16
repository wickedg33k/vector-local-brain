#!/usr/bin/env python3
"""
ytm_common.py -- shared helpers for the owner's YouTube Music account integration
(liked songs / named playlists / mixes) used by both ytm_auth.py (device-code
sign-in) and play.py (playback).

No password handling anywhere in this module. Auth is Google OAuth device-code
only: a Google Cloud OAuth client of type "TVs and Limited Input devices"
(client_id/secret) drives the flow, the owner approves on his phone/browser, and
the resulting oauth.json (refresh token) is all that's stored on disk.

Added 2026-09-26 for the Vector "play my liked songs" / "play my <name>
playlist" / "play my <name> mix" feature. See README.md.
"""
import logging
import os

VECTOR_MUSIC_DIR = "/opt/vector-music"
YTM_DIR = f"{VECTOR_MUSIC_DIR}/ytm"
OAUTH_PATH = f"{YTM_DIR}/oauth.json"
DEVICE_STATE_PATH = f"{YTM_DIR}/device_state.json"
ENV_PATH = f"{VECTOR_MUSIC_DIR}/.env"

log = logging.getLogger("vector-music.ytm")


def load_oauth_client_creds():
    """Read YTM_OAUTH_CLIENT_ID / YTM_OAUTH_CLIENT_SECRET from the process
    environment first, falling back to a local (chmod 600, not committed)
    KEY=VALUE .env file at /opt/vector-music/.env. Never logs values.
    Returns (client_id, client_secret), either of which may be '' if unset.
    """
    client_id = os.environ.get("YTM_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YTM_OAUTH_CLIENT_SECRET", "").strip()
    if (not client_id or not client_secret) and os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "YTM_OAUTH_CLIENT_ID" and not client_id:
                        client_id = v
                    elif k == "YTM_OAUTH_CLIENT_SECRET" and not client_secret:
                        client_secret = v
        except OSError as e:
            log.warning("could not read %s: %s", ENV_PATH, e)
    return client_id, client_secret


def get_ytmusic_client():
    """Return an authenticated ytmusicapi.YTMusic client for the owner's account,
    or None if the OAuth client creds or the approved oauth.json aren't
    available yet (caller should fall back to public search in that case).
    Never raises -- any failure here just means "not authenticated"."""
    if not os.path.exists(OAUTH_PATH):
        log.info("no oauth.json at %s -- not authenticated yet", OAUTH_PATH)
        return None
    client_id, client_secret = load_oauth_client_creds()
    if not client_id or not client_secret:
        log.warning(
            "oauth.json exists but YTM_OAUTH_CLIENT_ID/YTM_OAUTH_CLIENT_SECRET "
            "are not set (checked env + %s) -- cannot use it", ENV_PATH
        )
        return None
    try:
        from ytmusicapi import OAuthCredentials, YTMusic
    except ImportError as e:
        log.error("ytmusicapi not installed: %s", e)
        return None
    try:
        return YTMusic(
            OAUTH_PATH,
            oauth_credentials=OAuthCredentials(client_id=client_id, client_secret=client_secret),
        )
    except Exception as e:  # noqa: BLE001 -- any auth/parse failure -> unauthenticated
        log.warning("ytmusicapi auth failed (token expired/revoked?): %s", e)
        return None


# How many tracks to queue for a single "my ..." request. Keeps a bare "play
# my liked songs" (which could be hundreds of tracks) from turning into an
# unbounded queue; also bounds ytmusicapi/yt-dlp calls per voice request.
MY_MUSIC_QUEUE_LIMIT = 15


def resolve_liked_songs(yt, limit=MY_MUSIC_QUEUE_LIMIT):
    """Return up to `limit` videoIds from the owner's YT Music Liked Songs."""
    try:
        result = yt.get_liked_songs(limit=limit)
    except Exception as e:  # noqa: BLE001
        log.warning("get_liked_songs failed: %s", e)
        return []
    tracks = result.get("tracks", []) if isinstance(result, dict) else []
    return [t["videoId"] for t in tracks if t.get("videoId")][:limit]


def _best_playlist_match(playlists, name):
    """Fuzzy-match `name` against a list of ytmusicapi playlist dicts
    (each with a 'title'). Returns the best dict or None."""
    name_l = name.lower().strip()
    if not name_l:
        return None

    # 1) exact title match
    for p in playlists:
        if (p.get("title") or "").strip().lower() == name_l:
            return p

    # 2) substring containment either direction, scored by shared words
    name_words = set(name_l.split())
    best, best_score = None, 0
    for p in playlists:
        title_l = (p.get("title") or "").strip().lower()
        if not title_l:
            continue
        if name_l in title_l or title_l in name_l:
            score = len(name_words & set(title_l.split())) + 1
            if score > best_score:
                best_score, best = score, p
    if best:
        return best

    # 3) fallback: difflib close-match on title strings
    import difflib

    titles = [p.get("title", "") for p in playlists]
    match = difflib.get_close_matches(name, titles, n=1, cutoff=0.5)
    if match:
        for p in playlists:
            if p.get("title") == match[0]:
                return p
    return None


def resolve_named_playlist(yt, name, limit=MY_MUSIC_QUEUE_LIMIT):
    """Fuzzy-match `name` against the owner's library playlists and return up to
    `limit` videoIds from the best match. Empty list if nothing matches."""
    if not name:
        return []
    try:
        playlists = yt.get_library_playlists(limit=200)
    except Exception as e:  # noqa: BLE001
        log.warning("get_library_playlists failed: %s", e)
        return []
    best = _best_playlist_match(playlists, name)
    if not best or not best.get("playlistId"):
        log.info("no library playlist match for name=%r", name)
        return []
    log.info("matched playlist name=%r -> %r (id=%s)", name, best.get("title"), best["playlistId"])
    try:
        pl = yt.get_playlist(best["playlistId"], limit=limit)
    except Exception as e:  # noqa: BLE001
        log.warning("get_playlist(%s) failed: %s", best["playlistId"], e)
        return []
    tracks = pl.get("tracks", []) if isinstance(pl, dict) else []
    return [t["videoId"] for t in tracks if t.get("videoId")][:limit]


def resolve_mix(yt, name, limit=MY_MUSIC_QUEUE_LIMIT):
    """Resolve 'my <name> mix'. Tries, in order: a library playlist literally
    named '<name> mix' or '<name>', then a YT Music playlist search for
    '<name> mix'. Returns [] if nothing usable is found (caller falls back
    to public search)."""
    if name:
        ids = resolve_named_playlist(yt, f"{name} mix", limit=limit)
        if ids:
            return ids
        ids = resolve_named_playlist(yt, name, limit=limit)
        if ids:
            return ids
        try:
            results = yt.search(f"{name} mix", filter="playlists", limit=5)
        except Exception as e:  # noqa: BLE001
            log.warning("mix search failed for name=%r: %s", name, e)
            results = []
        for r in results or []:
            playlist_id = r.get("playlistId") or r.get("browseId")
            if not playlist_id:
                continue
            try:
                pl = yt.get_playlist(playlist_id, limit=limit)
            except Exception as e:  # noqa: BLE001
                log.warning("get_playlist(%s) failed during mix search: %s", playlist_id, e)
                continue
            tracks = pl.get("tracks", []) if isinstance(pl, dict) else []
            ids = [t["videoId"] for t in tracks if t.get("videoId")][:limit]
            if ids:
                return ids
        return []
    # bare "play my mix" with no name -> nothing sensible to disambiguate to;
    # let the caller fall back to public search rather than guessing.
    return []


def classify_my_music_request(raw_query: str):
    """Detect 'my liked songs' / 'my <name> playlist' / 'my <name> mix'
    inside a raw (pre strip_trigger) voice query. Returns (kind, name):
      ("liked", None)              -- Liked Songs
      ("playlist", "<name>"|None)  -- named/library playlist ("" -> None)
      ("mix", "<name>"|None)       -- named mix
      (None, None)                 -- not a 'my music' request; caller keeps
                                       the existing public-search behavior.
    """
    t = raw_query.strip().lower().strip('"')
    for lead in ("play me ", "play "):
        if t.startswith(lead):
            t = t[len(lead):]
            break
    if not t.startswith("my "):
        return None, None
    rest = t[len("my "):].strip()
    if not rest:
        return None, None

    if rest in ("liked songs", "likes", "liked music", "favorites", "favourites"):
        return "liked", None

    if rest.endswith(" playlist"):
        name = rest[: -len(" playlist")].strip()
        return "playlist", (name or None)
    if rest == "playlist":
        return "playlist", None

    if rest.endswith(" mix"):
        name = rest[: -len(" mix")].strip()
        return "mix", (name or None)
    if rest == "mix":
        return "mix", None

    return None, None


def resolve_my_music_ids(kind: str, name, limit=MY_MUSIC_QUEUE_LIMIT):
    """Top-level resolver: kind/name (from classify_my_music_request) -> list
    of videoIds, or [] if unauthenticated / no match (caller falls back to
    public search and logs it -- never raises)."""
    yt = get_ytmusic_client()
    if yt is None:
        log.info("my-music request kind=%s name=%r but no authenticated ytmusicapi client", kind, name)
        return []
    if kind == "liked":
        return resolve_liked_songs(yt, limit=limit)
    if kind == "playlist":
        if name:
            return resolve_named_playlist(yt, name, limit=limit)
        return resolve_liked_songs(yt, limit=limit)  # bare "play my playlist" -> liked songs
    if kind == "mix":
        return resolve_mix(yt, name, limit=limit)
    return []
