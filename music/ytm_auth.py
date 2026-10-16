#!/usr/bin/env python3
"""
ytm_auth.py -- non-interactive Google OAuth device-code sign-in for the owner's
YouTube Music account (ytmusicapi), for the Vector "play my liked songs" /
"play my <name> playlist" / "play my <name> mix" feature.

NO PASSWORD HANDLING. This drives the Google "TVs and Limited Input devices"
OAuth device-code flow: we print a short verification URL + user code, the owner
opens that URL on his phone/laptop, signs in with his own Google session, and
approves. We only ever see a device_code (short-lived, useless without also
knowing the approval) and, once approved, an OAuth refresh token written to
oauth.json (chmod 600). Never logs a token or device_code value.

Requires a Google Cloud OAuth client of type "TVs and Limited Input devices"
with the YouTube Data API v3 enabled (see /opt/vector-music/README.md for the
exact console steps). Client id/secret are read from the environment
(YTM_OAUTH_CLIENT_ID / YTM_OAUTH_CLIENT_SECRET) or a local
/opt/vector-music/.env file (chmod 600, KEY=VALUE, not committed anywhere).

Usage:
    ytm_auth.py start              # request a device code, print URL + code, exit
    ytm_auth.py poll [--timeout N] # poll until the owner approves (default 900s = 15min)
                                    # writes /opt/vector-music/ytm/oauth.json on success

`start` prints ONLY:
    VERIFICATION_URL=<url>
    USER_CODE=<code>
    EXPIRES_IN=<seconds>
to stdout -- that's what gets handed to the owner. Everything else goes to
ytm_auth.log.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytm_common import DEVICE_STATE_PATH, ENV_PATH, OAUTH_PATH, YTM_DIR, load_oauth_client_creds  # noqa: E402

LOG_PATH = "/opt/vector-music/ytm_auth.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
)
log = logging.getLogger("ytm-auth")


def _require_client_creds():
    client_id, client_secret = load_oauth_client_creds()
    if not client_id or not client_secret:
        msg = (
            "MISSING_CLIENT_CREDS: set YTM_OAUTH_CLIENT_ID and YTM_OAUTH_CLIENT_SECRET "
            f"(env vars, or KEY=VALUE lines in {ENV_PATH}, chmod 600). "
            "See README.md 'YouTube Music OAuth client setup' for exact Google Cloud console steps."
        )
        print(msg, file=sys.stderr)
        log.error("start/poll aborted: client creds missing")
        sys.exit(2)
    return client_id, client_secret


def cmd_start():
    from ytmusicapi.auth.oauth import OAuthCredentials

    client_id, client_secret = _require_client_creds()
    creds = OAuthCredentials(client_id=client_id, client_secret=client_secret)
    try:
        code = creds.get_code()
    except Exception as e:  # noqa: BLE001
        log.error("get_code() failed: %s", e)
        print(f"FAILED: could not request device code: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(YTM_DIR, exist_ok=True)
    state = {
        "device_code": code["device_code"],
        "interval": int(code.get("interval", 5)),
        "expires_in": int(code.get("expires_in", 1800)),
        "requested_at": int(time.time()),
    }
    with open(DEVICE_STATE_PATH, "w") as f:
        json.dump(state, f)
    os.chmod(DEVICE_STATE_PATH, 0o600)

    log.info(
        "device code requested (value not logged), expires_in=%s interval=%s",
        state["expires_in"], state["interval"],
    )
    # Only these three lines are meant to leave the box (handed to the owner).
    print(f"VERIFICATION_URL={code['verification_url']}")
    print(f"USER_CODE={code['user_code']}")
    print(f"EXPIRES_IN={state['expires_in']}")


def cmd_poll(timeout_s: int) -> int:
    from ytmusicapi.auth.oauth import OAuthCredentials
    from ytmusicapi.auth.oauth.token import RefreshingToken

    client_id, client_secret = _require_client_creds()
    if not os.path.exists(DEVICE_STATE_PATH):
        print("NO_STATE: run 'ytm_auth.py start' first", file=sys.stderr)
        sys.exit(2)
    with open(DEVICE_STATE_PATH) as f:
        state = json.load(f)

    creds = OAuthCredentials(client_id=client_id, client_secret=client_secret)
    device_code = state["device_code"]
    interval = max(int(state.get("interval", 5)), 5)
    budget = min(timeout_s, int(state.get("expires_in", 1800)))
    deadline = time.time() + budget

    log.info("poll starting, budget=%ss interval=%ss", budget, interval)
    while time.time() < deadline:
        try:
            raw = creds.token_from_code(device_code)
        except Exception as e:  # noqa: BLE001
            log.error("token_from_code() raised: %s", e)
            print(f"FAILED: {e}")
            return 1

        if "access_token" in raw:
            refresh_expires_in = raw.get("refresh_token_expires_in", raw["expires_in"])
            token = RefreshingToken(
                credentials=creds,
                access_token=raw["access_token"],
                refresh_token=raw["refresh_token"],
                scope=raw["scope"],
                token_type=raw["token_type"],
                expires_in=refresh_expires_in,
            )
            token.update(raw)  # sets expires_at from current time + expires_in
            token.store_token(OAUTH_PATH)
            os.chmod(OAUTH_PATH, 0o600)
            try:
                os.remove(DEVICE_STATE_PATH)
            except OSError:
                pass
            log.info("approved -- oauth.json written to %s (mode 600)", OAUTH_PATH)
            print("APPROVED")
            return 0

        err = raw.get("error", "unknown_error")
        if err == "authorization_pending":
            pass  # normal -- the owner hasn't approved yet
        elif err == "slow_down":
            interval += 5
            log.info("slow_down -- backing off to %ss", interval)
        elif err in ("expired_token", "access_denied"):
            log.error("device flow ended: %s", err)
            print(f"FAILED:{err}")
            return 1
        else:
            log.error("unexpected token endpoint error: %s", err)
            print(f"FAILED:{err}")
            return 1
        time.sleep(interval)

    log.error("poll timed out after %ss without approval", budget)
    print("TIMEOUT")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "poll"])
    ap.add_argument("--timeout", type=int, default=900, help="poll budget in seconds (default 900 = 15min)")
    args = ap.parse_args()
    if args.action == "start":
        cmd_start()
    else:
        sys.exit(cmd_poll(args.timeout))


if __name__ == "__main__":
    main()
