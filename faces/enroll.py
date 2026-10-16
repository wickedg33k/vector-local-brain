#!/usr/bin/env python3
"""Enroll a household member into vector-faces (local only).

Live (Vector looks at the person and takes ~20 frames):
    ~/vector-faces/enroll.py --name "Alex"
From existing photos (no robot involved):
    ~/vector-faces/enroll.py --name "Jordan" --from-dir /path/to/photos
Test the robot side without saving anything:
    ~/vector-faces/enroll.py --name "Test" --dry-run

The live flow needs wirepod_vector_sdk, which lives in the vector-faces
container, so on the host this script re-runs itself there via `docker exec`.
--from-dir needs only the Python stdlib and runs anywhere on the host.

Robot-control etiquette: the live flow holds an exclusive flock on
~/vector-pod/data/robot_control.lock for its whole duration (the same
lock vector-daycycle and vector-curious take), so it never fights them for
the robot. If the lock is busy it waits up to --lock-wait seconds.
"""
import argparse
import base64
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.json")
DEFAULTS = {
    "service_url": "http://127.0.0.1:8096",
    "robot_serial": os.environ.get("VECTOR_ESN", "YOUR_ESN"),
    "wirepod_api_base": "http://127.0.0.1:8080",
    "lock_file": os.path.expanduser("~/vector-pod/data/robot_control.lock"),
    "container": "vector-faces",
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


def say(cfg, text, tries=4):
    """Speak via wire-pod's HTTP API (same path curious.py uses). Retries a
    few times because wire-pod (vector-pod) may be mid-restart."""
    body = ("serial=%s&text=%s" % (cfg["robot_serial"], urllib.parse.quote(text))).encode()
    for attempt in range(1, tries + 1):
        req = urllib.request.Request(cfg["wirepod_api_base"] + "/api-sdk/say_text", data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                res = r.read().decode("utf-8", "replace").strip()
            print("  say(%r) -> %s" % (text, res))
            if "success" in res.lower():
                return True
        except Exception as e:
            print("  say(%r) attempt %d/%d failed: %s" % (text, attempt, tries, e))
        if attempt < tries:
            time.sleep(3)
    return False


def post_enroll(cfg, name, images, dry_run, replace, brighten, min_samples, source):
    q = urllib.parse.urlencode({"dry_run": int(dry_run), "replace": int(replace),
                                "brighten": int(brighten), "min_samples": min_samples,
                                "source": source})
    payload = json.dumps({"name": name, "images": [base64.b64encode(b).decode("ascii") for b in images]})
    req = urllib.request.Request(cfg["service_url"] + "/enroll?" + q, data=payload.encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e)}


def service_ok(cfg):
    try:
        with urllib.request.urlopen(cfg["service_url"] + "/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def report(status, res):
    print("\nservice: HTTP %s" % status)
    for k in ("name", "slug", "dry_run", "received", "accepted", "n_samples", "error"):
        if k in res:
            print("  %-10s %s" % (k, res[k]))
    for w in res.get("warnings") or []:
        print("  WARNING   %s" % w)
    rej = res.get("rejected") or []
    if rej:
        reasons = {}
        for r in rej:
            key = r["reason"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print("  rejected  %d %s" % (len(rej), reasons))


# ---------------------------------------------------------------- from-dir
def from_dir(cfg, args):
    files = sorted(os.path.join(args.from_dir, f) for f in os.listdir(args.from_dir)
                   if f.lower().endswith(IMG_EXT))
    if not files:
        sys.exit("no images in %s" % args.from_dir)
    print("enrolling %r from %d photo(s) in %s%s" % (args.name, len(files), args.from_dir,
                                                       " (dry run)" if args.dry_run else ""))
    images = []
    for p in files:
        with open(p, "rb") as f:
            images.append(f.read())
    status, res = post_enroll(cfg, args.name, images, args.dry_run, args.replace,
                              not args.no_brighten, args.min_samples or 1, "from-dir")
    report(status, res)
    return 0 if status == 200 else 1


# ---------------------------------------------------------------- live
def _reexec_in_container(cfg):
    if os.environ.get("FACES_IN_CONTAINER"):
        sys.exit("wirepod_vector_sdk missing inside the container?")
    docker = shutil.which("docker")
    if not docker:
        sys.exit("live enrollment needs docker (vector-faces container)")
    argv = [docker, "exec", "-i"] + (["-t"] if sys.stdin.isatty() else []) + \
           [cfg["container"], "python3", "/app/enroll.py"] + sys.argv[1:]
    os.execv(docker, argv)


def acquire_lock(path, wait_s):
    import fcntl
    fh = open(path, "a")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.time() - t0 > wait_s:
                fh.close()
                return None
            time.sleep(0.5)


def capture_frames(cfg, n_frames, seconds, head_angle):
    import threading
    import anki_vector
    from anki_vector.connection import ControlPriorityLevel
    from anki_vector.util import degrees

    robot = anki_vector.Robot(serial=cfg["robot_serial"],
                              behavior_control_level=ControlPriorityLevel.DEFAULT_PRIORITY,
                              default_logging=False, enable_face_detection=False,
                              cache_animation_lists=False)
    box = {"frames": [], "err": None}

    def _do():
        robot.connect(timeout=10)
        try:
            robot.behavior.set_lift_height(0.0, duration=0.3)
            robot.behavior.set_head_angle(degrees(head_angle), duration=0.4)
        except Exception as e:
            print("  head/lift move failed (continuing): %s" % e)
        gap = seconds / float(max(1, n_frames))
        for i in range(n_frames):
            t = time.time()
            img = robot.camera.capture_single_image()
            if img is not None:
                buf = io.BytesIO()
                img.raw_image.convert("RGB").save(buf, "JPEG", quality=92)
                box["frames"].append(buf.getvalue())
            time.sleep(max(0.0, gap - (time.time() - t)))

    th = threading.Thread(target=lambda: _safe(_do, box), daemon=True)
    th.start()
    th.join(seconds + 25)
    timed_out = th.is_alive()
    try:
        robot.disconnect()
    except Exception:
        pass
    if timed_out:
        print("  capture HARD TIMEOUT -- connection force-closed")
    if box["err"]:
        print("  capture error: %s" % box["err"])
    return box["frames"]


def _safe(fn, box):
    try:
        fn()
    except Exception as e:
        box["err"] = "%s: %s" % (type(e).__name__, e)


def live(cfg, args):
    try:
        import anki_vector  # noqa: F401
    except ImportError:
        _reexec_in_container(cfg)
    if not service_ok(cfg):
        sys.exit("vector-faces service is not healthy at %s" % cfg["service_url"])
    name = args.name
    print("live enrollment for %r%s: waiting for robot_control.lock ..." %
          (name, " (DRY RUN, nothing saved)" if args.dry_run else ""))
    lock = acquire_lock(cfg["lock_file"], args.lock_wait)
    if lock is None:
        sys.exit("robot_control.lock busy for %ss -- try again in a minute" % args.lock_wait)
    try:
        print("lock acquired; speaking prompt")
        if not args.no_speak:
            if not say(cfg, "Look at me, %s. Turn your head a little." % name):
                print("  (robot could not speak the prompt -- capturing anyway)")
            time.sleep(1.0)
        print("capturing %d frames over ~%ds ..." % (args.frames, args.seconds))
        t0 = time.time()
        raw = capture_frames(cfg, args.frames, args.seconds, args.head_angle)
        print("captured %d frame(s) in %.1fs" % (len(raw), time.time() - t0))
        # brighten (shared Vector auto-exposure fix), fail open
        frames = raw
        try:
            if "/mnt/vector-brain" not in sys.path:
                sys.path.append("/mnt/vector-brain")
            from brighten import brighten_jpeg
            frames = [(brighten_jpeg(b)[0] or b) for b in raw]
        except Exception as e:
            print("  brighten unavailable (%s) -- using raw frames" % e)
        if args.save_frames:
            os.makedirs(args.save_frames, exist_ok=True)
            for i, b in enumerate(frames):
                with open(os.path.join(args.save_frames, "frame_%02d.jpg" % i), "wb") as f:
                    f.write(b)
            print("saved frames to %s" % args.save_frames)
        if not frames:
            if not args.no_speak:
                say(cfg, "Sorry, my camera didn't work. Let's try again later.")
            return 1
        status, res = post_enroll(cfg, name, frames, args.dry_run, args.replace, False,
                                  args.min_samples or 5, "live")
        report(status, res)
        if not args.no_speak:
            if args.dry_run:
                say(cfg, "Face test done. I took %d pictures." % len(frames))
            elif status == 200:
                say(cfg, "Got it, nice to meet you, %s." % name)
            else:
                say(cfg, "Sorry %s, I couldn't see your face well enough. Let's try again." % name)
        return 0 if status == 200 else 1
    finally:
        lock.close()  # releases the flock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help='display name, e.g. "Alex"')
    ap.add_argument("--from-dir", help="enroll from existing photos in this directory")
    ap.add_argument("--dry-run", action="store_true", help="do everything except save")
    ap.add_argument("--replace", action="store_true", help="replace this person's samples instead of adding")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--head-angle", type=float, default=30.0, help="degrees, -22..45 (up is +)")
    ap.add_argument("--min-samples", type=int, default=0, help="default 5 live, 1 from-dir")
    ap.add_argument("--lock-wait", type=int, default=90)
    ap.add_argument("--save-frames", help="also write the (brightened) frames here")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--no-brighten", action="store_true", help="from-dir: skip brightening")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.from_dir:
        return from_dir(cfg, args)
    return live(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
