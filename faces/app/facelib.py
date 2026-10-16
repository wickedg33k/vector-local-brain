"""Face detection / embedding / household gallery for vector-faces.

YuNet (cv2.FaceDetectorYN) finds faces + 5 landmarks; SFace
(cv2.FaceRecognizerSF) turns an aligned 112x112 crop into a 128-d embedding.
Embeddings are L2-normalised so cosine similarity is a dot product.

Gallery layout (owned by the service user, dirs 0700, files 0600):
  <root>/people/<slug>/embeddings.npy   float32 (N,128)
  <root>/people/<slug>/meta.json        {name, slug, enrolled_at, updated_at, n_samples}
  <root>/people/<slug>/crops/*.jpg      a few aligned 112x112 face crops
"""
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

ROOT = os.environ.get("FACES_ROOT", "/app")
MODELS_DIR = os.path.join(ROOT, "models")
PEOPLE_DIR = os.path.join(ROOT, "people")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# SFace reference cosine threshold is 0.363 (LFW, same/different split).
# Vector's camera is low-res and dim, and a false "that's the owner" is worse
# than an "unknown", so we add a margin. See BUILD_NOTES.md for the numbers.
MATCH_THRESHOLD = float(os.environ.get("FACES_THRESHOLD", "0.42"))
# If the 2nd-best person is also above threshold and within this margin of
# the best, refuse to pick (siblings / parent+child can be close).
AMBIGUITY_MARGIN = float(os.environ.get("FACES_AMBIGUITY_MARGIN", "0.04"))
TOPK = 3                      # person score = mean of top-k sample sims
DET_SCORE = 0.70              # YuNet confidence to report a face at all
RECOG_MIN_FACE = 28           # px; smaller faces are reported as unknown
ENROLL_MIN_DET = 0.85         # enrollment quality gates
ENROLL_MIN_FACE = 48          # px (min of w,h) in the analysed image
ENROLL_MIN_SHARP = 25.0       # Laplacian variance of the aligned crop
ENROLL_OUTLIER_SIM = 0.30     # sample must be this close to its batch/person centroid
MAX_SAMPLES = 80              # per person (oldest dropped)
MAX_CROPS = 6
MAX_SIDE = 1280               # downscale big photos before detection

_brighten = None


def _get_brighten():
    """Shared Vector brightening (/mnt/vector-brain/brighten.py). Fail open."""
    global _brighten
    if _brighten is None:
        try:
            if "/mnt/vector-brain" not in sys.path:
                sys.path.append("/mnt/vector-brain")
            from brighten import brighten_jpeg
            _brighten = brighten_jpeg
        except Exception:
            _brighten = False
    return _brighten or None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s[:48]


def decode_image_bytes(data):
    """bytes (JPEG/PNG) or data-URI / base64 text -> raw image bytes."""
    if isinstance(data, str):
        data = data.strip()
        if data.startswith("data:"):
            data = data.split(",", 1)[1] if "," in data else ""
        return base64.b64decode(data)
    if data[:5] == b"data:":
        return base64.b64decode(data.split(b",", 1)[1])
    return data


def load_bgr(raw, brighten=True):
    """raw image bytes -> (BGR ndarray, scale, brighten_info). scale maps
    analysed-image coords back to original coords (orig = analysed / scale)."""
    info = None
    if brighten:
        fn = _get_brighten()
        if fn is not None:
            new, info = fn(raw)
            if new is not None:
                raw = new
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale, info


class FaceEngine:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), DET_SCORE, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        self.lock = threading.Lock()

    def detect(self, img):
        h, w = img.shape[:2]
        with self.lock:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
        if faces is None:
            return []
        return [f for f in faces]

    def embed(self, img, face):
        with self.lock:
            aligned = self.rec.alignCrop(img, face)
            feat = self.rec.feature(aligned)
        v = np.asarray(feat, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v, aligned

    @staticmethod
    def sharpness(aligned):
        g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _face_box(face, scale):
    x, y, w, h = [float(v) for v in face[:4]]
    return [int(round(x / scale)), int(round(y / scale)), int(round(w / scale)), int(round(h / scale))]


class Gallery:
    """In-memory view of people/, reloaded whenever a file there changes."""

    def __init__(self, root=PEOPLE_DIR):
        self.root = root
        self._sig = None
        self.people = {}  # slug -> {"meta":..., "emb": ndarray}
        self.lock = threading.RLock()
        os.makedirs(self.root, mode=0o700, exist_ok=True)

    def _signature(self):
        sig = []
        try:
            for e in sorted(os.scandir(self.root), key=lambda e: e.name):
                if e.is_dir():
                    p = os.path.join(e.path, "embeddings.npy")
                    m = os.path.join(e.path, "meta.json")
                    sig.append((e.name, os.path.getmtime(p) if os.path.exists(p) else 0,
                                os.path.getmtime(m) if os.path.exists(m) else 0))
        except FileNotFoundError:
            pass
        return tuple(sig)

    def refresh(self):
        with self.lock:
            sig = self._signature()
            if sig == self._sig:
                return
            people = {}
            for slug, _, _ in sig:
                d = os.path.join(self.root, slug)
                try:
                    with open(os.path.join(d, "meta.json")) as f:
                        meta = json.load(f)
                    emb = np.load(os.path.join(d, "embeddings.npy")).astype(np.float32)
                    if emb.ndim == 2 and emb.shape[0] > 0:
                        people[slug] = {"meta": meta, "emb": emb}
                except Exception:
                    continue
            self.people = people
            self._sig = sig

    def list(self):
        self.refresh()
        out = []
        for slug, p in sorted(self.people.items()):
            m = dict(p["meta"])
            m["n_samples"] = int(p["emb"].shape[0])
            out.append(m)
        return out

    def match(self, v):
        """-> (name, slug, score, runner_up_score)."""
        self.refresh()
        scored = []
        for slug, p in self.people.items():
            sims = p["emb"] @ v
            k = min(TOPK, sims.shape[0])
            s = float(np.mean(np.sort(sims)[-k:]))
            scored.append((s, slug))
        if not scored:
            return "unknown", None, 0.0, 0.0
        scored.sort(reverse=True)
        best, slug = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best < MATCH_THRESHOLD:
            return "unknown", None, best, second
        if second >= MATCH_THRESHOLD and (best - second) < AMBIGUITY_MARGIN:
            return "unknown", None, best, second
        return self.people[slug]["meta"].get("name", slug), slug, best, second

    def person_dir(self, slug):
        return os.path.join(self.root, slug)

    def save_person(self, name, slug, new_emb, crops, replace=False, source=None):
        """Append (or replace) samples for one person, atomically."""
        with self.lock:
            d = self.person_dir(slug)
            old_umask = os.umask(0o077)
            try:
                os.makedirs(os.path.join(d, "crops"), mode=0o700, exist_ok=True)
                now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                meta = {"name": name, "slug": slug, "enrolled_at": now, "updated_at": now,
                        "n_samples": 0, "sources": []}
                emb = np.zeros((0, new_emb.shape[1]), dtype=np.float32)
                mp = os.path.join(d, "meta.json")
                ep = os.path.join(d, "embeddings.npy")
                if not replace and os.path.exists(mp) and os.path.exists(ep):
                    with open(mp) as f:
                        meta.update({k: v for k, v in json.load(f).items() if k != "updated_at"})
                    emb = np.load(ep).astype(np.float32)
                if replace:
                    for fn in os.listdir(os.path.join(d, "crops")):
                        os.remove(os.path.join(d, "crops", fn))
                emb = np.vstack([emb, new_emb.astype(np.float32)])[-MAX_SAMPLES:]
                meta["name"] = name
                meta["updated_at"] = now
                meta["n_samples"] = int(emb.shape[0])
                srcs = list(meta.get("sources") or [])
                srcs.append({"at": now, "added": int(new_emb.shape[0]), "source": source or "api"})
                meta["sources"] = srcs[-20:]
                fd, tmp = tempfile.mkstemp(dir=d, suffix=".npy.tmp")
                with os.fdopen(fd, "wb") as f:
                    np.save(f, emb)
                os.replace(tmp, ep)
                fd, tmp = tempfile.mkstemp(dir=d, suffix=".json.tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump(meta, f, indent=1)
                os.replace(tmp, mp)
                # keep a few crops (newest win)
                cdir = os.path.join(d, "crops")
                stamp = time.strftime("%Y%m%d-%H%M%S")
                for i, c in enumerate(crops[:MAX_CROPS]):
                    cv2.imwrite(os.path.join(cdir, "%s-%02d.jpg" % (stamp, i)), c,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                existing = sorted(os.listdir(cdir))
                for fn in existing[:-MAX_CROPS]:
                    os.remove(os.path.join(cdir, fn))
            finally:
                os.umask(old_umask)
            self._sig = None
            return meta

    def delete(self, slug):
        with self.lock:
            d = self.person_dir(slug)
            if not slug or not os.path.isdir(d) or os.path.dirname(os.path.abspath(d)) != os.path.abspath(self.root):
                return False
            shutil.rmtree(d)
            self._sig = None
            return True


def analyse_for_enroll(engine, img):
    """One frame -> (embedding, aligned_crop, None) or (None, None, reason)."""
    faces = engine.detect(img)
    good = [f for f in faces if float(f[14]) >= 0.6]
    if len(good) == 0:
        return None, None, "no face"
    if len(good) > 1:
        return None, None, "%d faces" % len(good)
    f = good[0]
    det = float(f[14])
    if det < ENROLL_MIN_DET:
        return None, None, "low detector score %.2f" % det
    if min(float(f[2]), float(f[3])) < ENROLL_MIN_FACE:
        return None, None, "face too small (%dpx)" % int(min(f[2], f[3]))
    v, aligned = engine.embed(img, f)
    sharp = engine.sharpness(aligned)
    if sharp < ENROLL_MIN_SHARP:
        return None, None, "blurry (%.0f)" % sharp
    return v, aligned, None
