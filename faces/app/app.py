"""vector-faces: local household face recognition HTTP service.

Bound to 127.0.0.1 only (face data of the household, incl. minors -- nothing
leaves this host). No cloud calls, no telemetry.

  POST   /recognize      JPEG body | data-URI text | JSON {"image": ...} | multipart
                         -> [{"name", "slug", "score", "bbox": [x,y,w,h], "det_score"}]
                         ?brighten=0 to skip the shared Vector brightening.
  POST   /enroll         multipart (name=..., files) | JSON {"name", "images": [...]}
                         ?dry_run=1 ?replace=1 ?min_samples=N ?brighten=0 ?source=...
  GET    /people
  DELETE /people/<slug>
  GET    /health
"""
import json
import logging
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facelib  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vector-faces")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
STARTED = time.time()
ENGINE = facelib.FaceEngine()
GALLERY = facelib.Gallery()
MODEL_SHA = {
    os.path.basename(facelib.YUNET_PATH): facelib.sha256_file(facelib.YUNET_PATH),
    os.path.basename(facelib.SFACE_PATH): facelib.sha256_file(facelib.SFACE_PATH),
}


def _flag(name, default):
    v = request.args.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _images_from_request():
    """-> list of raw image bytes, plus an optional name from the body.
    Multipart -> uploaded files (+ form field "name"). Anything else is read
    raw, whatever Content-Type the client claimed (curl defaults to
    x-www-form-urlencoded, which must not be parsed as a form here)."""
    name = request.args.get("name")
    ctype = (request.content_type or "").lower()
    if ctype.startswith("multipart/"):
        name = request.form.get("name") or name
        imgs = []
        for key in request.files:
            for f in request.files.getlist(key):
                imgs.append(f.read())
        return imgs, name
    body = request.get_data(cache=False, parse_form_data=False)
    if not body:
        return [], name
    if body[:1] in (b"{", b"["):
        js = json.loads(body.decode("utf-8"))
        if isinstance(js, dict):
            name = js.get("name") or name
            items = js.get("images") or ([js["image"]] if js.get("image") else [])
        else:
            items = js
        return [facelib.decode_image_bytes(x) for x in items], name
    return [facelib.decode_image_bytes(body)], name


@app.get("/health")
def health():
    GALLERY.refresh()
    return jsonify({
        "status": "ok",
        "models": MODEL_SHA,
        "threshold": facelib.MATCH_THRESHOLD,
        "ambiguity_margin": facelib.AMBIGUITY_MARGIN,
        "people": len(GALLERY.people),
        "uptime_s": round(time.time() - STARTED, 1),
    })


@app.get("/people")
def people():
    return jsonify(GALLERY.list())


@app.delete("/people/<slug>")
def delete_person(slug):
    slug = facelib.slugify(slug)
    if GALLERY.delete(slug):
        log.info("deleted person %s", slug)
        return jsonify({"deleted": slug})
    return jsonify({"error": "not found", "slug": slug}), 404


@app.post("/recognize")
def recognize():
    t0 = time.time()
    try:
        imgs, _ = _images_from_request()
        if not imgs:
            return jsonify({"error": "no image"}), 400
        img, scale, _binfo = facelib.load_bgr(imgs[0], brighten=_flag("brighten", True))
    except Exception as e:
        return jsonify({"error": "bad image: %s" % e}), 400
    out = []
    for f in ENGINE.detect(img):
        det = float(f[14])
        item = {"name": "unknown", "slug": None, "score": 0.0,
                "bbox": facelib._face_box(f, scale), "det_score": round(det, 3)}
        if min(float(f[2]), float(f[3])) >= facelib.RECOG_MIN_FACE:
            v, _ = ENGINE.embed(img, f)
            name, slug, score, second = GALLERY.match(v)
            item.update({"name": name, "slug": slug, "score": round(score, 3),
                         "runner_up": round(second, 3)})
        out.append(item)
    out.sort(key=lambda x: -x["score"])
    log.info("recognize: %d face(s) %s (%.0f ms)", len(out),
             [(o["name"], o["score"]) for o in out], (time.time() - t0) * 1000)
    return jsonify(out)


@app.post("/enroll")
def enroll():
    try:
        imgs, name = _images_from_request()
    except Exception as e:
        return jsonify({"error": "bad request: %s" % e}), 400
    name = (name or "").strip()
    slug = facelib.slugify(name)
    if not slug:
        return jsonify({"error": "name required"}), 400
    if not imgs:
        return jsonify({"error": "no images"}), 400
    dry = _flag("dry_run", False)
    replace = _flag("replace", False)
    brighten = _flag("brighten", True)
    min_samples = int(request.args.get("min_samples", "3"))
    source = request.args.get("source", "api")[:64]

    embs, crops, rejected = [], [], []
    for i, raw in enumerate(imgs):
        try:
            img, _, _ = facelib.load_bgr(raw, brighten=brighten)
        except Exception as e:
            rejected.append({"index": i, "reason": "decode: %s" % e})
            continue
        v, crop, reason = facelib.analyse_for_enroll(ENGINE, img)
        if v is None:
            rejected.append({"index": i, "reason": reason})
            continue
        embs.append((i, v, crop))

    # Drop batch outliers (someone else walked into frame for a moment).
    warnings = []
    if len(embs) >= 3:
        M = np.stack([e[1] for e in embs])
        c = M.mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        keep = []
        for (i, v, crop), s in zip(embs, M @ c):
            if s < facelib.ENROLL_OUTLIER_SIM:
                rejected.append({"index": i, "reason": "outlier vs batch (%.2f)" % s})
            else:
                keep.append((i, v, crop))
        embs = keep
    # Must agree with what we already have for this person (unless replacing).
    GALLERY.refresh()
    if embs and not replace and slug in GALLERY.people:
        pe = GALLERY.people[slug]["emb"]
        pc = pe.mean(axis=0)
        pc /= (np.linalg.norm(pc) or 1.0)
        bc = np.stack([e[1] for e in embs]).mean(axis=0)
        bc /= (np.linalg.norm(bc) or 1.0)
        agree = float(bc @ pc)
        if agree < facelib.ENROLL_OUTLIER_SIM:
            return jsonify({"error": "new samples do not look like the existing %s samples "
                                     "(similarity %.2f); use replace=1 if intended" % (name, agree),
                            "accepted": 0, "rejected": rejected}), 409
    # Warn if this face already matches someone ELSE.
    if embs:
        bc = np.stack([e[1] for e in embs]).mean(axis=0)
        bc /= (np.linalg.norm(bc) or 1.0)
        for other, p in GALLERY.people.items():
            if other == slug:
                continue
            oc = p["emb"].mean(axis=0)
            oc /= (np.linalg.norm(oc) or 1.0)
            s = float(bc @ oc)
            if s >= facelib.MATCH_THRESHOLD:
                warnings.append("looks similar to %s (%.2f)" % (p["meta"].get("name", other), s))

    result = {"name": name, "slug": slug, "dry_run": dry, "received": len(imgs),
              "accepted": len(embs), "rejected": rejected, "warnings": warnings}
    if len(embs) < min_samples:
        result["error"] = "only %d usable face frame(s); need %d" % (len(embs), min_samples)
        log.info("enroll %s: rejected (%d usable)", slug, len(embs))
        return jsonify(result), 422
    if not dry:
        meta = GALLERY.save_person(name, slug, np.stack([e[1] for e in embs]),
                                   [e[2] for e in embs], replace=replace, source=source)
        result["n_samples"] = meta["n_samples"]
        log.info("enroll %s: +%d samples (total %d) source=%s", slug, len(embs),
                 meta["n_samples"], source)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host=os.environ.get("FACES_BIND", "127.0.0.1"),
            port=int(os.environ.get("FACES_PORT", "8096")), threaded=True)
