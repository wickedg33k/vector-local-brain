"""vector-coral: Google Coral USB Edge TPU object-detection HTTP service.

Serves /detect (image in -> detections out) and /health over Flask,
running inference on the Coral USB Accelerator via pycoral/tflite-runtime.
LAN-only by design (bind 0.0.0.0:8095, no auth) -- meant to sit behind
the LAN boundary only, called by Vector-robot vision tooling on gpu-host.
"""
import glob
import io
import json
import os
import threading
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from pycoral.adapters import common, detect
from pycoral.utils import edgetpu

APP_START = time.time()

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
CAPTURES_DIR = os.environ.get("CAPTURES_DIR", "/app/captures")
LABELS_PATH = os.path.join(MODELS_DIR, "coco_labels.txt")
MAX_CAPTURES = 1000  # kept as (image, json) pairs

# Two models available; pick per-request with ?model=, default is the
# better accuracy/latency tradeoff for a robot's small onboard camera feed.
MODEL_FILES = {
    "ssd_mobilenet_v2": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    "efficientdet_lite1": "efficientdet_lite1_384_ptq_edgetpu.tflite",
}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "ssd_mobilenet_v2")


def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[len(labels)] = line
    return labels


LABELS = load_labels(LABELS_PATH)

interpreters = {}
tpu_present = False
tpu_error = None
infer_lock = threading.Lock()  # single USB TPU -> serialize invoke() calls


def init_models():
    global tpu_present, tpu_error
    try:
        devices = edgetpu.list_edge_tpus()
        tpu_present = len(devices) > 0
    except Exception as e:  # noqa: BLE001
        tpu_error = str(e)
        tpu_present = False
    # list_edge_tpus() is a point-in-time enumeration and can be caught
    # mid re-enumeration too; the per-model load loop below is the real
    # source of truth, so tpu_present gets confirmed/overridden there.

    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        # On a cold-plugged/fresh-power Coral USB Accelerator, libedgetpu
        # must upload firmware on first open; the device then disconnects
        # and re-enumerates (1a6e:089a -> 18d1:9302) mid-open, which makes
        # the very first make_interpreter() call fail with
        # "Failed to load delegate from libedgetpu.so.1". Retry through it
        # instead of crashing the whole service on first boot.
        last_err = None
        interp = None
        for attempt in range(1, 6):
            try:
                interp = edgetpu.make_interpreter(path)
                interp.allocate_tensors()
                break
            except ValueError as e:  # noqa: PERF203
                last_err = e
                print(
                    f"[vector-coral] attempt {attempt}/5 loading '{key}' failed "
                    f"(likely USB re-enumeration after firmware upload): {e}",
                    flush=True,
                )
                time.sleep(2 * attempt)
        if interp is None:
            raise RuntimeError(f"could not load model '{key}' after retries: {last_err}")
        interpreters[key] = interp
        tpu_present = True
        tpu_error = None
        print(f"[vector-coral] loaded model '{key}' from {path}", flush=True)

    print(f"[vector-coral] edge tpu present: {tpu_present} (err={tpu_error})", flush=True)


init_models()

app = Flask(__name__)
capture_lock = threading.Lock()


def enforce_capture_limit():
    """Keep only the most recent MAX_CAPTURES (image, json) pairs across all date dirs."""
    files = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*", "*.jpg")), key=os.path.getmtime)
    excess = len(files) - MAX_CAPTURES
    if excess > 0:
        for f in files[:excess]:
            base = f[:-4]
            for ext in (".jpg", ".json"):
                p = base + ext
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


def save_capture(image_bytes, result):
    day = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(CAPTURES_DIR, day)
    os.makedirs(day_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S_%f")
    base = os.path.join(day_dir, ts)
    with open(base + ".jpg", "wb") as f:
        f.write(image_bytes)
    with open(base + ".json", "w") as f:
        json.dump(result, f)
    with capture_lock:
        enforce_capture_limit()


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "tpu_present": tpu_present,
            "tpu_error": tpu_error,
            "default_model": DEFAULT_MODEL,
            "models_loaded": list(interpreters.keys()),
            "uptime_s": round(time.time() - APP_START, 1),
        }
    )


@app.route("/detect", methods=["POST"])
def detect_route():
    model_key = request.args.get("model", DEFAULT_MODEL)
    if model_key not in interpreters:
        return jsonify({"error": f"unknown model '{model_key}', have {list(interpreters)}"}), 400

    try:
        min_score = float(request.args.get("min_score", 0.4))
    except ValueError:
        return jsonify({"error": "min_score must be a float"}), 400

    if request.files:
        file_key = next(iter(request.files))
        image_bytes = request.files[file_key].read()
    else:
        image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"error": "no image data (send raw JPEG body or multipart file)"}), 400

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not decode image: {e}"}), 400

    interp = interpreters[model_key]

    with infer_lock:
        _, scale = common.set_resized_input(
            interp, img.size, lambda size: img.resize(size, Image.ANTIALIAS)
        )
        t0 = time.time()
        interp.invoke()
        inference_ms = (time.time() - t0) * 1000.0
        objs = detect.get_objects(interp, score_threshold=min_score, image_scale=scale)

    detections = []
    for o in objs:
        b = o.bbox
        detections.append(
            {
                "label": LABELS.get(o.id, str(o.id)),
                "score": round(float(o.score), 4),
                "box": [int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax)],
            }
        )

    result = {
        "model": model_key,
        "inference_ms": round(inference_ms, 2),
        "detections": detections,
    }

    try:
        save_capture(image_bytes, result)
    except Exception as e:  # noqa: BLE001
        print(f"[vector-coral] capture save failed: {e}", flush=True)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, threaded=True)
