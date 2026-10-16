"""Auto-exposure correction for Vector camera frames (2026-10-15, hermes).

Vector's sensor is starved indoors: a live probe showed auto-exposure ON but
already pinned at its hardware ceiling (exposure 66 ms = max, gain 3.8 = max),
so frames come out mean-luma ~55-85 / median ~40-70 even in a lit room. The
camera can't be pushed further, so we fix it in software before a vision
model sees the frame.

brighten_jpeg(bytes) -> (new_bytes, info) ; new_bytes is None when no change.
brighten_data_uri(uri) -> (new_uri, info) ; new_uri is None when no change.
Fails open: any error -> (None, {"error": ...}). Needs Pillow + numpy.

Algorithm (only when dark: mean < DARK_MEAN or p95 < DARK_P95):
  0. gentle gray-world white balance (half strength, factors 0.85-1.2);
  1. linear stretch of luma percentiles p1..p98.5 to 0..255, applied identically
     to R,G,B (keeps colour), with the gain capped so noise isn't blown up;
  2. gamma (adaptive, clamped 0.60-0.90) to pull the mean toward ~TARGET_MEAN;
  3. mild contrast (x1.05).
Near-black frames (mean < BLACK_MEAN, e.g. lights off at night) are left
alone so the model still sees "dark" instead of amplified sensor noise.
"""
import base64
import io
import math
import os

DARK_MEAN = float(os.environ.get("BRIGHTEN_DARK_MEAN", "100"))
DARK_P95 = float(os.environ.get("BRIGHTEN_DARK_P95", "180"))
BLACK_MEAN = float(os.environ.get("BRIGHTEN_BLACK_MEAN", "12"))
TARGET_MEAN = float(os.environ.get("BRIGHTEN_TARGET_MEAN", "125"))
MAX_GAIN = float(os.environ.get("BRIGHTEN_MAX_GAIN", "3.0"))
GAMMA_MIN, GAMMA_MAX = 0.60, 0.90
HI_PCT = 98.5  # clip the top 1.5% (windows/lamps pin p99 and kill the stretch)
WB_STRENGTH = float(os.environ.get("BRIGHTEN_WB", "0.5"))  # 0 disables gentle gray-world WB
CONTRAST = 1.05
JPEG_QUALITY = 90


def _luma_stats(arr):
    import numpy as np
    y = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    p1, p50, p95, phi = np.percentile(y, [1, 50, 95, HI_PCT])
    return float(y.mean()), float(p1), float(p50), float(p95), float(phi)


def brighten_jpeg(data):
    try:
        import numpy as np
        from PIL import Image, ImageEnhance
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        mean, p1, p50, p95, p99 = _luma_stats(arr)
        info = {"mean_before": round(mean, 1), "p50_before": round(p50), "p95_before": round(p95)}
        if mean >= DARK_MEAN and p95 >= DARK_P95:
            info["skipped"] = "not dark"
            return None, info
        if mean < BLACK_MEAN:
            info["skipped"] = "near-black (lights off?)"
            return None, info
        if WB_STRENGTH > 0:
            # gentle gray-world white balance: Vector frames carry a green/cyan cast
            ch = arr.reshape(-1, 3).mean(axis=0)
            if float(ch.min()) > 1.0:
                f = (float(ch.mean()) / ch) ** WB_STRENGTH
                f = np.clip(f, 0.85, 1.2)
                arr = arr * f.astype(np.float32)
                info["wb"] = [round(float(x), 2) for x in f]
        lo, hi = p1, max(p99, p1 + 1.0)
        gain = 255.0 / (hi - lo)
        if gain > MAX_GAIN:
            # cap the stretch; keep the black point, let highlights stop short
            gain = MAX_GAIN
        out = np.clip((arr - lo) * gain, 0, 255) / 255.0
        m = float((out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114).mean())
        if 0.0 < m < 1.0:
            g = math.log(TARGET_MEAN / 255.0) / math.log(m)
        else:
            g = GAMMA_MAX
        g = min(GAMMA_MAX, max(GAMMA_MIN, g))
        out = np.power(out, g) * 255.0
        res = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")
        res = ImageEnhance.Contrast(res).enhance(CONTRAST)
        buf = io.BytesIO()
        res.save(buf, format="JPEG", quality=JPEG_QUALITY)
        a2 = np.asarray(res, dtype=np.float32)
        mean2, _, p50b, p95b, _ = _luma_stats(a2)
        info.update({"mean_after": round(mean2, 1), "p50_after": round(p50b), "p95_after": round(p95b),
                     "stretch": [round(lo), round(hi)], "gain": round(gain, 2), "gamma": round(g, 2)})
        return buf.getvalue(), info
    except Exception as e:  # fail open
        return None, {"error": "%s: %s" % (type(e).__name__, e)}


def brighten_data_uri(uri):
    try:
        if not isinstance(uri, str) or not uri.startswith("data:image") or "," not in uri:
            return None, {"skipped": "not a data:image URI"}
        header, b64 = uri.split(",", 1)
        if ";base64" not in header:
            return None, {"skipped": "not base64"}
        new, info = brighten_jpeg(base64.b64decode(b64))
        if new is None:
            return None, info
        return "data:image/jpeg;base64," + base64.b64encode(new).decode("ascii"), info
    except Exception as e:
        return None, {"error": "%s: %s" % (type(e).__name__, e)}
