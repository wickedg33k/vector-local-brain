#!/usr/bin/env python3
"""Transcribe a WAV file with faster-whisper. Prints JSON to stdout."""
import argparse, json, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device-index", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    lang = None if args.lang == "auto" else args.lang

    t0 = time.time()
    used_device = args.device
    used_compute = "int8_float16" if args.device == "cuda" else "int8"
    try:
        if args.device == "cuda":
            model = WhisperModel(
                args.model,
                device="cuda",
                device_index=args.device_index,
                compute_type="int8_float16",
            )
        else:
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
    except Exception as e:
        sys.stderr.write(f"[transcribe] {args.device} load failed ({e}); falling back to CPU int8\n")
        used_device = "cpu"
        used_compute = "int8"
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    load_s = time.time() - t0

    t1 = time.time()
    segments, info = model.transcribe(
        args.wav,
        language=lang,
        vad_filter=True,
        word_timestamps=False,
        beam_size=5,
    )
    seg_list = []
    for s in segments:
        seg_list.append({"start": s.start, "end": s.end, "text": s.text.strip()})
    infer_s = time.time() - t1

    # Explicitly drop the model / free CUDA context so a GPU run doesn't
    # linger and doesn't compete with Ollama for VRAM after we're done.
    del model

    out = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": seg_list,
        "model": args.model,
        "device": used_device,
        "compute_type": used_compute,
        "load_seconds": load_s,
        "infer_seconds": infer_s,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
