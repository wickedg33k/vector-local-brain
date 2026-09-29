#!/usr/bin/env bash
# transcribe.sh <url-or-path> [--lang auto|en] [--out DIR] [--gpu]
#
# Downloads (if URL) / uses local file, converts to 16k mono WAV, transcribes
# with faster-whisper, writes .txt/.srt/.md.
#
# Default device: CPU (int8). This box also runs Ollama's qwen3:32b as
# Vector's always-on brain (OLLAMA_KEEP_ALIVE=-1). CPU is the safe default so
# this tool can never touch it.
#
# --gpu opts into CUDA (device 1, the RTX 5000 with more headroom). It only
# runs if a conservative pre-flight check passes; otherwise it silently falls
# back to CPU and says so. This is not a theoretical worry: on 2026-09-28,
# GPU use here (with GPU1 showing ~5.8GB free) caused Ollama's scheduler to
# evict qwen3:32b ("predicted to exceed available memory") and then fail to
# reload for several minutes (GPU discovery watchdog timeout) while our
# process held VRAM. Require a much bigger margin before trying GPU again.
set -euo pipefail

BASE="$HOME/transcribe"
VENV="$BASE/venv"
OUT_DIR="$BASE/out"
LANG="auto"
INPUT=""
USE_GPU=0
MIN_FREE_MIB=11000   # empirically-informed safety margin, see note above

usage() { echo "Usage: $0 <url-or-path> [--lang auto|en] [--out DIR] [--gpu]" >&2; exit 1; }

[ $# -ge 1 ] || usage
INPUT="$1"; shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    --lang) LANG="$2"; shift 2 ;;
    --out)  OUT_DIR="$2"; shift 2 ;;
    --gpu)  USE_GPU=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

mkdir -p "$OUT_DIR"
TMPDIR=$(mktemp -d "$BASE/tmp/job.XXXXXX")
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

WALL_T0=$(date +%s.%N)

DEVICE="cpu"
if [ "$USE_GPU" = "1" ]; then
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ' || echo 0)
  QWEN_LOADED=$(curl -s --max-time 3 http://127.0.0.1:11434/api/ps 2>/dev/null | grep -c '"qwen3' || true)
  if [ "${FREE_MIB:-0}" -ge "$MIN_FREE_MIB" ]; then
    DEVICE="cuda"
    echo "[transcribe] GPU1 free=${FREE_MIB}MiB >= ${MIN_FREE_MIB}MiB margin — using GPU" >&2
  else
    echo "[transcribe] GPU1 free=${FREE_MIB}MiB < ${MIN_FREE_MIB}MiB safety margin (qwen3 loaded: $([ "$QWEN_LOADED" -gt 0 ] && echo yes || echo unknown)) — falling back to CPU to avoid evicting Vector's brain model" >&2
  fi
fi

# 1. Resolve input -> raw audio/video file + a name
if [[ "$INPUT" =~ ^https?:// ]]; then
  echo "[transcribe] downloading audio via yt-dlp: $INPUT" >&2
  source "$VENV/bin/activate"
  yt-dlp --js-runtimes deno:~/.local/bin/deno -x --audio-format wav --audio-quality 0 \
    -o "$TMPDIR/src.%(ext)s" \
    --no-playlist --no-simulate \
    --print "%(title)s" \
    "$INPUT" > "$TMPDIR/title.txt" 2> "$TMPDIR/ytdlp.log" || {
      cat "$TMPDIR/ytdlp.log" >&2; exit 1;
    }
  RAW_SRC=$(ls "$TMPDIR"/src.* | head -1)
  TITLE=$(tail -1 "$TMPDIR/title.txt" 2>/dev/null || echo "download")
  SOURCE_DESC="$INPUT"
else
  [ -f "$INPUT" ] || { echo "[transcribe] no such file: $INPUT" >&2; exit 1; }
  RAW_SRC="$INPUT"
  TITLE=$(basename "$INPUT")
  TITLE="${TITLE%.*}"
  SOURCE_DESC="$INPUT"
fi

# sanitize title for filenames
SAFE_NAME=$(echo "$TITLE" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//;s/_*$//' | cut -c1-80)
[ -n "$SAFE_NAME" ] || SAFE_NAME="transcript_$(date +%s)"

# 2. Convert to 16kHz mono WAV
WAV="$TMPDIR/audio.wav"
echo "[transcribe] converting to 16kHz mono wav" >&2
ffmpeg -y -loglevel error -i "$RAW_SRC" -ac 1 -ar 16000 -vn "$WAV"

DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$WAV" 2>/dev/null || echo "0")

# 3. Transcribe
echo "[transcribe] running faster-whisper on $DEVICE (lang=$LANG)" >&2
T0=$(date +%s.%N)
source "$VENV/bin/activate"
JSON_OUT="$TMPDIR/result.json"
DEV_INDEX_ARGS=()
[ "$DEVICE" = "cuda" ] && DEV_INDEX_ARGS=(--device-index 1)
python "$BASE/transcribe.py" "$WAV" --lang "$LANG" --device "$DEVICE" "${DEV_INDEX_ARGS[@]}" > "$JSON_OUT" 2> "$TMPDIR/whisper.log" || {
  cat "$TMPDIR/whisper.log" >&2; exit 1;
}
cat "$TMPDIR/whisper.log" >&2 || true
T1=$(date +%s.%N)
PROC_S=$(python3 -c "print(f'{$T1-$T0:.1f}')")

# 4. Render outputs
TXT="$OUT_DIR/${SAFE_NAME}.txt"
SRT="$OUT_DIR/${SAFE_NAME}.srt"
MD="$OUT_DIR/${SAFE_NAME}.md"

python3 - "$JSON_OUT" "$TXT" "$SRT" "$MD" "$TITLE" "$SOURCE_DESC" <<'PYEOF'
import json, sys, datetime

json_path, txt_path, srt_path, md_path, title, source = sys.argv[1:7]
with open(json_path) as f:
    data = json.load(f)

segs = data["segments"]

def srt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def mmss(t):
    t = int(t)
    return f"{t//60:02d}:{t%60:02d}"

with open(txt_path, "w") as f:
    f.write(" ".join(s["text"] for s in segs).strip() + "\n")

with open(srt_path, "w") as f:
    for i, s in enumerate(segs, 1):
        f.write(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{s['text']}\n\n")

dur = data.get("duration", 0)
with open(md_path, "w") as f:
    f.write(f"# {title}\n\n")
    f.write(f"- **Source:** {source}\n")
    f.write(f"- **Duration:** {mmss(dur)}\n")
    f.write(f"- **Language:** {data.get('language')} (p={data.get('language_probability', 0):.2f})\n")
    f.write(f"- **Model:** {data.get('model')} on {data.get('device')}/{data.get('compute_type')}\n\n")
    buf = []
    buf_start = None
    last_end = None
    for s in segs:
        if buf_start is None:
            buf_start = s["start"]
        if last_end is not None and s["start"] - last_end > 2.5 and buf:
            f.write(f"**[{mmss(buf_start)}]** " + " ".join(buf).strip() + "\n\n")
            buf = []
            buf_start = s["start"]
        buf.append(s["text"])
        last_end = s["end"]
    if buf:
        f.write(f"**[{mmss(buf_start)}]** " + " ".join(buf).strip() + "\n\n")

print(json.dumps({
    "language": data.get("language"),
    "duration": dur,
    "infer_seconds": data.get("infer_seconds"),
    "load_seconds": data.get("load_seconds"),
    "device": data.get("device"),
    "compute_type": data.get("compute_type"),
}))
PYEOF

WALL_T1=$(date +%s.%N)
WALL_S=$(python3 -c "print(f'{$WALL_T1-$WALL_T0:.1f}')")

echo "[transcribe] done" >&2
echo "TXT: $TXT"
echo "SRT: $SRT"
echo "MD:  $MD"
echo "duration_s: $DURATION"
echo "wall_seconds: $WALL_S"
echo "transcribe_seconds: $PROC_S"
