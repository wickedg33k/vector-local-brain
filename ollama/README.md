# ollama/ — model tuning and warm-up

Vector is a voice companion: a cold model load (~60 s for a 30B MoE) reads as
"broken". These settings keep every model resident and fully on the GPU.

## systemd override (`/etc/systemd/system/ollama.service.d/override.conf`)

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"   # or 127.0.0.1 if everything is on one host
Environment="OLLAMA_CONTEXT_LENGTH=32768"
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Keep a single override file: duplicating `OLLAMA_HOST` across several drop-ins
means editing one silently leaves the other in effect.

## Modelfiles

Ollama's default GPU-layer heuristic left part of the models on the CPU on our
2-GPU host. Re-creating them with `num_gpu 99` (offload every layer) and an
explicit context fixed time-to-first-token. Keep a backup tag of the original
first so you can roll back.

```bash
ollama cp qwen3:30b-a3b-instruct-2507-q4_K_M qwen3-30b-a3b-orig-bak
ollama create qwen3:30b-a3b-instruct-2507-q4_K_M -f Modelfile.chat
ollama cp gemma3:4b gemma3-4b-orig-bak
ollama create gemma3:4b -f Modelfile.vision
```

`Modelfile.chat` (main voice model):
```
FROM qwen3:30b-a3b-instruct-2507-q4_K_M
PARAMETER num_gpu 99
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1
```

`Modelfile.vision` (camera questions, `VISION_MODEL=gemma3:4b` in `brain-proxy/start.sh`):
```
FROM gemma3:4b
PARAMETER num_gpu 99
PARAMETER num_ctx 8192
```

The tiny command classifier used by the wire-pod patches (`qwen3:1.7b`,
tagged `qwen3:1.7b-classify`) is deliberately pinned to the CPU
(`PARAMETER num_gpu 0`, `num_ctx 2048`) so it never evicts the big model's KV
cache.

## Warm-up after reboot (root crontab)

```cron
@reboot sleep 90  && curl -s -m 120 127.0.0.1:11434/api/generate -d '{"model":"qwen3:1.7b-classify","keep_alive":-1}' >/dev/null
@reboot sleep 100 && curl -s -m 200 127.0.0.1:11434/api/generate -d '{"model":"qwen3:30b-a3b-instruct-2507-q4_K_M","keep_alive":-1,"prompt":"hi"}' >/dev/null
@reboot sleep 260 && curl -s -m 90  127.0.0.1:11434/api/generate -d '{"model":"gemma3:4b","keep_alive":-1,"prompt":"hi"}' >/dev/null
```

The staggered sleeps load the small model first, then the chat model, then the
vision model, so they don't contend for VRAM during boot.
