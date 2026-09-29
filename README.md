# vector-local-brain

A local, offline-first "brain" for the [Anki/Digital Dream Labs
Vector](https://www.anki.com/) desk robot — wake word → local LLM →
speech, entirely on your own hardware, with a Retrieval-Augmented Generation
(RAG) memory layer, an idle "curious" wandering/vision mode, a persistent
mood/diary system, optional Coral Edge TPU object detection, and an
optional escalation path to a bigger remote assistant when the local model
comes up short.

**Not affiliated with Anki, Digital Dream Labs, or the official Vector app
in any way.** This is a hobby project built on top of the excellent
open-source [wire-pod](https://github.com/kercre123/wire-pod) /
[WireOS](https://github.com/kercre123/wire-pod) project, which replaces
Vector's (now-defunct) cloud backend with a local one.

## What this actually is

Vector's official cloud backend has been shut down for years; wire-pod is
the community project that lets the robot keep working by pointing it at a
backend you run yourself. This repo is what we built **on top of** wire-pod:
a local LLM brain with memory, a wandering/curiosity mode, computer vision,
and a small fleet of support services — all running on hardware you own, no
cloud dependency required (aside from an optional escalation hook).

## Architecture

```
                    ┌─────────────────────┐
   wifi             │   Vector robot       │
◄──────────────────►│  (WireOS + camera)   │
                    └─────────┬────────────┘
                              │ wire-pod protocol (wake word, STT, intents)
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         GPU host                              │
   │                                                                 │
   │  ┌────────────┐   /v1/chat/completions   ┌──────────────────┐ │
   │  │  wire-pod   │◄────────────────────────►│  brain-proxy      │ │
   │  │ (patched,   │   llm_first routing        │  (server.py)     │ │
   │  │  Go, Docker)│                             │  - RAG (rag.py)  │ │
   │  └─────┬──────┘                             │  - mood/diary    │ │
   │        │ STT: vosk                            │  - escalation ↗ │ │
   │        │                                       └────────┬────────┘ │
   │        │ /api-sdk/play_animation                        │          │
   │        │                                                 ▼          │
   │  ┌─────▼──────┐        ┌────────────┐       ┌──────────────────┐  │
   │  │  curious    │──────►│   Ollama    │◄──────│  Obsidian vault   │  │
   │  │ (wander +   │  LLM  │ (local LLM  │  RAG  │  (persistent      │  │
   │  │  vision loop)│ calls│  + vision)  │ reads │   memory/notes)   │  │
   │  └─────┬──────┘        └────────────┘       └──────────────────┘  │
   │        │ object detection                                          │
   │        ▼                                                            │
   │  ┌────────────┐        ┌────────────────┐                          │
   │  │ Coral       │        │  Dashboard      │                        │
   │  │ Edge TPU    │        │  (read-only     │                        │
   │  │ service     │        │   web UI)       │                        │
   │  └────────────┘        └────────────────┘                          │
   └──────────────────────────────────────────────────────────────┘
                              │ (optional, off by default)
                              ▼
                  ┌───────────────────────────┐
                  │  Remote escalation         │
                  │  assistant (any HTTP LLM   │
                  │  endpoint you point it at) │
                  └───────────────────────────┘
```

## Components

| Dir | What it is |
|---|---|
| `wire-pod-patches/` | Our diff/patches against upstream wire-pod, plus the build/deploy script and an example `apiConfig.json` |
| `brain-proxy/` | The LLM proxy wire-pod talks to: injects RAG context from an Obsidian-style vault, tracks a lightweight mood/battery state, keeps same-day conversational memory, and optionally escalates unanswerable questions to a remote assistant |
| `curious/` | An idle-time "wander and be curious" mode: periodically takes a snapshot, describes what it sees, asks a short spoken question about it, and folds the answer into memory |
| `coral/` | A small Flask/Docker service wrapping a Google Coral USB Edge TPU for fast local object detection, used by `curious/` |
| `dashboard/` | A read-only web dashboard showing recent conversations, curious-mode activity, and mood/battery state |
| `health/` | A one-line-per-run stability probe (ping + a couple of HTTP checks) for cron |
| `transcribe/` | Optional: a standalone `faster-whisper` CLI wrapper for transcribing arbitrary audio/video, independent of the robot pipeline |
| `log-receiver/` | A tiny UDP syslog-style listener so you can forward the robot's own logs off-device |
| `docs/robot-tweaks.md` | Small robot-side (WireOS) edits and findings that don't belong in this repo as files |

## Features

- **Local-first**: wake word → STT → LLM → TTS all run on your own
  hardware via wire-pod + Ollama. No cloud dependency for normal operation.
- **`llm_first` routing**: voice goes to the LLM first, wire-pod's built-in
  intent graph is the fallback (patched into wire-pod — see
  `wire-pod-patches/`).
- **RAG memory**: a scoped, auditable retrieval layer over a small
  Obsidian-style markdown vault, with an always-on always-included slice
  plus similarity-ranked retrieval for the rest.
- **Mood + same-day memory**: the brain proxy tracks lightweight
  mood/battery state and recent same-day conversation turns so replies have
  continuity across a session.
- **Curious/wander mode**: during idle time, the robot looks around, asks
  about what it sees, and (optionally) drives home to charge — all
  config-driven (quiet hours, gap timing, battery thresholds, etc. — see
  `curious/config.example.json`).
- **Coral Edge TPU vision**: sub-30ms local object detection, used to give
  curious mode something specific to ask about.
- **Escalation hook**: optional, rate-limited fallback to a bigger remote
  LLM assistant when the local model can't answer — fails open to the local
  model's own answer on any error/timeout.
- **Stability tooling**: a health probe, a log receiver, and a
  build/deploy script with a last-known-good rollback image.

## Hardware used

- A GPU host running Docker: 2× 16GB-class GPUs (one for the always-on
  local LLM via Ollama, kept warm; the other with headroom for occasional
  GPU-accelerated transcription), enough RAM to run wire-pod + brain-proxy +
  curious + Ollama concurrently.
- A Google Coral USB Accelerator (Edge TPU) for object detection.
- A Vector robot (any WireOS-compatible unit) on the same LAN.
- Everything is otherwise commodity: no special networking, no cloud
  compute required.

## Setup

1. **wire-pod**: follow [wire-pod's own setup docs](https://github.com/kercre123/wire-pod)
   to get your robot escaped and talking to a wire-pod instance you control.
   Then apply our patches — see `wire-pod-patches/README.md`.
2. **Ollama**: install and pull the models referenced in
   `wire-pod-patches/apiConfig.example.json` / `curious/config.example.json`
   (an instruct model for chat, a small vision model, an embedding model
   for RAG).
3. **brain-proxy**: copy `brain-proxy/.env.example`, fill in placeholders,
   point wire-pod's `apiConfig.json` `knowledge.endpoint` at brain-proxy
   instead of Ollama directly (brain-proxy proxies to Ollama and injects
   RAG context). Run with `start.sh` / `stop.sh` / `watchdog.sh`.
4. **A markdown vault** for RAG: any folder of `.md` notes works; point
   `VECTOR_VAULT_ROOT` at it. Start empty — it grows as you use it.
5. **curious/** (optional): copy `config.example.json` → `config.json`,
   fill in your robot's ESN, build/run the Docker container.
6. **coral/** (optional, needed for curious mode's object descriptions):
   install the udev rule (`coral/udev/`), download the two EdgeTPU models
   from the [`google-coral/test_data`](https://github.com/google-coral/test_data)
   repo into `coral/models/` (not included in this repo — see
   `coral/README.md` for exact filenames), build/run.
7. **dashboard/, health/, log-receiver/, transcribe/**: all optional,
   independently useful, see each directory.

## Robot-side tweaks

A handful of small edits live on the robot itself (WireOS), not in this
repo — journald disk cap, a `user_intent_map.json` typo fix, syslog
forwarding, a `/etc/hosts` pin, and a couple of debugging findings (a
low-memory overlay, a connect-storm fault, and an intent-routing gotcha).
See **`docs/robot-tweaks.md`**.

## Credits

- [wire-pod](https://github.com/kercre123/wire-pod) by
  [kercre123](https://github.com/kercre123) and contributors — the
  local-backend project that makes any of this possible. MIT licensed.
- [WireOS](https://github.com/kercre123/wire-pod) — the companion
  robot-side firmware.
- The [`anki_vector` Python SDK](https://github.com/digital-dream-labs/vector-python-sdk) —
  Apache-2.0 licensed, used client-side for the utility scripts under
  `curious/`.
- [Google Coral](https://coral.ai/) / `pycoral` / `tflite-runtime` for the
  Edge TPU vision service.

## License

MIT — see `LICENSE`. `wire-pod-patches/` is a diff against wire-pod (also
MIT); see `wire-pod-patches/README.md` for attribution details.
