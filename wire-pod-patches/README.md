# wire-pod patches

This directory is **our diff against upstream [wire-pod](https://github.com/kercre123/wire-pod)**,
not a fork copy. wire-pod is MIT-licensed (Copyright (c) 2022 Kerigan
Creighton) — see the root `LICENSE` file. Full credit to
[kercre123](https://github.com/kercre123) and contributors for wire-pod and
WireOS; this repo would not exist without that project.

## What's here

- `tracked-files.diff` — a `git diff` against upstream `chipper/` for files
  wire-pod already ships (apply with `git apply` from a wire-pod checkout's
  `chipper/` directory, or read it directly — it's the fastest way to see
  exactly what we changed and why, since every hunk is comment-annotated).
- `new-files/` — new files we added under `chipper/pkg/wirepod/...` that
  don't exist upstream. Copy these into the matching path in your wire-pod
  checkout.
- `deploy.sh` — our build/deploy script: builds a Docker image from a
  patched wire-pod checkout, swaps it in for the running container, and
  waits for a few post-deploy health signals before declaring success.
  Treat it as a worked example, not a drop-in script — it has our paths
  and container names baked in; read it before running it.
- `apiConfig.example.json` — wire-pod's own runtime config
  (`data/chipper/apiConfig.json`), with all placeholder values. This is
  where the LLM persona/system prompt, the Ollama endpoint, and
  `llm_first` routing get configured. Copy it to `apiConfig.json` in your
  wire-pod data dir and fill in the placeholders.

## Key patches, in plain English

- **`llm_first` routing** (`apiConfig.example.json` + `matchIntentSend.go`
  in the diff) — when enabled, voice input goes to the LLM first and only
  falls back to wire-pod's built-in intent graph if the LLM's response
  says it can't help. This flips wire-pod's default priority (intent graph
  first, LLM as fallback).
- **`classify.go`** — a tiny, fast LLM call (capped at ~12 tokens) that
  classifies a voice-sourced utterance's intent before the main LLM call,
  so cheap/obvious commands ("turn around", "go home") don't pay full LLM
  latency.
- **`pendingintent.go`** — tracks an intent that's waiting on a follow-up
  before it fires (e.g. a clarifying question), so the next utterance can
  resolve it instead of starting a new turn from scratch.
- **`kgsim_cmds.go` panic guard** — upstream's command simulator could
  panic on certain malformed/edge-case command sequences; wrapped in a
  recover so one bad command can't take the whole voice pipeline down.
- **Conversation mode** (`conversation.go`) — after an LLM reply finishes
  speaking, re-opens the mic for a follow-up (like it heard "Hey Vector"
  again) instead of requiring the wake word every single turn.
- **`/api-sdk/play_animation` and `/api-sdk/list_anim_triggers`**
  (`sdkapp/animation.go`, `sdkapp/server.go` in the diff) — new HTTP
  endpoints so external tools (like `vector-curious`) can trigger a named
  animation or discover what's available, without going through the voice
  pipeline.
- **The "live-9" keyphrase fix** and **extra explore keyphrases**
  (`intent-data/en-US.json` in the diff) — keyphrase/intent-data tuning
  from iterating through several deploy cycles (the `live-N` naming in
  `deploy.sh`/commit comments is just our own build numbering, nothing
  wire-pod-specific).
- **`pendinghermes.go` / the escalation hook in the diff** — an optional
  "if the LLM says it doesn't know, ask a bigger remote assistant and
  speak its answer instead" path. Wired up via `brain-proxy/` in this
  repo; fully optional and off by default unless configured.

## Applying

```bash
git clone https://github.com/kercre123/wire-pod.git
cd wire-pod/chipper
git apply /path/to/vector-local-brain/wire-pod-patches/tracked-files.diff
cp -r /path/to/vector-local-brain/wire-pod-patches/new-files/pkg/* pkg/
cp /path/to/vector-local-brain/wire-pod-patches/apiConfig.example.json ../data/chipper/apiConfig.json
# edit apiConfig.json's placeholders (endpoint, robotName, etc.)
```

Upstream moves; the diff was cut against a specific commit and may need
minor conflict resolution against a newer wire-pod `main`.
