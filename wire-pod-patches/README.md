# wire-pod patches

A `git format-patch` series against upstream
[wire-pod](https://github.com/kercre123/wire-pod), **not** a fork copy.
wire-pod is MIT-licensed (Copyright (c) 2022 Kerigan Creighton); its `chipper/`
server is derived from Digital Dream Labs' open-sourced chipper (MIT,
Copyright (c) 2021 Digital Dream Labs). The patches only touch files under
`chipper/` and are offered under the same MIT terms — see the root `LICENSE`
and `THIRD_PARTY_NOTICES.md`. Full credit to
[kercre123](https://github.com/kercre123) and contributors for wire-pod and WireOS.

## What's here

- `patches/0001-…0010-*.patch` — the series, cut against upstream commit
  **`347c45f`** (2026-08-27, "Merge pull request #521"). It also applies
  cleanly (`git am -3`) to upstream `main` as of 2026-10-14 (`dc6ae0c`).
- `apiConfig.example.json` — wire-pod's runtime config
  (`data/chipper/apiConfig.json`) with placeholders: persona prompt, Ollama /
  brain-proxy endpoint, `llm_first`, conversation mode, speech pace.
- `customIntents.example.json` — the custom intents the patches and `music/`
  rely on (music, dance, stop, sentry, end-conversation). The `exec` paths are
  container paths; bind-mount `music/` at `/opt/vector-music`.
- `deploy.sh` — our build/deploy script (worked example: read before running).

## The series

| # | Patch | What it does |
|---|---|---|
| 0001 | baseline | `llm_first` routing (LLM first, intent graph as fallback), conversation mode v1 (re-open the mic after a reply), pending-intent tracking, the optional escalation hook (`pendinghermes.go`), `/api-sdk/play_animation` + `/api-sdk/list_anim_triggers`, stream-end / VAD tuning, LLM-flush and behaviour-control timeouts, keyphrase tuning in `intent-data/en-US.json` |
| 0002 | step1 | `kgsim_cmds.go` panic guard for a missing `doIntent` parameter |
| 0003 | step2 | fast spoken ack the moment the LLM stream opens + wake-word interrupt grace period |
| 0004 | step3 | "sentry mode on/off/auto" voice command (`sentrycmd.go`) |
| 0005–0006 | step4 | serial fast command classifier (`classify.go`, small CPU-pinned model) before the LLM fallback, plus its e2e timing gate |
| 0007 | step5 | eyes-only "thinking" animation while the LLM works (track-locked so it never moves wheels/lift) |
| 0008 | step6 | **conversation mode up to 100 turns** (was 6) + **kill phrase** ("end conversation", "that's all Vector", "goodbye Vector") + fix: the follow-up timeout is now measured from the end of speech, not from the start of the round trip incl. TTS |
| 0009 | step7 | **fuzzy kill-phrase matching** — tolerant of Vosk hearing "end" as "in"/"and", word order, and a trailing "stop"; checked before plugins, custom intents, core intents, the classifier and the LLM |
| 0010 | step8 | **speech pace**: configurable TTS `DurationScalar` via `knowledge.speech_duration_scalar` (default 1.05, ~10% slower and clearer than stock) |

## Applying

```bash
git clone https://github.com/kercre123/wire-pod.git
cd wire-pod
git checkout 347c45f            # or stay on main; the series also applies there
git am /path/to/vector-local-brain/wire-pod-patches/patches/*.patch
cp /path/to/vector-local-brain/wire-pod-patches/apiConfig.example.json  data/chipper/apiConfig.json   # edit placeholders
cp /path/to/vector-local-brain/wire-pod-patches/customIntents.example.json data/chipper/customIntents.json
```

New `apiConfig.json` keys used by the patches (all optional):

| Key | Default | Meaning |
|---|---|---|
| `knowledge.llm_first` | `false` | send voice to the LLM before the intent graph |
| `knowledge.conversation_mode` | `false` | auto re-listen after each reply |
| `knowledge.conversation_timeout_sec` | `12` | silence before conversation mode gives up |
| `knowledge.speech_duration_scalar` | `1.05` | TTS pace (1.0 = stock) |
| `STT.end_of_speech_silence_ms` | | silence that ends an utterance |

The Vosk end-to-end test expects a `stttest_noise.pcm` fixture that is **not**
shipped (it was a room recording); the test skips cleanly without it.
