# music/ — "play a song", dance mode, and stop-on-wake

Runs **inside the wire-pod container** as custom-intent exec scripts
(bind-mount this directory at `/opt/vector-music`), plus two tiny watchers that
run on the Docker host.

| File | Where it runs | What it does |
|---|---|---|
| `play.py` | wire-pod container (custom intent `play_music_custom` / `dance_music_custom`) | Searches + downloads one track with `yt-dlp`, converts it to Vector's 16 kHz mono WAV and streams it to the robot speaker via the Python SDK. Detaches a worker so the intent returns immediately. |
| `stop.sh` | wire-pod container (`stop_music_custom`) | Kills the worker from its pidfile; idempotent. |
| `stop_on_voice.sh` | Docker host (systemd unit in `systemd/`) | Tails `docker logs vector-pod`; when a new voice capture starts more than 5 s after a song started, runs `stop.sh` so you can talk to Vector without shouting over the music. |
| `duck_watcher.sh` | Docker host (optional, superseded) | Earlier approach: lowers volume during voice capture. Lowering master volume mid-stream ended the SDK audio stream on our unit, so `stop_on_voice.sh` is the one we run. |
| `ytm_common.py`, `ytm_auth.py` | wire-pod container (optional) | "play my liked songs / my <name> playlist" via `ytmusicapi` + Google OAuth device-code flow. Client id/secret come from env (`YTM_OAUTH_CLIENT_ID`, `YTM_OAUTH_CLIENT_SECRET`) or `/opt/vector-music/.env` (chmod 600, git-ignored). Without them it falls back to a plain public search. |

## Dance mode

"dance party" / "dance to the music" picks an upbeat track and dances along:

- A lightweight, numpy-only **BPM estimate** of the downloaded WAV
  (`estimate_bpm`, falls back to 115 BPM) paces the moves.
- Moves are Vector's own stock `anim_dancebeat_*` clips (the ones "Hey
  Vector, dance" uses), bracketed by `DanceBeatGetIn` / `DanceBeatGetOut` and
  played back-to-back with no manual sleeps. Faster songs favour the quick
  eye/head-nod clips, slower songs the bigger head + lift + body clips. The old
  hand-rolled head/lift loop was removed because it stuttered and kept raising
  the arm.
- The worker holds the shared **robot-control lock** (`fcntl.flock` on
  `robot_control.lock`, also used by `curious/` and `faces/enroll.py`) so two
  tools never drive the robot at once.

## Setup

```bash
# inside the wire-pod image (or a derived image): python3, ffmpeg, a venv with
#   pip install yt-dlp wirepod_vector_sdk requests   (+ ytmusicapi, optional)
# bind-mount this dir at /opt/vector-music, then merge the music entries from
# wire-pod-patches/customIntents.example.json into data/chipper/customIntents.json
sudo cp systemd/vector-music-stop-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vector-music-stop-watcher
```

`ROBOT_SERIAL` (your ESN) must be set in the container environment.

Downloading audio with `yt-dlp` may be restricted by the terms of the service
you download from; use it only for content you are entitled to play.
