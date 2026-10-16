# faces/ — local household face recognition

A small Flask service (Docker, `--network host`, bound to `127.0.0.1:8096`
only) that tells Vector *who* is in front of it, so the brain proxy can say
"People you can currently see: Alex" and curious mode can greet people by the
names **you** enrolled. Nothing leaves the host: no cloud APIs, no telemetry,
CPU only.

- **Detection**: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  (`cv2.FaceDetectorYN`) — faces + 5 landmarks.
- **Embedding**: [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
  (`cv2.FaceRecognizerSF`) — aligned 112×112 crop → 128-d embedding, L2-normalised.
- **Matching**: per-person score = mean of the top-3 cosine similarities
  against that person's samples. Default threshold **0.42**; if the runner-up
  is also above threshold and within 0.04 of the best, the answer is
  "unknown" (siblings / parent-child look-alikes). Faces under 28 px are
  "unknown". Override with `FACES_THRESHOLD`, `FACES_AMBIGUITY_MARGIN`.
- Every frame goes through `brain-proxy/brighten.py` first (Vector's camera is
  dim; fail-open if the import is unavailable).

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/recognize` | JPEG body, data-URI text, JSON `{"image": ...}` or multipart → `[{name, slug, score, bbox, det_score}]`. `?brighten=0` skips brightening. |
| `POST` | `/enroll` | multipart (`name=`, files) or JSON `{"name", "images": [...]}`. `?dry_run=1`, `?replace=1`, `?min_samples=N`. |
| `GET` | `/people` | enrolled people + sample counts |
| `DELETE` | `/people/<slug>` | forget someone |
| `GET` | `/health` | |

## Models (not included — download and verify)

```bash
mkdir -p models && cd models
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
sha256sum -c SHA256SUMS
```

| File | sha256 | License |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT (Shiqi Yu) |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

## Enrolling people

```bash
cp config.example.json config.json      # set robot_serial + lock_file
./run.sh                                # build + start the container
./enroll.py --name "Alex"               # live: Vector looks at the person, ~20 frames
./enroll.py --name "Jordan" --from-dir /path/to/photos   # from existing photos
./enroll.py --name "Test" --dry-run     # exercise the robot side, save nothing
curl -s 127.0.0.1:8096/people
```

The live flow takes the shared robot-control flock (`lock_file`) for its whole
duration so it never fights curious mode, music or the day-cycle script for
the robot, says a short prompt, captures frames with the head tilted up and
the lift down (no wheel motion), then POSTs them to `/enroll`. Enrollment gates
per frame: exactly one face, detector score ≥ 0.85, face ≥ 48 px, sharpness
check; outliers vs. the batch centroid are dropped; max 80 samples per person.

## Privacy

The face DB (`people/<slug>/{embeddings.npy, meta.json, crops/}`) is created
with dirs `0700` / files `0600` and is **git-ignored**, as are `work/`,
`config.json`, `current_people.json` and the model weights. Only enroll people
who have agreed to it, and get a parent's OK for minors. Delete someone with
`DELETE /people/<slug>`.

## Integration

- `curious/` POSTs each daytime capture to `/recognize` and writes
  `current_people.json` (entries kept ~10 min). Config knob `face_source`:
  `local` (default), `robot` (Vector's built-in face names) or `both`.
- `brain-proxy/server.py` reads `current_people.json` (`VECTOR_CURRENT_PEOPLE_PATH`)
  and injects "People you can currently see: …" for names seen in the last
  120 s, plus a static rule never to claim to see anyone not listed, and a
  short summary of that person's own note (`20-agents/vector/people/`).
- During the hourly explorer, `curious/` checks for known faces every few
  seconds and turns to greet them (see the root README changelog).
