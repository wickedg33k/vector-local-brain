# Robot-side tweaks

These changes live **on the Vector robot itself** (WireOS), not in this repo.
They're documented here as snippets/findings rather than shipped as files,
since they're small, hand-applied edits to an existing robot filesystem.

## 1. Cap journald's disk usage

Vector's onboard storage is small; an unbounded journal can fill it.

`/etc/systemd/journald.conf.d/cap.conf`:
```ini
[Journal]
RuntimeMaxUse=8M
```
Then `systemctl restart systemd-journald`.

## 2. `user_intent_map.json` typo fix + added intents

The stock map had a typo that silently broke the "explore" app-triggered
intent: the app_intent key was `"explore_start"` instead of the value the
rest of the system actually expects, `"intent_explore_start"`. Voice-sourced
intents worked fine (they don't go through this map), which is what made
this confusing to track down — see finding #6 below.

```diff
- "explore_start": "intent_explore_start",
+ "intent_explore_start": "intent_explore_start",
```

We also added a few extra explore-flavored keyphrases in wire-pod's
`intent-data` (see `wire-pod-patches/`) so more natural phrasings route to
the same intent instead of falling through to the LLM.

## 3. Forward robot syslog to a LAN receiver

`syslog-ng` (already present on WireOS) forwards to the tiny UDP listener in
`log-receiver/` so robot-side logs are visible off-device without SSH:

```
destination d_udp_receiver {
    udp("POD_HOST" port(5514));
};
log {
    source(src);
    destination(d_udp_receiver);
};
```
Replace `POD_HOST` with wherever `log-receiver/udp_syslog_listener.py` runs.

## 4. Pin the pod host in `/etc/hosts`

WireOS resolves its pod host via mDNS by default, which occasionally
misbehaves. Pinning it removes a class of intermittent-connection bugs:
```
POD_HOST_IP  escapepod.local
```

## 5. The red-box low-memory overlay

If you see a red bounding-box overlay appear on Vector's face/eyes with no
other symptoms, it's WireOS's low-memory indicator, not a camera or vision
pipeline bug. It clears on its own once free memory recovers; it's a signal
to look at what's eating RAM on the robot (usually a runaway vision/SDK
client), not at the wire-pod container.

## 6. Fault 914 from SDK connect storms

Fault code 914 on the robot was traced to **connect storms** — an SDK
client (ours or a stray process) retrying `vector.Robot(...)` connections in
a tight loop, usually after the robot briefly drops off wifi or wire-pod
restarts mid-connection. Fix is client-side backoff, not a robot setting:
never retry an SDK connect more than once every few seconds, and give up
after a handful of tries rather than looping forever.

## 7. AppIntent-sourced intents silently dropped

App-triggered intents (via `robot_control` / the SDK's `AppIntent`, as
opposed to a spoken wake word) were being accepted by wire-pod but never
reaching the matcher — voice-sourced intents worked the whole time, which
made this look like an intent-matching bug rather than a routing one. Root
cause was upstream of the intent matcher (see item #2 above and the
`classify.go` / `matchIntentSend.go` patches in `wire-pod-patches/`) — the
fix was making sure app-sourced intents get the same normalization voice
intents already got, not a matcher change.
