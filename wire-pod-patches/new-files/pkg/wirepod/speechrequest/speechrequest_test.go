package speechrequest

import "testing"

// --- endOfSpeechReached: pure VAD gating logic, 2026-09-27 "he is cutting
// me off" fix. All units are 10ms VAD frames. ---

func TestEndOfSpeechReached_OldAggressiveThresholdNoLongerTriggersAlone(t *testing.T) {
	// The OLD behavior: 23 inactive frames (230ms) + >18 active frames was
	// enough to end the stream. With the new minTotalFrames floor
	// (150 frames = 1.5s default), that alone must NOT be enough if total
	// elapsed audio is still short.
	inactiveFrames := 23
	activeFrames := 19
	totalFrames := 45 // 450ms total -- well under the new 1.5s floor
	silenceFrames := endOfSpeechSilenceFrames() // resolves to the new default (110 frames / 1.1s)
	minTotal := minTotalAudioFrames()           // resolves to the new default (150 frames / 1.5s)

	if endOfSpeechReached(inactiveFrames, activeFrames, totalFrames, silenceFrames, minTotal) {
		t.Fatalf("expected the old aggressive case to NOT end speech under the new thresholds")
	}
}

func TestEndOfSpeechReached_RequiresFullSilenceWindow(t *testing.T) {
	silenceFrames := 110 // 1.1s
	minTotal := 150      // 1.5s
	activeFrames := 50
	totalFrames := 200 // total is fine

	if endOfSpeechReached(109, activeFrames, totalFrames, silenceFrames, minTotal) {
		t.Fatalf("expected 1090ms of silence (109 frames) to NOT be enough when the threshold is 110")
	}
	if !endOfSpeechReached(110, activeFrames, totalFrames, silenceFrames, minTotal) {
		t.Fatalf("expected exactly 110 frames (1.1s) of silence to be enough")
	}
}

func TestEndOfSpeechReached_RequiresMinTotalAudio(t *testing.T) {
	silenceFrames := 110
	minTotal := 150 // 1.5s
	inactiveFrames := 110
	activeFrames := 50

	if endOfSpeechReached(inactiveFrames, activeFrames, 149, silenceFrames, minTotal) {
		t.Fatalf("expected 1490ms total (149 frames) to NOT be enough when the floor is 150")
	}
	if !endOfSpeechReached(inactiveFrames, activeFrames, 150, silenceFrames, minTotal) {
		t.Fatalf("expected exactly 150 frames (1.5s) total to be enough")
	}
}

func TestEndOfSpeechReached_RequiresMinActiveFrames(t *testing.T) {
	// Unchanged floor: >18 active frames (matches the original behavior),
	// so a long silence with almost no real speech still doesn't trigger.
	silenceFrames := 110
	minTotal := 150
	if endOfSpeechReached(200, 10, 300, silenceFrames, minTotal) {
		t.Fatalf("expected too few active frames to NOT trigger end of speech")
	}
	if !endOfSpeechReached(200, 19, 300, silenceFrames, minTotal) {
		t.Fatalf("expected 19 active frames (just above the 18 floor) to be sufficient")
	}
}

// --- config resolution: 0/unset falls back to the new defaults ---

func TestEndOfSpeechSilenceFrames_DefaultsWhenUnset(t *testing.T) {
	got := endOfSpeechSilenceFrames()
	want := DefaultEndOfSpeechSilenceMs / vadFrameMs
	if got != want {
		t.Fatalf("got %d frames, want %d (default %dms / %dms per frame)", got, want, DefaultEndOfSpeechSilenceMs, vadFrameMs)
	}
}

func TestMinTotalAudioFrames_DefaultsWhenUnset(t *testing.T) {
	got := minTotalAudioFrames()
	want := DefaultMinTotalAudioMs / vadFrameMs
	if got != want {
		t.Fatalf("got %d frames, want %d (default %dms / %dms per frame)", got, want, DefaultMinTotalAudioMs, vadFrameMs)
	}
}
