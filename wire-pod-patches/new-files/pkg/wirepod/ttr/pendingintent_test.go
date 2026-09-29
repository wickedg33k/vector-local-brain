package wirepod_ttr

import (
	"errors"
	"testing"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
)

func TestRecordAndTakePendingIntent_Basic(t *testing.T) {
	esn := "test-esn-pending-basic"
	RecordPendingIntent(esn, "intent_explore_start")

	if !hasPendingIntent(esn) {
		t.Fatalf("expected a pending intent to be recorded")
	}

	got, ok := TakePendingIntent(esn)
	if !ok {
		t.Fatalf("expected TakePendingIntent to find the queued intent")
	}
	if got != "intent_explore_start" {
		t.Fatalf("got %q, want intent_explore_start", got)
	}
	if hasPendingIntent(esn) {
		t.Fatalf("expected TakePendingIntent to consume the entry")
	}
}

func TestTakePendingIntent_NoneQueued(t *testing.T) {
	_, ok := TakePendingIntent("esn-with-nothing-queued-ever")
	if ok {
		t.Fatalf("expected ok=false when nothing is queued")
	}
}

func TestRecordPendingIntent_ReplacesForSameESN(t *testing.T) {
	esn := "test-esn-pending-replace"
	RecordPendingIntent(esn, "intent_explore_start")
	RecordPendingIntent(esn, "intent_system_charger") // a second command this turn should replace, not queue twice

	got, ok := TakePendingIntent(esn)
	if !ok || got != "intent_system_charger" {
		t.Fatalf("expected the most recent intent (intent_system_charger), got %q ok=%v", got, ok)
	}
	if hasPendingIntent(esn) {
		t.Fatalf("expected only one entry to have existed")
	}
}

// --- FirePendingIntentAfterRelease: this is the "fires only after release
// is called" contract -- recording a pending intent must NOT itself cause a
// fire; only calling FirePendingIntentAfterRelease does. The robot
// connection is stubbed out (fireIntentFn), so these pass a nil
// *vector.Vector -- fine, since the stub never dereferences it. ---

func withStubFireIntent(t *testing.T, fn func(intentName string, robot *vector.Vector) error) {
	t.Helper()
	orig := fireIntentFn
	fireIntentFn = fn
	t.Cleanup(func() { fireIntentFn = orig })
}

func TestFirePendingIntentAfterRelease_DoesNotFireBeforeCalled(t *testing.T) {
	esn := "test-esn-fire-not-yet"
	called := false
	withStubFireIntent(t, func(intentName string, robot *vector.Vector) error {
		called = true
		return nil
	})

	RecordPendingIntent(esn, "intent_explore_start")
	// Recording alone must not fire it -- only FirePendingIntentAfterRelease does.
	if called {
		t.Fatalf("expected recording a pending intent to NOT fire it immediately")
	}
	if !hasPendingIntent(esn) {
		t.Fatalf("expected the intent to still be queued before release")
	}
}

func TestFirePendingIntentAfterRelease_FiresQueuedIntent(t *testing.T) {
	esn := "test-esn-fire-ok"
	var gotIntent string
	called := 0
	withStubFireIntent(t, func(intentName string, robot *vector.Vector) error {
		called++
		gotIntent = intentName
		return nil
	})

	RecordPendingIntent(esn, "intent_explore_start")
	FirePendingIntentAfterRelease(esn, nil)

	if called != 1 {
		t.Fatalf("expected exactly one fire, got %d", called)
	}
	if gotIntent != "intent_explore_start" {
		t.Fatalf("got %q, want intent_explore_start", gotIntent)
	}
	if hasPendingIntent(esn) {
		t.Fatalf("expected the queued intent to be consumed after firing")
	}
}

func TestFirePendingIntentAfterRelease_NoOpWhenNothingQueued(t *testing.T) {
	called := 0
	withStubFireIntent(t, func(intentName string, robot *vector.Vector) error {
		called++
		return nil
	})

	FirePendingIntentAfterRelease("esn-nothing-queued-for-fire", nil)

	if called != 0 {
		t.Fatalf("expected no fire when nothing was queued, called=%d", called)
	}
}

func TestFirePendingIntentAfterRelease_ErrorFailsSafe(t *testing.T) {
	esn := "test-esn-fire-error"
	withStubFireIntent(t, func(intentName string, robot *vector.Vector) error {
		return errors.New("rpc error: unavailable (flaky wifi)")
	})

	RecordPendingIntent(esn, "intent_explore_start")

	// Must not panic.
	FirePendingIntentAfterRelease(esn, nil)

	if hasPendingIntent(esn) {
		t.Fatalf("expected the entry to be consumed even when the fire errored")
	}
}

// --- IsAllowedLLMIntent (whitelist check, split out of DoIntent) ---

func TestIsAllowedLLMIntent_Whitelisted(t *testing.T) {
	if !IsAllowedLLMIntent("intent_explore_start") {
		t.Fatalf("expected intent_explore_start to be whitelisted")
	}
}

func TestIsAllowedLLMIntent_NotWhitelisted(t *testing.T) {
	if IsAllowedLLMIntent("intent_clock_settimer_extend") {
		t.Fatalf("expected intent_clock_settimer_extend (a parameterized intent) to NOT be whitelisted for doIntent")
	}
}

func TestIsAllowedLLMIntent_TrimsWhitespace(t *testing.T) {
	if !IsAllowedLLMIntent("  intent_explore_start  ") {
		t.Fatalf("expected whitespace to be trimmed before the whitelist check")
	}
}
