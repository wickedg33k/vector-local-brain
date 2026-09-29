package wirepod_ttr

import (
	"testing"
	"time"
)

// --- behaviorControlShouldRelease: the adaptive watchdog decision,
// 2026-09-27, replacing the flat 30s deadline that cut off a legitimate
// ~35s+ getImage (vision) reply. ---

func TestBehaviorControlShouldRelease_ActiveReplyIsNotCutOff(t *testing.T) {
	// The exact regression: a getImage turn easily exceeds 30s (intro ->
	// capture -> ~6s vision model -> answer) but keeps making progress the
	// whole time -- must NOT be released just because total hold time
	// passed the old flat 30s mark, as long as activity is recent.
	should, _ := behaviorControlShouldRelease(45*time.Second, 2*time.Second, 20*time.Second, 90*time.Second)
	if should {
		t.Fatalf("expected an actively-progressing 45s-held reply to NOT be released")
	}
}

func TestBehaviorControlShouldRelease_InactivityTriggersRelease(t *testing.T) {
	should, reason := behaviorControlShouldRelease(25*time.Second, 21*time.Second, 20*time.Second, 90*time.Second)
	if !should || reason != "inactivity watchdog" {
		t.Fatalf("expected inactivity watchdog to release, got should=%v reason=%q", should, reason)
	}
}

func TestBehaviorControlShouldRelease_JustUnderInactivityIsFine(t *testing.T) {
	should, _ := behaviorControlShouldRelease(25*time.Second, 19999*time.Millisecond, 20*time.Second, 90*time.Second)
	if should {
		t.Fatalf("expected just-under-inactivity-timeout to still be fine")
	}
}

func TestBehaviorControlShouldRelease_AbsoluteCapAlwaysWins(t *testing.T) {
	// Even with activity 1ms ago, the absolute cap is a hard ceiling --
	// this is what keeps the original freeze-safety-net property (control
	// can never be held forever) even for a pathological case that somehow
	// keeps "touching" activity indefinitely.
	should, reason := behaviorControlShouldRelease(90*time.Second, 1*time.Millisecond, 20*time.Second, 90*time.Second)
	if !should || reason != "absolute cap" {
		t.Fatalf("expected the absolute cap to release regardless of recent activity, got should=%v reason=%q", should, reason)
	}
}

func TestBehaviorControlShouldRelease_JustUnderAbsoluteCapIsFine(t *testing.T) {
	should, _ := behaviorControlShouldRelease(89999*time.Millisecond, 1*time.Millisecond, 20*time.Second, 90*time.Second)
	if should {
		t.Fatalf("expected just-under-the-absolute-cap to still be fine")
	}
}

// --- TouchBehaviorControlActivity / getBehaviorControlActivity ---

func TestTouchBehaviorControlActivity_UpdatesTimestamp(t *testing.T) {
	esn := "test-esn-bcontrol-touch"
	before := getBehaviorControlActivity(esn) // zero value if never touched

	TouchBehaviorControlActivity(esn)
	after := getBehaviorControlActivity(esn)

	if !after.After(before) {
		t.Fatalf("expected TouchBehaviorControlActivity to advance the recorded timestamp, before=%v after=%v", before, after)
	}
}
