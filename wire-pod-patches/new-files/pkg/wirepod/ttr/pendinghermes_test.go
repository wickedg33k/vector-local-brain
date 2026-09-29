package wirepod_ttr

import (
	"testing"
	"time"
)

func TestConsumePendingHermesQuestion_NotArmed(t *testing.T) {
	if ConsumePendingHermesQuestion("esn-never-armed-hermes") {
		t.Fatalf("expected false when nothing was ever recorded")
	}
}

func TestConsumePendingHermesQuestion_ArmedAndFresh(t *testing.T) {
	esn := "test-esn-hermes-fresh"
	RecordPendingHermesQuestion(esn)

	if !ConsumePendingHermesQuestion(esn) {
		t.Fatalf("expected a freshly armed flag to be consumed as true")
	}
	// Consuming clears it -- a second call must report false.
	if ConsumePendingHermesQuestion(esn) {
		t.Fatalf("expected the flag to be cleared after being consumed once")
	}
}

func TestConsumePendingHermesQuestion_Expired(t *testing.T) {
	esn := "test-esn-hermes-expired"
	recordPendingHermesQuestionWithExpiry(esn, time.Now().Add(-1*time.Second))

	if ConsumePendingHermesQuestion(esn) {
		t.Fatalf("expected an expired flag to report false")
	}
	// Still consumed (removed) even though expired -- no lingering entry.
	if ConsumePendingHermesQuestion(esn) {
		t.Fatalf("expected the expired entry to have been removed on first consume")
	}
}

func TestRecordPendingHermesQuestion_ReplacesForSameESN(t *testing.T) {
	esn := "test-esn-hermes-replace"
	recordPendingHermesQuestionWithExpiry(esn, time.Now().Add(-1*time.Second)) // armed but already expired
	RecordPendingHermesQuestion(esn)                                          // re-armed fresh

	if !ConsumePendingHermesQuestion(esn) {
		t.Fatalf("expected the fresh re-arm to replace the expired entry, not stack alongside it")
	}
}
