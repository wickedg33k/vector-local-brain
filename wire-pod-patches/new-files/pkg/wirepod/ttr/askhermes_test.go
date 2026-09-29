package wirepod_ttr

import (
	"testing"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// --- extractHermesQuestion: the Go port of ask_hermes.py's TRIGGER_RE
// stripping, WITHOUT the Python fallback-to-original-text bug. ---

func TestExtractHermesQuestion_BareTriggerIsEmpty(t *testing.T) {
	cases := []string{
		"ask hermes",
		"hey hermes",
		"tell hermes",
		"hermes",
		"ask her me",
		"hermies",
	}
	for _, c := range cases {
		t.Run(c, func(t *testing.T) {
			got := extractHermesQuestion(c)
			if got != "" {
				t.Fatalf("expected empty question for bare trigger %q, got %q", c, got)
			}
		})
	}
}

func TestExtractHermesQuestion_RealQuestionIsExtracted(t *testing.T) {
	cases := map[string]string{
		"ask hermes what's the weather":       "what's the weather",
		"hey hermes, what time is it":         "what time is it",
		"tell hermes to remind me about taxes": "remind me about taxes",
	}
	for input, want := range cases {
		t.Run(input, func(t *testing.T) {
			got := extractHermesQuestion(input)
			if got != want {
				t.Fatalf("got %q, want %q", got, want)
			}
		})
	}
}

// --- customIntentHandler's ask_hermes branch uses this length check ---

func TestMinHermesQuestionLen_BoundaryBehavior(t *testing.T) {
	// "hi" (2 chars) should be treated as too short; "yes" (3 chars) as
	// long enough -- documents the exact boundary the fix uses.
	if len("hi") >= minHermesQuestionLen {
		t.Fatalf("expected 2-char input to be below minHermesQuestionLen (%d)", minHermesQuestionLen)
	}
	if len("yes") < minHermesQuestionLen {
		t.Fatalf("expected 3-char input to meet minHermesQuestionLen (%d)", minHermesQuestionLen)
	}
}

// --- runAskHermesWithQuestion: not-configured fallback (safe to test --
// never reaches exec.Command since intent_ask_hermes isn't found) ---

func TestRunAskHermesWithQuestion_NotConfigured(t *testing.T) {
	orig := vars.CustomIntents
	vars.CustomIntents = []vars.CustomIntent{
		{Name: "some_other_custom_intent"},
	}
	t.Cleanup(func() { vars.CustomIntents = orig })

	got := runAskHermesWithQuestion(nil, "test-esn-hermes-not-configured", "what's the weather")
	if got {
		t.Fatalf("expected false when intent_ask_hermes isn't configured")
	}
}
