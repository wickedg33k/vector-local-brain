package wirepod_ttr

import "testing"

// --- parseClassifyResponse ---

func TestParseClassifyResponse_ExactIntentName(t *testing.T) {
	got := parseClassifyResponse("intent_explore_start")
	if got != "intent_explore_start" {
		t.Fatalf("got %q, want intent_explore_start", got)
	}
}

func TestParseClassifyResponse_WhitespaceAndPunctuationTrimmed(t *testing.T) {
	got := parseClassifyResponse("  intent_system_charger.\n")
	if got != "intent_system_charger" {
		t.Fatalf("got %q, want intent_system_charger", got)
	}
}

func TestParseClassifyResponse_QuotedIntentName(t *testing.T) {
	got := parseClassifyResponse(`"intent_imperative_dance"`)
	if got != "intent_imperative_dance" {
		t.Fatalf("got %q, want intent_imperative_dance", got)
	}
}

func TestParseClassifyResponse_NoneCaseInsensitive(t *testing.T) {
	for _, in := range []string{"NONE", "none", "None.", "  none  "} {
		if got := parseClassifyResponse(in); got != "" {
			t.Fatalf("input %q: got %q, want \"\" (NONE)", in, got)
		}
	}
}

func TestParseClassifyResponse_EmptyString(t *testing.T) {
	if got := parseClassifyResponse(""); got != "" {
		t.Fatalf("got %q, want \"\"", got)
	}
}

func TestParseClassifyResponse_HallucinatedIntentNameRejected(t *testing.T) {
	// Not on AllowedLLMIntents -- must fail safe to "", never be trusted.
	if got := parseClassifyResponse("intent_made_up_thing"); got != "" {
		t.Fatalf("got %q, want \"\" (unwhitelisted name must be rejected)", got)
	}
}

func TestParseClassifyResponse_RamblingExplanationRejected(t *testing.T) {
	// Model ignored "nothing else" -- first line isn't a bare intent name,
	// so this must fail safe rather than partial-match.
	got := parseClassifyResponse("I think the answer is intent_explore_start because they said roam")
	if got != "" {
		t.Fatalf("got %q, want \"\" (rambling output must be rejected, not partial-matched)", got)
	}
}

func TestParseClassifyResponse_LeakedThinkTagStripped(t *testing.T) {
	got := parseClassifyResponse("<think>the user wants to explore</think>intent_explore_start")
	if got != "intent_explore_start" {
		t.Fatalf("got %q, want intent_explore_start", got)
	}
}

func TestParseClassifyResponse_OnlyFirstLineConsidered(t *testing.T) {
	got := parseClassifyResponse("intent_explore_start\nsome extra text on a second line")
	if got != "intent_explore_start" {
		t.Fatalf("got %q, want intent_explore_start", got)
	}
}

// --- buildClassifyPrompt ---

func TestBuildClassifyPrompt_ContainsUtteranceAndAllAllowedIntents(t *testing.T) {
	prompt := buildClassifyPrompt("go to your charger")
	if !containsStr(prompt, "go to your charger") {
		t.Fatalf("expected prompt to contain the utterance verbatim")
	}
	for _, name := range AllowedLLMIntents {
		if !containsStr(prompt, name) {
			t.Fatalf("expected prompt to list allowed intent %q", name)
		}
	}
	if !containsStr(prompt, "NONE") {
		t.Fatalf("expected prompt to instruct a NONE fallback")
	}
	if !containsStr(prompt, "/no_think") {
		t.Fatalf("expected prompt to end with /no_think for qwen3")
	}
}

func containsStr(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (func() bool {
		for i := 0; i+len(needle) <= len(haystack); i++ {
			if haystack[i:i+len(needle)] == needle {
				return true
			}
		}
		return false
	})()
}

// --- stripThinkTags ---

func TestStripThinkTags_RemovesBlock(t *testing.T) {
	got := stripThinkTags("<think>reasoning here</think>NONE")
	if got != "NONE" {
		t.Fatalf("got %q, want NONE", got)
	}
}

func TestStripThinkTags_NoTagsUnchanged(t *testing.T) {
	got := stripThinkTags("intent_explore_start")
	if got != "intent_explore_start" {
		t.Fatalf("got %q, want unchanged", got)
	}
}

func TestStripThinkTags_UnterminatedLeftAlone(t *testing.T) {
	in := "<think>never closes intent_explore_start"
	got := stripThinkTags(in)
	if got != in {
		t.Fatalf("got %q, want unchanged input for an unterminated tag", got)
	}
}
