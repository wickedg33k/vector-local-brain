package wirepod_ttr

import (
	"testing"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// realIntents is a representative slice pulled verbatim from
// chipper/intent-data/en-US.json (keyphrases copied exactly) covering the
// four cases in the 2026-09-26c bug report, so these tests exercise real
// production keyphrase data, not a hand-picked toy list.
func realIntents() []vars.JsonIntent {
	return []vars.JsonIntent{
		{
			Name:       "intent_imperative_eyecolor",
			Keyphrases: []string{"eye color", "colo", "i call her", "i foller", "icolor", "ecce", "erior", "ichor", "agricola", "change", "oracular", "oracle", "set your eye color to"},
		},
		{
			Name:       "intent_greeting_goodmorning",
			Keyphrases: []string{"morning", "mourning", "mooning", "it bore", "afternoon", "after noon", "after whom", "good morning"},
		},
		{
			Name:       "intent_imperative_affirmative",
			Keyphrases: []string{"yes", "correct", "sure", "yes please"},
		},
		{
			Name:       "intent_clock_settimer_extend",
			Keyphrases: []string{"timer", "time for", "time of for", "time or", "time of", "set a timer for"},
		},
	}
}

func TestFindIntentMatch_TurnYourEyesRed_GoesToLLM(t *testing.T) {
	// This is the exact regression case from the 2026-09-26c bug report.
	// With the word-boundary fix, "yes" no longer even matches inside
	// "eyes" (no boundary between 'e' and 'y'), so this now falls through
	// to the LLM cleanly with no CORE keyword match and no SOCIAL
	// deferral either -- eye color changes for this phrasing are handled
	// by the LLM's own {{doIntent||intent_imperative_eyecolor}} mechanism,
	// not by keyword matching, so "goes to LLM" IS the correct outcome
	// here, not a bug.
	res := findIntentMatch("turn your eyes red", realIntents(), true)
	if res.Found {
		t.Fatalf("expected no keyword match (should fall through to LLM), got intent %q", res.IntentName)
	}
}

func TestFindIntentMatch_Yes_GoesToLLM(t *testing.T) {
	res := findIntentMatch("yes", realIntents(), true)
	if res.Found {
		t.Fatalf("expected SOCIAL-only match to fall through to LLM, got intent %q", res.IntentName)
	}
	if !res.FellBackFromSocial {
		t.Fatalf("expected FellBackFromSocial=true (deferred SOCIAL match with no CORE alternative)")
	}
}

func TestFindIntentMatch_GoodMorning_GoesToLLM(t *testing.T) {
	res := findIntentMatch("good morning", realIntents(), true)
	if res.Found {
		t.Fatalf("expected SOCIAL-only match to fall through to LLM, got intent %q", res.IntentName)
	}
	if !res.FellBackFromSocial {
		t.Fatalf("expected FellBackFromSocial=true")
	}
}

func TestFindIntentMatch_SetATimer_MatchesTimer(t *testing.T) {
	res := findIntentMatch("set a timer for 5 minutes", realIntents(), true)
	if !res.Found {
		t.Fatalf("expected a CORE match for the timer intent, got none (fell through to LLM)")
	}
	if res.IntentName != "intent_clock_settimer_extend" {
		t.Fatalf("expected intent_clock_settimer_extend, got %q (keyphrase %q)", res.IntentName, res.Keyphrase)
	}
}

// TestFindIntentMatch_SocialDoesNotBlockLaterCoreMatch is a synthetic
// regression test using a SOCIAL keyphrase long enough (>4 chars) that the
// word-boundary fix alone does NOT prevent the collision -- isolating and
// proving the scan-order fix specifically (continue past a deferred SOCIAL
// match instead of aborting the whole search). Before the 2026-09-26c fix,
// this test fails: the old code stopped at "please" and never reached
// "widgetcolor".
func TestFindIntentMatch_SocialDoesNotBlockLaterCoreMatch(t *testing.T) {
	intents := []vars.JsonIntent{
		{Name: "intent_imperative_affirmative", Keyphrases: []string{"please"}},
		{Name: "intent_test_widgetcolor", Keyphrases: []string{"widgetcolor"}},
	}
	res := findIntentMatch("please change to widgetcolor", intents, true)
	if !res.Found {
		t.Fatalf("expected the later CORE match (widgetcolor) to be found, got none -- the SOCIAL match at 'please' incorrectly blocked the scan")
	}
	if res.IntentName != "intent_test_widgetcolor" {
		t.Fatalf("expected intent_test_widgetcolor, got %q", res.IntentName)
	}
	if len(res.DeferredSocialLogs) != 1 {
		t.Fatalf("expected exactly one deferred SOCIAL log entry, got %d: %v", len(res.DeferredSocialLogs), res.DeferredSocialLogs)
	}
}

func TestFindIntentMatch_LLMFirstOff_SocialMatchesNormally(t *testing.T) {
	// Sanity check: with llm_first OFF, SOCIAL intents behave exactly like
	// any other intent (matched immediately, no deferral) -- unchanged
	// from pre-2026-09-26 behavior.
	res := findIntentMatch("yes", realIntents(), false)
	if !res.Found || res.IntentName != "intent_imperative_affirmative" {
		t.Fatalf("expected direct match to intent_imperative_affirmative with llm_first off, got Found=%v Name=%q", res.Found, res.IntentName)
	}
}

func TestKeyphraseMatches_ShortKeyphrasesRequireWordBoundary(t *testing.T) {
	cases := []struct {
		voiceText, keyphrase string
		want                 bool
	}{
		{"turn your eyes red", "yes", false},  // "yes" inside "eyes" -- no boundary, must NOT match
		{"say yes please", "yes", true},       // "yes" as its own word -- must match
		{"the snow is falling", "no", false},  // "no" inside "snow" -- must NOT match
		{"say no thanks", "no", true},         // "no" as its own word -- must match
		{"good morning vector", "morning", true}, // >4 chars: plain substring behavior unchanged
	}
	for _, c := range cases {
		got := keyphraseMatches(c.voiceText, c.keyphrase)
		if got != c.want {
			t.Errorf("keyphraseMatches(%q, %q) = %v, want %v", c.voiceText, c.keyphrase, got, c.want)
		}
	}
}
