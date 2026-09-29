package wirepod_ttr

import (
	"testing"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// commandIntents mirrors the real chipper/intent-data/en-US.json entries
// this test cares about, copied verbatim as of the 2026-09-27 doIntent fix
// (see intent-data/en-US.json.bak-20260927-doint for the pre-change
// version) -- explore/dance/charger, the three families the 2026-09-27
// bug report asked to check.
func commandIntents() []vars.JsonIntent {
	return []vars.JsonIntent{
		{
			Name:       "intent_explore_start",
			Keyphrases: []string{"start", "plor", "owing", "tailoring", "oding", "oring", "pling", "start exploring", "explore mode", "explorer mode", "exploring mode", "go explore", "go exploring"},
		},
		{
			Name:       "intent_system_charger",
			Keyphrases: []string{"charge", "home", "go to your", "church", "find your ch", "charger"},
		},
		{
			Name:       "intent_imperative_dance",
			Keyphrases: []string{"dance", "dancing", "thence", "dance to the beat", "the beat", "boogie", "to the music"},
		},
	}
}

// TestFindIntentMatch_ExploreModeVariants_MatchCoreDirectly covers the
// exact phrasings the 2026-09-27 bug report asked for: these must match
// intent_explore_start as a CORE keyword match (Found=true), never fall
// through to the LLM, regardless of llm_first.
func TestFindIntentMatch_ExploreModeVariants_MatchCoreDirectly(t *testing.T) {
	phrases := []string{
		"explore mode",
		"go and explore mode",
		"explorer mode",
		"exploring mode",
		"go explore",
		"go exploring",
	}
	for _, phrase := range phrases {
		t.Run(phrase, func(t *testing.T) {
			res := findIntentMatch(phrase, commandIntents(), true)
			if !res.Found {
				t.Fatalf("expected %q to match a CORE intent, got no match (FellBackFromSocial=%v)", phrase, res.FellBackFromSocial)
			}
			if res.IntentName != "intent_explore_start" {
				t.Fatalf("expected %q to match intent_explore_start, got %q", phrase, res.IntentName)
			}
		})
	}
}

// TestFindIntentMatch_DanceAndCharger_AlreadyMatchCoreDirectly is the
// "sanity-check dance / go home / go to your charger" ask -- these already
// matched before this round's change (via existing substring keyphrases),
// confirmed here so a future edit can't silently regress them.
func TestFindIntentMatch_DanceAndCharger_AlreadyMatchCoreDirectly(t *testing.T) {
	cases := []struct {
		phrase       string
		wantIntent   string
	}{
		{"dance", "intent_imperative_dance"},
		{"do a dance", "intent_imperative_dance"},
		{"go home", "intent_system_charger"},
		{"go to your charger", "intent_system_charger"},
	}
	for _, c := range cases {
		t.Run(c.phrase, func(t *testing.T) {
			res := findIntentMatch(c.phrase, commandIntents(), true)
			if !res.Found {
				t.Fatalf("expected %q to match a CORE intent, got no match", c.phrase)
			}
			if res.IntentName != c.wantIntent {
				t.Fatalf("expected %q to match %s, got %q", c.phrase, c.wantIntent, res.IntentName)
			}
		})
	}
}
