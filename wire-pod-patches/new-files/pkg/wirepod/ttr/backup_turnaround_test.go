package wirepod_ttr

import (
	"testing"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// backupTurnaroundIntents mirrors the real chipper/intent-data/en-US.json
// entries this test cares about, copied verbatim as of the 2026-09-28
// bare-"back"/bare-"around" misfire fix (see
// intent-data/en-US.json.bak-20260928-eof for the pre-change version).
// Bug report: "go back to it ... you were doing good" matched
// intent_imperative_backup on the bare "back" keyphrase and reversed the
// robot instead of letting the LLM answer. "look around the room" matched
// intent_imperative_turnaround on the bare "around" keyphrase and spun the
// robot. Fix: require explicit phrasing for both.
func backupTurnaroundIntents() []vars.JsonIntent {
	return []vars.JsonIntent{
		{
			Name:       "intent_imperative_backup",
			Keyphrases: []string{"back up", "backup", "backwards", "go backwards", "reverse", "move back", "beck"},
		},
		{
			Name:       "intent_imperative_turnaround",
			Keyphrases: []string{"turn around", "one eighty", "one ate he"},
		},
		{
			Name:       "intent_system_charger",
			Keyphrases: []string{"charge", "home", "go to your", "church", "find your ch", "charger"},
		},
	}
}

// TestFindIntentMatch_BackupMisfires_NoLongerMatch is the exact 2026-09-28
// regression case plus the coordinator's other named phrases: none of
// these should match intent_imperative_backup any more now that the bare
// "back" keyphrase is gone. With no CORE match, these fall through to the
// LLM (Found=false or a different intent), which is the desired outcome.
func TestFindIntentMatch_BackupMisfires_NoLongerMatch(t *testing.T) {
	phrases := []string{
		"go back to it you were doing good",
		"i'm back",
		"im back",
		"come back",
		"back to the charger",
	}
	for _, phrase := range phrases {
		t.Run(phrase, func(t *testing.T) {
			res := findIntentMatch(phrase, backupTurnaroundIntents(), true)
			if res.Found && res.IntentName == "intent_imperative_backup" {
				t.Fatalf("expected %q to NOT match intent_imperative_backup, but it did (keyphrase %q)", phrase, res.Keyphrase)
			}
		})
	}
}

// TestFindIntentMatch_BackupExplicitPhrases_StillMatch confirms the
// explicit backup phrasings the coordinator asked for still work as CORE
// matches.
func TestFindIntentMatch_BackupExplicitPhrases_StillMatch(t *testing.T) {
	phrases := []string{
		"back up",
		"backup",
		"backwards",
		"go backwards",
		"reverse",
		"move back",
	}
	for _, phrase := range phrases {
		t.Run(phrase, func(t *testing.T) {
			res := findIntentMatch(phrase, backupTurnaroundIntents(), true)
			if !res.Found {
				t.Fatalf("expected %q to match a CORE intent, got no match", phrase)
			}
			if res.IntentName != "intent_imperative_backup" {
				t.Fatalf("expected %q to match intent_imperative_backup, got %q", phrase, res.IntentName)
			}
		})
	}
}

// TestFindIntentMatch_TurnaroundMisfire_NoLongerMatches is the
// "look around the room" case from the bug report: this should NOT spin
// the robot now that bare "around" is gone.
func TestFindIntentMatch_TurnaroundMisfire_NoLongerMatches(t *testing.T) {
	phrases := []string{
		"look around the room",
		"look around",
		"what's around here",
	}
	for _, phrase := range phrases {
		t.Run(phrase, func(t *testing.T) {
			res := findIntentMatch(phrase, backupTurnaroundIntents(), true)
			if res.Found && res.IntentName == "intent_imperative_turnaround" {
				t.Fatalf("expected %q to NOT match intent_imperative_turnaround, but it did (keyphrase %q)", phrase, res.Keyphrase)
			}
		})
	}
}

// TestFindIntentMatch_TurnAround_StillMatches confirms the explicit
// "turn around" phrasing (and the numeric variants) still work.
func TestFindIntentMatch_TurnAround_StillMatches(t *testing.T) {
	phrases := []string{
		"turn around",
		"can you turn around",
		"one eighty",
	}
	for _, phrase := range phrases {
		t.Run(phrase, func(t *testing.T) {
			res := findIntentMatch(phrase, backupTurnaroundIntents(), true)
			if !res.Found {
				t.Fatalf("expected %q to match a CORE intent, got no match", phrase)
			}
			if res.IntentName != "intent_imperative_turnaround" {
				t.Fatalf("expected %q to match intent_imperative_turnaround, got %q", phrase, res.IntentName)
			}
		})
	}
}
