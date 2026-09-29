package wirepod_ttr

import "testing"

// TestGetActionsFromString_SingleSegmentTagNoPanic is the exact 2026-09-29
// crash repro: panic: runtime error: index out of range [1] with length 1
// at kgsim_cmds.go:292, triggered by the LLM emitting {{intent_play_popawheelie}}
// (no "||" separator) instead of {{doIntent||intent_play_popawheelie}}.
func TestGetActionsFromString_SingleSegmentTagNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("GetActionsFromString panicked on a no-|| tag: %v", r)
		}
	}()
	acts := GetActionsFromString("{{intent_play_popawheelie}}")
	// "intent_play_popawheelie" is not a recognized ValidLLMCommands name
	// (that's "doIntent", not a bare intent name), so CmdParamToAction
	// safely logs+ignores it -- no action is added for the tag itself. The
	// trailing empty text after "}}" still produces a harmless no-op
	// sayText("") action, same as the pre-existing (unrelated to this fix)
	// behavior for any tag at the very end of the string. The point of
	// this test is that it must never panic.
	if len(acts) != 1 || acts[0].Action != ActionSayText || acts[0].Parameter != "" {
		t.Fatalf("expected a single no-op sayText(\"\") action, got %v", acts)
	}
}

func TestGetActionsFromString_TextThenSingleSegmentTagNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("GetActionsFromString panicked: %v", r)
		}
	}()
	_ = GetActionsFromString("Sure thing! {{intent_play_popawheelie}} enjoy")
}

func TestGetActionsFromString_MultipleSingleSegmentTagsNoPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("GetActionsFromString panicked: %v", r)
		}
	}()
	_ = GetActionsFromString("{{foo}}{{bar}}{{baz}}")
}

// Regression guard: the normal two-segment {{command||param}} form must
// still work exactly as before the length-guard fix.
func TestGetActionsFromString_NormalCommandWithParamStillWorks(t *testing.T) {
	acts := GetActionsFromString("{{doIntent||intent_explore_start}}")
	// Pre-existing (unrelated to the length-guard fix) behavior: a tag at
	// the very end of the string also yields a trailing no-op sayText("")
	// for whatever (nothing) follows "}}". The doIntent action itself is
	// what matters here.
	if len(acts) != 2 {
		t.Fatalf("expected 2 actions (doIntent + trailing no-op text), got %v", acts)
	}
	if acts[0].Action != ActionDoIntent || acts[0].Parameter != "intent_explore_start" {
		t.Fatalf("expected doIntent(intent_explore_start) first, got %+v", acts[0])
	}
	if acts[1].Action != ActionSayText || acts[1].Parameter != "" {
		t.Fatalf("expected trailing no-op sayText(\"\"), got %+v", acts[1])
	}
}

func TestGetActionsFromString_CommandWithParamAndTrailingTextStillWorks(t *testing.T) {
	acts := GetActionsFromString("{{playAnimationWI||happy}} great job")
	if len(acts) != 2 {
		t.Fatalf("expected 2 actions (animation + trailing text), got %v", acts)
	}
	if acts[0].Action != ActionPlayAnimationWI || acts[0].Parameter != "happy" {
		t.Fatalf("expected playAnimationWI(happy) first, got %+v", acts[0])
	}
	if acts[1].Action != ActionSayText || acts[1].Parameter != "great job" {
		t.Fatalf("expected trailing sayText(great job), got %+v", acts[1])
	}
}

func TestGetActionsFromString_PlainTextNoTags(t *testing.T) {
	acts := GetActionsFromString("just talking, no commands here")
	if len(acts) != 1 || acts[0].Action != ActionSayText {
		t.Fatalf("expected a single sayText action, got %v", acts)
	}
}
