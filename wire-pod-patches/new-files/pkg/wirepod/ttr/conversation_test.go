package wirepod_ttr

import (
	"errors"
	"testing"
	"time"

	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// --- decideConversationContinuation: the pure state-machine tests ---

func baseDecisionInput() conversationDecisionInput {
	return conversationDecisionInput{
		ConversationModeEnabled: true,
		ReplyText:               "Sure, here's the weather today.",
		ListenCount:             0,
		MaxListens:              DefaultConversationMaxListens,
		TimeSinceLastUserSpeech: 2 * time.Second,
		Timeout:                 12 * time.Second,
	}
}

func TestDecideConversationContinuation_OK(t *testing.T) {
	d := decideConversationContinuation(baseDecisionInput())
	if !d.ShouldListen || d.Reason != "ok" {
		t.Fatalf("expected ShouldListen=true, reason=ok, got %+v", d)
	}
}

func TestDecideConversationContinuation_Disabled(t *testing.T) {
	in := baseDecisionInput()
	in.ConversationModeEnabled = false
	d := decideConversationContinuation(in)
	if d.ShouldListen || d.Reason != "disabled" {
		t.Fatalf("expected disabled, got %+v", d)
	}
}

func TestDecideConversationContinuation_DoIntentSuppresses(t *testing.T) {
	in := baseDecisionInput()
	in.ReplyText = "Sure! {{doIntent||intent_explore_start}} Off I go."
	d := decideConversationContinuation(in)
	if d.ShouldListen || d.Reason != "doIntent" {
		t.Fatalf("expected doIntent to suppress auto-listen, got %+v", d)
	}
}

func TestDecideConversationContinuation_NewVoiceRequestSuppresses(t *testing.T) {
	// The LLM already asked for a follow-up itself via the stock
	// {{newVoiceRequest||now}} command (see kgsim_cmds.go DoNewRequest) --
	// our own auto-listen must not ALSO fire, or the mic opens twice.
	in := baseDecisionInput()
	in.ReplyText = "Anything else? {{newVoiceRequest||now}}"
	d := decideConversationContinuation(in)
	if d.ShouldListen || d.Reason != "newVoiceRequest" {
		t.Fatalf("expected newVoiceRequest to suppress auto-listen, got %+v", d)
	}
}

func TestDecideConversationContinuation_CapEndsIt(t *testing.T) {
	in := baseDecisionInput()
	in.ListenCount = DefaultConversationMaxListens // already at the cap
	d := decideConversationContinuation(in)
	if d.ShouldListen || d.Reason != "cap" {
		t.Fatalf("expected cap to end the conversation, got %+v", d)
	}
}

func TestDecideConversationContinuation_CapAllowsOneBelowLimit(t *testing.T) {
	in := baseDecisionInput()
	in.ListenCount = DefaultConversationMaxListens - 1
	d := decideConversationContinuation(in)
	if !d.ShouldListen {
		t.Fatalf("expected one below the cap to still be allowed, got %+v", d)
	}
}

func TestDecideConversationContinuation_TimeoutEndsIt(t *testing.T) {
	in := baseDecisionInput()
	in.TimeSinceLastUserSpeech = 13 * time.Second
	in.Timeout = 12 * time.Second
	d := decideConversationContinuation(in)
	if d.ShouldListen || d.Reason != "timeout" {
		t.Fatalf("expected timeout to end the conversation, got %+v", d)
	}
}

func TestDecideConversationContinuation_JustUnderTimeoutIsOK(t *testing.T) {
	in := baseDecisionInput()
	in.TimeSinceLastUserSpeech = 11999 * time.Millisecond
	in.Timeout = 12 * time.Second
	d := decideConversationContinuation(in)
	if !d.ShouldListen {
		t.Fatalf("expected just-under-timeout to still continue, got %+v", d)
	}
}

// --- EndConversation / state storage ---

func TestEndConversation_ResetsActiveState(t *testing.T) {
	esn := "test-esn-end-conv"
	placeConversationState(ConversationState{ESN: esn, Active: true, ListenCount: 4})

	// silence, a CORE intent match, or the state machine itself all call
	// EndConversation the same way.
	EndConversation(esn)

	cs := getConversationState(esn)
	if cs.Active || cs.ListenCount != 0 {
		t.Fatalf("expected reset state after EndConversation, got %+v", cs)
	}
}

func TestGetConversationState_UnknownESNIsZeroValue(t *testing.T) {
	cs := getConversationState("never-seen-this-esn")
	if cs.Active || cs.ListenCount != 0 {
		t.Fatalf("expected zero-value state for unknown ESN, got %+v", cs)
	}
}

// --- connectRobotByESN / TriggerFollowUpListen ---

func TestConnectRobotByESN_NotRegistered(t *testing.T) {
	// vars.BotInfo.Robots has nothing matching this ESN, so this must fail
	// before ever attempting a network dial (vector.New is only reached
	// after a match is found).
	_, err := connectRobotByESN("esn-definitely-not-registered-anywhere")
	if err == nil {
		t.Fatalf("expected an error for an unregistered ESN")
	}
}

// --- MaybeContinueConversation: the orchestration, with the RPC stubbed ---

func withStubTrigger(t *testing.T, fn func(esn string) error) {
	t.Helper()
	orig := triggerFollowUpListenFn
	triggerFollowUpListenFn = fn
	t.Cleanup(func() { triggerFollowUpListenFn = orig })
}

func withConversationConfig(t *testing.T, enabled bool, timeoutSec int) {
	t.Helper()
	origEnabled := vars.APIConfig.Knowledge.ConversationMode
	origTimeout := vars.APIConfig.Knowledge.ConversationTimeoutSec
	vars.APIConfig.Knowledge.ConversationMode = enabled
	vars.APIConfig.Knowledge.ConversationTimeoutSec = timeoutSec
	t.Cleanup(func() {
		vars.APIConfig.Knowledge.ConversationMode = origEnabled
		vars.APIConfig.Knowledge.ConversationTimeoutSec = origTimeout
	})
}

func TestMaybeContinueConversation_TriggersAndIncrementsListenCount(t *testing.T) {
	withConversationConfig(t, true, 12)
	called := 0
	var gotESN string
	withStubTrigger(t, func(esn string) error {
		called++
		gotESN = esn
		return nil
	})
	esn := "test-esn-trigger-ok"
	EndConversation(esn)

	MaybeContinueConversation(esn, "Sure, here you go.", time.Now())

	if called != 1 {
		t.Fatalf("expected the trigger to fire exactly once, called=%d", called)
	}
	if gotESN != esn {
		t.Fatalf("expected trigger to be called with esn=%q, got %q", esn, gotESN)
	}
	cs := getConversationState(esn)
	if !cs.Active || cs.ListenCount != 1 {
		t.Fatalf("expected Active=true, ListenCount=1, got %+v", cs)
	}
}

func TestMaybeContinueConversation_TriggerErrorFailsSafe(t *testing.T) {
	withConversationConfig(t, true, 12)
	withStubTrigger(t, func(esn string) error {
		return errors.New("connection refused (flaky wifi)")
	})
	esn := "test-esn-trigger-err"
	placeConversationState(ConversationState{ESN: esn, Active: true, ListenCount: 2})

	// Must not panic, and must drop the conversation rather than leaving
	// stale "Active" state around.
	MaybeContinueConversation(esn, "Sure, here you go.", time.Now())

	cs := getConversationState(esn)
	if cs.Active {
		t.Fatalf("expected conversation to be dropped after a trigger error, got %+v", cs)
	}
}

func TestMaybeContinueConversation_CapStopsAfterMaxListens(t *testing.T) {
	withConversationConfig(t, true, 12)
	called := 0
	withStubTrigger(t, func(esn string) error {
		called++
		return nil
	})
	esn := "test-esn-cap"
	EndConversation(esn)

	for i := 0; i < DefaultConversationMaxListens+1; i++ {
		MaybeContinueConversation(esn, "Sure, here you go.", time.Now())
	}

	if called != DefaultConversationMaxListens {
		t.Fatalf("expected exactly %d triggers before the cap stops it, got %d", DefaultConversationMaxListens, called)
	}
	cs := getConversationState(esn)
	if cs.Active {
		t.Fatalf("expected conversation to end once the cap is hit, got %+v", cs)
	}
}

func TestMaybeContinueConversation_DisabledNeverTriggers(t *testing.T) {
	withConversationConfig(t, false, 12)
	called := 0
	withStubTrigger(t, func(esn string) error {
		called++
		return nil
	})
	esn := "test-esn-disabled"
	EndConversation(esn)

	MaybeContinueConversation(esn, "Sure, here you go.", time.Now())

	if called != 0 {
		t.Fatalf("expected no trigger when conversation_mode is disabled, called=%d", called)
	}
}

func TestMaybeContinueConversation_TimeoutNeverTriggers(t *testing.T) {
	withConversationConfig(t, true, 1) // 1s timeout
	called := 0
	withStubTrigger(t, func(esn string) error {
		called++
		return nil
	})
	esn := "test-esn-timeout"
	EndConversation(esn)

	oldSpeech := time.Now().Add(-5 * time.Second) // well past the 1s timeout
	MaybeContinueConversation(esn, "Sure, here you go.", oldSpeech)

	if called != 0 {
		t.Fatalf("expected no trigger once the timeout has elapsed, called=%d", called)
	}
	cs := getConversationState(esn)
	if cs.Active {
		t.Fatalf("expected conversation to be ended by the timeout, got %+v", cs)
	}
}

func TestMaybeContinueConversation_DoIntentNeverTriggers(t *testing.T) {
	withConversationConfig(t, true, 12)
	called := 0
	withStubTrigger(t, func(esn string) error {
		called++
		return nil
	})
	esn := "test-esn-dointent"
	EndConversation(esn)

	MaybeContinueConversation(esn, "Sure! {{doIntent||intent_explore_start}}", time.Now())

	if called != 0 {
		t.Fatalf("expected no trigger when the reply used doIntent, called=%d", called)
	}
}
