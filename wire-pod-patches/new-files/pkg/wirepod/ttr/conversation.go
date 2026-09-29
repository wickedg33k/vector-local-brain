package wirepod_ttr

// conversation.go -- "conversation mode" for Vector, added 2026-09-26d
// (Hermes, the owner-approved). Goal: after wire-pod's LLM path finishes
// speaking a reply and releases behavior control, the robot re-opens its
// mic for a follow-up (same as if it heard "Hey Vector") so the owner doesn't
// have to repeat the wake word for every turn of a conversation.
//
// Design constraints (see the brief this was built from):
//   - Fully config-gated: Knowledge.ConversationMode (bool) and
//     Knowledge.ConversationTimeoutSec (int, seconds).
//   - Fail-safe: any error triggering the follow-up listen (flaky Wi-Fi,
//     robot unreachable) is logged and the conversation is dropped
//     silently -- never retried, never panics.
//   - Ends the conversation (does not re-listen) when: the next utterance
//     is silence/noaudio; a CORE intent was matched; the conversation has
//     been running longer than the timeout since the last user speech; or
//     a safety cap of consecutive auto-listens is hit.
//   - Does not re-listen when the LLM reply itself contained a
//     {{doIntent||...}} command, or when the LLM already asked for a
//     follow-up itself via the existing (stock wire-pod) {{newVoiceRequest||now}}
//     command -- that command already triggers a new voice request via
//     AppIntent (see DoNewRequest in kgsim_cmds.go), so auto-triggering on
//     top of it would double-open the mic.
//
// Fixed 2026-09-26e (Hermes): the original trigger simulated a button press
// via an HTTP GET to the robot's debug console webserver on port 8889 (the
// same mechanism /api-sdk/trigger_wake_word used). That port is CLOSED on
// production escape-pod firmware -- confirmed by TCP connect failures/curl
// timeouts against the owner's real robot -- so the trigger never fired.
// Auto-listen now reuses the SAME mechanism stock wire-pod already uses for
// {{newVoiceRequest||now}}: DoNewRequest's AppIntent("knowledge_question")
// call over the robot's already-open gRPC connection (port 443), which
// production firmware does accept. See DoNewRequest in kgsim_cmds.go.
//
// The state machine itself (decideConversationContinuation) is a pure
// function with no I/O and no global state, so it's unit-testable directly.
// Everything else in this file is the thin I/O layer around it (per-ESN
// state storage, the RPC that actually re-opens the mic).

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// DefaultConversationMaxListens is the safety cap on consecutive
// auto-triggered listens within one conversation, so something like a TV
// left on in the background can't keep the robot listening forever.
const DefaultConversationMaxListens = 6

// DefaultConversationTimeoutSec mirrors the fallback also applied in
// vars.ReadConfig for configs written before this field existed.
const DefaultConversationTimeoutSec = 12

// conversationPostSpeechDelay is how long we wait after speech finishes
// (and behavior control is released) before re-opening the mic, so the
// robot's own speaker output doesn't get picked back up by its mic.
// DoNewRequest (below) adds its own additional ~333ms delay before the
// actual AppIntent call, which is fine -- a little extra margin here is
// harmless and only makes the "don't hear himself" guarantee stronger.
const conversationPostSpeechDelay = 400 * time.Millisecond

// ConversationState tracks the auto-listen conversation for one robot
// (keyed by ESN). Zero value is the correct "no active conversation" state.
type ConversationState struct {
	ESN         string
	Active      bool
	ListenCount int
}

var (
	conversationStates []ConversationState
	conversationMu     sync.Mutex
)

// getConversationState returns the current state for esn, or a fresh
// zero-value state if none is tracked yet.
func getConversationState(esn string) ConversationState {
	conversationMu.Lock()
	defer conversationMu.Unlock()
	for _, cs := range conversationStates {
		if cs.ESN == esn {
			return cs
		}
	}
	return ConversationState{ESN: esn}
}

// placeConversationState upserts state for its ESN.
func placeConversationState(cs ConversationState) {
	conversationMu.Lock()
	defer conversationMu.Unlock()
	for i, existing := range conversationStates {
		if existing.ESN == cs.ESN {
			conversationStates[i] = cs
			return
		}
	}
	conversationStates = append(conversationStates, cs)
}

// EndConversation resets any tracked conversation state for esn. Call this
// whenever something should stop the auto-listen chain: silence/noaudio,
// a matched CORE intent, or the state machine itself deciding not to
// continue (cap/timeout/disabled/doIntent/newVoiceRequest).
func EndConversation(esn string) {
	placeConversationState(ConversationState{ESN: esn})
}

// conversationDecisionInput bundles everything decideConversationContinuation
// needs as plain values, so it never touches global config/state/network and
// stays trivially unit-testable.
type conversationDecisionInput struct {
	ConversationModeEnabled bool
	ReplyText               string
	ListenCount             int
	MaxListens              int
	TimeSinceLastUserSpeech time.Duration
	Timeout                 time.Duration
}

type conversationDecision struct {
	ShouldListen bool
	// Reason is a short machine-checkable tag for logging/tests:
	// "disabled" | "doIntent" | "newVoiceRequest" | "cap" | "timeout" | "ok"
	Reason string
}

// decideConversationContinuation is the pure state-machine decision: given
// the current turn's inputs, should wire-pod re-open the mic for a
// follow-up? See the file header for the full rule list.
func decideConversationContinuation(in conversationDecisionInput) conversationDecision {
	if !in.ConversationModeEnabled {
		return conversationDecision{ShouldListen: false, Reason: "disabled"}
	}
	if strings.Contains(in.ReplyText, "{{doIntent") {
		return conversationDecision{ShouldListen: false, Reason: "doIntent"}
	}
	if strings.Contains(in.ReplyText, "{{newVoiceRequest") {
		// The LLM already asked for a follow-up itself via the stock
		// wire-pod mechanism (AppIntent "knowledge_question", see
		// DoNewRequest) -- don't also trigger our own, or the mic would
		// open twice.
		return conversationDecision{ShouldListen: false, Reason: "newVoiceRequest"}
	}
	if in.ListenCount >= in.MaxListens {
		return conversationDecision{ShouldListen: false, Reason: "cap"}
	}
	if in.TimeSinceLastUserSpeech > in.Timeout {
		return conversationDecision{ShouldListen: false, Reason: "timeout"}
	}
	return conversationDecision{ShouldListen: true, Reason: "ok"}
}

// conversationTimeout resolves the configured timeout, falling back to the
// default if unset (defensive -- vars.ReadConfig already normalizes this on
// load, this is a second fail-safe for callers/tests that build config in
// memory without going through ReadConfig).
func conversationTimeout() time.Duration {
	secs := vars.APIConfig.Knowledge.ConversationTimeoutSec
	if secs <= 0 {
		secs = DefaultConversationTimeoutSec
	}
	return time.Duration(secs) * time.Second
}

// connectRobotByESN mirrors the connection pattern already used in
// StreamingKGSim/KGSim: look up the robot's GUID+IP in vars.BotInfo.Robots
// by ESN and open a fresh SDK connection. Kept ESN-driven (rather than
// threading the *vector.Vector the caller already has through
// MaybeContinueConversation) so the trigger stays self-contained and
// callable from anywhere that only has an ESN -- and so a stale/closed
// connection from the just-finished response can't take the follow-up
// trigger down with it.
func connectRobotByESN(esn string) (*vector.Vector, error) {
	for _, bot := range vars.BotInfo.Robots {
		if bot.Esn == esn {
			return vector.New(vector.WithSerialNo(esn), vector.WithToken(bot.GUID), vector.WithTarget(bot.IPAddress+":443"))
		}
	}
	return nil, fmt.Errorf("no robot registered for ESN %s", esn)
}

// triggerFollowUpListenFn is a package-level indirection so tests can stub
// out the actual robot RPC. Production code should never reassign this
// outside of tests.
var triggerFollowUpListenFn = TriggerFollowUpListen

// TriggerFollowUpListen re-opens the robot's mic for a follow-up using the
// SAME mechanism stock wire-pod already uses for {{newVoiceRequest||now}}
// (see DoNewRequest in kgsim_cmds.go): AppIntent("knowledge_question") over
// the robot's gRPC connection (port 443). This is what production
// escape-pod firmware actually accepts -- see the 2026-09-26e fix note in
// the file header for why the earlier HTTP/port-8889 approach didn't work.
func TriggerFollowUpListen(esn string) error {
	robot, err := connectRobotByESN(esn)
	if err != nil {
		return err
	}
	return DoNewRequest(robot)
}

// MaybeContinueConversation is called once an LLM reply has finished
// speaking and behavior control has been released (see the end of
// StreamingKGSim in kgsim.go). It decides -- via decideConversationContinuation
// -- whether to re-open the robot's mic for a follow-up, and if so, does it.
//
// lastUserSpeechAt should be the timestamp the current turn's user utterance
// was received (captured at the top of StreamingKGSim), NOT the time this
// function is called -- that's what lets the timeout rule catch a
// pathologically slow LLM/TTS round trip eating into the conversation
// window.
//
// Fails safe: any error is logged and the conversation state is reset
// (dropped silently), never retried.
func MaybeContinueConversation(esn, replyText string, lastUserSpeechAt time.Time) {
	cs := getConversationState(esn)

	decision := decideConversationContinuation(conversationDecisionInput{
		ConversationModeEnabled: vars.APIConfig.Knowledge.ConversationMode,
		ReplyText:               replyText,
		ListenCount:             cs.ListenCount,
		MaxListens:              DefaultConversationMaxListens,
		TimeSinceLastUserSpeech: time.Since(lastUserSpeechAt),
		Timeout:                 conversationTimeout(),
	})

	if !decision.ShouldListen {
		if decision.Reason != "disabled" {
			logger.Println("Conversation mode: not continuing for " + esn + " (reason: " + decision.Reason + ")")
		}
		EndConversation(esn)
		return
	}

	// Give the robot's own speech a moment to finish leaving the speaker
	// before we open the mic back up, so he doesn't hear himself.
	time.Sleep(conversationPostSpeechDelay)

	if err := triggerFollowUpListenFn(esn); err != nil {
		logger.Println("Conversation mode: failed to trigger follow-up listen for " + esn + ": " + err.Error())
		EndConversation(esn)
		return
	}

	cs.ESN = esn
	cs.Active = true
	cs.ListenCount++
	placeConversationState(cs)
	logger.Println(fmt.Sprintf("Conversation mode: auto-listen triggered for %s (turn %d/%d)", esn, cs.ListenCount, DefaultConversationMaxListens))
}
