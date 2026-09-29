package wirepod_ttr

// pendinghermes.go -- added 2026-09-27. the owner: "he is cutting me off" --
// when the ask_hermes custom intent fires with an empty/very short
// question (just "ask hermes" with nothing after it), Vector now asks for
// clarification and listens for the actual question as a follow-up,
// instead of either (a) sending the bare trigger phrase itself to Hermes
// as if it were the question (the OLD behavior --
// ask_hermes.py's extract_question fell back to the full original text,
// including the trigger phrase, when nothing was left after stripping it),
// or (b) saying nothing useful.
//
// This has to live pod-side (not in ask_hermes.py): routing the NEXT,
// separate voice request to ask_hermes instead of the LLM is a wire-pod
// intent-routing decision (ProcessTextAll in matchIntentSend.go), which a
// backgrounded Python script invoked via exec has no way to influence.
//
// See matchIntentSend.go: customIntentHandler's intent_ask_hermes case
// records the pending flag (via RecordPendingHermesQuestion) and speaks
// the reprompt when the extracted question is too short; ProcessTextAll
// checks ConsumePendingHermesQuestion at the very top, before any other
// routing, and if set, sends the ENTIRE next utterance straight to
// intent_ask_hermes's exec as the question (runAskHermesWithQuestion),
// bypassing CORE/SOCIAL keyword matching and the LLM entirely for that
// turn.

import (
	"sync"
	"time"
)

// pendingHermesQuestionTTL bounds how long a "what do you want to ask
// Hermes?" reprompt stays armed. If nothing (or silence) follows within
// this window, the flag simply expires -- the next real utterance goes
// through normal routing again.
const pendingHermesQuestionTTL = 15 * time.Second

// minHermesQuestionLen is the shortest (trimmed) extracted question that's
// treated as "real" rather than empty/a bare trigger phrase.
const minHermesQuestionLen = 3

// hermesRepromptText is spoken when ask_hermes fires with no real question
// attached.
const hermesRepromptText = "Sure, what do you want to ask Hermes?"

// hermesRepromptFollowUpDelay is how long to wait after starting the
// reprompt speech before re-opening the mic for the follow-up -- KGSim
// speaks asynchronously with no synchronous "done" signal back to the
// caller, so this is a fixed, generous estimate for a short phrase (same
// ad-hoc-timing convention already used throughout this codebase, e.g.
// DoNewRequest's 333ms delay, conversationPostSpeechDelay).
const hermesRepromptFollowUpDelay = 2500 * time.Millisecond

type pendingHermesEntry struct {
	ESN       string
	ExpiresAt time.Time
}

var (
	pendingHermesQuestions   []pendingHermesEntry
	pendingHermesQuestionsMu sync.Mutex
)

// RecordPendingHermesQuestion arms the reprompt-followup flag for esn,
// replacing any existing one.
func RecordPendingHermesQuestion(esn string) {
	recordPendingHermesQuestionWithExpiry(esn, time.Now().Add(pendingHermesQuestionTTL))
}

// recordPendingHermesQuestionWithExpiry is the same as
// RecordPendingHermesQuestion but takes an explicit expiry -- split out so
// tests can arm an already-expired entry without a real 15s wait.
func recordPendingHermesQuestionWithExpiry(esn string, expiresAt time.Time) {
	pendingHermesQuestionsMu.Lock()
	defer pendingHermesQuestionsMu.Unlock()
	entry := pendingHermesEntry{ESN: esn, ExpiresAt: expiresAt}
	for i, e := range pendingHermesQuestions {
		if e.ESN == esn {
			pendingHermesQuestions[i] = entry
			return
		}
	}
	pendingHermesQuestions = append(pendingHermesQuestions, entry)
}

// ConsumePendingHermesQuestion reports whether esn has a non-expired
// pending "what do you want to ask Hermes?" flag, removing it either way
// (expired or not) so a stale entry never lingers to be found twice.
func ConsumePendingHermesQuestion(esn string) bool {
	pendingHermesQuestionsMu.Lock()
	defer pendingHermesQuestionsMu.Unlock()
	for i, e := range pendingHermesQuestions {
		if e.ESN == esn {
			pendingHermesQuestions = append(pendingHermesQuestions[:i], pendingHermesQuestions[i+1:]...)
			return time.Now().Before(e.ExpiresAt)
		}
	}
	return false
}
