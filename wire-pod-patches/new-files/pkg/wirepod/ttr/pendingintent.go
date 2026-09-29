package wirepod_ttr

// pendingintent.go -- added 2026-09-27. Defers a whitelisted LLM
// {{doIntent||...}} call until AFTER the KG response has finished speaking
// AND behavior control has been released, instead of firing it immediately
// (in a goroutine, after a fixed short delay) WHILE wire-pod still holds
// OVERRIDE_BEHAVIORS for speech.
//
// Root cause of the 2026-09-27 bug ("explore mode" got a spoken
// confirmation -- "Exploring mode activated." -- but the robot never
// moved): DoIntent (kgsim_cmds.go) fired AppIntent from a goroutine ~333ms
// after being called, but PerformActions kept right on speaking whatever
// text followed the {{doIntent||...}} marker in the same LLM response,
// which meant wire-pod was still holding OVERRIDE_BEHAVIORS behavior
// control when the AppIntent for "explore" landed on the robot -- so the
// explore behavior got dropped/overridden by the robot's own behavior
// system instead of actually running.
//
// The fix has two halves:
//  1. PerformActions' ActionDoIntent case (kgsim_cmds.go) now calls
//     RecordPendingIntent and stops speaking immediately (returns
//     disconnect=true), instead of calling DoIntent and continuing.
//  2. kgsim.go's StreamingKGSim calls FirePendingIntentAfterRelease at the
//     SAME point it already calls conversation.go's MaybeContinueConversation
//     -- right after behavior control has been released for this turn.
//
// Conversation mode already skips auto-listening after a doIntent
// (decideConversationContinuation checks the raw reply text for
// "{{doIntent", which is still present in fullfullRespText regardless of
// this change) -- verified by TestDecideConversationContinuation_DoIntentSuppresses
// and TestMaybeContinueConversation_DoIntentNeverTriggers in
// conversation_test.go, unchanged by this fix.

import (
	"sync"
	"time"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
)

// pendingIntentPostReleaseDelay is how long FirePendingIntentAfterRelease
// waits after behavior control is released before firing the AppIntent
// call. The release itself is sent by a separate goroutine (BControl in
// bcontrol.go) with no synchronous confirmation channel back to the
// caller, so this gives the ControlRelease message time to actually land
// on the robot before we ask it to run a new behavior.
const pendingIntentPostReleaseDelay = 300 * time.Millisecond

type pendingIntentEntry struct {
	ESN    string
	Intent string
}

var (
	pendingIntents   []pendingIntentEntry
	pendingIntentsMu sync.Mutex
)

// RecordPendingIntent queues a doIntent call for esn, replacing any
// previously queued (unfired) one for the same ESN -- only the most recent
// command a turn asked for should ever fire.
func RecordPendingIntent(esn, intentName string) {
	pendingIntentsMu.Lock()
	defer pendingIntentsMu.Unlock()
	for i, p := range pendingIntents {
		if p.ESN == esn {
			pendingIntents[i].Intent = intentName
			return
		}
	}
	pendingIntents = append(pendingIntents, pendingIntentEntry{ESN: esn, Intent: intentName})
}

// TakePendingIntent removes and returns the queued intent name for esn, if
// any. Returns ok=false if nothing is queued.
func TakePendingIntent(esn string) (string, bool) {
	pendingIntentsMu.Lock()
	defer pendingIntentsMu.Unlock()
	for i, p := range pendingIntents {
		if p.ESN == esn {
			pendingIntents = append(pendingIntents[:i], pendingIntents[i+1:]...)
			return p.Intent, true
		}
	}
	return "", false
}

// hasPendingIntent reports whether esn currently has a queued intent,
// without consuming it. Test-facing helper.
func hasPendingIntent(esn string) bool {
	pendingIntentsMu.Lock()
	defer pendingIntentsMu.Unlock()
	for _, p := range pendingIntents {
		if p.ESN == esn {
			return true
		}
	}
	return false
}

// fireIntentFn is a package-level indirection so tests can stub out the
// actual robot RPC without a live connection. Production code should never
// reassign this outside of tests.
var fireIntentFn = DoIntentNow

// FirePendingIntentAfterRelease is called from the SAME release point as
// conversation.go's MaybeContinueConversation (see the end of
// StreamingKGSim in kgsim.go, right after behavior control is released for
// a finished response). No-op if nothing is queued for esn -- safe to call
// unconditionally on every release.
//
// Fails safe: an RPC error is logged and dropped, never retried, never
// panics -- matches the fail-safe convention already used for conversation
// mode's own trigger (see MaybeContinueConversation).
//
// Known edge case: if the response is interrupted (touched/waked, see
// InterruptKGSimWhenTouchedOrWaked in kgsim_interrupt.go) rather than
// ending normally, this function is never called for that turn, so a
// queued intent stays queued and will fire at the next turn's release
// point instead. This is intentionally not special-cased -- interruption
// is rare, and RecordPendingIntent always replaces (never queues multiple)
// so a stale intent can only ever be at most one turn old.
func FirePendingIntentAfterRelease(esn string, robot *vector.Vector) {
	intentName, ok := TakePendingIntent(esn)
	if !ok {
		return
	}
	time.Sleep(pendingIntentPostReleaseDelay)
	if err := fireIntentFn(intentName, robot); err != nil {
		logger.Println("Pending doIntent (" + intentName + ") failed for " + esn + ": " + err.Error())
	}
}
