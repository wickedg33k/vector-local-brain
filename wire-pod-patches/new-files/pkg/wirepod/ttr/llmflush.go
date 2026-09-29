package wirepod_ttr

// llmflush.go -- added 2026-09-27. Fixes a bug where an LLM reply with no
// terminal punctuation was silently dropped: StreamingKGSim only ever
// moved streamed text out of the in-progress buffer (fullRespText) into
// the speakable queue (fullRespSlice) when it saw ".", "?", "!" or "..."
// go by. A reply that never contained one of those (a short reply made
// MORE likely by the 2026-09-26e brevity instruction) left fullRespSlice
// empty for the whole stream, which the old EOF handler treated as
// len(fullRespSlice)==0 -> "LLM returned no response" -> the function
// returned an error and nothing was ever spoken. Evidence (2026-09-27,
// the owner): "asked from his wife face to status" -> pod logged
// "LLM stream response:" (nothing printed), "Intent Sent:
// intent_greeting_hello", "LLM debug: there is content after the last
// punctuation mark", "LLM stream finished" -- and nothing was spoken; the
// robot showed the cloud-with-exclamation error.
//
// The fix, split into pure/testable pieces:
//   - flushTrailingText: always carries forward whatever's left in the
//     not-yet-punctuation-split buffer as a final chunk, replacing the old
//     fragile newStr/TrimPrefix reconciliation in the EOF handler (which
//     only worked by accident for the single-trailing-fragment case and
//     did nothing for a reply with NO punctuation at all).
//   - needsFallbackResponse: decides whether, after flushing, there is
//     genuinely nothing to say and no action (doIntent etc.) to perform --
//     in which case a short spoken fallback should be used instead of
//     silence, so the robot never errors out with nothing spoken.
//
// See kgsim.go's StreamingKGSim EOF handling for how these are wired in.

import (
	"strings"
)

// llmEmptyReplyFallback is spoken when the LLM's reply produces no
// speakable text and no action, so the robot never silently errors out.
const llmEmptyReplyFallback = "Hmm, say that again?"

// llmRawLogTruncateLen bounds how much of the raw LLM text gets logged for
// diagnosis (added 2026-09-27 per this bug's postmortem -- there was no
// visibility into the actual raw text that triggered the silent-drop).
const llmRawLogTruncateLen = 300

// truncateForLog truncates s to at most max runes for logging, appending a
// marker if it was cut. Rune-safe (doesn't split a multi-byte character).
func truncateForLog(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "...(truncated)"
}

// flushTrailingText appends trailingText (the not-yet-punctuation-split
// buffer, i.e. fullRespText at EOF) to flushedChunks (fullRespSlice as
// accumulated mid-stream) if it has any non-whitespace content. Pure: does
// not mutate its inputs, returns a new slice.
//
// This covers all three of the 2026-09-27 bug report's cases:
//   - No punctuation ANYWHERE in the reply: flushedChunks is empty,
//     trailingText is the ENTIRE reply -> becomes the sole chunk.
//   - A sentence flushed mid-stream, then a trailing fragment with no
//     closing punctuation: trailingText (just the fragment) is appended
//     after the already-flushed chunk(s).
//   - The reply ends exactly on a punctuation mark, nothing after:
//     trailingText is empty -> flushedChunks is returned unchanged.
func flushTrailingText(flushedChunks []string, trailingText string) []string {
	if strings.TrimSpace(trailingText) == "" {
		return flushedChunks
	}
	out := make([]string, 0, len(flushedChunks)+1)
	out = append(out, flushedChunks...)
	out = append(out, trailingText)
	return out
}

// needsFallbackResponse reports whether, after parsing every chunk's
// commands (via GetActionsFromString), nothing will actually be said or
// done -- meaning a fallback phrase should be spoken instead of silence.
// A single {{command}} with no surrounding text (e.g. a bare
// {{doIntent||...}}) correctly reports false: an action WILL happen, no
// fallback needed even though there's no spoken text.
func needsFallbackResponse(chunks []string) bool {
	for _, c := range chunks {
		for _, a := range GetActionsFromString(c) {
			if a.Action != ActionSayText {
				return false
			}
			if strings.TrimSpace(a.Parameter) != "" {
				return false
			}
		}
	}
	return true
}
