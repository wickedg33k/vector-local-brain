package speechrequest

// streamend.go -- added 2026-09-27. Fixes a regression from the more
// lenient end-of-speech VAD (2026-09-27 "he is cutting me off" fix): the
// robot's OWN stream (its own end-of-utterance detection or a stream
// deadline) can now close BEFORE our ~1.1s-silence VAD decides, which
// GetNextStreamChunk surfaces as io.EOF. STT backends used to treat ANY
// stream error as a hard failure and discard whatever audio had already
// been accepted -- evidence: 2 of 5 recent voice requests logged
// "(Bot ..., Vosk) Processing... / Using general recognizer / EOF /
// Intent Sent: intent_system_noaudio", dropping real speech.
//
// ShouldFinalizeOnStreamEnd is the pure decision (no I/O, no recognizer)
// behind the fix, used by pkg/wirepod/stt/vosk/Vosk.go's STT: when the
// stream ends, finalize with whatever was already accepted instead of
// erroring out, as long as SOME audio was accepted. Only a stream error
// with zero audio ever accepted is still treated as a real error --
// "only send noaudio if the final transcript is truly empty" is enforced
// naturally downstream (preqs/intent.go et al already check
// strings.TrimSpace(transcribedText) == "") once this returns a real
// (possibly empty) transcript instead of an error.
import "io"

// ShouldFinalizeOnStreamEnd reports whether a stream-read error should be
// treated as "the robot ended the stream, finalize what we have" rather
// than a hard failure.
func ShouldFinalizeOnStreamEnd(streamErr error, framesAccepted int) bool {
	return streamErr == io.EOF && framesAccepted > 0
}
