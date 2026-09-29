package speechrequest

import (
	"errors"
	"io"
	"testing"
)

func TestShouldFinalizeOnStreamEnd_EOFWithAudio(t *testing.T) {
	// The exact regression: robot closes the stream (EOF) before our VAD
	// decides, but we already accepted real audio -- must finalize, not error.
	if !ShouldFinalizeOnStreamEnd(io.EOF, 3) {
		t.Fatalf("expected EOF with accepted audio to finalize instead of erroring")
	}
}

func TestShouldFinalizeOnStreamEnd_EOFWithNoAudio(t *testing.T) {
	// Never got a single frame -- genuinely nothing to finalize, still an error.
	if ShouldFinalizeOnStreamEnd(io.EOF, 0) {
		t.Fatalf("expected EOF with zero accepted audio to still be treated as an error")
	}
}

func TestShouldFinalizeOnStreamEnd_OtherErrorNeverFinalizes(t *testing.T) {
	// A real transport error (not a clean stream-end) should still be a
	// hard error even if some audio came in first.
	if ShouldFinalizeOnStreamEnd(errors.New("connection reset"), 5) {
		t.Fatalf("expected a non-EOF error to still be treated as an error")
	}
}
