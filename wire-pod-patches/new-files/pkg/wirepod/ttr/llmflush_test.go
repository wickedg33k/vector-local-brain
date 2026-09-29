package wirepod_ttr

import "testing"

// --- flushTrailingText: the three 2026-09-27 bug report scenarios ---

func TestFlushTrailingText_NoPunctuationAtAll(t *testing.T) {
	// The exact failure mode: fullRespSlice never got anything mid-stream
	// (no "." "?" "!" "..." ever seen), so the whole reply is sitting in
	// the trailing buffer at EOF.
	got := flushTrailingText(nil, "I think it's daytime right now")
	want := []string{"I think it's daytime right now"}
	if len(got) != 1 || got[0] != want[0] {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFlushTrailingText_SentenceThenUnpunctuatedTrailer(t *testing.T) {
	// A full sentence flushed mid-stream, then a trailing fragment with no
	// closing punctuation.
	got := flushTrailingText([]string{"Sure!"}, " and it looks sunny out")
	want := []string{"Sure!", " and it looks sunny out"}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFlushTrailingText_OnlyACommand(t *testing.T) {
	// A reply that's nothing but a bare {{command}} -- no punctuation, so
	// it never got flushed mid-stream either.
	got := flushTrailingText(nil, "{{doIntent||intent_explore_start}}")
	want := []string{"{{doIntent||intent_explore_start}}"}
	if len(got) != 1 || got[0] != want[0] {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFlushTrailingText_EndsExactlyOnPunctuation(t *testing.T) {
	// Nothing left over -- must not append an empty/whitespace chunk.
	got := flushTrailingText([]string{"Sure!", "That works."}, "")
	want := []string{"Sure!", "That works."}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFlushTrailingText_WhitespaceOnlyTrailerIsDropped(t *testing.T) {
	got := flushTrailingText([]string{"Sure!"}, "   ")
	if len(got) != 1 {
		t.Fatalf("expected whitespace-only trailer to be dropped, got %v", got)
	}
}

func TestFlushTrailingText_DoesNotMutateInput(t *testing.T) {
	in := []string{"Sure!"}
	_ = flushTrailingText(in, "more")
	if len(in) != 1 || in[0] != "Sure!" {
		t.Fatalf("expected input slice to be left untouched, got %v", in)
	}
}

// --- needsFallbackResponse ---

func TestNeedsFallbackResponse_RealTextNeedsNoFallback(t *testing.T) {
	if needsFallbackResponse([]string{"I think it's daytime right now"}) {
		t.Fatalf("expected real spoken text to need no fallback")
	}
}

func TestNeedsFallbackResponse_BareCommandNeedsNoFallback(t *testing.T) {
	// The command WILL do something (e.g. explore) even though nothing is
	// spoken -- no fallback phrase should be injected on top of it.
	if needsFallbackResponse([]string{"{{doIntent||intent_explore_start}}"}) {
		t.Fatalf("expected a bare command to need no fallback (the action itself is the response)")
	}
}

func TestNeedsFallbackResponse_TrulyEmptyNeedsFallback(t *testing.T) {
	if !needsFallbackResponse(nil) {
		t.Fatalf("expected a completely empty response to need the fallback")
	}
}

func TestNeedsFallbackResponse_WhitespaceOnlyNeedsFallback(t *testing.T) {
	if !needsFallbackResponse([]string{"   "}) {
		t.Fatalf("expected whitespace-only content to need the fallback")
	}
}

// --- truncateForLog ---

func TestTruncateForLog_ShortStringUnchanged(t *testing.T) {
	if got := truncateForLog("hello", 300); got != "hello" {
		t.Fatalf("got %q, want unchanged", got)
	}
}

func TestTruncateForLog_LongStringTruncated(t *testing.T) {
	long := make([]rune, 400)
	for i := range long {
		long[i] = 'a'
	}
	got := truncateForLog(string(long), 300)
	if len(got) <= 300 {
		t.Fatalf("expected a truncation marker appended, got len=%d", len(got))
	}
	if got[:300] != string(long[:300]) {
		t.Fatalf("expected the first 300 runes to be preserved unchanged")
	}
}
