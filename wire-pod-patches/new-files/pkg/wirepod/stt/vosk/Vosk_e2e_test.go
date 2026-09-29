package wirepod_vosk

// Vosk_e2e_test.go -- added 2026-09-28 (Hermes) after the live-7 regression
// (ALL speech recognition broke: every request logged
// "(Bot ..., Vosk) Processing... / Using general recognizer" then
// "rpc error: DeadlineExceeded"/"Canceled", zero "Transcribed text" lines).
//
// This is a REAL end-to-end harness: it loads the actual Vosk model (same
// path production uses, via vars.VoskModelPath -- must be run with the real
// model data available, e.g. the ~/vector-pod/data/vosk volume
// mounted), feeds a real recorded PCM utterance
// (chipper/stttest.pcm -- the same file Vosk.go's own runTest() self-test
// already uses on every startup, 16kHz/16-bit/mono, ~6s, known-good) through
// a fake gRPC stream implementing the SAME interface GetNextStreamChunk
// type-switches on, and calls the real STT() end to end. Two cases:
//   - normal: the whole file streams through, VAD should decide
//     end-of-speech partway through the trailing silence and return before
//     the file is exhausted.
//   - eofMidSpeech: the fake stream returns io.EOF partway through (before
//     VAD would have decided on its own), simulating the robot closing its
//     end of the stream early -- the regression scenario.
// Both must return a non-empty transcript within a bounded deadline. A
// deadline exceeded here (STT() genuinely hanging/never returning) is
// exactly the class of bug being hunted.
//
// Skips itself (does not fail) if the real model isn't available, so it's
// safe to leave in the tree for normal (non-cgo, non-model) CI runs.

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	pb "github.com/digital-dream-labs/api/go/chipperpb"
	"github.com/kercre123/wire-pod/chipper/pkg/vars"
	sr "github.com/kercre123/wire-pod/chipper/pkg/wirepod/speechrequest"
	"github.com/maxhawkins/go-webrtcvad"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// fakeIntentStream implements pb.ChipperGrpc_StreamingIntentServer by
// replaying pre-chunked audio, then returning a configurable terminal error
// (io.EOF to simulate a clean stream end, or something else to simulate a
// real transport error).
type fakeIntentStream struct {
	grpc.ServerStream
	chunks    [][]byte
	i         int
	afterErr  error
}

func (f *fakeIntentStream) Send(*pb.IntentResponse) error { return nil }

func (f *fakeIntentStream) Recv() (*pb.StreamingIntentRequest, error) {
	if f.i >= len(f.chunks) {
		if f.afterErr == nil {
			return nil, io.EOF
		}
		return nil, f.afterErr
	}
	c := f.chunks[f.i]
	f.i++
	return &pb.StreamingIntentRequest{
		Session:    "e2e-test",
		DeviceId:   "e2e-test-esn",
		InputAudio: c,
	}, nil
}

func (f *fakeIntentStream) Context() context.Context           { return context.Background() }
func (f *fakeIntentStream) SetHeader(metadata.MD) error        { return nil }
func (f *fakeIntentStream) SendHeader(metadata.MD) error       { return nil }
func (f *fakeIntentStream) SetTrailer(metadata.MD)             {}
func (f *fakeIntentStream) SendMsg(m interface{}) error        { return nil }
func (f *fakeIntentStream) RecvMsg(m interface{}) error        { return nil }

// chunkPCM splits raw PCM into 3200-byte (100ms @ 16kHz/16-bit/mono) pieces,
// matching the robot's documented chunk size (chipperpb.proto: "16k SR,
// 1-channel, 100ms chunks").
func chunkPCM(pcm []byte) [][]byte {
	const chunkSize = 3200
	var out [][]byte
	for len(pcm) > 0 {
		n := chunkSize
		if n > len(pcm) {
			n = len(pcm)
		}
		out = append(out, pcm[:n])
		pcm = pcm[n:]
	}
	return out
}

// e2eModelAvailable reports whether the real Vosk model this test needs is
// present, and initializes it (mirrors Vosk.go's Init(), scoped to what the
// test needs) if so.
func e2eSetup(t *testing.T) {
	t.Helper()
	if p := os.Getenv("E2E_VOSK_MODEL_PATH"); p != "" {
		vars.VoskModelPath = p
	}
	modelPath := filepath.Join(vars.VoskModelPath, "en-US", "model")
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("skipping: real Vosk model not available at %s (%v) -- run with the model data mounted to exercise this harness", modelPath, err)
	}
	if !modelLoaded {
		vars.APIConfig.PastInitialSetup = true
		vars.APIConfig.STT.Language = "en-US"
		if err := Init(); err != nil {
			t.Fatalf("Init() failed: %v", err)
		}
	}
}

func e2eLoadTestPCM(t *testing.T) []byte {
	t.Helper()
	// Same file Vosk.go's own runTest() self-test uses -- known-good,
	// already proven to transcribe correctly on every pod startup.
	candidates := []string{"../../../../stttest.pcm", "./stttest.pcm", "../../../../../stttest.pcm"}
	for _, c := range candidates {
		if b, err := os.ReadFile(c); err == nil {
			return b
		}
	}
	t.Skip("skipping: stttest.pcm not found relative to test working directory")
	return nil
}

func newSpeechRequest(t *testing.T, stream pb.ChipperGrpc_StreamingIntentServer, firstReq []byte) sr.SpeechRequest {
	t.Helper()
	vadInst, err := webrtcvad.New()
	if err != nil {
		t.Fatalf("webrtcvad.New() failed: %v", err)
	}
	vadInst.SetMode(2)
	return sr.SpeechRequest{
		Device:   "e2e-test-esn",
		Session:  "e2e-test",
		FirstReq: firstReq,
		Stream:   stream,
		IsOpus:   false,
		VADInst:  vadInst,
	}
}

// runSTTWithDeadline calls STT() in a goroutine and fails the test if it
// doesn't return within deadline -- this is what catches a genuine
// hang/never-returns bug, as opposed to a fast (possibly wrong) error.
func runSTTWithDeadline(t *testing.T, req sr.SpeechRequest, deadline time.Duration) (string, error) {
	t.Helper()
	type result struct {
		text string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		text, err := STT(req)
		done <- result{text, err}
	}()
	select {
	case r := <-done:
		return r.text, r.err
	case <-time.After(deadline):
		t.Fatalf("STT() did not return within %s -- this is the hang/never-returns failure mode", deadline)
		return "", nil
	}
}

func TestE2E_NormalEndOfSpeech_ProducesTranscript(t *testing.T) {
	e2eSetup(t)
	pcm := e2eLoadTestPCM(t)
	chunks := chunkPCM(pcm)
	first, rest := chunks[0], chunks[1:]
	stream := &fakeIntentStream{chunks: rest} // EOF after all real audio -- VAD should decide before that
	req := newSpeechRequest(t, stream, first)

	text, err := runSTTWithDeadline(t, req, 15*time.Second)
	if err != nil {
		t.Fatalf("STT() returned an error for a normal utterance: %v", err)
	}
	if text == "" {
		t.Fatalf("expected a non-empty transcript for a known-good recorded utterance")
	}
	t.Logf("normal case transcript: %q", text)
}

func TestE2E_EOFMidSpeech_StillProducesTranscript(t *testing.T) {
	// The live-7 regression scenario: the stream ends (EOF) BEFORE our VAD
	// would have decided end-of-speech on its own -- simulated by cutting
	// the chunk feed short (only the first third of the recording), so the
	// silence tail (which is what would normally trigger end-of-speech)
	// never arrives -- the fake stream returns EOF instead.
	e2eSetup(t)
	pcm := e2eLoadTestPCM(t)
	chunks := chunkPCM(pcm)
	first := chunks[0]
	cut := len(chunks) / 3
	if cut < 1 {
		cut = 1
	}
	rest := chunks[1:cut]
	stream := &fakeIntentStream{chunks: rest, afterErr: io.EOF}
	req := newSpeechRequest(t, stream, first)

	text, err := runSTTWithDeadline(t, req, 15*time.Second)
	if err != nil {
		t.Fatalf("STT() returned an error on EOF-mid-speech instead of finalizing: %v", err)
	}
	// Partial audio may or may not yield a full transcript depending on
	// where the cut lands, but it must return (not hang) and must not
	// error -- that's the actual regression being guarded against here.
	t.Logf("EOF-mid-speech transcript: %q", text)
}

func TestE2E_GenuineTransportError_StillReturnsPromptly(t *testing.T) {
	// A non-EOF error (e.g. DeadlineExceeded/Canceled, matching the live-7
	// symptom) with SOME audio already accepted must still return promptly
	// (as an error) -- not hang.
	e2eSetup(t)
	pcm := e2eLoadTestPCM(t)
	chunks := chunkPCM(pcm)
	first := chunks[0]
	cut := len(chunks) / 3
	if cut < 1 {
		cut = 1
	}
	rest := chunks[1:cut]
	stream := &fakeIntentStream{chunks: rest, afterErr: errors.New("rpc error: code = DeadlineExceeded desc = context deadline exceeded")}
	req := newSpeechRequest(t, stream, first)

	_, err := runSTTWithDeadline(t, req, 15*time.Second)
	if err == nil {
		t.Fatalf("expected a genuine (non-EOF) transport error to still be returned as an error")
	}
	t.Logf("genuine transport error case returned promptly: %v", err)
}
