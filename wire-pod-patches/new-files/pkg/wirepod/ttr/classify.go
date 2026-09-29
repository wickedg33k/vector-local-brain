package wirepod_ttr

// classify.go -- added 2026-09-29 (live-10, Hermes).
//
// Problem (evidenced 2026-09-29 13:32Z, robot docked/idle): when the LLM
// chat path decides the user wants a built-in action, it emits
// {{doIntent||...}}, which wire-pod defers via pendingintent.go and fires
// with AppIntent once behavior control is released. In production this is
// unreliable -- the log shows
// "UserIntentComponent.Update.PendingIntentNotCleared.ForceClear" ->
// "@behavior.voice_command.dropped explore_start App": nothing claims the
// app-sourced pending intent and it gets force-cleared. The SAME intent
// arriving as the voice STREAM's own intent result (exactly what a
// keyphrase CORE match produces, e.g. saying "explore mode" verbatim ->
// "startExploringVoice") works every time -- voice-sourced intents are
// reliable, app-sourced ones are not.
//
// Fix: before falling through to the LLM chat path (StreamingKGSim), make
// one small, fast, NON-STREAMING classification call directly to Ollama
// (bypassing the vector-brain RAG/Hermes-escalation proxy entirely -- this
// needs to be fast and side-effect free) asking "is this utterance one of
// these N whitelisted commands, or none of them?". If it answers with a
// command, send THAT as the voice stream's intent result via the exact
// same ParamChecker/prehistoricParamChecker -> IntentPass path a keyphrase
// CORE match already uses -- never touching AppIntent/doIntent at all for
// this turn. If it answers NONE, times out, or errors, fall straight
// through to the existing SOCIAL/LLM path unchanged -- callers must treat
// every non-match as "proceed exactly as before", never as an error to
// surface. The deferred doIntent/pendingintent.go handling stays in place
// completely unchanged, as a secondary fallback for whatever the LLM chat
// path itself still decides to doIntent (e.g. a command buried in a longer
// conversational turn that the classifier -- deliberately conservative --
// didn't flag).

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/kercre123/wire-pod/chipper/pkg/logger"
	"github.com/kercre123/wire-pod/chipper/pkg/vars"
)

// classifyTimeout bounds the fast classification call. On timeout,
// ClassifyCommandIntent fails open (returns ok=false) so a slow/loaded
// model never adds latency to -- or blocks -- a normal conversational
// turn; the caller just proceeds to the existing LLM chat path exactly as
// if the classifier didn't exist.
const classifyTimeout = 3 * time.Second

// classifyMaxTokens is deliberately tiny -- the expected reply is a single
// intent name (longest: "intent_imperative_turnaround", ~30 chars) or the
// word NONE.
const classifyMaxTokens = 12

// classifyOllamaURL talks directly to Ollama, NOT through the vector-brain
// proxy at 127.0.0.1:11500 -- that proxy does RAG context injection, vision
// routing, and Hermes-escalation heuristics, none of which this fast
// yes/no-style call wants or needs. wire-pod's container runs with
// --network host, so 127.0.0.1 here is the same host Ollama listens on
// (gpu-host).
const classifyOllamaURL = "http://127.0.0.1:11434/v1/chat/completions"

// classifyKeepAlive is sent on every classify call so a burst of
// classification traffic never contributes to the model unloading between
// turns (see also the vector-brain proxy's own keep_alive fix for the main
// chat path, server.py).
const classifyKeepAlive = "24h"

// classifyIntentDescriptions gives the classifier model a short
// description for each candidate intent. Deliberately scoped to EXACTLY
// the same names as AllowedLLMIntents (kgsim_cmds.go) -- the same
// whitelist doIntent already enforces -- so the classifier can never
// return anything that wouldn't have been allowed anyway.
var classifyIntentDescriptions = map[string]string{
	"intent_explore_start":         "start exploring / roaming / wandering around on its own",
	"intent_system_charger":        "go to / return to / drive to its charger or dock",
	"intent_system_sleep":          "go to sleep",
	"intent_clock_time":            "say/tell the current time",
	"intent_imperative_volumeup":   "turn its speaking volume up / louder",
	"intent_imperative_volumedown": "turn its speaking volume down / quieter",
	"intent_imperative_eyecolor":   "change its eye color",
	"intent_photo_take_extend":     "take a photo / picture / selfie",
	"intent_imperative_dance":      "dance",
	"intent_play_fistbump":         "do a fist bump",
	"intent_play_popawheelie":      "pop a wheelie trick",
	"intent_imperative_forward":    "drive/move forward",
	"intent_imperative_backup":     "drive/move backward",
	"intent_imperative_turnleft":   "turn left",
	"intent_imperative_turnright":  "turn right",
	"intent_imperative_turnaround": "turn all the way around",
	"intent_imperative_come":       "come here / approach the user",
	"intent_imperative_lookatme":   "look at the user",
}

// buildClassifyPrompt is pure (no I/O) so it's directly unit-testable.
func buildClassifyPrompt(voiceText string) string {
	var b strings.Builder
	b.WriteString("You are a strict intent classifier for a small robot. ")
	b.WriteString("Decide whether the utterance below is a direct command asking the robot to do ONE of the listed actions right now. ")
	b.WriteString("Reply with EXACTLY one of the intent names below, verbatim, and nothing else -- no punctuation, no explanation, no quotes. ")
	b.WriteString("If the utterance is not clearly and directly asking for one of these actions (this includes questions, statements, opinions, greetings, general chat, or anything ambiguous), reply with exactly: NONE\n\n")
	for _, name := range AllowedLLMIntents {
		desc, ok := classifyIntentDescriptions[name]
		if !ok {
			continue
		}
		b.WriteString(name + ": " + desc + "\n")
	}
	b.WriteString("\nUtterance: \"")
	b.WriteString(voiceText)
	b.WriteString("\"\n/no_think")
	return b.String()
}

type classifyChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type classifyChatRequest struct {
	Model       string                `json:"model"`
	Messages    []classifyChatMessage `json:"messages"`
	Temperature float32               `json:"temperature"`
	MaxTokens   int                   `json:"max_tokens"`
	Stream      bool                  `json:"stream"`
	KeepAlive   string                `json:"keep_alive,omitempty"`
	// ReasoningEffort disables qwen3's chain-of-thought (2026-09-29 fix,
	// found during live-10 testing): /no_think ALONE was not enough --
	// Ollama's OpenAI-compat layer still burned the whole max_tokens
	// budget on a separate "reasoning" field before ever reaching content,
	// leaving content empty with finish_reason="length". Matches the same
	// fix already used for the main chat path (vector-brain's server.py).
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

type classifyChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// stripThinkTags defensively removes a <think>...</think> block, in case a
// qwen3 model leaks one even with /no_think + reasoning_effort=none.
// Pure/testable.
func stripThinkTags(s string) string {
	for {
		start := strings.Index(s, "<think>")
		if start == -1 {
			return s
		}
		end := strings.Index(s, "</think>")
		if end == -1 || end < start {
			// Unterminated -- bail rather than loop/garble.
			return s
		}
		s = strings.TrimSpace(s[:start] + s[end+len("</think>"):])
	}
}

// parseClassifyResponse extracts a clean, verified intent name (or "" for
// "no match") from the raw model output. Pure/testable: only an EXACT
// match against AllowedLLMIntents is accepted -- a hallucinated name,
// extra words, a rambling explanation the model added despite
// instructions, or genuinely empty output all fail safe to "" (treated by
// the caller exactly like NONE).
func parseClassifyResponse(raw string) string {
	s := stripThinkTags(strings.TrimSpace(raw))
	s = strings.Trim(s, " \t\n\r.,!?\"'`")
	if idx := strings.IndexAny(s, "\n\r"); idx != -1 {
		s = strings.TrimSpace(s[:idx])
	}
	s = strings.Trim(s, " \t.,!?\"'`")
	if s == "" || strings.EqualFold(s, "NONE") {
		return ""
	}
	if IsAllowedLLMIntent(s) {
		return s
	}
	return ""
}

// classifyHTTPClient is package-level so production gets one shared client
// with classifyTimeout as a hard ceiling (belt-and-braces alongside the
// per-request context timeout below).
var classifyHTTPClient = &http.Client{Timeout: classifyTimeout}

// classifyModel resolves which model to classify with -- the same model
// configured for the main knowledge/chat path, falling back to the known
// default if unset.
func classifyModel() string {
	model := strings.TrimSpace(vars.APIConfig.Knowledge.Model)
	if model == "" {
		model = "qwen3:32b"
	}
	return model
}

// ClassifyCommandIntent makes a fast, non-streaming classification call
// directly against Ollama to decide whether voiceText is a direct command
// for one of AllowedLLMIntents. Returns (intentName, true) on a confident
// match, ("", false) on NONE / timeout / any error -- callers MUST treat
// false as "fall through to the existing SOCIAL/LLM path unchanged", never
// as a failure to surface to the user.
func ClassifyCommandIntent(voiceText string) (string, bool) {
	reqBody := classifyChatRequest{
		Model: classifyModel(),
		Messages: []classifyChatMessage{
			{Role: "user", Content: buildClassifyPrompt(voiceText)},
		},
		Temperature:     0,
		MaxTokens:       classifyMaxTokens,
		Stream:          false,
		KeepAlive:       classifyKeepAlive,
		ReasoningEffort: "none",
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		logger.Println("classify: failed to marshal request: " + err.Error())
		return "", false
	}

	ctx, cancel := context.WithTimeout(context.Background(), classifyTimeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, classifyOllamaURL, bytes.NewReader(body))
	if err != nil {
		logger.Println("classify: failed to build request: " + err.Error())
		return "", false
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := classifyHTTPClient.Do(httpReq)
	if err != nil {
		logger.Println("classify: request failed/timed out after " + time.Since(start).String() + ": " + err.Error())
		return "", false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logger.Println("classify: non-200 from Ollama (" + resp.Status + ") after " + time.Since(start).String())
		return "", false
	}

	var parsed classifyChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		logger.Println("classify: failed to decode response: " + err.Error())
		return "", false
	}
	if len(parsed.Choices) == 0 {
		logger.Println("classify: empty choices in response")
		return "", false
	}

	intent := parseClassifyResponse(parsed.Choices[0].Message.Content)
	if intent == "" {
		logger.Println("classify: '" + voiceText + "' -> NONE (" + time.Since(start).String() + ")")
		return "", false
	}
	logger.Println("classify: '" + voiceText + "' -> " + intent + " (" + time.Since(start).String() + ")")
	return intent, true
}
