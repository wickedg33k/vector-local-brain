package sdkapp

import (
	"testing"

	"github.com/fforchino/vector-go-sdk/pkg/vectorpb"
)

func TestValidatePlayAnimationParams_BothEmpty(t *testing.T) {
	if err := validatePlayAnimationParams("", ""); err == nil {
		t.Fatalf("expected an error when neither trigger nor anim is given")
	}
}

func TestValidatePlayAnimationParams_TriggerOnly(t *testing.T) {
	if err := validatePlayAnimationParams("sleepy_trigger", ""); err != nil {
		t.Fatalf("expected no error with a trigger given, got %v", err)
	}
}

func TestValidatePlayAnimationParams_ClipOnly(t *testing.T) {
	if err := validatePlayAnimationParams("", "anim_tired_01"); err != nil {
		t.Fatalf("expected no error with a clip given, got %v", err)
	}
}

func TestValidatePlayAnimationParams_BothGivenIsFine(t *testing.T) {
	// Not an error at validation time -- playAnimationWithControl prefers
	// trigger over clip if both happen to be set.
	if err := validatePlayAnimationParams("sleepy_trigger", "anim_tired_01"); err != nil {
		t.Fatalf("expected no error when both are given, got %v", err)
	}
}

func TestAnimationTriggerNames_Extracts(t *testing.T) {
	resp := &vectorpb.ListAnimationTriggersResponse{
		AnimationTriggerNames: []*vectorpb.AnimationTrigger{
			{Name: "GreetAfterLongTime"},
			{Name: "SleepyTired"},
			{Name: "FistBumpRequestOnce"},
		},
	}
	got := animationTriggerNames(resp)
	want := []string{"GreetAfterLongTime", "SleepyTired", "FistBumpRequestOnce"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestAnimationTriggerNames_EmptyResponse(t *testing.T) {
	resp := &vectorpb.ListAnimationTriggersResponse{}
	got := animationTriggerNames(resp)
	if len(got) != 0 {
		t.Fatalf("expected no names from an empty response, got %v", got)
	}
}

func TestAnimationTriggerNames_NilResponse(t *testing.T) {
	// GetAnimationTriggerNames() on a nil *ListAnimationTriggersResponse is
	// safe (generated protobuf getters nil-check their receiver) -- make
	// sure our wrapper doesn't panic on it either, since a defensive caller
	// might pass one through.
	var resp *vectorpb.ListAnimationTriggersResponse
	got := animationTriggerNames(resp)
	if len(got) != 0 {
		t.Fatalf("expected no names from a nil response, got %v", got)
	}
}
