package sdkapp

// animation.go -- added 2026-09-26e (Hermes) for the day-cycle scheduler,
// whose separate Python SDK connection gets UNAUTHENTICATED 401 talking to
// the robot directly. Exposes wire-pod's OWN already-working robot
// connection over HTTP instead, at /api-sdk/play_animation and
// /api-sdk/list_anim_triggers (wired in server.go's SdkapiHandler switch).
//
// play_animation takes behavior control briefly (same OVERRIDE_BEHAVIORS
// request / ControlGrantedResponse / ControlRelease pattern already used in
// pkg/wirepod/ttr/bcontrol.go for the LLM's speech path), plays either an
// animation TRIGGER (?trigger=, resolved to a contextual clip by the
// robot's own behavior system) or an exact clip (?anim=, same call shape
// DoPlayAnimation/DoPlayAnimationWI in pkg/wirepod/ttr/kgsim_cmds.go use
// for {{playAnimationWI||...}}), then releases control. The whole
// operation is bounded by animationRPCTimeout so a stuck robot can't hang
// the HTTP request forever.

import (
	"context"
	"fmt"
	"time"

	"github.com/fforchino/vector-go-sdk/pkg/vector"
	"github.com/fforchino/vector-go-sdk/pkg/vectorpb"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
)

// animationRPCTimeout bounds both /api-sdk/list_anim_triggers and
// /api-sdk/play_animation end to end (control request + play + release for
// the latter). Kept at the requested "<=10s" ceiling.
const animationRPCTimeout = 10 * time.Second

// validatePlayAnimationParams is the pure parameter-validation logic for
// /api-sdk/play_animation, split out so it's unit-testable without a robot
// connection: exactly the "must provide trigger= or anim=" rule.
func validatePlayAnimationParams(trigger, clip string) error {
	if trigger == "" && clip == "" {
		return fmt.Errorf("must provide trigger= (an animation trigger name) or anim= (an exact clip name)")
	}
	return nil
}

// animationTriggerNames is the pure extraction logic for
// /api-sdk/list_anim_triggers, split out so it's unit-testable without a
// robot connection: pulls the plain trigger name strings out of a
// ListAnimationTriggersResponse.
func animationTriggerNames(resp *vectorpb.ListAnimationTriggersResponse) []string {
	triggers := resp.GetAnimationTriggerNames()
	names := make([]string, 0, len(triggers))
	for _, t := range triggers {
		names = append(names, t.GetName())
	}
	return names
}

// playAnimationWithControl acquires behavior control, plays trigger (if
// non-empty) or clip (if trigger is empty), then releases control -- all
// bounded by timeout. validatePlayAnimationParams should be called first by
// the caller; this function itself just prefers trigger over clip if
// (unexpectedly) both are set.
func playAnimationWithControl(robot *vector.Vector, trigger, clip string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	stream, err := robot.Conn.BehaviorControl(ctx)
	if err != nil {
		return fmt.Errorf("opening behavior control stream: %w", err)
	}

	if err := stream.Send(&vectorpb.BehaviorControlRequest{
		RequestType: &vectorpb.BehaviorControlRequest_ControlRequest{
			ControlRequest: &vectorpb.ControlRequest{
				Priority: vectorpb.ControlRequest_OVERRIDE_BEHAVIORS,
			},
		},
	}); err != nil {
		return fmt.Errorf("requesting behavior control: %w", err)
	}

	granted := make(chan error, 1)
	go func() {
		for {
			resp, err := stream.Recv()
			if err != nil {
				granted <- err
				return
			}
			if resp.GetControlGrantedResponse() != nil {
				granted <- nil
				return
			}
		}
	}()

	select {
	case err := <-granted:
		if err != nil {
			return fmt.Errorf("waiting for control grant: %w", err)
		}
	case <-ctx.Done():
		return fmt.Errorf("timed out waiting for behavior control grant")
	}

	var playErr error
	switch {
	case trigger != "":
		_, playErr = robot.Conn.PlayAnimationTrigger(ctx, &vectorpb.PlayAnimationTriggerRequest{
			AnimationTrigger: &vectorpb.AnimationTrigger{Name: trigger},
			Loops:            1,
		})
	case clip != "":
		_, playErr = robot.Conn.PlayAnimation(ctx, &vectorpb.PlayAnimationRequest{
			Animation: &vectorpb.Animation{Name: clip},
			Loops:     1,
		})
	default:
		playErr = fmt.Errorf("no trigger or anim clip name given")
	}

	// Best-effort release regardless of whether play succeeded -- never
	// leave wire-pod holding behavior control on an error.
	if err := stream.Send(&vectorpb.BehaviorControlRequest{
		RequestType: &vectorpb.BehaviorControlRequest_ControlRelease{
			ControlRelease: &vectorpb.ControlRelease{},
		},
	}); err != nil {
		logger.Println("play_animation: failed to release behavior control cleanly: " + err.Error())
	}
	stream.CloseSend()

	return playErr
}
