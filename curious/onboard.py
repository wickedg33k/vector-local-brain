import anki_vector
from anki_vector.messaging import protocol
with anki_vector.Robot("YOUR_ESN", behavior_control_level=None) as robot:
    req = protocol.OnboardingInputRequest(onboarding_mark_complete_and_exit=protocol.OnboardingMarkCompleteAndExit())
    resp = robot.conn.run_coroutine(robot.conn.grpc_interface.SendOnboardingInput(req)).result(timeout=20)
    print("RESPONSE:", resp)
