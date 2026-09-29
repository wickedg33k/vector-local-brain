import anki_vector, io, base64, json, urllib.request, time
from anki_vector.util import degrees
L = anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY
shots = []
with anki_vector.Robot("YOUR_ESN", behavior_control_level=L) as robot:
    robot.behavior.say_text("Let me look around my desk.")
    for turn, head in [(0, 5), (-35, 5), (70, 5), (-35, 20)]:
        if turn: robot.behavior.turn_in_place(degrees(turn))
        robot.behavior.set_head_angle(degrees(head)); time.sleep(0.8)
        img = robot.camera.capture_single_image().raw_image
        b = io.BytesIO(); img.save(b, format="JPEG", quality=90); shots.append(base64.b64encode(b.getvalue()).decode())
    robot.behavior.say_text("Got it. Thinking about what I saw.")
content = [{"type": "text", "text": "These are 4 photos from a small desk robot looking around a desk (center, left, right, center-higher). List every distinct object you can identify, with its approximate position (center/left/right), and transcribe ALL readable text, brands and model names exactly. Be concise, one bullet per object."}]
for s in shots: content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + s}})
body = {"model": "qwen2.5vl:3b", "stream": False, "messages": [{"role": "user", "content": content}]}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=240)
print(json.load(r)["choices"][0]["message"]["content"])
