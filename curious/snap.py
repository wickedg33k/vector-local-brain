import anki_vector, io, base64, json, urllib.request
with anki_vector.Robot("YOUR_ESN", behavior_control_level=anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
    img = robot.camera.capture_single_image().raw_image
buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92); open("/app/snap.jpg","wb").write(buf.getvalue())
b64 = base64.b64encode(buf.getvalue()).decode()
body = {"model":"qwen2.5vl:3b","stream":False,"messages":[{"role":"user","content":[
 {"type":"text","text":"Describe the main object in front of the camera, and transcribe ALL readable text exactly (brand, model, labels). Be precise."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}]}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=120)
print(json.load(r)["choices"][0]["message"]["content"])
