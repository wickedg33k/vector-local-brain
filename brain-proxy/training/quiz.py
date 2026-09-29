import json, urllib.request, time, os
qs = [("How long will a Pixel 10 get updates?", "7 years (retest after budget fix)"), ("What is the Cape Hatteras Lighthouse known for?", "tallest brick lighthouse in US")]
out = []
for q, exp in qs:
    body = {"model": "qwen3:32b", "stream": False, "max_tokens": 90, "messages": [{"role": "system", "content": "You are Vector, a small desk robot. Reply in 1-2 short sentences."}, {"role": "user", "content": q}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11500/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=60))
    a = r["choices"][0]["message"]["content"].strip().replace("\n", " ")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.append(f"{ts} Q={q!r} expected={exp!r} secs={time.time()-t:.1f} A={a[:200]!r}")
open(os.path.expanduser("~/vector-brain/training/eval.log"), "a").write("\n".join(out) + "\n")
print("\n".join(out))
