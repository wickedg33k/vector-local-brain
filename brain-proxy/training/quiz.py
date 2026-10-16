"""Tiny spot-check: ask the brain proxy a couple of questions and append
timing + answers to the eval log. Edit QUESTIONS to probe your own vault."""
import json, os, time, urllib.request

PROXY_URL = os.environ.get("VECTOR_BRAIN_URL", "http://127.0.0.1:11500") + "/v1/chat/completions"
MODEL = os.environ.get("VECTOR_QUIZ_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M")
EVAL_LOG = os.environ.get("VECTOR_EVAL_LOG", os.path.expanduser("~/vector-brain/training/eval.log"))

QUESTIONS = [
    ("How many legs does a spider have?", "eight"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
]

out = []
for q, exp in QUESTIONS:
    body = {"model": MODEL, "stream": False, "max_tokens": 90, "messages": [
        {"role": "system", "content": "You are Vector, a small desk robot. Reply in 1-2 short sentences."},
        {"role": "user", "content": q}]}
    t = time.time()
    req = urllib.request.Request(PROXY_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    a = r["choices"][0]["message"]["content"].strip().replace("\n", " ")
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.append(f"{ts} Q={q!r} expected={exp!r} secs={time.time()-t:.1f} A={a[:200]!r}")
os.makedirs(os.path.dirname(EVAL_LOG), exist_ok=True)
with open(EVAL_LOG, "a") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
