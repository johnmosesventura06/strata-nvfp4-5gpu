#!/usr/bin/env python3
"""Client A decodes a long answer; 3 s later client B sends a long prompt.
Reports A's decode rate while B's prompt is read, A's rate before/after, and B's time to first token."""
import json, random, string, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8190"
FILL = ("The report continues with notes on operations, staffing, the budget for the next quarter, the maintenance "
        "schedule, open questions from the review board, and the list of suppliers that still owe documents. ")


def prompt(ptoks):
    m = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    ctx = "".join(f"[{m}-{i}] {FILL}" for i in range(max(0, int(ptoks / 38)))) if ptoks > 200 else ""
    return f"{ctx}\n\nReference {m}. Write a very long, detailed essay about the history of bridges. Many paragraphs."


def stream(ptoks, max_tokens, out, text=None):
    payload = {"messages": [{"role": "user", "content": text or prompt(ptoks)}], "max_tokens": max_tokens, "temperature": 0,
               "reasoning_effort": "none", "stream": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    out["t0"] = time.time(); out["ticks"] = []; out["text"] = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except Exception:
                continue
            for ch in d.get("choices", []):
                dl = ch.get("delta") or {}
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                if piece:
                    out["ticks"].append(time.time())
                    out["text"] += piece


def rate(ticks, a, b):
    n = sum(1 for t in ticks if a <= t < b)
    return n / (b - a) if b > a else 0.0


def main():
    label = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    res = []
    texts = []
    for rd in range(rounds):
        A, B = {}, {}
        random.seed(int(sys.argv[3]) + rd if len(sys.argv) > 3 else 1000 + rd)   # the same prompts in every run: their texts compare across runs
        pa, pb = prompt(1000), prompt(16000)
        ta = threading.Thread(target=stream, args=(1000, 1500, A, pa)); ta.start()
        time.sleep(3.0)
        tb = threading.Thread(target=stream, args=(16000, 60, B, pb)); tb.start()
        tb.join(); ta.join()
        b0, b1 = B["t0"], B["ticks"][0]
        res.append(dict(during=rate(A["ticks"], b0 + 0.5, b1 - 0.2), before=rate(A["ticks"], A["ticks"][0] + 0.3, b0),
                        after=rate(A["ticks"], b1 + 0.5, min(A["ticks"][-1], b1 + 6)), ttft_b=b1 - b0))
        texts.append({"A": A["text"], "B": B["text"]})
        r = res[-1]
        print(f"{label}: A decode before {r['before']:.1f} | during B's prompt {r['during']:.1f} | after {r['after']:.1f} "
              f"pieces/s; B first token {r['ttft_b']:.2f} s", flush=True)
        time.sleep(3)
    json.dump(texts, open(f"/home/ai-host/strata-nvfp4/bench-conc/texts-{label}.json", "w"))
    print(f"{label} MEAN: during {sum(r['during'] for r in res)/len(res):.1f} pieces/s, B TTFT {sum(r['ttft_b'] for r in res)/len(res):.2f} s")


main()
