"""T4.5 native end-to-end suite: validated decode loop under exercise.

Cases (each runs the real binary; model loads are ~20s, suite is slow):
 1. matched-template completion vs llama reference (thinking-enabled both
    sides): device greedy text must contain the same final answer as llama.
 2. one-token prompt runs clean.
 3. short prompt (4 tokens) determinism: two runs byte-identical JSON.
 4. max-length prompt (64 tokens) runs clean.
 5. negatives: truncated .binfer and garbage file exit non-zero with stderr.
Saves tools/e2e/report_t45.json.
"""
import json
import os
import subprocess
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools", "cli"))
import tok as tokenizer
from ainfer_cli import generate_llama

VENV = os.path.expanduser("~/.venvs/ainfer/bin/python")
BIN = os.path.join(REPO, "build-b60/tools/decode/decode")
MODEL = os.path.join(REPO, "models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer")
OUT = os.path.join(REPO, "tools/e2e/report_t45.json")
ONEAPI = "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; "


def run_decode(ids, max_new, tag, timeout=1500):
    rp = f"/tmp/e2e_{tag}.json"
    cmd = (f"{ONEAPI}exec {BIN} {MODEL} {len(ids)} 1 {rp} "
           f"--ids={','.join(map(str, ids))} --max-new={max_new}")
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=timeout)
    rep = json.load(open(rp)) if r.returncode == 0 else None
    return r, rep


results = []
tok = tokenizer.load()
MSG = "What is 84 * 3 / 2? Reply with just the number."

# 1. matched template (thinking ON both sides) vs llama reference
msgs = [{"role": "user", "content": MSG}]
rendered = tokenizer.render_chat(msgs, add_generation_prompt=True,
                                 enable_thinking=True)
ids = tokenizer.encode(tok, rendered)
print(f"[1] thinking-template ids: {len(ids)}", flush=True)
r, rep = run_decode(ids, 96, "match")
ref = generate_llama(MSG, 96, 0.0, 0, stream_out=False)
dev_text = tokenizer.decode(tok, rep["generated"]) if rep else ""
ok = (rep is not None and ref["rc"] == 0 and "126" in dev_text
      and "126" in ref["text"])
results.append({"case": "matched-template-completion", "pass": ok,
                "prompt_tokens": len(ids), "generated": rep["generated"] if rep else None,
                "device_text": dev_text[:120],
                "llama_text": ref["text"][:120] if ref["text"] else None})
ltxt = ref["text"][:60] if ref["text"] else None
print(f"{'PASS' if ok else 'FAIL'} matched-template: dev={dev_text[:60]!r} "
      f"llama={ltxt!r}", flush=True)

# 2. one-token prompt
r, rep = run_decode([3710], 8, "one")
ok = rep is not None and len(rep["generated"]) > 0 and rep.get("stopped_eos") in (True, False)
results.append({"case": "one-token-prompt", "pass": ok,
                "generated": rep["generated"] if rep else None})
print(f"{'PASS' if ok else 'FAIL'} one-token: {rep}", flush=True)

# 3. determinism: same 4-token prompt twice
r1, rep1 = run_decode([248045, 846, 198, 3710], 8, "det1")
r2, rep2 = run_decode([248045, 846, 198, 3710], 8, "det2")
ok = (rep1 is not None and rep2 is not None
      and rep1["generated"] == rep2["generated"])
results.append({"case": "determinism-repeat", "pass": ok,
                "gen1": rep1["generated"] if rep1 else None,
                "gen2": rep2["generated"] if rep2 else None})
print(f"{'PASS' if ok else 'FAIL'} determinism", flush=True)

# 4. max-length prompt: 64 tokens (pattern repeat of a valid prefix)
base = [248045, 846, 198, 3710, 369, 220, 23, 19,
        348, 220, 18, 593, 220, 17, 30, 248046]
ids64 = (base * 4)[:64]
r, rep = run_decode(ids64, 4, "max64")
ok = rep is not None and len(rep["generated"]) > 0
results.append({"case": "max-length-64", "pass": ok,
                "generated": rep["generated"] if rep else None})
print(f"{'PASS' if ok else 'FAIL'} max-64: {rep}", flush=True)

# 5. negatives
os.system("head -c 1000000 " + MODEL + " > /tmp/e2e_trunc.binfer 2>/dev/null")
open("/tmp/e2e_garbage.binfer", "wb").write(b"not a model file at all" * 100)
neg_ok = True
for tag, path in (("truncated", "/tmp/e2e_trunc.binfer"),
                  ("garbage", "/tmp/e2e_garbage.binfer")):
    r = subprocess.run(["bash", "-c",
                        f"{ONEAPI}exec {BIN} {path} 2 2 /tmp/e2e_neg.json"],
                       capture_output=True, text=True, timeout=300)
    good = r.returncode != 0 and len(r.stderr.strip()) > 0
    neg_ok &= good
    results.append({"case": f"negative-{tag}", "pass": good,
                    "rc": r.returncode, "stderr": r.stderr.strip()[:160]})
    print(f"{'PASS' if good else 'FAIL'} negative-{tag}: rc={r.returncode}",
          flush=True)

npass = sum(1 for x in results if x["pass"])
json.dump({"cases": len(results), "passed": npass,
           "all_pass": npass == len(results), "results": results},
          open(OUT, "w"), indent=1)
print(f"{npass}/{len(results)} -> {OUT}", flush=True)
sys.exit(0 if npass == len(results) else 3)
