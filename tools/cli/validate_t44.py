"""T4.4 validation: determinism (same seed x2 identical), second prompt,
edge prompt. Saves tools/cli/report_t44.json. Each CLI call loads the 17GB
model (~20s incl. load), so this takes ~10 min total."""
import json
import os
import subprocess
import sys

VENV = os.path.expanduser("~/.venvs/ainfer/bin/python")
CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ainfer_cli.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_t44.json")


def run(msg, mtok=32, seed=0, tag=""):
    rp = f"/tmp/cli_{tag}.json"
    r = subprocess.run(
        [VENV, CLI, "--message", msg, "--max-tokens", str(mtok),
         "--timings", "--report", rp],
        capture_output=True, text=True, timeout=1500)
    rep = json.load(open(rp))
    rep["rc_outer"] = r.returncode
    return rep


results = []
r1 = run("What is 84 * 3 / 2? Reply with just the number.", tag="det1")
r2 = run("What is 84 * 3 / 2? Reply with just the number.", tag="det2")
det = r1["text"] == r2["text"] and r1["rc"] == 0
results.append({"case": "determinism-same-seed", "pass": det,
                "text": r1["text"][:120]})
print(f"{'PASS' if det else 'FAIL'} determinism: {r1['text'][:80]!r}")

r3 = run("Name the capital of France. Reply with just the name.", tag="cap")
ok3 = r3["rc"] == 0 and len(r3["text"]) > 0
results.append({"case": "second-prompt", "pass": ok3,
                "text": r3["text"][:120]})
print(f"{'PASS' if ok3 else 'FAIL'} second-prompt: {r3['text'][:80]!r}")

r4 = run("Hi.", tag="short")
ok4 = r4["rc"] == 0
results.append({"case": "short-prompt", "pass": ok4,
                "text": r4["text"][:120]})
print(f"{'PASS' if ok4 else 'FAIL'} short-prompt: {r4['text'][:80]!r}")

npass = sum(1 for r in results if r["pass"])
json.dump({"cases": len(results), "passed": npass,
           "all_pass": npass == len(results),
           "note": "backend=llama.cpp SYCL reference (native T4.2 backend pending); "
                   "total_s includes ~16-20s model load",
           "results": results,
           "timings": {k: r1[k] for k in ("total_s", "prompt_tokens")}},
          open(OUT, "w"), indent=1)
print(f"{npass}/{len(results)} -> {OUT}")
sys.exit(0 if npass == len(results) else 3)
