"""T6.2 decode/prefill sweep on the adopted loop (decode_l0, AINFER_PROFILE=1).

Points (prompt length x gen): 1x8, 16x8, 64x8, 256x4. Prompts are prefixes /
repeats of validated id sequences (no new tokenization ground to cover).
Per point: wall time, generated tokens + sanity (nonempty, finite top5),
per-class profile medians, derived steady s/token and prefill tok/s.
Saves tools/t62/report_t62.json.
"""
import json
import os
import re
import subprocess
import sys
import time

REPO = "/mnt/usb/AInfer"
BIN = os.path.join(REPO, "build-b60/tools/decode/decode_l0")
MODEL = os.path.join(REPO, "models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer")
SPVDIR = os.path.join(REPO, "build-b60/tools/cmdlist")
OUT = os.path.join(REPO, "tools/t62/report_t62.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ONEAPI = "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; "

BASE16 = [248045, 846, 198, 3710, 369, 220, 23, 19,
          348, 220, 18, 593, 220, 17, 30, 248046]
POINTS = [
    ("p001", [3710], 8),
    ("p016", BASE16[:16], 8),
    ("p064", (BASE16 * 4)[:64], 8),
    ("p256", (BASE16 * 16)[:256], 4),
]


def run(ids, max_new, tag, timeout=1500):
    rp = f"/tmp/sweep_{tag}.json"
    cmd = (f"{ONEAPI}AINFER_PROFILE=1 exec {BIN} {MODEL} {len(ids)} 1 "
           f"{SPVDIR} {rp} --ids={','.join(map(str, ids))} "
           f"--max-new={max_new}")
    t0 = time.time()
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=timeout)
    dt = time.time() - t0
    rep = json.load(open(rp)) if r.returncode == 0 else None
    prof = {}
    for m in re.finditer(r"^(linear_layer|attn_layer|tail|embed|control|"
                         r"readback|topk)\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         r.stderr + r.stdout, re.M):
        prof[m.group(1)] = float(m.group(2)) / 1000.0
    tops = re.findall(r"TOP5@s\d+: (.+?) \|", r.stdout + r.stderr)
    finite = all(v not in ("nan", "inf", "-inf")
                 for t in tops
                 for v in re.findall(r"\((-?\d+\.\d+|nan|inf)", t))
    return r, rep, dt, prof, finite


rows = []
for tag, ids, mg in POINTS:
    r, rep, dt, prof, finite = run(ids, mg, tag)
    lin = prof.get("linear_layer", 0)
    att = prof.get("attn_layer", 0)
    tail = prof.get("tail", 0)
    steady = 48 * lin + 16 * att + tail
    prefill = 48 * lin + 16 * att  # same lists, no tail/logits
    ntok = len(rep["generated"]) if rep else 0
    ok = rep is not None and ntok > 0 and finite
    rows.append({"point": tag, "prompt": len(ids), "max_new": mg,
                 "pass": ok, "generated": ntok, "finite_top5": finite,
                 "wall_s": round(dt, 1),
                 "lin_ms": round(lin * 1000, 3), "attn_ms": round(att * 1000, 3),
                 "tail_ms": round(tail * 1000, 3),
                 "steady_s_per_tok": round(steady, 4),
                 "steady_tok_per_s": round(1 / steady, 2) if steady else 0,
                 "prefill_tok_per_s": round(1 / prefill, 2) if prefill else 0})
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: P={len(ids)} gen={ntok} "
          f"wall={dt:.0f}s steady={steady * 1000:.0f}ms/tok "
          f"({1 / steady:.1f} t/s) prefill~{1 / prefill:.0f} t/s", flush=True)

npass = sum(1 for x in rows if x["pass"])
json.dump({"device": "B60", "backend": "raw-L0 recorded lists",
           "points": len(rows), "passed": npass,
           "all_pass": npass == len(rows), "rows": rows},
          open(OUT, "w"), indent=1)
print(f"{npass}/{len(rows)} -> {OUT}", flush=True)
sys.exit(0 if npass == len(rows) else 3)
