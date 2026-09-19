#!/usr/bin/env python3
"""Measure prefill throughput scaling across prompt lengths P in {8, 16, 32, 64, 128, 256}."""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_BIN = os.path.join(REPO_ROOT, "tools", "bench_258v", "bench_258v")
BINFER_MODEL = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
SPV_PATH = os.path.join(REPO_ROOT, "tools", "kernels_258v", "all_kernels.spv")

sysroot_lib = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = sysroot_lib + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")

# Prompt tokens: repeated valid vocabulary tokens
vocab_sample = [151644, 8948, 198, 2610, 525, 264, 10925, 151645]

lengths = [8, 16, 32, 64, 128, 256]
results = []

for P in lengths:
    prompt_ids = (vocab_sample * ((P + len(vocab_sample) - 1) // len(vocab_sample)))[:P]
    ids_csv = ",".join(str(x) for x in prompt_ids)
    rpt_file = f"/tmp/bench_prefill_p{P}.json"
    cmd = [
        BENCH_BIN,
        BINFER_MODEL,
        f"--spv={SPV_PATH}",
        f"--ids={ids_csv}",
        f"--decode-tokens=4",
        f"--warmup-runs=1",
        f"--measured-runs=3",
        f"--report={rpt_file}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode == 0 and os.path.exists(rpt_file):
        with open(rpt_file) as f:
            d = json.load(f)
            timing = d.get("timing_fields", {})
            pref_ms = timing.get("prefill_time_ms", 0.0)
            pref_tps = timing.get("prefill_tok_per_s", 0.0)
            results.append({
                "prompt_tokens": P,
                "prefill_time_ms": pref_ms,
                "prefill_tok_per_s": pref_tps,
            })
            print(f"P={P:3d}: {pref_ms:6.2f} ms ({pref_tps:6.2f} tok/s)")
            sys.stdout.flush()
        os.remove(rpt_file)
    else:
        print(f"P={P:3d}: FAILED\n{res.stderr}")

out_path = os.path.join(REPO_ROOT, "tools", "bench_258v", "report_prefill_scaling.json")
with open(out_path, "w") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nScaling report written to: {out_path}")
