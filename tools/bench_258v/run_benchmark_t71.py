#!/usr/bin/env python3
"""Task T7.1: Standardized benchmark harness with isolated timing fields for 258V.

Reports:
  - model load time (ms)
  - tokenization time (ms)
  - prefill time (ms)
  - prefill tokens/s
  - first decode latency (ms)
  - cold TTFT (ms)
  - warm TTFT (ms)
  - sustained decode tokens/s
  - p50/p90/p95/p99 inter-token jitter (ms)
  - memory RSS and device arena footprints

Emits: tools/bench_258v/report_bench_t71.json
"""

import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as ainfer_tok

BENCH_BIN = os.path.join(REPO_ROOT, "tools", "bench_258v", "bench_258v")
BINFER_MODEL = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
SPV_PATH = os.path.join(REPO_ROOT, "tools", "kernels_258v", "all_kernels.spv")
REPORT_PATH = os.path.join(REPO_ROOT, "tools", "bench_258v", "report_bench_t71.json")


def measure_tokenization():
    print("[T7.1] Measuring isolated tokenization latency...")
    t0 = time.perf_counter()
    tokenizer = ainfer_tok.load()
    t_tok_load = (time.perf_counter() - t0) * 1000.0

    prompts = [
        "Hello! Can you write a Python function to solve the two-sum problem?",
        "Explain the architectural differences between DeltaNet linear attention and standard multi-head attention.",
        "Implement a thread-safe circular buffer in C++ using std::atomic and cache-line padding.",
    ]

    timings = []
    token_counts = []
    for p in prompts:
        # Warmup
        ainfer_tok.encode(tokenizer, p)
        # Measured runs
        t_start = time.perf_counter()
        for _ in range(50):
            ids = ainfer_tok.encode(tokenizer, p)
        t_end = time.perf_counter()
        avg_ms = ((t_end - t_start) / 50.0) * 1000.0
        timings.append(avg_ms)
        token_counts.append(len(ids))

    mean_tok_ms = sum(timings) / len(timings)
    print(f"  Tokenizer load time: {t_tok_load:.2f} ms")
    print(f"  Mean tokenization time: {mean_tok_ms:.3f} ms (across {len(prompts)} prompts)")

    sample_prompt = "<|im_start|>user\nWrite a concise quicksort algorithm in C++.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    sample_ids = ainfer_tok.encode(tokenizer, sample_prompt)

    return {
        "tokenizer_load_time_ms": t_tok_load,
        "mean_tokenization_time_ms": mean_tok_ms,
        "sample_prompt_tokens": len(sample_ids),
        "sample_ids": sample_ids,
    }


def run_cpp_bench(prompt_ids, decode_tokens=64, measured_runs=3):
    print(f"[T7.1] Running C++ Level Zero benchmark harness ({len(prompt_ids)} prompt tok, {decode_tokens} gen tok)...")
    ids_csv = ",".join(str(x) for x in prompt_ids)
    cmd = [
        BENCH_BIN,
        BINFER_MODEL,
        f"--spv={SPV_PATH}",
        f"--ids={ids_csv}",
        f"--decode-tokens={decode_tokens}",
        f"--warmup-runs=1",
        f"--measured-runs={measured_runs}",
        f"--report={REPORT_PATH}",
    ]

    env = dict(os.environ)
    sysroot_lib = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
    env["LD_LIBRARY_PATH"] = sysroot_lib + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"ERROR: bench_258v failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        sys.exit(1)

    print(res.stdout)
    with open(REPORT_PATH, "r") as f:
        cpp_report = json.load(f)
    return cpp_report


def main():
    tok_data = measure_tokenization()
    cpp_data = run_cpp_bench(tok_data["sample_ids"], decode_tokens=64, measured_runs=3)

    # Augment report with tokenization data and end-to-end TTFT
    timing = cpp_data["timing_fields"]
    timing["tokenization_time_ms"] = tok_data["mean_tokenization_time_ms"]
    timing["tokenizer_load_time_ms"] = tok_data["tokenizer_load_time_ms"]
    timing["e2e_cold_ttft_ms"] = timing["cold_ttft_ms"] + timing["tokenization_time_ms"]
    timing["e2e_warm_ttft_ms"] = timing["warm_ttft_ms"] + timing["tokenization_time_ms"]

    cpp_data["status"] = "PASSED"
    cpp_data["summary"] = {
        "model_load_s": round(timing["model_load_time_ms"] / 1000.0, 3),
        "prefill_tok_per_s": round(timing["prefill_tok_per_s"], 2),
        "warm_ttft_ms": round(timing["warm_ttft_ms"], 2),
        "e2e_warm_ttft_ms": round(timing["e2e_warm_ttft_ms"], 2),
        "first_decode_latency_ms": round(timing["first_decode_latency_ms"], 2),
        "sustained_decode_tok_per_s": round(timing["sustained_decode_tok_per_s"], 2),
        "p50_jitter_ms": round(timing["inter_token_jitter"]["p50_ms"], 2),
        "p95_jitter_ms": round(timing["inter_token_jitter"]["p95_ms"], 2),
        "total_committed_gib": round(cpp_data["memory_footprint"]["total_committed_gib"], 2),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(cpp_data, f, indent=2)

    print(f"[T7.1] Final standardized benchmark report saved to {REPORT_PATH}")
    print("Summary:")
    print(json.dumps(cpp_data["summary"], indent=2))


if __name__ == "__main__":
    main()
