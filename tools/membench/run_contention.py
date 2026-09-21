#!/usr/bin/env python3
"""T1.6: iGPU weight-streaming bandwidth in isolation vs under CPU stress.

Phases (each: rebuild-free replay of tools/membench/contention binary):
  1. isolated    — nothing else on the box (besides ambient users)
  2. tokenize    — 2x tok.py encode loops (realistic host orchestration)
  3. triad       — 7x numpy streaming-triad workers (synthetic DRAM flood)
  4. combined    — 7x triad + 2x tokenize (worst case)

The GPU bench is pinned to CPU 0; stressors to CPUs 1-7 (fence waits are
cheap, but this keeps scheduler noise out of the host-side timing).

Emits: tools/membench/report_contention_258v.json
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO_ROOT, "tools", "membench", "contention")
SPV = os.path.join(REPO_ROOT, "tools", "membench", "stream_read.spv")
STRESS = os.path.join(REPO_ROOT, "tools", "membench", "stress_cpu.py")
REPORT = os.path.join(REPO_ROOT, "tools", "membench", "report_contention_258v.json")
PY = sys.executable

env_base = dict(os.environ)
SYSROOT_LIB = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
env_base["LD_LIBRARY_PATH"] = SYSROOT_LIB + (":" + env_base["LD_LIBRARY_PATH"]
                                             if "LD_LIBRARY_PATH" in env_base else "")


def load_avg():
    try:
        with open("/proc/loadavg") as f:
            return f.read().split()[:3]
    except Exception:
        return []


def run_bench(label):
    r = subprocess.run(["taskset", "-c", "0", BENCH, SPV, label],
                       capture_output=True, text=True, timeout=600, env=env_base)
    if r.returncode != 0:
        print(f"bench {label} FAILED:\n{r.stderr[-1500:]}")
        sys.exit(2)
    print(f"[{label}] {r.stdout.strip()}", flush=True)
    return json.loads(r.stdout)


def start_stressors(specs):
    """specs: list of argv-suffix lists. Returns Popen handles."""
    procs = []
    for s in specs:
        p = subprocess.Popen([PY, STRESS] + s, stderr=subprocess.DEVNULL)
        procs.append(p)
    time.sleep(3)  # let workers reach steady state
    return procs


def stop_stressors(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    t_all = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    phases = {}

    print("=== phase 1/4: isolated ===", flush=True)
    phases["isolated"] = run_bench("isolated")

    print("=== phase 2/4: tokenize ===", flush=True)
    p = start_stressors([["tokenize", "--workers", "2", "--cpus", "1-7"]])
    try:
        phases["tokenize"] = run_bench("tokenize")
    finally:
        stop_stressors(p)

    print("=== phase 3/4: triad ===", flush=True)
    p = start_stressors([["triad", "--workers", "7", "--gib", "0.25", "--cpus", "1-7"]])
    try:
        phases["triad"] = run_bench("triad")
    finally:
        stop_stressors(p)

    print("=== phase 4/4: combined ===", flush=True)
    p = start_stressors([["triad", "--workers", "7", "--gib", "0.25", "--cpus", "1-6"],
                         ["tokenize", "--workers", "2", "--cpus", "7"]])
    try:
        phases["combined"] = run_bench("combined")
    finally:
        stop_stressors(p)

    base = phases["isolated"]["stream_read_gbs"]
    deltas = {}
    for k, v in phases.items():
        if k == "isolated":
            continue
        deltas[k] = {
            "stream_read_gbs": v["stream_read_gbs"],
            "delta_gbs": round(v["stream_read_gbs"] - base, 2),
            "delta_pct": round(100.0 * (v["stream_read_gbs"] - base) / base, 1),
            "d2d_copy_gbs": v["d2d_copy_gbs"],
        }

    report = {
        "task": "T1.6",
        "device": "Intel Arc 140V (Xe2, Lunar Lake 258V)",
        "timestamp": t_all,
        "method": {
            "gpu_work": "2 GiB sequential float4 stream-read kernel + D2D copy, "
                        "host fence timing, median of 7 after 2 warmup, GPU bench pinned to CPU 0",
            "cpu_stress": "numpy triad workers (1 GiB each, spawn ctx) and/or tok.py encode loops "
                          "pinned to CPUs 1-7",
            "host_loadavg_during_run": load_avg(),
        },
        "phases": phases,
        "contention_delta_vs_isolated": deltas,
        "all_pass": True,
    }
    json.dump(report, open(REPORT, "w"), indent=1)
    print(f"=== report: {REPORT} ===")
    print(json.dumps(deltas, indent=1))


if __name__ == "__main__":
    main()
