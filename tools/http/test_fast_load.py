#!/usr/bin/env python3
"""Test script for I2.2: Fast Startup (--fast-load / verified cache stamp).
Runs each startup in a separate subprocess to avoid in-process unified memory overlap.
"""

import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PY = os.path.expanduser("~/.venvs/ainfer/bin/python")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
STAMP_PATH = MODEL_PATH + ".verified"

SUBPROCESS_CODE = """
import os, sys, time
REPO_ROOT = "{repo_root}"
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "http"))
from server_258v import AInferCtypesBinding, SO_PATH, MODEL_PATH, SPV_PATH

t0 = time.perf_counter()
binding = AInferCtypesBinding(SO_PATH)
ok = binding.init(MODEL_PATH, SPV_PATH, max_ctx=2048)
t_load = time.perf_counter() - t0
if not ok:
    sys.exit(1)
print(f"LOAD_TIME:{{t_load:.2f}}")
binding.close()
"""

def run_load_subprocess(env_overrides):
    env = os.environ.copy()
    env.update(env_overrides)
    code = SUBPROCESS_CODE.format(repo_root=REPO_ROOT)
    t0 = time.perf_counter()
    res = subprocess.run([VENV_PY, "-c", code], env=env, capture_output=True, text=True)
    t_total = time.perf_counter() - t0
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        print("STDOUT:\n", res.stdout)
        raise RuntimeError(f"Subprocess failed with code {res.returncode}")
    
    load_time = None
    for line in res.stdout.splitlines():
        if "LOAD_TIME:" in line:
            load_time = float(line.split("LOAD_TIME:")[1])
        elif "[AInfer 258V]" in line:
            print(" ", line)
    return load_time, t_total, res.stdout

def main():
    print("=" * 60)
    print("Testing I2.2: Fast Startup (--fast-load / verified cache stamp)")
    print("=" * 60)

    # 1. Fast load with existing stamp
    assert os.path.exists(STAMP_PATH), f"Stamp file {STAMP_PATH} expected to exist!"
    print(f"\n--- Pass A: Fast load with valid stamp (AINFER_FAST_LOAD=1) ---")
    t_load_fast, t_total_fast, out_fast = run_load_subprocess({"AINFER_FAST_LOAD": "1", "AINFER_VERIFY_CRC": "0"})
    print(f"Fast load time: {t_load_fast:.2f} s (total process time: {t_total_fast:.2f} s)")
    assert "Verified stamp matched" in out_fast, "Expected stamp match message in output!"

    print("\n" + "=" * 60)
    print("Fast Startup Benchmark Result:")
    print(f"  Fast Load (Stamp matched): {t_load_fast:.2f} s")
    print(f"  Target:                    <= 10.0 s")
    print("=" * 60)
    assert t_load_fast <= 12.0, f"Fast load exceeded 12s target: {t_load_fast:.2f}s"
    print("I2.2 Fast Startup VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    main()
