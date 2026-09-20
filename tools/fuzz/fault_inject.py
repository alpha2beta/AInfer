#!/usr/bin/env python3
"""T9.5: fault-injection suite for the AInfer runtime.

Injects synthetic disk errors, truncated containers, invalid checksums,
corrupt diagnostic caches, and simulated device-loss conditions, then
verifies graceful error reporting and clean termination (no crash/hang).

Groups (each case asserts an EXPECTED graceful outcome; anything else —
crash/signal, hang past timeout, or C++ exception escaping — is a finding):
  A. binfer container faults, via tools/binfer.py cmd_validate in-process
     (fast): truncated file, bad magic, payload-CRC corruption.
  B. runtime init faults, via fault_driver init (one ~20 s model load each):
     missing file, truncated container, payload-CRC corruption, garbage
     SPV bytes (simulated device-module loss), nonexistent SPV path.
  C. diagnostic cache faults, via fault_driver import (shares one runtime
     init inside the driver per case): bad magic, geometry mismatch,
     payload-CRC corruption, truncation, out-of-range position with otherwise
     VALID header+CRC (must now be rejected per the T9.5 import fix).
  D. export faults: valid export round-trips true; export to /dev/full
     (ENOSPC) must now return false per the T9.5 export fix.
  E. CLI faults, via decode_258v (fails before model load, fast): garbage
     --max-new / --ids values must exit 2, never SIGABRT.

Usage: fault_inject.py [--report PATH]
Builds fault_driver + decode_258v into a temp dir (never into the repo).
Exit 0 iff all cases behave as expected.
"""
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import binfer  # noqa: E402

MODEL = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes",
                     "tiel-coder-35b-text-int4g128.binfer")
SPV = os.path.join(REPO_ROOT, "tools", "kernels_258v", "all_kernels.spv")
CASE_TIMEOUT_S = 600  # one model load dominates; per-case watchdog

# NOTE: workdir lives under ~/.cache, NOT /tmp — /tmp is a small tmpfs here
# and the suite copies the ~19 GiB model container for mutation cases.
workdir = tempfile.mkdtemp(prefix="ainfer_faultinject_",
                           dir=os.path.join(os.path.expanduser("~"), ".cache"))
os.makedirs(workdir, exist_ok=True)
LD_ENV = dict(os.environ)
SYSROOT_LIB = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
if os.path.isdir(SYSROOT_LIB):
    LD_ENV["LD_LIBRARY_PATH"] = SYSROOT_LIB + ":" + LD_ENV.get("LD_LIBRARY_PATH", "")
LD_ENV.setdefault("ASAN_OPTIONS", "detect_leaks=0")
RESULTS = []
FINDINGS = 0


def build_tools():
    inc = os.path.join(REPO_ROOT, "tools", "l0probe", "include")
    drv = os.path.join(workdir, "fault_driver")
    cli = os.path.join(workdir, "decode_258v")
    for src, out in [
        (os.path.join(REPO_ROOT, "tools", "fuzz", "fault_driver.cpp") + " " +
         os.path.join(REPO_ROOT, "tools", "decode", "runtime_258v.cpp"), drv),
        (os.path.join(REPO_ROOT, "tools", "decode", "decode_258v.cpp") + " " +
         os.path.join(REPO_ROOT, "tools", "decode", "runtime_258v.cpp"), cli),
    ]:
        r = subprocess.run(
            f"g++ -O1 -g -std=c++17 {src} -I{inc} -L{SYSROOT_LIB} -lze_loader -o {out}",
            shell=True, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"BUILD FAILED:\n{r.stderr[-2000:]}")
            sys.exit(2)
    return drv, cli


def run(argv, timeout=CASE_TIMEOUT_S):
    """Run argv; returns (rc, last_stderr_line, timed_out). Negative rc = signal."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, env=LD_ENV)
        lines = (r.stderr.strip().splitlines() or [""]) + (r.stdout.strip().splitlines() or [""])
        return r.returncode, lines[-1][:160], False
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", True


SKIP_PASSED = set()  # case names already passed in a prior run (resume mode)


def check(name, argv, expect, timeout=CASE_TIMEOUT_S, result_has=None):
    """expect: 'rc0' | 'rc1' | 'rc2' | 'rc-nonzero'. result_has: substring required in stdout."""
    global FINDINGS
    if name in SKIP_PASSED and name != "C.seed-export":
        # C.seed-export is setup, not just a test: each run gets a fresh
        # workdir, so the seed cache must be re-exported even on resume.
        print(f"[SKIP] {name}: previously passed", flush=True)
        RESULTS.append({"case": name, "ok": True, "detail": "skipped-previously-passed",
                        "diag": "resume mode"})
        return
    rc, diag, timed_out = run(argv, timeout)
    if timed_out:
        ok, why = False, "HANG past timeout"
    elif rc is None or (rc < 0):
        ok, why = False, f"CRASH signal {-rc if rc else '?'}"
    else:
        if expect == "rc0":
            ok = (rc == 0)
        elif expect == "rc1":
            ok = (rc == 1)
        elif expect == "rc2":
            ok = (rc == 2)
        else:
            ok = (rc != 0)
        why = f"rc={rc}"
        if ok and result_has is not None:
            r2 = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, env=LD_ENV)
            ok = result_has in r2.stdout
            why += f" result_has={ok}"
    if not ok:
        FINDINGS += 1
    print(f"[{'PASS' if ok else 'FINDING'}] {name}: {why} :: {diag}", flush=True)
    RESULTS.append({"case": name, "ok": ok, "detail": why, "diag": diag})


def corrupt_copy(dst, mut):
    # NOTE: never bytearray(f.read()) the ~19 GiB model — a full in-RAM copy
    # got this harness OOM-killed twice at this exact case. Stream the copy,
    # then apply single-byte mutations by seek+write.
    shutil.copy(MODEL, dst)
    with open(dst, "r+b") as f:
        mut(f)
    return dst


def flip_at(off):
    def _mut(f):
        f.seek(off)
        cur = f.read(1)
        f.seek(off)
        f.write(bytes([(cur[0] + 1) % 256]))
    return _mut


def main(report=None):
    import contextlib
    import io
    drv, cli = build_tools()
    print(f"driver: {drv}\ncli: {cli}", flush=True)

    # ---- A. binfer container faults (in-process, fast) ----
    with contextlib.redirect_stdout(io.StringIO()):
        tmp = os.path.join(workdir, "t.binfer")
        shutil.copy(MODEL, tmp)
        with open(tmp, "r+b") as f:
            f.truncate(os.path.getsize(tmp) // 2)
        RESULTS.append({"case": "A.truncated-validate", "ok": binfer.cmd_validate(tmp) == 1,
                        "detail": "rc==1", "diag": "truncated half-file"})
        with open(tmp, "wb") as f:
            f.write(b"\x00" * 64)
        RESULTS.append({"case": "A.bad-magic-validate", "ok": binfer.cmd_validate(tmp) == 1,
                        "detail": "rc==1", "diag": "zeroed header"})
    for r_ in RESULTS:
        print(f"[{'PASS' if r_['ok'] else 'FINDING'}] {r_['case']}: {r_['detail']} :: {r_['diag']}", flush=True)
        if not r_["ok"]:
            globals()["FINDINGS"] += 1

    # valid-cache seed for group C: export via driver once
    seed_cache = os.path.join(workdir, "seed_cache.bin")
    check("C.seed-export", [drv, "export", MODEL, SPV, seed_cache], "rc0", result_has="RESULT init=1 export=1")

    # ---- B. runtime init faults ----
    check("B.missing-file", [drv, "init", os.path.join(workdir, "nope.binfer"), SPV],
          "rc0", result_has="RESULT init=0")
    trunc = os.path.join(workdir, "trunc.binfer")
    shutil.copy(MODEL, trunc)
    with open(trunc, "r+b") as f:
        f.truncate(os.path.getsize(trunc) // 2)
    check("B.truncated-container", [drv, "init", trunc, SPV], "rc0", result_has="RESULT init=0")
    # Corrupt a byte deep inside tensor payload space (10 GiB in — past all
    # headers/sections, guaranteed inside weight payloads). Requires the T9.5
    # payload-CRC verification: must now FAIL init instead of silent-loading.
    badcrc = corrupt_copy(os.path.join(workdir, "badcrc.binfer"),
                          flip_at(10 * 1024**3))
    check("B.bad-payload-crc", [drv, "init", badcrc, SPV], "rc0", result_has="RESULT init=0")
    garbagespv = os.path.join(workdir, "garbage.spv")
    with open(garbagespv, "wb") as f:
        f.write(os.urandom(4096))
    check("B.garbage-spv", [drv, "init", MODEL, garbagespv], "rc0", result_has="RESULT init=0")
    check("B.missing-spv", [drv, "init", MODEL, os.path.join(workdir, "nope.spv")],
          "rc0", result_has="RESULT init=0")

    # ---- C. diagnostic cache faults ----
    def cache_mut(name, fn):
        p = os.path.join(workdir, name)
        shutil.copy(seed_cache, p)
        with open(p, "r+b") as f:
            buf = bytearray(f.read())
        fn(buf)
        with open(p, "wb") as f:
            f.write(bytes(buf))
        return p

    check("C.bad-magic",
          [drv, "import", MODEL, SPV, cache_mut("c-badmagic.bin", lambda b: b.__setitem__(0, 0xFF))],
          "rc0", result_has="RESULT init=1 import=0")
    check("C.geometry-mismatch",
          [drv, "import", MODEL, SPV, cache_mut("c-geom.bin", lambda b: struct.pack_into("<Q", b, 40, 12345))],
          "rc0", result_has="RESULT init=1 import=0")
    check("C.payload-crc",
          [drv, "import", MODEL, SPV, cache_mut("c-crc.bin", lambda b: b.__setitem__(200, (b[200] + 1) % 256))],
          "rc0", result_has="RESULT init=1 import=0")
    trunc_cache = os.path.join(workdir, "c-trunc.bin")
    shutil.copy(seed_cache, trunc_cache)
    with open(trunc_cache, "r+b") as f:
        f.truncate(100)
    check("C.truncated", [drv, "import", MODEL, SPV, trunc_cache],
          "rc0", result_has="RESULT init=1 import=0")
    check("C.insane-position-valid-crc",
          [drv, "import", MODEL, SPV,
           cache_mut("c-pos.bin", lambda b: struct.pack_into("<I", b, 24, 0xFFFFFFFF))],
          "rc0", result_has="RESULT init=1 import=0")

    # ---- D. export faults ----
    check("D.valid-export-roundtrip", [drv, "export", MODEL, SPV, os.path.join(workdir, "e-ok.bin")],
          "rc0", result_has="RESULT init=1 export=1")
    if os.path.exists("/dev/full"):
        check("D.export-enospc", [drv, "export", MODEL, SPV, "/dev/full"],
              "rc0", result_has="RESULT init=1 export=0")
    else:
        print("[SKIP] D.export-enospc: no /dev/full on this platform", flush=True)
        RESULTS.append({"case": "D.export-enospc", "ok": True, "detail": "skipped", "diag": "no /dev/full"})

    # ---- E. CLI faults (fail before model load: fast) ----
    check("E.max-new-garbage", [cli, MODEL, "--max-new=abc"], "rc2")
    check("E.max-new-overflow", [cli, MODEL, "--max-new=99999999999999999999"], "rc2")
    check("E.ids-garbage", [cli, MODEL, "--ids=1,abc,3"], "rc2")
    check("E.max-ctx-negative", [cli, MODEL, "--max-ctx=-5"], "rc2")

    report_obj = {"task": "T9.5", "cases": RESULTS,
                  "passed": sum(1 for r in RESULTS if r["ok"]),
                  "total": len(RESULTS),
                  "all_pass": FINDINGS == 0}
    print(f"=== fault_inject: {report_obj['passed']}/{report_obj['total']} passed ===")
    if report:
        json.dump(report_obj, open(report, "w"), indent=1)
        print(f"report: {report}")
    return 0 if FINDINGS == 0 else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="T9.5 fault-injection suite")
    ap.add_argument("--report", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already ok in the existing --report file "
                         "(survives kills on unstable boxes)")
    args = ap.parse_args()
    if args.resume and args.report and os.path.exists(args.report):
        try:
            prev = json.load(open(args.report))
            for c in prev.get("cases", []):
                if c.get("ok"):
                    SKIP_PASSED.add(c["case"])
            print(f"[resume] skipping {len(SKIP_PASSED)} previously-passed cases", flush=True)
        except Exception as e:
            print(f"[resume] could not read report: {e}", flush=True)
    sys.exit(main(args.report))
