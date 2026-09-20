#!/usr/bin/env python3
"""T9.4: structure-aware fuzzer for the .binfer container parser.

Target: tools/binfer.py cmd_validate / _validate_inner (pure Python, fast).
Seed: first 8 KiB of the real pinned container (valid magic, version,
sections incl. MoE Section 6, real directory entries).
Mutators: byte flips, truncation, u32/u64 field overwrites at header offsets
(puts huge n/scount on purpose — T9.4 caps must turn these into fast INVALID),
section-header shuffling.

Contract under test: cmd_validate() must ALWAYS return 0/1 promptly —
no escaping exception, no hang, no pathological allocation. Any escaping
exception type, return code outside {0,1}, or per-case time over the hang
budget is a fuzzer FAILURE (exit 1 at the end).

Usage: fuzz_binfer.py [--iters N] [--seed SEED] [--report PATH]
"""
import argparse
import os
import random
import struct
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import binfer  # noqa: E402

SEED_CONTAINER = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes",
    "tiel-coder-35b-text-int4g128.binfer")
SEED_BYTES = 8192
HANG_BUDGET_S = 5.0  # per-case wall clock: anything slower is a hang finding

# Header field offsets in the .binfer container (from _validate_inner layout)
MAGIC_OFF = 0
VERSION_OFF = 8
FLAGS_OFF = 12
N_OFF = 16        # u64 tensor count
TABLE_OFF = 24    # u64 section-table offset
SCOUNT_OFF = 32   # u32 section count


def load_seed():
    with open(SEED_CONTAINER, "rb") as f:
        data = f.read(SEED_BYTES)
    assert data[:8] == b"BINFER\x00\x01", "seed container magic mismatch"
    return bytearray(data)


def mutate(rng, seed):
    """Return one mutated input exercising a distinct parser path."""
    out = bytearray(seed)
    choice = rng.randrange(10)
    if choice == 0:  # sparse byte flips
        for _ in range(rng.randrange(1, 8)):
            out[rng.randrange(len(out))] = rng.randrange(256)
    elif choice == 1:  # truncation (incl. empty / sub-header)
        cut = rng.choice([0, 1, 7, 8, 15, 16, 31, 47, 48,
                          rng.randrange(len(out))])
        out = out[:cut]
    elif choice == 2:  # huge tensor count (T9.4 hang primitive)
        struct.pack_into("<Q", out, N_OFF, rng.choice([0, 1, 713, 100000, 100001,
                                                       2**32, 2**64 - 1]))
    elif choice == 3:  # huge section count (T9.4 hang primitive)
        struct.pack_into("<I", out, SCOUNT_OFF, rng.choice([0, 1, 7, 1024, 1025,
                                                            2**24, 2**32 - 1]))
    elif choice == 4:  # corrupt magic / version / flags
        field = rng.choice([MAGIC_OFF, VERSION_OFF, FLAGS_OFF])
        ln = {MAGIC_OFF: 8, VERSION_OFF: 4, FLAGS_OFF: 4}[field]
        for i in range(ln):
            out[field + i] = rng.randrange(256)
    elif choice == 5:  # wild table offset / total
        struct.pack_into("<Q", out, TABLE_OFF, rng.choice([0, 8, 2**40, 2**64 - 1]))
    elif choice == 6:  # dense random block overwrite
        pos = rng.randrange(max(1, len(out) - 64))
        ln = rng.randrange(1, 65)
        for i in range(ln):
            out[pos + i] = rng.randrange(256)
    elif choice == 7:  # shuffle 32-byte section-header slots in place
        base = 64
        slots = (len(out) - base) // 32
        if slots >= 2:
            a, b = rng.sample(range(slots), 2)
            ao, bo = base + 32 * a, base + 32 * b
            out[ao:ao + 32], out[bo:bo + 32] = out[bo:bo + 32], out[ao:ao + 32]
    elif choice == 8:  # flip every byte in a random 4-byte field to 0xFF
        pos = rng.randrange(max(1, len(out) - 4))
        out[pos:pos + 4] = b"\xff\xff\xff\xff"
    else:  # single-bit flips across the header
        for _ in range(rng.randrange(1, 5)):
            i = rng.randrange(len(out))
            out[i] ^= 1 << rng.randrange(8)
    return bytes(out)


def run_case(data, tmpdir, idx):
    """Run one case; returns (verdict, seconds). Verdict in {ok, escape, hang, badrc}."""
    import contextlib
    import io
    path = os.path.join(tmpdir, f"fz{idx}.binfer")
    with open(path, "wb") as f:
        f.write(data)
    t0 = time.perf_counter()
    try:
        # cmd_validate is chatty (prints INVALID per case); silence it —
        # only the return code and escaping exceptions matter here.
        with contextlib.redirect_stdout(io.StringIO()):
            rc = binfer.cmd_validate(path)
    except Exception as e:  # noqa: BLE001 - ANY escape is a finding
        return (f"escape:{type(e).__name__}:{e}", time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    if dt > HANG_BUDGET_S:
        return (f"hang:{dt:.1f}s", dt)
    if rc not in (0, 1):
        return (f"badrc:{rc}", dt)
    return ("ok", dt)


def main():
    ap = argparse.ArgumentParser(description="T9.4 .binfer parser fuzzer")
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=94321)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    seed = load_seed()
    rng = random.Random(args.seed)
    tmpdir = tempfile.mkdtemp(prefix="fuzz_binfer_")
    findings = []
    worst = 0.0
    t_all = time.perf_counter()
    for i in range(args.iters):
        data = mutate(rng, seed)
        verdict, dt = run_case(data, tmpdir, i)
        worst = max(worst, dt)
        if verdict != "ok":
            findings.append({"iter": i, "verdict": verdict})
            print(f"[FINDING] iter={i} {verdict}", flush=True)
            if len(findings) >= 20:
                print("finding cap reached; stopping early")
                break
        if (i + 1) % 100000 == 0:
            print(f"[{i + 1}/{args.iters}] worst-case {worst * 1000:.1f} ms, "
                  f"findings {len(findings)}", flush=True)
    total_s = time.perf_counter() - t_all
    done = (i + 1) if 'i' in dir() else 0
    report = {
        "task": "T9.4",
        "target": "tools/binfer.py cmd_validate",
        "seed": args.seed,
        "iterations": done,
        "findings": findings,
        "worst_case_s": worst,
        "total_s": total_s,
        "all_pass": not findings,
    }
    print(f"=== fuzz_binfer: {done} iters, {len(findings)} findings, "
          f"worst {worst * 1000:.1f} ms, {total_s:.1f}s total ===")
    if args.report:
        import json
        json.dump(report, open(args.report, "w"), indent=1)
        print(f"report: {args.report}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
