#!/usr/bin/env python3
"""T1.6 CPU-side memory stressors (run alongside the iGPU bench).

Modes (each saturates DRAM channels from the CPU side):
  triad     streaming a=b+c*d triad over --gib GiB, --workers processes
  tokenize  tight tok.py encode loop (realistic host-orchestration load)

Runs until killed (the driver manages lifetime) or --seconds elapses.
Writes periodic RSS/throughput heartbeats to stderr (not parsed).
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def triad_worker(gib, seconds):
    import numpy as np
    n = int(gib * 1024**3 // 8)
    rng = np.random.default_rng(os.getpid())
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    t0 = time.perf_counter()
    it = 0
    while True:
        # streaming triad + reductions (keeps traffic honest, defeats DCE)
        np.multiply(c, d, out=a)
        np.add(a, b, out=a)
        s = float(a.sum())
        it += 1
        if it == 1:
            sys.stderr.write(f"[triad pid={os.getpid()}] {gib} GiB, checksum={s:.3f}\n")
            sys.stderr.flush()
        if seconds and (time.perf_counter() - t0) > seconds:
            break


def tokenize_worker(seconds, worker_id):
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
    import tok as ainfer_tok
    tok = ainfer_tok.load()
    prompts = [
        "Write a concise quicksort algorithm in C++.",
        "Explain the architectural differences between DeltaNet linear attention and standard multi-head attention.",
        "Implement a thread-safe circular buffer in C++ using std::atomic and cache-line padding.",
        "Summarize the Python GIL trade-offs for multithreaded tokenization workloads.",
    ]
    t0 = time.time()
    n = 0
    while True:
        for p in prompts:
            ainfer_tok.encode(tok, p)
            n += 1
        if n == len(prompts):
            sys.stderr.write(f"[tokenize id={worker_id}] warmed up\n")
            sys.stderr.flush()
        if seconds and (time.time() - t0) > seconds:
            break
    sys.stderr.write(f"[tokenize id={worker_id}] {n} encodes done\n")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description="T1.6 CPU memory stressor")
    ap.add_argument("mode", choices=["triad", "tokenize"])
    ap.add_argument("--workers", type=int, default=4)
    # NOTE: footprint per worker is 4x this (a/b/c/d arrays). Keep total
    # workers*gib*4 well under RAM: 7 workers x 0.25 GiB = 7 GiB total.
    # A 19 GiB total once OOM-hung this suite; streaming pressure comes from
    # access rate (arrays still >> LLC), not footprint size.
    ap.add_argument("--gib", type=float, default=0.25, help="GiB per triad array (x4 arrays per worker)")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until killed")
    ap.add_argument("--cpus", default="", help="taskset CPU list, e.g. 1-7")
    args = ap.parse_args()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for w in range(args.workers):
        if args.mode == "triad":
            p = ctx.Process(target=triad_worker, args=(args.gib, args.seconds))
        else:
            p = ctx.Process(target=tokenize_worker, args=(args.seconds, w))
        p.start()
        procs.append(p)
    if args.cpus:
        for p in procs:
            os.system(f"taskset -pc {args.cpus} {p.pid} >/dev/null 2>&1")
    sys.stderr.write(f"[stress] {args.mode} x{args.workers} started: {[p.pid for p in procs]}\n")
    sys.stderr.flush()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
    sys.stderr.write("[stress] done\n")


if __name__ == "__main__":
    main()
