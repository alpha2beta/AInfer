#!/bin/bash
# Benchmark discipline driver (cookbook 6.8): discarded warmup + repeated
# measured runs. Usage: bench_ab.sh <label> <reps> -- <command...>
# Runs once discarded (warmup: page cache, GPU clocks, kernel caches),
# then <reps> measured runs, printing each result line containing
# "tok/s" or "chunks x" plus median wall time. Cold-prefix rule: the
# caller must use matched natural prompts, never filler, for published
# numbers (filler is fine for timing-only sweeps; label it).
set -u
LABEL="$1"; REPS="$2"; shift 2
[ "${1:-}" = "--" ] && shift
echo "===== $LABEL: 1 warmup (discarded) + $REPS measured ====="
"$@" > /tmp/bench_warm.log 2>&1 || { echo "WARMUP FAILED"; tail -3 /tmp/bench_warm.log; exit 1; }
echo "warmup done (discarded)"
: > /tmp/bench_times.txt
for i in $(seq 1 "$REPS"); do
  "$@" > /tmp/bench_rep$i.log 2>&1 || { echo "REP $i FAILED"; tail -3 /tmp/bench_rep$i.log; exit 1; }
  grep -hE "tok/s|chunks x" /tmp/bench_rep$i.log | head -2
done
echo "===== $LABEL done ====="
