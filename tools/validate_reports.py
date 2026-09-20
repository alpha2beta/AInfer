#!/usr/bin/env python3
"""T8.2: strict-parse every tracked JSON evidence report (Gate A).

Scope: tools/*/report_*.json + tools/*/golden_*.json + tools/*/corpus_*.json
+ tools/*/bench_*.json + reference/*.json + repo-root *.json (Stamp 14:
golden/corpus were gaps; B60-R1: bench schema added).
Exits 0 with a PASS line, or 1 listing each unparseable file.
Host-only, no GPU. Registered as ctest `reports_json`.
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = [
    "tools/*/report_*.json",
    "tools/*/golden_*.json",
    "tools/*/corpus_*.json",
    "tools/*/bench_*.json",
    "reference/*.json",
    "*.json",
]


def main():
    files = set()
    for pat in PATTERNS:
        files.update(glob.glob(os.path.join(REPO, pat)))
    files = sorted(f for f in files if os.path.isfile(f))
    bad = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                json.load(fh)
        except Exception as e:  # noqa: BLE001 - report then fail
            bad.append((os.path.relpath(f, REPO), str(e)[:120]))
    if bad:
        print(f"REPORTS-JSON FAIL: {len(bad)}/{len(files)} unparseable")
        for f, e in bad:
            print(f"  BAD {f}: {e}")
        return 1
    print(f"REPORTS-JSON-OK: {len(files)}/{len(files)} strict-parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
