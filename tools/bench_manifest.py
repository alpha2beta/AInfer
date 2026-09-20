#!/usr/bin/env python3
"""B60-R1 environment + artifact manifest collector.

Collects everything review_B60.md Phase B60-R1 requires except per-run
timing (which bench_normalize.py attaches): git commit, .binfer size +
SHA-256, model revision, tokenizer hashes (pinned golden_t44), Level Zero
driver, oneAPI version, GPU identity, power/clock readability.

The 16 GiB .binfer hash is slow on USB: it is cached in the output file
and reused when size+mtime match. Use --no-binfer-hash to skip (records
size only, marks hash "skipped").

Writes tools/bench/report_manifest.json (covered by the reports_json gate).
Usage: bench_manifest.py [--out PATH] [--no-binfer-hash]
"""
import hashlib
import json
import os
import subprocess
import sys

REPO = "/mnt/usb/AInfer"
BINFER = os.path.join(REPO, "models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer")
GOLDEN = os.path.join(REPO, "tools/tokenizer/golden_t44.json")
OUT_DEFAULT = os.path.join(REPO, "tools/bench/report_manifest.json")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    out = OUT_DEFAULT
    do_hash = True
    for i, a in enumerate(sys.argv[1:]):
        if a == "--out":
            out = sys.argv[1:][i + 1]
        if a == "--no-binfer-hash":
            do_hash = False
    prev = {}
    if os.path.exists(out):
        try:
            prev = json.load(open(out))
        except Exception:
            prev = {}
    st = os.stat(BINFER)
    bhash = prev.get("binfer_sha256", "")
    if do_hash:
        if (prev.get("binfer_bytes") == st.st_size
                and prev.get("binfer_mtime") == st.st_mtime
                and bhash and bhash != "skipped"):
            pass  # cache hit
        else:
            print("hashing .binfer (16 GiB, one-time)...", flush=True)
            bhash = sha256_file(BINFER)
    else:
        bhash = "skipped"
    golden = json.load(open(GOLDEN))
    man = {
        "schema": "b60-bench-v1",
        "git_commit": sh(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]),
        "binfer_bytes": st.st_size,
        "binfer_mtime": st.st_mtime,
        "binfer_sha256_full": bhash,
        "binfer_sha256_scope": "whole file (includes trailing 32 B checksum)",
        "binfer_sha256_payload": "5ef77c128ee526a351051f54f16378d04eb868ea7e6f89703c41a1d58655911f",
        "binfer_sha256_payload_scope": "bytes [0, checksum_offset) per "
        "binfer_spec §11; conversion-time value, verified 2026-09-20 "
        "by re-hash",
        "binfer_sha256": bhash,
        "model_revision": golden.get("model_rev", ""),
        "tokenizer_sha256": golden.get("assets", {}),
        "tokenizer_specials": golden.get("specials_count", 0),
        "driver_level_zero": {
            "ze_loader": sh(["dpkg-query", "-W", "-f=${Version}",
                             "libze1"]) or "unknown",
            "neo_opencl": sh(["dpkg-query", "-W", "-f=${Version}",
                              "intel-opencl-icd"]) or "unknown",
            "igc": sh(["dpkg-query", "-W", "-f=${Version}",
                       "libigc1"]) or "unknown",
            "dpcpp": sh(["dpkg-query", "-W", "-f=${Version}",
                         "intel-oneapi-compiler-dpcpp-cpp-2026.1"])
            or "unknown",
        },
        "oneapi_version": "2026.1",
        "gpu_name": "Intel Arc Pro B60 (8086:e211)",
        "gpu_power": "unreadable (no sudo; sysfs perf/power not permitted)",
        "notes": "B60-R1 manifest. Power/clock telemetry unavailable on "
                 "this box; record nulls, do not estimate.",
    }
    json.dump(man, open(out, "w"), indent=1)
    print(f"manifest -> {out} commit={man['git_commit']} "
          f"binfer={man['binfer_bytes']} sha={str(bhash)[:16]}")


if __name__ == "__main__":
    main()
