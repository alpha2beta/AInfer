#!/usr/bin/env python3
"""B60-R1 run normalizer: decode_l0 report + run metadata -> schema result.

Joins a raw decode_l0 JSON report with the invocation metadata the report
cannot know (KV precision, MTP mode, chunk size, warm state, timing
boundaries, comparison class) and emits a bench_schema.json-conformant
result file. Missing optional timing fields become null (never estimated).

Enforces the schema contract: prefill excludes load/tokenization unless
boundaries say so; accepted-token rate is separate from eval rate; the
manifest (report_manifest.json) commit is pinned into every result.

Usage:
  bench_normalize.py --report /tmp/pf.json --command "..." --kv bf16|int8
    --mtp off|depth-1|depth-2|adaptive --chunk 256 --warm cold|warm
    [--context N] [--class local_reproduced|same_class_external|
    different_checkpoint|unverified] [--load-s X] [--tok-s X]
    [--ttft-cold X] [--ttft-warm X] [--vram BYTES]
    [--out tools/bench/report_<name>.json]
  bench_normalize.py --check <result.json>  (schema-conformance check only)
"""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"
SCHEMA = os.path.join(REPO, "tools/bench/bench_schema.json")
MANIFEST = os.path.join(REPO, "tools/bench/report_manifest.json")


def fail(msg):
    print(f"bench_normalize FAIL: {msg}", flush=True)
    return 1


def check_result(d):
    s = json.load(open(SCHEMA))
    errs = []
    for k in s["result_required"]:
        if k not in d:
            errs.append(f"missing required: {k}")
    for k in ("prompt_tokens", "context_occupancy", "generated_tokens",
              "decode_tokens_per_second"):
        if d.get(k) is None:
            errs.append(f"null required: {k}")
    md = d.get("metadata", {})
    for k in s["result_metadata_required"]:
        if k not in md:
            errs.append(f"missing metadata: {k}")
    b = md.get("boundaries", {})
    for k in s["boundaries_required"]:
        if k not in b:
            errs.append(f"missing boundaries: {k}")
    rt = md.get("run_type")
    if rt not in s["run_type_enum"]:
        errs.append("bad run_type")
    has_pre = d.get("prefill_seconds") is not None
    if rt == "decode_only" and has_pre:
        errs.append("decode_only must have null prefill")
    if rt == "prefill_decode" and not has_pre:
        errs.append("prefill_decode needs prefill_seconds")
    if md.get("comparison_class") not in s["comparison_class_enum"]:
        errs.append("bad comparison_class")
    if md.get("warm_state") not in s["warm_state_enum"]:
        errs.append("bad warm_state")
    if md.get("kv_precision") not in s["kv_enum"]:
        errs.append("bad kv_precision")
    if md.get("mtp_mode") not in s["mtp_mode_enum"]:
        errs.append("bad mtp_mode")
    return errs


def main():
    a = sys.argv[1:]
    if a[:1] == ["--check"] or (len(a) == 2 and a[0] == "--check"):
        d = json.load(open(a[1]))
        errs = check_result(d)
        if errs:
            print("SCHEMA-FAIL:")
            for e in errs:
                print(f"  {e}")
            return 1
        print(f"SCHEMA-OK: {a[1]}")
        return 0
    g = {"report": None, "command": "", "kv": "bf16", "mtp": "off",
         "chunk": 256, "warm": "warm", "context": None, "run-type": None,
         "class": "local_reproduced", "load-s": None, "tok-s": None,
         "ttft-cold": None, "ttft-warm": None, "vram": None, "out": None}
    i = 0
    while i < len(a):
        k = a[i].lstrip("-")
        if k in g:
            g[k] = a[i + 1]
            i += 2
        else:
            return fail(f"unknown flag --{a[i].lstrip('-')}")
    if not g["report"]:
        return fail("--report required")
    rep = json.load(open(g["report"]))
    prompt = rep.get("prompt", [])
    gen = rep.get("generated", [])
    pre_ms = rep.get("prefill_ms")
    dec_ms = rep.get("decode_ms")
    mtp = rep.get("mtp")
    n_prompt = len(prompt)
    ctx = int(g["context"]) if g["context"] else n_prompt + len(gen)
    pre_tps = (n_prompt / (pre_ms / 1000.0)) if pre_ms else None
    dec_tps = rep.get("decode_tok_per_sec")
    acc_tps = None
    alpha = None
    if mtp and dec_ms:
        # Accepted-token throughput: accepted tokens share the same wall.
        # mtp.accepted counts accepted drafts; total output = len(gen).
        alpha = mtp.get("alpha")
        acc_tps = len(gen) / (dec_ms / 1000.0) if dec_ms else None
    man = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    out = {
        "schema": "b60-bench-v1",
        "manifest_commit": man.get("git_commit", "unknown"),
        "prompt_tokens": n_prompt,
        "context_occupancy": ctx,
        "generated_tokens": len(gen),
        "warm_state": g["warm"],
        "model_load_seconds": float(g["load-s"]) if g["load-s"] else None,
        "tokenization_seconds": float(g["tok-s"]) if g["tok-s"] else None,
        "cache_initialization_seconds": None,
        "prefill_seconds": (pre_ms / 1000.0) if pre_ms else None,
        "prefill_tokens_per_second": pre_tps,
        "first_decode_seconds": None,
        "cold_ttft_seconds": float(g["ttft-cold"]) if g["ttft-cold"] else None,
        "warm_ttft_seconds": float(g["ttft-warm"]) if g["ttft-warm"] else None,
        "decode_tokens_per_second": dec_tps,
        "accepted_tokens_per_second": acc_tps,
        "mtp_acceptance_rate": alpha,
        "peak_vram_bytes": int(g["vram"]) if g["vram"] else None,
        "metadata": {
            "command": g["command"],
            "run_type": g["run-type"] or ("prefill_decode" if pre_ms else "decode_only"),
            "kv_precision": g["kv"],
            "mtp_mode": g["mtp"],
            "chunk_size": int(g["chunk"]),
            "warm_state": g["warm"],
            "comparison_class": g["class"],
            "evidence": os.path.relpath(g["report"], REPO)
            if g["report"].startswith(REPO) else g["report"],
            "boundaries": {
                "model_load_included": g["load-s"] is not None,
                "tokenization_included": g["tok-s"] is not None,
                "prefill_start": "first chunk-list exec (chunk path) or "
                                 "first loop step (decode path)",
                "prefill_end": "last chunk-list fence (chunk path)",
            },
        },
    }
    errs = check_result(out)
    if errs:
        print("SCHEMA-FAIL (internal):")
        for e in errs:
            print(f"  {e}")
        return 1
    if not g["out"]:
        return fail("--out required")
    json.dump(out, open(g["out"], "w"), indent=1)
    print(f"BENCH-OK: {g['out']} prefill={pre_tps} decode={dec_tps} "
          f"alpha={alpha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
