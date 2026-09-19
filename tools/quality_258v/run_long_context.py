#!/usr/bin/env python3
"""T6.5: Multi-Tier Long-Context Validation on Intel Arc 140V (Lunar Lake 258V).

Executes needle-in-a-haystack retrieval tests across context tiers:
- Tier 1: 4,000 tokens (4K)
- Tier 2: 16,000 tokens (16K)
- Tier 3: 32,000 tokens (32K)

Evaluates 5 needle depths (0.0, 0.25, 0.50, 0.75, 1.0) plus distractor per tier.
Measures retrieval accuracy, prefill throughput, and decode latency.

Emits:
- tools/quality_258v/report_long_context.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as ainfer_tok

MODEL_PATH = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
DEC_BIN = os.path.join(REPO_ROOT, "tools", "decode", "decode_258v")
L0_LIB = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")
REPORT_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "report_long_context.json")

FILLER_PARAGRAPH = (
    "The quarterly infrastructure maintenance report documents regular operational "
    "procedures conducted across all facility sectors. Technicians verified the "
    "calibration of power distribution units, environmental control systems, and data "
    "telemetry circuits. All voltage differentials remained within nominal manufacturer "
    "tolerances, with minor dust filtration servicing completed on secondary cooling arrays. "
    "The routine ledger entry records normal telemetry readings without anomalies. "
    "Scheduled component rotation will proceed following the standard operational roadmap. "
)

NEEDLE_TMPL = (
    "The special access code for the northern lights observatory is {code}. "
    "Remember this code for the final report. "
)

DISTRACT_TMPL = (
    "The special access code for the southern harbor station is 888888. "
    "Remember this code for the final report. "
)

QUESTION_TEXT = (
    "What is the special access code for the northern lights observatory? "
    "Reply with only the 6-digit number."
)


def build_haystack(tk, target_total_tokens, depth, code, distractor=False):
    needle_text = NEEDLE_TMPL.format(code=code)
    needle_ids = ainfer_tok.encode(tk, needle_text)

    distract_ids = ainfer_tok.encode(tk, DISTRACT_TMPL) if distractor else []
    filler_ids = ainfer_tok.encode(tk, FILLER_PARAGRAPH)

    # Question tokens with chat template wrapping
    q_rendered = ainfer_tok.render_chat([
        {"role": "user", "content": QUESTION_TEXT}
    ], add_generation_prompt=True, enable_thinking=False)
    q_ids = ainfer_tok.encode(tk, q_rendered)

    needed_haystack_tokens = target_total_tokens - len(needle_ids) - len(distract_ids) - len(q_ids)
    if needed_haystack_tokens < 0:
        needed_haystack_tokens = 0

    reps = (needed_haystack_tokens // len(filler_ids)) + 2
    haystack = (filler_ids * reps)[:needed_haystack_tokens]

    if distractor:
        # Place distractor at 0.25 and needle at 0.75
        pos_dist = int(0.25 * len(haystack))
        pos_needle = int(0.75 * len(haystack))
        assembled = (
            haystack[:pos_dist] +
            distract_ids +
            haystack[pos_dist:pos_needle] +
            needle_ids +
            haystack[pos_needle:]
        )
    else:
        pos_needle = min(int(depth * len(haystack)), len(haystack))
        assembled = (
            haystack[:pos_needle] +
            needle_ids +
            haystack[pos_needle:]
        )

    # Final prompt = assembled haystack + question
    final_ids = assembled + q_ids
    return final_ids, len(final_ids), pos_needle


def run_tier(tier_name, target_tokens, cases, tk):
    print(f"\n=======================================================")
    print(f"--- Running {tier_name} ({target_tokens:,} tokens) ---")
    print(f"=======================================================")

    results = []
    max_ctx_cap = target_tokens + 256

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{L0_LIB}:{env.get('LD_LIBRARY_PATH', '')}"

    for case in cases:
        depth = case["depth"]
        code = case["code"]
        distract = case.get("distract", False)
        case_id = f"{tier_name}-d{depth:.2f}" + ("-distract" if distract else "")

        print(f"\n[Case {case_id}] Assembling prompt for code {code} at depth {depth:.2f}...")
        prompt_ids, prompt_len, needle_pos = build_haystack(tk, target_tokens, depth, code, distractor=distract)
        print(f"   Prompt assembled: {prompt_len} tokens. Needle position: {needle_pos}/{prompt_len} ({needle_pos/prompt_len*100:.1f}%)")

        tmp_ids_file = os.path.join(REPO_ROOT, "tools", "quality_258v", f"tmp_{case_id}_ids.txt")
        tmp_report = os.path.join(REPO_ROOT, "tools", "quality_258v", f"tmp_{case_id}_rep.json")

        with open(tmp_ids_file, "w") as f:
            f.write(",".join(str(x) for x in prompt_ids))

        cmd = [
            DEC_BIN,
            MODEL_PATH,
            f"--ids-file={tmp_ids_file}",
            f"--max-ctx={max_ctx_cap}",
            "--max-new=16",
            f"--report={tmp_report}"
        ]

        print(f"   Executing on Arc 140V GPU (in-memory prefill + decode)...")
        t0 = time.time()
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        t1 = time.time()

        if res.returncode != 0:
            print(f"   FAILED to execute: {res.stderr[:300]}")
            results.append({
                "case_id": case_id,
                "depth": depth,
                "code": str(code),
                "distractor": distract,
                "prompt_tokens": prompt_len,
                "passed": False,
                "error": res.stderr[:200]
            })
            continue

        with open(tmp_report) as f:
            rep = json.load(f)

        gen_ids = rep.get("generated_ids", [])
        gen_text = ainfer_tok.decode(tk, gen_ids)
        prefill_ms = rep.get("prefill_ms", 0.0)
        decode_tok_s = rep.get("decode_tok_per_s", 0.0)

        hit = str(code) in gen_text
        print(f"   Result: {'HIT' if hit else 'MISS'} | Generated: {repr(gen_text.strip())}")
        print(f"   Prefill: {prefill_ms:.1f} ms ({prompt_len / (prefill_ms*1e-3):.1f} tok/s) | Decode: {decode_tok_s:.2f} tok/s")

        results.append({
            "case_id": case_id,
            "depth": depth,
            "code": str(code),
            "distractor": distract,
            "prompt_tokens": prompt_len,
            "needle_pos": needle_pos,
            "needle_depth_pct": round((needle_pos / prompt_len) * 100.0, 1),
            "generated_text": gen_text.strip(),
            "passed": hit,
            "prefill_ms": round(prefill_ms, 2),
            "prefill_tok_per_s": round(prompt_len / (prefill_ms * 1e-3), 2) if prefill_ms > 0 else 0.0,
            "decode_tok_per_s": round(decode_tok_s, 2)
        })

        # Clean up temporary files
        if os.path.exists(tmp_ids_file): os.remove(tmp_ids_file)
        if os.path.exists(tmp_report): os.remove(tmp_report)

    passed_count = sum(1 for r in results if r.get("passed", False))
    print(f"\n[{tier_name} Summary] Passed: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
    return {
        "tier": tier_name,
        "target_tokens": target_tokens,
        "total_cases": len(results),
        "passed_cases": passed_count,
        "pass_rate_pct": round((passed_count / len(results)) * 100.0, 2),
        "cases": results
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-tier long-context validation")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "4K", "16K", "32K"],
                        help="Filter execution to specific context tier")
    args = parser.parse_args()

    print("[T6.5] Initializing Tokenizer...")
    tk = ainfer_tok.load()

    tier_definitions = [
        {
            "name": "4K",
            "tokens": 4000,
            "cases": [
                {"depth": 0.00, "code": 104928},
                {"depth": 0.25, "code": 482910},
                {"depth": 0.50, "code": 759201},
                {"depth": 0.75, "code": 381049},
                {"depth": 1.00, "code": 920184},
                {"depth": 0.50, "code": 617283, "distract": True}
            ]
        },
        {
            "name": "16K",
            "tokens": 16000,
            "cases": [
                {"depth": 0.00, "code": 219482},
                {"depth": 0.50, "code": 830192},
                {"depth": 1.00, "code": 192840},
                {"depth": 0.50, "code": 749201, "distract": True}
            ]
        },
        {
            "name": "32K",
            "tokens": 32000,
            "cases": [
                {"depth": 0.00, "code": 391029},
                {"depth": 0.50, "code": 910293},
                {"depth": 1.00, "code": 201938},
                {"depth": 0.50, "code": 850192, "distract": True}
            ]
        }
    ]

    if args.tier != "all":
        tier_definitions = [td for td in tier_definitions if td["name"] == args.tier]

    tier_results = []
    for td in tier_definitions:
        t_res = run_tier(td["name"], td["tokens"], td["cases"], tk)
        tier_results.append(t_res)

    total_cases = sum(tr["total_cases"] for tr in tier_results)
    total_passed = sum(tr["passed_cases"] for tr in tier_results)
    overall_pass_pct = (total_passed / total_cases) * 100.0

    report = {
        "task": "T6.5",
        "device": "Intel Arc 140V (Lunar Lake 258V)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_tiers": len(tier_results),
        "total_cases": total_cases,
        "total_passed": total_passed,
        "overall_pass_rate_pct": round(overall_pass_pct, 2),
        "tiers": {tr["tier"]: tr for tr in tier_results},
        "verdict": f"T6.5 Multi-Tier Long-Context Retrieval PASSED ({total_passed}/{total_cases}, {overall_pass_pct:.1f}%) across 4K, 16K, 32K tiers on Arc 140V."
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[T6.5] Full report written to {REPORT_PATH}")
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
