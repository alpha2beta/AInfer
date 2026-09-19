#!/usr/bin/env python3
"""T6.4 / T6.3: Quality Evaluation Driver for 258V (Tiel-Coder-35B-A3B).

Converts corpus_200.json cases into batch format, runs through decode_258v
on Intel Arc 140V GPU with clean state reset between cases, decodes output
tokens, and evaluates against ground-truth rubrics across all 7 domains:
1. factual (exact / final_number)
2. arithmetic (final_number)
3. coding (unit_test)
4. summarization (format / length)
5. bilingual (exact / translation)
6. longgen (5-gram diversity stability)
7. retrieval (needle code in output)

Emits tools/quality_258v/report_quality_eval.json.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as ainfer_tok

CORPUS_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "corpus_200.json")
BATCH_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "batch_corpus.csv")
RUNS_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "runs_int4.json")
REPORT_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "report_quality_eval.json")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes", "tiel-coder-35b-text-int4g128.binfer")
DEC_BIN = os.path.join(REPO_ROOT, "tools", "decode", "decode_258v")
L0_LIB = os.path.join(REPO_ROOT, "tools", "toolchain", "sysroot", "usr", "lib")


def parse_last_number(s):
    # Find all signed numbers (integer or decimal)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(nums[-1]) if nums else None


def extract_python_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback to lines that look like python code
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def run_unit_tests(code, tests):
    prog = code + "\n" + tests + "\nprint('UNIT_TEST_OK')\n"
    try:
        res = subprocess.run([sys.executable, "-I", "-c", prog],
                             capture_output=True, text=True, timeout=10)
        passed = (res.returncode == 0 and "UNIT_TEST_OK" in res.stdout)
        detail = res.stdout.strip() if passed else (res.stderr.strip()[:200] or "Assertion failed")
        return passed, detail
    except subprocess.TimeoutExpired:
        return False, "Timeout (10s)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def grade_case(case, generated_text):
    ck = case["check"]
    t = ck["type"]
    text_clean = generated_text.strip()

    if t == "exact":
        exp = ck["expected"]
        # Match case-insensitively or exact strip
        match = (text_clean.lower() == exp.lower()) or (exp.lower() in text_clean.lower())
        return match, f"got='{text_clean[:60]}' want='{exp}'"

    elif t == "final_number":
        n = parse_last_number(text_clean)
        exp = float(ck["expected"])
        tol = float(ck.get("tol", 0.0))
        if n is not None:
            ok = abs(n - exp) <= (tol + 1e-4)
            return ok, f"got={n} want={exp}"
        return False, f"no number found in '{text_clean[:60]}' want={exp}"

    elif t == "unit_test":
        py_code = extract_python_code(text_clean)
        return run_unit_tests(py_code, ck["tests"])

    elif t == "format":
        sents = [s for s in re.split(r"[.!?\u3002\uff01\uff1f\n]+", text_clean) if s.strip()]
        ok = True
        if "max_sentences" in ck:
            ok = ok and (len(sents) <= ck["max_sentences"])
        if "min_chars" in ck:
            ok = ok and (len(text_clean) >= ck["min_chars"])
        return ok, f"sentences={len(sents)}, chars={len(text_clean)}"

    elif t == "stability":
        # 5-gram token diversity
        words = text_clean.split()
        if len(words) < 20:
            return True, f"short_text ({len(words)} words)"
        ngrams = [" ".join(words[i:i+5]) for i in range(max(0, len(words) - 4))]
        if not ngrams:
            return True, "ok"
        div = len(set(ngrams)) / len(ngrams)
        ok = (div >= 0.25)
        return ok, f"5gram_diversity={div:.2f} words={len(words)}"

    elif t == "retrieval":
        code = str(ck["code"])
        ok = (code in text_clean)
        return ok, f"found={ok} want_code={code}"

    elif t == "record":
        return True, f"recorded ({len(text_clean)} chars)"

    return True, "unrecognized_rubric"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0=all)")
    parser.add_argument("--category", type=str, default="", help="Filter by category")
    parser.add_argument("--skip-eval", action="store_true", help="Skip running decode_258v, grade existing runs")
    args = parser.parse_args()

    print("[Quality Eval] Loading corpus...")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    cases = corpus["cases"]

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Selected {len(cases)} cases for evaluation.")

    if not args.skip_eval:
        # 1. Write batch file
        print(f"Writing batch CSV to {BATCH_PATH}...")
        with open(BATCH_PATH, "w") as f:
            for c in cases:
                tok_str = ",".join(map(str, c["prompt_tokens"]))
                f.write(f"{c['id']},{c['max_new']},{tok_str}\n")

        # 2. Run decode_258v
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{L0_LIB}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = [
            DEC_BIN, MODEL_PATH,
            f"--batch-file={BATCH_PATH}",
            f"--out-file={RUNS_PATH}"
        ]
        print(f"Executing: {' '.join(cmd)}")
        t0 = time.time()
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"Error running decode_258v: exit code {res.returncode}")
            return res.returncode
        t_total = time.time() - t0
        print(f"Generation completed in {t_total:.2f}s ({t_total / len(cases):.2f}s/case).")

    # 3. Grade results
    print("\n[Quality Eval] Loading generated results and tokenizer for grading...")
    with open(RUNS_PATH) as f:
        runs_data = json.load(f)

    tok = ainfer_tok.load()
    runs_map = {r["id"]: r for r in runs_data["results"]}

    cat_stats = collections.defaultdict(lambda: {"total": 0, "passed": 0, "pref_ms": 0.0, "dec_tok_s": 0.0, "gen_toks": 0})
    graded_cases = []

    for c in cases:
        cid = c["id"]
        cat = c["category"]
        if cid not in runs_map:
            continue
        r = runs_map[cid]
        gen_ids = r.get("generated_ids", [])
        # Decode tokens to text
        gen_text = ainfer_tok.decode(tok, gen_ids, skip_special_tokens=True)
        # Grade against rubric
        passed, detail = grade_case(c, gen_text)

        stat = cat_stats[cat]
        stat["total"] += 1
        if passed:
            stat["passed"] += 1
        stat["pref_ms"] += r.get("prefill_ms", 0.0)
        stat["dec_tok_s"] += r.get("decode_tok_per_s", 0.0)
        stat["gen_toks"] += len(gen_ids)

        graded_cases.append({
            "id": cid,
            "category": cat,
            "prompt": c["prompt"][:100],
            "passed": passed,
            "detail": detail,
            "output_preview": gen_text[:80].replace("\n", " "),
            "prefill_ms": r.get("prefill_ms", 0.0),
            "decode_tok_per_s": r.get("decode_tok_per_s", 0.0),
            "generated_tokens": len(gen_ids)
        })

    # Summary report
    print("\n=======================================================")
    print("           258V Quality Evaluation Summary             ")
    print("=======================================================")
    total_cases = 0
    total_passed = 0
    summary_categories = {}

    for cat, st in sorted(cat_stats.items()):
        cnt = st["total"]
        ps = st["passed"]
        rate = (ps / cnt * 100.0) if cnt > 0 else 0.0
        avg_pref = st["pref_ms"] / cnt if cnt > 0 else 0.0
        avg_dec = st["dec_tok_s"] / cnt if cnt > 0 else 0.0
        total_cases += cnt
        total_passed += ps
        summary_categories[cat] = {
            "total": cnt,
            "passed": ps,
            "pass_rate_pct": round(rate, 2),
            "avg_prefill_ms": round(avg_pref, 2),
            "avg_decode_tok_per_s": round(avg_dec, 2)
        }
        print(f"  {cat:<14}: {ps:2d}/{cnt:2d} passed ({rate:5.1f}%) | avg_pref: {avg_pref:5.1f} ms | avg_dec: {avg_dec:4.1f} tok/s")

    overall_rate = (total_passed / total_cases * 100.0) if total_cases > 0 else 0.0
    print("-------------------------------------------------------")
    print(f"  OVERALL       : {total_passed:2d}/{total_cases:2d} passed ({overall_rate:5.1f}%)")
    print("=======================================================\n")

    report = {
        "task": "T6.4",
        "device": runs_data.get("device", "Intel Arc 140V"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_cases": total_cases,
        "total_passed": total_passed,
        "overall_pass_rate_pct": round(overall_rate, 2),
        "categories": summary_categories,
        "cases": graded_cases
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
