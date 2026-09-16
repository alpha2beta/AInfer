#!/usr/bin/env python3
"""T8.6: grading (T8.5 scores) + margin-aware divergence analysis.

Reads runs/{int4,int8kv,llama}/*.json + runs/bf16/*.json (teacher-forced
BF16 top-K along the INT4 trajectory, where available) + corpus_t85.json.
Writes tools/quality/report_t85.json (category scores per config) and
tools/quality/report_t86.json (per-divergence margin/overlap/reconvergence).
Usage: analyze_t86.py [--grade-only]
"""
import json
import os
import re
import subprocess
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TK = None


def text_of(cfg, rep_path):
    global TK
    if TK is None:
        TK = tokenizer.load()
    d = json.load(open(rep_path))
    if cfg == "llama":
        return d.get("raw", "").strip()
    gen = d.get("generated", [])
    if isinstance(gen, str):
        return gen
    if not gen:
        return ""
    # Skip specials for grading (EOS/im_end leak in with skip=False).
    return tokenizer.decode(TK, gen, skip_special_tokens=True)


def gen_ids(rep_path):
    d = json.load(open(rep_path))
    gen = d.get("generated", [])
    return gen if isinstance(gen, list) else []


def last_number(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(nums[-1]) if nums else None


def grade(case, text):
    ck = case["check"]
    t = ck["type"]
    if t == "exact":
        return text.strip() == ck["expected"], text.strip()[:200]
    if t == "final_number":
        n = last_number(text)
        ok = n is not None and abs(n - ck["expected"]) <= ck.get("tol", 0.0)
        return ok, f"got={n} want={ck['expected']}"
    if t == "unit_test":
        code = extract_code(text)
        ok, detail = run_tests(code, ck["tests"])
        return ok, detail
    if t == "format":
        sents = [s for s in re.split(r"[.!?\u3002\uff01\uff1f]+", text.strip()) if s.strip()]
        ok = True
        if "max_sentences" in ck:
            ok = ok and len(sents) <= ck["max_sentences"]
        if "min_chars" in ck:
            ok = ok and len(text.strip()) >= ck["min_chars"]
        return ok, f"sents={len(sents)} chars={len(text.strip())}"
    if t == "stability":
        toks = text.split()
        grams = [" ".join(toks[i:i + 5]) for i in range(max(0, len(toks) - 4))]
        div = len(set(grams)) / max(1, len(grams))
        rep_ok = div > 0.3 or len(toks) < 40
        return rep_ok, f"tok5div={div:.2f} ntok={len(toks)}"
    return None, text[:200]  # record: no grade


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    return body.strip()


def run_tests(code, tests):
    prog = code + "\n" + tests + "\nprint('UNIT-OK')\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", prog],
                           capture_output=True, text=True, timeout=20)
        ok = r.returncode == 0 and "UNIT-OK" in r.stdout
        return ok, (r.stdout[-150:] + r.stderr[-150:]).strip()[:300]
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:200]


def main():
    corp = json.load(open(os.path.join(HERE, "corpus_t85.json")))["cases"]
    cfgs = ["int4", "int8kv", "llama"]
    scores = {c: {k: {"pass": 0, "total": 0, "graded": 0} for k in cfgs}
              for c in {x["category"] for x in corp}}
    per_case = {}
    for case in corp:
        if case["category"] == "retrieval" and "gen" in case["check"]:
            continue  # graded separately by check_retrieval
        row = {"id": case["id"], "cat": case["category"]}
        for cfg in cfgs:
            p = os.path.join(HERE, "runs", cfg, case["id"] + ".json")
            if not os.path.exists(p):
                row[cfg] = "missing"
                continue
            try:
                text = text_of(cfg, p)
            except Exception as e:  # noqa: BLE001
                row[cfg] = f"read-error: {e}"[:100]
                continue
            g, detail = grade(case, text)
            row[cfg] = {"grade": g, "detail": detail,
                        "text": text[:500]}
            if g is not None:
                s = scores[case["category"]][cfg]
                s["total"] += 1
                s["pass"] += 1 if g else 0
        per_case[case["id"]] = row
    r85 = {"task": "T8.5-quality-benchmark", "scores": scores,
           "cases": per_case}
    json.dump(r85, open(os.path.join(HERE, "report_t85.json"), "w"), indent=1,
              ensure_ascii=False)
    # T8.6 divergence: BF16-margin part filled by bf16_replay outputs.
    divs = []
    for cid, row in per_case.items():
        if not isinstance(row.get("int4"), dict):
            continue
        bfp = os.path.join(HERE, "runs", "bf16", cid + ".json")
        bf = json.load(open(bfp)) if os.path.exists(bfp) else None
        i4ids = gen_ids(os.path.join(HERE, "runs", "int4", cid + ".json"))
        entry = {"id": cid, "int4_grade": row["int4"]["grade"],
                 "int8kv_grade": row.get("int8kv", {}).get("grade")
                 if isinstance(row.get("int8kv"), dict) else None,
                 "bf16": bf is not None}
        if bf:
            entry.update(summarize_divergence(bf, i4ids))
        divs.append(entry)
    r86 = {"task": "T8.6-margin-divergence", "divergences": divs}
    json.dump(r86, open(os.path.join(HERE, "report_t86.json"), "w"), indent=1)
    n = len(per_case)
    print(f"graded {n} cases -> report_t85.json + report_t86.json")


def summarize_divergence(bf, i4ids):
    """bf: {positions: [{pos, int4_tok, bf_top: [...], bf_topv: [...]}]}."""
    first, overlaps, margins = None, [], []
    reconv = True
    for rec in bf["positions"]:
        top, topv = rec["bf_top"], rec["bf_topv"]
        margins.append(round(topv[0] - topv[1], 4) if len(topv) > 1 else None)
        overlaps.append(rec["int4_tok"] in top)
        if rec["int4_tok"] != top[0] and first is None:
            first = {"pos": rec["pos"], "int4_tok": rec["int4_tok"],
                     "bf_top1": top[0],
                     "margin": round(topv[0] - topv[1], 4) if len(topv) > 1 else None,
                     "int4_in_bf_top": rec["int4_tok"] in top,
                     "overlap5": len(set(top))}
    if first is not None:
        tail = [r for r in bf["positions"] if r["pos"] > first["pos"]]
        reconv = any(r["int4_tok"] == r["bf_top"][0] for r in tail[-5:]) if tail else False
    return {"first_divergence": first,
            "bf_margin_mean": round(sum(m for m in margins if m is not None)
                                   / max(1, len([m for m in margins if m is not None])), 4),
            "int4_in_bf_top5_rate": round(sum(overlaps) / max(1, len(overlaps)), 3),
            "reconvergence": reconv}


if __name__ == "__main__":
    main()
