#!/usr/bin/env python3
"""T8.5 runner: greedy generations for corpus_t85 short cases (all but
retrieval/longgen-specials) across configs int4 | int8kv | llama.

int4/int8kv: decode_l0 via --ids-file (chat rendered thinking-off by tok),
  report JSON per case. int8kv sets AINFER_KV8=1.
llama: llama-cli -m <Q4_K_XL> with the same rendered text, --temp 0 -s 0.
Resumable: skips cases with an existing report. Run in background with nohup.
Usage: run_t85.py [--cfg int4|int8kv|llama] [--only ID] [--limit N]
"""
import json
import os
import subprocess
import sys
import time

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer

MODEL = os.path.join(REPO, "models", "Qwen3.8-27B",
                     "qwen3.8-27b-text-int4g128.binfer")
SPVDIR = os.path.join(REPO, "build-b60", "tools", "cmdlist")
DEC = os.path.join(REPO, "build-b60", "tools", "decode", "decode_l0")
LLAMA = "/home/yanchun/llama.arc/build/bin/llama-cli"
LLAMA_MODEL = "/mnt/usb/Test/Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))


def decode_report(path):
    d = json.load(open(path))
    gen = d.get("generated", d.get("gen", []))
    if isinstance(gen, str):
        return gen
    tk = tokenizer.load()
    return tokenizer.decode(tk, gen)


def run_decode(tk, ids, max_new, report, env_extra):
    with open("/tmp/t85_ids.txt", "w") as f:
        f.write(",".join(map(str, ids)))
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [DEC, MODEL, "1", "1", SPVDIR, report,
           "--ids-file=/tmp/t85_ids.txt", f"--max-new={max_new}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                       env=env)
    if r.returncode != 0:
        raise RuntimeError(f"decode rc={r.returncode}: {r.stderr[-500:]}")


EOS_MARKERS = ("<|im_end|>", "<|endoftext|>")


def strip_llama_chrome(out_text):
    # Same rules as tools/cli/ainfer_cli.py (banner, echo, thinking, perf).
    lines, body, in_think = out_text.splitlines(), [], False
    for ln in lines:
        s = ln.strip()
        if s == "[Start thinking]":
            in_think = True
            continue
        if s == "[End thinking]":
            in_think = False
            continue
        if in_think or not s:
            continue
        if s.startswith("> "):
            # Single-turn generation echo carries the ANSWER on this line.
            body.append(s[2:])
            continue
        if s.startswith(">") or s.startswith("[ Prompt:") or s in (
                "Exiting...",) or set(s) <= {"\u2584", "\u2588", "\u2580",
                                             " ", "#", "X", "K", "M", "N",
                                             "i", "x", "D", "P", "e", "8",
                                             "E", "B", "C", ".", "/"}:
            continue
        if "available commands:" in s or s.startswith("/"):
            continue
        if s.startswith(("build      :", "model      :", "ftype      :",
                         "modalities :", "Loading model")):
            continue
        body.append(ln)
    text = "\n".join(body).strip()
    for m in EOS_MARKERS:
        i = text.find(m)
        if i >= 0:
            text = text[:i].strip()
            break
    return text


def run_llama(user_message, max_new, nctx, report):
    # Proven pattern (tools/cli/ainfer_cli.py, fwd_s3.py): --single-turn,
    # RAW user message via stdin; the GGUF-bundled Qwen template owns
    # chat formatting (asymmetry vs decode_l0's tok-rendered ids, documented
    # in report_t85.json).
    inner = (f"{LLAMA} -m {LLAMA_MODEL} -ngl 99 --no-warmup "
             f"--no-display-prompt --simple-io --single-turn -n {max_new} "
             f"-c {nctx} --temp 0.0 -s 0 -t 8 --reasoning off")
    # Harness constraint: single-turn stdin consumes the FIRST LINE only;
    # newlines are flattened to spaces (content identical, documented).
    flat = " ".join(user_message.splitlines())
    p = subprocess.Popen(
        ["bash", "-c",
         "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec "
         + inner],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    try:
        out_text, err = p.communicate(input=flat + "\n", timeout=1500)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        out_text, err = p.communicate()
        rc = 124
    if rc != 0:
        raise RuntimeError(f"llama rc={rc}: {err[-500:]}")
    json.dump({"gen": strip_llama_chrome(out_text or ""),
               "stderr_tail": (err or "")[-300:]}, open(report, "w"))


def main():
    cfg = "int4"
    only, limit, skip = None, None, 0
    for i, a in enumerate(sys.argv[1:]):
        if a == "--cfg":
            cfg = sys.argv[1:][i + 1]
        if a == "--only":
            only = sys.argv[1:][i + 1]
        if a == "--limit":
            limit = int(sys.argv[1:][i + 1])
        if a == "--skip":
            skip = int(sys.argv[1:][i + 1])
    corp = json.load(open(os.path.join(HERE, "corpus_t85.json")))["cases"]
    todo = [c for c in corp
            if c["category"] != "retrieval"
            and not (only and c["id"] != only)]
    todo = todo[skip:]
    if limit:
        todo = todo[:limit]
    outdir = os.path.join(HERE, "runs", cfg)
    os.makedirs(outdir, exist_ok=True)
    tk = tokenizer.load()
    done, fail = 0, 0
    t0 = time.time()
    for c in todo:
        rep = os.path.join(outdir, c["id"] + ".json")
        if os.path.exists(rep):
            done += 1
            continue
        try:
            rendered = tokenizer.render_chat(
                [{"role": "user", "content": c["prompt"]}],
                add_generation_prompt=True, enable_thinking=False)
            if cfg == "llama":
                run_llama(c["prompt"], c["max_new"],
                          len(tokenizer.encode(tk, rendered)) + c["max_new"] + 16,
                          rep)
            else:
                ids = tokenizer.encode(tk, rendered)
                env_extra = {"AINFER_KV8": "1"} if cfg == "int8kv" else {}
                run_decode(tk, ids, c["max_new"], rep, env_extra)
            done += 1
        except Exception as e:  # noqa: BLE001 - record and continue
            fail += 1
            json.dump({"error": str(e)[:300]}, open(rep, "w"))
            print(f"FAIL {c['id']}: {e}", flush=True)
        if (done + fail) % 20 == 0:
            print(f"{cfg}: {done} ok {fail} fail "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"{cfg} DONE: {done} ok {fail} fail ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
