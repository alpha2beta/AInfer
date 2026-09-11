#!/usr/bin/env python3
"""AInfer CLI generation loop (T4.4).

Prompt -> our tokenizer (chat template + exact encode) -> backend ->
streamed text + timings + stopping. Backend is pluggable; the only backend
today is the llama.cpp reference (`llama`) used to prove the loop. The native
runtime backend hooks up once T4.2 lands (same interface: generate(ids)).

Usage:
    ainfer_cli.py --message "What is 84 * 3 / 2?" [--max-tokens N] [--temp T]
                  [--seed S] [--timings] [--report PATH]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tokenizer"))
import tok as tokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CLI = os.path.expanduser("~/llama.arc/build/bin/llama-cli")
LLAMA_BUILD = "ddd4ec142 (10217), IntelLLVM 2026.1.1, SYCL (matches T0.5 baseline)"
GGUF = "/mnt/usb/Test/Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf"
ONEAPI_SETUP = "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1"

EOS_MARKERS = ("<|im_end|>", "<|endoftext|>")


def build_prompt(message, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": message})
    return tokenizer.render_chat(msgs, add_generation_prompt=True,
                                 enable_thinking=False)


def generate_llama(user_message, max_tokens, temp, seed, stream_out=True):
    """Run llama-cli reference backend; yield text chunks; return stats.

    The reference backend owns templating (GGUF-bundled Qwen template): the
    user message goes via stdin as a single chat turn. OUR rendered prompt
    (T4.3) is used for token accounting and parity, and is what the native
    T4.2 backend will consume directly.
    """
    import shlex
    inner = " ".join(shlex.quote(x) for x in
                     [LLAMA_CLI, "-m", GGUF, "-ngl", "99", "--no-warmup",
                      "--no-display-prompt", "--simple-io", "--single-turn",
                      "-n", str(max_tokens), "--temp", str(temp),
                      "-s", str(seed)])
    t0 = time.time()
    p = subprocess.Popen(["bash", "-c", f"{ONEAPI_SETUP}; exec {inner}"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        out_text, err = p.communicate(input=user_message + "\n", timeout=1500)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        out_text, err = p.communicate()
        rc = 124
    t1 = time.time()
    first_at = None  # chunk-level TTFT unavailable in single-turn batch mode
    text = out_text or ""
    # strip chat-UI chrome: banner, '> echo', thinking block, perf line
    lines = text.splitlines()
    body = []
    in_think = False
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
    # stopping: cut at first EOS marker (reference backend may echo it)
    stop_hit = None
    for m in EOS_MARKERS:
        i = text.find(m)
        if i >= 0:
            stop_hit = m
            text = text[:i]
            break
    return {"text": text, "rc": rc, "stop": stop_hit,
            "ttft_s": None, "total_s": t1 - t0,
            "stderr_tail": (err or "")[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="What is 84 * 3 / 2?")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timings", action="store_true")
    ap.add_argument("--report", default=None)
    ap.add_argument("--backend", default="llama", choices=["llama"])
    a = ap.parse_args()

    tok = tokenizer.load()
    rendered = build_prompt(a.message, a.system)
    ids = tokenizer.encode(tok, rendered)
    if a.timings:
        print(f"[prompt tokens: {len(ids)}]", flush=True)
    if a.backend == "llama":
        st = generate_llama(a.message, a.max_tokens, a.temp, a.seed)
    if a.timings:
        n_out = len(tokenizer.encode(tok, st["text"])) if st["text"] else 0
        dt = st["total_s"]
        ttft = f"{st['ttft_s']:.2f}s" if st["ttft_s"] else "n/a(batch)"
        print(f"{st['text']}\n[TTFT: {ttft} total: {st['total_s']:.2f}s "
              f"out_tokens~{n_out} tok/s~{n_out / dt if dt > 0 else 0:.1f} "
              f"stop={st['stop']} rc={st['rc']}]")
    if a.report:
        json.dump({"prompt_tokens": len(ids), "max_tokens": a.max_tokens,
                   "temp": a.temp, "seed": a.seed,
                   "text": st["text"], "stop": st["stop"], "rc": st["rc"],
                   "ttft_s": st["ttft_s"], "total_s": st["total_s"]},
                  open(a.report, "w"), indent=1)
    return 0 if st["rc"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
