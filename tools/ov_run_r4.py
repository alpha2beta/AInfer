import json, re, subprocess, sys

PROMPTS = {
    "math": "What is 84 * 3 / 2? Reply with just the number.",
    "rain": "Explain how rain forms, step by step, for a curious ten-year-old.",
}

def run(with_long=False, max_new=128):
    cmd = [sys.executable, "tools/ov_bench_qwen38.py", "--max-new", str(max_new)]
    if with_long:
        cmd.append("--long")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    blocks = []
    cur = None
    for line in out.splitlines():
        m = re.match(r"prompt_toks\(ours,proxy\)=(\d+)", line)
        if m:
            cur = {"prompt_tokens": int(m.group(1))}
        m = re.match(r"tokens=(\d+)/(\d+) ttft=([\d.]+)ms total=([\d.]+)ms decode=([\d.]+)t/s", line)
        if m and cur is not None:
            cur.update(generated_tokens=int(m.group(1)),
                       ttft_ms=float(m.group(3)),
                       total_ms=float(m.group(4)),
                       decode_tps=float(m.group(5)))
            blocks.append(cur)
            cur = None
    return blocks

def main():
    with_long = "--long" in sys.argv
    max_new = 32 if "--short" in sys.argv else 128
    tag = "r4_ov"
    for phase in ("warm", "meas"):
        blocks = run(with_long, max_new)
        out = []
        for b in blocks:
            gen = b["generated_tokens"]
            dec_ms = b["total_ms"] - b["ttft_ms"]
            out.append({
                "phase": phase,
                "prompt": [0] * b["prompt_tokens"],
                "generated": [0] * gen,
                "prefill_ms": b["ttft_ms"],
                "decode_ms": dec_ms,
                "decode_tok_per_sec": gen / (dec_ms / 1000.0),
                "gap_p50_ms": None,
                "source": "openvino_genai VLMPipeline text-only GPU.0",
            })
        json.dump(out, open(f"tools/bench/evidence/{tag}_{phase}.json", "w"), indent=1)
        print(phase, [(b["prompt_tokens"], b["generated_tokens"], round(b["ttft_ms"]), round(b["total_ms"]), round(b["decode_tps"], 2)) for b in blocks])

if __name__ == "__main__":
    main()
