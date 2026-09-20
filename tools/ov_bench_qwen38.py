#!/usr/bin/env python3
"""OpenVINO GenAI reference benchmark: Qwen3.8-27B INT4 on B60 (GPU.0).

Counterpart to the AInfer e2e numbers (decode_l0, greedy). Measures TTFT,
prefill tok/s, decode tok/s via a timestamping streamer. Greedy
(do_sample=False). Optional --draft enables the bundled MTP draft model
(openvino_mtp_model.xml) for speculative decoding.
Usage: ov_bench_qwen38.py [--device GPU.0] [--draft] [--max-new N]
"""
import os
import sys
import time

MODEL = "/mnt/usb/ov_qwen38_int4"


class StampStreamer:
    def __init__(self, t_start=None):
        self.t0 = t_start
        self.stamps = []
        self.text = []

    def __call__(self, s):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        self.stamps.append(now - self.t0)
        self.text.append(s)
        return False  # keep generating

    def report(self, prompt, max_new):
        import statistics
        n = len(self.stamps)
        ttft = self.stamps[0] if n else float("nan")
        total = self.stamps[-1] if n else float("nan")
        gaps = [b - a for a, b in zip(self.stamps, self.stamps[1:])]
        dec = (n - 1) / (total - ttft) if n > 1 and total > ttft else float("nan")
        print(f"prompt={prompt!r:.60}")
        print(f"tokens={n}/{max_new} ttft={ttft*1000:.1f}ms "
              f"total={total*1000:.0f}ms decode={dec:.2f}t/s "
              f"gap_p50={statistics.median(gaps)*1000:.1f}ms" if gaps else "")
        print("text:", "".join(self.text)[:160].replace("\n", "\\n"))
        return {"tokens": n, "ttft_ms": ttft * 1000, "total_ms": total * 1000,
                "decode_tps": dec}


def main():
    import openvino_genai as ov_genai
    device = "GPU.0"
    use_draft = False
    max_new = 128
    for i, a in enumerate(sys.argv[1:]):
        if a == "--device":
            device = sys.argv[1:][i + 1]
        if a == "--draft":
            use_draft = True
        if a == "--max-new":
            max_new = int(sys.argv[1:][i + 1])
    print(f"openvino_genai {ov_genai.__version__} device={device} "
          f"draft={use_draft}", flush=True)
    t_load0 = time.perf_counter()
    # VLM dir (Qwen3.8): VLMPipeline handles text-only via ChatHistory.
    # --draft: MTP speculative decoding via bundled openvino_mtp_model
    # (copied to /mnt/usb/ov_draft_mtp/openvino_model.*).
    if use_draft:
        dm = ov_genai.draft_model("/mnt/usb/ov_draft_mtp", device)
        pipe = ov_genai.VLMPipeline(MODEL, device, draft_model=dm)
    else:
        pipe = ov_genai.VLMPipeline(MODEL, device)
    print(f"load={(time.perf_counter()-t_load0):.1f}s", flush=True)
    cfg = ov_genai.GenerationConfig()
    cfg.do_sample = False
    cfg.max_new_tokens = max_new
    # Bypass the bundled chat template (expects unset jinja kwargs):
    # render with the pinned AInfer template (parity with decode_l0 runs)
    # and feed pre-tokenized inputs.
    sys.path.insert(0, os.path.join("/mnt/usb/AInfer", "tools", "tokenizer"))
    sys.path.insert(0, os.path.join("/mnt/usb/AInfer", "tools"))
    import tok as tokenizer
    otok = tokenizer.load()
    gtok = ov_genai.Tokenizer(MODEL)
    prompts = [
        "What is 84 * 3 / 2? Reply with just the number.",
        "Explain how rain forms, step by step, for a curious ten-year-old.",
    ]
    if "--long" in sys.argv:
        fill = ("The northern lights observatory tracks auroral activity "
                "every night. Researchers log solar wind speed, cloud cover, "
                "and instrument status. Supplies arrive by snowmobile twice "
                "a month during winter. ")
        prompts += ["Describe this research station in detail: " + fill * 6,
                    "Summarize the following station log: " + fill * 24]
    for p in prompts:
        h = ov_genai.ChatHistory()
        h.append({"role": "user", "content": p})
        rendered = tokenizer.render_chat(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            enable_thinking=False)
        print(f"prompt_toks(ours,proxy)={len(tokenizer.encode(otok, rendered))}",
              flush=True)
        t0 = time.perf_counter()
        st = StampStreamer(t0)
        cfg2 = ov_genai.GenerationConfig()
        cfg2.do_sample = False
        cfg2.max_new_tokens = max_new
        if use_draft:
            cfg2.num_assistant_tokens = 5
        pipe.generate(history=h, generation_config=cfg2, streamer=st)
        wall = time.perf_counter() - t0
        r = st.report(p, max_new)
        print(f"wall={wall*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
