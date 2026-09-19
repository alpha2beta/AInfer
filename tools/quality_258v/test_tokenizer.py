#!/usr/bin/env python3
"""T6.2: Tokenizer regression and chat-template validation suite for 258V.

Tests tokenization, round-trip encoding/decoding, special token preservation,
and chat template formatting across English, Chinese, code, and thinking tags.
Validates 100% parity against Hugging Face transformers AutoTokenizer.
Emits tools/quality_258v/report_tokenizer.json.
"""

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "tokenizer"))
import tok as ainfer_tok

from transformers import AutoTokenizer

MODEL_DIR = os.path.join(REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes")
REPORT_PATH = os.path.join(REPO_ROOT, "tools", "quality_258v", "report_tokenizer.json")


def main():
    print("[T6.2] Loading HF AutoTokenizer and AInfer Tokenizer...")
    t0 = time.time()
    hf_tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    my_tok = ainfer_tok.load()
    t_load = time.time() - t0
    print(f"Loaded in {t_load:.2f}s. Vocab size: HF={len(hf_tok)}, AInfer={my_tok.get_vocab_size(with_added_tokens=True)}")

    results = {
        "task": "T6.2",
        "model": "Tiel-Coder-35B-A3B-Genesis-Hermes",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocab_size": len(hf_tok),
        "tests": [],
        "all_passed": True,
    }

    # 1. Raw Text Token Parity Tests
    test_texts = [
        # English
        "The quick brown fox jumps over the lazy dog.",
        "AInfer is an optimized batch-one inference engine for Intel Arc 140V Xe2 GPU.",
        # Chinese
        "北京是中国的首都，也是一座具有三千多年历史的古都。",
        "人工智能和混合专家模型（MoE）正在改变计算硬件的未来。",
        # Code - Python
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        # Code - C++ / OpenCL
        "#include <level_zero/ze_api.h>\nvoid kernel_gemv(__global half* W, __global float* X) { /* xe2 */ }",
        # Structured Markdown & JSON
        "```json\n{\"model\": \"Tiel-Coder-35B-A3B\", \"experts\": 256, \"active\": 8}\n```",
        # Whitespace and indentation stress
        "    \t\n  \n\t\t\tline with mixed spaces and tabs\n\n\n",
        # Numbers, math, symbols
        "Calculate: 3.1415926535 * (42 / 7) + 1.2e-4 = ?",
        # Special characters & emoji
        "🚀 Intel Core Ultra 7 258V + Arc 140V (Xe2) @ 8533 MT/s LPDDR5X! ✨"
    ]

    print("\n--- 1. Testing Raw Text Tokenization Parity ---")
    for idx, text in enumerate(test_texts):
        hf_ids = hf_tok.encode(text)
        my_ids = ainfer_tok.encode(my_tok, text)
        match = (hf_ids == my_ids)
        if not match:
            results["all_passed"] = False
        results["tests"].append({
            "name": f"text_parity_{idx}",
            "type": "encode_parity",
            "text_sample": text[:40],
            "num_tokens": len(hf_ids),
            "match": match,
            "hf_ids": hf_ids,
            "ainfer_ids": my_ids
        })
        print(f"  [{idx+1}/{len(test_texts)}] Match: {match} ({len(hf_ids)} tokens) | Sample: {repr(text[:30])}")

    # 2. Special Tokens Preservation
    print("\n--- 2. Testing Special Tokens Preservation ---")
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>"
    ]
    for st in special_tokens:
        hf_id = hf_tok.convert_tokens_to_ids(st)
        my_ids = ainfer_tok.encode(my_tok, st)
        is_single = (len(my_ids) == 1 and my_ids[0] == hf_id)
        if not is_single:
            results["all_passed"] = False
        results["tests"].append({
            "name": f"special_token_{st}",
            "type": "special_token",
            "token": st,
            "expected_id": hf_id,
            "encoded_ids": my_ids,
            "match": is_single
        })
        print(f"  Special token '{st}': id={hf_id}, encoded={my_ids}, SingleToken: {is_single}")

    # 3. Round-trip Decode Parity
    print("\n--- 3. Testing Round-Trip Decode Parity ---")
    for idx, text in enumerate(test_texts):
        encoded = ainfer_tok.encode(my_tok, text)
        decoded = ainfer_tok.decode(my_tok, encoded)
        match = (decoded == text)
        results["tests"].append({
            "name": f"round_trip_{idx}",
            "type": "round_trip",
            "match": match,
            "original": text,
            "decoded": decoded
        })
        print(f"  [{idx+1}/{len(test_texts)}] Round-trip match: {match}")
        if not match:
            results["all_passed"] = False

    # 4. Chat Template Validation (enable_thinking=False and enable_thinking=True)
    print("\n--- 4. Testing Chat Template Formatting Parity ---")
    conversations = [
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a quicksort function in C++."}
        ],
        [
            {"role": "user", "content": "Hello! Tell me about Lunar Lake."}
        ],
        [
            {"role": "system", "content": "You are an expert mathematician."},
            {"role": "user", "content": "What is 17 * 19?"},
            {"role": "assistant", "content": "17 * 19 is 323."},
            {"role": "user", "content": "Now divide it by 17."}
        ]
    ]

    for c_idx, conv in enumerate(conversations):
        for think in [False, True]:
            hf_rendered = hf_tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=think)
            my_rendered = ainfer_tok.render_chat(conv, add_generation_prompt=True, enable_thinking=think)
            text_match = (hf_rendered == my_rendered)

            hf_ids = hf_tok.encode(hf_rendered)
            my_ids = ainfer_tok.encode(my_tok, my_rendered)
            token_match = (hf_ids == my_ids)

            test_ok = (text_match and token_match)
            if not test_ok:
                results["all_passed"] = False

            results["tests"].append({
                "name": f"chat_template_conv{c_idx}_think_{think}",
                "type": "chat_template",
                "enable_thinking": think,
                "text_match": text_match,
                "token_match": token_match,
                "num_tokens": len(hf_ids),
                "rendered_preview": my_rendered[:60].replace("\n", "\\n")
            })
            print(f"  Conv {c_idx} (think={think}): TextMatch={text_match}, TokenMatch={token_match} ({len(hf_ids)} tok)")

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[T6.2] All tests completed! all_passed={results['all_passed']}. Report written to: {REPORT_PATH}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
