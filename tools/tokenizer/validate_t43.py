"""T4.3 validation: our tokenizer vs HF AutoTokenizer (exact match required).

Checks: plain-text encode, chat-markup encode, all 33 specials round-trip,
decode with specials, chat-template render parity (same kwargs both sides),
prompt_ids end-to-end. Writes tools/tokenizer/report_t43.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok as ours

from transformers import AutoTokenizer
ref = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "..", "models", "Qwen3.8-27B"),
    trust_remote_code=False)

otok = ours.load()
results = []


def check(name, cond, detail=""):
    results.append({"case": name, "pass": bool(cond), "detail": detail})
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


plain = ["Hello, world!", "What is 84 * 3 / 2?",
         "Explain quantum computing in simple terms.",
         "Write a Python function that reverses a string.",
         "Unicode: caf\u00e9 \u4e2d\u6587 \U0001f600",
         ""]  # empty edge case
for t in plain:
    a = ours.encode(otok, t)
    b = ref.encode(t)
    check(f"encode:{t[:24]!r}", a == b, f"ours={a[:8]} ref={b[:8]}")

chat = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
check("encode:chat-markup", ours.encode(otok, chat) == ref.encode(chat),
      str(ref.encode(chat)))

addeds = ours.added_special_tokens()
check("specials:count-33", len(addeds) == 33, str(len(addeds)))
rt_ok = True
for i, content in addeds.items():
    a = ours.encode(otok, content)
    # each special must encode to its own single id
    if a != [i]:
        rt_ok = False
        check(f"special-single:{content}", False, f"got={a} want=[{i}]")
        break
    if ref.encode(content) != [i]:
        rt_ok = False
        check(f"special-ref:{content}", False, str(ref.encode(content)))
        break
check("specials:all-single-id", rt_ok)

dec = ours.decode(otok, ref.encode(chat))
check("decode:chat-markup", dec == ref.decode(ref.encode(chat)), repr(dec[:40]))

# chat template parity: same kwargs both sides (thinking disabled => minimal)
msgs = [{"role": "user", "content": "Hi"}]
r1 = ours.render_chat(msgs, add_generation_prompt=True, enable_thinking=False)
r2 = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                             enable_thinking=False)
check("chat-template:minimal", r1 == r2, f"ours={r1!r} ref={r2!r}")
msgs2 = [{"role": "system", "content": "Be brief."},
         {"role": "user", "content": "What is 84 * 3 / 2?"}]
r1 = ours.render_chat(msgs2, add_generation_prompt=True, enable_thinking=False)
r2 = ref.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True,
                             enable_thinking=False)
check("chat-template:system", r1 == r2)
# default HF path (thinking on) must also match when kwargs match
r1 = ours.render_chat(msgs, add_generation_prompt=True, enable_thinking=True,
                      ) if False else None
try:
    r1 = ours.render_chat([{"role": "user", "content": "Hi"}],
                          add_generation_prompt=True, enable_thinking=True,
                          )
    import jinja2  # noqa
    r2 = ref.apply_chat_template([{"role": "user", "content": "Hi"}],
                                 tokenize=False, add_generation_prompt=True,
                                 enable_thinking=True)
    check("chat-template:thinking", r1 == r2, f"ours={r1[:60]!r}")
except Exception as e:
    check("chat-template:thinking", False, f"{type(e).__name__}: {e}")

pid = ours.prompt_ids(otok, msgs)
check("prompt_ids:e2e",
      pid == ours.encode(otok, ours.render_chat(msgs)), str(pid))

npass = sum(1 for r in results if r["pass"])
report = {"cases": len(results), "passed": npass, "all_pass": npass == len(results),
          "results": results,
          "note": "reference = transformers AutoTokenizer(Qwen2Tokenizer) on pinned dir"}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_t43.json")
json.dump(report, open(out, "w"), indent=1)
print(f"{npass}/{len(results)} passed -> {out}")
sys.exit(0 if npass == len(results) else 3)
