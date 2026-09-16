#!/usr/bin/env python3
"""T8.4: pinned tokenizer golden. stdlib + tok only (no transformers, no HF).

Golden file golden_t44.json carries explicit {label, input, ids} records plus
pinned template strings, prompt_ids, asset sha256, and specials count.
Asserts every value; exit 0 GOLDEN-OK, else 1 with mismatch list.
Runs under the project venv (needs `tokenizers`, `jinja2`).
Registered as ctest `tokenizer_golden` (host-only, FAST tier).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fails.append(name)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    golden = json.load(open(os.path.join(here, "golden_t44.json")))
    for fn, meta in golden["assets"].items():
        p = os.path.join(tok.MODEL_DIR, fn)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        check(f"asset-sha256:{fn}", h == meta["sha256"], h[:16])
    otok = tok.load()
    check("specials-count-33",
          len(tok.added_special_tokens()) == golden["specials_count"] == 33)
    chat_ids = None
    for rec in golden["encode"]:
        got = tok.encode(otok, rec["input"])
        check(f"encode:{rec['label']}", got == rec["ids"],
              str(got[:6]) if got != rec["ids"] else "")
        if rec["label"] == "chat_markup":
            chat_ids = rec["ids"]
    dec_t = tok.decode(otok, chat_ids, skip_special_tokens=True)
    dec_f = tok.decode(otok, chat_ids, skip_special_tokens=False)
    check("decode:skip-true-roundtrip",
          "user" in dec_t and "<|im_start|>" not in dec_t, repr(dec_t[:24]))
    check("decode:skip-false-roundtrip", dec_f == [
        r for r in golden["encode"] if r["label"] == "chat_markup"][0]["input"])
    msgs_min = [{"role": "user", "content": "Hi"}]
    check("template:minimal",
          tok.render_chat(msgs_min, add_generation_prompt=True,
                          enable_thinking=False)
          == golden["templates"]["minimal_no_think"])
    check("template:system",
          tok.render_chat([{"role": "system", "content": "Be brief."},
                           {"role": "user", "content": "What is 84 * 3 / 2?"}],
                          add_generation_prompt=True, enable_thinking=False)
          == golden["templates"]["system_no_think"])
    check("prompt_ids:e2e",
          tok.prompt_ids(otok, msgs_min) == golden["prompt_ids_e2e"])
    for rec in golden["encode"]:
        if rec["label"].startswith(("chat_", "malformed_")):
            continue
        check(f"roundtrip:{rec['label']}",
              tok.decode(otok, rec["ids"], skip_special_tokens=False)
              == rec["input"])
    if fails:
        print(f"GOLDEN FAIL: {len(fails)} mismatches: {fails}")
        return 1
    print(f"GOLDEN-OK: pinned tokenizer suite "
          f"(v{golden['version']}, {golden['model_rev'][:9]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
