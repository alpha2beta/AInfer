"""T7.4 64K needle corpus: 5 haystack variants (needle at 0/25/50/75/100% depth)
for long-context retrieval quality once chunked-prefill production lands.
Target: 65024-token prompt (leaves 512 for decode under 65544 sizing).

Haystack is synthetic filler (programmatic paragraphs — no copyrighted text);
needle is a distinctive access-code sentence with a per-depth magic number.
IDs are built directly (filler ids cycled + needle ids spliced) so the total
is exact; text is rendered for the record via detokenization.
Writes tools/t74/needle_corpus.json (ids + needle spans + expected codes).
"""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer

PROMPT_TOKENS = 65024
NEEDLE_TMPL = ("Among the routine maintenance logs, one entry stands out. "
               "The special access code for the northern lights observatory "
               "is {code}. Please remember this code for the final report. ")
FILLER_TMPL = ("The quarterly maintenance review covers filter replacements, "
               "calibration checks, and inventory reconciliation across all "
               "sectors. Technicians noted nominal readings on every panel, "
               "with only minor dust accumulation on the outer vents. The "
               "next scheduled inspection will follow the standard rotation. "
               "All measurements were recorded in the central ledger. ")


def ids_of(tk, text):
    return tokenizer.encode(tk, text)


def main():
    tk = tokenizer.load()
    filler = ids_of(tk, FILLER_TMPL)
    assert len(filler) > 0
    codes = [739521, 184963, 502817, 926438, 317654]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    variants = []
    for vi, (depth, code) in enumerate(zip(depths, codes)):
        needle = ids_of(tk, NEEDLE_TMPL.format(code=code))
        body_len = PROMPT_TOKENS - len(needle)
        reps = body_len // len(filler) + 1
        body = (filler * reps)[:body_len]
        at = min(int(depth * body_len), body_len)
        ids = body[:at] + needle + body[at:]
        assert len(ids) == PROMPT_TOKENS, len(ids)
        text = tokenizer.decode(tk, ids)
        assert str(code) in text, "needle code lost in round-trip"
        variants.append({"variant": vi, "depth": depth, "code": code,
                         "needle_span": [at, at + len(needle)],
                         "prompt_tokens": len(ids), "ids": ids})
        print(f"variant {vi}: depth {depth} code {code} "
              f"span {at}..{at + len(needle)}", flush=True)
    out = {"task": "T7.4-64K-needle-corpus", "prompt_tokens": PROMPT_TOKENS,
           "variants": variants}
    with open(os.path.join(REPO, "tools", "t74",
                           "needle_corpus.json"), "w") as f:
        json.dump(out, f)
    print(f"WROTE needle_corpus.json "
          f"({os.path.getsize(os.path.join(REPO, 'tools', 't74', 'needle_corpus.json'))} bytes)",
          flush=True)


if __name__ == "__main__":
    main()
