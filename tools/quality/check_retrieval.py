#!/usr/bin/env python3
"""Check one retrieval case: expected code in generated text."""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer


def main(cid):
    d = f"/mnt/usb/retr/{cid}"
    man = json.load(open(os.path.join(d, "manifest.json")))
    gen = json.load(open(os.path.join(d, "gen.json")))
    tk = tokenizer.load()
    text = tokenizer.decode(tk, gen["generated"])
    hit = man["code"] in text
    print(f"{cid}: code={man['code']} text={text[:80]!r} "
          f"-> {'HIT' if hit else 'MISS'}", flush=True)
    json.dump({"hit": hit, "text": text}, open(os.path.join(d, "check.json"), "w"))
    return 0 if hit else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
