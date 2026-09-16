"""Check needle retrieval: decode generated ids for the variant's magic code.
Usage: needle_check.py <variant> <generated_ids.json | ids,comma,...>
Prints HIT/MISS by detokenizing and searching for the code string.
"""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer


def main():
    vi = int(sys.argv[1])
    corp = json.load(open(os.path.join(REPO, "tools", "t74",
                                       "needle_corpus.json")))
    v = corp["variants"][vi]
    code = str(v["code"])
    arg = sys.argv[2]
    if arg.endswith(".json"):
        d = json.load(open(arg))
        gen = d.get("generated", d)
    else:
        gen = [int(x) for x in arg.split(",") if x.strip()]
    tk = tokenizer.load()
    text = tokenizer.decode(tk, gen)
    hit = code in text
    print(f"variant {vi} code {code}: {'HIT' if hit else 'MISS'} "
          f"({len(gen)} tokens)", flush=True)
    if not hit:
        print("tail: " + text[-200:], flush=True)
    return 0 if hit else 1


if __name__ == "__main__":
    sys.exit(main())
