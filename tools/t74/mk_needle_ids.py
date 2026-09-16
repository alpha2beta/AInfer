"""Write /tmp/ids_needle<v>.txt: variant context ids + question ids.
Usage: mk_needle_ids.py <variant>
"""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer

Q = (" What is the special access code for the northern lights observatory? "
     "Reply with just the number.")


def main():
    vi = int(sys.argv[1])
    corp = json.load(open(os.path.join(REPO, "tools", "t74",
                                       "needle_corpus.json")))
    v = corp["variants"][vi]
    tk = tokenizer.load()
    q = tokenizer.encode(tk, Q)
    ids = v["ids"] + q
    assert len(v["ids"]) == 65024
    with open(f"/tmp/ids_needle{vi}.txt", "w") as f:
        f.write(",".join(map(str, ids)))
    print(f"variant {vi}: P={len(ids)} (ctx 65024 + Q {len(q)})", flush=True)


if __name__ == "__main__":
    main()
