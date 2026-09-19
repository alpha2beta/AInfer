#!/usr/bin/env python3
"""T8.5 retrieval cases: haystack + needle(s) + distractors at 4K/8K/16K/32K.

For each corpus gen-config: builds exact-ctx token ids, writes fp32 embed
chunks <prefix>_<ch>.bin (M=256, BF16 embed table, headers-free path) and a
question ids file. Decode uses the merged single-binary path
(decode_l0 --prefill-chunks). Check: expected code in generated text.
Usage: mk_retrieval.py [--case ID]  (writes /tmp/retr/<id>/ + manifest)
"""
import json
import os
import sys

import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import tok as tokenizer
import fwd_cpu as F

M = 256
FILLER = ("The northern lights observatory tracks auroral activity every night. "
          "Researchers log solar wind speed, cloud cover, and instrument status. "
          "Supplies arrive by snowmobile twice a month during winter. ")
NEEDLE = "The special access code for the {place} observatory is {code}."
DECOY = "The old access code for the {place} observatory was {code} (expired)."
QUESTION = (" What is the special access code for the {place} observatory? "
            "Reply with just the number.")
PLACES = ["northern lights", "southern cross", "polar dawn", "arctic circle",
          "midnight sun"]


def build(ctx, depth, code, place, multi=False, distract=False):
    tk = tokenizer.load()
    filler_ids = tokenizer.encode(tk, FILLER)
    needle_ids = tokenizer.encode(tk, NEEDLE.format(place=place, code=code))
    parts, pos = [], 0
    target = int(ctx * depth)
    while pos + len(filler_ids) < target:
        parts.append(filler_ids)
        pos += len(filler_ids)
    parts.append(needle_ids)
    pos += len(needle_ids)
    if multi:
        code2 = str(int(code) + 111111)
        n2 = tokenizer.encode(tk, NEEDLE.format(
            place=PLACES[(PLACES.index(place) + 2) % len(PLACES)], code=code2))
        while pos + len(filler_ids) < int(ctx * (depth + 0.2)):
            parts.append(filler_ids)
            pos += len(filler_ids)
        parts.append(n2)
        pos += len(n2)
    if distract:
        dcode = str(int(code) - 222222)
        dec = tokenizer.encode(tk, DECOY.format(place=place, code=dcode))
        while pos + len(filler_ids) < int(ctx * (depth + 0.1)):
            parts.append(filler_ids)
            pos += len(filler_ids)
        parts.append(dec)
        pos += len(dec)
    while pos + len(filler_ids) <= ctx:
        parts.append(filler_ids)
        pos += len(filler_ids)
    if pos < ctx:  # exact pad with a filler prefix slice
        parts.append(filler_ids[:ctx - pos])
        pos = ctx
    ids = [i for p in parts for i in p][:ctx]
    assert len(ids) == ctx, (len(ids), ctx)
    return tk, ids


def main():
    only = None
    if "--case" in sys.argv:
        only = sys.argv[sys.argv.index("--case") + 1]
    corp = [c for c in json.load(
        open(os.path.join(REPO, "tools", "quality", "corpus_t85.json")))["cases"]
        if c["category"] == "retrieval" and "gen" in c["check"]]
    emb = None
    for ci, c in enumerate(corp):
        if only and c["id"] != only:
            continue
        g = c["check"]["gen"]
        code = str(100000 + (ci * 137279) % 900000)
        place = PLACES[ci % len(PLACES)]
        multi = "multi" in g["tag"]
        distract = "distract" in g["tag"]
        tk, ids = build(g["ctx"], g["depth"], code, place, multi, distract)
        d = f"/mnt/usb/retr/{c['id']}"
        os.makedirs(d, exist_ok=True)
        if emb is None:
            emb = F.load_bf16(
                ["model.language_model.embed_tokens.weight"])[
                "model.language_model.embed_tokens.weight"]
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"id": c["id"], "ctx": g["ctx"], "depth": g["depth"],
                       "code": code, "place": place,
                       "nchunks": g["ctx"] // M}, f)
        for ch in range(g["ctx"] // M):
            h = emb[torch.tensor(ids[ch * M:(ch + 1) * M])].numpy().astype("float32")
            h.tofile(os.path.join(d, f"ch_{ch}.bin"))
        q = tokenizer.encode(tk, QUESTION.format(place=place))
        # Merged path needs FULL ids (context + question tail, cf. 64K
        # merge: P=65043, impP=TC=65024); guard requires P >= TC.
        full = ids + q
        with open(os.path.join(d, "ids.txt"), "w") as f:
            f.write(",".join(map(str, full)))
        with open(os.path.join(d, "q.txt"), "w") as f:
            f.write(",".join(map(str, q)))
        print(f"{c['id']}: ctx={g['ctx']} chunks={g['ctx']//M} code={code}",
              flush=True)


if __name__ == "__main__":
    main()
