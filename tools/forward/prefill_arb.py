"""T7.4 prefill-quality arbitration: chunked-prefill + host lm_head top-1 vs
the certified recorded loop top-1, on the first 64 ids of the 70-token e2e
prompt (chunk A: 0..31, chunk B: 32..63 — matches chunk64mc geometry).

Modes:
  embed: BF16 embed table gather -> /tmp/arb_in_a.bin, /tmp/arb_in_b.bin
         (fed to chunk64mc via CHUNK64MC_HOST_IN=/tmp/arb_in)
  score: BF16 lm_head top-1 over /tmp/chunk64mc_out_b.bin ->
         /tmp/arb_top1.json + agreement vs decode_l0 top5_per_step
         (positions 32..63) with near-tie margins from loop top-5 values.
Usage: prefill_arb.py embed|score [--loop base_e2e.json]
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/mnt/usb/AInfer"
sys.path.insert(0, os.path.join(REPO, "tools", "forward"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import fwd_cpu as F

IDS70 = [248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13,
         5044, 1683, 15060, 1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814,
         47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321, 30246, 303,
         279, 1534, 4087, 13, 248046, 198, 248045, 846, 198, 3710, 369, 220,
         23, 19, 348, 220, 18, 593, 220, 17, 30, 17308, 440, 1066, 279, 1324,
         13, 248046, 198, 248045, 74455, 198, 248068, 198]
assert len(IDS70) == 70
M = 32


def do_embed():
    emb = F.load_bf16(["model.language_model.embed_tokens.weight"])[
        "model.language_model.embed_tokens.weight"]
    for ch, path in (0, "/tmp/arb_in_a.bin"), (1, "/tmp/arb_in_b.bin"):
        ids = IDS70[ch * M:(ch + 1) * M]
        h = emb[torch.tensor(ids)].numpy().astype(np.float32)
        assert h.shape == (M, 5120), h.shape
        h.tofile(path)
        print(f"wrote {path} ({h.shape})", flush=True)
    del emb


def do_score(_loop_path=None):
    # Reference: T=64 INT4 chain + BF16 head (/tmp/arb_ref.json). The recorded
    # loop only stores gen-step top-5, so prefill arbitration rides on the
    # INT4-float chain (same quant level — isolates fp16-chunk-path noise);
    # BF16-vs-INT4 is separately characterized (T1.4 batch 53/60).
    ref = json.load(open("/tmp/arb_ref.json"))
    lm = F.load_bf16(["lm_head.weight"])["lm_head.weight"]
    got = np.fromfile("/tmp/chunk64mc_out_b.bin", dtype=np.float32)
    assert got.size == M * 5120
    x = torch.from_numpy(got.reshape(M, 5120))
    with torch.no_grad():
        logits = (x @ lm.T).numpy()
    del lm
    top1 = logits.argmax(axis=1).tolist()
    agree, near_tie, hard = 0, 0, 0
    det = []
    for m in range(M):
        pos = 32 + m
        want = ref["top1"][pos]
        got1 = top1[m]
        ok = got1 == want
        agree += ok
        row = {"pos": pos, "want": want, "got": got1, "agree": bool(ok),
               "ref_margin": ref["top1_margin"][pos]}
        if not ok:
            if ref["top1_margin"][pos] < 1.0:
                near_tie += 1
            else:
                hard += 1
        det.append(row)
    print(f"arbitration: {agree}/{M} top-1 vs INT4-float chain "
          f"(near-tie misses: {near_tie}, hard misses: {hard})", flush=True)
    for r in det:
        if not r["agree"]:
            print(f"  pos {r['pos']}: want {r['want']} got {r['got']} "
                  f"ref_margin {r['ref_margin']}", flush=True)
    json.dump({"agree": agree, "n": M, "near_tie": near_tie, "hard": hard,
               "top1_chunkB": top1, "detail": det},
              open("/tmp/arb_top1.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        do_embed()
    elif sys.argv[1] == "score":
        lp = sys.argv[3] if len(sys.argv) > 3 else "/tmp/base_e2e.json"
        do_score(lp)
    else:
        sys.exit(2)
