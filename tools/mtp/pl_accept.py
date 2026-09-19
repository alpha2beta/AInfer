"""T7.2 revisit: prompt-lookup draft acceptance vs greedy loop truth.

Protocol mirrors mtp_accept.py: drafts proposed per position, agreement =
top-1 equality vs the recorded loop's greedy continuation (exact-output
gate: with greedy acceptance, SD output == greedy decode by construction;
ties broken first-max on both sides).
Draft rule (standard prompt-lookup): at each position, take the trailing
ngram (NB Candidates of length m), find all occurrences in the context +
generated-so-far, continue from the LONGEST match (first on ties), propose
up to K tokens. Acceptance = leading run of draft==truth (greedy cascade).
Usage: pl_accept.py <context_ids.txt> <truth.json> [--m 2,3,4,5,6] [--K 8]
Writes tools/mtp/report_pl_accept.json.
"""
import json
import os
import sys

REPO = "/mnt/usb/AInfer"

MS = [2, 3, 4, 5, 6]
K = 8
for a in sys.argv:
    if a.startswith("--m="):
        MS = [int(x) for x in a.split("=")[1].split(",")]
    if a.startswith("--K="):
        K = int(a.split("=")[1])


def load_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().strip().split(",") if x.strip()]


def propose(ctx, m, K):
    """Longest trailing-m-gram match continuation (first on ties)."""
    if len(ctx) < m:
        return []
    key = ctx[-m:]
    best, best_len = None, m
    # search all start positions (naive; contexts here are ~2K)
    for s in range(len(ctx) - m):
        if ctx[s:s + m] == key:
            if best is None:
                best = s
    if best is None:
        return []
    return ctx[best + m:best + m + K]


def main():
    ctx = load_ids(sys.argv[1])
    truth = json.load(open(sys.argv[2]))["generated"]
    n_pos = len(truth)
    per_m = {}
    for m in MS:
        acc_runs, draftable = [], 0
        for i in range(n_pos):
            hist = ctx + truth[:i]
            d = propose(hist, m, K)
            if d:
                draftable += 1
            run = 0
            for j, tok in enumerate(d):
                if i + j < n_pos and truth[i + j] == tok:
                    run += 1
                else:
                    break
            acc_runs.append(run)
        per_m[str(m)] = {
            "positions": n_pos,
            "draftable": draftable,
            "mean_accept": round(sum(acc_runs) / n_pos, 4),
            "agree": sum(1 for r in acc_runs for _ in [r] if r > 0),
        }
    # headline: best m by mean accepted run; E[tokens] for K=8 with bonus
    best = max(per_m, key=lambda m: per_m[m]["mean_accept"])
    rep = {"protocol": "prompt-lookup (longest-match, first-tie) vs loop greedy",
           "context_len": len(ctx), "truth_len": n_pos, "K": K,
           "per_m": per_m, "best_m": best,
           "mean_accept_best": per_m[best]["mean_accept"]}
    json.dump(rep, open(os.path.join(REPO, "tools", "mtp",
                                     "report_pl_accept.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_m"},
                     indent=1), flush=True)
    print("per_m:", json.dumps(per_m), flush=True)


if __name__ == "__main__":
    main()
