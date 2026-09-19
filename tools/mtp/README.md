# MTP (Multi-Token Prediction) — implementation reference

Speculative-decoding path for Qwen3.8-27B on Intel Arc Pro B60: MTP-head
drafts + dual-token verification (depth-1) + chained depth-2 with adaptive
dispatch. **Status: landed and measured** — depth-1 +27% @alpha 0.71,
depth-2 up to +73% @alpha 1.0 (25.56 vs 14.80 tok/s), all bitwise identical
to greedy; adaptive default within ~2–4% of the better fixed policy per
prompt. Written as a porting reference, e.g. for a Qwen3.5-A3B MoE model on
Intel Core Ultra 7 258V: steal the design, re-measure the economics on your
hardware.

Contents of this folder (copies for porting reference; canonical originals
stay in `../t72/` — script output paths below point here):

| File | Role |
|---|---|
| `README.md` | this reference |
| `mtp_accept.py` | MTP-head acceptance measurement (CPU streamed BF16; writes `report_accept.json`) |
| `pl_accept.py` | prompt-lookup acceptance vs greedy loop truth (writes `report_pl_accept.json`) |
| `report_t72.json` | MTP-1 dataflow recovery, draft-cost model, adoption gate |
| `report_accept.json` / `report_accept_full.json` | acceptance samples (alpha 0.585 / pooled 0.624) |
| `report_pl_accept.json` | prompt-lookup: alpha 0/65 (DEAD) |
| `report_mtpdraft.json` | device draft-slice verdict (`MTPDRAFT-OK`, 6.4 ms/draft) |
| `report_mtp_revisit.json` | 64K-economics revisit verdict (RE-DEFER with triggers) |
| `mtpdraft_replay.cpp` | byte-copy of the draft-slice harness (`../cmdlist/`, build-registered there — edit only the original) |
| `gemvm3_replay.cpp` | byte-copy of the triple-lane GEMV harness (`../cmdlist/`, ctest `gemvm3_replay`): M3 vs M2+M1 bitwise on real weights, 1.62× |
| `mtp_kernels_ref.cpp` | verbatim extracts from `../cmdlist/kernels.cpp`: `Int4GemvM2` (dual-token GEMV) + `Int4GemvM3` (triple-lane GEMV) + `Concat2`; header cites origin commit + line ranges. Reference only, not built |

Related code kept in place (build-registered, do not move):

- `../cmdlist/mtpdraft_replay.cpp` — draft-slice harness (ctest `mtpdraft_replay`): one MTP-1 draft as a single recorded raw-L0 list over 4 positions with real `.binfer` weights (~0.9 GB spans); checks host parity + bitwise reset-determinism.
- `../cmdlist/kernels.cpp` — `Int4GemvM2` (purely additive): streams P/S once, unpacks nibbles once, dual dp4a against two activation vectors in GRF; 1 WI per L0 group. Plus 10-line `Concat2` (`_ZTS7Concat2`).
- `../decode/decode_l0.cpp` (MTP engine, behind `--mtp` / `AINFER_MTP=1`): MTP weight allocs, draft-list recording, 64 dual-token verification lists (`embM2`, `layersM2`, `tailM2`, `commitSpecR`), accept/reject loop — plus depth-2: 64 triple lists (`embM3`, `layersM3`, `tailM3`, `commitSpec2R`), chained second-draft list (`mtpDraft2R` re-applies the MTP module to `(embed(d1), h_mtp1)` via `dChH`/`dCtrlDraft2`), two-level specular states, accept-0/1/2, and the **adaptive depth controller** (per-round M2/M3 dispatch on trailing d1-alpha ≥ 0.8 + M3-rate ≥ 18.5, M2-seeded discovery, on-demand draft-2, probe backoff 8→32; `AINFER_MTP2=0/1` forces M2/M3). Default-path delta is one `[Normal Decode]` perf line; stop-count fixed to match base (`<= G` guards).

## 1. MTP-1 dataflow (Qwen3.5-family head layout)

Per step: trunk hidden `h[t+1]` + embedded next token `embed(ids[t+1])` → draft for `t+2`:

1. `pre_fc_norm_embedding` on embed, `pre_fc_norm_hidden` on h (RMSNorm BF16).
2. `Concat2` → 10240-wide vector.
3. Stream-fuse FC `mtp.fc.weight` 5120×10240 (INT4 sym-g128, same quant as trunk).
4. One full-attention decoder layer (`mtp.layers.0.*`: q/k/v/o + GQA + MLP) with **its own 1-slot BF16 KV cache**; RoPE tables and `DecodeControl` struct format shared with trunk.
5. `mtp.norm` → **shared** `lm_head` (248320 vocab) → two-stage device argmax.

Tensor inventory: 7 INT4 mats + 8 BF16 norms + embed rows. **Porting check #1:** dump the target checkpoint's `mtp.*` namespace first — if names/shapes match, the dataflow ports verbatim. MoE trunk doesn't change the head, but some MoE releases drop it (no head → MTP off the table).

## 2. Measurement protocol (reuse verbatim)

1. **Draft correctness (device):** harness-first, real weights — hidden parity 4.46e-08, exact token, top-5 5/5, reset-deterministic. Cost on B60: **6.4 ms/draft** vs 68 ms full step (~0.06 draft fraction).
2. **Acceptance (CPU, no device):** streamed one-shard BF16 trunk forward for `h` + greedy tokens; MTP layer via HF `Qwen3_5DecoderLayer`; greedy acceptance = draft==trunk. Result: **alpha 0.585–0.624 pooled**, just under the 0.6 sustained gate (≥20 samples + quality gate required).
3. **Verify-cost model:** C(8,65K) ≈ 1.7 s = 1.09 fixed (measured flat 1093–1096 ms/chunk for W=8–256) + 0.57 attention (192 rows × 65K × 45.7 ns/score slope). Speedup = E(k,α) tokens/round ÷ (draft + verify cost).

## 3. Economics that drove the deferral (re-derive on your hardware)

- Short context: **0.24× (a loss)** — verify 1.1 s vs 8×0.1 s sequential. Any adoption is long-context-only mode.
- 64K: **~2.5× prize** — E(8, 0.62) = 2.62 tokens/round; 2.62 × 1.65 s ÷ (1.7 + 0.05) s. Real but with unproven links (verify-list hardening, resample path) and no production 64K traffic.
- Prompt-lookup: E≈1.0, C_dec/C_verify < 1 at every length. Dead; re-test only for highly templated workloads.
- Revisit triggers: (1) real 64K decode traffic, (2) chained-draft proof, (3) remeasured long-context alpha, (4) a speculative-regression quality gate.

## 4. Depth-2 + adaptive dispatch (measured 2026-09-19, B60)

- Chained drafts: d1 = MTP(confirmed), d2 = MTP(d1, h_mtp1); triple verify
  (t, d1, d2) in one shot via `Int4GemvM3` (weights stream once: 308 µs vs
  500 µs M2+M1 sequential, 1.62×; bitwise 0/36864).
- Measured curve, all bitwise identical to greedy: alpha 0.67–0.71 → 18.7 t/s
  (≈ depth-1); alpha 1.0 → 25.56 t/s (+73% vs 14.80 base, +16% over depth-1
  ceiling). Adaptive default: 18.99 / 24.50 on those prompts (within ~2–4%
  of the better fixed policy, no per-prompt tuning).
- Ceiling: 25.7 t/s max at 116.8 ms rounds (third lane costs +26 ms/round:
  +0.35 ms/linear-layer, +0.33 ms/full-layer, +3.4 ms second draft). The
  29.6 bar needs verify-cost cuts (~17 ms) or depth-3, not more of the same.
- Porting note: the adaptive controller is pure host logic (~40 lines, no
  device work) — the highest-EV piece to copy after the kernels. Depth
  selection composes with MoE verify (batched, sublinear cost) favorably.

## 5. Porting checklist (A3B MoE on Ultra 7 258V)

1. Checkpoint audit (`mtp.*` present? same 7+8 set?).
2. L0 device picker pins `0x8086:0xe211` (B60) — parameterize it (`mtpdraft_replay.cpp:128-137`); Lunar Lake iGPU is a different ID.
3. 258V is unified LPDDR5x (no VRAM): static arenas still work, but profile host-visible shared allocations vs explicit device arenas.
4. MoE trunk verify cost: sparse active params help per-token cost, but expert dispatch and small-batch MoE efficiency on Xe2 are the unknowns — measure roof + one MoE-layer GEMV before projecting speedup.
5. Kernels (`Int4GemvM2`, `Concat2`, recorded-list pattern) port as-is under INT4 sym-g128; retune work-group sizing for 8 Xe cores.
6. Build scale if greenlit: MTP kernels → 1-layer MTP KV → draft/verify/accept loop surgery (~2–3 turns on B60); keep behind a flag gate like `opt_mtp`.
7. Run the CPU acceptance job first: cheap, device-free, and alpha < 0.6 kills the project before any kernel is written.
