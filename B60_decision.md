# B60 Continuation Decision (B60-R8)

Date: 2026-09-22. Branch: `B60`. Baseline: `CHUNK_M=512` + `AINFER_PP=1` + `AINFER_BATCH=1`, BF16 KV.
Criteria: `review_B60.md` §6. Evidence reports: `tools/bench/report_r4_*`, `report_r7_*`, `report_m512*`, `report_r2_*`.

## Aligned local comparisons (same B60, same model family)

| Metric | AInfer (INT4g128, BF16 KV, recorded loop) | llama.cpp SYCL `build-sycl-f16` (Q4_K_XL, f16 KV) | Ratio |
|---|---|---|---|
| Prefill P512 | 35.7 tok/s (`r5_pp`, M256+PP) / ~38 @M512 stack | 515.9 tok/s (`report_r4_llama_p512`) | ~13.6–14.4× behind |
| Prefill P4096 | 35.2 tok/s (116.3 s, `r7_normal_p4096`) | 485.7 tok/s (`report_r4_llama_p4096`) | ~13.8× behind |
| Decode (32 tok, P~256–512 ctx) | 13.95 tok/s (`report_r4_ours_decode32_p256`) | 14.85 tok/s (`r4_llama` tg) | within 6.1% |
| OV GenAI prefill (prior, same box) | same AInfer path | 516 @235tok / 1044 @884tok (`report_r4_ov_*`) | ~15–30× behind |

## Criterion evaluation (`review_B60.md` §6.1–6.3)

- §6.1 "prefill 2× over ~30 tok/s": NOT MET (+24% to ~35 tok/s).
- §6.1 "64K −30%": NOT MET (−16% at P16K: 578.6 s vs 691.2 s; P64K untested on new baseline).
- §6.1 "decode within 15% of best aligned non-speculative": MET (13.95 vs 14.85, −6.1%).
- §6.1 "measurable advantage": PARTIAL — deterministic recorded Level Zero execution, per-tensor quality parity (`271 16` holds across PP/BATCH/M512), full evidence chain (154/154 gate). No memory/long-context-stability advantage demonstrated vs llama (llama fits 17.5 GB model + KV in 23 GB VRAM; AInfer needs 6 GiB PP repack + BF16 KV).
- §6.3 freeze trigger "prefill >2× slower than aligned llama/OV": MET against (13.8×). Other freeze triggers not met (decode competitive; investigations partially passed: PP+BATCH+M512 stack +24.6% ≥ 15% go bar; recurrence proven negligible rather than failed).
- MTP (§6.1 within 20% of speculative best): unmeasured — MTP head deferred, text-only v1. Cannot claim.

## Decision: continue as a RESEARCH runtime (§6.2), not a performance/production runtime

> A model-specific Intel GPU kernel and runtime research platform, not a general-purpose production inference server.

Rationale: the prefill gap is architectural (13.8× vs SYCL llama, 15–30× vs OV), not tuning; R7 showed the available fused-attention v1 path is negative on B60 (+56% at P16K) and linear GEMM, though improved (+24.6% stack), cannot close an order of magnitude. Decode parity and determinism keep the branch valuable for kernel research (DPAS layouts, DeltaNet state analysis, recorded-execution correctness).

## Conditions for revisiting

1. B60-native attention blocking (not 258V 64-wide tiling) that turns `prefill_full` (44.5% @P16K) from bottleneck into parity.
2. INT8-KV path (`AINFER_KV8=1`) with quality gate, or quantized-KV XMX processing per B60-I4.
3. Remaining P6 fusion (11/18 kernels) + `cvt` fusion — expected single-digit %, tracked but not decision-relevant.
4. MTP depth-2 acceptance measurement before any speculative claim.

Serving features (P3) stay out of scope until condition 1 moves prefill by multiples, not percent.
