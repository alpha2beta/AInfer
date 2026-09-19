# AInfer — Project Status (as of 2026-09-19, Stamp 15)

## What it is

A custom Intel Arc Pro B60 (24 GB VRAM) inference engine for **Qwen3.8-27B** (text-only, 4K context; 64K chunked prefill supported), using SYCL/ESIMD/Level Zero. Target: beat llama.cpp SYCL baseline (~14.86 tok/s). **Achieved and exceeded: +43% to +73% throughput lead (21.24–25.56 tok/s)** via MTP speculative verification.

---

## Overall Health: Green

| Phase | Scope | Status |
|---|---|---|
| 0 — Environment & Foundations | T0.1–T0.5 | ✅ Done (5/5) |
| 1 — Model Spec & Reference | T1.1–T1.6 | ✅ Done (6/6 — T1.5 closed by T8.5/T8.6) |
| 2 — Container & Exporter | T2.1–T2.6 | ✅ Done (6/6) |
| 3 — Kernel Microbenchmarks | T3.1–T3.9 | ✅ Done (9/9) |
| 4 — End-to-End Runtime | T4.1–T4.5 | ✅ Done (5/5) |
| 5 — Static Scheduling & Memory | T5.1–T5.6 | ✅ Done (6/6) |
| 6 — Performance Tuning | T6.1–T6.4 | ✅ Done (4/4) |
| 7 — Optional Features | T7.1–T7.4 | ✅ Done (4/4 — T7.2 MTP closed, depth-1 & depth-2 chained adaptive) |
| 8 — Release Hardening | T8.1–T8.11 | 🔶 Partial (7/11 — P0 & P1 closed) |
| X — Cross-cutting | X1–X2 | ✅ Done (2/2) |

**CTest:** 48/48 registered (Stamp 15 full suite 48/48 green, 1197.9 s; fast gate 40/40 green, 85.9 s; 102/102 reports parse, tokenizer golden 25/25)

---

## Production Runtime

- **Default decode:** `decode_l0` — 66 recorded L0 command lists, zero per-token allocation/construction. Token-identical to streamed-CPU BF16 greedy on tested prompts.
- **Speculative decode:** `decode_l0 ... --mtp` — depth-1 & depth-2 chained drafts + adaptive M2/M3 dispatch, 100% bitwise greedy determinism, 21.24–25.56 tok/s.
- **64K chunked prefill → decode (single binary):** `MERGE-OK` — in-process 64K needle 5/5 HIT, zero re-uploads, zero disk.
- **HTTP API:** Demo-grade only (T7.3) — per-request process + single-flight lock. Persistent worker is T8.8.

## Key Numbers

| Metric | Value |
|---|---|
| Baseline decode (`AINFER_TOP5=0`) | 15.02 tok/s (66.58 ms/tok) vs llama.cpp SYCL 14.86 tok/s (+1.1%) |
| Speculative decode (`--mtp` / `AINFER_MTP=1`) | **21.24–25.56 tok/s** (47.09–39.12 ms/tok, adaptive M2/M3) vs llama.cpp SYCL 14.86 tok/s (**+43% to +73% throughput lead**) |
| Speculative greedy determinism | **100% bitwise token identical** to non-speculative baseline (zero quality loss) |
| Quality (greedy top-1, 6 prompts) | 47/48 across 6 prompts |
| 200-case quality suite (T8.5) | fact 40/40, code 30/30, summ 20/20, long 20/20, arith 32/40, biling 27/28 |
| Retrieval benchmark (T8.5) | 20/20 (15 merged-path 4K–32K + 5 legacy 64K) |
| Margin divergence (T8.6) | INT4 in BF16 top-5 100%; ZERO quant-attributable grade flips; INT8-KV identical |
| INT8-KV top-1 preservation | 53/53 |
| 64K needle retrieval | 5/5 HIT |

---

## Phase 8 — Release Hardening

| Task | Priority | Status |
|---|---|---|
| **T8.1** STATUS.md | P0 | ✅ Done (2026-09-16) |
| **T8.2** Strict JSON report validation | P0 | ✅ Done (102/102 parse, `reports_json` ctest gate) |
| **T8.3** Weight-stats anomaly classification | P0 | ✅ Done (136/136 classified) |
| **T8.4** Tokenizer golden suite expansion | P0 | ✅ Done (27/27 parity, 25/25 golden) |
| **T8.5** 200+ case quality benchmark | P1 | ✅ Done (2026-09-17, `report_t85.json`) |
| **T8.6** Margin-aware divergence report | P1 | ✅ Done (2026-09-17, `report_t86.json`) |
| **T8.7** Single-binary prefill→decode merge | P2 | ✅ Done (2026-09-15, `report_merge.json`) |
| **T8.8** Persistent HTTP worker | P3 | 🔴 Open |
| **T8.9** Typed arena spans/checked offsets | P4 | 🔴 Open |
| **T8.10** Fault injection & sanitizer track | P4 | 🔴 Open |
| **T8.11** Controlled AInfer vs llama.cpp benchmark | P5 | 🔴 Open |

---

## Deferred

- **Hybrid attention** (`AINFER_ATTN=hybrid`) — validated, env-gated, +25% decode speed, but deferred as default until launch-count reduction proven first.

---

## Summary

All P0 and P1 release-hardening tasks are completed. Gate C is satisfied, and Phase 1 is now fully closed (6/6). Phase 7 is fully closed (4/4) with MTP speculative decoding certified in production delivering a sustained +43% to +73% throughput lead over llama.cpp SYCL with 100% bitwise greedy determinism. The remaining open tasks are P3 (T8.8 persistent HTTP worker), P4 memory-safety hardening (T8.9/T8.10), and P5 controlled baseline benchmarking (T8.11).
