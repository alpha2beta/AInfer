# AInfer — Project Status (as of 2026-09-16, Stamp 13)

## What it is

A custom Intel Arc Pro B60 (24 GB VRAM) inference engine for **Qwen3.8-27B** (text-only, 4K context), using SYCL/ESIMD/Level Zero. Target: beat llama.cpp SYCL baseline (~14.86 tok/s).

---

## Overall Health: Green

| Phase | Scope | Status |
|---|---|---|
| 0 — Environment & Foundations | T0.1–T0.5 | ✅ Done (5/5) |
| 1 — Model Spec & Reference | T1.1–T1.6 | 🔶 Partial (5/6 — T1.5 quality suite open) |
| 2 — Container & Exporter | T2.1–T2.6 | ✅ Done (6/6) |
| 3 — Kernel Microbenchmarks | T3.1–T3.9 | ✅ Done (9/9) |
| 4 — End-to-End Runtime | T4.1–T4.5 | ✅ Done (5/5) |
| 5 — Static Scheduling & Memory | T5.1–T5.6 | ✅ Done (6/6) |
| 6 — Performance Tuning | T6.1–T6.4 | ✅ Done (4/4) |
| 7 — Optional Features | T7.1–T7.4 | 🔶 Partial (3/4 — T7.2 MTP deferred) |
| X — Cross-cutting | X1–X2 | ✅ Done (2/2) |

**CTest:** 47/47 green (Stamp 13, 2026-09-16, ~1133 s)

---

## Production Runtime

- **Default decode:** `decode_l0` — 66 recorded L0 command lists, zero per-token allocation/construction. Token-identical to streamed-CPU BF16 greedy on tested prompts.
- **64K chunked prefill → decode (single binary):** `MERGE-OK` — in-process 64K needle 5/5 HIT, zero re-uploads, zero disk.
- **HTTP API:** Demo-grade only (T7.3) — per-request process + single-flight lock.

## Key Numbers

| Metric | Value |
|---|---|
| Short-context decode | ~14.7 tok/s (≈ llama.cpp parity) |
| Sustained P64/G64 | ~10.6 tok/s |
| Quality (greedy top-1) | 47/48 across 6 prompts |
| INT8-KV top-1 preservation | 53/53 |
| 64K needle retrieval | 5/5 HIT |

---

## Phase 8 — Release Hardening (Current/Open Work)

| Task | Priority | Status |
|---|---|---|
| **T8.1** STATUS.md | P0 | ✅ Done (2026-09-16) |
| **T8.2** Strict JSON report validation | P0 | ✅ Done (91/91 parse, `reports_json` ctest gate) |
| **T8.3** Weight-stats anomaly classification | P0 | ✅ Done (136/136 classified) |
| **T8.4** Tokenizer golden suite expansion | P0 | ✅ Done (27/27 parity, 25/25 golden) |
| **T8.5** 200+ case quality benchmark | P1 | 🔴 Open |
| **T8.6** Margin-aware divergence report | P1 | 🔴 Open |
| **T8.7** Single-binary prefill→decode merge | P2 | ✅ Done (2026-09-15) |
| **T8.8** Persistent HTTP worker | P3 | 🔴 Open |
| **T8.9** Typed arena spans/checked offsets | P4 | 🔴 Open |
| **T8.10** Fault injection & sanitizer track | P4 | 🔴 Open |
| **T8.11** Controlled AInfer vs llama.cpp benchmark | P5 | 🔴 Open |

---

## Deferred

- **T7.2 MTP speculation** — re-deferred 2026-09-15. Prompt-lookup alpha ≈ 0 (dead end). MTP-head gives ~2.5× only at 64K. Reopens on: production 64K traffic + chained-draft proof + 64K alpha confirmation.
- **Hybrid attention** (`AINFER_ATTN=hybrid`) — validated, env-gated, +25% decode speed, but deferred as default until launch-count reduction proven first.

---

## Summary

All P0 release-hardening tasks are now done. The project is engineering-complete with no open blocking scope. Next priorities are the P1 quality tasks (T8.5 200+ benchmark, T8.6 divergence report) and optionally the P3 persistent HTTP worker (T8.8).
