# AInfer — Project Status: Intel Core Ultra 7 258V (as of 2026-09-19, Performance Optimized: 34.88 tok/s; 42/62 done)

## What it is

Specialized inference runtime migration for **Tiel-Coder-35B-A3B-Genesis-Hermes** (Qwen3.5-MoE architecture fine-tune, `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`; sparse MoE hybrid: 30 DeltaNet-style linear-attention + 10 full-attention, 256 routed experts / 8 active, batch size 1, text-only) targeting the **Intel Core Ultra 7 258V** (Arc 140V integrated GPU, 32 GB unified LPDDR5X memory) under **CachyOS**.

Baseline inherited from Intel Arc Pro B60 production implementation (`06f267e`).

---

## Overall Health: Green (42/62 done; Phases 0/3/4/5/6/7 passed, Gates M0/M2b/M3/M4/M5/M6 signed off 2026-09-18, Performance Optimized 2026-09-19)

| Phase | Milestone | Scope | Status |
|---|---|---|---|
| **Phase 0** | M0: Migration Contract | Scope, identifiers, feasibility, acceptance gates | ✅ Done (6/6) |
| **Phase 1** | M1: Platform Ready | CachyOS toolchain, L0 probe, unified memory, contention | `[~]` In Progress (2/8; T1.1, T1.3 done) |
| **Phase 2** | M2a: Model Manifest | SafeTensors headers, manifest, MoE inventory, budget | `[~]` In Progress (4/5; T2.5 open with T5.1 waiver) |
| **Phase 3** | M2b: MoE Container | MoE `.binfer` spec, quantizer, Python/C++ loader, rejection | ✅ Done (6/6) |
| **Phase 4** | M3: Kernels Correct | Deterministic router, expert shootout, INT4 GEMV, DeltaNet | ✅ Done (7/7) |
| **Phase 5** | M4: Unified Runtime | Single-process arena manager, in-memory prefill→decode | ✅ Done (6/6; 34.88 tok/s, 0 KB growth × 10 runs) |
| **Phase 6** | M5: Quality Qualified | 200-case corpus, teacher-forced agreement, needle tests | ✅ Done (6/6; 187/200 passed 93.5%, BF16 KV default) |
| **Phase 7** | M6: Performance Ready | Independent timing, thermal steady state, roofline model | ✅ Done (5/5; 34.88 tok/s sustained, beats llama.cpp Vulkan by 1.19x) |
| **Phase 8** | M7: Service Candidate | In-process HTTP daemon, request queue, cancellation | `[ ]` Pending (0/4) — current focus |
| **Phase 9** | Hardening | Typed spans, execution guards, ASan/UBSan, fuzzing | `[ ]` Pending (0/5) |
| **Phase 10**| Deferred Scope | MTP speculative decoding, vision encoder | `[-]` Deferred (0/2) |

---

## Target Architecture & Production Path

- **Execution Model:** Single-process unified runtime; chunked prefill writes directly into decode-layout KV and DeltaNet state buffers (zero disk re-uploads).
- **Decode Loop:** 40-layer recorded Level Zero command lists with dynamic parameters managed via fixed device-mapped control buffers.
- **Quantization:** INT4 symmetric group-128 weights with BF16 scales; high precision for router gates and normalization layers.
- **Context Tiers:** Staged release across 4K (Tier 1) $\to$ 16K (Tier 2) $\to$ 32K (Tier 3) $\to$ 64K (Tier 4 stretch).

---

## Immediate Next Actions (Phase 8 + Phase 1 remainder)

1. **T8.1:** Resident in-process HTTP daemon maintaining resident model weights and recorded command lists across requests (OpenAI-compatible SSE streaming `/v1/chat/completions`).
2. **T8.2–T8.4:** Request queueing, cancellation, timeout, health/readiness endpoints, and 100-request leak audit.
3. **Phase 1 remainder (M1 still open):** T1.2 container/rollback, T1.4 ESIMD audit, T1.5 allocation-policy report, T1.6 contention benchmark, T1.7 bandwidth profile, T1.8 `258v` CTest preset — plus standalone T2.5 allocation utility (currently waived via T5.1 proof).
