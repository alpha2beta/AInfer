# AInfer — Project Status: Intel Core Ultra 7 258V (as of 2026-09-20, Baseline Stamp 3: 47/62 done, Gate M7 flagged)

## What it is

Specialized inference runtime migration for **Tiel-Coder-35B-A3B-Genesis-Hermes** (Qwen3.5-MoE architecture fine-tune, `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`; sparse MoE hybrid: 30 DeltaNet-style linear-attention + 10 full-attention, 256 routed experts / 8 active, batch size 1, text-only) targeting the **Intel Core Ultra 7 258V** (Arc 140V integrated GPU, 32 GB unified LPDDR5X memory) under **CachyOS**.

Baseline inherited from Intel Arc Pro B60 production implementation (`06f267e`).

---

## Overall Health: Yellow (47/62 done; Phases 0/3/4/5/6/7 passed, Gates M0/M2b/M3/M4/M5/M6 signed off; Gate M7 flagged for re-verification at Stamp 3, 2026-09-20 — see `STATUS.md` §4/§7)

| Phase | Milestone | Scope | Status |
|---|---|---|---|
| **Phase 0** | M0: Migration Contract | Scope, identifiers, feasibility, acceptance gates | ✅ Done (6/6) |
| **Phase 1** | M1: Platform Ready | CachyOS toolchain, L0 probe, unified memory, contention | `[~]` In Progress (3/8; T1.1, T1.3, T1.4 done) |
| **Phase 2** | M2a: Model Manifest | SafeTensors headers, manifest, MoE inventory, budget | `[~]` In Progress (4/5; T2.5 open with T5.1 waiver) |
| **Phase 3** | M2b: MoE Container | MoE `.binfer` spec, quantizer, Python/C++ loader, rejection | ✅ Done (6/6) |
| **Phase 4** | M3: Kernels Correct | Deterministic router, expert shootout, INT4 GEMV, DeltaNet | ✅ Done (7/7) |
| **Phase 5** | M4: Unified Runtime | Single-process arena manager, in-memory prefill→decode | ✅ Done (6/6; 35.54 tok/s decode committed, prefill 335.52 tok/s @P=256 (328.29 @P=441), 0 KB growth × 10 runs) |
| **Phase 6** | M5: Quality Qualified | 200-case corpus, teacher-forced agreement, needle tests | ✅ Done (6/6; 187/200 passed 93.5%, BF16 KV default) |
| **Phase 7** | M6: Performance Ready | Independent timing, thermal steady state, roofline model | ✅ Done (5/5; 35.54 tok/s committed, beats llama.cpp Vulkan by 1.21x — see uncommitted-regression note in `STATUS.md`) |
| **Phase 8** | M7: Service Candidate | In-process HTTP daemon, request queue, cancellation | `[~]` In Progress (3/4; T8.4 leak audit FAILED per committed evidence, re-run needed) |
| **Phase 9** | Hardening | Typed spans, execution guards, ASan/UBSan, fuzzing | `[ ]` Pending (0/5) |
| **Phase 10**| Deferred Scope | MTP speculative decoding, vision encoder | `[~]` In Progress (T10.1 single-token done; dual-token uncommitted; T10.2 deferred) |

---

## Target Architecture & Production Path

- **Execution Model:** Single-process unified runtime; chunked prefill writes directly into decode-layout KV and DeltaNet state buffers (zero disk re-uploads).
- **Decode Loop:** Unified recorded Level Zero command lists with dynamic parameters managed via fixed device-mapped control buffers.
- **Quantization:** INT4 symmetric group-128 weights with BF16 scales; high precision for router gates and normalization layers.
- **Context Tiers:** Staged release across 4K (Tier 1) $\to$ 16K (Tier 2) $\to$ 32K (Tier 3) $\to$ 64K (Tier 4 stretch).

---

## Immediate Next Actions (Stamp 3 follow-ups)

1. **T8.4 re-run:** the committed HTTP stress evidence (`report_http_stress.json`) reads `status: FAILED` with only 20/100 requests and a confirmed 6.5 MB RSS growth — re-run the harness at its actual 100-request default and commit a genuine PASSED/zero-leak result.
2. **Root-cause and commit the dual-token MTP work:** `int4_gemv_m2`/`--speculative` code and its regenerated benchmarks (`report_bench_t71.json`, `report_llama_comparison.json`) sit uncommitted with an unresolved 14% decode regression (35.54→30.48 tok/s) and 122% slower model load — investigate before committing.
3. **Phase 1 remainder (M1 still open):** T1.2 container/rollback, T1.5 allocation-policy report, T1.6 contention benchmark, T1.7 bandwidth profile, T1.8 `258v` CTest preset — plus standalone T2.5 allocation utility (currently waived via T5.1 proof).
4. **Phase 9 hardening** (T9.1–T9.5) once the above are resolved.
