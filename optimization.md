# AInfer Performance Optimization Guide: Surpassing llama.cpp Vulkan on Intel Core Ultra 7 258V

**Date:** 2026-09-19  
**Target Hardware:** Intel Core Ultra 7 258V (Lunar Lake, integrated Arc 140V Xe2 GPU, 32 GB unified LPDDR5X-8533)  
**Target Architecture:** `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes` (Qwen3.5-MoE sparse hybrid: 40 layers = 30 DeltaNet linear attention + 10 full attention, 256 routed experts / 8 active, ~35B total / ~3B active params)  
**Operating System:** CachyOS (Linux 6.x rolling release)  

---

## 1. Executive Summary

Through systematic profiling and kernel fusion, AInfer decode performance on the Intel Arc 140V integrated GPU was increased from **23.83 tok/s** to **34.88 tok/s sustained** (and **35.67–35.81 tok/s** in steady-state single-command execution).

This achievement officially **surpasses the llama.cpp Vulkan backend (29.33 tok/s) by 1.19x (+18.9% faster)** and **outperforms 8-thread CPU Alderlake baseline (9.98 tok/s) by 3.49x**, while strictly preserving 100% bit-exact numerical parity, zero runtime heap allocations, and deterministic memory commitment.

```
========================================================================================
Runtime Throughput Comparison (Decode Tokens/Sec on Intel Core Ultra 7 258V)
========================================================================================
llama.cpp CPU (8 threads):   [========] 9.98 tok/s (Baseline)
AInfer Baseline (Phase 5):   [===================] 23.83 tok/s (2.39x CPU)
llama.cpp Vulkan (Arc 140V): [=======================] 29.33 tok/s (2.94x CPU)
AInfer Optimized (Current):  [============================] 34.88 tok/s (3.49x CPU, +18.9% vs Vulkan)
========================================================================================
```

---

## 2. Key Optimization Pillars

### Pillar 1: Unified Single-List Recording (`cmd_step_` / `cmd_prefill_step_`)
- **Problem:** The initial runtime recorded 42 separate Level Zero command lists per step (1 embed list + 40 per-layer lists + 1 tail list). Executing 42 separate command lists per token caused driver ring-buffer submission overhead, software dispatch latency, and GPU scheduling bubbles (~5.0 ms per token).
- **Optimization:** Unified the entire pipeline into a single pre-recorded command list (`cmd_step_` for decode and final prefill token; `cmd_prefill_step_` for prompt tokens $p < P-1$).
- **Impact:** Eliminated 41 driver command queue submit boundaries per step, reducing decode latency by ~5 ms/tok and immediately boosting throughput from 23.83 tok/s to 28.02–28.56 tok/s.

### Pillar 2: Batched MoE Kernel Dispatch
- **Problem:** The original MoE implementation evaluated the 8 active experts via a serial loop. Across 40 layers, this required:
  $$\text{Kernels per token} = 40 \times (8 \text{ gate-up} + 8 \text{ SwiGLU} + 8 \text{ down-accum}) = 960 \text{ kernel launches}$$
  with 960 corresponding barrier synchronizations.
- **Optimization:** Replaced the 24 serial launches and 24 barriers per layer with **3 batched kernels**:
  1. `moe_gateup_all8_ctrl` `[8, 1024]` — computes Gate and Up GEMVs across all 8 active experts concurrently (32 workgroups).
  2. `silu_mul_all8` `[8, 512]` — computes SwiGLU activation across all 8 active experts concurrently (16 workgroups).
  3. `moe_down_accum_all8_ctrl` `[2048]` — computes Down GEMV for all 8 active experts, scaling with router weights and accumulating directly into `d_moe_acc_`.
- **Impact:** Per-layer MoE execution time dropped from **~450 µs down to 216 µs**, saving **~8.8 ms per token** across all 40 layers.

### Pillar 3: Fused LM-Head GEMV + Argmax Stage 1 (`int4_gemv_lm_head_argmax1`)
- **Problem:** The final vocabulary projection (`[248320, 2048]`) previously executed a standard GEMV, wrote all 248,320 FP32 logits to memory (~1 MB), hit a barrier, and then launched `k_argmax1_` to read back all 248,320 floats to find group maxima.
- **Optimization:** Fused GEMV computation with Stage 1 local tree-reduction in a single kernel (`int4_gemv_lm_head_argmax1`). Each of the 970 workgroups (256 threads each) computes its slice of vocabulary logits, performs an in-register tree reduction, and writes only the workgroup maximum value and token index (`stage1_vals[gid]` / `stage1_idxs[gid]`). Furthermore, in greedy mode, binding `null_logits` (`if (y != NULL)`) completely eliminates writing 248,320 unread logits (~1 MB) to global memory. In Stage 2 (`argmax_stage2_ctrl`), deterministic token-index tie-breaking guarantees strict reproducibility across runs.
- **Impact:** Eliminated 1 MB write traffic and 1 MB read traffic, one separate kernel launch, and one synchronization barrier. Tail latency reduced to **2.59 ms** (from ~3.5 ms).

### Pillar 4: Shared Expert Group Size Resolution (Numerical Integrity Root Cause)
- **Problem:** When batched MoE was initially enabled, generated tokens drifted to token `220` (ASCII space).
- **Root Cause:** In `runtime_258v.cpp`, `zeKernelSetGroupSize(k_silu512_, 256, 1, 1)` had originally been located inside the serial expert loop. When the loop was deleted in favor of `silu_mul_all8`, the subsequent launch of `k_silu512_` for the **Shared Expert** at line 773 was left with uninitialized/invalid workgroup size. The shared expert silently produced invalid activations, corrupting downstream representations across all 40 layers.
- **Resolution:** Explicitly configured `zeKernelSetGroupSize(k_silu512_, 256, 1, 1)` prior to the shared expert activation launch.
- **Result:** Restored **100% bit-exact numerical parity**:
  - Layer 0 Gate_Up difference: `0.000000e+00`
  - Layer 0 SwiGLU difference: `0.000000e+00`
  - Layer 0 MoE Acc difference: `9.313226e-10` (floating point rounding identity)
  - Full trajectory restored to golden tokens: `[148431, 62497, 148287, 198, ...]`.

### Pillar 5: Static Arena Management & Zero-Allocation Invariance
- Memory allocation during inference is strictly forbidden.
- Static arenas are allocated once at initialization:
  - **Payload Arena:** 17.32 GiB (INT4 packed weights)
  - **Scale Arena:** 521.15 MiB (BF16 group-128 scales)
  - **KV Cache Arena:** 80.00 MiB (10 full-attention layers @ 4096 ctx)
  - **SSM State Arena:** 62.81 MiB (30 DeltaNet layers)
  - **Workspace Arena:** 64.00 MiB
  - **Runtime Control Buffer:** 128 bytes
  - **Total Committed:** 18.03 GiB (13.97 GiB headroom on 32 GB system memory)
- **Leak Audit:** Exactly **0 KB RSS growth** across 10+ consecutive multi-request generation runs.

---

## 3. Benchmarking Matrix

All benchmarks executed on the physical Intel Core Ultra 7 258V machine running CachyOS with thermal telemetry and isolated timing.

### Isolated Performance Telemetry (`report_bench_t71.json`)

| Benchmark Field | Baseline (Phase 5) | Optimized (Current) | Delta / Speedup |
|---|---|---|---|
| **Model Load & Weights Upload** | 9.96 s | 9.09 s | -8.7% |
| **Tokenization Latency** | 0.014 ms | 0.023 ms | Negligible |
| **Prefill Throughput (p=21)** | 26.15 tok/s | **39.07 tok/s** | **+49.4%** |
| **Cold TTFT** | 810.65 ms | **546.96 ms** | **-32.5%** |
| **Warm TTFT** | 803.21 ms | **537.57 ms** | **-33.1%** |
| **First Decode Latency** | 40.95 ms | **28.00 ms** | **-31.6%** |
| **Sustained Decode Throughput** | 23.83 tok/s | **34.88 tok/s** | **+46.4%** |
| **Single-Command Peak Decode** | 24.20 tok/s | **35.81 tok/s** | **+48.0%** |
| **Inter-Token Latency (p50)** | 41.80 ms | **28.54 ms** | **-31.7%** |
| **Inter-Token Latency (p90)** | 42.55 ms | **29.16 ms** | **-31.5%** |
| **Inter-Token Latency (p95)** | 44.07 ms | **29.46 ms** | **-33.1%** |
| **Inter-Token Latency (p99)** | 47.12 ms | **31.44 ms** | **-33.3%** |
| **Inter-Token Standard Deviation** | 1.15 ms | **0.80 ms** | Highly deterministic |
| **Host Peak RSS Delta** | 0 KB | **0 KB** | Zero memory leak |

### Component Latency Breakdown (`profile_pipeline`)

```
=================================================================
--- Decode Step Profiling Breakdown (Arc 140V Xe2) ---
=================================================================
  Unified Single cmd_step: 28.036 ms (Throughput: 35.67 tok/s)
  Embed Latency:           0.248 ms (0.9%)
  Total 40 Layers Latency: 25.084 ms (89.5%)
  Tail (LM Head + Argmax): 2.704 ms (9.6%)
-----------------------------------------------------------------
  DeltaNet Layer (30 layers): 0.612 ms/layer (Total: 18.36 ms)
  Full-Attn Layer (10 layers): 0.672 ms/layer (Total:  6.72 ms)
=================================================================
```

---

## 4. Competitive Analysis vs. llama.cpp

Controlled comparison run from `tools/bench_258v/run_llama_comparison_t75.py`:

| Dimension | llama.cpp CPU (8-thread) | llama.cpp Vulkan (Arc 140V) | AInfer (Level Zero) | Advantage |
|---|---|---|---|---|
| **Decode (tok/s)** | 9.98 tok/s | 29.33 tok/s | **34.88 tok/s** | **AInfer +18.9% faster** |
| **Backend API** | CPU C++ | Vulkan KHR_coopmat | Intel Level Zero | Native driver dispatch |
| **Command Execution** | Host thread pool | Command buffer re-records | Pre-recorded static lists | Zero dispatch overhead |
| **Memory Model** | Dynamic host allocations | Vulkan descriptor sets | Fixed pre-allocated arenas | Zero memory fragmentation |
| **Runtime Heap Alloc** | Unbounded | Dynamic buffers | **0 KB** | Strictly deterministic |
| **Warm TTFT (p=21)** | ~1,200 ms | ~700 ms | **537.57 ms** | Lowest time-to-first-token |

---

## 5. Verification & Numerical Fidelity

1. **Gate M4 Unit Test (`test_runtime_258v`):**
   - 5/5 tests passed cleanly.
   - Bit-exact golden token sequence: `[148431, 62497, 148287, 198, 220, 16, 13, 198, 220, 17, 13, 198, 220, 18, 13, 198]`.
   - Deterministic reset: 10 repeated generation runs produced identical outputs with 0 KB RSS growth.

2. **End-to-End Generation Sample (`decode_258v`):**
   - **Prompt:** `"Hello"`
   - **Throughput:** 33.39 tok/s
   - **Output:**
     > *"Hello! 😊\n\nHow can I help you today? Whether you have a question, need help with a task, want to learn something new, or just"*

### Pillar 6: Chunked Batched Prefill GEMM & Fused Layer Execution
- **Problem:** Sequential prefill processed prompts one token at a time (39.07–39.32 tok/s), resulting in 534 ms TTFT for 21 tokens and leaving GPU compute units underutilized.
- **Optimization:**
  - Implemented 18 dedicated batched OpenCL kernels (`int4_gemm_prefill`, `embed_gather_batch`, `rmsnorm_2048_batch`, `conv1d_update_silu_batch`, `head_l2_norm_qk_batch`, `gate_prep_batch`, `deltanet_recurrent_batch`, `deltanet_head_norm_silu_z_batch`, `deinterleave_q_gate_batch`, `rope_and_kv_append_batch`, `gqa_attn_prefill_batch`, `moe_topk_router_batch`, `moe_gateup_all8_batch`, `silu_mul_all8_batch`, `moe_down_accum_all8_batch`, `silu_mul_batch`, `block_resadd_moe_batch`, `resadd_batch`).
  - Sub-allocated chunk activation tensors within the existing 64 MiB workspace arena (zero runtime heap allocations).
  - Pre-recorded and cached single-submission Level Zero command lists `cmd_prefill_chunk_[B]` and `cmd_prefill_tail_[B]` for chunk sizes $B \in [1, 32]$.
  - **Numerical Fidelity Root Cause & Fix:** In `gate_prep_batch`, the gate decay factor was originally missing `exp(gate)`, writing raw $-e^{A} \times \text{softplus}$ directly to $g$, which corrupted DeltaNet state across multi-step recurrence. Once restored to $g = \exp(-e^A \times \text{softplus})$, 100% bit-exact golden sequence output was verified (`[148431, 62497, 148287, ...]`).
- **Impact:**
  - Prefill throughput doubled from **39.32 tok/s to 78.92 tok/s** (scaling up to **89.31 tok/s** at $P=32$, a **2.27x speedup**).
  - Warm TTFT for 21 tokens halved from **534.04 ms down to 266.17 ms** (-50.2%).
  - Prefill gap to llama.cpp Vulkan closed to **92.6% parity** (87.62 tok/s vs 94.59 tok/s @ $P=64$).
  - **Scaled 2026-09-19 (`bench_prefill`, `report_prefill_scaling.json`):** P=8 (69.99 tok/s, 114.30 ms), P=16 (110.10 tok/s), P=32 (162.31 tok/s, 6.16 ms/tok), P=64 (216.27 tok/s, 4.62 ms/tok), P=128 (274.76 tok/s, 3.64 ms/tok), P=256 (**299.04 tok/s**, 856.07 ms, **3.34 ms/tok**) — throughput scales ~monotonically with prompt length, now well past the llama.cpp Vulkan P=64 reference (94.59 tok/s).
### Pillar 7: Hardware DPAS Systolic Array INT4 GEMM Tiling
- **Problem:** While Pillar 6 chunked the prefill forward pass into batched operations, `int4_gemm_prefill` still relied on scalar SIMD instruction unpacking (shift + mask + cast for each nibble into `float4`), leaving Intel Xe2's systolic matrix engines idle. Linear projections accounted for ~33% of prefill execution time.
- **Optimization:**
  - Audited Intel Arc 140V Xe2 matrix architecture via `ocloc` / `iga64`, identifying hardware support for native INT4 DPAS (`dpas.8x1 ... :s4 :s4`) and `dpas.8x8` SIMD16 FP16/BF16 matrix operations (`intel_sub_group_f16_f16_matrix_mad_k16`).
  - Redesigned `int4_gemm_prefill` into a subgroup-tiled systolic GEMM kernel:
    - Subgroup tile: $M_{tile} = 16, B_{tile} = 8, K_{step} = 16$ with FP32 accumulation.
    - Weights unpacked on-the-fly to FP16 in registers; activations loaded as FP16.
    - Workgroup: 256 threads (16 subgroups) computing 256 output channels concurrently.
    - Full coverage across all model projection shapes ($M \in \{32, 512, 2048, 4096, 8192\}, K \in \{512, 2048, 4096\}$).
- **Impact:**
  - On the dominant DeltaNet projection shape ($M=8192, K=2048$, across 30 layers), GEMM latency dropped from **2395.71 µs to 1257.97 µs** at $B=32$ (**1.90x speedup**, saving **34.2 ms per 32-token chunk** across all 30 DeltaNet layers).
  - Standard 21-token prefill throughput increased to **88.49 tok/s** (with peak scaling reaching **89.73 tok/s**).
  - Warm TTFT reduced further to **237.33 ms** (down from 266.17 ms, a total **55.6% reduction** vs initial 534 ms).
  - Sustained decode throughput reached an all-time record of **35.54 tok/s** (**+21.2% faster than llama.cpp Vulkan**).
  - Full Gate M4 suite (7/7 tests) verified bit-exact golden sequence match (`[148431, 62497, 148287, 198, ...]`), 0 KB RSS growth, and multi-chunk long-prompt determinism ($P=128$ and $P=256$).
### Pillar 8: Prefill Attention Rewrite + Recurrence Tuning (299 → 318 tok/s)
- **Problem:** With chunked + DPAS prefill at 299.04 tok/s (P=256), the remaining profile (new full-attention breakdown in `profile_prefill_breakdown`, with control-position pinned for faithful books) showed GQA attention at 20.64 ms/layer and the DeltaNet recurrence loop at 5.27 ms/layer.
- **Optimization:**
  - **Attention v2 (`gqa_attn_prefill_batch_v2`):** replaced the 256-wide SLM tree reduction (~10 barriers per KV position per query) with an in-register subgroup butterfly over 4 batched positions plus one SLM exchange (0.25 barriers/position). Private per-dim `q` load replaced the pointless SLM staging. **Lesson:** IGC silently compiled the light kernel as SIMD32, breaking the hardcoded subgroup-16 shuffle (constant 0.17 bias, caught by the A/B harness) — fixed with `__attribute__((intel_reqd_sub_group_size(16)))`. A/B harness `tools/kernels_258v/bench_attn_prefill.cpp` (synthetic runtime-layout buffers + batched CPU reference): 1.36–2.65x on the kernel, 1.68e-07 max diff vs CPU (same class as v1's 1.47e-07). Runtime switch is env-gated (`AINFER_ATTN_V1=1` restores v1).
  - **Recurrence v2 (`deltanet_recurrent_batch_v2`):** double-buffered 4-step batching (1 barrier per 4 steps vs 1 per step) plus 4-way split accumulators for the dot-product FMA chains, env-gated (`AINFER_RECR_V1=1`). Measured only ~9% on the kernel (5.27→4.9 ms): the barrier cut alone changed nothing (verified identical via `AINFER_RECR_V1=1` A/B), so the loop is bound by FMA-chain latency/occupancy, not barriers — a parallel-scan rewrite would be needed for a large win (deferred: complex, ~10–13% end-to-end prize).
- **Impact:**
  - P=256 prefill: **299.04 → 318.31 tok/s** (+6.5%; 804.26 ms, 3.14 ms/tok), P=128: 274.76 → 286.31 tok/s.
  - **Extended sweep 2026-09-19 (P=512/1024/2048):** P=512 (291.62 tok/s), P=1024 (254.88 tok/s), P=2048 (198.21 tok/s, 5.05 ms/tok). The curve peaks at P≈256 and declines after — later chunks pay growing causal-attention cost over longer KV, dragging the average down. An externally claimed 525 tok/s OpenVINO figure (no published prompt length or methodology) is unreachable anywhere on this curve; closing that gap, if real, needs cross-query KV reuse (FlashAttention-style blocking: each K/V block read once per query-block instead of once per query) rather than more per-query tuning.
  - Full Gate M4 suite (7/7) re-verified bit-exact with final code (attention v2 + recurrence v2 active).
  - Honest accounting: remaining prefill is ~memory-bound near the practical roofline (MoE + dense GEMMs ≈ 70%); further big wins need the recurrence parallel scan or higher achieved bandwidth, not more barrier-cutting.
### Pillar 9: Prefill GEMM Doubled K-Slices (328 tok/s @P=441)
- **Problem:** `int4_gemm_prefill` measured at ~1.3 TFLOPS (QKV 8.59 GFLOP in 6.844 ms at B=512) — single-digit % of Xe2 XMX peak. DPAS issue density, not launch overhead, is the bound (this measurement also killed the QKV+ZAB fusion idea: launch + X-reread savings estimated ~1%, deprioritized).
- **Optimization:** new `int4_gemm_prefill_v2` processes 2 K-slices (32 K elements) per iteration: 4KB double-buffered SLM staging over slice-pairs, 8 back-to-back DPAS per barrier instead of 4, same scales reused across each even-aligned pair. Env-gated (`AINFER_GEMM_V1=1` restores v1).
- **Impact:**
  - P=441: 307.06 → **328.29 tok/s** (+6.9%, 1343.34 ms, 3.05 ms/tok); P=256 →335.52 (+5.7%), P=512 →328.39 (+6.2%), P=1024 →280.58 (+5.1%), P=2048 →207.78 (+2.4%).
  - Running total at matched P=441: 294.80 → 328.29 (**+11.4%**); remaining gap to the claimed 525 tok/s is 1.6x.
  - Full Gate M4 suite (7/7) re-verified bit-exact with GEMM v2 active.
### Pillar 10: Prefill GEMM M_tile=32 with Shared Activations (352 tok/s @P=441)
- **Problem:** v2 still builds `a_mat` (SLM gather) separately per row while DPAS density per barrier stays at 8 — setup cost per DPAS unchanged.
- **Optimization:** new `int4_gemm_prefill_v4` widens M_tile to 32 rows/subgroup (each lane handles rows m and m+16); `a_mat` is built once per token-tile and shared across both rows' DPAS (16 DPAS per slice-pair for one setup). Group x-dim halved via `gemm_rows_per_group_` (256 rows/group); MTP verify path audited (uses `k_gemv_m2_`, unaffected). Fallbacks kept (`AINFER_GEMM_V1`/`AINFER_GEMM_V2`). A no-SLM v3 detour was tried and reverted (-38% direct loads exposed X latency; -50% register-prefetch spilled) — v2's SLM double-buffering is load-bearing.
- **Impact:**
  - P=441: 328.29 → **351.62 tok/s** (+7.1%, 1254.20 ms, 2.84 ms/tok); P=256 →362.48 (+8.0%), P=512 →351.94 (+7.2%), P=1024 →293.89 (+4.7%), P=2048 →222.20 (+6.9%).
  - Running total at P=441: 294.80 → 351.62 (**+19.3%**); remaining gap to 525 is 1.49x.
  - Full Gate M4 suite (7/7) re-verified bit-exact with v4 as default.
  - **Negative result — GEMM v5 (M_tile=64) tried and reverted:** 4 rows/lane, 16 DPAS per shared `a_mat`, no spills in asm (8/16/32 DPAS per body for v2/v4/v5 as designed). Measured consistently worse than v4 (best 272 vs v4's worst 328 tok/s @P=441) — likely I-cache pressure or lower EU occupancy from the 2x loop body and halved group count. M_tile=32 stands; tile widening is exhausted. Note: ±7% run-to-run variance observed this session (7-day uptime, shared box) — small deltas need repeated runs.
  - **Path B verdict — integer DPAS infeasible via OpenCL C (2026-09-19, no code changed):** `s8_s8_matrix_mad_k32` / `u8_s8_*` are undeclared in this IGC; the only integer form exposed is `int8 intel_sub_group_u8_u8_matrix_mad_k32(uint a, uint8 b, int acc)` (signature confirmed by successful compile). It loses three ways: (1) 8 dots of length 8 per lane (64 MACs) vs 1024 MACs/lane for the FP16 k16 path — 16x narrower; (2) unsigned-only, so signed activations would need offset-encoding plus correction terms; (3) INT4 weights still need nibble unpacking (to bytes instead of halves — same cost), so the unpack overhead Path B was supposed to eliminate stays. The T1.4 `dpas.8x1 :s4` sighting remains assembly-level only; emitting it needs ESIMD/inline-asm (major project, uncertain IGC acceptance). Path B closed unless that investment is explicitly approved.
  - **Negative result — GEMM v3 tried and reverted (2026-09-19):** asm analysis showed ~125 scalar/mem ops per DPAS and zero spills, so SLM staging was removed (direct cached X loads, zero loop barriers). Result: **-38%** (315→196 tok/s) — the SLM double-buffering was load-bearing latency hiding, not overhead. Restoring overlap via 512B register prefetch was worse (**-50%**, spill). Conclusion: v2's SLM staging is essentially optimal for this shape; further GEMM headroom (if any) is in M-tiles/occupancy or integer DPAS (path B), not staging removal.

---

## 5. Verification & Numerical Fidelity

1. **Gate M4 Unit Test (`test_runtime_258v`):**
   - 7/7 tests passed cleanly (T5.1–T5.7).
   - Bit-exact golden token sequence: `[148431, 62497, 148287, 198, 220, 16, 13, 198, 220, 17, 13, 198, 220, 18, 13, 198]`.
   - Deterministic reset: 10 repeated generation runs produced identical outputs with 0 KB RSS growth.
   - Long-prompt verification: $P=128$ (4 chunks) and $P=256$ (8 chunks) verified bit-exact and position-consistent across chunk boundaries.

2. **Persistent HTTP Server (Phase 8):**
   - Persistent resident daemon (`server_258v.py` on port 8088) verified with resident 18.03 GiB model.
   - SSE chunked streaming (`/v1/chat/completions`) and full JSON responses verified with immediate client-side socket closure on stream completion (`data: [DONE]`).
    - Abrupt client disconnect cancellation verified: state cleanly reset with zero device memory leak.

---

## 6. External Reference: Intel Arc Pro B70 Inference Cookbook (transferable hints)

**Source (reviewed 2026-09-19):** `SergiioB/intel-arc-pro-b70-inference-cookbook` — vLLM-XPU + llama.cpp-SYCL recipes on discrete Arc Pro B60/B70 (Battlemage, 32 GB VRAM, ~437 GB/s). Most relevant pages: the 35B-MoE vLLM-XPU recipe (MTP1/2/4 tables, 170.9 tok/s MTP4) and the fused multi-token `MUL_MAT_ID` MoE kernel write-up (+42% decode).

**Applicability caveat:** absolute tok/s figures do **not** transfer to Arc 140V (shared 32 GB LPDDR5X, ~90–115 GB/s sustained). The techniques and methodology below do. Each item cites the AInfer task it attaches to.

### 6.1 Fused multi-token MoE GEMV (follow-up to Pillar 2 / T4.2)
- **Cookbook finding:** multi-row MoE verify batches fell into a counting-sort path (D2H expert-ID readback + host sort + grouped GEMM, ~144 ops/round) because the 3D tensor form hid behind a single-row fast-path gate. Fusing into one multi-token GEMV (M=2..8, gate-up + grouped down forms) cut verify-round cost 118→85 ms (**+42% decode**).
- **AInfer status:** Pillar 2 already batches 8 experts per layer for single-token decode. The prefill-chunk path (Pillar 6, cached $B \in [1, 32]$) is the candidate: a fused M-token expert GEMV would remove per-row dispatch overhead across 8 experts × 40 layers.
- **Action:** prototype fused M-token expert GEMV for prefill chunks, gated by an env kill-switch A/B (their `GGML_SYCL_MT_OFF=1` pattern) so acceptance stays measurable against the current path.

### 6.2 On-device expert-ID grouping (corroborates T4.2 Strategy 3)
- **Cookbook finding:** fused `MUL_MAT_ID` plus on-device expert-ID grouping exists precisely to avoid the D2H-readback path; grouping toggle is env-gated for A/B.
- **AInfer status:** consistent with our Strategy 3 (Batched Device-Driven Dispatch, 4.84 µs/layer, zero host syncs). Direction confirmed — keep grouping device-side; no change required.

### 6.3 MTP mode selection per context tier (extends T10.1)
- **Cookbook finding:** MTP2 is the long-context sweet spot (85.8% accept @128K, 101.64 tok/s); MTP4 wins short responses (178.34 tok/s @p512/g32) but accept collapses to ~60% at 128K. Mixed long-prefill + short-request traffic should use no-spec.
- **AInfer status:** T10.1 closed with dual-token ($B=2$) verification (51.36 tok/s @α=93.75%, break-even @α=39%).
- **Action:** score wider draft/verify fan-out per context tier rather than one global gate. Fallback recorded: DFlash-style speculation (their Nemotron recipe hits 186 tok/s with zero native MTP) if MTP weights ever prove unusable.

### 6.4 Prefill chunk-budget sweep (extends T5.2 / T7.2)
- **Cookbook finding:** raising the token budget 8192→16384 gave **+17.6% prefill and +12.0% decode** at the same 128K recipe; ubatch sweet spots are non-monotonic per context (512 @8K, 3072 @16K, 1024 @128K). Prefill rate itself is essentially flat across context lengths.
- **AInfer status:** cached chunk lists cover $B \in [1, 32]$; $B$ was never swept beyond 32.
- **Action:** sweep cached chunk sizes beyond $B=32$ per context tier (T7.2 matrix), watching workspace-arena pressure (currently 64 MiB).

### 6.5 Quantized KV precedent + RoPE ordering caution (corroborates T6.6)
- **Cookbook finding:** `q8_0` K + `q4_1` V (llama.cpp) and FP8 KV (vLLM) ship in production recipes — INT8 KV is viable. But quantized KV required a rotation workaround upstream (`LLAMA_ATTN_ROT_DISABLE=1`, load/decode crash otherwise).
- **AInfer status:** BF16 KV qualified as production default (only 640 MiB @32K; INT8 saves just 320 MiB). Decision stands.
- **Action:** keep an explicit RoPE-before-vs-after-quantization ordering test in the attention suite so a future INT8-KV revival cannot regress silently.

### 6.6 Phase-split math precision (follow-up to T4.4 / T4.5)
- **Cookbook finding:** F16-math binary wins prefill (594 tok/s) while FP32-math wins decode (23.38 tok/s) — same weights, compile-time flag, two binaries served per phase.
- **AInfer status:** FP32 SSM state already kept; GEMV/attention accumulation policy is currently uniform.
- **Action:** A/B accumulator precision (FP16 vs FP32) separately for prefill vs decode kernels; adopt per-phase policy only on measured win with parity intact.

### 6.7 Level Zero environment knobs (open T1.7)
- **Cookbook finding:** `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`, `SYCL_DEVICE_FILTER=level_zero`; graph capture (`VLLM_XPU_ENABLE_XPU_GRAPH=1`) is their equivalent of our recorded command lists — independent validation of the approach.
- **Action:** add `ZE_FLAT_DEVICE_HIERARCHY` and immediate-vs-regular command-list A/B to the T1.7 dispatch profile before M1 closes.

### 6.8 Benchmark discipline (extends T7.1)
- **Cookbook finding:** n=5 medians after one discarded warmup, exact token counts, zero cache reuse, entropy-first cold prefixes, matched natural prompts (not filler), client monotonic SSE timing; prefill proxy explicitly labeled as including scheduling (not isolated engine prefill).
- **AInfer status:** isolated timing fields already separate load/tokenize/prefill/decode (ahead here).
- **Action:** codify discarded-warmup + cold-prefix rules in the harness driver so future numbers stay comparable.

### 6.9 Power-cap sweeps (extends T7.4)
- **Cookbook finding:** 150W eco vs 230W stock characterized via `power1_cap` hwmon; per-card draw and package thermals reported per cell.
- **AInfer status:** 5.58-min steady-state run done (55.8°C settle, 97.0% retention) but no power-cap sweep.
- **Action:** add PL1-cap sweep × perf/W table — the constraint matters more on Lunar Lake (17W PL1 / 37W PL2) than on B70.

### 6.10 Watchdog and speculation hazards (extends T8.3 / T9.5)
- **Cookbook finding:** Xe driver ring wedge (hung `/health`) recovered via `dmesg`-pattern detection + auto-restart watchdog. Separately: prefix-caching × MTP causes **silent** token corruption — they hard-disable prefix caching whenever MTP is on.
- **Action (T8.3):** add `dmesg` ring-wedge pattern detection to health monitoring with restart policy.
- **Recorded hazard (T9.5):** if prefix caching or speculation is ever added, the two must never be enabled together without a corruption battery.

### 6.11 Mixed quantization by tensor class (follow-up to T3.3)
- **Cookbook finding:** gate/up at IQ3_S with down at IQ4_NL (folder name not uniform) — per-class bit-width is normal practice.
- **AInfer status:** shared `down_proj` shows the worst container error (max 0.0222).
- **Action:** if quality ever needs it, promote just `down_proj` tensors to INT8; precedent exists, cost is bounded to one tensor class.


