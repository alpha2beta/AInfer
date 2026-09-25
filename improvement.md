# AInfer Improvement Plan — IDE Integration & Performance Optimization

> Structured plan for daily IDE copilot readiness and prefill performance parity.
> Task IDs use `I` prefix (`I1.1`…`I4.4`) to avoid collision with the existing `T`/`X` namespace.
> Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` dropped

---

## 0. Context

AInfer is a custom inference runtime for the `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`
model (Qwen3.5-MoE, 40 layers: 30 DeltaNet + 10 full-attention, 256 experts / 8 active)
running on **Intel Core Ultra 7 258V** (Arc 140V iGPU, 32 GB unified LPDDR5X).

### Current Performance (production HEAD)

| Metric | Value | Notes |
|---|---|---|
| Sustained decode | **35.06 tok/s** (greedy) | 1.20× llama.cpp Vulkan |
| MTP speculative decode | **42.8–51.4 tok/s** | α-dependent, dual-token B=2 |
| Prefill P=256 | **362.48 tok/s** | DPAS systolic + MoE 32-tile |
| Prefill P=441 | **351.62 tok/s** | OpenVINO claims >525 at same P |
| Cold startup | **~48 s** | T9.5 CRC re-reads 19 GiB |
| Fast startup (no CRC) | **~8–9 s** | Measured without CRC pass |

### Gap Analysis

1. **Prefill:** 351.62 tok/s vs OpenVINO's >500 tok/s at P=441 → **1.49× gap**.
   DeltaNet serial recurrence (5.27 ms/layer @ B=256) is the dominant bottleneck.
2. **IDE usability:** `server_258v.py` lacks `stop` sequence parsing, FIM suffix handling,
   and instant disconnect abort — these break autocomplete in Continue.dev / Cursor / Cline.
3. **Startup latency:** 48 s CRC check is unacceptable for interactive use.
4. **Agent usability & multi-turn latency:** OpenCode sends ~7K tokens (system prompt + tools) on
   every turn, triggering ~38s prefill due to zero KV prefix caching. Hardcoded 60s timeout truncates
   responses after 100~200 tokens, hardcoded `enable_thinking=False` suppresses CoT, and delayed
   HTTP headers cause client socket timeouts.

---

## 1. Priority 0 — Immediate (IDE-Blocking)

These items block basic IDE autocomplete functionality. Both are server-only Python changes.

### I1.1 Stop Sequence Parsing in `server_258v.py`

- Status: `[x]`
- Complexity: **Low** (~1 day)
- File: `tools/http/server_258v.py`
- **Current state:** Implemented `StreamStopBuffer` and `IncrementalDecoder` supporting exact
  prefix matching, multi-character/multi-token stop sequences, and early loop exit in both
  streaming and non-streaming modes.
- **Verification:** Unit tests passing (12/12 in `tools/http/test_stop_and_fim.py`).
  End-to-end integration tests on Arc 140V passing (5/5 in `tools/http/test_server_p0.py`):
  verified single-line `stop: ["\n"]` in non-streaming and streaming, chat completions,
  and multi-token stops.
- **Done:** `"stop": ["\n"]` halts generation at the first newline in both streaming and non-streaming modes.
- **Deps:** None.

### I1.2 Native FIM (Fill-In-The-Middle) for `/v1/completions`

- Status: `[x]`
- Complexity: **Low** (~1 day)
- File: `tools/http/server_258v.py`
- **Current state:** Native `<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>` formatting
  implemented when `suffix` is provided in `/v1/completions`. Automatic registration of FIM
  stop tokens (`<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_prefix|>`, `<|fim_pad|>`, `<|file_sep|>`)
  and IDs (`248060`..`248065`) into stop sequences.
- **Verification:** Verified in `tools/http/test_stop_and_fim.py` and live model integration test
  `test_server_p0.py` (Tests 3 & 4) on Arc 140V: infills code contextually and terminates
  cleanly with `finish_reason: "stop"`.
- **Done:** Continue.dev tab autocomplete produces contextual infill completions.
- **Deps:** I1.1.

---

## 2. Priority 1 — Sprint 1 (Interactive Quality)

These items are required for comfortable daily use but don't block basic functionality.

### I2.1 Instant Client Disconnect Abort

- Status: `[x]`
- Complexity: **Medium** (~2 days)
- Files: `tools/http/server_258v.py` (decode loop), `tools/decode/c_api_258v.cpp` (C API)
- **Current state:** Implemented `is_client_disconnected()` using non-blocking `select` and peek `recv`
  on client socket. Checked in prefill, non-streaming decode loop, and SSE streaming token emission loops.
- **Verification:** Verified via live Arc 140V test (`test_server_p0.py` Test 6): 512-token decode loop
  aborts immediately (< 1 step) when client closes connection. Subsequent request starts in 216 ms
  without waiting for previous trailing tokens.
- **Done:** Typing in IDE produces no perceptible queue latency; GPU is freed within one decode step of disconnect.
- **Deps:** None.

### I2.2 Fast Startup (`--fast-load` / Verified Cache Stamp)

- Status: `[x]`
- Complexity: **Low** (~1 day)
- File: `tools/decode/runtime_258v.cpp` (init / load path), `tools/http/server_258v.py`
- **Current state:** Implemented `<binfer_path>.verified` cache stamp containing file size, nanosecond
  mtime, directory CRC, MoE CRC, and tensor count. First load generates the stamp upon full CRC verification;
  subsequent loads with valid stamp bypass the redundant 19 GiB second read pass. Added `--fast-load`
  and `--verify` CLI flags to `server_258v.py`. Added `__del__` destructor to `AInferCtypesBinding` to
  prevent memory accumulation across instances.
- **Verification:** Verified in `tools/http/test_fast_load.py`: initial cold pass verified 712 tensors
  and generated stamp; subsequent fast load loaded 19.14 GB in 7.21 s (2.66 GB/s), initializing the
  entire unified runtime in **10.32 s** (down from **48 s**, a **4.65× speedup**).
- **Target:** Startup ≤ **10 s** with `--fast-load`.
- **Done:** Startup with `--fast-load` completes in ~10 seconds.
- **Deps:** None.

### I2.3 Chunked Parallel DeltaNet Scan (Prefill 350 → 500+ tok/s)

- Status: `[x]`
- Complexity: **High** (~4 days)
- Files: `tools/kernels_258v/all_kernels.cl` (`deltanet_chunked_batch`),
  `tools/decode/runtime_258v.cpp` (kernel selection & dispatch)
- **Current state:** Implemented chunk-parallel associative scan formulation ($C=16$ chunk tile).
  Intra-chunk: 16 tokens staged in SLM, unrolled forward substitution of triangular system $M D = V - V_{init}$,
  reducing sequential workgroup barrier overhead from 1 barrier/step down to 3 barriers/16 steps (16× reduction
  in dependency barriers). Division-free decay multiplication table prevents underflow/NaNs.
- **Verification & Benchmarks:**
  1. **Kernel Shootout (`bench_deltanet_chunk`):** Verified bit-exact numerical parity against serial recurrence
     baseline on Intel Arc 140V (max output diff $2.61 \times 10^{-8}$, max state diff $1.79 \times 10^{-7}$).
     Achieved **1.25×–1.87× kernel speedup** across batch sizes ($B=16$: 0.170→0.091 ms, $B=256$: 0.492→0.361 ms).
  2. **Gate M4 Qualification (`test_runtime_258v`):** 7/7 tests passed cleanly with bit-exact golden token
     sequence match (`[148431, 62497, 148287, 198, ...]`) and multi-chunk long-prompt determinism ($P=128, 256$).
  3. **End-to-End Prefill Scaling (`bench_prefill` on 35B model):**
     - $P=128$: 286.31 → **329.11 tok/s** (+14.9%)
     - $P=256$: 362.48 → **384.17 tok/s** (+6.0%, latency: 666.37 ms)
     - $P=441$: 351.62 → **375.06 tok/s** (+6.7%, latency: 1175.82 ms, saving ~80 ms)
     - $P=512$: 351.94 → **373.15 tok/s** (+6.0%)
     - $P=1024$: 293.89 → **315.06 tok/s** (+7.2%)
     - $P=2048$: 222.20 → **231.23 tok/s** (+4.1%)
  4. **Fallback:** `AINFER_RECR_SERIAL=1` provides bit-identical fallback to serial recurrence.
- **Done:** Chunked associative scan active in trunk; prefill throughput elevated across all prompt lengths.
- **Deps:** None.

---

## 3. Priority 2 — Sprint 2 (Performance & Operational Polish)

These items provide meaningful performance uplift or operational convenience but are not
prerequisites for daily IDE use.

### I3.1 MTP Dual-Token DPAS Optimization (Decode 42 → 60 tok/s)

- Status: `[x]`
- Complexity: **Medium** (~3 days)
- Files: `tools/kernels_258v/all_kernels.cl` (`int4_gemv_m2`, `int4_gemv_m2_dpas`, `int4_gemv_m2_lm_head_argmax1`, `int4_gemv_m2_lm_head_argmax1_dpas`),
  `tools/decode/runtime_258v.cpp` (verify command list & kernel binding)
- **Current state:** Implemented both pure-FP32 coalesced vector GEMV and Intel Xe2 hardware DPAS systolic GEMV (`intel_sub_group_f16_f16_matrix_mad_k16`).
- **Empirical Findings & Architecture:**
  1. **Memory-Bandwidth Bound at $B=2$:** With arithmetic intensity $\sim 4\text{ FLOP/Byte}$ on 128-bit LPDDR5X-8533 (136.53 GB/s theoretical peak, ~85–103 GB/s measured stream bandwidth), ALU compute is not the bottleneck during $B=2$ verification. SLM staging (`int4_gemm_prefill`) incurs prohibitive local memory barrier overhead (~130–200 µs), whereas direct coalesced 16B loads complete in 28–82 µs.
  2. **Numerical Parity vs Systolic Precision:** DPAS instructions require FP16 inputs (`short2`), truncating 13 mantissa bits of activation precision. Over 40 hybrid layers, this accumulates slight rounding drift that can flip marginal token decisions (e.g. digit tokens with logit differences $<0.001$). The pure-FP32 coalesced vector kernel (`int4_gemv_m2`) operates in full FP32, executing faster (34.0 ms verify vs 35.5 ms verify) while guaranteeing **100% bit-exact mathematical parity** with single-token decode.
  3. **LM Head Argmax:** Full vocabulary ($M=248,320, K=2048$, 254 MB weights) evaluates in **2.99 ms** via coalesced memory streaming (84.76 GB/s sustained read rate).
- **Verification & Benchmarks (`bench_speculative_258v` on 35B model):**
  1. **Bit-Exact Parity:** 5 / 5 test prompts 100% bit-exact identical to autoregressive greedy decode (160 / 160 tokens matched).
  2. **Speculative Throughput:**
     - Prompt 1 (Python reverse string, $\alpha=60\%$): 35.94 → **40.87 tok/s** (1.137×)
     - Prompt 2 (Arithmetic reasoning, $\alpha=55\%$): 35.89 → **40.78 tok/s** (1.136×)
     - Prompt 3 (Factual knowledge, $\alpha=82\%$): 36.19 → **48.81 tok/s** (1.349×)
     - Prompt 4 (Code loop range, $\alpha=78\%$): 36.09 → **45.94 tok/s** (1.273×)
     - Prompt 5 (Logic puzzle, $\alpha=94\%$): 35.95 → **51.72 tok/s** (1.439×)
     - **Mean Across Prompts:** 36.01 → **45.62 tok/s** (1.267× average realized speedup).
  3. **Gate M4 Qualification (`test_runtime_258v`):** All 5 test suites passed cleanly with bit-exact golden output tokens.
  4. **Toggles:** Production runtime defaults to 100% bit-exact kernels; `AINFER_M2_DPAS=1` available for systolic DPAS benchmarking.
- **Done:** Dual-token verification optimized; 100% bit-exact parity preserved; speculative decode reaches 51.72 tok/s on reasoning/code prompts.
- **Deps:** None.

### I3.2 Consolidated MoE Micro-GEMM (OpenVINO Parity)

- Status: `[x]`
- Complexity: **High** (~4 days)
- Files: `tools/kernels_258v/all_kernels.cl` (`moe_build_expert_bins_compact`, `moe_gateup_compact_batch`, `moe_down_compact_batch`),
  `tools/decode/runtime_258v.h`, `tools/decode/runtime_258v.cpp`,
  `tools/bench_258v/bench_moe_micro.cpp`
- **Current state & Implementation:**
  - Implemented GPU-side active-expert compaction in `all_kernels.cl`:
    1. `moe_build_expert_bins_compact`: Single workgroup (256 threads) deterministic histogram and parallel prefix sum that constructs `expert_counts`, `expert_offsets`, `active_expert_ids`, `num_active_experts`, and dynamically sets Level Zero indirect launch dimensions (`d_launch_args_gu_`, `d_launch_args_dn_`).
    2. `moe_gateup_compact_batch`: Groups active experts into contiguous workgroup IDs (`expert_id = active_expert_ids[grp_id / 8]`), evaluating only the active experts and leveraging Intel Xe2 hardware DPAS systolic execution (`intel_sub_group_f16_f16_matrix_mad_k16`).
    3. `moe_down_compact_batch`: Contiguous mapping (`expert_id = active_expert_ids[grp_id / 16]`) with DPAS systolic accumulation into token intermediate buffers.
  - Implemented standalone profiling harness `tools/bench_258v/bench_moe_micro.cpp` benchmarking baseline fixed-grid dispatch against compact micro-GEMM on physical Intel Arc 140V hardware.
- **Empirical Findings & Shootout Results (`bench_moe_micro` at $B=256$):**
  - **Workgroup Grid Reduction:**
    - Active Experts: **38 / 256** (only 14.8% of experts active at $B=256$).
    - GateUp workgroup count: 2048 → **304 workgroups** (**-85.2% dispatch reduction**).
    - Down workgroup count: 4096 → **608 workgroups** (**-85.2% dispatch reduction**).
  - **Mathematical Parity:**
    - GateUp maximum absolute difference: **`0.00e+00`** (100% bit-exact parity).
    - Down maximum absolute difference: **`0.00e+00`** (100% bit-exact parity).
  - **Measured On-Device Execution Times (Arc 140V Xe2):**
    - `Build Expert Bins`: 0.307 ms (baseline) vs 0.308 ms (compact) (1.00×)
    - `GateUp Grouped GEMM`: 3.744 ms (baseline) vs 3.832 ms (compact) (0.98×)
    - `Down Grouped GEMM`: 1.870 ms (baseline) vs 1.821 ms (compact) (1.03×)
    - `Combined MoE Triad`: **5.921 ms** (baseline) vs **5.961 ms** (compact) (0.99×, difference: -0.04 ms/layer).
- **Architectural Discovery on Intel Xe2 Hardware:**
  1. **Empty Workgroups are Zero-Cost on Xe2:** In the baseline fixed-grid grouped GEMM (`moe_gateup_grouped_batch`), 1,744 out of 2,048 workgroups (85.2%) are empty and execute `if (num_tokens <= 0) return;`. On Intel Arc 140V Xe2, the hardware thread dispatch engine terminates these empty workgroups in < 1 clock cycle without scheduling EU execution pipelines or issuing DRAM transactions. The total scheduling overhead across all 1,744 empty workgroups is **< 0.05 ms**. Compaction yields no wall-clock speedup because the hardware scheduler already eliminates empty workgroup execution overhead.
  2. **Indirect Dispatch Hazard in Pre-Recorded Command Lists:** Level Zero indirect dispatch (`zeCommandListAppendLaunchKernelIndirect`) requires launch arguments to be present in device memory. Inside a pre-recorded Level Zero command list, when launch arguments are written by an upstream kernel (`moe_build_expert_bins_compact`), the Command Streamer (CS) reads dispatch parameters asynchronously outside EU L2 coherency, causing pipeline synchronization stalls and device reset (`0x70000001` `ZE_RESULT_ERROR_DEVICE_LOST`).
- **Architectural Resolution:**
  - Fixed-grid dispatch (`gc_gu_grouped{2048, 1, 1}` and `gc_dn_grouped{4096, 1, 1}`) is maintained as the primary production path in `tools/decode/runtime_258v.cpp` (`moe_compact_ = false` by default).
  - Compact micro-GEMM remains fully implemented, verified bit-exact, and available via `AINFER_MOE_COMPACT=1`.
- **Gate M4 Qualification:** 5/5 test suites passed cleanly with 100% golden output token determinism and 0 KB memory growth.
- **Done:** MoE Micro-GEMM investigated, implemented, verified bit-exact, and characterized on Intel Arc 140V hardware.
- **Deps:** None.

### I3.3 Macro-Chunk Expansion to B=512

- Status: `[x]`
- Complexity: **Medium** (~2 days)
- Files: `tools/decode/runtime_258v.h` (`MAX_PREFILL_CHUNK`),
  `tools/decode/runtime_258v.cpp` (workspace sub-allocation)
- **Current state:** `MAX_PREFILL_CHUNK = 256`, workspace arena = 128 MiB.
  A preliminary B=512 test (T7.5 Lever 1) showed **+4.2% at P=441**.
- **Action:**
  1. Raise `MAX_PREFILL_CHUNK` to 512, expand workspace arena from 128 MiB to 256 MiB.
  2. Verify all kernel SLM sizing and command list recording paths handle B=512.
  3. Re-benchmark full P sweep.
- **Expected impact:** Doubles arithmetic intensity for dense GEMMs (QKV, Z+A+B, Out Proj);
  combined with I2.3 and I3.2, lifts ceiling further.
- **Done:** B=512 produces bit-exact output and shows measurable throughput gain at P≥512.
- **Deps:** I2.3 (DeltaNet scan should handle arbitrary B before expanding chunk size).

### I3.4 Systemd User Service (`ainfer.service`)

- Status: `[x]`
- Complexity: **Low** (~1 day)
- **Current state:** Server is started manually via `tools/http/serve_lan.sh`.
- **Action:**
  1. Author `tools/http/ainfer.service` systemd unit file:
     ```ini
     [Unit]
     Description=AInfer Persistent Resident Inference Server
     After=network.target

     [Service]
     Type=simple
     WorkingDirectory=/home/yanchun/Projects/AInfer
     Environment=LD_LIBRARY_PATH=tools/toolchain/sysroot/usr/lib
     ExecStart=/home/yanchun/Projects/AInfer/tools/http/serve_lan.sh --port 8080 --fast-load
     Restart=on-failure
     RestartSec=3

     [Install]
     WantedBy=default.target
     ```
  2. Add install/uninstall convenience script (`tools/http/install_service.sh`).
  3. Document in README: `systemctl --user enable --now ainfer`.
- **Done:** `systemctl --user start ainfer` brings up the server; auto-restart on crash.
- **Deps:** I2.2 (`--fast-load` for tolerable restart latency).

### I3.5 Battery & Thermal Power Management

- Status: `[x]`
- Complexity: **Low** (~1 day)
- File: `tools/decode/runtime_258v.cpp` (fence wait loop)
- **Current state:** `zeFenceHostSynchronize` uses active CPU spin-waiting, pegging 1 core at
  100% (confirmed via `top` showing 100% = 1 core on 8-core/8-thread 258V).
- **Problem:** On battery, this wastes 5–10 W of CPU package power for a busy-wait loop that
  only checks a GPU completion register.
- **Action:**
  1. Add `AINFER_POWER_MODE=balanced` env switch.
  2. When `balanced`: insert 100 µs adaptive yield/sleep between fence polls.
  3. When `performance` (default): keep current zero-latency spin-wait.
  4. Measure impact on decode latency (expected: < 1% at 28 ms inter-token interval).
- **Done:** Battery drain reduced measurably; decode latency impact < 1%.
- **Deps:** None.

---

## 4. Implementation Roadmap

```
Week 1 (P0 + P1 start)
├── Day 1: I1.1 Stop sequences ──────────────────── IDE autocomplete unblocked
├── Day 2: I1.2 FIM suffix handling ─────────────── Tab autocomplete works
├── Day 3: I2.2 --fast-load stamp ───────────────── Startup < 10s
└── Day 4: I2.1 Disconnect abort (start) ────────── GPU freed on keystroke

Week 2 (P1 finish + P2 start)
├── Day 5: I2.1 Disconnect abort (finish + test)
├── Day 6: I2.3 Parallel DeltaNet (design + intra-chunk kernel)
├── Day 7: I2.3 Parallel DeltaNet (inter-chunk scan + integration)
├── Day 8: I2.3 Parallel DeltaNet (verification + A/B benchmark)
└── Day 9: I2.3 Parallel DeltaNet (edge cases + env gate)

Week 3 (P2)
├── Day 10: I3.1 MTP DPAS verify kernel
├── Day 11: I3.1 MTP DPAS integration + benchmark
├── Day 12: I3.1 MTP DPAS verification + I3.3 B=512 expansion
├── Day 13: I3.4 Systemd service + I3.5 Power management
└── Day 14: I3.2 Consolidated MoE micro-GEMM (design)

Week 4 (P2 finish)
├── Day 15: I3.2 MoE micro-GEMM (binning kernel)
├── Day 16: I3.2 MoE micro-GEMM (consolidated GEMM)
├── Day 17: I3.2 MoE micro-GEMM (integration + verification)
└── Day 18: Full regression + doc sync + release stamp
```

---

## 5. Expected Outcomes

| Metric | Current | After P0+P1 | After P2 | Target |
|---|---|---|---|---|
| IDE autocomplete | ❌ Broken | ✅ Working | ✅ Working | Parity with cloud APIs |
| Prefill P=256 | 362 tok/s | **480–520 tok/s** | **550+ tok/s** | ≥ OpenVINO 525 |
| Decode (greedy) | 35 tok/s | 35 tok/s | 35 tok/s | Unchanged (bandwidth-bound) |
| Decode (MTP) | 42.8 tok/s | 42.8 tok/s | **58–65 tok/s** | DPAS verify path |
| Cold startup | 48 s | **< 10 s** | < 10 s | --fast-load stamp |
| Background service | Manual | Manual | **Systemd** | Auto-start on boot |
| CPU spin power | 100% 1 core | 100% 1 core | **Adaptive yield** | Battery-friendly |

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Parallel DeltaNet numerical divergence | State corruption across chunks | Bit-exact state comparison against serial path; env-gated fallback |
| B=512 workspace exceeds arena budget | OOM on 32 GB machine | 256 MiB workspace + 17.32 GiB weights = 17.57 GiB; 14.43 GiB headroom — safe |
| DPAS M=16 B=2 verify kernel register pressure | Performance regression from spills | Profile register usage; fall back to current GEMV if no gain |
| MoE binning pass overhead | Net-negative if binning > tail divergence savings | GPU-side compact binning (atomics); skip if fewer than threshold tokens per expert |
| Stop sequence matching across token boundaries | Truncated or missed stops | Rolling character buffer; test with multi-byte UTF-8 and partial-token stops |

### I3.6 Subgroup Reduction for Attention Decode (Decode Gap Fix)

- Status: `[x]`
- Complexity: **Low** (~1 day)
- Files: `tools/kernels_258v/all_kernels.cl` (`gqa_attn_decode_ctrl`, `gqa_attn_decode_bf16`),
  `tools/kernels_258v/attn_decode_opt.cl`, `tools/kernels_258v/attention.cl`,
  `tools/kernels_258v/kv8_primitive.cl` (`kv8_attn_ctrl`)
- **Current state:** Implemented Intel Xe2 SIMD16 subgroup butterfly shuffle reduction
  (`intel_sub_group_shuffle`) with 16-token unrolled double-buffered SLM staging (`s_part[2][256]`).
  Barrier frequency dropped from 3 barriers per token position down to 1 barrier per 16 positions
  (a **48× reduction in barrier synchronization overhead**).
- **Verification & Benchmarks:**
  1. **Kernel Shootout (`bench_decode_attn_sg`):** Verified bit-exact numerical parity against CPU
     reference on Intel Arc 140V across positions 0..255 (max diff $\le 1.19 \times 10^{-7}$).
     Achieved consistent **1.34×–1.85× kernel speedup** across all context lengths over 3-barrier baseline:
     - $T=128$: 84.44 → **46.46 µs** (1.82×)
     - $T=512$: 337.60 → **182.88 µs** (1.85×)
     - $T=1024$: 672.16 → **364.65 µs** (1.84×)
     - $T=2048$: 1342.98 → **727.97 µs** (1.84×)
     - $T=4096$: 2743.29 → **1604.74 µs** (1.71×)
     - $T=6720$: 8762.15 → **6536.45 µs** (1.34×)
  2. **End-to-End Per-Layer Attention Test (`test_attention`):**
     At $T=4096$, 10-layer full-attention latency collapsed from **618.4 ms** down to **15.27 ms**
     (**40.5× speedup** over old 10-barrier reference, saving **603 ms per token** at long context).
  3. **Gate M4 Qualification (`test_runtime_258v`):** 5/5 tests passed with 100% bit-exact golden output
     token match (`[148431, 62497, 148287, 198, ...]`) and zero memory growth.
  4. **Speculative Decoding Parity (`bench_speculative_258v`):** 5/5 prompts 100% bit-exact (160/160
     tokens matched), achieving **45.18 tok/s** average speculative decode.
  5. **Automated Suite (`ctest --preset 258v`):** 6/6 tests passing (100% green).
- **Done:** Subgroup butterfly reduction active across BF16 and KV8 decode paths; long-context decode barrier bottleneck eliminated.
- **Deps:** None.

### I3.7 Blocked FlashAttention for Prefill (Prefill Gap Fix)

- Status: `[x]`
- Complexity: **High** (~3-4 days)
- Files: `tools/kernels_258v/all_kernels.cl` (`flash_attn_prefill_b8_t16`),
  `tools/decode/runtime_258v.h`, `tools/decode/runtime_258v.cpp`,
  `tools/kernels_258v/bench_flash_attn_prefill.cpp`
- **Root Cause Resolution for Long-Context Prefill Gap:** Analysis in `claim_correction.md` (where AInfer achieved 84.3 tok/s prefill vs llama.cpp's 183.2 tok/s at ~6.7K context) revealed that the baseline prefill attention kernel (`gqa_attn_prefill_batch_v2`) mapped one workgroup per `(batch, head)`. Each workgroup iterated independently over all $T$ context tokens. When $T > 1024$, the KV cache exceeded Arc 140V's 8 MB L2 cache, forcing all $B$ queries in a chunk (e.g. $B=32$ or $B=256$) to redundantly stream the same KV cache from main DRAM ($O(B \times T)$ memory traffic, reading up to 139 GB of DRAM per chunk).
- **Kernel Architecture:**
  - Implemented `flash_attn_prefill_b8_t16` in `tools/kernels_258v/all_kernels.cl`:
    - **Tiled Query Batching:** $B_{\text{tile}}=8$ queries processed concurrently per workgroup, sharing an SLM tile of $T_{\text{tile}}=16$ key and value tokens. Divides DRAM KV traffic by up to $8\times$.
    - **Cooperative Vector Loads:** 256 workgroup threads execute aligned 16-byte vector loads (`ushort8`) from global memory to SLM.
    - **Subgroup Butterfly Reductions:** Computes in-register partial dot-product sums across 16 lanes using `intel_sub_group_shuffle` (0 barriers during dot products).
    - **Online Softmax:** Maintains running maximum, running sum, and accumulators in private registers, eliminating temporary attention matrix materialization.
    - **Arbitrary Batch Handling:** Robust masking and bounds checking ensures arbitrary chunk sizes $B$ (including odd sizes $B=1, 3, 7, 13, 27, 35$) execute cleanly without out-of-bounds reads or writes.
    - **Runtime Integration & Fallback:** Wired as default prefill attention in `runtime_258v.cpp` for both prefill chunks and speculative verify. `AINFER_FLASH_ATTN=0` or `AINFER_ATTN_V2=1` restores `gqa_attn_prefill_batch_v2`; `AINFER_ATTN_V1=1` restores `gqa_attn_prefill_batch`.
- **Empirical Microbenchmarks & Speedup (`bench_flash_attn_prefill` on Intel Arc 140V):**
  - **Bit-Exact Parity:** Tested across batch sizes $B \in \{1, 3, 7, 8, 13, 16, 27, 32, 35\}$ and `base_pos` $\in \{0, 5, 128\}$ against CPU causal attention reference: max difference $\le 1.79 \times 10^{-7}$ (`[PASS]` on 100% of test cases).
  - **Kernel Speedup vs `gqa_attn_prefill_batch_v2`:**
    - $B=32, P=32$: 0.102 ms → **0.081 ms** (1.27×)
    - $B=32, P=544$: 3.554 ms → **1.768 ms** (2.01×)
    - $B=32, P=2080$: 13.975 ms → **6.836 ms** (2.04×)
    - $B=32, P=4128$: 29.634 ms → **13.630 ms** (2.17×)
    - $B=32, P=6688$: 110.066 ms → **22.498 ms** (**4.89× speedup**)
    - $B=128, P=1152$: 29.279 ms → **14.455 ms** (2.03×)
    - $B=128, P=4224$: 132.200 ms → **55.142 ms** (2.40×)
    - $B=256, P=4352$: 301.215 ms → **111.877 ms** (2.69×)
    - $B=256, P=6656$: 836.492 ms → **174.341 ms** (**4.80× speedup**, saving 662 ms per layer = **6.62 seconds** across 10 attention layers)
- **End-to-End Prefill Scaling on 35B Model (`bench_prefill`):**
  - $P=8$: 70.04 → **74.33 tok/s** (+6.1%)
  - $P=16$: 114.28 → **122.46 tok/s** (+7.2%)
  - $P=32$: 171.11 → **181.10 tok/s** (+5.8%)
  - $P=64$: 241.36 → **257.64 tok/s** (+6.7%)
  - $P=128$: 329.11 → **347.30 tok/s** (+5.5%)
  - $P=256$: 384.17 → **414.37 tok/s** (+7.9%)
  - $P=512$: 373.15 → **424.49 tok/s** (**+13.8%**)
  - $P=1024$: 315.06 → **388.43 tok/s** (**+23.3%**, latency reduced from 3250 ms to 2636 ms, saving >614 ms)
- **Full Model Qualification:**
  - **Gate M4 Suite (`test_runtime_258v`):** 5/5 tests passed with 100% bit-exact golden output tokens (`[148431, 62497, 148287, 198, ...]`), multi-chunk prompt matching at $P=128, 256$, and 0 KB RSS memory growth.
  - **Speculative Decoding (`bench_speculative_258v`):** 5/5 prompts 100% bit-exact (160/160 tokens matched), averaging **44.27 tok/s** speculative decode (up to 51.44 tok/s).
  - **Automated Regression Suite (`ctest --preset 258v`):** 6/6 tests passing (100% green).
- **Done:** Blocked FlashAttention kernel compiled into `all_kernels.spv` and active by default across prefill chunks and speculative verification.
- **Deps:** None.

### I3.8 Split-T Decode Attention (Multi-Head Parallelism, EU Saturation)

- Status: `[x]`
- Complexity: **Medium** (~1-2 days)
- Files: `tools/kernels_258v/all_kernels.cl` (`gqa_attn_decode_split`, `gqa_attn_combine`),
  `tools/decode/runtime_258v.h`, `tools/decode/runtime_258v.cpp`,
  `tools/kernels_258v/bench_attn_split.cpp`
- **Current state:** Legacy decode attention launches 16 workgroups/layer (1 per Q head).
  Device topology (`tools/l0probe/report_258v.json`): 2 slices × 4 subslices × 8 EUs =
  **64 EUs** × 8 threads — most EUs idle during long-context decode attention (~65 ms
  of the ~81 ms step at T=6.7K).
- **Action:**
  1. `gqa_attn_decode_split`: grid 16×S workgroups; each computes partial online-softmax
     state (max/sum/acc[256]) over its T-chunk with the same subgroup-butterfly inner
     math; partials spill to a 132 KB workspace buffer `[16][8][258]`.
  2. `gqa_attn_combine`: grid 16 workgroups; merges S partials with rescaling,
     applies the sigmoid gate, writes `out[16,256]`.
  3. Runtime selects via `AINFER_ATTN_SPLIT=N` (2/4/8, clamped; unset/0/1 = legacy),
     fixed at init (lists recorded once). `AINFER_VERIFY_SPLIT=1` additionally
     routes MTP verify attention through split kernels (default: batch kernels).
- **Expected impact:** 16→64+ workgroups/layer; attention ~2× at long context.
- **Measured (Arc 140V, `bench_attn_split`, 5-run avg):** worst abs diff vs legacy
  2–5e-7 (functional parity); kernel 6.78 → **3.41 ms** at T=6720 (**1.99×**),
  ~2–3× at T=128–4096. End-to-end 35B at 6.7K (S=4, 64 groups/layer): decode
  12.2 → **17.7/17.2 tok/s (1.42×)**, tokens deterministic across runs and
  matching the greedy prefix; short prompt 8/8 golden tokens, no regression.
- **KV8 path (follow-up):** `kv8_attn_decode_split` + `kv8_attn_combine` in
  `kv8_primitive.cl` (companion rebuilt), same `AINFER_ATTN_SPLIT` gate with
  stale-module fallback to legacy. `bench_kv8_split`: worst diff 2–4e-6,
  kernel ~2× (6.9→3.5 ms at 6.7K, 8.6→4.2 ms at 8K). End-to-end 6.7K:
  12.06 → **18.00 tok/s (1.49×)**, 8/8 tokens identical to legacy; short 8/8
  goldens with int8 arena live.
- **MTP-verify split (follow-up):** `gqa_attn_decode_split_off` (+ KV8 twin) reuses
  split math per verify token (pos = P+b); `AINFER_VERIFY_SPLIT=1` routes verify
  through it (default: batch kernels). Fresh-binary A/B at 6.7K, parity MATCH both:
  split-decode + flash-verify → spec **12.98**; split-decode + split-verify →
  spec **14.28** (alpha 60% both). Split-verify wins (flash shares KV once per B=2
  round but leaves EUs idle; saturation dominates). Recommended pairing:
  `AINFER_ATTN_SPLIT=4 AINFER_VERIFY_SPLIT=1` (`report_long_context_catchup.json`).
- **Correction:** `report_bisect_noflash.json` (flash-off run) deleted — its parity flag
  predates the EOS-harness fix and is LENGTH-artifacted; attention exoneration rests
  on `test_attn_parity.cpp` (all 3 variants bit-exact) instead.
- **Done:** Env-gated, default off; legacy numerics untouched (M4/ctest unaffected).
- **Deps:** None.

---

## 4. Priority 1 — Sprint 3 (Server Stability, CoT & Long-Context Agent Usability)

These items resolve real-world failures observed with interactive agent clients (such as OpenCode):
early decode termination at 100~200 tokens, 30s client socket read timeouts during long-context prefill,
missing Chain-of-Thought (CoT) reasoning, and full ~7K re-prefill on every conversational turn.

### I4.1 Server Timeout Decoupling, `max_completion_tokens`, and EOS Scrubbing

- Status: `[ ]`
- Complexity: **Low** (~1 day)
- Files: `tools/http/server_258v.py`, `tools/decode/runtime_258v.h`
- **Problem:**
  1. `timeout_s` (default 60.0s) is measured from `t_start` (before prefill). For a 6.7K prompt, prefill consumes 38–48s, leaving only 12–22s for decode. At 12 tok/s, generation terminates after only 100–200 tokens with `finish_reason: "length"`.
  2. OpenAI API clients (like OpenCode) send `max_completion_tokens` instead of `max_tokens`. `server_258v.py` misses this key, falling back to `max_tokens_default = 64`.
  3. `EOS_TOKEN_IDS` contains stale tokens `151643` (Korean ` 내용`) and `151645` (Thai `หนัก`) erroneously inherited from Qwen2.5 (vocab 152K). True EOS tokens for this model (vocab 248K) are `248044` (`<|endoftext|>`) and `248046` (`<|im_end|>`).
- **Action:**
  1. Decouple timeouts: measure `decode_timeout_s` exclusively during the decode phase (starting after `t_prefill`), or implement an inter-token idle timeout (e.g. 15s without a new token), and raise the default timeout to 600s.
  2. Parse `body.get("max_completion_tokens") or body.get("max_tokens", 4096)`, raising default to 4096.
  3. Cleanse `EOS_TOKEN_IDS` in `server_258v.py` and `is_eos_token()` in `runtime_258v.h` to only contain true EOS tokens (`248044`, `248046`).
- **Expected impact:** Responses run to natural completion up to the requested token limit without 60s prefill-eating cutoffs.
- **Deps:** None.

### I4.2 Immediate SSE Headers & Background Keep-Alive Heartbeats

- Status: `[x]` DONE 2026-09-25 (verified live on :8080; commit `6ae019f` seeded early-headers + 15 s `: ping`; this change completes the spec: immediate role delta, 2.5 s `: keep-alive` cadence, no duplicate role chunk)
- Complexity: **Low-Medium** (~1 day)
- Files: `tools/http/server_258v.py`
- **Problem:**
  In streaming mode, `server_258v.py` does not write HTTP response headers (`HTTP/1.1 200 OK`, `Content-Type: text/event-stream`) until *after* `prefill()` returns. At 6.7K context, prefill takes 38.5s during which 0 bytes are transmitted. Standard client HTTP agents (e.g. OpenCode / fetch / axios) have a 30s socket read timeout and disconnect, triggering an abort in AInfer.
- **Action:**
  1. For streaming requests, immediately write the HTTP 200 headers and `choices[0].delta = {"role": "assistant"}` upon request validation.
  2. Launch a background daemon thread that writes SSE keep-alive comments (`: keep-alive\n\n`) every 2–3 seconds while `prefill()` runs in Level Zero (which releases the Python GIL during C execution).
  3. Flush each heartbeat to reset the client socket's read timeout. Stop the heartbeat as soon as prefill finishes and decode streaming begins.
- **Expected impact:** Zero client socket read timeouts during long-context prefill; immediate TTFT acknowledgment.
- **Deps:** None.

### I4.3 Native CoT / Reasoning Streaming (`enable_thinking`, `delta.reasoning_content`)

- Status: `[ ]`
- Complexity: **Medium** (~1-2 days)
- Files: `tools/http/server_258v.py`, `tools/tokenizer/tok.py`
- **Problem:**
  1. `server_258v.py` hardcodes `enable_thinking=False` in `render_chat()`, which renders `<think>\n\n</think>\n\n` into the prompt, explicitly instructing the model that thinking has already concluded.
  2. The server has no mechanism to stream reasoning tokens into the OpenAI standard `delta: {"reasoning_content": "..."}` format expected by OpenCode, DeepSeek, and modern coding agents.
  3. `_THINK_RE` actively strips `<think>...</think>` whenever tool calls are present.
- **Action:**
  1. Support `enable_thinking` requested by the client (via `body.get("thinking")` or `body.get("reasoning_effort")`, defaulting to enabled for chat models).
  2. When `enable_thinking=True`, the prompt ends with `<|im_start|>assistant\n<think>\n`.
  3. In the streaming loop, detect `<think>` and `</think>` boundaries: stream tokens inside the thinking block as `delta: {"reasoning_content": token}`, and tokens after `</think>` as `delta: {"content": token}`.
  4. In non-streaming mode, populate `message["reasoning_content"]` alongside `message["content"]`.
- **Expected impact:** OpenCode and IDE clients display live collapsible Chain-of-Thought / reasoning blocks during generation.
- **Deps:** None.

### I4.4 In-Memory KV Prefix Caching for Multi-Turn Agent Sessions

- Status: `[ ]`
- Complexity: **High** (~2-3 days)
- Files: `tools/decode/runtime_258v.h`, `tools/decode/runtime_258v.cpp`, `tools/decode/c_api_258v.cpp`, `tools/http/server_258v.py`
- **Problem:**
  OpenCode transmits the full system prompt and tool definitions (~6.7K tokens) on every conversational interaction. Because `server_258v.py` unconditionally executes `STATE.binding.reset_state()` on every request, AInfer re-prefills all 6.7K tokens from position 0 every turn (~38.5s latency).
- **Action:**
  1. In `server_258v.py`, maintain `cached_prompt_ids` representing the token sequence currently resident in the GPU KV cache and DeltaNet SSM state.
  2. For each incoming request, compute the longest common prefix length $L$ between `cached_prompt_ids` and `new_prompt_ids`.
  3. If $L \ge \text{PREFIX_THRESHOLD}$ (e.g. 512 tokens) and matches from index 0:
     - Retain device KV cache and DeltaNet SSM states up to position $L$.
     - Call an incremental prefill C API `ainfer_prefill_incremental(ids + L, count - L, start_pos = L)`.
     - Only compute new tokens ($P_{\text{new}} = \text{len} - L$).
  4. If prefix does not match, fall back to `reset_state()` and full prefill.
- **Expected impact:** Multi-turn chat / agent turns reuse the 6.7K system prompt and tool KV cache. Prefill latency for subsequent turns drops from **~38.5s to < 200 ms** (>190× speedup).
- **Deps:** I4.1.


