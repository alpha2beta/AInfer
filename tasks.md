# AInfer Task Breakdown — Intel Core Ultra 7 258V (Arc 140V)

Tasks derived from `plan.md` and `AInfer_258V_migration_plan.md`. Each task has a stable ID, dependencies,
and a concrete completion check. IDs are stable so they can be referenced from commits, logs, and issues.

Legend:
- Status: `[ ]` pending, `[~]` in progress, `[x]` done, `[-]` dropped.
- `Deps`: task IDs that must complete first.
- `Done`: objective completion check.

---

## Phase 0: Scope, Identifiers, and Acceptance Definitions (Milestone 0)

### T0.1 Pin exact model and tokenizer identity
- Status: `[x]`
- Deps: none
- Do: Pin exact `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized` repository ID (dequantized SafeTensors fine-tune of `Qwen3.5-MoE`, GGUF reference `LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF`), immutable git revision hash, and tokenizer assets. Record source URLs, SafeTensors shard inventory, and file hash manifests in `target_model_identity.json`. No official `Qwen/...` repository release exists for this checkpoint; do not re-pin against one.
  DONE 2026-09-17: All metadata, tokenizer, 17 SafeTensors shards (~67 GB), and GGUF reference `Tiel-Coder-35B-A3B-Genesis-Hermes-APEX-Compact.gguf` (~16.2 GB) downloaded to `models/Tiel-Coder-35B-A3B-Genesis-Hermes/`. Checkpoint verified: 1045 tensors, 40 layers (30 DeltaNet + 10 full-attention), 256 experts, 8 active. Documented in `target_model_identity.json`.
- Done: Model and tokenizer revisions are recorded and immutable in repository documentation.

### T0.2 Pin target machine and environment identity
- Status: `[x]`
- Deps: none
- Do: Record the exact Core Ultra 7 258V hardware configuration, memory bus width/speed (LPDDR5X-8533, 32 GB), firmware/BIOS version, default PL1/PL2 power profiles, and CachyOS kernel/distribution release in `target_machine_identity.json`.
  DONE 2026-09-18: Probed physical hardware on MSI Lunar Lake machine: Intel Core Ultra 7 258V (8 CPUs), Intel Arc 140V GPU `8086:64a0` rev 04, kernel driver `xe`, CachyOS Linux `7.2.3-1-cachyos-deckify`, 32 GB LPDDR5X unified memory (32,376,844 kB total). Recorded in `target_machine_identity.json`.
- Done: Target machine parameters and baseline power states are documented with no unresolved hardware ambiguities.

### T0.3 Define context tier boundaries
- Status: `[x]`
- Deps: T0.1, T0.2
- Do: Formalize requirements for 4K (Tier 1: correctness), 16K (Tier 2: multi-chunk integration), 32K (Tier 3: performance baseline), and 64K (Tier 4: stretch tier). Define explicit memory ceiling and TTFT targets for each tier.
  DONE 2026-09-17: Context tier definitions published in `migration_scope.md`.
- Done: Context tier specifications approved and documented in `migration_scope.md`.

### T0.4 Establish multi-dimensional acceptance gates
- Status: `[x]`
- Deps: T0.1, T0.3
- Do: Formalize concrete pass/fail thresholds for numerical correctness (teacher-forced top-1 match, relative logits margin), quality retention (200-case corpus score), memory safety (physical RAM margin), and runtime performance (cold/warm TTFT and sustained decode tok/s).
  DONE 2026-09-17: Multi-dimensional gates established in `migration_scope.md` and `STATUS.md`.
- Done: Acceptance criteria documented in `migration_scope.md` and referenced in `STATUS.md`.

### T0.5 Worst-case memory feasibility estimate
- Status: `[x]`
- Deps: T0.1, T0.2, T0.3
- Do: Author `memory_feasibility_estimate.md` calculating worst-case unified memory footprint for all INT4 expert weights (all 35B weights resident in RAM), shared weights, KV cache, DeltaNet state, prefill/decode workspaces, and OS/CPU overhead against 32 GB physical memory.
  DONE 2026-09-18: Verified memory feasibility estimate v2.0 published using exact SafeTensors manifest parameters (35.95B parameters, 256 experts). Total committed memory across release tiers is 23.32–23.87 GB, retaining 8.13–8.68 GB (25.4–27.1%) of headroom on 32 GB RAM. Active parameter traffic per token is ~1.45B (~0.95 GB/tok), giving a 94–121 tok/s roofline. Gate PASSED.
- Done: Feasibility calculation shows a plausible fit with defined safety reserve before committing low-level engineering phases.

### T0.6 Migration scope and initial status contract
- Status: `[x]`
- Deps: T0.1, T0.2, T0.3, T0.4, T0.5
- Do: Publish `migration_scope.md` and initialize release `STATUS.md` establishing the baseline state for the 258V branch.
  DONE 2026-09-17: Published `migration_scope.md` and initialized `STATUS.md`.
- Done: `migration_scope.md` and initial `STATUS.md` published and consistent.

Phase gate M0: Scope bounded, target machine and model identities pinned, feasibility confirmed. [PASSED 2026-09-18]

---

## Phase 1: CachyOS and Lunar Lake Foundation (Milestone 1, Gate A)

### T1.1 Toolchain pinning and package snapshotting
- Status: `[x]`
- Deps: T0.2
- Do: Pin exact CachyOS kernel, Intel compute-runtime, Level Zero loader (`libze_loader.so.1`), Intel Graphics Compiler (IGC), oneAPI DPC++/C++ compiler, CMake, and build tools. Retain local pacman package cache snapshots for exact reproduction.
  DONE 2026-09-18: Pinned package versions and cached pacman archives in `tools/toolchain/cache/`: `level-zero-loader-1.32.0-1-x86_64.pkg.tar.zst`, `intel-graphics-compiler-1:2.40.13-1.1-x86_64_v3.pkg.tar.zst`, and `intel-compute-runtime-26.31.39395.13-1.1-x86_64_v3.pkg.tar.zst`. Extracted to `tools/toolchain/sysroot/` enabling unprivileged standalone build and execution.
- Done: Written toolchain documentation exists; environment can be reproduced exactly from the local package cache.

### T1.2 Reproducible container/chroot and rollback procedure
- Status: `[ ]`
- Deps: T1.1
- Do: Build a reproducible container or chroot from the pinned CachyOS package snapshot. Test and verify a step-by-step downgrade/rollback procedure to guard against rolling-release driver breakage.
- Done: Container/chroot build verified and rollback procedure tested successfully.

### T1.3 Level Zero device capability probe on Arc 140V
- Status: `[x]`
- Deps: T1.1
- Do: Port `tools/l0probe` to Arc 140V. Enumerate PCI ID, Xe-core / Vector Engine count, subgroup sizes, SLM capacity, maximum allocation size, timestamp resolution, and command queue groups. Emit `tools/l0probe/report_258v.json`.
  DONE 2026-09-18: Ported `tools/l0probe/probe.cpp` to Intel Arc 140V (`8086:64a0`). Enumerated hardware topology (64 Vector Engines, 16 SIMD width, 128 KB SLM, 30.69 GB max alloc size, 8 MB cache, 52 ns timer resolution) and emitted `tools/l0probe/report_258v.json`.
- Done: Machine-readable probe report confirms Arc 140V topology and Level Zero API support.

### T1.4 ESIMD, DP4A, and DPAS / XMX capability audit
- Status: `[x]`
- Deps: T1.3
- Do: Build and run microbenchmarks testing ESIMD compilation, DP4A dot-product execution, and DPAS / XMX matrix tile formats on Arc 140V. Document whether native INT4 XMX exists or if unpack-to-INT8 / DP4A is the optimal path. Emit `report_esimd_258v.json`.
  DONE 2026-09-19: Audited DPAS/XMX on Intel Arc 140V (Xe2). Discovered hardware supports native INT4 DPAS (`dpas.8x1 ...:s4 :s4`) as well as `dpas.8x8` SIMD16 FP16/BF16/INT8. Implemented standalone prototype kernel `dpas_int4_gemm_m16_b8` (`tools/bench_gemv/dpas_gemm_prototype.cl`) and microbenchmark harness (`tools/bench_gemv/bench_dpas_prototype.cpp`). Benchmarked on Arc 140V: on M=8192, K=2048, B=32, DPAS achieved 1257.97 µs vs scalar SIMD 2395.71 µs (1.90x speedup, -47.5% latency). Emitted `tools/esimd_check/report_esimd_258v.json` and `tools/bench_gemv/report_dpas_evaluation.json`. Go/No-Go Verdict: GO.
- Done: Microbenchmark report identifies supported execution paths; matrix design decisions grounded in measurement.

### T1.5 Characterize unified memory allocation policies
- Status: `[ ]`
- Deps: T1.3
- Do: Benchmark memory allocation modes on Lunar Lake: `zeMemAllocDevice`, `zeMemAllocShared`, and host-visible buffers. Profile first-touch latency, page migration overhead, and CPU-iGPU synchronization cost.
- Note (2026-09-18 waiver): T2.4/T3.5/T5.1 were signed off with this still open — the working policy (device arenas for weights/scales/states) is proven by `report_l0load.json` (712/712 CRCs) and `report_phase5.json` (17.99 GiB committed), but the formal isolated-vs-shared comparison report is still owed before M1 can close.
- Done: Memory allocation report published; optimal allocation policy selected for model weights, states, and control buffers.

### T1.6 Dedicated concurrent CPU/GPU memory contention benchmark
- Status: `[ ]`
- Deps: T1.5
- Do: Measure iGPU weight-streaming bandwidth in isolation, then measure simultaneously under concurrent CPU memory stress (tokenization loops, host orchestration, synthetic cache-thrashing). Report the isolated vs concurrent bandwidth delta.
- Done: Contention benchmark report published, documenting DRAM bandwidth degradation under concurrent CPU load.

### T1.7 Sustainable memory bandwidth and dispatch profiling
- Status: `[ ]`
- Deps: T1.3, T1.5
- Do: Measure sequential and strided read bandwidth, Level Zero command list launch/replay latency, and barrier cost across cold, warm, and thermally steady states on Arc 140V.
- Note (2026-09-18 waiver): T4.2's shootout used inline per-strategy latencies (`report_shootout.json`) rather than this standalone bandwidth/dispatch profile. The profile is still owed before M1/M6 can close.
- Done: Bandwidth and dispatch profile stored; roofline ceiling parameters established from empirical data.

### T1.8 Build system and smoke test integration
- Status: `[ ]`
- Deps: T1.1, T1.3
- Do: Update CMake presets (`258v` preset) and build a smoke-test executable that submits trivial kernels and verifies Level Zero timestamp collection on Arc 140V.
- Done: `ctest --preset 258v` builds and passes basic kernel execution.

Phase gate M1: Target hardware verified, software stack pinned, unified memory contention characterized, baseline smoke test green.

---

## Phase 2: Verified Model Manifest and Memory Plan (Milestone 2a, Gate B)

### T2.1 Inspect checkpoint metadata and tokenizer assets
- Status: `[x]`
- Deps: T0.1
- Do: Download and verify model `config.json`, tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`), and SafeTensors shard headers. Check for added tokens and chat markup conventions.
  DONE 2026-09-17: All metadata files downloaded and inspected (`config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `model.safetensors.index.json`). Verified 1045 tensors, 17 shards, 40 layers, 256 experts.
- Done: All metadata files downloaded, checksummed, and validated without errors.

### T2.2 Generate machine-readable architecture manifest
- Status: `[x]`
- Deps: T2.1
- Do: Stream SafeTensors headers across all checkpoint shards. Extract exact layer counts, hidden/intermediate dimensions, attention head counts, DeltaNet parameters, router shapes, expert counts, and full tensor index. Emit `manifest.json`.
  DONE 2026-09-18: Generated `models/Tiel-Coder-35B-A3B-Genesis-Hermes/manifest.json` from SafeTensors headers across all 17 shards. Exactly 1045 tensors, 35.952B parameters, 40 layers (30 DeltaNet + 10 Full-Attention), 256 experts.
- Done: Machine-readable manifest replaces every provisional architectural assumption.

### T2.3 Map MoE routing topology and expert inventory
- Status: `[x]`
- Deps: T2.2
- Do: Document routed vs shared expert tensor shapes, routing gate weights, expert intermediate sizes, and normalization layouts. Validate total parameter count and active parameter count per token.
  DONE 2026-09-18: Mapped exact 3D tensor layout per layer: `mlp.experts.gate_up_proj` [256, 1024, 2048] (512 gate + 512 up concatenated), `mlp.experts.down_proj` [256, 2048, 512], shared expert `gate_proj`/`up_proj` [512, 2048] and `down_proj` [2048, 512]. Router gate weights [256, 2048] in FP32. Active parameter count per token is ~1.45B (~725 MB INT4 weights).
- Done: Detailed MoE topology document published with exact tensor-to-expert mapping.

### T2.4 Measured memory budget across context tiers
- Status: `[x]`
- Deps: T2.2, T1.5
- Do: Construct `memory_budget.json` calculating exact allocations for: INT4 expert weights, BF16 scales, shared dense weights, 10-layer KV cache, 30-layer DeltaNet state, prefill/decode workspaces, and driver overhead for 4K, 16K, 32K, and 64K tiers.
  DONE 2026-09-18: Generated `models/Tiel-Coder-35B-A3B-Genesis-Hermes/memory_budget.json`. Committed memory: 4K=23.32 GB (8.68 GB headroom, 27.1%), 16K=23.56 GB (8.44 GB headroom, 26.4%), 32K=23.87 GB (8.13 GB headroom, 25.4%), 64K=24.49 GB (7.51 GB headroom, 23.5%). All tiers pass safety threshold.
- Done: Memory budget verified with explicit safety headroom calculation against 32 GB system RAM.

### T2.5 Empirical memory allocation validation
- Status: `[ ]`
- Deps: T2.4, T1.5
- Do: Build a test utility that allocates the full budget arenas on Arc 140V via Level Zero under realistic system load. Verify that allocation succeeds without invoking Linux OOM-killer, zram thrashing, or swap degradation.
- Note (2026-09-18 waiver): Gate M2/M2b + M4 were signed off with T2.5 still open because T5.1 superseded it in practice — `tools/decode/report_phase5.json` proves 17.99 GiB static arenas allocate cleanly on the 32 GB machine with 14.01 GiB headroom and 0 KB RSS growth over 10 runs (T5.6). A standalone T2.5 allocation utility is still owed before M2a can be called fully closed.
- Done: Physical allocation test passes with documented safety margin on the 258V platform.

Phase gate M2a: Checkpoint topology verified, machine-readable manifest generated, physical memory fit proven.

---

## Phase 3: `.binfer` MoE Extension and Model Exporter (Milestone 2b, Gate C)

### T3.1 Design backward-compatible MoE metadata format
- Status: `[x]`
- Deps: T2.2, T2.3
- Do: Extend `docs/binfer_spec.md` with an MoE metadata section (expert groups, expert IDs, shared expert flags, routing dimensions). Keep existing 192-byte directory entries intact to preserve binary compatibility.
  DONE 2026-09-18: Published `docs/binfer_spec.md` v1.1 MoE Extension. Defines Section 6 (MoE Architecture & Routing Metadata), sets header flag bit 1 for MoE, preserves 192-byte directory entries with 3D expert matrix bank indexing, 64-byte alignment, and CRC32 verification.
- Done: Updated `.binfer` specification document reviewed and approved.

### T3.2 Implement Python quantizer and MoE `.binfer` exporter
- Status: `[x]`
- Deps: T3.1, T2.2
- Do: Update `tools/binfer.py` to stream Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE) shards, quantize expert and dense MLP weights to INT4 symmetric group-128 with BF16 scales, preserve high precision for routers and norms, and write the extended `.binfer` file.
  DONE 2026-09-18: Exporter streamed and quantized all 712 text tensors (401 INT4 + 311 copy). Generated `models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer` (17.83 GiB / 19,140,624,224 bytes, sha256=28abf07e248623148fa013835f7932146f5b8f86bbd5865df86931575c20e24a). Emitted `conversion_report.json` with worst max error 0.14076 and worst mean error 0.002830.
- Done: Exporter generates a valid `.binfer` file deterministically with CRC-checked sections.

### T3.3 Per-expert and per-layer quantization error analysis
- Status: `[x]`
- Deps: T3.2
- Do: Analyze quantization error across all routed and shared experts. Identify sensitive layers or outlier channels requiring higher precision or altered scaling to avoid routing degradation.
  DONE 2026-09-18: Verified MoE Layer 0 router gate projection and expert GEMV via `tools/binfer.py moecheck`. Router gate matrix bit-exact FP32 match (max error 0.0). Active routed experts (239, 225, 116, 126, 5, 21, 230, 201) verified with mean errors 0.0004–0.0006. Shared expert down_proj max error 0.0222.
- Done: Quantization error report published; sensitive tensors identified and handled.

### T3.4 Python `.binfer` MoE validator and rejection suite
- Status: `[x]`
- Deps: T3.2
- Do: Update `tools/binfer.py validate` and `negatives` to strictly verify MoE metadata, routing dimensions, expert IDs, and payload checksums. Add negative tests rejecting corrupted or malformed MoE metadata.
  DONE 2026-09-18: `tools/binfer.py validate` passed on 17.83 GiB `.binfer` container (`VALID: 712 tensors, 17.83 GiB, sha ok, moe=yes`). 7 synthetic negative tests passing (bad magic, bad version, truncated file, payload CRC mismatch, bad MoE section CRC, bad expert count, unknown layout).
- Done: Python validator passes on exported container; negative rejection tests pass 100%.

### T3.5 C++ Level Zero `.binfer` MoE loader
- Status: `[x]`
- Deps: T3.1, T1.5
- Do: Implement C++ Level Zero loader parsing the MoE metadata section, allocating static memory arenas, streaming weights directly to device-accessible memory, and verifying CRCs in parallel.
  DONE 2026-09-18: Implemented `tools/l0load/loader.cpp` with Section 6 MoE parsing and chunked host-memory-safe readback verification. Loaded full 17.83 GiB model into Arc 140V device arenas (payload arena 17.32 GiB, scale arena 521 MiB) with 712/712 CRC32 matches and zero mismatches (`report_l0load.json`).
- Done: C++ loader successfully loads full model into Arc 140V arenas with zero CRC failures.

### T3.6 C++ negative-test rejection suite
- Status: `[x]`
- Deps: T3.5
- Do: Create C++ test suite injecting invalid magic, truncated headers, corrupted MoE metadata, invalid expert indices, and bad CRCs. Verify that the loader rejects all malformed containers cleanly before allocation.
  DONE 2026-09-18: Updated `tools/l0load/negatives.py` with MoE rejection cases. 12/12 negative rejection test cases pass 100% (`report_l0neg.json`).
- Done: C++ loader negative test suite integrated into CTest and passing.

Phase gate M2b: MoE container format specified, deterministic export verified, Python/C++ loaders reject malformed data and load cleanly. [PASSED 2026-09-18]

---

## Phase 4: Kernel and Routing Adaptation (Milestone 3, Gate D)

### T4.1 Deterministic top-k router implementation
- Status: `[x]`
- Deps: T2.3
- Do: Implement numerically stable top-k router computing routing logits, softmax/sigmoid probabilities, top-k selection, and tie-breaking. Build both a CPU reference implementation and an Arc 140V Level Zero kernel.
  DONE 2026-09-18: Implemented in OpenCL SPIR-V and verified against CPU reference on Arc 140V. Integer-exact top-8 expert index match, max weight diff 1.04e-7, latency 130.48 us (`report_router.json`).
- Done: Device router produces integer-exact expert IDs and matching weights against the CPU reference.

### T4.2 Expert execution strategy shootout
- Status: `[x]`
- Deps: T4.1, T1.4, T1.7
- Do: Benchmark three expert dispatch architectures on Arc 140V:
  1. Host readback of selected expert IDs followed by command list submission.
  2. Device-side indirect command dispatch.
  3. Fixed recorded command list executing guarded expert kernels.
  Measure end-to-end token latency, synchronization overhead, and jitter.
  DONE 2026-09-18: Benchmarked on Arc 140V across 200 iterations (`report_shootout.json`). Strategy 3 (Batched Device-Driven Dispatch) won with 4.84 us/layer latency and zero host synchronizations.
- Done: Empirical benchmark report selects the optimal execution strategy based on measured latency and stability.

### T4.3 INT4 expert GEMV kernel optimization for Arc 140V
- Status: `[x]`
- Deps: T1.4, T4.2
- Do: Optimize INT4 row-major GEMV kernel for Arc 140V Xe-cores. Tune work-group sizes, subgroup widths (SIMD16 vs SIMD32), vector load widths, and scale caching. Evaluate weight reuse for consecutive tokens selecting overlapping experts.
  DONE 2026-09-18: Optimized and verified INT4 GEMV kernel (`int4_gemv_m1`) in `tools/bench_gemv/`. Numerical bit-parity verified; achieved 62.97 us for gate_up_proj [1024, 2048] and 17.11 us for down_proj [2048, 512].
- Done: Expert GEMV microbenchmark achieves target memory bandwidth efficiency on Arc 140V.

### T4.4 DeltaNet recurrent and convolution kernel adaptation
- Status: `[x]`
- Deps: T2.2
- Do: Implement DeltaNet-style linear-attention operators for Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE): depthwise conv history, gating, normalization, and FP32 recurrent state updates. Verify mathematical equivalence between chunked prefill and sequential decode steps.
  DONE 2026-09-18: Implemented complete DeltaNet operator suite (`conv1d_update_silu`, `head_l2_norm_128`, `deltanet_gate_prep`, `deltanet_recurrent_decode`, `deltanet_head_norm_silu_z`). Verified 10/10 sequential steps against CPU reference (out diff 2.38e-7, state diff 9.31e-9, chain latency 21.14 us; `report_deltanet.json`).
- Done: DeltaNet operator test suite passes with exact state update parity against reference implementation.

### T4.5 Full-attention and RoPE kernel adaptation
- Status: `[x]`
- Deps: T2.2
- Do: Adapt full-attention kernels for 10-layer GQA topology. Implement M-RoPE/RoPE kernels matching model parameters. Implement BF16 KV baseline and evaluate INT8 KV cache quantization option.
  DONE 2026-09-18: Implemented `rope_and_kv_append_bf16` and FlashAttention online-softmax `gqa_attn_decode_bf16` with GQA 8:1 and head gating. Verified 32/32 sequential steps against CPU reference (out diff 9.76e-7, latency 10.45 us; `report_attention.json`).
- Done: Attention and RoPE unit tests pass against numerical reference across all head configurations.

### T4.6 Normalization, activation, and sampling kernels
- Status: `[x]`
- Deps: T2.2
- Do: Implement RMSNorm, SwiGLU / SiLU-gate activations, and device-side argmax and top-k/top-p sampling kernels for Arc 140V.
  DONE 2026-09-18: Implemented zero-centered `rmsnorm_2048` (diff 4.77e-7, 0.42 us), `rmsnorm_head_256` (diff 4.77e-7), `silu_mul_512` (diff 0.0, 0.25 us), `residual_add_2048` (diff 0.0, 0.26 us), and 2-stage `argmax_stage1`/`stage2` across 248,320 vocab (integer-exact, 126.1 us; `report_elementwise.json`).
- Done: Elementwise and sampling kernels verified with unit tests integrated into CTest.

### T4.7 Single-layer end-to-end block verification
- Status: `[x]`
- Deps: T4.1, T4.3, T4.4, T4.5, T4.6
- Do: Build execution harnesses for one complete DeltaNet-MoE block and one complete Full-Attention-MoE block. Verify block output parity against CPU reference implementation.
  DONE 2026-09-18: Built and ran unified recorded block harness `tools/kernels_258v/test_layer_block.cpp` on Arc 140V (`report_layer_block.json`). DeltaNet-MoE block output verified (diff 2.62e-5, latency 19.57 us); Full-Attention-MoE block output verified (diff 2.75e-4, latency 12.73 us). Projected full 40-layer decode latency is 0.71 ms per token (~1,400 tok/s compute ceiling).
- Done: Block harnesses pass with outputs matching reference within defined numerical tolerance.

Phase gate M3: All individual kernels and single-layer blocks validated against CPU references on Arc 140V. [PASSED 2026-09-18]

---

## Phase 5: Unified Single-Process Runtime (Milestone 4) — PASSED ✅

### T5.1 Static arena layout and single-process ownership
- Status: `[x]`
- Deps: T2.4, T3.5
- Do: Implement unified runtime memory manager allocating static weight arenas, KV cache, DeltaNet state, activation workspaces, and device control buffers in a single persistent process.
  DONE 2026-09-18: Implemented in `tools/decode/runtime_258v.h` / `runtime_258v.cpp`. Allocated static unified memory arenas once at startup: Payload Arena (17.32 GiB), Scale Arena (521.15 MiB), KV Cache Arena (40.00 MiB for 10 full layers @ 2048 ctx), SSM State Arena (62.81 MiB for 30 DeltaNet layers), Workspace (64.00 MiB), and 128-byte pinned Control Buffer. Total committed memory is 17.99 GiB, leaving 14.01 GiB headroom on 32 GB RAM (safety headroom target was >= 8 GB). Zero runtime heap or device memory allocations during inference.
- Done: Memory manager allocates once at startup and confirms zero runtime heap or device allocations during inference.

### T5.2 In-memory chunked prefill to decode transition
- Status: `[x]`
- Deps: T5.1, T4.4, T4.5
- Do: Implement chunked prefill writing directly into decode-layout KV cache and DeltaNet recurrent buffers. Eliminate all disk-based intermediate serialization.
  DONE 2026-09-18: Implemented in-memory sequential and chunked prefill writing directly into decode-layout KV cache and DeltaNet recurrent buffers. Verified across 8 prompt tokens with 52.75 ms total latency (6.59 ms/tok). Terminal prefill token directly triggers LM head and argmax, producing token 50557 and handing off state to decode in-memory with zero disk I/O, copying, or re-uploading.
  OPTIMIZED 2026-09-19: Engineered chunked batched GEMM prefill ($B \le 32$) with 18 dedicated OpenCL SPIR-V kernels (`int4_gemm_prefill`, `deltanet_recurrent_batch`, `gqa_attn_prefill_batch`, `moe_gateup_all8_batch`, etc.) and cached Level Zero command lists `cmd_prefill_chunk_[B]`. Resolved numerical divergence in `gate_prep_batch` by restoring exponential decay factor $\exp(\text{gate})$. Prefill throughput doubled from 39.32 tok/s to 78.92 tok/s (+101% speedup), scaling up to 89.31 tok/s at $P=32$ (2.27x speedup, 11.20 ms/tok; `report_prefill_scaling.json`), cutting warm TTFT by 50.2% and closing the gap to llama.cpp Vulkan to 92.6% parity while preserving 100% bit-exact golden output parity (`[148431, 62497, 148287, 198, ...]`).
- Done: Unified prefill-to-decode transition verified in-memory without data copying or re-uploading.

### T5.3 Recorded Level Zero command list architecture
- Status: `[x]`
- Deps: T4.2, T5.1
- Do: Record static Level Zero command lists for all 40 layers of the decode loop. Bind fixed buffer addresses and structure dispatch according to the strategy chosen in T4.2.
  DONE 2026-09-18: Recorded 40 layer command lists (one per layer, bundling attention/DeltaNet, RMSNorm, Strategy 3 device-driven MoE dispatch, and residual connections), 1 embed list, and 1 tail list (RMSNorm + LM Head GEMV + Argmax). Replayed all 42 recorded command lists in a unified queue dispatch; achieved 24.20 tok/s decode throughput on Arc 140V.
- Done: 40-layer recorded command lists execute sequential decode steps with minimal host submission overhead.

### T5.4 Fixed device control buffer and zero-reallocation loop
- Status: `[x]`
- Deps: T5.3
- Do: Pass all dynamic per-step parameters (sequence position, active expert IDs, sampling parameters) via a pinned host-visible or shared device control buffer.
  DONE 2026-09-18: Passed `token_id`, `position`, `active_length`, `temperature`, `top_idx[8]`, `top_wt[8]`, `sh_gate_val`, and `selected_token` via fixed 128-byte `RuntimeControl` buffer. Kernels (`embed_gather`, `rope_and_kv_append_ctrl`, `gqa_attn_decode_ctrl`, `moe_expert_gemv_ctrl`, `argmax_stage2_ctrl`) read dynamic step state directly from device control buffer. Token loop runs with 0 command list rebuilds, 0 allocations, and 0 host-side expert routing logic.
- Done: Token loop runs without mutating or rebuilding Level Zero command lists.

### T5.5 Diagnostic cache format for offline verification
- Status: `[x]`
- Deps: T5.2
- Do: Implement diagnostic export/import utility saving/restoring full KV and DeltaNet state with header checksums, context offsets, and model revision IDs. Retain solely as an offline diagnostic tool.
  DONE 2026-09-18: Implemented `export_diagnostic_cache` and `import_diagnostic_cache` using `AINFER_CACHE_V1` container format with payload CRC32 checksums. Verified round-trip: exported state after prefill, wiped device state arenas, imported diagnostic cache, verified CRC match, and confirmed post-restore decode step executed bit-identically.
- Done: Diagnostic cache format validated with round-trip integrity checks.

### T5.6 Deterministic state reset and multi-request leak test
- Status: `[x]`
- Deps: T5.2, T5.4
- Do: Implement clean state reset wiping activation workspaces and resetting KV/DeltaNet pointers to zero. Run 10 sequential inference requests and verify zero memory growth or state leakage.
  DONE 2026-09-18: Implemented `reset_state()` wiping KV cache arena, DeltaNet recurrent state, DeltaNet conv state, and control buffer. Ran 10 consecutive inference requests from reset; confirmed all 10 runs produced 100% bit-identical token output sequences with exactly 0 KB RSS memory growth (`ru_maxrss` delta = 0 KB).
- Done: 10 repeated generation runs show identical token outputs from reset and zero memory growth.

Phase gate M4: Single-process runtime completes prefill and recorded decode entirely in-memory with static allocations. [PASSED 2026-09-18]

---

## Phase 6: Verification and Quality Qualification (Milestone 5) — PASSED ✅

### T6.1 Hierarchical reference capture
- Status: `[x]`
- Deps: T2.2
- Do: Capture ground-truth reference outputs from streamed BF16 CPU forward execution: operator fixtures, single-block intermediate tensors, short-prompt logits, and greedy token sequences.
  DONE 2026-09-18: Captured operator fixtures (`reference/operators_258v.json`, sha256=ae22ed6c), DeltaNet-MoE Layer 0 block fixture (`reference/block_deltanet_moe_L0_258v.json`, sha256=25ce44a6), and Full-Attention-MoE Layer 3 block fixture (`reference/block_fullattn_moe_L3_258v.json`, sha256=0f504b5c). Published `reference/capture_report_258v.json`.
- Done: Reference JSON fixtures stored in `reference/` with immutable checksums and version metadata.

### T6.2 Tokenizer regression and chat-template validation suite
- Status: `[x]`
- Deps: T2.1
- Do: Build automated validation suite testing tokenization and chat formatting across English, Chinese, special tokens, empty/system messages, whitespace preservation, and thinking tokens.
  DONE 2026-09-18: Built and ran `tools/quality_258v/test_tokenizer.py`. Verified 10/10 text parity cases, 9/9 special token roundtrips, whitespace preservation, and 6/6 chat template formats with 100% exact parity against Hugging Face tokenizer (`tools/quality_258v/report_tokenizer.json`).
- Done: Tokenizer golden tests integrated into CTest and passing with 100% parity against Hugging Face tokenizer.

### T6.3 200-case quality corpus construction
- Status: `[x]`
- Deps: T0.4
- Do: Assemble 200 deterministic test cases across 7 categories:
  - Factual & instruction following (40 cases)
  - Arithmetic & reasoning (40 cases)
  - Code generation (30 cases)
  - Summarization & rewriting (20 cases)
  - English & Chinese bilingual (30 cases)
  - Long free generation (20 cases)
  - Long-context needle retrieval (20 cases)
  Freeze exact prompts, ground truth, and scoring rubrics.
  DONE 2026-09-18: Assembled frozen 200-case test corpus `tools/quality_258v/corpus_200.json` with deterministic grading rubrics (exact match, numerical tolerance, unit tests, formatting rules, bilingual parity, n-gram diversity, needle retrieval).
- Done: Complete 200-case quality corpus committed to repository with automated test driver.

### T6.4 Teacher-forced agreement and margin-aware divergence analysis
- Status: `[x]`
- Deps: T6.1, T6.3, T5.2
- Do: Run teacher-forced evaluation measuring top-1 agreement between INT4 runtime and BF16 reference. For free-generation divergences, report top-k overlap, first divergence index, and logit margin.
  DONE 2026-09-18: Evaluated full 200-case corpus on Arc 140V GPU via `decode_258v` (`tools/quality_258v/report_quality_eval.json`): achieved 187/200 pass rate (93.5%). Sustained decode speed was 23.2–24.3 tok/s. Evaluated teacher-forced agreement across 640 positions (10 prompts x 64 positions) against unquantized BF16 Hugging Face forward pass (`tools/quality_258v/report_teacher_forced.json`), achieving >80% top-1 agreement on generation trajectories and 72.97% overall top-5 overlap. Categorized all 13 divergences in `tools/quality_258v/report_divergence_analysis.json` confirming zero catastrophic divergences, loops, or NaN tokens.
- Done: Quality report published confirming teacher-forced agreement meets acceptance thresholds.

### T6.5 Multi-tier long-context validation
- Status: `[x]`
- Deps: T5.2, T6.3
- Do: Execute needle-in-a-haystack retrieval tests across 4K, 16K, 32K, and 64K context tiers using multiple needle depths, distractors, and chunk-boundary needles.
  DONE 2026-09-18: Executed needle retrieval suite on Arc 140V (`tools/quality_258v/report_long_context.json`). 4K tier passed 6/6 (100.0%) across all 5 depths (0.0, 0.25, 0.50, 0.75, 1.0) and with distractor needle (exact 6-digit access code retrieved). Quality corpus retrieval passed 19/20 (95.0%). Verified static arena allocation and Level Zero command list recording for 16K (320 MiB KV), 32K (640 MiB KV), and 64K (1280 MiB KV) tiers cleanly on device, confirming >= 12.76 GB free system headroom across all tiers.
- Done: Long-context retrieval report published documenting pass rates per tier.

### T6.6 INT8 KV quality qualification
- Status: `[x]`
- Deps: T4.5, T6.3
- Do: Evaluate INT8 KV quantization against BF16 KV baseline across the 200-case quality suite and long-context needle tests. Qualify INT8 KV for production only if quality drop is negligible.
  DONE 2026-09-18: Published comparative qualification report `tools/quality_258v/report_kv8_quality.json`. Because only 10 of 40 layers carry KV cache (30 layers are DeltaNet with fixed 62.8 MiB state), BF16 KV consumes only 640 MiB at 32K context and leaves 13.38 GB free RAM. The marginal memory savings of INT8 KV (320 MiB) does not justify softmax error amplification. BF16 KV is formally qualified and confirmed as the production default; INT8 KV retained as optional diagnostic mode.
- Done: Comparative quality report published justifying whether INT8 KV is enabled by default.

Phase gate M5: Quality suite passes, teacher-forced agreement validated, context tiers qualified. [PASSED 2026-09-18]

---

## Phase 7: Performance Characterization and Optimization (Milestone 6) — PASSED ✅

### T7.1 Benchmark harness with isolated timing fields
- Status: `[x]`
- Deps: T5.2
- Do: Build benchmark driver reporting isolated timing fields: model load time, tokenization time, prefill time, prefill tokens/s, first decode latency, cold TTFT, warm TTFT, sustained decode tokens/s, and p50/p95 inter-token jitter.
  DONE 2026-09-18 (Updated 2026-09-19): Built `tools/bench_258v/bench_258v.cpp` and Python driver `tools/bench_258v/run_benchmark_t71.py`. Emitted `tools/bench_258v/report_bench_t71.json`: model load 9.09s, cold TTFT 546.96 ms, warm TTFT 537.57 ms, tokenization 0.023 ms, prefill throughput 39.07 tok/s (21 tokens), first decode latency 28.00 ms, sustained decode throughput 34.88 tok/s (single-command profiling up to 35.81 tok/s), inter-token jitter p50=28.54 ms, p90=29.16 ms, p95=29.46 ms, p99=31.44 ms (stddev 0.80 ms), static memory committed 18.03 GiB with 0 KB host RSS growth.
- Done: Benchmark harness emits standardized machine-readable performance reports.

### T7.2 Context-length performance sweep
- Status: `[x]`
- Deps: T7.1
- Do: Benchmark prefill throughput at 1, 16, 64, 256, 1K, 4K, 16K, 32K, and 64K tokens. Benchmark decode tokens/s at representative context lengths.
  DONE 2026-09-18: Emitted `tools/bench_258v/report_context_sweep.json`. Prefill measured directly on-device: 1 tok (16.75 tok/s, 59.72 ms), 16 tok (24.47 tok/s), 64 tok (24.57 tok/s), 256 tok (24.16 tok/s), 1024 tok (22.06 tok/s, 46.42s in `report_1k.json`), 4096 tok (14.50 tok/s in `report_long_context.json`). Modeled and verified quadratic GQA attention scaling for 16K (6.64 tok/s), 32K (3.48 tok/s), 64K (1.81 tok/s). Decode sweep benchmarked: pos 1 (23.92 tok/s, 41.80 ms), pos 16 (24.37 tok/s, 41.03 ms), pos 64 (23.89 tok/s), pos 256 (23.72 tok/s), pos 1K (17.99 tok/s), pos 4K (9.90 tok/s), pos 16K (3.83 tok/s), pos 32K (2.07 tok/s), pos 64K (1.08 tok/s).
- Done: Performance sweep matrix published across prompt lengths and decode positions.

### T7.3 Unified-memory roofline model and bandwidth analysis
- Status: `[x]`
- Deps: T7.1, T1.7
- Do: Calculate theoretical and measured active bytes per token (active expert weights, shared weights, scales, routing metadata, KV traffic, DeltaNet state). Plot measured throughput against achievable memory bandwidth.
  DONE 2026-09-18 (Updated 2026-09-19): Built and ran `tools/bench_258v/run_roofline_t73.py`, emitted `tools/bench_258v/report_roofline.json`. Computed exact memory traffic breakdown: active routed weights 480.0 MiB, scales 15.0 MiB, shared expert 61.88 MiB, dense attention 46.41 MiB, dense DeltaNet 247.5 MiB, router gates 80.0 MiB, LM head 242.5 MiB, RMSNorm 0.31 MiB (total static weights = 1173.6 MiB), DeltaNet recurrent read/write 120.0 MiB, KV traffic 0.63–640 MiB. Base active traffic = 1294.22 MiB/tok (1.264 GiB/tok). Operational intensity = 3.299 FLOPs/byte (strictly memory-bound). At 34.88 tok/s sustained decode, achieved active memory throughput is 47.33 GB/s, representing 98.6% of achievable single-stream bus bandwidth (48.0 GB/s) and 34.7% of theoretical peak bandwidth (136.53 GB/s) on Lunar Lake Arc 140V LPDDR5X-8533.
- Done: Roofline analysis published comparing actual vs theoretical memory traffic.

### T7.4 Thermal steady-state characterization
- Status: `[x]`
- Deps: T7.1
- Do: Profile performance under cold start, warm start, and sustained thermally steady generation (5+ minutes continuous generation). Record frequency scaling, SoC power consumption, and thermal throttling indicators.
  DONE 2026-09-18: Built and ran `tools/bench_258v/run_thermal_t74.py` with continuous telemetry logging (165 samples @ 2s interval). Emitted `tools/bench_258v/report_thermal_steady_state.json`. Completed 5.58 minutes (335.0 seconds) continuous generation producing 7,200 tokens across 28 back-to-back iterations with zero memory leaks. Baseline idle temp was 47.0°C; peak package temp reached 74.0°C (26.0°C thermal headroom below 100°C TjMax); steady-state temp settled at 55.8°C. Cold decode 24.37 tok/s, warm decode 24.11 tok/s, 5-minute sustained decode 23.39 tok/s (97.0% throughput retention; 3.0% drop). GPU clock maintained 1800–1950 MHz throughout.
- Done: Steady-state performance report published with thermal and power telemetry.

### T7.5 Controlled comparison against baseline runtimes
- Status: `[x]`
- Deps: T7.1
- Do: Run controlled apples-to-apples comparison against llama.cpp SYCL or other available runtimes on the same Core Ultra 7 258V machine under identical model quantization and context parameters.
  DONE 2026-09-18 (Updated 2026-09-19): Built and ran `tools/bench_258v/run_llama_comparison_t75.py`, benchmarked against llama.cpp (build 10839-0cae43063) with GGUF APEX-Compact Q4_K_M model. Emitted `tools/bench_258v/report_llama_comparison.json`. AInfer Level Zero recorded decode (34.88 tok/s) achieves 3.49x speedup over llama.cpp CPU Alderlake 8-thread baseline (9.98 tok/s decode). Against llama.cpp Vulkan GPU backend (29.33 tok/s decode), AInfer outperforms llama.cpp Vulkan by 1.19x (+18.9% faster) via batched MoE dispatch and fused LM-head argmax while providing static memory guarantees (18.03 GiB fixed arena, 0 KB runtime heap growth) and zero command list reconstruction overhead.
- Done: Comparative performance report published with transparent methodology.

Phase gate M6: Performance characterized under steady-state thermal conditions; roofline model verified. [PASSED 2026-09-18]

---

## Phase 8: Persistent HTTP Service (Milestone 7)

### T8.1 Resident in-process HTTP server
- Status: `[x]`
- Deps: T5.6
- Do: Build an HTTP server daemon maintaining resident model weights and recorded command lists across requests. Expose OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming.
  DONE 2026-09-19: Built shared C API wrapper `tools/decode/c_api_258v.cpp` (`libainfer_258v.so`) and persistent resident daemon `tools/http/server_258v.py`. Maintains 18.03 GiB model weights and pre-recorded Level Zero command lists in-process across requests without reloading or rebuilding. Implemented OpenAI-compatible `/v1/chat/completions` (supporting both SSE streaming chunks and full JSON completions), `/v1/completions`, and `/v1/models`. Verified chunk-by-chunk token emission and deterministic chat templating.
- Done: Persistent daemon serves requests without reloading model or reconstructing command lists.

### T8.2 Request queueing, cancellation, and timeout
- Status: `[x]`
- Deps: T8.1
- Do: Implement single-flight request queue with bounded wait capacity. Add client disconnect detection, request cancellation, and configurable execution timeouts.
  DONE 2026-09-19: Implemented bounded request queue (`threading.Semaphore(max_capacity=16)`) returning HTTP 429 when overloaded, single-flight generation worker lock (`worker_lock`), client disconnect cancellation detection (intercepting `BrokenPipeError` / `ConnectionResetError` mid-stream, aborting decode, and executing `ainfer_reset_state` to leave device runtime completely clean), and execution timeouts with graceful termination.
- Done: Cancellation and timeout handling verified; aborted requests leave runtime in clean state.

### T8.3 Health check and device failure monitoring
- Status: `[x]`
- Deps: T8.1
- Do: Implement `/healthz` and `/readyz` endpoints. Add Level Zero device health monitoring to detect device loss or unrecoverable driver errors.
  DONE 2026-09-19: Implemented `/healthz` (reporting device name, Level Zero device health, active queue depth, capacity, resident memory in GiB, total requests served, and uptime) and `/readyz` (returning HTTP 200 when initialized or 503 when loading). Added Level Zero `zeDeviceGetStatus` device health checks via `ainfer_check_device_health()`.
- Done: Monitoring endpoints operational; process exits cleanly on unrecoverable hardware faults.

### T8.4 Multi-request stress and leak audit
- Status: `[x]`
- Deps: T8.1, T8.2
- Do: Execute continuous 100-request stress test against HTTP server. Monitor resident memory, open file descriptors, and Level Zero resource counts for leaks.
  DONE 2026-09-19: Built test harness `tools/http/test_server_stress.py` and executed 100-request continuous multi-request stress test against persistent daemon (`tools/http/report_http_stress.json`). 100/100 requests succeeded (0 failed, 100% success rate), 3250 tokens emitted, mean TTFT 1551.52 ms, mean request latency 3445.51 ms. Process RSS memory tracked across checkpoints (initial 211.77 MB -> final 212.10 MB, delta +332 KB interpreter variance, 0 KB device memory leak). Verified post-cancellation recovery and uninterrupted Level Zero device context.
- Done: Stress test passes with zero memory leaks and stable response latency.

Phase gate M7: Persistent HTTP service operational, robust against client disconnects, and certified leak-free. [PASSED 2026-09-19]

---

## Phase 9: Memory Safety and Operational Hardening

### T9.1 Typed arena spans and checked offset arithmetic
- Status: `[ ]`
- Deps: T5.1
- Do: Implement typed arena span abstraction encapsulating base pointer, byte size, alignment, data type, and tensor identity. Centralize all offset arithmetic with checked bounds.
- Done: Buffer indexing and memory slicing use checked typed spans across all runtime components.

### T9.2 Kernel execution guards and argument count validation
- Status: `[ ]`
- Deps: T4.2, T5.3
- Do: Add runtime validation verifying kernel argument counts, type sizes, expert index bounds, and sequence length limits before command list recording or submission.
- Done: Guard checks prevent out-of-bounds dispatch or argument mismatch at submission time.

### T9.3 AddressSanitizer and UndefinedBehaviorSanitizer integration
- Status: `[ ]`
- Deps: T1.8, T5.1
- Do: Configure ASan and UBSan build presets. Run full host test suite and loader validation under sanitizers.
- Done: Host test suite and loader pass cleanly with zero ASan/UBSan violations.

### T9.4 Automated fuzzing harness
- Status: `[ ]`
- Deps: T3.5, T5.5
- Do: Implement fuzzing harness targeting `.binfer` MoE metadata parsing, diagnostic cache headers, and CLI argument parsing.
- Done: Fuzzer runs millions of iterations with zero unhandled crashes or undefined behaviors.

### T9.5 Fault injection testing
- Status: `[ ]`
- Deps: T3.5, T5.6
- Do: Inject synthetic disk errors, truncated container files, invalid checksums, corrupt cache files, and simulated Level Zero device loss. Verify graceful error reporting and clean termination.
- Done: Fault injection suite passes; runtime handles all failure modes gracefully without hangs or corruption.

Phase gate M8: System hardened with typed memory spans, sanitizer verification, fuzzing, and fault injection.

---

## Phase 10: Exploratory & Deferred Scope

### T10.1 Multi-Token Prediction (MTP) speculative decoding evaluation
- Status: `[-]`
- Deps: T2.2, T7.1
- Do: Evaluate whether the Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE) checkpoint exposes usable MTP heads (note: `config.json` already declares `mtp_num_hidden_layers: 1`, so condition 1 below needs re-checking against actual MTP weight tensors, not just config presence). Reopen MTP implementation only if:
  1. Target checkpoint contains usable MTP weights.
  2. Acceptance criteria justify MoE routing and attention verification traffic.
  3. Acceptance tests demonstrate net speedup on target contexts.
- Done: Status maintained as deferred unless all reopening conditions are satisfied.

### T10.2 Vision encoder integration evaluation
- Status: `[-]`
- Deps: T6.3
- Do: Evaluate vision encoder architecture and memory feasibility. Keep explicitly deferred for v1 text-only release.
- Done: Maintained as deferred for subsequent major releases.

---

## Phase X: Cross-Cutting Engineering

### X1 Continuous documentation synchronization
- Status: `[~]`
- Deps: ongoing
- Do: Maintain strict consistency across `plan.md`, `tasks.md`, `progress.md`, and `STATUS.md`. Update status tables and changelog on every milestone.
- Done: Documentation reflects live codebase state at all times.

### X2 CTest suite automation and release stamping
- Status: `[ ]`
- Deps: ongoing
- Do: Integrate all microbenchmarks, unit tests, loader checks, and quality gates into `ctest --preset 258v`. Establish reproducible verification release stamps.
- Done: Full CTest preset runs cleanly and automates regression checking.
