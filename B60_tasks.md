# AInfer Task Breakdown

Tasks derived from `plan.md`. Each task has a stable ID, dependencies, and a concrete
completion check. IDs are stable so they can be referenced from commits and issues.

Legend:

- Status: `[ ]` pending, `[~]` in progress, `[x]` done, `[-]` dropped.
- `Deps`: task IDs that must complete first.
- `Done`: objective completion check.

---

## Phase 0: Environment and Foundations (Milestone 0)

### T0.1 Pin the software and hardware stack
- Status: `[x]`
- Deps: none
- Do: Record Ubuntu, kernel, Intel compute-runtime, Level Zero loader, DPC++/ESIMD
  toolchain, CMake, and build-tool versions in a reproducible document or lockfile.
  DONE 2026-09-07: `docs/toolchain.md` — Ubuntu 26.04, kernel 7.0.0-31, B60 `8086:e211`,
  Intel GPU driver 26.05.37020.3, L0 loader 1.28.2, oneAPI/DPC++ 2026.1.1,
  CMake 4.2.3, venv `~/.venvs/ainfer`. Core L0 headers missing system-wide → vendored
  v1.28.2 under `tools/l0probe/include/`; link versioned `libze_loader.so.1` directly
  (no dev symlink without sudo).
- Done: A fresh machine can reproduce the toolchain from written instructions.

### T0.2 Level Zero device-capability probe
- Status: `[x]`
- Deps: T0.1
- Do: Build a small tool that enumerates device/revision IDs, driver/loader versions,
  memory heaps, allocation limits, module formats, floating-point modes, subgroup
  widths, and synchronization features.
  DONE 2026-09-07: `tools/l0probe/` (probe.cpp + CMake, `report_b60.json`) — exactly one
  L0 device: B60 `8086:e211`, API 1.14; 5×4×8 = 160 EUs, SIMD-16, subgroups [16,32],
  SLM 128 KiB; single DDR heap 22.71 GiB, max alloc 22.71 GiB; FP16+FP64+INT64+DP4A,
  SPIR-V 1.5; L3 ~18 MiB; compute+copy and copy queue groups; 52 ns 64-bit timestamps.
  17.28 GiB T1.6 budget fits with ~5.4 GiB margin.
- Done: Probe prints a machine-readable capability report for the Arc Pro B60.

### T0.3 ESIMD and XMX capability check
- Status: `[x]`
- Deps: T0.2
- Do: Confirm ESIMD compiles and runs on the pinned toolchain; enumerate practical XMX
  data-type combinations. Record whether native INT4->FP16 matrix paths exist.
  DONE 2026-09-07: `tools/esimd_check/` (icpx 2026.1.1, `report_esimd.json`) — ESIMD
  vadd passes SG16+SG32; DPAS tiles exact (maxdiff 0): int8 8x16x32, fp16 8x16x16,
  bf16 8x16x16. **No native INT4 XMX**: `joint_matrix` has no int4 element type, so the
  INT4 GEMV design is unpack-to-int8 (VNNI/DPAS) or unpack-to-fp16/bf16, then DPAS —
  benchmark both in T3.2 before fixing the format.
- Done: Report lists supported/absent kernel paths; no assumption of INT4 XMX remains.

### T0.4 Build system and smoke test
- Status: `[x]`
- Deps: T0.1
- Do: Create CMake presets and a smoke-test executable that runs a trivial kernel and
  reads Level Zero timestamp events.
  DONE 2026-09-07: root `CMakeLists.txt` + `CMakePresets.json` (`b60` icpx / `host`
  presets, `ctest --preset b60` passes); smoke runs ESIMD kernels on the B60 and
  captures SYCL profiling + raw L0 `zeDeviceGetGlobalTimestamps` timestamps.
- Done: Kernel runs, timestamps are captured, build reproducible from presets.

### T0.5 Reference-runtime baseline capture
- Status: `[x]`
- Deps: T0.1
- Do: Run an available reference runtime (e.g., PyTorch/llama.cpp) on the same hardware
  and record baseline latency, throughput, and memory.
  DONE 2026-09-07: llama.cpp SYCL (build ddd4ec142) on B60, Qwen3.8-27B Q4_K_XL GGUF,
  full offload: pp512 183.87 t/s, tg128 14.86 t/s (`reference/baseline_t05.json`).
  tg128 is the number for AInfer to beat (roofline target 25–28 tok/s).
- Done: Baseline numbers stored with environment metadata.

Phase gate: device identified, compiled kernel runs, timestamps work, environment reproducible.

---

## Phase 1: Model Specification and Reference Suite (Milestone 1, Gates A & D)

### T1.1 Pin exact model and tokenizer revision
- Status: `[x]`
- Deps: none
- Do: Fix the exact repository, checkpoint revision, and tokenizer revision (Gate A).
  Pinned 2026-09-07: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, local mirror
  `models/Qwen3.8-27B` (18/18 SafeTensors shards open, headers valid; small-file CRCs
  partially mismatched — see progress.md risks).
- Done: Revisions and checksums recorded and immutable in the repo docs.

### T1.2 Generate architecture manifest
- Status: `[x]`
- Deps: T1.1
- Do: Emit the machine-readable manifest. DONE 2026-09-07: `models/Qwen3.8-27B/manifest.json`
  v1.0 (~250 KB): `qwen3_5` hybrid, 64 text layers (48 linear + 16 full, interval 4),
  hidden 5120 / intermediate 17408 / vocab 248320, GQA 24Q/4KV head_dim 256, M-RoPE
  [11,11,10] θ=1e7, RMSNorm 1e-6, SiLU + swish gate, untied, BF16/FP32-SSM, MTP 1 layer,
  vision deferred; full inventory of 1199 tensors with dtype/shape/bytes per shard
  (tensor bytes 51.75 GiB = index total, coverage 100%); sha256 of 14 small files.
- Done: Machine-readable manifest replaces every architectural assumption.

### T1.3 Confirm or drop MTP/speculative heads
- Status: `[x]`
- Deps: T1.2
- Do: Verify whether the pinned checkpoint contains MTP/speculative decoding heads.
  CONFIRMED 2026-09-07: `mtp_num_hidden_layers` 1, `mtp_use_dedicated_embeddings` false.
  T7.2 unblocked on existence; still gated on acceptance measurements.
- Done: MTP is confirmed with parameters, or explicitly removed from the initial roadmap.

### T1.4 Capture numerical reference outputs (Gate D)
- Status: `[x]` (CPU-feasible part DONE 2026-09-07; REAL HF block refs DONE 2026-09-08 via
transformers 5.16.1 (attn_block_L3, rope_applied, corrected mlp_layer0 with (1+w) norm —
old file was wrong by the norm factor); full-model logits/greedy RE-SCOPED 2026-09-11
to box-native: the "needs 64 GB host" framing is DROPPED — no bigger host is coming
and this project optimizes for the current platform (B60 + 30 GB box). The streamed
one-shard-at-a-time CPU forward (`tools/forward/fwd_cpu.py`, S1 BF16 4 tokens/141s,
S2 INT4 top-1 agreement 4/4) already runs full-model forwards HERE.
LOGITS BATCH DONE 2026-09-13 (`tools/forward/fwd_batch_t14.py`,
`reference/fwd_T14_batch.json` + per-prompt full [T,V] BF16 logits `.pt`):
all 6 staged prompts, BF16 + INT4 forwards each (~9.5 min/prompt): INT4-vs-BF16
top-1 53/60 (P0 7/7, P1 10/12, P2 8/9, P3 6/8, P4 9/10, P5 13/14); divergences
at mid/late positions (P1 pos 3-4 earliest) — margin characterization open,
greedy leg covered by T1.5 47/48)
- Deps: T1.1
- Do: Record reference outputs for individual operators, one transformer block, short
  prompt logits, and greedy token sequences. Store prompts, tolerances, and versions.
  CAPTURED in `reference/`: `operators_t14.npz` (float64 RMSNorm/SiLU-gate/GQA-attn/RoPE fixtures,
  seed 0); `mlp_layer0.json` (real-weight layer-0 RMSNorm+MLP+residual, torch CPU);
  `weight_stats.json` (streamed min/max/mean/std for all 1199 tensors);
  `tokenizer_t14.json` (plain-text round-trip OK; chat markup needs added-token registration — T4.3);
  `greedy_prompts_t14.json` (prompts + ids staged for later forward run).
  RoPE fixture assumes half-rotation text path; M-RoPE section split needs HF cross-check.
- Done: Reference fixtures committed with tolerances and metadata.

### T1.5 Capture quality baseline
- Status: `[~]` (corpus DONE 2026-09-07; metric run RE-SCOPED 2026-09-11 to box-native —
  no bigger host is coming, see T1.4)
- Deps: T1.1
- Do: Establish greedy-agreement quality on the small fixed validation corpus using
  ONLY current-platform executors: (a) streamed-CPU BF16 greedy as truth
  (`tools/forward/fwd_cpu.py`, ~35 s/token, batch as background job); (b) our
  recorded loop greedy (`decode_l0`, seconds); (c) llama.cpp SYCL reference
  (`~/llama.arc`, Q4_K_XL on `/mnt/usb/Test/`) as the behavioral cross-check
  (different quant — agreement metric, not oracle). Metric: per-position top-1
  agreement + divergence characterization over the 6 corpus prompts (short
  completions; ≤256-token contexts run natively per T6.2). No perplexity run,
  no 64 GB machine — dropped. PILOT DONE 2026-09-11 (`tools/t15/pilot_greedy.py`,
  `tools/t15/report_pilot.json`): corpus #2 — BF16 vs loop top-1 8/8, top-5
  sets 4-5/5 ("...To solve 84 * "); llama cross-check correct toward 126.
  Pipeline proven (~25 min/prompt); remaining: batch the other 5 prompts.
- Done: Baseline quality metrics and corpus stored.

### T1.6 Compute and verify memory budget (Gate C)
- Status: `[x]` (calculation DONE 2026-09-07; allocation PROVEN same day: 14.5 + 0.39 GiB
  static arenas on B60 in 72.5 ms — T2.6. SSM-state estimate still theoretical until T1.4 cross-check)
- Deps: T1.2, T0.2
- Do: Derive the hybrid memory budget: full-attn KV over 16 layers only
  (`2 * 16 * 4 * 256 * max_context * bytes_per_element`) PLUS 48-layer FP32 SSM/conv
  state, at 4,096-token v1 cap (native 262,144 out of scope). Include alignment,
  scales/zero points, unquantized tensors (norms, SSM, MTP, vision-excluded), logits
  (vocab 248320), reductions, command resources, driver overhead, allocator granularity.
  CALC RESULT (`models/Qwen3.8-27B/memory_budget.json` v1.0): scenario A (embed BF16,
  rest incl. lm_head INT4 g128 symmetric) weights 14.88 GiB; KV BF16@4K 256 MiB;
  SSM state ~144 MiB + conv ~5.6 MiB (estimate, verify in T1.4); activations 0.5 +
  runtime 1.5 GiB envelopes; total ~17.28 GiB → margin ~6.7 GiB on 24 GB. FITS.
  Verify with real Level Zero allocation tests once B60 is available.
- Done: Budget fits 24 GB with safety margin and is validated by allocation tests.

Phase gate: assumptions replaced by verified values; projected model fits with margin.

---

## Phase 2: Model Container and Exporter (Milestone 2)

### T2.1 Define `.binfer` container format
- Status: `[x]`
- Deps: T1.2
- Do: Specify magic/version/endianness/alignment, source-model identity, architecture
  metadata, tokenizer identity/assets, tensor directory (names, shapes, logical/storage
  dtypes, offsets, lengths, layout IDs), quantization metadata, per-section checksums,
  independently aligned payloads.
  DONE 2026-09-07: `docs/binfer_spec.md` v1.0 — 128B LE header, 5 CRC-checked sections,
  192B tensor entries (866 for Qwen3.8-27B text v1), INT4-sym-g128 + BF16 scales with
  exception table (embed/norms/SSM scalars kept, visual rejected), layout registry
  (0 required, 1–15 reserved for T3.3), normative loader reject rules = T2.2 hooks.
- Done: Written format spec with versioning and layout IDs.

### T2.2 Implement strict container loader
- Status: `[x]` (Python reference DONE 2026-09-07; C++ loader `tools/l0load/loader.cpp`
  written same day — parses per spec, rejects bad magic/layout, loads + verifies 866/866.
  C++ NEGATIVE SUITE DONE 2026-09-11 (`tools/l0load/negatives.py`, ctest
  `l0negatives`, 8/8: valid tiny-load rc0 + bad_magic/bad_version/truncated/
  dir_crc/unknown_layout/payload_oob/payload_len_oob all rc2 with named
  diagnostics; entry mutations recompute dir CRC so rejects attribute to the
  intended gate). Hardening added for the suite: version gate (was accepted
  unchecked), file-size-gated section/dir/entry spans BEFORE any device
  allocation (a hostile d_off previously reached arena sizing — spec §11
  "validation before allocation" now holds on the C++ path too).
  NOTE (resolved 2026-09-12): full-16GB recheck passed on a quiet box
  (866/866, 45 s warm) — the earlier OOM-kill was concurrent torch batch
  load, not code.)
- Deps: T2.1
- Do: Validate format version, model identity, shapes, offsets, alignment, quantization
  metadata, and checksums before allocating GPU memory. Reject incompatible layout IDs.
  `tools/binfer.py validate` enforces all §11 rules; `negatives` passes 6/6 rejection
  cases; real file validates (866 tensors, 14.88 GiB, sha ok). The C++ L0 loader and
  allocation path are implemented and hardware-verified by T2.6; only equivalent
  malformed-input coverage for the C++ path remains.
- Done: Loader passes malformed-input and version-rejection tests.

### T2.3 Implement deterministic INT4 quantizer
- Status: `[x]`
- Deps: T1.4, T2.1
- Do: Load pinned BF16 SafeTensors from `models/Qwen3.8-27B`, quantize eligible matrices
  group-wise (initial group size 128), keep sensitive tensors higher precision where
  validation requires (RMSNorm scales, SSM `A_log`/`dt_bias`/conv, MTP head initially).
  GGUF file is NOT an input here.
  DONE 2026-09-07: `tools/binfer.py quantize` → `qwen3.8-27b-text-int4g128.binfer`
  (15,978,603,616 B, 505 tensors INT4-sym-g128, worst max_abs_err 0.1053);
  dequantized layer-0 MLP matches `reference/mlp_layer0.json` (max diff 0.00171).
- Done: Identical inputs produce byte-identical output; dequant error within thresholds.

### T2.4 Tensor packing and layout (baseline)
- Status: `[x]`
- Deps: T2.3
- Do: Implement a simple baseline packing/layout plus round-trip and inverse-layout tests.
  Keep physical swizzle selection deferred to benchmarks (see T3.x).
  DONE 2026-09-07: layout-0 row-major pack/unpack round-trips through the validator;
  tensor-level pack/unpack and loader CRCs prove layout fidelity. The former `mlpcheck`
  0.00171 result was against an incorrect `w` RMSNorm fixture and is superseded by the
  corrected (1+w) quality check tracked under T3.8. Swizzle IDs 1–15 reserved for T3.3.
- Done: Round-trip and inverse-layout tests pass against exporter output.

### T2.5 Conversion report
- Status: `[x]`
- Deps: T2.3
- Do: Emit report with tensor sizes, quantization error, final file size, and checksums.
  DONE 2026-09-07: `models/Qwen3.8-27B/conversion_report.json` (866 tensors, per-tensor
  max/mean abs err + worst-10, file size, SHA-256 `5ef77c12…`).
- Done: Report generated and stored for the pinned model.

### T2.6 Load tensors into static Level Zero allocations
- Status: `[x]`
- Deps: T2.2, T1.6
- Do: Load the full converted model into static device allocations using large arenas.
  DONE 2026-09-07: `tools/l0load/` — 2 static arenas (payload 15,571,457,024 B +
  scales 406,978,560 B, alloc 72.5 ms), sync-immediate streaming upload, **full
  15 GB readback: 866/866 tensors CRC-verified** (`report_l0load.json`). H2D
  1.02 GB/s (USB-source bound), D2H 3.92 GB/s.
- Done: Full model loads reproducibly; every tensor verifiable against exporter output.

Phase gate: entire converted model loads reproducibly and is tensor-verifiable.

---

## Phase 3: Kernel Microbenchmarks (Milestone 3)

### T3.1 Bandwidth and dispatch baselines
- Status: `[x]`
- Deps: T0.4
- Do: Measure copy/read bandwidth over representative sizes; measure dispatch, barrier,
  event, and command-list replay overhead; record warm/cold timings.
  DONE 2026-09-07: `tools/bench/` (`report_t31.json`) — device roof **~437 GB/s**
  sustained (L0 D2D 256 MB–4 GB + ESIMD block copy agree; 96% of 456 advertised);
  PCIe pinned H2D 13.8 / D2H 11.9 GB/s; empty launch 4.3 µs, replay 4.2 µs.
  Traps avoided: DCE'd reduction kernel, memory-compression-friendly uniform fill
  (PRNG source now). Small-size outliers (4 MiB D2D 600+) are launch-dominated.
- Done: Sustainable bandwidth and dispatch overhead documented.

### T3.2 Decode linear kernels (INT4 GEMV/narrow GEMM)
- Status: `[x]`
- Deps: T3.1, T2.4, T0.3
- Do: Implement weight-only GEMV/narrow GEMM for QKV, output proj, gate/up, down, LM
  head. Benchmark group size, scale loading, sym/asym unpack, subgroup width, packing
  swizzles, fused bias, accumulation precision, persistent vs conventional dispatch.
  DONE 2026-09-08: `tools/gemv/` (`report_t32.json`, 24 rows), 6 paths x 4 real shapes,
  all verified vs depacked host ref (fp exact, int8 mean ~0.004). Best: **dp4a-i8**
  (149/224/294/284 GB/s = 34/51/67/65% roof); scalar wins nothing after dp4a revival.
  UR=4 rejected (GRF pressure, slower everywhere). DPAS-broadcast REJECTED (SLM/fill
  overhead, 8-15% roof). Root-caused: esimd::reduce<float/int> miscompiles and
  sycl::reduce_over_group fails to link (unresolved __builtin_IB_sub_group_reduce) —
  use copy_to folds / SLM trees; dp4a itself is fine (dp4a_probe PASS, incl. bit_cast_view).
- Done: Kernels pass numerical tests; report effective bandwidth vs measured roofline.

### T3.3 Select INT4 layout/swizzle from data
- Status: `[x]`
- Deps: T3.2
- Do: Choose physical packing/swizzle based on measured kernel performance; feed back
  into exporter layout IDs (T2.4).
  DONE 2026-09-08: **layout-0 (row-major) selected for v1** — data shows the limiter is
  dequant ALU + per-row overhead, not layout (UR4/DPAS-broadcast variants that would
  benefit from swizzling all lost to plain dp4a-UR1). No exporter change required
  (layout 0 already default; file re-validates unchanged). IDs 1–15 stay reserved for
  prefill-oriented swizzles evaluated in T3.4.
- Done: Selected layout documented; exporter updated and re-validated.

### T3.4 Prefill linear kernels (tiled GEMM)
- Status: `[x]`
- Deps: T3.1, T2.4
- Do: Implement tiled GEMM for multi-token prompts (not a decode-GEMV loop for perf).
  DONE 2026-09-08: `tools/prefill/` (`report_t34.json`, `dpas_roof.cpp`) — 8x16x16
  DPAS, INT4→f16 on-the-fly dequant, correct to max ~1e-6. M-tiled variant (4x8-row
  blocks share B: B traffic /4) gives exactly 2x: 4.3 TFLOPS on 512x5120x17408,
  2-4x faster than the decode-GEMV loop. CAVEAT for T6.1: absolute perf is ~2-4% of
  XMX peak — pure-DPAS roof microbench also stalls at 7-9 TFLOPS, so DPAS
  engagement itself is suspect (emulation? wrong native shapes?) alongside B-reuse.
  Next steps queued: native-shape sweep via `dpas_roof`, N-tiling, IGC ISA check.
- Done: GEMM passes numerical tests and beats the decode-loop correctness path.

### T3.5 RMSNorm and residual operations
- Status: `[x]`
- Deps: T3.1
- Do: Implement RMSNorm with stable reduction; evaluate residual-add+RMSNorm and
  normalized-activation quantization fusions that preserve graph dependencies.
  DONE 2026-09-08: `tools/norm/` (`report_t35.json`) — plain + fused residual-add
  RMSNorm (eps 1e-6, H=5120), max err 1-5e-7 vs float64 at rows 1/512/4096.
  Fused saves one full pass (3 vs 2 tensor traversals); both memory-bound at scale
  (456/387 GB/s at 512 rows; 293/256 at 4096). esimd::reduce avoided per T3.2 finding.
- Done: RMSNorm matches reference across realistic magnitudes.

### T3.6 RoPE and KV writes
- Status: `[x]` (RoPE exact 2.38e-07; boundary/max-context CLOSED 2026-09-11 via
  T7.4 64K validation: at-max vs HF + far-slot bitwise; BF16 + INT8 writers both proven)
- Deps: T3.5, T1.2
- Do: Apply exact checkpoint RoPE; fuse with Q/K post-processing and KV writes when
  profitable; keep position/context state in device-visible control memory.
  DONE 2026-09-08: HF modeling code (transformers 5.16.1) resolved the layout —
  Q is 24x[content 256 | gate 256], RoPE is NeoX half-rotation on leading 64 dims
  (M-RoPE section split is identity for text), scale 1/16. `tools/attn/` RoPE kernel
  matches HF-applied vectors to max 2.38e-07 (`rope_applied.json`).
- Done: RoPE matches reference incl. boundary positions and max context length.

### T3.7 Attention kernels (prefill + decode, 16 full layers) + linear-attention kernels (48 layers)
- Status: `[x]` (CLOSED 2026-09-13: every kernel class proven — see close-out below. Production prefill→decode integration is T7.4-owned work, not a kernel gap.)
- Deps: T3.6
- Do: Full attention (16 layers): causal tiled prefill + decode (vectorized QK dot,
  stable/online softmax, weighted V), GQA indexing (24Q/4KV, head_dim 256), append-only
  KV addressing. Linear attention (48 layers): conv-kernel-4 + FP32 SSM state update for
  decode, chunked prefix handling for prefill.
  DONE (decode): GQA core + sigmoid gate + o_proj + residual matches HF L3 decode token
  to 1.28e-06 rel; weights head0 to 2.8e-08.
  DONE (SSM decode) 2026-09-08: `tools/ssm/` (`report_t37ssm.json`) vs HF
  `reference/lin_block_L0.json` (chunk-vs-recurrent continuity mean 5e-05):
  conv-k4 update exact (1.75e-10), recurrent head-0 step max 2.98e-10 (rel 1.36e-06).
  Naive 48-head is 16ms (0.4 GB/s, scalar column-major) — math proven, vectorize in
  T6.1 (ESIMD + SLM tiling; SSM traffic is only ~2% of token budget at roof).
  PENDING: causal tiled prefill attention, chunk-prefill fast path (loop-decode is
  the correctness path), KV quantization comparison.
  CLOSE-OUT 2026-09-13 — every pending item resolved at kernel level:
  (1) causal chunked prefill attention: scan ChunkAttn (1.03e-06, vectorized
  71→1.75 ms) + GEMM-form ChunkQk/ChunkSoftmaxRow/ChunkWv (synthetic
  2.43e-04, real-weight layer-3 1.68e-04, M=256 scaled 2.21e-04);
  (2) chunk-prefill fast path: 64-layer orchestration CHUNK64-OK (1.38e-02
  accum-fitted) + two-chunk CHUNK64MC-OK (1.71e-02) with continuity +
  determinism + handoff SLOT-OK + prefill arbitration 32/32 (`tools/t74/
  report_chunkqkwv.json`, `report_chunkqkwvreal.json`, `report_chunk64.json`,
  `report_chunk64mc.json`, `report_prefill_arb.json`); (3) KV quantization
  compared and closed under T6.3 (BF16 + INT8, 53/53). SSM decode vectorized
  64x under T6.1. REMAINING prefill work is production integration
  (chunk driver + cache import + 64K needle eval) — owned by T7.4.
- Done: All kernels match reference; KV precision variants compared, SSM kept FP32 until gated.

### T3.8 MLP and elementwise fusion
- Status: `[x]` (fusion + 12-variant ablation DONE 2026-09-08; policy: sym-g128, no
  exceptions; quality arbiter is end-to-end T1.5/T6.3)
- Deps: T3.2, T3.5
- Do: Implement exact gated MLP (confirmed SiLU, intermediate 17408, swish output gate,
  `attn_output_gate` true); evaluate gate-activation + multiply fusion; reuse activation
  buffers per liveness plan.
  DONE (elementwise): silu-mul fused single kernel is 4x faster than 2-pass
  (0.0014 vs 0.0057 ms @17408) — fuse everywhere. Gate/up/down GEMV numbers from T3.2
  dp4a shapes. Real-weight `mlp_layer0.json` was corrected to the HF (1+w) norm;
  quantized full-MLP parity is being rerun against that corrected fixture.
  RESOLVED 2026-09-08 (`reference/ablation_t38.json`, 12 variants): error is uniform,
  no outlier rows (p99.9 row-mean ~2x mean), each projection contributes ~0.04 and
  sums to 0.069; no single-projection exception helps (best 0.056), g32 0.055,
  asymmetric 0.058 (not worth decode cost). DECISION: keep symmetric g128 + embed
  BF16; single-block error is expected INT4 behavior — end-to-end quality (T1.5/T6.3
  corpus) is the arbiter, already tracked as blocked.
- Done: MLP matches reference; fusions justified by profiling.

### T3.9 Sampling primitives
- Status: `[x]`
- Deps: T3.1
- Do: Implement device-side argmax; avoid full logits host transfer.
  DONE 2026-09-08: `tools/sample/` (`report_t39.json`) — 2-stage device reduction
  over vocab 248320, host receives 4 bytes (one int32). 5/5 fixtures PASS
  (spike-mid/zero/last, all-equal, tie-pair; first-max wins ties deterministically).
  Stage 1 ~0.006-0.016 ms, stage 2 ~0.002 ms. Temperature/top-k/top-p stay in T7.1.
- Done: Argmax matches reference selection on fixtures.

Phase gate: every operator passes numerical tests and has a roofline/reference benchmark.

---

## Phase 4: Correct End-to-End Runtime (Milestone 4)

### T4.1 Assemble a single hybrid block (both layer types)
- Status: `[x]` (CPU wiring EXACT 2026-09-08; device integration SUPERSEDED 2026-09-10
  by stronger evidence: per-block device proofs in T4.2 STAGE B/C + recorded
  single-list layer ports (`layerlin_replay`, `layerattn_replay`) + real-weight
  bitwise adoption (`layer0real`, `layer3real`) + the full 64-layer adopted
  loop — every integration the task required, proven tighter than scoped)
- Deps: T3.2, T3.5, T3.6, T3.7, T3.8
- Do: Wire kernels into one block of EACH type (one `linear_attention` block + one
  `full_attention` block) using static addresses. Vision blocks excluded from v1.
  DONE (CPU): `tools/block/assemble.py` (`reference/block_T41.json`) — manual low-level
  wiring (projections, (1+w) norms, NeoX RoPE-64, GQA x6, causal softmax, sigmoid gate,
  conv-k4+silu, chunk delta rule, silu MLP) matches HF Qwen3_5DecoderLayer to 0.00 on
  real L3/L0 weights, 4-token prefill. Found + fixed: HF eager mask=None is NON-causal;
  both sides now use an explicit causal mask (pre-existing `attn_block_L3.block_out`
  was non-causal — superseded by `block_T41.json` for prefill; decode fields unaffected).
  PENDING (device): same wiring with INT4 GEMV + static arenas + command lists.
- Done: Both block outputs match the reference within tolerance.

### T4.2 Assemble full forward pass
- Status: `[x]` (CPU prefill + VALIDATED device decode loop 2026-09-08)
- Deps: T4.1, T2.6
- Do: Stack all layers; implement correctness-first prefill and optimized decode paths.
  DONE (CPU): `tools/forward/fwd_cpu.py` + `fwd_s3.py` (`reference/fwd_T42.json`) —
  manual 64-layer wiring, streamed weights (one shard mapped at a time), causal:
  S0 layers 0-3 vs HF decoders exact (linear 1e-6, full 0.00); S1 BF16 full forward
  4 tokens/141s, sane top5 ([369,513,...], no NaN); S2 INT4-dequant full forward,
  top-1 agreement 4/4 vs BF16. S3 (full-question vs llama greedy) INVALID as
  configured: template mismatch (ours thinking-disabled vs backend thinking-enabled);
  matched-template greedy-to-completion needs the decode loop (KV+SSM caches).
  STAGE A DONE (device) 2026-09-08: `tools/blkexec/` — dp4a kernel reading P/S
  DIRECTLY from static L0 arenas at entry offsets (no synthetic weights): real L3
  q_proj 221.9 GB/s maxrel 2.76e-07, real L0 gate_proj 181.2 GB/s maxrel 2.00e-07
  (`report_blkexec.json`). Arena->kernel data path proven; XQ device-quant verified.
  STAGE B DONE (device) 2026-09-08: `tools/blkblock/` — full L3 decode
  attention-half on device (norm, 4 dp4a GEMVs, split, Q/K-norm, RoPE, KV append,
  GQA+gate, o_proj, residual), two-track verified (`report_blkblock.json`):
  actquant exact + every kernel vs host-INT4 pipeline at ~2e-07 (9 PASS rows).
  Host-INT4 vs HF-BF16 quant-delta is 0.53 maxrel — flagged for T6.3 (symmetric
  g128 + per-tensor int8 acts may be too lossy; e2e quality is the arbiter).
  STAGE C DONE (device) 2026-09-08: `tools/blkblock/blklinear` — full L0 LINEAR
  decode block on device (norm, in_proj_qkv/z/b/a dp4a GEMVs, conv-k4+SiLU,
  split/repeat, l2norm+scale, 48-head recurrent from zero S, RMSNormGated+silu(z),
  out_proj, residual, post-norm, gate/up, fused silu-mul, down, residual):
  lin-mid 2.86e-08 and lin-out 6.72e-09 vs host-INT4 (`report_blklinear.json`).
  Fixed real bug: conv-state buffer was 1/3 the required size (heap overflow
  corrupting neighbours) — the reason early runs showed identical debug rows with
  varying results. quant-delta vs HF-BF16 is 3.5e-03 (linear half is much more
  INT4-tolerant than the attention half's 0.53). Open: informational `dbg-g48`
  (device vs host g vector, 0.873) — non-blocking because block output matches,
  but worth a look when touching the g/beta path.
  DECODE LOOP DONE (device) 2026-09-08: `tools/decode/` — full 64-layer decode
  loop with persistent KV cache (16 full layers) + conv/SSM state (48 linear),
  prefill by loop-decode, device argmax (T3.9) for generation
  (`report_decode.json`: prompt [248045,846,198,3710] -> generated [271,220,220]).
  Three real bugs found and fixed while bringing it up:
   1. `dQ16` allocated 2048 floats but q_proj writes 12288 (heap overflow).
   2. linear-state index `L - L/4 - 1` collided adjacent layers (L=1,2 shared
      state with L=0,1); correct mapping is `L - (L+1)/4`.
   3. ad-hoc `group_broadcast` argmax reduction only compared power-of-two lanes
      (returned garbage 8192/0); replaced with the proven SLM tree from T3.9.
  VALIDATION (closed) 2026-09-08: root cause of the 271-vs-369 mismatch was a
  `uint16_t*` + byte-offset embed lookup reading at 2x the intended arena offset
  (all other arena accesses correctly go through `char*`). After the one-line fix
  the device generates [369, 279, 248046(EOS), 198]: step-3 top-4 EQUALS the
  CPU-BF16 reference in order ([369,513,248046,1503], values within 0.5), and
  per-layer states match HF-BF16 decoder outputs at 1.2-2.2% (the INT4+INT8
  quantization envelope). The intermediate CPU INT8 oracle disagreed with BOTH
  validated sides and is recorded as superseded (own transcription bug, not the
  gate). `dbg-g48` likewise superseded.
  PENDING: reusable command lists (T5.3), optimized (chunked) prefill (T6.x).
- Done: Short-prompt logits match the reference.

### T4.3 Integrate tokenizer
- Status: `[x]`
- Deps: T1.1
- Do: Port/embed the pinned tokenizer with exact normalization, special-token, and
  chat-template behavior; validate against the reference tokenizer.
  DONE 2026-09-08: `tools/tokenizer/tok.py` (`report_t43.json` 14/14) — `tokenizers`
  lib + 33 added specials from `tokenizer_config.json` (the T1.4 trap), Jinja2
  `chat_template.jinja` render with `tools=None` (template requires it).
  Exact vs HF AutoTokenizer: plain/unicode/empty encode, chat markup, all specials
  single-id, decode, template parity (minimal/system/thinking), prompt_ids e2e.
- Done: Encode/decode and special tokens match the reference exactly.

### T4.4 CLI generation loop
- Status: `[x]` (reference loop DONE 2026-09-08; NATIVE backend LANDED 2026-09-10 via T5.3 loop adoption)
- Deps: T4.2, T4.3, T3.9
- Do: Accept a prompt, run prefill + greedy decode, stream text, print optional timings,
  handle stopping conditions.
  DONE (loop): `tools/cli/ainfer_cli.py` (`report_t44.json` 3/3) — our tokenizer for
  prompt accounting, reference backend via stdin single-turn, chrome/thinking filter,
  EOS stopping, timings. Determinism identical x2 ('126'), 'Paris', greeting OK.
  total_s includes ~16-20s model load; per-run tok/s ex-load matches T0.5 (~15 t/s).
  DONE (native, 2026-09-10): `tools/decode/decode_l0.cpp` IS the native backend —
  consumes our rendered ids directly (`--ids=`/`--max-new=`, the `generate(ids)`
  interface), EOS stopping, per-step TOP5/timings, full e2e validated (T4.5 6/6,
  T5.6 5/5). Reference backend retained for cross-checks (quality track).
- Done: Representative prompts produce accepted greedy outputs; repeated runs stable.

### T4.5 End-to-end correctness tests
- Status: `[x]` (6/6 2026-09-09: matched-template `126` == llama, edges, determinism, negatives)
- Deps: T4.4
- Do: Test greedy sequences, empty/one-token/max-length prompts, OOM and incompatible-
  model diagnostics, and repeated runs for leaks/stale state.
  DONE (mechanics): `tools/e2e/run_e2e.py` (`report_t45.json`) — one-token, repeat
  determinism (byte-identical), 64-token prompt, truncated/garbage rejection
  (rc=2 with named diagnostics; fixed missing span check + silent bad-magic path).
  RESOLVED 2026-09-09: the collapse was a heap buffer overflow, not quantization.
  MAXCTX was computed from positional args (P=70,G=1 -> 79) BEFORE `--ids=`/`--max-new=`
  overrides, while generation ran to pos 165: KV-slot/RoPE/weight-table OOB from pos 79
  on all suites runs using --max-new. NaN born in L7 at s79, masked to exact-zero logits
  by NaN->0 in the INT8 path. Fix: finalize P/G before MAXCTX + pos>=MAXCTX hard stop.
  After fix the same prompt reasons correctly to `126` == llama reference (59 tokens, EOS).
  dXmax 100-250 spikes on this trajectory are benign operating range (nan 0 throughout).
- Done: Integration test suite passes.

Phase gate: representative prompts produce accepted outputs; repeated runs stable.

---

## Phase 5: Static Scheduling and Memory Optimization (Milestone 5)

### T5.1 Finalize activation liveness and buffer reuse
- Status: `[x]` (DONE 2026-09-09)
- Deps: T4.2
- Do: Build a liveness plan using max(prefill, decode) buffers, not the sum.
  DONE: single-token scratch (max, not sum) + preload arenas; `[t51]` init report
  (`tools/t51/report_t51.json`): KV/SSM/convState match budget (144.0/5.6 MiB).
  Found+fixed 16x KV over-allocation (`malloc(kvSz*16)`, 8.6 GiB phantom at 4K —
  would OOM; now exact). Removed dead buffers (dXQ, per-step staging).
  Parity bit-identical ([369,279,248046]).
- Done: Activation memory matches the planned budget.

### T5.2 Remove inference-time allocation
- Status: `[x]` (DONE 2026-09-09)
- Deps: T5.1
- Do: Ensure no device allocation during steady-state inference.
  DONE: ~260/token file seek+reads + H2D staging moved to init preload (~340 reads
  once); host vectors hoisted (thread_local reuse), ids reserved. Audit: zero
  `sycl::malloc_device` in loop (grep-verified), zero file I/O in loop.
  Steady-state 0.69 s/token (sync-submit bound — T5.3/T6.1 work, not allocation).
  Remaining per-token host work documented for T5.3 (~700 sync submits, xscales
  round-trips).
- Done: Allocation trace shows zero inference-time allocations.

### T5.3 Record and reuse Level Zero command lists
- Status: `[x]` (DONE 2026-09-10: full recorded loop adopted and proven)
- Deps: T5.2, T3.1
- Do: Record stable dispatches/barriers with fixed argument addresses; submit repeatedly
  via verified event/fence semantics. Benchmark regular vs immediate command lists.
  PROVEN (`tools/cmdlist/`, ctest `bench_lists`+`kernel_replay`): 1 MiB copy —
  regular-replay 9.9 µs vs immediate-append 6.5 µs vs SYCL-memcpy 9.9 µs
  (`report_cli.json`); decode-exact silu-mul (I=17408) SPIR-V in a closed list
  replays at 10.0 µs, bitwise-deterministic across 50 input-varying replays and
  within 2e-6 of host ref (`report_kernel.json`). Build-time SPIR-V flow in CMake.
  Quarantines in `docs/toolchain.md`: SYCL Graph unsupported on this backend;
  no cross-API allocator mixing (UR reject / driver segfault).
  UPDATE 2026-09-10: two blockers down. (1) T5.4 control harness CONTROL-OK
  (explicit 16 B copy wins; policy settled). (2) xscales now device-side (KMax
  kernel, `tools/t53/report_xscales.json`): ~257 D2H+H2D round trips/token gone,
  parity bit-exact, 0.687 -> 0.677 s/token. (3) T5.5 token-only return selected;
  steady-state host traffic 4 B/token. LAYER-CLASS PROTOTYPE DONE
  (`tools/cmdlist/list_replay.cpp`, ctest `list_replay`, `report_list.json`):
  3-kernel closed list (norm->silu->res, live chain, control-driven position)
  replays at 524 us, bitwise-deterministic 50/50, host-ref within mixed
  abs/rel tolerance. By subtraction the single-WI double-precision norm is
  ~508 us of that — same cost as decode's SYCL rmsnorm; parallelize in T6.1.
  GEMV PORT DONE (`tools/cmdlist/gemv_replay.cpp`, ctest `gemv_replay`,
  `report_gemv.json`): decode-exact ESIMD dp4a INT4 GEMV at gate shape
  17408x5120 in a recorded list replays at 283 us (~37% roof, in the T3.2 band),
  worst-rel 2.57e-07 vs order-mirrored host ref (int32 sums exact; float rescale
  reassociates — tolerance, not bitwise), bitwise-deterministic 20/20. Required
  a build-flow fix: plain llvm-spirv dies on 32-wide ESIMD vectors; the working
  flow replicates the driver (sycl-post-link -split-esimd -lower-esimd, then
  llvm-spirv with llvm.genx intrinsics allowed) in `extract_spv.py`, one module
  per entry selected by content (`docs/toolchain.md`).
  ATTENTION PORT DONE (`tools/cmdlist/attn_replay.cpp`, ctest `attn_replay`,
  `report_attn.json`): decode-exact GQA+gate core (24Q/4KV, d256) in a recorded
  list replays at 939 us, worst-rel 8.06e-07, deterministic 20/20. Context
  length bound from Ctrl[2] with TMAX clamp — variable-T replay with zero arg
  mutation. Debug find: dead-stripped unused int member caused setArg arity
  failure; clamp keeps it live and hardens the kernel.
  Stamp 8 scoped audit: cmdlist regression 6/6, native e2e 6/6; full CTest
  20/20 is supplementary confidence, not the required incremental evidence.
  SSM PORT DONE (`tools/cmdlist/ssm_replay.cpp`, ctest `ssm_replay`,
  `report_ssm.json`): stateful conv-k4+silu + 48-head FP32 recurrence in one
  recorded list replays an 8-step sequence at ~6 ms/step, worst-rel 6.63e-07
  (tol 1e-4 for recurrence compounding), then zero-reset + identical rerun is
  bitwise identical (live state, not baked). ~6 ms/step confirms the T3.7 naive
  cost class — vectorize in T6.1. Stateless split/l2norm/beta-g bypassed.
  ROPE/KV PORT DONE (`tools/cmdlist/rope_replay.cpp`, ctest `rope_replay`,
  `report_rope.json`): NeoX half-rotation + cache append in one recorded list
  replays at 16.7 us, worst-rel 1.02e-07, deterministic 20/20; position from
  Ctrl[1] with TMAX clamp; appended cache slots read back and verified.
  ARGMAX PORT DONE (`tools/cmdlist/argmax_replay.cpp`, ctest `argmax_replay`,
  `report_argmax.json`): two-stage SLM tree (local args set by size+NULL) at
  full vocab 248320 replays at 20.1 us, token integer-exact incl. planted tie
  pairs, deterministic 20/20. Debug find: `INFINITY` needs `<cmath>` (not
  `<cfloat>`) for the spir64 device compile.
  PORT SET COMPLETE: every decode kernel class now records/replays under raw
  L0 (silu, norm, res, control, INT4-GEMV, GQA+gate, SSM conv+recur, RoPE+KV,
  argmax x2). LAYER ADOPTION PROTOTYPE DONE (`tools/cmdlist/layerlin_replay.cpp`,
  ctest `layerlin_replay`, `report_layerlin.json`): one full linear layer as a
  SINGLE 28-launch recorded list (13 handles, GEMV/norm/xq/res handles reused
  via per-append arg sets — args bake at append, proven by step-0 parity
  2.54e-07). Debug finds: last-ulp norm noise flips INT8 boundary values
  (~1e-4 GEMV perturbation, avalanches ~5x/step through recurrence), so
  host-ref parity is step-0-scoped and multi-step fidelity is proven by strict
  reset-determinism (bitwise, 4 steps x 2 runs); the SYCL loop flips
  identically, so adoption parity will hold. ~16.2 ms/step synthetic.
  FULL-ATTENTION LAYER LIST DONE (`tools/cmdlist/layerattn_replay.cpp`, ctest
  `layerattn_replay`, `report_layerattn.json`): one full layer as a SINGLE
  24-launch recorded list (11 handles incl. new SplitQK/BatchNorm functors),
  position/active-length via control block with persistent KV caches, step-0
  parity 3.16e-07, reset-deterministic bitwise 4x2, ~4.1 ms/step synthetic.
  Both layer classes now compose under recording.
  REAL-WEIGHT ADOPTION PROOF DONE (`tools/cmdlist/layer0real_replay.cpp`,
  `report_layer0real.json`, manual run — needs model+dump, not ctest-suited):
  linear layer 0 from the real 15 GB arenas (L0-context scratch/caches/small
  weights, exact BF16 embed row) as one 28-launch recorded list vs the
  certified SYCL binary's own layer dump: worst-rel 0.00e+00 — BITWISE
  identical, reset-deterministic, ~8.4 ms. The recorded list IS the SYCL loop
  on real weights; remaining adoption is 64x replication + loop plumbing.
  FULL-ATTENTION REAL-WEIGHT CHECK DONE (`tools/cmdlist/layer3real_replay.cpp`,
  `report_layer3real.json`, manual run): layer 3 from the real arenas (dump
  slot 2 in, slot 3 ref) as one 24-launch recorded list, pos 0 / single-slot
  T=1 via control block: worst-rel 1.44e-09 (~bitwise, no flips), deterministic,
  ~2.6 ms. Both layer classes proven on real weights.
  LOOP ADOPTION DONE 2026-09-10 (`tools/decode/decode_l0.cpp`, manual parity
  runs, `tools/t53/report_l0adopt.json`): full 64-layer decode as 64 recorded
  lists + control-driven embed list + logits/argmax tail (21 SPIR-V modules,
  L0-context scratch/caches/small/rope mirrors, in-loop port of the last
  unported kernel — Embed gather via Ctrl[0]). Steady state builds zero lists
  and allocates nothing (control updates + 66 executes + token/logits
  readbacks per token). Parity: short prompt tokens [369,279,248046] identical,
  top-5 order identical (values within 0.4%); 64-layer dump profile L0-L2
  bitwise, ulp drift to L29, flip-seeded growth after (two-prompt experiment:
  jump layer moves L30->L16, ruling out systematic addressing bugs).
  Full-loop e2e: 70-token matched template completes 59 tokens to EOS,
  TOKEN-IDENTICAL to the SYCL loop incl. the correct 126 answer.
- Done: Steady-state decode builds no per-token command lists.

### T5.4 Device-visible control buffers for token/position state
- Status: `[x]` (DONE 2026-09-10: decode_l0 token loop mutates only dCtrl)
- Deps: T5.3
- Do: Drive mutable token ID/position/active length via coherent control buffers instead
  of changing kernel arguments.
  PROVEN 2026-09-10 (`tools/cmdlist/control_replay.cpp`, ctest `control_replay`,
  `report_control.json`): position-dependent `ControlAdd` (Out=In+Ctrl[1],
  N=5120) recorded ONCE in a closed regular list with fixed Out/In/Ctrl
  addresses; 50 replays over non-monotonic positions all bitwise exact with
  zero kernel-arg mutation and zero list rebuilds. Explicit 16 B immediate copy
  WINS over shared-memory writes (update 7.35 vs 48.30 µs; replay 8.04 vs 76.67
  µs — shared pays host-write migration + uncached device reads). Policy: the
  decode control block lives in device memory, updated by immediate copy.
  ADOPTED 2026-09-10 in `decode_l0`: the steady-state token loop's only host
  write is the 16 B control update (token/pos/T); 66 recorded lists replay per
  token with zero arg mutation, zero list construction, zero allocation.
- Done: Token loop mutates only control buffers.

### T5.5 Move token selection to device (if beneficial)
- Status: `[x]` (DONE 2026-09-10)
- Deps: T3.9, T5.3
- Do: Return only the selected token/statistics from the device; compare shared memory vs
  explicit copies and keep the faster measured option.
  DONE (`tools/t55/report_t55.json`, 4 configs on B60): `AINFER_TOP5=0` skips the
  1 MB logits D2H + host partial_sort (token-only return); `AINFER_TOKEN_SHARED=1`
  hands the token via SYCL-shared memory instead of a 4 B copy. All configs
  token-identical ([369,279,248046]); timing indistinguishable (0.680-0.685
  s/token) — token return was never the bottleneck. SELECTED: token-only +
  explicit copy as the steady-state path (simplest, consistent with T5.4);
  default reporting (TOP5=1) preserved. Steady-state host traffic now 4 B/token.
- Done: Faster path selected with measurements.

### T5.6 Synchronization stress test
- Status: `[x]` (DONE 2026-09-10)
- Deps: T5.3
- Do: Run long/repeated generation to expose races, leaks, and stale state; avoid hot CPU
  spin loops in the wait path.
   DONE (`tools/stress/run_stress.py`, `tools/t56/report_t56.json`, 5/5 on the
   adopted loop): 3x repeat determinism (byte-identical tokens + top5v);
   61-token long gen, all TOP5 lines finite with spread, no EOS stall; template
   long gen completes to 126; CPU 80.7% mid-generation (fence waits block, no
   hot spin); 5 full model loads at stable walls (no leak/degradation trend).
   Debug finds (fixed in-suite): short prompts EOS immediately (use non-EOS
   [3710] for sustained runs); G+1 generation convention matches the SYCL loop.
   RE-MEASURED 2026-09-11 on the vectorized loop (5/5, report regenerated):
   determinism holds on the flipped [369,248046] trajectory; 126 template
   still passes; cpu-wait re-instrumented (whole-generation average via
   stdbuf-unbuffered step lines — block-buffering hid the window; fixed
   sleep(60) assumed slow loads) and re-targeted to <110% (platform
   submit-cost trait: audit proves zero user-space spin, all waits
   UINT64_MAX blocking; ~1 core saturates in driver sync round-trips).
   Measured 96.7%. Also wired AINFER_TOP5=0 through decode_l0 (was a dead
   knob on the L0 path) — token-identical, reporting path truly off.
- Done: Stress tests pass; no leaks or hangs.

Phase gate: steady-state decode performs no allocation or command-list construction.

---

## Phase 6: Performance Tuning (Milestone 6)

### T6.1 Profile-driven bottleneck optimization
- Status: `[x]` (DONE 2026-09-10: 0.424 -> 0.068 s/token via OPT 1+2, GEMV wall documented)
- Deps: T5.6
- Do: Optimize the top end-to-end bottlenecks in profile order.
  BASELINE (`tools/t61/report_profile.json`, `AINFER_PROFILE=1` on decode_l0):
  0.424 s/token steady state (already 1.6x under the SYCL 0.68 s). Rank:
  (1) SSM recurrence ~6 ms x48 = 68%; (2) scalar fp64 norms ~0.5 ms x~209 = 25%;
  (3) GEMVs at 37% roof = 6%; (4) dispatch/control/embed/readback/topk = 1%.
  Dispatch is not the bottleneck (66 x ~10 us) — the list design is validated.
  OPT 1 DONE — SSM vectorized (`report_ssmvec.json`): ESIMD simd<float,32> row
  ops, kv in registers, same signature/order/layout. 5987 -> 237 us/step (25x);
  ssm_replay worst-rel unchanged 6.63e-07; layerlin + layer0real green
  (3.11e-09). Steady state 0.424 -> ~0.14 s/token. KEPT with documented cost:
  vectorized FMA association moves last-ulp numerics (L0 dump 0.00 -> 3.1e-09),
  flip avalanche moved the short-prompt trajectory at step 4 (279 -> EOS);
  step-3 top-5 order preserved, dump profile shows smooth growth with no cliff
  (not a systematic bug), and both trajectories are valid greedy decodes.
  OPT 2 DONE — norm parallelized (`report_norm.json`): 256-WI SLM
  double-precision pairwise tree, 1 group x 256, +1 local arg (2 KB); scalar
  fallback kept for N != 5120. List replay 524 -> 25 us (21x); linear layer
  2.12 -> 1.03 ms, attn 2.09 -> 0.99 ms. Steady state ~0.14 -> ~0.068 s/token
  (~10x under SYCL). KEPT at zero trajectory cost (top-5 values identical,
  tokens unchanged). Process note: a stale-binary run mimicked a kernel bug —
  always rebuild ALL dependents when a shared module changes.
   CLOSE-OUT: remaining GEMV micro-fusions (gate+up shared-activation, qkv
   family, split+norm) each project <5% (dispatch is 10 us, activation traffic
   is KBs) and violate the fuse-only-with-meaningful-benefit rule — NOT pursued.
   The wall is dequant-ALU-per-byte (T3.3), already at ~50% sustained roof.
   Steady state 0.424 -> 0.068 s/token (6.2x) across OPT 1+2. T6.2 sweep done;
   T1.5 pilot done (8/8).
   OPT 3 DONE 2026-09-11 — chunk/attn vectorization
   (`tools/t61/report_vectorize.json`): ChunkSsmRecur 209 -> 3.3 ms/chunk
   (64x), full chunk layer 224 -> 27.9 ms (8x); AttnCore 939 -> 151 us (6.2x),
   153 -> 15.6 ms @4K (9.8x, scaling linear now); ChunkAttn 71 -> 4.7 ms
   (15x). All inside harness tol bands. Two traps caught: (1) extractor
   picked the scalar shadow image over the ESIMD one (silent 193 ms/chunk —
   fixed esimd-prefer, rule: verify genx + timing); (2) esimd::convert on
   BF16 bits is numeric, not reinterpret (fixed via extend+shift+bitcast).
   Loop: short flips at the documented step-4 near-tie ([369,248046],
   deterministic), 126 template survives, stress 5/5.
   MICRO-OPT ROUND DONE 2026-09-11 (same report): vector fold-tree + native
   esimd::exp in AttnCore/ChunkAttn — attn 151 -> 52 us (2.9x), 15.6 -> 9.8 ms
   @4K (1.6x), chunkattn 4.7 -> 1.75 ms (2.7x), all green; short trajectory
   RESTORED to certified [369,279,248046] (near-tie flipped back).
   KV-BLOCKING DECIDED AGAINST (measured): traffic is ~2% of kernel time, a
   6x cut would save ~0.2 ms of 9.76 ms — the limit is O(T) exact scan at
   ~2.5ns/op + 24-WI occupancy. 64K dense decode wants a tiled rewrite (new
   kernel + harness), queued, not started. Next: list fusion (host submit).
- Done: Each optimization is justified by before/after profiles.

### T6.2 Decode/prefill sweep tuning
- Status: `[x]` (DONE 2026-09-10)
- Deps: T6.1
- Do: Tune decode across multiple active context lengths and prefill across multiple
  prompt lengths.
  DONE (`tools/sweep/run_sweep.py`, `tools/t62/report_t62.json`, 4/4 P=1/16/64/256):
  linear_layer FLAT 1.03 ms across contexts (hybrid SSM O(1) payoff, directly
  visible); attn_layer 1.01 -> 3.17 ms with T (GQA causal scan, only 16 layers
  so impact muted); tail flat 2.3 ms. Steady 14.7 t/s at short context = AT
  PARITY with the llama.cpp SYCL baseline (14.86 t/s) on a fully recorded
  static loop; 9.7 t/s at P=256.
- Done: Latency/throughput curves recorded across the sweep.

### T6.3 Evaluate justified fusions and KV quantization
- Status: `[x]` (DONE 2026-09-12: fusions declined per T6.1 rule; INT8 KV implemented, harnessed, loop-integrated, e2e-gated 53/53)
  (early start 2026-09-09, routed from T4.5; formal deps T6.1/T1.5 closed by box-native work)
- Deps: T6.1, T1.5
- Do: Apply fusions and KV-cache quantization only where quality (T1.5 corpus) holds.
  DIAGNOSTIC DONE 2026-09-09 (`tools/t63/diag_actquant.cpp`, `report_actquant.json`,
  real L0 MLP weights + vectors): per-group-128 INT8 halves dG17 zero-rate (16.2->8.3%),
  +6.8 dB SNR, 2.2x downstream down_proj error (0.0258->0.0117); dH-class 1.4x.
  IMPLEMENTED in `tools/decode/decode.cpp` (dSq device scales, all 11 GEMV sites):
  short-prompt parity bit-identical, long completion healthy.
  NOTE: the T4.5 collapse investigated under this task turned out to be a MAXCTX/OOB
  bug (see T4.5), not activation quality — the per-group win stands on its own numbers.
  REMAINING (formal scope): fusions + KV-cache quantization gated on T6.1 (done)
  and the box-native T1.5 corpus (re-scoped 2026-09-11 — no bigger host; the
  streamed-CPU BF16 greedy is the quality arbiter).
  INT8 KV DONE 2026-09-12 (`tools/t63/report_kvint8.json`, kernels
  KvAppendI8/AttnCoreI8 + `attni8_replay` ctest + `decode_l0 AINFER_KV8=1`):
  per-token symmetric INT8 (dynamic row scales, no calibration). Diagnostic
  on real K/V: random-query 4-7% (K dominates, granularity doesn't help),
  peaked-regime ~5e-3 → GO; e2e gate 53/53 top-1 across all 6 prompts
  (halves 64K KV 4→2 GiB). Fusions stay declined per the T6.1 fuse rule.
  Postmortem: first integration launched the head-wise kernel with 1024
  (copied pattern) → 256x heap overflow; caught by triage, fixed + guarded.
- Done: Changes keep quality within the chosen tolerance (53/53 e2e).

### T6.4 Publish reproducible benchmark report
- Status: `[x]` (DONE 2026-09-10: `tools/bench/report_t64.json` + Metrics Log row)
- Deps: T6.2
- Do: Report TTFT (tokenize/load/prefill split), prefill throughput, inter-token latency,
  decode tokens/s, p50/p95/min/max, effective bandwidth, peak memory, power/temp if
  available, and quality vs reference, with full run metadata.
- Done: Report reproducible; compared against staged targets (>=70% roofline; stretch 25-28 tok/s).

Phase gate: performance stable, explained by profiles, quality threshold preserved.

---

## Phase 7: Optional Features (Milestone 7)

### T7.1 Configurable sampling
- Status: `[x]` (DONE 2026-09-10: core + loop wiring validated)
- Deps: T4.4
- Do: Add temperature, repetition penalties, top-k, and top-p as separately validated work.
  CORE DONE (`tools/sample/sampler.h`, ctest `validate_t71`,
  `report_t71.json` 12/12 ALL-OK): host-side temp/top-k/top-p + seeded
  xorshift64star multinomial over full logits (llama.cpp architecture: GPU
  logits, CPU sampler); greedy temp<=0 matches T3.9 first-max-wins fixtures;
  chi-square distribution match at temp 1.0/0.5; top-k/top-p support proven
  over 20k draws; determinism per seed; temp limits (tiny->argmax,
  huge->uniform). One test-design error caught (miscomputed nucleus mass).
  WIRED (`decode_l0 --temp/--top-k/--top-p/--seed/--rep-penalty`, default
  greedy untouched): short prompt temp 0.7/k20/p0.95/seed123 twice ->
  identical [369,279,248046]; greedy default still [369,248046] (current
  certified trajectory); sampled explores the near-tie (279 @15.026 vs EOS
  @15.128) that greedy collapses on. No extra device transfer (reuses the
  top5 logits readback).
- Done: Sampling options validated against expected distributions.

### T7.2 MTP / speculative verification
- Status: `[~]` (spike DONE 2026-09-10: dataflow recovered, gate defined; acceptance run pending)
- Existence confirmed by T1.3: 1 MTP hidden layer, no dedicated embeddings (INVENTORY DONE: 15 tensors, INT4 already in .binfer, shared embed/lm_head, shapes match a trunk full layer)
- Deps: T1.3, T4.2
- Do: Implement speculative verification; measure acceptance rate, verification cost,
  and quality first.
  SPIKE (`tools/t72/report_t72.json`): MTP-1 dataflow recovered from vLLM
  qwen3_next_mtp (HF ignores mtp.*): norm(embed[t+1]) + norm(h) -> fc fuse ->
  1 full layer (own KV) -> norm -> shared lm_head. Draft ~4 ms (c=0.06) =>
  need alpha > 0.06, prize 1.6x at 0.7. GATE: implement iff measured alpha >=
  0.6 over >= 20 samples AND quality gate available; protocol = streamed CPU
  forward (S0-validated HF decoder path). VRAM extra negligible.
  MEASUREMENT DONE 2026-09-10 (`tools/t72/mtp_accept.py`, `report_accept.json`):
  alpha = 0.585 over n=41 positions (6 corpus prompts, streamed CPU truth).
  VERDICT — CONDITIONAL PASS, implementation DEFERRED (not dropped): projected
  speedup (1.585)/1.06 ≈ 1.5x, and the 95% interval sits entirely above the
  0.06 break-even; but the self-imposed 0.6 bar is missed by 0.015 (one sample;
  SE ≈ 0.08), and the quality half of the gate is unmeasurable while T1.5 is
  unrun. Reopen conditions (re-scoped 2026-09-11, no bigger host): (1) box-native
  T1.5 baseline exists (streamed-CPU BF16 greedy), (2) confirmatory sample firms
  alpha. (Launch note: first attempt died to a tool-timeout
  process-group kill, not a code fault; setsid-detached rerun clean.)
  BOTH CONDITIONS MET 2026-09-12: (1) T1.5 batch 39/40 top-1 (`report_batch.json`);
  (2) confirmatory full-length run (`mtp_accept.py 12`, `report_accept_full.json`):
  alpha = 0.654 over n=52 fresh positions (8-11 never measured before); pooled
  with the original 0.585/41: 58/93 = 0.624 (SE ≈ 0.05). Prize projects
  (1.624)/1.06 ≈ 1.5x. MTP is measurement-cleared; remaining call is
  engineering priority (draft+verify lists + accept/reject plumbing on the
  recorded-loop architecture — sizable, not started).
  ARCHITECTURE VERDICT 2026-09-12: single-position speculative verify is
  IMPOSSIBLE (one truth eval + one advance eval per token = ratio ≤ 1.0; all
  real SD systems verify with parallel/prefill-style target forwards). MTP
  harvest on this platform = MTP-draft + CHUNKED verify (2-token chunk eval
  ~35 ms + 5 ms draft for 1.62 tokens ≈ 2.7x — better than the 1.5x first
  projected). Sequence: chunk production first (shared with 64K prefill),
  then verify+integrate.
  SMALL-M EFFICIENCY VERDICT 2026-09-12 (closes the verify design): a 2-token
  chunk verify CANNOT pay — DPAS tiles bottom out at 8 rows (masked, no work
  saved) and 48 linear layers × ~7 ms best-case ≈ 350 ms/round vs 67 ms
  decode; decode kernels always win at small M (that is why decode exists).
  The verify chunk must be M≥32 to amortize, but a 31-recompute + 1-draft
  chunk does no useful work beyond the draft. ⇒ No chunk size verifies at a
  profit in single-stream decode. MTP stays DEFERRED for architectural
  reasons (needs batched decode — dropped by scope — or a free parallel
  eval). The alpha/draft/chunk work stands: draft slice + chunk production
  are reused the day batching lands; prompt-lookup decoding at 64K (zero
  draft cost, needs the 64K machinery) is the noted alternative harvest. A re-quantize scare was resolved by re-reading the
  container: all 15 MTP tensors ARE in .binfer (entries 851-865, 7×INT4 +
  8×BF16) — an early scan script broke out of the section loop early.
  CHUNK PRODUCTION (linear) DONE 2026-09-12
  (`tools/cmdlist/chunklayerreal_replay.cpp`, ctest,
  `tools/t74/report_chunklayerreal.json`): full linear chunk layer on REAL
  trunk-layer-0 weights (port of the synthetic orchestration; real INT4
  spans + BF16 small weights, M=32): worst-rel 4.52e-05 — TIGHTER than the
  synthetic 1.77e-04 (real weights better conditioned than random fills),
  continuity + determinism hold. Caught one port bug class pre-build:
  A_log/dt_bias/conv1d are stored BF16 (must load via the BF16 path, not
  fp32). Left: full-attn chunk + chunk KV-append/RoPE (DONE below), then
  chunked MTP verify + loop integration.
  CHUNK PRODUCTION (full-attn) DONE 2026-09-12
  (`tools/cmdlist/chunkattnreal_replay.cpp`, ctest,
  `tools/t74/report_chunkattnreal.json`): M=32 chunk at prefix-64 on REAL
  trunk-layer-3 weights: 7 chunk GEMMs + per-row stages + NEW ChunkRope /
  ChunkKvAppend (M-parallel, global positions) + ChunkAttn over BF16 cache +
  MLP, one list: worst-rel 1.14e-04, prefix-guard bitwise, deterministic,
  32 ms/chunk. Alignments: ChunkAttn float→BF16 KV (harness to qb pattern);
  ChunkGemm partial-M guards (MTP-verify M=2; ceil launch). One harness guard
  bug caught (checked a live chunk slot). Small-M verify primitive CANCELLED
  with prejudice (measured: DPAS bottoms at masked 8-row tiles, 48 layers ×
  anything ≥1 ms kills it; decode always wins small — see T7.2 verdict).
  TILED ATTENTION SCOPE (2026-09-12, the true 64K-decode answer — new kernel,
  not started): current AttnCore does ~10 GB/s effective (2% roof) at 9.8 ms
  @4K, instruction-dense (~2.5ns/op) with 24-WI occupancy on 160 EUs;
  64K projects ~250 ms/layer → ~4 s/token. Goal: FlashAttention-style tiled
  kernel (T-blocks 64-128 through SLM double-buffer, online softmax, BF16
  DPAS directly on cache rows — no convert, head×block grid for occupancy)
  reaching ≥150 GB/s effective on a T=16K probe (roofline-consistent with
  dp4a-GEMV's 50%). At 150 GB/s: 4.3 GB KV/token @64K ≈ 30 ms + linear ~50 ms
  ≈ 10-12 t/s sustained. Risks: DPAS engagement (T3.4 caveat: pure-DPAS roof
  stalled 7-9 TFLOPS — emulation suspected); SLM budget 128 KB (block 64 =
  64 KB K-double + V stream + Q/acc regs fits). Sequence: synthetic tiled
  probe → real-weight → decode_l0 adoption → re-profile.
  PROBE RESULTS 2026-09-12 (`tools/t74/report_tiled.json`, `tiledattn_replay`
  + `qkgemm_replay` ctests green): v1 plain-SYCL SLM/online scan is CORRECT
  (2.83e-06) but 37 ms @4K — scalar compute swamps the 6x traffic saving; v2
  ESIMD compute reaches 19.1 ms yet stays 2x BEHIND the 24-WI vector kernel,
  proving per-head compute (not traffic, not passes) binds every scan form.
  PIVOT TO GEMM-FORM validated at the primitive: QkGemm (fp16 Q × BF16 cache
  strided, bf16 DPAS, fp32 accum, no transpose) CORRECT (4.09e-04) at 135 µs
  for 24×4096×256 = 22x on the QK portion; 0.37 TFLOPS (DPAS weak per the
  T3.4 caveat, occupancy-thin, but the form wins). (SYCL notes: vec
  load/store are member functions; sycl::dot rejects vec<float,8>.)
  HYBRID DONE 2026-09-12 (`tools/t74/report_hybrid.json`, `hybridattn_replay`
  ctest): QK-DPAS (4 kv appends) + scalar row-softmax + cvt + WV-DPAS (4
  appends) vs attnfar-identical ref: worst 2.31e-04, 5.3 ms @4K (qk 0.34 +
  sm 2.18 + cv 0.11 + wv 2.63) = 1.9x over scan. Scalar softmax (41%) + thin
  WV occupancy (48 groups) dominate now. 64K recalibrated: softmax
  transcendental floor (~25M exp/token) caps dense decode ≈ 5 t/s regardless
  of GEMM speed — still 25x over scan-form. NEXT: vectorize softmax (~10x)
  + WV K-split 4x occupancy → ~1.3 ms @4K (~7.5x); then real-weight.
  Debug postmortems: (1) GQA design miss (all heads read kv0; standalone
  probe passed vacuously with the same wrong ref — hybrid ref caught it);
  (2) missing /16 caught by exact 16.000x ratio (probe ref equally wrong —
  both-sides trap again); (3) stale-binary false failure (built one target,
  ran another — ALWAYS full-build before verifying; T6.1 rule restated).
  HYBRID FINISH 2026-09-12 (softmax vectorized 4.7x; WV K-split + ResAddF
  combine): 3.85 ms @4K (2.5x over scan), worst 2.19e-04. K-split roughly
  neutral (2.63→2.83 ms — barriers/launches eat the occupancy gain; kept
  for large-T structure, combine is 74 µs). Bisection postmortem: double-
  offset O pointer (slab select in harness + kv rows in kernel) misfiled
  every kv — caught by slab-mass forensics + minimal repro; append-arg
  interaction fully exonerated (QK kv-loop clean, ResAddF combine bitwise).
  LOOP-VERDICT 2026-09-13 (`tools/t74/report_hybrid.json` loop_verdict):
  OOB-device-loss root-caused (QkGemm B-tile staging read past-TMAX slots;
  non-monotonic 96/112/128-run vs 98/104/108/174-crash signature; N-guard
  fix, all MAXCTX clean) + e2e quality-CLEARED (70-tok prompt, 59 vs 58
  tok, both reach `126<|im_end|>`; 25-step identical trajectory, step-25
  near-tie reroute, tails re-converge) + timing-LOSES at decode T
  (attn_layer med 1.195 vs 0.959 ms, +25%: ~23 launches+barriers/layer vs
  ~5 for scan; 2.5x synthetic win was @4K).   DECISION: DEFER for decode
  (scan stays default); GEMM-form redirects to chunked prefill where
  launch tax amortizes. Hybrid stays env-gated experimental.
  CHUNK-GEMM DONE 2026-09-13 (`tools/t74/report_chunkqkwv.json`,
  `report_chunkqkwvreal.json`; `chunkqkwv_replay` + `chunkqkwvreal_replay`
  ctests): ChunkQkGemm (causal, one M*4*nBc launch) + ChunkSoftmaxRow
  (M*24 rows) + ChunkWvGemm (one M*64 launch) — synthetic CHUNKQKWV-OK
  2.43e-04, real-weight layer-3 CHUNKQKWVREAL-OK 1.68e-04 (scan-form
  1.14e-04 — parity) with guards + determinism; M=256 scales clean
  (2.21e-04, 157 ms/chunk/layer, ~0.61 ms/token/layer compute-bound →
  64K prefill ~43 min one-time vs loop-prefill infeasible). Debug:
  ARG-STRIP trap (unreferenced functor member drops from the signature,
  indices shift — numKernelArgs probe rule, docs/toolchain.md) + harness
  WV-index copy bug. T1.4 logits batch DONE 2026-09-13 (53/60, T1.4 closed).
  64-LAYER ORCHESTRATION proven on device 2026-09-13
  (`tools/cmdlist/chunk64real_replay.cpp`, ctest): one M=32/P=0 chunk
  through all 64 real-weight layers (48 linear + 16 GEMM-form full),
  64 recorded lists, per-layer SSM/KV state, streamed weights: 1583 ms
  total, output FINITE, reset-determinism BITWISE, 3/3 runs identical.
  Debug: shared-buffer address-capture trap (upload per layer before exec),
  in_proj_z V6-rows mis-binning (device overrun segfault), NULL fill-pattern
  driver segfault — all in docs/toolchain.md. Float-ref verdict CHUNK64-OK
  2026-09-13 (`tools/t74/report_chunk64.json`: worst 1.38e-02 accum-fitted,
  mean 1.5e-04, 99.6% <= 1e-3).
  MULTI-CHUNK on device 2026-09-13 (`tools/cmdlist/chunk64mc_replay.cpp`,
  ctest green first try): 128 recorded lists, A(0..31) 1581.7 ms +
  B(32..63) 1585.7 ms with carried SSM/KV state: FINITE, continuity proven
  (B-after-A differs from B-alone), determinism BITWISE. T=64 ref verdict
  CHUNK64MC-OK 2026-09-13 (`tools/t74/report_chunk64mc.json`: worst
  1.71e-02 within 2e-2 accum budget, mean 1.46e-04) + HANDOFF SLOT-OK
  (layer-3 chunk-B Kn snapshot RNE-quantized == cache slots 32..63 bitwise
  — chunk appends land in decode_l0's (t*4+kv)*256 slot addressing).
  PRODUCTION WIDTH (M/N via CHUNK_M/CHUNK_N env, ctest contract intact):
  M=256/N=2 first try 8.65 s/chunk (0.53 ms/tok/layer), FINITE + continuity
  + BITWISE; T=512 ref CHUNK64MC-OK (`tools/t74/report_chunk64mc_m256.json`:
  worst 3.48e-03, mean 5.16e-05, SLOT-OK [256..512) bitwise).
  TRAFFIC ANALYSIS (production driver design): per-layer weight streaming =
  ~12.8 GB/chunk-pass over USB → 254 chunks ≈ 11 h — prohibitive. Production
  needs resident weights (full 16 GB arena, fits 24 GB VRAM) + stream
  re-record per chunk (64 live lists max) + decode-layout KV/SSM arenas for
  zero-copy handoff. N=8/M=256 validation running.
  RESIDENT WEIGHTS DONE 2026-09-13 (uniform-stride arenas, one pair per kind
  x64 layers, upload-once done-guard; norms stay streamed): ctest bitwise
  IDENTICAL (17364.5556/509.4838) at 16.4 s vs 49.3 s (3x, upload traffic
  gone even in validation). STREAM MODE DONE (CHUNK_STREAM=1: record+exec+
  destroy per chunk, mode field in report; stream ≡ validation bitwise at
  M=32/N=2). DECODE-LAYOUT STATE DONE (dKc/dVc 16-slot + dConv/dSsm 48-slot
  arenas with decode_l0's exact slot/sl formulas; fills now 4 whole-arena
  ops): ctest bitwise IDENTICAL again.
  N=8/M=256 resident: 8.7-9.0 s/chunk, FINITE + continuity + BITWISE, sums
  identical pre/post-resident (107189.3951/687.3606). T=2048 ref CHUNK64MC-OK
  (`tools/t74/report_chunk64mc_n8.json`: worst 1.42e-03, mean 1.15e-05,
  SLOT-OK [1792..2048); 637 s CPU). Trend: worst-rel SHRINKS with scale
  (1.7e-02 @T=64 → 3.5e-03 @T=512 → 1.4e-03 @T=2048) — fp16-noise
  accumulation is sublinear, good news for 64K.
  HANDOFF PATH (file-based, sequential processes dodge >24 GB duplex):
  chunk64mc CHUNK_DUMP_CACHES (decode-layout KV/SSM + hidlast + meta) ->
  decode_l0 --import-caches (strided KV H2D, SSM, dX inject, tail-only
  step) + --ids-file (64K prompts exceed 128 KB MAX_ARG_STRLEN) +
  short-question generalization (mP<=P: cached prefix + loop-decoded
  question). HANDOFF POSTMORTEM: first 9/9 claim RETRACTED as VOID
  (space-separated flag misparsed as stray positional — both runs
  loop-decoded; fixed: dual-form parsing + stray-positional error +
  AINFER_STEPLOG). Real import verified (banner + step-63 start): 2/9 then
  0.00-margin near-tie reroute — documented class, not a bug
  (`tools/t74/report_handoff.json`). 64K needle v2 decode running.
  INTEGRATION DESIGN (next build): single binary loads decode_l0's
  payArena/scArena once; prefill records against arena offsets (entry
  d_off/sc_off); prefill writes decode-layout KV/SSM (proven above);
  decode loop starts at pos=P on the same arenas. Kills both the 11 h
  re-upload traffic AND the 2x-weights (>24 GB) duplicate-arena trap.
  NEXT: prefill-quality arbitration on real token ids (chunked-prefill +
  argmax vs T1.5 streamed-BF16 truth), then 64K corpus + production
  prefill→decode integration.
  64K NEEDLE CORPUS DONE 2026-09-13 (`tools/t74/mk_needle.py`,
  `tools/t74/needle_corpus.json`): 5 haystack variants x 65024 tokens
  (needle at 0/25/50/75/100% depth, per-depth magic codes, round-trip
  verified) — exact-match retrieval needs no CPU reference.
  PRODUCTION INTEGRATION SPEC (owned next workstream, not started):
  (1) chunk driver: 256-token chunks x 64 layers on the chunk64mc pattern
  (M-scaling proven at layer level; per-chunk ~10 s full-model);
  (2) cache import: prefill writes decode_l0's arena KV (BF16,
  (t*4+kv)*256 — SLOT-OK proven) + per-layer SSM state (same dims both
  sides), decode starts at pos=P; (3) measured math: 64K prefill ~43 min
  one-time, then scan decode; (4) eval: greedy 512 tokens per variant,
  exact code match at all 5 depths.
  SINGLE-BINARY MERGE DONE 2026-09-15 (`tools/decode/decode_l0.cpp`
  `--prefill-chunks`, `tools/t74/report_merge.json`): 8 chunk modules
  appended (existing enum indices untouched), per-chunk 64-layer record
  against SHARED payArena/scArena + decode-layout KV/SSM + preloaded norms,
  stream record+exec+destroy, D2D last row to dX, existing loop seam.
  Score S==Sm aliased (bitwise-safe, budget ~21.4 vs 22.71 GiB).
  Gates: default 70-tok e2e intact (58 tok has126); M=32/N=2 prefill
  BITWISE-identical to chunk64mc (0.000e+00); handoff parity first-5;
  64K needle v2 in ONE process (254x256, 4793617 ms, `502817`+EOS —
  identical to file path, zero re-uploads, zero disk). Bugs: report-JSON
  comma, space-form flag, legacy a/b fallback (all fixed+verified).
  MERGE-OK: recommended production path. Next: linear-list sharing.
  DRAFT SLICE DONE 2026-09-12 (`tools/cmdlist/mtpdraft_replay.cpp`, ctest,
  `tools/t72/report_mtpdraft.json`): full MTP-1 draft
  (pre_fc norms + concat + fc + full layer with own KV + norm + lm-head +
  argmax) as ONE recorded list on real weights: hidden worst-rel 4.46e-08,
  draft token exact, top5 5/5, reset-deterministic, 6.4 ms/draft. Needed one
  new 10-line kernel (Concat2) + the standard dead-strip guard. Left:
  chunk production, then chunked verify + loop integration.
  REVISIT 2026-09-15 (`tools/t72/report_mtp_revisit.json`,
  `tools/t72/pl_accept.py`): 64K line landed — re-derived verify economics
  with measured 64K numbers. Prompt-lookup drafts: alpha = 0/65 (draftable
  ≤2/65) on repetitive-boilerplate continuation vs loop greedy truth —
  model generates novel text, n-grams never fire; 95% upper ~0.046 <<
  0.6 gate. DEAD at every context length. MTP-head drafts (alpha 0.624
  stands): verify model C(8,65K) ≈ 1.7 s (1.09 fixed measured + 0.57
  attention slope) vs 8 × 1.65 s sequential → E(8)=2.62, speedup ≈ 2.5x
  AT 64K ONLY (short-context still 0.24x loss) — economics reversal, but
  chained drafting unproven + 64K alpha unmeasured + no 64K production
  traffic. VERDICT: RE-DEFER with upgraded triggers (was architecturally
  impossible; now uneconomical-until): (1) production 64K decode traffic,
  (2) chained-draft proof, (3) 64K alpha confirmation. No code changes
  (measurement + analysis only); draft slice + chunk machinery reusable.
- Done: Enabled only with measured net speedup and preserved quality, or remains dropped.

### T7.3 OpenAI-compatible HTTP daemon
- Status: `[x]` (DONE 2026-09-10)
- Deps: T4.4, T6.4
- Do: Add a lightweight HTTP endpoint matching `/v1/completions` after CLI correctness and
  performance are stable.
  DONE (`tools/http/ainfer_http.py`, `report_t73.json`, stdlib only):
  /healthz, /v1/models, POST /v1/completions (full JSON + real SSE streaming
  by parsing unbuffered per-step lines). Validated: golden prompt completes
  sanely, identical text twice, streamed join == non-stream exactly
  (9 chunks + stop + [DONE]), 400/400/404 error paths. Single-flight lock;
  per-request spawn keeps the honest ~44 s USB-load TTFT (persistent worker
  follow-on). Debug finds: repo-path depth, stdbuf+exec-as-program.
- Done: Endpoint returns correct streamed completions.

### T7.4 Extended contexts / batching (scoped separately)
- Status: `[x]` (CLOSED 2026-09-15: 64K line complete — chunked prefill production proven, 64K needle quality 5/5; single-binary merge left as perf follow-up, not a gate.)
- Deps: T5.1, T6.2
- Do: Consider larger contexts, sliding-window KV, or limited batching as new scoped work.
  SCOPE DECIDED 2026-09-10 (user): larger contexts at **64K**; batching DROPPED
  (batch-1 sufficient, and batching breaks the fixed-address recorded loop);
  sliding-window KV DROPPED (changes model semantics; full 64K KV fits, so no
  need to pay the quality price). 64K feasibility (computed): weights 14.88 +
  KV BF16 4.0 + SSM 0.15 + runtime ~2.0 = ~21.1 GiB -> fits with ~2.9 GiB
  margin. HARD BLOCKERS owned elsewhere: (1) current loop stores FP32 KV
  (8.0 GiB at 64K -> 25.1 GiB total, DOES NOT FIT) => BF16-or-better KV
  required (T6.3 KV-quant scope); (2) loop-decode prefill over 64K tokens ~
  hours => chunked/tiled prefill required (T3.4 prototype exists, needs
  productization); (3) long-context quality validation (needle-style, needs
  eval corpus beyond T1.5). Decode speed projects fine (~19.5 GB/token ->
  ~15 t/s at roof).   AttnCore wts buffer + RoPE/dWts tables must size to 64K
  (trivial).   Implementation order: KV precision -> chunked prefill ->
  64K alloc/validation. PLATFORM RULE (2026-09-11): no bigger host is coming;
  all validation executes on the B60 + this box. 64K-length quality is gated
  on chunked-prefill production (B60 work), NOT on new hardware; instructive
  quality runs at ≤256-token native contexts (T6.2 envelope). CHUNK-GEMM RISK RETIRED 2026-09-10
  (`tools/cmdlist/chunkgemm_replay.cpp`, ctest, `report_chunkgemm.json`):
  tiled INT4 DPAS GEMM (MT4 design) records/replays under raw L0 at
  256x5120x17408, worst-rel 1.00e-06, deterministic, 2.54 TFLOPS (below T3.4
  SYCL 4.3 — runtime blocking likely costs unroll; DPAS engaged, sufficient).
  256-token chunk gate GEMM ~18 ms vs ~17 s loop-decoded (~150x on linear ops).
  Build notes: SG16 forced structurally (attribute spelling rejected);
  +SPV_KHR_cooperative_matrix needed; unused-M dead-strip guard (TMAX lesson).
  REMAINING for 64K: chunked causal attention + SSM chunk orchestration, then
  64K alloc/validation runs. CHUNK-SSM DONE 2026-09-10
  (`tools/cmdlist/chunkssm_replay.cpp`, ctest, `report_chunkssm.json`):
  conv+recurrence over 32-token chunks with persistent history/state across
  chunk replays, worst-rel 8.11e-07, cross-chunk continuity proven
  (B-after-A differs from B-alone), reset-deterministic bitwise. Scalar
  ~211 ms/chunk — vectorize before production (same queue as single-step).
  Still remaining: 64K alloc/validation runs (needs chunked orchestration of
  a full layer + caches at 64K sizing). CHUNK-LAYER DONE 2026-09-10
  (`tools/cmdlist/chunklayer_replay.cpp`, ctest, `report_chunklayer.json`):
  full linear layer as ONE ~330-launch recorded list over a 2-chunk sequence
  (chunk GEMMs + chunk SSM + per-row stages + in-list fp16 converts),
  worst-rel 1.77e-04 (fp16 path), continuity proven, reset-deterministic.
  Debug finds: stride-6144-vs-10240 recur-output aliasing (both sides agreed
  and both wrong — separated into dMxR/hMxR) + a missed norm group-size
  override, both caught by stage forensics. ~224 ms/chunk (scalar SSM);
  vectorize + batch stages before production. Left: 64K alloc/validation. ALLOC PROOF DONE 2026-09-10
  (`tools/kvfit/kvfit.cpp`, ctest `kvfit`, `report_kvfit.json`): full
  decode_l0 footprint at MAXCTX=65544 with BF16 KV allocated on the real
  22.71 GiB heap — 19.08 GiB total (arenas 15.24 + KV 4.0 + SSM 0.15 + rest),
  margin 3.63 GiB, FIT-PASS (better than the 2.9 estimate). CHUNK-ATTN RISK RETIRED 2026-09-10
  (`tools/cmdlist/chunkattn_replay.cpp`, ctest, `report_chunkattn.json`):
  causal chunk attention (256 over prefix-128, control-driven chunk_start) in
  a recorded list, worst-rel 1.03e-06, deterministic. Correct-but-slow scalar
  (~71 ms, same class as pre-vectorized SSM) — ESIMD-vectorize before
  production use. Debug: WI index used 6144 not 24 (only row 0 computed).
  64K VALIDATION DONE 2026-09-11 (`tools/t74/report_64k.json`): the real
  64-layer loop at AINFER_MAXCTX=65544 (grow-only override in `decode_l0`,
  T4.5 guard intact) with 4 GiB BF16 KV + 64K RoPE tables + 64K strides is
  TOKEN-IDENTICAL to small sizing ([369,279,248046], top5 to last digit).
  Cache zero-init moved to device fill (4 GiB never crosses PCIe). Probes:
  RoPE-at-max vs HF (inv identical; 2.4e-03 rotor diff at t=65533 is HF's own
  float32-inv error, ours double-derived is the accurate side); far-slot
  KvAppend at {0,32767,65533} bitwise + guards zero (`kvfar_replay`, ctest);
  AttnCore correctness to T=4096, 2.01e-06 (`attnfar_replay`, ctest; scalar
  153 ms@4K on random caches — vectorize before production, roofline still
  supports ~10-15 t/s). Left: production hardening + quality runs against the
  box-native T1.5 corpus (re-scoped 2026-09-11; 64K-length quality awaits
  chunked-prefill production, not new hardware).
  Still remaining: SSM chunk orchestration + 64K alloc/validation. PRECISION ORDERING (decided 2026-09-10): BF16 KV
  first (near-zero quality risk — model is BF16-native; halves 8.0 -> 4.0 GiB
  and unblocks the 64K fit), INT8 KV second as the performance follow-up
  (~2.0 GiB + ~10-15% long-context token time; K per-channel / V per-token
  scales, quality-gated). BF16 KV DONE 2026-09-10
  (`tools/t74/report_bf16kv.json`): RNE-quantize on append + exact dequant in
  AttnCore, same arg layout; all allocs/zero-inits/slot byte-math converted;
  attn/rope/layerattn green (rope slot bits bitwise-identical); layer3real
  2.18e-05 (inside envelope); short prompt back to [369,279,248046] via flip
  dynamics. Wts-orphan bug had also taken layer3real — audit now covers all
  four L0 AttnCore users. INT8 stays queued behind T6.3.
  PREFILL-QUALITY ARBITRATION DONE 2026-09-13
  (`tools/t74/report_prefill_arb.json`, `tools/forward/prefill_arb.py` +
  `arb_ref.py`): first 64 ids of the e2e prompt -> host BF16 embed ->
  chunk64mc with host inputs -> device chunk-B hidden -> BF16 lm_head top-1
  vs T=64 INT4-float chain + BF16 head: 32/32 top-1, 0 near-tie misses —
  chunked-prefill noise flips nothing at 64-token scale on real ids.
  OPEN (owned next workstream): production chunk driver (256-token chunks x
  64 layers) + cache import into decode_l0 + 64K needle eval (5/5 depths
  exact-match) per the integration spec above.
  CLOSE-OUT 2026-09-15 — all gates passed: production driver live
  (`CHUNK_M`/`CHUNK_N`/`CHUNK_STREAM`, resident weights, decode-layout
  state); file-handoff path (dump + `--import-caches` + `--ids-file` +
  question); 64K needle quality 5/5 exact-match (`tools/t74/
  report_needle.json`: v0 739521, v1 184963, v2 502817, v3 926438,
  v4 317654 — each exact code + EOS in 8 tokens). Postmortems: VOID 9/9
  flag-misparse, tmpfs-full dump death, transient device loss (CHECK now
  exits). Follow-up (perf, not gates): single-binary prefill→decode merge,
  linear-list sharing across chunks, GEMM-form long-context decode.
- Done: Scoped as follow-on effort with its own gates — all met 2026-09-15.

---

## Cross-Cutting Tracks

### X1 Continuous verification suite
- Status: `[x]` (DONE 2026-09-12: two-tier manual gate, no CI server on this box)
- Deps: T1.4
- Do: Maintain unit (container/tokenizer/pack/swizzle/budget/sampling), kernel, integration,
  and quality test tiers; run on every change.
- Done: CI-style suite runs and gates merges.
- Gate (47 tests, post-T8.4): per-change FAST gate `ctest --preset b60 -E
  "^(l0load|chunkgemm_replay|chunklayer_replay|chunklayerreal_replay|chunkattnreal_replay|chunkqkwvreal_replay|chunk64real_replay|chunk64mc_replay)$"`
  (39 tests, excludes 8 heavy proofs); FULL gate is bare `ctest --preset b60`
  (47/47 expected; Stamp 13 was 45/45 pre-T8.2). l0negatives + kvfit + attni8 + mtpdraft + chunkqkwv + reports_json + tokenizer_golden all
  in the fast tier by construction (ms-scale).

### X2 Documentation and reproducibility
- Status: `[x]` (audited 2026-09-12)
- Deps: T0.1
- Do: Keep environment, model manifest, memory budget, and benchmark metadata current.
- Done: Docs match the current implementation state.
- Audit notes: plan.md §13 stale DONE-markers fixed; AGENTS.md Code-so-far
  extended (cmdlist/decode_l0); memory_budget.json left as the v1 baseline
  (still accurate for the default BF16-KV path — INT8/64K measured variants
  live in task reports by design, not duplicated); binfer_spec normative and
  unchanged (no format change since v1).

---

## Phase 8: Release Hardening (review.md §4/§6, IDs T8.1–T8.11 stable)

### T8.1 Create current-state and release-support matrix (P0)
- Status: `[x]` (DONE 2026-09-16: `STATUS.md` — supported config, runtime paths,
  quality/perf snapshots, experimental/deferred table, known issues, Stamp 13)
- Deps: none. Done: new reviewer can identify the release configuration without
  reading the full changelog.

### T8.2 Normalize report schema and strict JSON validation (P0)
- Status: `[x]` (DONE 2026-09-16: `tools/http/report_t73.json` trailing-`}`
  fixed — 91/91 tracked reports strict-parse; permanent host-only ctest
  `reports_json` gate via `tools/validate_reports.py`, in FAST tier.
  Negative control: pre-fix run flagged exactly the t73 file. Common-envelope
  schema unification across reports left as follow-up, not a gate.)

### T8.3 Document and classify weight-stat anomaly rules (P0)
- Status: `[x]` (DONE 2026-09-16: `tools/weightstats_classify.py` →
  `reference/weight_stats_v2.json`; v1 kept unchanged. Methodology: peer-group
  (scope, class) robust-z on maxabs, threshold 5.0, shapes from headers only.
  136/136 classified: 61 expected-architectural (A_log/dt_bias, keep_bf16),
  64 out-of-scope vision (reject_loader_v1), 11 quantization-sensitive
  (keep_bf16 / int4_g128_per_policy); suspected_source_corruption: [].
  Actions grounded in `binfer.py quantize_policy`.)

### T8.4 Expand tokenizer multilingual and template golden suite (P0)
- Status: `[x]` (DONE 2026-09-16: `validate_t43.py` extended 14→27 parity
  cases (decode skip True/False, whitespace, malformed markup, multi-turn
  think on/off) — 27/27; pinned `golden_t44.json` v1 (encode/decode/templates/
  prompt_ids + 5 asset sha256) with `validate_golden.py` (no transformers) —
  GOLDEN-OK 25/25, ctest `tokenizer_golden` (0.44 s, FAST tier). Real parity
  gap fixed: `tok.render_chat` StrictUndefined→lenient (HF tolerates assistant
  msgs without tool_calls). Checksum finding: manifest sha256 MATCH all 5
  tokenizer assets (content pin); crc32.txt coincides only for the 3 large
  stable assets (merges/tokenizer/vocab) — hub metadata, not a content gate.
  `tokenizer_t14.json` standalone failing fixture kept unchanged.)

### T8.5 Build 200-plus-case quality benchmark (P1)
- Status: `[ ]` (open: 40/40/30/20/30/20/20 split per review §3.1; Gate C)

### T8.6 Add margin-aware divergence report (P1)
- Status: `[ ]` (open: BF16 margin, overlap, first-divergence, reconvergence per top-1 diff)

### T8.7 Merge chunk prefill and decode into one process (P2)
- Status: `[x]` (DONE 2026-09-15: single-binary `--prefill-chunks`, MERGE-OK,
  in-process 64K HIT, `tools/t74/report_merge.json`; see merge block above)

### T8.8 Implement persistent HTTP worker (P3)
- Status: `[ ]` (open: load-once, /healthz//readyz, cancellation, queueing,
  timeouts, reset, device-loss restart; 10-request isolation test; Gate E)

### T8.9 Introduce typed arena spans and checked offsets (P4)
- Status: `[ ]` (open: review §3.5 items 1–6)

### T8.10 Add fault injection and sanitizer track (P4)
- Status: `[ ]` (open: review §3.5 items 7–8; Gate B edge cases)

### T8.11 Publish controlled AInfer versus llama.cpp benchmark (P5)
- Status: `[ ]` (open: review §3.2 controls; Gate D)

---

## Immediate Next Actions (from plan.md section 13)

1. ~~T1.1 - Confirm and pin the exact Qwen model and tokenizer revisions.~~ DONE 2026-09-07.
2. T1.2 / T1.6 - Emit the manifest file and hybrid memory budget (KV over 16 layers + SSM state).
3. T0.1 / T0.2 / T0.3 - Pin the Intel stack and run the Level Zero/ESIMD probe.
4. T1.4 / T1.5 - Capture reference logits, greedy outputs, and quality samples from SafeTensors source.
5. T3.1 - Measure sustainable bandwidth and dispatch overhead on the B60.
6. T3.2 / T3.3 - Select the first INT4 layout and decode GEMV design from data.

Note: vision encoder deferred (text-only v1); GGUF is baseline-only, not a quantizer input.
