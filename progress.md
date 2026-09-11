# AInfer Implementation Progress

> Template for tracking implementation status against `tasks.md` and `plan.md`.
> Update this file as work progresses. Keep `tasks.md` as the source of truth for scope;
> use this file for live status, evidence, and history.
>
> Status legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` dropped
> Copy the relevant task table row status into the Dashboard on each update and append to Changelog.

## 0. Meta

- Last updated: 2026-09-10
- Updated by: assistant
- Current focus: T6.1 profile-driven optimization of the adopted loop (Phase 5 complete 6/6, gate closed; stress 5/5 green)
- Overall health: Green
- Headline: Model pinned (Qwen3.8-27B@1d4bf0f, SafeTensors mirror validated); MTP confirmed; text-only v1, vision deferred

### Verification Stamp

> ✅ **CHECKED — 2026-09-07**
>
> Reviewed `tasks.md`, tracked artifacts, and executable verification on the Ubuntu/B60
> host. `tools/binfer.py validate`, `negatives`, and `mlpcheck` pass; the real container is
> 15,978,603,616 bytes with 866 tensors, 505 INT4 tensors, SHA-256 prefix `5ef77c128ee526a3`,
> and layer-0 MLP max absolute difference 0.00171. Status rows accurately distinguish
> completed work from partial work. This stamp does not certify the pending C++ negative-test
> suite, full-model logits/quality baseline, or later runtime phases. (2026-09-07 update:
> since stamping, the C++ L0 loader loaded + CRC-verified 866/866 tensors on the B60 —
> see T2.6 changelog entry; standalone C++ rejection tests still pending.)

### Verification Stamp 2

> ✅ **CHECKED — 2026-09-07 (post-B60 implementation audit)**
>
> Rebuilt the `b60` preset and reran all hardware tests added since the previous stamp:
> CTest 3/3 passed (`esimd_smoke`, `l0load`, `bench_t31`). The probe selects PCI
> `8086:e211`; ESIMD SG16/SG32 and XMX INT8/FP16/BF16 tiles pass exactly; raw L0
> timestamps work; the C++ loader allocated the 15.98 GB payload+scale arenas and
> read back 866/866 tensors with zero CRC failures. The rerun measured 437.07 GB/s
> D2D and 430.8 GB/s verified ESIMD copy, with 4.26 µs launch and 4.78 µs replay.
> llama.cpp SYCL baseline rerun: pp512 183.97 t/s, tg128 14.81 t/s (consistent with
> the stored 183.87/14.86 baseline). This stamp certifies Phase 0 and completed
> T1.6/T2.3-T2.6/T3.1 evidence only; T2.2 remains partial until C++ negative tests,
> and it does not certify T1.4/T1.5 or later kernels/runtime phases.

### Verification Stamp 3

> ✅ **CHECKED — 2026-09-08 (kernel/reference audit)**
>
> Rebuilt the current `b60` preset and reran CTest: 8/8 passed, including full
> 15 GB loader readback, bandwidth/dispatch, DP4A probe, 24-row GEMV shootout,
> RMSNorm, prefill GEMM, and attention/RoPE/MLP-elementwise tests. Artifacts parse
> and agree with recorded evidence: DP4A GEMV peaks at 293.6 GB/s, RMSNorm has six
> verified cases, prefill MT4 peaks at 4.327 TFLOPS, RoPE max error is 2.38e-07,
> attention weights 2.84e-08, and decode attention+o_proj+residual 1.28e-06 relative.
> The audit found and corrected status drift: duplicate T3.3 row removed; T3.6 and
> T3.8 downgraded to partial because KV boundary/fusion work and corrected-(1+w)
> quantized full-MLP parity remain. This stamp certifies completed T3.2-T3.5 and
> the explicitly completed portions of T3.6-T3.8; it does not certify prefill/SSM
> attention, max-context KV behavior, full block assembly, or later runtime phases.
>
> Post-stamp corrected `mlpcheck`: max 1.62375, mean 0.063877 (~6.9% of mean
> |reference|). The old 0.00171 claim is invalid/superseded; T3.8 remains partial
> pending mixed-precision or calibration work.

### Verification Stamp 4

> ✅ **CHECKED — 2026-09-08 (T4.2/T4.3/T4.4 audit)**
>
> Rebuilt the current B60 preset and reran the complete hardware suite: **10/10
> CTest targets passed** (`esimd_smoke`, `l0load`, `bench_t31`, `dp4a_probe`,
> `gemv_t32`, `norm_t35`, `prefill_t34`, `attn_t36`, `ssm_t37`, `sample_t39`).
> The CPU forward artifact contains S0/S1/S2/S3; tokenizer validation is 14/14;
> CLI validation is 3/3; device argmax is 5/5; all referenced reports parse.
> T4.2 CPU evidence is accepted: streamed 64-layer BF16 prefill is finite, S0
> interval layers match HF, and full INT4-vs-BF16 top-1 agrees 4/4. The S3 llama
> comparison is explicitly invalid as configured because the two paths used
> different thinking/chat templates; no cross-backend quality claim is made.
> This stamp certifies the hardware suite and CPU T4.2/T4.3 evidence only; native
> device full-forward/decode, T4.5, T5.x, and T1.5 quality remain open.

### Verification Stamp 5

> ✅ **CHECKED — 2026-09-08 (device block/decode-loop audit)**
>
> Rebuilt the current B60 preset and reran the full suite: **14/14 CTest
> targets passed**, adding `blkexec`, `blkblock`, `blklinear`, and `decode` to
> Stamp 4's coverage. The full 15 GB arena-backed path is exercised by real
> Qwen weights: direct-arena DP4A GEMVs pass, L3 attention-half has 9/9
> device-vs-host-INT4 checks at about 2e-07, and L0 linear block mid/final
> outputs pass at 2.86e-08/6.72e-09 after fixing its 3x-undersized conv-state
> allocation. The 64-layer loop persists 16-layer KV plus 48-layer conv/SSM
> state and generates `[271, 220, 220]` from the recorded 4-token prompt.
>
> This stamp certifies execution, arena addressing, and stage-wise
> device-vs-host-INT4 arithmetic. It does **not** certify token-equivalent
> generation: device uses INT4 weights plus per-tensor INT8 activations, while
> the existing CPU comparison uses INT4 plus FP32 activations. Their top-5 sets
> differ (device top-1 271 vs CPU 369); the existing logit std is close
> (2.18 vs 2.33), but that is insufficient for acceptance. A CPU oracle with
> the identical INT8 activation policy is the next correctness gate.
>
> ### Verification Stamp 6
>
> > ✅ **CHECKED — 2026-09-08 (device loop validation)**
> >
> > The 271-vs-369 mismatch that motivated Stamp 5's oracle is resolved: root
> > cause was a `uint16_t*` + byte-offset embedding lookup reading at twice the
> > intended arena offset (one-line fix; all other arena accesses already went
> > through `char*`). After the fix the native loop generates `[369, 279,
> > 248046(EOS), 198]`: step-3 top-4 EQUALS the CPU-BF16 reference in order
> > ([369,513,248046,1503], values within 0.5), per-layer states match HF-BF16
> > decoder outputs at 1.2–2.2% (the INT4+INT8 envelope), and generation stops
> > correctly at EOS. T4.2 is marked done; the intermediate INT8 CPU oracle
> > disagreed with both validated sides and is recorded as superseded.
> > Remaining: T4.5 native end-to-end suite (including a proper matched-template
> > greedy-to-completion comparison vs the llama reference), T5.x static
> > scheduling, and T1.5/T6.3 quality work.

### Verification Stamp 7

> ✅ **CHECKED — 2026-09-10 (post-Stamp-6 runtime/scheduling audit)**
>
> Audited the implementation and artifacts added since Stamp 6. The native
> `tools/e2e/run_e2e.py` result is **6/6**: the 70-token matched thinking
> template completes through `126` (matching llama), one-token and 64-token
> cases are finite, repeated generation is token-identical, and truncated and
> garbage containers are rejected with diagnostics. The prior collapse was
> correctly reclassified: `MAXCTX` was sized before `--ids` / `--max-new`
> overrides, causing cache/RoPE OOB access at position 79. The corrected order
> and position guard are present and the native suite passes.
>
> T6.3's early activation-policy change is independently justified on real L0
> weights: per-group-128 INT8 dG17 scales reduce zero-rate 16.17% to 8.27%,
> improve SNR 30.29 to 37.06 dB, and reduce downstream down-projection
> mean-relative error 0.02581 to 0.01166. The decode loop uses those group
> scales at every GEMV site. T5.1/T5.2's static-liveness audit verifies
> init-only small-weight loading and no release-path allocation, file reads,
> vector resize, or small-weight H2D in the token loop. It also corrected a
> 16x KV allocation bug (8.0 GiB phantom allocation at 4K); exact FP32 KV is
> 512 MiB at 4K, not the earlier erroneous 537 MiB report.
>
> Rebuilt the B60 configuration and ran the full suite: **16/16 CTest passed**
> (547.95 s). After tightening the final report-vector allocation, rebuilt and
> reran the affected targets: **decode, bench_lists, kernel_replay 3/3 passed**.
> T5.3 mechanics are accepted: regular-list replay is 9.89 us for its 1 MiB
> copy benchmark; a closed raw-L0 list replays the decode-exact I=17408
> silu-mul SPIR-V kernel at 10.03 us, deterministic across 50 changing inputs
> and within its 2e-6 reference tolerance. SYCL Graph is unavailable on this
> backend; raw L0 is the validated recording path.
>
> This stamp does **not** certify whole-decode command-list replay: ~700
> synchronous SYCL submissions and activation-scale D2H/H2D round trips remain.
> It also does not close T1.5's quality baseline, T4.1/T4.4 integration status,
> KV quantization, or the remaining Phase 5 stress/scheduling work.

### Verification Stamp 8

> ✅ **CHECKED — 2026-09-10 (post-Stamp-7 recorded-kernel audit)**
>
> This is a scoped incremental audit. It certifies the raw-L0 command-list
> work added after Stamp 7, not the pre-existing foundational kernels. The
> required cmdlist regression suite is **6/6**: `bench_lists`, `kernel_replay`,
> `control_replay`, `list_replay`, `gemv_replay`, and `attn_replay`.
>
> `extract_spv.py` correctly follows the required ESIMD-compatible pipeline:
> `objcopy` bundle extraction, `sycl-post-link -split=auto -split-esimd
> -lower-esimd`, then `llvm-spirv` with the driver's GenX intrinsic allowance
> and extension set. It emits content-selected per-entry modules, avoiding
> unstable post-link image numbering. The direct `llvm-spirv` path is correctly
> excluded because it fails on ESIMD's 32-wide vector representation.
>
> The recorded kernels are accepted: closed-list silu-mul is deterministic;
> `DecodeControl` updates drive a recorded position-dependent kernel without
> arg mutation (explicit 16 B copy remains the measured policy); the live
> norm->silu->res list is deterministic across 50 replays; decode-exact ESIMD
> INT4 GEMV at 17408x5120 is deterministic across 20 replays with 2.57e-07
> worst relative error; and decode-exact GQA+gate is deterministic across 20
> replays with variable Ctrl[2]-bound context lengths T=1..64 and 8.06e-07
> worst relative error. The attention port's `TMAX` clamp fixes the discovered
> dead-argument issue and hardens its loop bound.
>
> Native runtime regression remains green: `tools/e2e/run_e2e.py` is **6/6**,
> including the matched-template `126` completion, determinism, max-64, and
> malformed-container diagnostics. A complete B60 build plus **20/20 CTest**
> pass (645.10 s) was also run as additional confidence evidence; it was not
> required to establish this incremental stamp.
>
> This stamp does **not** certify a recorded full decode loop. The raw-L0 ports
> for conv/SSM recurrence, RoPE/KV append, and argmax remain absent, and
> `decode.cpp` still builds roughly 700 synchronous SYCL submissions/token.
> T5.4 is mechanism-complete but awaits raw-L0 decode integration; T1.5 quality
> metrics and KV quantization remain open.

### Verification Stamp 9

> ✅ **CHECKED — 2026-09-10 (Phase 5/6 closure audit)**
>
> Audited all implementation and evidence added after Stamp 8: T5.1/T5.2
> static scheduling, the full T5.3 raw-L0 port set plus 64-layer adoption
> (`decode_l0`), T5.4 control policy, T5.5 token return, T5.6 stress, T6.1
> profile baseline with SSM-vectorize and norm-parallelize, T6.2 sweep, T6.4
> benchmark report, and the Phase 5 gate closure. Status rows verified:
> T5.1–T5.6 `[x]`, T6.1/T6.2/T6.4 `[x]`; Phase 5 `[x]` 6/6 with gate closed;
> Phase 6 `[~]` 3/4. All 41 `report*.json` evidence files under `tools/` parse
> (verified).
>
> Rebuilt from a visible configure and ran the complete B60 suite: **25/25
> CTest passed** (472.87 s). One failure occurred mid-audit and was
> root-caused before stamping: the AttnCore Wts refactor orphaned
> `layerattn_replay` (arg 7 never set → NULL pointer → deterministic
> DEVICE_LOST at first execute). Fixed, re-verified green, and recorded as a
> standing rule — kernel-signature changes require a repo-wide call-site
> audit. Corrected before stamping: stale SPIR-V flow comments, a wrong
> position-sequence in `report_list.json`, `report_control.json` build-flow
> metadata, missing output-capacity reserves in `decode_l0`, and two
> miscounted `_ZTS` entry names (caught loudly by content selection, as
> designed).
>
> Native SYCL e2e rerun on the current tree: **6/6**. Recorded-loop
> matched-template rerun on the current (post-optimization) binary: completes
> 78 tokens to EOS with the correct answer 126. The trajectory is NOT
> token-identical to the SYCL loop — expected and accepted: OPT 1+2 moved
> last-ulp numerics and flip dynamics rerouted the reasoning trace
> (documented T6.1 cost). The e2e bar (completion + correct answer) holds on
> both loops.
>
> This stamp does **not** certify a T1.5 quality baseline, T6.3 fusions or KV
> quantization, chunked prefill, MTP/speculation, HTTP, or any Phase 7
> optional. Power/thermal telemetry remains unavailable on this box, so the
> T6.4 report's energy entries stay explicitly empty.

## 1. Dashboard

| Phase | Scope | Done / Total | Status |
| ----- | ----- | ------------ | ------ |
| 0 - Environment & Foundations | T0.1-T0.5 | 5 / 5 | `[x]` |
| 1 - Model Spec & Reference | T1.1-T1.6 | 4 / 6 (+2 partial) | `[~]` |
| 2 - Container & Exporter | T2.1-T2.6 | 5 / 6 (+1 partial) | `[~]` |
| 3 - Kernel Microbenchmarks | T3.1-T3.9 | 7 / 9 (+2 partial) | `[~]` |
| 4 - End-to-End Runtime | T4.1-T4.5 | 3 / 5 (+2 partial) | `[~]` |
| 5 - Static Scheduling & Memory | T5.1-T5.6 | 6 / 6 | `[x]` |
| 6 - Performance Tuning | T6.1-T6.4 | 3 / 4 | `[~]` |
| 7 - Optional Features | T7.1-T7.4 | 2 / 4 | `[~]` |
| X - Cross-cutting | X1-X2 | 0 / 2 | `[ ]` |

Next up:

1. T6.1 (close-out) - optional small win: fuse KMax+Quantize into one launch to remove the last per-GEMV host sync pair (~5%); GEMV/tail micro-fusions rejected as sub-5%.
2. T6.3 - the remaining actionable item is the quality arbiter (T1.5 corpus), which needs a 64 GB+ host; KV quantization gated behind it.
3. T7.1 - configurable sampling (temperature/top-k/top-p) as the next feature, following the plan's greedy-first ordering.

Blocked / waiting:

- T1.4 logits/greedy, T1.5 metrics - need 64 GB+ machine with `qwen3_5` modeling code
- (unblocked by move: T0.2/T0.3/T3.1/T1.6-alloc/T2.6 - B60 present)

## 2. Phase Gates

| Gate | Depends on | Status | Date | Evidence / Notes |
| ---- | ---------- | ------ | ---- | ---------------- |
| Phase 0: device identified, kernel runs, env reproducible | T0.1-T0.5 | `[x]` | 2026-09-07 | Toolchain pinned, probe/smoke/bench/loader all run on B60, reference baseline stored |
| Phase 1: assumptions replaced, model fits with margin | T1.1-T1.6 | `[ ]` | YYYY-MM-DD |  |
| Phase 2: model loads reproducibly, tensors verifiable | T2.1-T2.6 | `[ ]` | YYYY-MM-DD |  |
| Phase 3: operators pass tests, roofline benchmarks exist | T3.1-T3.9 | `[ ]` | YYYY-MM-DD |  |
| Phase 4: accepted outputs, stable repeated runs | T4.1-T4.5 | `[ ]` | YYYY-MM-DD |  |
| Phase 5: no alloc or cmd-list construction in steady state | T5.1-T5.6 | `[x]` | 2026-09-10 | decode_l0: 66 recorded lists, zero per-token construction/allocation (T5.1-T5.3); control-only mutation (T5.4); 4 B token return (T5.5); stress 5/5 (T5.6) |
| Phase 6: stable perf, explained by profiles, quality kept | T6.1-T6.4 | `[ ]` | YYYY-MM-DD |  |

## 3. Phase 0 — Environment and Foundations

| Task | Title | Status | Owner | Started | Finished | Evidence (commit/PR/artifact) | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | ------------------------------ | ----- |
| T0.1 | Pin software and hardware stack | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `docs/toolchain.md`; L0 headers vendored v1.28.2 |  |
| T0.2 | Level Zero device-capability probe | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `tools/l0probe/report_b60.json`: 160 EUs, 22.71 GiB heap, SG [16,32] | Deps: T0.1 |
| T0.3 | ESIMD and XMX capability check | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `report_esimd.json`: ESIMD SG16/32 ok; DPAS int8/fp16/bf16 exact; INT4 native = false | Deps: T0.2 |
| T0.4 | Build system and smoke test | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `CMakePresets.json` (b60/host); `ctest --preset b60` 1/1 pass; SYCL+L0 timestamps | Deps: T0.1 |
| T0.5 | Reference-runtime baseline capture | `[x]` | assistant | 2026-09-07 | 2026-09-07 | llama.cpp SYCL: pp512 183.87, tg128 14.86 t/s (`baseline_t05.json`) | Deps: T0.1 |

## 4. Phase 1 — Model Specification and Reference Suite

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T1.1 | Pin exact model and tokenizer revision | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `models/Qwen3.8-27B` (18/18 shards, headers valid); rev `1d4bf0f` | Gate A |
| T1.2 | Generate architecture manifest | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `models/Qwen3.8-27B/manifest.json` v1.0; 1199 tensors, 100% index coverage, 14 sha256 | Deps: T1.1 |
| T1.3 | Confirm or drop MTP/speculative heads | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `mtp_num_hidden_layers`=1, no dedicated embeddings | Gates T7.2 (existence unblocked) |
| T1.4 | Capture numerical reference outputs | `[~]` | assistant | 2026-09-07 |  | `reference/`: operator fixtures, real-weight MLP-L0, 1199-tensor stats, tokenizer report; logits/greedy blocked (no full-model run on this PC) | Deps: T1.1; Gate D |
| T1.5 | Capture quality baseline | `[~]` | assistant | 2026-09-07 |  | `reference/corpus_t15.json` committed (6 prompts + ids); metric run blocked | Deps: T1.1 |
| T1.6 | Compute and verify memory budget | `[x]` | assistant | 2026-09-07 | 2026-09-07 | Budget FITS + proven: 14.5/0.39 GiB arenas allocated on B60 (T2.6) | Deps: T1.2, T0.2; Gate C |

Key pins (T1.1 done 2026-09-07):

- Model repo / revision: `Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; local `models/Qwen3.8-27B`, 18 shards, ~55.6 GB declared
- Tokenizer: `tokenizer.json` + `vocab.json` + `merges.txt` + `chat_template.jinja` local; checksums not yet pinned (small-file CRCs partially mismatched)
- Layers / hidden / intermediate / vocab: 64 text (48 linear + 16 full, interval 4) / 5120 / 17408 / 248320; vision depth 27 (deferred)
- Attention / heads / RoPE / norm / MLP: GQA 24Q/4KV head_dim 256; linear-attn 16Kx128/48Vx128 conv-4 FP32 state; M-RoPE [11,11,10] theta 1e7 partial 0.25; RMSNorm 1e-6; SiLU + swish output gate; untied
- MTP present? YES — 1 hidden layer, `mtp_use_dedicated_embeddings` false

## 5. Phase 2 — Model Container and Exporter

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T2.1 | Define `.binfer` container format | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `docs/binfer_spec.md` v1.0; 866-entry dir, reject rules = T2.2 hooks | Deps: T1.2 |
| T2.2 | Implement strict container loader | `[~]` | assistant | 2026-09-07 |  | Python ref: real file VALID, 6/6 negatives pass; C++ port pending | Deps: T2.1 |
| T2.3 | Implement deterministic INT4 quantizer | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `.binfer` 15,978,603,616 B, sha `5ef77c12…`, 505 tensors, worst err 0.1053 | Deps: T1.4, T2.1 |
| T2.4 | Tensor packing and layout (baseline) | `[x]` | assistant | 2026-09-07 | 2026-09-07 | layout-0 tensor round-trip + loader CRC; old 0.00171 MLP claim superseded | Deps: T2.3 |
| T2.5 | Conversion report | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `conversion_report.json` (866 tensors, worst-10, sha) | Deps: T2.3 |
| T2.6 | Load tensors into static L0 allocations | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `report_l0load.json`: 866/866 CRC-verified; H2D 1.02 / D2H 3.92 GB/s | Deps: T2.2, T1.6 |

## 6. Phase 3 — Kernel Microbenchmarks

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T3.1 | Bandwidth and dispatch baselines | `[x]` | assistant | 2026-09-07 | 2026-09-07 | `report_t31.json`: device roof ~437 GB/s; PCIe 13.8/11.9; launch 4.3 µs | Deps: T0.4 |
| T3.2 | Decode linear kernels (INT4 GEMV) | `[x]` | assistant | 2026-09-08 | 2026-09-08 | 6x4 shootout verified; dp4a-i8 best (34-67% roof); reduce/dp4a root-caused | Deps: T3.1, T2.4, T0.3 |
| T3.3 | Select INT4 layout/swizzle from data | `[x]` | assistant | 2026-09-08 | 2026-09-08 | layout-0 selected (limiter is ALU, not layout); UR4 + DPAS-broadcast rejected | Deps: T3.2 |
| T3.4 | Prefill linear kernels (tiled GEMM) | `[x]` | assistant | 2026-09-08 | 2026-09-08 | DPAS GEMM correct (1e-6); MT4 2x to 4.3 TFLOPS; DPAS-engagement queued for T6.1 | Deps: T3.1, T2.4 |
| T3.5 | RMSNorm and residual ops | `[x]` | assistant | 2026-09-08 | 2026-09-08 | plain+fused exact (1e-7); fused saves a pass; 456/387 GB/s | Deps: T3.1 |
| T3.6 | RoPE and KV writes | `[~]` | assistant | 2026-09-08 |  | RoPE max 2.38e-07; KV boundary/max-context + explicit fused-write kernel pending | Deps: T3.5, T1.2 |
| T3.7 | Attention kernels (prefill + decode) | `[~]` | assistant | 2026-09-08 |  | decode GQA e2e 1.28e-06; SSM decode proven (conv 1.75e-10, recurrent 2.98e-10); 48h vectorize + prefill-tiled + KV-quant pending | Deps: T3.6 |
| T3.8 | MLP and elementwise fusion | `[x]` | assistant | 2026-09-08 | 2026-09-08 | silu-mul fused 4x; 12-variant ablation: keep sym-g128, arbiter is e2e quality | Deps: T3.2, T3.5 |
| T3.9 | Sampling primitives (argmax) | `[x]` | assistant | 2026-09-08 | 2026-09-08 | 5/5 fixtures PASS; 4-byte host xfer; first-max ties | Deps: T3.1 |

## 7. Phase 4 — Correct End-to-End Runtime

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T4.1 | Assemble single transformer block | `[~]` | assistant | 2026-09-08 |  | CPU wiring exact (0.00) both types vs HF decoder; device INT4 wiring pending | Deps: T3.2, T3.5-T3.8 |
| T4.2 | Assemble full forward pass | `[x]` | assistant | 2026-09-08 | 2026-09-08 | device loop VALIDATED: top-4 == BF16 in order, per-layer 1-2% vs HF; coherent gen + EOS | Deps: T4.1, T2.6 |
| T4.3 | Integrate tokenizer | `[x]` | assistant | 2026-09-08 | 2026-09-08 | 14/14 exact vs HF; 33 specials + template parity | Deps: T1.1 |
| T4.4 | CLI generation loop | `[~]` | assistant | 2026-09-08 |  | loop proven vs reference (determinism, stopping, timings); native backend pending T4.2 | Deps: T4.2, T4.3, T3.9 |
| T4.5 | End-to-end correctness tests | `[x]` | assistant | 2026-09-08 | 2026-09-09 | 6/6 (`report_t45.json`): matched-template 126 == llama, edges, determinism, negatives | Deps: T4.4 |

## 8. Phase 5 — Static Scheduling and Memory Optimization

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T5.1 | Finalize activation liveness / reuse | `[x]` | assistant | 2026-09-09 | 2026-09-09 | `[t51]` footprint matches budget; fixed 16x KV over-alloc (8.0 GiB phantom at 4K); parity identical | Deps: T4.2 |
| T5.2 | Remove inference-time allocation | `[x]` | assistant | 2026-09-09 | 2026-09-09 | 260+/token file reads -> init preload; zero device/host allocs in loop; 0.69 s/tok steady-state | Deps: T5.1 |
| T5.3 | Record and reuse L0 command lists | `[x]` | assistant | 2026-09-09 | 2026-09-10 | decode_l0: 66 recorded lists, zero per-token construction; short parity identical, 64-layer profile + token-identical 59-token e2e | Deps: T5.2, T3.1 |
| T5.4 | Device-visible control buffers | `[x]` | assistant | 2026-09-10 | 2026-09-10 | decode_l0 loop's only host write is the 16 B control update; zero arg mutation | Deps: T5.3 |
| T5.5 | Move token selection to device | `[x]` | assistant | 2026-09-10 | 2026-09-10 | token-only+explicit selected (all configs 0.680-0.685 s/tok, identical tokens); steady-state host traffic 4 B/token | Deps: T3.9, T5.3 |
| T5.6 | Synchronization stress test | `[x]` | assistant | 2026-09-10 | 2026-09-10 | 5/5: determinism-x3, 61-tok long gen finite, template 126, CPU 80.7% (no spin), stable walls | Deps: T5.3 |

## 9. Phase 6 — Performance Tuning

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T6.1 | Profile-driven bottleneck optimization | `[x]` | assistant | 2026-09-10 | 2026-09-10 | 0.424 -> 0.068 s/token via SSM+norm opts; GEMV wall documented | Deps: T5.6 |
| T6.2 | Decode/prefill sweep tuning | `[x]` | assistant | 2026-09-10 | 2026-09-10 | linear flat 1.03 ms (SSM O(1)); attn 1.01->3.17 ms with T; 14.7 t/s at parity with llama baseline | Deps: T6.1 |
| T6.2 | Decode/prefill sweep tuning | `[ ]` |  | YYYY-MM-DD | YYYY-MM-DD |  | Deps: T6.1 |
| T6.3 | Justified fusions and KV quantization | `[~]` | assistant | 2026-09-09 |  | per-group-128 acts implemented+validated (2.2x err cut); MAXCTX/OOB root-caused under this task; fusions+KV-quant remain | Deps: T6.1, T1.5 |
| T6.4 | Reproducible benchmark report | `[x]` | assistant | 2026-09-10 | 2026-09-10 | `tools/bench/report_t64.json` + Metrics Log; staged targets judged (3/5) | Deps: T6.2 |

## 10. Phase 7 — Optional Features

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| T7.1 | Configurable sampling | `[x]` | assistant | 2026-09-10 | 2026-09-10 | 14/14 sampler tests + wired flags; seed-determinism proven, greedy default kept | Deps: T4.4 |
| T7.2 | MTP / speculative verification | `[ ]` |  | YYYY-MM-DD | YYYY-MM-DD |  | Deps: T1.3 (done — 1 layer), T4.2; gated on acceptance measurements |
| T7.3 | OpenAI-compatible HTTP daemon | `[x]` | assistant | 2026-09-10 | 2026-09-10 | real SSE streaming verified (join == full); determinism + error paths; single-flight | Deps: T4.4, T6.4 |
| T7.4 | Extended contexts / batching | `[~]` | assistant | 2026-09-10 |  | scoped: 64K contexts; batching + sliding-window dropped; blockers = BF16 KV + chunked prefill | Deps: T5.1, T6.2; separately scoped |

## 11. Cross-Cutting Tracks

| Task | Title | Status | Owner | Started | Finished | Evidence | Notes |
| ---- | ----- | ------ | ----- | ------- | -------- | -------- | ----- |
| X1 | Continuous verification suite | `[ ]` |  | YYYY-MM-DD | YYYY-MM-DD |  | Deps: T1.4 |
| X2 | Documentation and reproducibility | `[ ]` |  | YYYY-MM-DD | YYYY-MM-DD |  | Deps: T0.1 |

## 12. Metrics Log

Record benchmark snapshots here; link full reports under Evidence above (esp. T0.5, T3.1-T3.2, T6.4).

| Date | Task | Config (prompt/gen/context, sampling) | TTFT | Decode tok/s | p50/p95 ITL | Peak VRAM | Env (driver/toolchain/model rev) | Notes |
| ---- | ---- | ------------------------------------- | ---- | ------------ | ----------- | --------- | -------------------------------- | ----- |
| YYYY-MM-DD | _e.g. T0.5_ |  |  |  |  |  |  | baseline |
| 2026-09-10 | T6.4 | P64/gen64/ctx≤128, greedy EOS | ~50 s (44 load + ~6 prefill) | 10.6 sustained (14.7 short-ctx) | 94 / 103 ms | ~15.1 GiB computed | 26.05 / oneAPI 2026.1.1 / 1d4bf0f | `tools/bench/report_t64.json`; power telemetry N/A; 50% roof, parity w/ llama SYCL |

Quality tracking (T1.5, T6.3):

| Date | Corpus / Metric | Reference score | Current score | Delta | Notes |
| ---- | --------------- | --------------- | ------------- | ----- | ----- |
| YYYY-MM-DD |  |  |  |  |  |

## 13. Decisions

| Date | Task(s) | Decision | Rationale | Alternatives rejected |
| ---- | ------- | -------- | --------- | --------------------- |
| 2026-09-07 | T1.1 | Pin `Qwen/Qwen3.8-27B@1d4bf0f`, SafeTensors BF16 mirror as source | Headers validated 18/18; GGUF would cause double quantization | GGUF as quantizer input |
| 2026-09-07 | scope | Text-only v1, vision encoder deferred; 4K context cap | 24 GB budget; native 262K + 27 vision layers out of scope for v1 | Full multimodal v1 |
| 2026-09-07 | T3.7/T1.6 | Hybrid memory/kernel plan (16-layer KV + 48-layer SSM state) | Naive 64-layer KV formula overestimates this architecture | Full-transformer formula |
| 2026-09-09 | T6.3 | Per-group-128 (not per-tensor) INT8 activation scales in decode GEMVs | Halves zero-rate, +6.8dB SNR, 2.2x downstream error on real L0 vectors at negligible cost (136 scales/GEMV) | Per-tensor (kept nowhere); per-token (unneeded) |
| 2026-09-10 | T7.4 | Larger contexts at 64K; batching + sliding-window dropped | 64K fits only with BF16-or-better KV (21.1 vs 25.1 GiB); chunked prefill mandatory; batch-1 recorded loop kept | Sliding-window (quality price unneeded); batching (breaks fixed-address design) |
| 2026-09-10 | T5.4 | DecodeControl block in device memory, updated by 16 B immediate copy | Explicit wins decisively over shared (7.35+8.04 vs 48.30+76.67 us; shared pays migration + uncached reads); confirms plan.md no-zero-copy-assumption | Shared/coherent control writes |

## 14. Risks & Blockers

| Raised | Task(s) | Issue | Impact | Owner | Mitigation / Next step | Resolved |
| ------ | ------- | ----- | ------ | ----- | ---------------------- | -------- |
| 2026-09-07 | T1.1 | Small-file CRCs partially mismatched (chat_template, generation_config, tokenizer_config) | Low; weights headers valid, but tokenizer assets need checksum pinning |  | Re-verify line endings/encoding; pin checksums in manifest | open |
| 2026-09-07 | T0.2/T0.3/T3.1 | No Arc Pro B60 access on prior PC | Gates B/C alloc tests + all perf work blocked |  | Resolved by platform move — B60 present, T0.x unblocked | 2026-09-07 |
| 2026-09-07 | T3.7 | 48 linear-attention layers need new SSM kernels unscoped in original plan | Medium; kernel roadmap expanded, estimates uncertain |  | Prototype CPU reference first; benchmark on B60 before committing layout | open |

## 15. Changelog

Newest first. One line per meaningful status change.

| Date | Change |
| ---- | ------ |
| 2026-09-11 | OPT3 vectorize DONE: ChunkSsmRecur 64x, AttnCore 6-10x, ChunkAttn 15x; extractor esimd-shadow trap + BF16-convert trap caught by harnesses. 33/33 CTest, stress 5/5 (cpu-wait re-instrumented, TOP5=0 real). 126 survives; short flips at known near-tie. Next: KV-blocking + list fusion. |
| 2026-09-11 | T1.5 pilot PASS: streamed-CPU BF16 greedy vs loop INT4 agree top-1 8/8, top-5 sets 4-5/5 ("To solve 84 * "); llama cross-check correct toward 126. First end-to-end quality number; pipeline box-native (~25 min/prompt), batch of 5 left. |
| 2026-09-11 | PLATFORM RULE: no 64 GB+ host coming — all "needs bigger machine" framings dropped. T1.4/T1.5 re-scoped to box-native (streamed-CPU BF16 greedy = truth, loop = system, llama SYCL = cross-check); T6.3/T7.2 gates and T7.4 quality reframed the same way. |
| 2026-09-11 | T7.4 64K validated: real loop at MAXCTX=65544 token-identical to small sizing; device-fill zero-init; RoPE-at-max vs HF, far-slot KV bitwise, AttnCore correct to 4K. Left: production hardening (vectorize) + quality vs box-native T1.5 corpus. |
| 2026-09-10 | T7.4 chunk-layer done: full linear layer as one ~330-launch recorded list over 2 chunks, worst-rel 1.77e-04 (fp16 path), continuity + reset-determinism proven; stride-aliasing + group-size bugs caught by stage forensics. Left: 64K validation runs. |
| 2026-09-10 | T7.4 64K alloc proven: full footprint at MAXCTX=65544 with BF16 KV allocates on the real heap — 19.08/22.71 GiB, 3.63 GiB margin, FIT-PASS. Left: full-layer chunk orchestration + 64K validation runs. |
| 2026-09-10 | T7.4 chunk-SSM done: persistent conv/SSM state across 32-token chunk replays, worst-rel 8.11e-07, continuity proven + reset-deterministic; scalar ~211 ms/chunk queued for vectorization. Left: 64K alloc/validation. |
| 2026-09-10 | T7.4 chunk-attention retired: causal 256-over-128 chunk in recorded list, worst-rel 1.03e-06, deterministic; scalar ~71 ms needs ESIMD before production. Debug: WI index 6144-vs-24. Left: SSM chunk orchestration + 64K validation. |
| 2026-09-10 | T7.4 chunk-GEMM risk retired: tiled INT4 DPAS GEMM under raw L0 (256x5120x17408), 2.54 TFLOPS, worst-rel 1e-06, deterministic; ~150x over loop-decode on linear ops. Remaining for 64K: chunked attention + SSM orchestration, then alloc/validation. |
| 2026-09-10 | T7.4 BF16 KV done: RNE-quantize on append, cache bytes halved; harnesses green, layer3real 2.18e-05, short prompt [369,279,248046]; 64K fit unblocked (pending chunked prefill). Also fixed the Wts orphan in layer3real. |
| 2026-09-10 | T7.4 scoped (user): larger contexts at 64K; batching dropped (batch-1 sufficient, breaks fixed-address design); sliding-window dropped (quality price unneeded — 64K fits). 64K math: BF16 KV ~21.1 GiB fits, FP32 KV ~25.1 GiB does not → BF16 KV + chunked prefill are the owned blockers. |
| 2026-09-10 | T7.2 acceptance measured: alpha = 0.585 (n=41, streamed CPU truth) -> ~1.5x projected, interval above break-even; CONDITIONAL PASS but implementation DEFERRED (0.6 bar missed by one sample; T1.5 quality gate blocked). Reopen on quality baseline + confirmatory sample. |
| 2026-09-10 | T7.2 spike: MTP-1 dataflow recovered (vLLM source; HF has none), 15 tensors already INT4 in .binfer, draft ~4 ms (c=0.06), break-even alpha > 0.06 / prize 1.6x at 0.7; gate = measured alpha >= 0.6 + quality. Acceptance run next. |
| 2026-09-10 | T7.3 done: OpenAI-compatible daemon (`tools/http/ainfer_http.py`, stdlib only) — full JSON + real SSE streaming verified byte-equal, determinism + 400/404 paths; single-flight lock, per-request spawn TTFT documented. Phase 7 at 2/4. |
| 2026-09-10 | T7.1 done: sampler wired into `decode_l0` (temp/top-k/top-p/seed/rep-penalty flags); seed-123 twice identical, greedy default preserved, near-tie explored correctly. Phase 7 at 1/4. |
| 2026-09-10 | T7.1 sampler core done: host temp/top-k/top-p + seeded multinomial, 12/12 statistical contract tests ALL-OK (chi-square at temp 1.0/0.5, supports, limits, determinism); greedy matches T3.9 fixtures. Repetition penalty + loop wiring remain. |
| 2026-09-10 | Verification Stamp 9: Phase 5/6 closure audit — 25/25 CTest (472.87 s), e2e 6/6, L0 matched completion with 126 on current binary (78 tokens, trajectory differs by documented flip physics). Orphan-harness root-caused pre-stamp. Does not certify T1.5/T6.3 quality, prefill, or Phase 7. |
| 2026-09-10 | Audit find: AttnCore Wts refactor orphaned `layerattn_replay` (arg 7 never set → NULL pointer → deterministic DEVICE_LOST at first execute). Fixed + green. Rule: kernel-signature changes require a repo-wide call-site audit; unset L0 args fail silently at setArg time and explode at execute time. |
| 2026-09-10 | T6.4 done: reproducible benchmark (`tools/bench/report_t64.json` + Metrics Log) — TTFT/load/prefill split, p50/p95 ITL (94/103 ms), 14.7 t/s at parity with llama baseline, staged targets judged 3/5 (roofline + stretch missed on the documented ALU wall). Phase 6 at 3/4. |
| 2026-09-10 | T6.1/T6.2 done: sweep 4/4 (P=1/16/64/256) — linear flat 1.03 ms, attn scales with T, 14.7 t/s at parity with llama.cpp baseline; Phase 6 at 2/4. Remaining: T6.3 fusions/KV-quant (needs T1.5) + T6.4 report. |
| 2026-09-10 | T6.1 OPT 2 (norm parallelize): 256-WI SLM double tree, list 524 -> 25 us (21x), steady ~0.068 s/token (~10x under SYCL); kept at zero trajectory cost. Next: GEMV/tail efficiency. |
| 2026-09-10 | T6.1 OPT 1 (SSM vectorize): 5987 -> 237 us/step (25x), steady state ~0.14 s/token; kept with documented flip cost (short trajectory moved at step 4, top-5 order kept, no-cliff dump profile). Next: norm parallelization. |
| 2026-09-10 | T6.1 baseline: adopted loop profiles at 0.424 s/token (1.6x under SYCL 0.68 s); rank SSM 68% / norms 25% / GEMV 6% / dispatch 1% (`tools/t61/report_profile.json`, `AINFER_PROFILE=1`). Next: SSM vectorization, then norm parallelization. |
| 2026-09-10 | T5.6 done + Phase 5 gate CLOSED (6/6): stress suite 5/5 on the adopted loop (determinism-x3, 61-token finite long gen, template 126, CPU 80.7% no spin, stable walls). Next: T6.1 profiling. |
| 2026-09-10 | T5.3/T5.4 DONE: `decode_l0` full 64-layer recorded loop adopted — 66 lists, zero per-token construction/allocation; short parity identical, 64-layer profile (L0-2 bitwise, flip growth after, two-prompt experiment rules out systematic bugs), 59-token e2e token-identical incl. 126. Phase 5 at 5/6 (T5.6 stress remains). |
| 2026-09-10 | T5.3 real-weight attn LAYER3REAL-OK: recorded 24-launch layer-3 list from real arenas (slot 2 in, slot 3 ref), worst-rel 1.44e-09, deterministic. Both classes proven; remaining is 64x replication + loop plumbing (+ in-loop embed). |
| 2026-09-10 | T5.3 real-weight proof LAYER0REAL-OK: recorded 28-launch linear-layer-0 list from real arenas is BITWISE identical (0.00e+00) to the certified SYCL dump, reset-deterministic. Adoption remaining is 64x replication + loop plumbing. |
| 2026-09-10 | T5.3 full-attn list LAYERATTN-OK: one attention layer as a single 24-launch recorded list (control-driven pos/T, persistent KV), step-0 parity 3.16e-07, reset-deterministic bitwise. Both layer classes compose; 64-layer decode adoption remains. |
| 2026-09-10 | T5.3 layer adoption LAYERLIN-OK: full linear layer as one 28-launch recorded list, step-0 parity 2.54e-07, reset-deterministic bitwise 4x2; INT8 flip avalanche (~5x/step) scopes host-ref to step 0, multi-step fidelity via determinism. Full-attn list + 64-layer adoption remain. |
| 2026-09-10 | T5.3 port set COMPLETE: RoPE/KV ROPE-OK (16.7 us, worst-rel 1.02e-07) + argmax ARGMAX-OK (20.1 us, integer-exact token incl. ties). Every decode kernel class now records/replays under raw L0; only loop adoption remains. |
| 2026-09-10 | T5.3 SSM port SSM-OK: stateful conv-k4+silu + 48-head FP32 recurrence in one recorded list, 8-step sequence at ~6 ms/step, worst-rel 6.63e-07, zero-reset rerun bitwise identical. Only RoPE/KV + argmax ports remain before loop adoption. |
| 2026-09-10 | Verification Stamp 8: scoped post-Stamp-7 audit accepted cmdlist 6/6 and native e2e 6/6; full CTest 20/20 recorded as supplementary confidence, not required incremental evidence. Raw-L0 ports now cover silu/norm/res, INT4 GEMV, GQA+gate; SSM/conv, RoPE/KV, argmax and decode adoption remain. |
| 2026-09-10 | T5.3 attention port ATTN-OK: GQA+gate core (24Q/4KV) in recorded list, 939 us, worst-rel 8.06e-07, deterministic 20/20; variable-T via control block, TMAX clamp (also fixed a dead-arg setArg failure). SSM/RoPE/argmax ports remain. |
| 2026-09-10 | T5.3 GEMV port GEMV-OK: decode-exact dp4a INT4 GEMV (17408x5120) in recorded list, 283 us (~37% roof), worst-rel 2.57e-07, deterministic 20/20. Fixed build flow (post-link esimd lowering before llvm-spirv) in `extract_spv.py`; one module per entry. Attn/SSM ports remain. |
| 2026-09-10 | T5.3 layer-class prototype LIST-OK: 3-kernel closed list (norm->silu->res, control-driven) replays at 524 us, bitwise-deterministic 50/50; ~508 us is the scalar fp64 norm (T6.1 work). GEMV/attn SPIR-V port remains for full-loop recording. |
| 2026-09-10 | T5.5 done: token-only return (`AINFER_TOP5=0`) + shared-vs-explicit handoff measured — all 4 configs token-identical at 0.680-0.685 s/token; explicit kept (tie goes to simpler + T5.4-consistent). Phase 5 at 3/6. |
| 2026-09-10 | T5.3 enabler: xscales device-side (KMax kernel) — ~257 activation D2H + 257 scale H2D round trips/token eliminated, parity bit-exact ([369,279,248046], s3 top5 identical), 0.687 -> 0.677 s/token; decode ctest passes. Loop host traffic now only token + logits diagnostics. |
| 2026-09-10 | T5.4 harness CONTROL-OK: `ControlAdd` recorded once with fixed addresses, 50 non-monotonic-position replays all bitwise exact, no arg mutation/rebuild. Explicit 16 B copy beats shared (7.35+8.04 vs 48.30+76.67 us); ctest `control_replay` passes. Decode-loop adoption remains (needs raw-L0 migration). |
| 2026-09-10 | Verification Stamp 7: audited post-Stamp-6 work; rebuilt B60 and CTest 16/16 passed (547.95 s), native e2e 6/6 passed. Corrected audit findings before stamp: all release-loop host scratch preallocated; T5.1 report fixed FP32 KV@4K to 512 MiB and phantom 16x allocation to 8.0 GiB. Next: T5.4 control buffer, then full replay. |
| 2026-09-09 | T5.1/T5.2 done: small-weight preload arenas kill ~260/token file reads + staging copies (init-only now); host vectors hoisted, ids reserved; audit shows zero inference-time allocs. Fixed 16x KV over-allocation (phantom 8.0 GiB at 4K). Parity bit-identical; steady-state 0.69 s/tok (sync-submit bound, queued T5.3/T6.1). |
| 2026-09-09 | T6.3 diagnostic: per-group-128 INT8 acts halves dG17 zero-rate (16.2->8.3%), +6.8dB SNR, 2.2x down_proj error cut (`tools/t63/diag_actquant.cpp`, `report_actquant.json`, real L0 weights); implemented in decode loop (dSq scales, 11 sites), short parity bit-identical. |
| 2026-09-09 | ROOT CAUSE (was T4.5 collapse): MAXCTX sized from positional G=1 before --ids=/--max-new= overrides -> KV/RoPE OOB writes from pos 79; NaN born L7 masked to zero-logits by NaN->0 in INT8 path. Fix: finalize P/G before MAXCTX + hard pos guard. Same prompt now reasons to 126 == llama; e2e suite 6/6, T4.5 done. dXmax 100-250 spikes are benign operating range. |
| 2026-09-08 | T4.5 mechanics 5/6: determinism byte-identical, one-token/max-64 clean, truncated+garbage rejected rc=2 (added span bounds check + bad-magic message). Greedy completion degenerates (repetition collapse, NaN/zero poison past step ~79); prime suspect per-tensor INT8 acts vs outliers (dXmax ~400) -> T6.3. |
| 2026-09-08 | ✅ Verification Stamp 5: rebuilt B60 preset; CTest 14/14 including arena-backed block and full decode-loop targets. Execution/stage arithmetic accepted; token equivalence explicitly remains unverified pending a matched INT8-activation CPU oracle. |
| 2026-09-08 | T4.2-device decode loop: full 64-layer loop with KV+SSM caches + device argmax runs (generated [271,220,220]); fixed 3 real bugs (q_proj buffer overflow 2048 vs 12288, linear state-index collision, broken argmax reduction). Token-exact validation pending: device (INT4+INT8 acts) differs from CPU (INT4+fp32 acts) — logit std matches (2.18 vs 2.33), mean shifted, top-5 disjoint; need apples-to-apples int8 CPU reference (T4.5/T6.3). |
| 2026-09-08 | T4.2-device stage C: full L0 linear decode block on device (lin-mid 2.9e-08, lin-out 6.7e-09); fixed 3x-undersized conv-state buffer (heap overflow); linear-half quant-delta 3.5e-03 vs 0.53 for attention half. |
| 2026-09-08 | T4.2-device stage B: full L3 attention-half on device, 9/9 kernel rows PASS at ~2e-07 vs host-INT4; quant-delta 0.53 flagged for T6.3 (sym-g128 + per-tensor int8 may be too lossy). |
| 2026-09-08 | T4.2-device stage A: arena-backed real-weight dp4a GEMV passes (L3 q_proj 221.9 GB/s, L0 gate_proj 181.2 GB/s, maxrel ~2e-07); arena-to-kernel path proven. Full-block wiring + decode loop remain. |
| 2026-09-08 | ✅ Verification Stamp 4: current B60 preset rebuilt; complete hardware suite 10/10 passed; CPU T4.2 S0/S1/S2 evidence, T4.3 14/14, T4.4 3/3, T3.9 5/5 rechecked. S3 template-mismatch comparison remains excluded from quality claims. |
| 2026-09-08 | T4.2 CPU prefill proven (S0 exact, S1 sane top5, S2 INT4 4/4 agree); S3 mismatch traced to template paths, not weights — matched-template completion needs decode loop. |
| 2026-09-08 | T4.3 done (14/14 exact vs HF incl. 33 specials + template parity); T4.4 loop proven vs llama SYCL reference (determinism, stopping, timings; native backend pending T4.2). |
| 2026-09-08 | T3.9 done: device argmax 5/5 fixtures PASS (4-byte host transfer, first-max ties); Phase 3 at 7/9. |
| 2026-09-08 | T4.1 CPU wiring exact (0.00) both block types vs HF decoder; mask-convention bug found+fixed (HF eager mask=None is non-causal); old L3 block_out superseded by block_T41.json for prefill. |
| 2026-09-08 | T3.7 SSM decode proven: HF `lin_block_L0.json` (chunk-vs-recurrent mean 5e-05); conv-k4 exact 1.75e-10, recurrent head-0 max 2.98e-10; naive 48-head 16ms flagged for T6.1 vectorization. T4.1 unblocked for both block types. |
| 2026-09-08 | ✅ Verification Stamp 3: rebuilt B60 preset, CTest 8/8; audited T3.2-T3.8 artifacts; removed duplicate T3.3 row; downgraded T3.6/T3.8 to partial; fixed `mlpcheck` to use Qwen (1+w) RMSNorm. |
| 2026-09-08 | Audit correction: old MLP diff 0.00171 was invalid (wrong `w` norm fixture). Correct (1+w) INT4 check: max 1.62375, mean 0.063877 (~6.9% mean-relative); T3.8 remains partial. |
| 2026-09-08 | T3.8 closed: 12-variant ablation shows uniform error, no cheap exception (best 0.056 single, asym 0.058); policy stands (sym-g128, embed BF16); e2e quality is the arbiter. |
| 2026-09-08 | T3.2 done: 5-path x 4-shape GEMV shootout, all verified; vec-i8 best (26% roof), scalar wins one shape (29%), DPAS-broadcast rejected; toolchain findings logged (reduce/dp4a/group-reduce). |
| 2026-09-08 | T3.3 done: dp4a revived (root cause was reduce miscompile, not the intrinsic) — dp4a-i8 reaches 34-67% roof; UR4 rejected; layout-0 selected, IDs 1-15 reserved for prefill. |
| 2026-09-08 | T3.6/T3.8 done, T3.7 partial: HF modeling resolved Q/gate/RoPE/RMSNorm-(1+w) (mlp_layer0 was wrong, fixed); decode token e2e 1.28e-06; silu-mul fused 4x. T4.1 unblocked for full-attention blocks. |
| 2026-09-08 | T3.5/T3.4 done: RMSNorm exact (1e-7), fused saves a pass; DPAS prefill GEMM correct (1e-6), MT4 2x to 4.3 TFLOPS (2-4x over decode loop); DPAS-engagement shortfall queued for T6.1. |
| 2026-09-07 | ✅ Verification Stamp 2: rebuilt B60 preset; CTest 3/3; probe/ESIMD/XMX/timestamps pass; C++ loader readback 866/866; D2D 437.07 GB/s; llama.cpp baseline rerun 183.97 pp / 14.81 tg. Corrected stale T2.2 C++-loader note. |
| 2026-09-07 | T3.1/T0.5 done: device roof ~437 GB/s (96% of 456), PCIe 13.8/11.9, launch 4.3 µs; llama.cpp SYCL baseline pp512 183.87 / tg128 14.86 t/s. Phase 0 complete (5/5). |
| 2026-09-07 | T2.6/T1.6 done: first B60 execution — 2 static arenas, full 15 GB readback 866/866 CRC-verified; C++ loader written (T2.2 port advanced). |
| 2026-09-07 | T0.3/T0.4 done: ESIMD SG16/32 + DPAS int8/fp16/bf16 exact on B60 (no native INT4 XMX → unpack-then-DPAS design); `CMakePresets.json` b60/host, `ctest` 1/1 pass with SYCL+L0 timestamps. |
| 2026-09-07 | Platform move to Ubuntu + Arc Pro B60 (`8086:e211`); venv at `~/.venvs/ainfer`; T2.3/T2.4/T2.5 done (`.binfer` 15,978,603,616 B, sha `5ef77c12…`, VALID, mlpcheck diff 0.00171); T2.2 Python ref done (6/6 negatives). |
| 2026-09-07 | T2.1 done: `docs/binfer_spec.md` v1.0 (866-entry dir, INT4-sym-g128 policy, file ≈ 14.9 GiB). |
| 2026-09-07 | T1.4/T1.5 partial: `reference/` with operator fixtures, real-weight MLP-L0, 1199-tensor stats, tokenizer finding (added-token registration needed), staged prompts/corpus; full-model runs blocked. |
| 2026-09-07 | T1.6 extended: 32K context fits (scenario A + KV BF16 ≈ 20.4 GB, margin ≈ 3.6 GB); scaling table added to `memory_budget.json`. |
| 2026-09-07 | T1.6 calc done: `memory_budget.json` v1.0 — scenario A total ~17.28 GiB, margin ~6.7 GiB, FITS 24 GB; L0 validation pending B60. |
| 2026-09-07 | T1.2 done: `models/Qwen3.8-27B/manifest.json` v1.0 (1199 tensors, full index coverage, tensor bytes = index total 51.75 GiB). |
| 2026-09-07 | Pinned Qwen3.8-27B@1d4bf0f (T1.1 done, T1.3 MTP confirmed, T1.2 in progress); plan/tasks updated for hybrid arch, text-only v1, hybrid memory formula; GGUF demoted to baseline-only. |
| YYYY-MM-DD | Initialized `progress.md` from `tasks.md`; all tasks pending. |
