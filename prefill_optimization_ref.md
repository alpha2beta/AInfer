# AInfer Prefill Optimization Reference

Standalone summary of the prefill optimization campaign on Intel Core Ultra 7 258V
(Arc 140V Xe2 iGPU, 32 GB LPDDR5X) for `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes`
(Qwen3.5-MoE, 40 layers = 30 DeltaNet + 10 full-attention, 256 experts / 8 active,
INT4-g128 + BF16 scales). Full detail lives in `optimization.md` Pillars 6–10;
task traceability in `tasks.md` T5.2/T7.2/T7.5; history in `progress.md`.

**Headline:** prefill at matched P=441 went **294.80 → 351.62 tok/s (+19.3%)**;
peak **362.48 tok/s at P=256**. Every step kept Gate M4 7/7 bit-exact
(golden tokens `[148431, 62497, 148287, 198, ...]`) and zero heap allocations.

Reproduce: `g++ -O3 -std=c++17 tools/decode/runtime_258v.cpp
tools/bench_258v/bench_prefill.cpp -Itools/l0probe/include
-Ltools/toolchain/sysroot/usr/lib -lze_loader -o tools/bench_258v/bench_prefill`
then run with `LD_LIBRARY_PATH=tools/toolchain/sysroot/usr/lib`.
Reports land in `tools/bench_258v/report_prefill_scaling.json`.
Kernel variants are env-gated (`AINFER_ATTN_V1`, `AINFER_RECR_V1`,
`AINFER_GEMM_V1/V2` restore older kernels; current defaults are the winners).

---

## 1. Scoreboard (tok/s, higher is better)

| Step | P=256 | P=441 | P=512 | P=1024 | P=2048 | Δ vs prev @P=441 |
|---|---|---|---|---|---|---|
| Sequential baseline | — | — | — | — | — | — |
| Pillar 6: chunked batched GEMM (18 kernels, $B \le 32$) | 89.31 | — | — | — | — | — |
| + DPAS systolic GEMM (Pillar 7) | 89.73 | — | — | — | — | — |
| + scaling harness (`bench_prefill`, P=8…256) | 299.04 | 294.80 | — | — | — | — |
| + attention v2, recurrence v2 (Pillar 8) | 318.31 | — | — | — | — | — |
| + chunks-512, GEMM v2 K-slices (Pillar 9) | 335.52 | 328.29 | 328.39 | 280.58 | 207.78 | +11.4% cumul. |
| + GEMM v4 M_tile=32 (Pillar 10) | **362.48** | **351.62** | 351.94 | 293.89 | 222.20 | +19.3% cumul. |

The curve peaks at P≈256 and declines after (causal KV growth per chunk).
External claim to beat: 525 tok/s (OpenVINO, P=441 probe) — current gap 1.49x.

## 2. What worked (keep)

- **Chunked batched prefill (Pillar 6).** 18 dedicated batched kernels +
  cached per-B Level Zero command lists. 39.32 → 78.92 tok/s (+101%).
  Critical fix: `gate_prep_batch` was missing the outer `exp(gate)` decay —
  corrupted DeltaNet state across recurrence; restoring it recovered bit-exactness.
- **DPAS systolic GEMM (Pillar 7).** INT4 weights unpacked on the fly to FP16,
  `intel_sub_group_f16_f16_matrix_mad_k16`, M_tile=16/B_tile=8/K_step=16.
  1.90x on the dominant DeltaNet projection (M=8192, K=2048).
- **Attention v2 (Pillar 8).** Replaced the 256-wide SLM tree reduction
  (~10 barriers/KV-position/query) with in-register subgroup butterfly +
  one SLM exchange (0.25 barriers/position). 1.36–2.65x on the kernel (A/B
  harness `tools/kernels_258v/bench_attn_prefill.cpp`, 1.68e-07 vs CPU).
  **Lesson:** IGC silently compiled the light kernel as SIMD32, breaking the
  subgroup-16 shuffle (constant 0.17 bias) — fixed with
  `__attribute__((intel_reqd_sub_group_size(16)))`. Always A/B new kernels;
  never trust the default SIMD width.
- **Chunks-512 (Pillar 9).** `MAX_PREFILL_CHUNK` 256→512, workspace 128→256 MiB
  (overflow guard verified silent). +4–6% from better MoE grouping
  (~14 vs ~8 slots/expert) and fewer chunk boundaries. M4 bit-exact.
- **GEMM v2 doubled K-slices (Pillar 9).** 2 K-slices per iteration, 8 back-to-back
  DPAS per barrier (4KB double-buffered SLM). +6.9% @P=441.
- **GEMM v4 M_tile=32 (Pillar 10).** Each lane handles 2 rows; `a_mat` SLM-gather
  built once and shared across both rows' DPAS (16 DPAS per slice-pair for one
  setup). +7.1% @P=441. `gemm_rows_per_group_` halves m-groups (all 3
  `append_gemm` sites checked; MTP verify path uses `k_gemv_m2_`, unaffected).

## 3. What failed (do not retry without new evidence)

- **GEMM v3 without SLM staging: −38%.** Direct cached X loads + zero barriers
  exposed load latency — the SLM double-buffer is load-bearing latency hiding,
  not overhead. Register-prefetch variant: −50% (spill). **Rule: never remove
  the staging; attack setup density instead.**
- **GEMM v5 M_tile=64: slower in every run** (best 272 vs v4's worst 328 @P=441),
  asm-confirmed zero spills — I-cache/occupancy suspected. **M_tile=32 stands;
  tile widening is exhausted.**
- **QKV+ZAB fusion: ~1% prize, deprioritized.** Launch + X-reread savings are
  noise next to DPAS throughput. Measured before deciding — do the same for
  any future fusion proposal.
- **Integer DPAS via OpenCL C: infeasible.** `s8_s8`/`u8_s8` builtins don't exist
  in this IGC; the only integer form (`u8_u8_matrix_mad_k32`) is 16x narrower
  per lane (64 vs 1024 MACs), unsigned-only, and still needs nibble unpacking.
  Reaching `s4` ops needs ESIMD/inline-asm (major project, uncertain acceptance,
  breaks bit-exact prefill). Path closed unless explicitly approved.
- **Recurrence barrier-cutting: ~1.5% end-to-end.** Double-buffered 4-step
  batching + 4-way accumulators moved the kernel only 5.27→4.9 ms — the loop is
  FMA-chain/occupancy bound, not barrier bound (proven by identical timing with
  `AINFER_RECR_V1=1`). A parallel scan is the only large win left here
  (~10–13% prize, high complexity/risk).

## 4. Where time goes now (B=512 probe, per-layer)

DeltaNet layer (~37 ms): QKV 6.8 · ZAB 4.1 · recurrence 9.7 · out-proj 3.1 ·
MoE router 2.5 + GateUp 5.5 + Down 2.5 + tail. Full-attn layer (~50 ms):
QKV 6.8 · attention 27.2 · out-proj 3.1 · MoE ~10. GEMMs ≈ 60–86% of layer time
at ~1.3 TFLOPS (single-digit % of XMX peak) — DPAS throughput is the ceiling.

## 5. OpenVINO 525 comparison (status)

Matched-P=441 measurement: ours 351.62 vs claimed 525 (1.49x gap) with decode
at parity (35.0 vs 35.54) → compute-kernel efficiency gap, not bandwidth.
Confounders: Windows 101.8826 vs Linux compute-runtime, official OV INT4 vs
INT4-g128, single-table unverified source. Remaining unbuilt idea with a real
prize: cross-query KV reuse (FlashAttention-style blocking, each K/V block read
once per query-block) — helps mainly long-P. ESIMD integer path parked above.

## 6. Methodology notes (earned the hard way)

- A/B every kernel with an env-gated fallback before making it default; M4 7/7
  bit-exact is the acceptance bar for prefill changes (token-level equivalence
  only if bit-exactness is ever deliberately surrendered).
- ±7% run-to-run variance observed on a long-uptime shared box — repeat runs
  before concluding deltas <10%; one 268-tok/s outlier nearly misled a decision.
- Microbench absolutism misleads: profiler attention figures were inflated ~3x
  by stale control-buffer position (fixed by pinning); always cross-validate
  harness vs profiler (6.665 vs 6.687 ms agreement is the standard).
- Rebuild pipeline: `ocloc compile -file all_kernels.cl -device lnl` from
  `tools/toolchain/sysroot` is byte-deterministic — verify with `cmp` before
  benchmarking a new kernel. SPV files are gitignored build artifacts.
- Unique anchors when inserting kernels (`__kernel void <name>(`), after one
  duplicated-insertion build break from a repeated section header.
