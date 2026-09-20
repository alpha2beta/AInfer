# AInfer (258v branch) — Qwen3.5-MoE inference on Intel Core Ultra 7 258V

Port of the AInfer inference runtime to Lunar Lake: integrated Arc 140V
Xe2 GPU, 32 GB unified LPDDR5X-8533, CachyOS. Target model
`symrex/Tiel-Coder-35B-A3B-Genesis-Hermes` (Qwen3.5-MoE sparse hybrid:
40 layers = 30 DeltaNet linear attention + 10 full attention, 256 routed
experts / 8 active, ~35B total / ~3B active params).

Related branches: `B60` (Qwen3.8-27B dense on Intel Arc Pro B60),
`main` (initial snapshot).

## Measured status (see `optimization.md` for the full record)

| Path | Result |
|---|---|
| Decode, sustained | **34.88 tok/s** (+18.9% vs llama.cpp Vulkan 29.33; 3.49× vs 8-thread CPU) |
| Prefill, chunked batched + DPAS tiling | **up to 352 tok/s** @P=441 (from 39.3; peak 89.3 @P=32 early) |
| Warm TTFT (P=21) | **237 ms** (from 534 ms, −55.6%) |
| Quality | Gate M4 7/7 bit-exact golden sequence; 0 KB RSS growth; deterministic reset |

Key techniques: unified single-list recording, batched 8-expert MoE dispatch
(3 kernels replace 960 launches), fused LM-head GEMV+argmax, chunked
batched prefill GEMM, DPAS systolic tiling (M_tile=32; v5/M64 and no-SLM
reverted as negative results), attention butterfly + recurrence tuning.
OpenVINO GenAI on the same class of hardware: ~35 tok/s decode parity,
~525 tok/s prefill claim under investigation.

## Layout

- `tools/kernels_258v/` — OpenCL kernels (`all_kernels.cl`: MoE, DeltaNet, attention, GEMM) + per-kernel reports + A/B bench harnesses
- `tools/bench_258v/` — benchmarks (`bench_258v`, `bench_prefill*`, scaling sweeps, llama-comparison runner)
- `tools/decode/runtime_258v.*` — resident runtime: static arenas (18.03 GiB committed), pre-recorded command lists; `test_runtime_258v.cpp` (Gate M4)
- `tools/http/server_258v.py` — persistent resident daemon (SSE `/v1/chat/completions`, port 8088)
- `prefill_optimization_review.md` — chunked-prefill technical review; `target_machine_identity.json` / `target_model_identity.json` — pinned environment
- `optimization.md` — full optimization record (Pillars 1–10 + B70 cookbook transfer notes)
- `optimization_258v_MoE.md` — earlier snapshot of the same program
- `B60_*.md` — frozen copies of the B60 branch docs for cross-reference

## Environment notes

CachyOS rolling, Intel GPU stack for Lunar Lake (Level Zero / NEO).
Unified memory: arenas live in shared LPDDR5X (13.97 GiB headroom at
18.03 committed). Power matters here (17W PL1 / 37W PL2) — report PL1
with every benchmark. 5.58-min steady-state run settles at 55.8 °C.

## Docs

- `optimization.md` — current performance truth (read first)
- `AInfer_258V_migration_plan.md`, `migration_scope.md` — port scope
- `STATUS.md`, `plan.md`, `tasks.md`, `progress.md` — branch-local status
  (note: several `B60_*` files are frozen B60 copies, not live docs)
