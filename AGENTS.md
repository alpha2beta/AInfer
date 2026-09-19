# AGENTS.md — AInfer

## What this repo is

Custom NInfer-style inference runtime for pinned model
**`symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`** (dequantized SafeTensors
release of the `LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF` fine-tune; base
architecture `Qwen3.5-MoE` / `model_type: qwen3_5_moe` — **not** an official `Qwen/...`
repo release, despite earlier planning docs assuming one)
(sparse MoE hybrid: 40 text layers = 30 DeltaNet-style linear-attention + 10 full-attention,
256 routed experts / 8 active per token, batch size 1, text-only — the checkpoint also ships
a vision tower which is excluded from the v1 `.binfer` export). Target hardware is
**Intel Core Ultra 7 258V** (integrated Arc 140V GPU, Xe2, 32 GB unified LPDDR5X memory)
running under **CachyOS**.

Branch `258v` is the active migration track. The prior production implementation for
Intel Arc Pro B60 (`8086:e211`, Qwen3.8-27B) is archived under `B60_*` (`B60_plan.md`,
`B60_tasks.md`, `B60_progress.md`, `B60_STATUS.md`).

## Doc trio — update consistently

- `plan.md` = design. `tasks.md` = scope; task IDs (`T0.1`…`X2`) are stable, never renumber.
  `progress.md` = live status (mirrors tasks phase tables + changelog).
  `STATUS.md` = single-source current release truth.
- After finishing work: mark status in `tasks.md`, update `progress.md` dashboard/rows/changelog.

## Key domain facts (do not re-derive)

- KV cache covers **10 full layers only** (out of 40 layers total); 30 linear layers carry FP32 SSM state (~60 MiB).
  Naive 40-layer KV math overestimates ~4×.
- All 35B model parameters must reside in 32 GB unified system memory in INT4 (~17.5 GB weights + scales).
  Measured headroom on 32 GB RAM is 8+ GB for OS and workspaces (see `memory_feasibility_estimate.md`).
- Primary runtime path is **single-process in-memory** prefill $\to$ decode. Disk-based cache handoff is diagnostic-only.
- Target OS is CachyOS (rolling release): toolchain packages must be pinned with local pacman cache snapshots.
- B60 baseline documents and reports are preserved with `B60_` prefix for regression benchmarking.

## Environment (Ubuntu + Arc Pro B60)

- Repo lives at `/mnt/usb/AInfer` on a USB mount: **no symlinks allowed** (venv must live
  on ext4, e.g. `~/.venvs/ainfer`), all files show executable bits (ignore that).
- Python: project venv at `~/.venvs/ainfer/bin/python` (uv-managed; torch 2.14 CPU,
  safetensors, numpy, tokenizers). System `python3` (3.14) has no pip and no sudo —
  do not fight it, use the venv. No `py` launcher here.
- GPU: B60 at `03:00.0` (`8086:e211`); oneAPI 2026.1 under `/opt/intel/oneapi`
  (source `setvars.sh` before using `icpx`/Level Zero dev tools).
- Box: 12 CPUs, 30 GB RAM — full BF16 model (55 GB) still doesn't fit RAM; on-device
  (24 GB VRAM) execution is the path, not host reference runs.
- safetensors: keep the one-shard-at-a-time discipline from the Windows notes below
  (mapping all 18 × ~4 GB shards at once is fragile on any box).
  Chunk tensors >256 MB through fp32 (lm_head → 4.7 GB otherwise).
- numpy has no BF16: use torch for BF16 stats; raw bytes via `bytes(t.untyped_storage())`
  (`.tobytes()` does not exist on UntypedStorage here).

## Prior-box notes (Windows + pwsh, kept for reference)

- Python was `py -3` (bare `python`/`python3` hit broken MS Store stubs); no heredocs
  in pwsh; temp scripts under `C:\Users\yshar\AppData\Local\Temp\opencode\`.
  Those notes no longer apply on this box.

## Code so far

- `tools/binfer.py` — quantizer + writer (T2.3), §11 validator/loader reference (T2.2).
  Subcommands: `quantize | validate | negatives | mlpcheck` (run from repo root).
  `quantize` streams ~55 GB → ~15 GB `.binfer`; allow up to 2 h timeout.
- `reference/` — operator fixtures, real-weight MLP-L0, 1199-tensor stats, tokenizer finding
  (chat markup needs added-token registration from `tokenizer_config.json`), staged T1.4/T1.5
  prompts. Do not regenerate casually (full pass streams 55 GB).
- `tools/cmdlist/` — all device kernels (`kernels.cpp`) + SPIR-V extraction
  (`extract_spv.py`: content-selected modules, ESIMD images preferred) + replay
  harnesses, one per kernel/layer class (`ctest --preset b60`).
- `tools/decode/decode_l0.cpp` — the native recorded-loop backend (66 lists;
  env: `AINFER_MAXCTX` grow-only, `AINFER_TOP5=0` token-only, `AINFER_KV8=1`
  INT8 KV, `AINFER_PROFILE=1`).
- Generated artifacts live next to the model: `manifest.json`, `memory_budget.json`,
  `conversion_report.json`, `*.binfer`.

## Tooling notes

- `default.edit` requires a prior `default.read` of the file and exact-whitespace `oldString`.
- `default.write` may create the deliverable directly (specs, scripts, reports); never for
  docs not requested.
