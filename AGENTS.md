# AGENTS.md — AInfer

## What this repo is

Custom NInfer-style inference runtime plan for pinned model `Qwen/Qwen3.8-27B`
(hybrid: 64 text layers = 48 linear-attention + 16 full-attention, MTP head,
vision encoder **deferred**). Target HW is Intel Arc Pro B60 (`8086:e211`, Battlemage G21).

## Doc trio — update consistently

- `plan.md` = design. `tasks.md` = scope; task IDs (`T0.1`…`X2`) are stable, never renumber.
  `progress.md` = live status (mirrors tasks phase tables + changelog).
- After finishing work: mark status in `tasks.md`, update `progress.md` dashboard/rows/changelog.

## Key domain facts (do not re-derive)

- KV cache covers **16 full layers only**; 48 linear layers carry FP32 SSM state (~144 MiB).
  Naive 64-layer KV math overestimates ~4×.
- v1 is **text-only, 4K context**. Native 262K + vision are out of scope.
- `Dirk-Qwen3.8-27B-UD-Q4_K_S.gguf` (repo root) is **baseline-only**; quantizer source is
  `models/Qwen3.8-27B/` SafeTensors BF16 (`@1d4bf0f`, see `manifest.json`).
- `docs/binfer_spec.md` is normative for the container. Sharp edges: section 5 (tensor dir)
  has **no length prefix**, table entries are 32 B (`I2Q3I`), dir entries 192 B with 12 B reserved.

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
- Generated artifacts live next to the model: `manifest.json`, `memory_budget.json`,
  `conversion_report.json`, `*.binfer`.

## Tooling notes

- `default.edit` requires a prior `default.read` of the file and exact-whitespace `oldString`.
- `default.write` may create the deliverable directly (specs, scripts, reports); never for
  docs not requested.
