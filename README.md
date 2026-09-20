# AInfer (B60 branch) — INT4 recorded-loop inference for Qwen3.8-27B on Intel Arc Pro B60

Custom NInfer-style inference runtime for pinned model `Qwen/Qwen3.8-27B`
(64 text layers = 48 linear-attention + 16 full-attention, MTP head;
vision encoder deferred). Target hardware: Intel Arc Pro B60 (`8086:e211`,
Battlemage G21, 24 GB VRAM). Text-only v1, 4K default context, 64K via
chunked prefill.

Related branches: `258v` (Qwen3.5-MoE port on Intel Core Ultra 7 258V),
`main` (initial snapshot).

## Measured status (see `STATUS.md` for current truth)

| Path | Result |
|---|---|
| Default INT4 decode, greedy | ~14.9 tok/s, token-identical to streamed-CPU BF16 reference |
| MTP speculative decode (`--mtp`, adaptive depth-2) | 19–25.6 tok/s by alpha, bitwise identical |
| Chunked prefill → decode, single binary | 64K in ~80 min; needle retrieval 20/20 (15 new + 5 legacy) |
| 200-case quality suite (INT4 / INT8-KV / llama.cpp) | no quant-attributable grade flips; INT8-KV delta none |
| Full test suite | `ctest --preset b60` green (Stamps 12–15 in `progress.md`) |

External references (same card/model class): llama.cpp SYCL ~14.9–17 tok/s
(FP16 build), llama FP16 + draft-mtp 29.6 tok/s, OpenVINO GenAI INT4
23.3 tok/s decode / ~1046 tok/s prefill. Our prefill (~30 tok/s) is the
known gap — see `progress.md` prefill program entries.

## Layout

- `models/Qwen3.8-27B/` — SafeTensors BF16 source (quantizer input; GGUF is baseline-only)
- `tools/binfer.py` — quantizer + `.binfer` container writer (`quantize | validate | negatives | mlpcheck`)
- `tools/cmdlist/` — all device kernels (`kernels.cpp`) + SPIR-V extraction + replay harnesses (one per kernel/layer class)
- `tools/decode/decode_l0.cpp` — native recorded-loop backend: default decode (66 lists), `--prefill-chunks` single-binary 64K, `--mtp` speculative, `--import-caches` file handoff
- `tools/forward/` — streamed-CPU BF16/INT4 reference forwards (box-native truth)
- `tools/quality/` — 200-case corpus, runners, BF16 margin analysis (`report_t85.json`, `report_t86.json`)
- `tools/mtp/` — MTP reference copies for porting (README + kernels + reports)
- `tools/tokenizer/` — tokenizer with added-token registration + pinned golden
- `reference/` — operator fixtures, real-weight references, staged prompts
- `docs/binfer_spec.md` — normative container spec; `docs/toolchain.md` — oneAPI/Level Zero quarantines

## Build / test / run

Environment: Ubuntu + oneAPI 2026.1 (`source /opt/intel/oneapi/setvars.sh`
first), project venv at `~/.venvs/ainfer/bin/python` (uv-managed). Repo
lives on a USB mount: no symlinks; venv must live on ext4.

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake --preset b60 && cmake --build build-b60 --target decode_l0
ctest --preset b60            # full suite (~20 min)
# 70-token e2e:
./build-b60/tools/decode/decode_l0 models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer \
  1 1 build-b60/tools/cmdlist out.json --ids-file=ids.txt --max-new=64
```

Key env flags: `AINFER_MAXCTX` (grow-only), `AINFER_TOP5=0` (token-only),
`AINFER_KV8=1` (INT8 KV), `AINFER_PROFILE=1`, `AINFER_MTP2=0/1` (force
depth-1/2), `CHUNK_M`, `AINFER_FLASH=1` (fused attention, experimental),
`AINFER_GEMM_DB=1` (experimental, negative on B60).

## Docs

- `STATUS.md` — current release truth (read this first)
- `plan.md` — design; `tasks.md` — scope (stable task IDs); `progress.md` — engineering log
- `review.md` — 2026-09-15 external review + T8.1–T8.11 roadmap (Phases 8 mostly closed; open: persistent HTTP worker, memory-safety hardening, controlled benchmark)
- `AGENTS.md` — environment + repo conventions for agents
