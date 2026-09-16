# AInfer — Current Status (release truth)

> Single-source current state. Updated 2026-09-16 (T8.1).
> `progress.md` is the engineering log; this file is the release overview.
> If they disagree, this file wins — fix the other one.

## 1. Supported configuration

| Item | Value |
|---|---|
| Hardware | Intel Arc Pro B60 (`8086:e211`), 24 GB VRAM |
| Toolchain | oneAPI 2026.1 (`source /opt/intel/oneapi/setvars.sh --force`), Level Zero, ESIMD/SYCL |
| Host | Ubuntu, 12 CPUs, 30 GB RAM; Python venv `~/.venvs/ainfer` (torch 2.14 CPU, safetensors, numpy, tokenizers) |
| Test entry | `ctest --preset b60` (47 tests incl. host-only `reports_json` + `tokenizer_golden` gates, T8.2/T8.4) |
| Model | `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, SafeTensors BF16 source (`models/Qwen3.8-27B/`, 18 shards, ~55.6 GB) |
| Architecture | 64 text layers = 48 linear-attention + 16 full-attention (`L%4==3`); 1 MTP head; vision encoder deferred |
| Scope | **v1 text-only, 4K context.** Native 262K and vision are out of scope |
| Container | `models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer`, 15978603616 B, 866 tensors |
| Quantization (default) | INT4 symmetric group-128, BF16 scales, BF16 embeddings |
| KV format (default) | INT8 (`AINFER_KV8=1`); BF16 KV retained for import/handoff paths |
| Baseline (external) | `Dirk-Qwen3.8-27B-UD-Q4_K_S.gguf` — cross-check only, never a quantizer source |

## 2. Runtime paths

| Path | Command | Status |
|---|---|---|
| Default decode | `decode_l0 --prompt=…` / `--ids=` / `--ids-file=` (+`--max-new=`) | **Production.** 66 recorded L0 lists, static arenas, token-identical to streamed-CPU BF16 greedy on tested prompts |
| 64K chunked prefill → decode (single binary) | `decode_l0 --prefill-chunks=N` with `CHUNK64MC_HOST_IN=<prefix>` + `CHUNK_M` | **Production** (`MERGE-OK`, `tools/t74/report_merge.json`). Shared arenas, zero re-uploads, zero disk; bitwise-identical prefill vs harness; 64K needle v2 in-process HIT |
| File cache handoff | `--import-caches=<dir>` (+`CHUNK_DUMP_CACHES` producer) | Fallback only. Superseded by single-binary path |
| HTTP API | T7.3 OpenAI-style SSE server | **Demo-grade**: per-request process + single-flight lock. Persistent worker is T8.8 |

### Key environment flags

`AINFER_MAXCTX` (grow-only, 64K via 65544) · `AINFER_TOP5=0` (token-only)
· `AINFER_KV8=1` (INT8 KV) · `AINFER_PROFILE=1` · `AINFER_STEPLOG=1`
· `AINFER_ATTN=hybrid` (**experimental**, deferred as default: +25% at decode T)
· `CHUNK_M` / `CHUNK_N` / `CHUNK_STREAM=1` · `CHUNK_DUMP_CACHES` / `AINFER_PREFILL_DUMP`

## 3. Quality snapshot (limited — not a production claim)

| Gate | Result | Evidence |
|---|---|---|
| INT4-vs-BF16 teacher-forced batch (T1.4) | 53/60 top-1 | `reference/fwd_T14_batch.json` |
| Greedy agreement, 6 prompts × 48 pos (T1.5) | 47/48 (sole diff: narrow-margin reorder) | `reference/greedy_prompts_t14.json` |
| INT8-KV top-1 preservation | 53/53 | T6.3 |
| 64K needle retrieval, 5 depths | 5/5 HIT (file handoff) + v2 in-process HIT | `tools/t74/report_needle.json`, `report_merge.json` |
| Full CTest suite | 45/45, 1133.8 s | Stamp 13 |

Corpus is <200 cases: no coding/multilingual/long-free-generation coverage.
Broad quality validation is **T8.5** (open).

## 4. Performance snapshot (parity, not lead)

| Metric | Value |
|---|---|
| Device roof | ~437 GB/s |
| ChunkGemm | 2.54 TFLOPS |
| Short-context decode | ~14.7 tok/s vs llama.cpp SYCL ~14.86 tok/s (near-parity) |
| Sustained P64/G64 | ~10.6 tok/s |
| 64K chunked prefill (M=256) | ~80 min (`prefill_ms 4793617.1`), O(W) 11.9→26.8 s/chunk |
| Decode at 65K | ~1.65–2 s/step |

No controlled apples-to-apples benchmark published yet (**T8.11**, open).

## 5. Experimental and deferred features

| Feature | State | Reopen condition |
|---|---|---|
| Hybrid decode (`AINFER_ATTN=hybrid`) | Validated, env-gated, **deferred as default** | Launch-count reduction proven first |
| MTP speculation | **Re-deferred** 2026-09-15 (prompt-lookup alpha≈0; MTP-head ~2.5x at 64K only) | 64K traffic + chained-draft proof + 64K alpha (`tools/t72/report_mtp_revisit.json`) |
| Batching / sliding window / KV-blocking | Dropped (batch=1 explicit) | — |
| Persistent HTTP worker | Open (**T8.8**) | — |
| 200+ case quality suite | Open (**T8.5/T8.6**) | — |
| Memory-safety hardening (typed spans, sanitizers, fault injection) | Open (**T8.9/T8.10**) | — |

## 6. Known issues

- All 91 tracked JSON reports strict-parse (permanent `reports_json` ctest gate, T8.2).
- Tokenizer assets pinned: manifest sha256 verified for all 5 files; `golden_t44.json` + `tokenizer_golden` gate (T8.4). crc32.txt is hub metadata, not a content gate.
- Weight-stats anomalies classified with documented methodology (`weight_stats_v2.json`, T8.3); 0 suspected corruption.
- No GPU power telemetry on this box (`xpu-smi` no device, `intel_gpu_top` i915-only).
- Documented numerics class: near-tie last-ulp flips (e.g. step-25 0.9%, step-65 0.00-margin).

## 7. Last verification

**Stamp 13 (2026-09-16):** full `ctest --preset b60` 45/45 green, 0 failed.
Certifies the single-binary merge regression-free. Does not certify MTP,
hybrid-as-default, broad quality, or serving readiness.
