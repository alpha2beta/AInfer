# AInfer — Current Status (release truth)

> Single-source current state. Updated 2026-09-19 (T7.2/T8.6, Stamp 15).
> `progress.md` is the engineering log; this file is the release overview.
> If they disagree, this file wins — fix the other one.

## 1. Supported configuration

| Item | Value |
|---|---|
| Hardware | Intel Arc Pro B60 (`8086:e211`), 24 GB VRAM |
| Toolchain | oneAPI 2026.1 (`source /opt/intel/oneapi/setvars.sh --force`), Level Zero, ESIMD/SYCL |
| Host | Ubuntu, 12 CPUs, 30 GB RAM; Python venv `~/.venvs/ainfer` (torch 2.14 CPU, safetensors, numpy, tokenizers) |
| Test entry | `ctest --preset b60` (48 tests incl. host-only `reports_json` + `tokenizer_golden` gates, T8.2/T8.4; fast tier 40/40 in 86s) |
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
| Speculative decode | `decode_l0 ... --mtp` | **Production.** Depth-1 & Depth-2 chained drafts + adaptive M2/M3 dispatch, 100% bitwise greedy determinism, 21.24–25.56 tok/s |
| 64K chunked prefill → decode (single binary) | `decode_l0 --prefill-chunks=N` with `CHUNK64MC_HOST_IN=<prefix>` + `CHUNK_M` | **Production** (`MERGE-OK`, `tools/t74/report_merge.json`). Shared arenas, zero re-uploads, zero disk; bitwise-identical prefill vs harness; 64K needle v2 in-process HIT |
| File cache handoff | `--import-caches=<dir>` (+`CHUNK_DUMP_CACHES` producer) | Fallback only. Superseded by single-binary path |
| HTTP API | T7.3 OpenAI-style SSE server | **Demo-grade**: per-request process + single-flight lock. Persistent worker is T8.8 |

### Key environment flags

`AINFER_MAXCTX` (grow-only, 64K via 65544) · `AINFER_TOP5=0` (token-only)
· `AINFER_KV8=1` (INT8 KV) · `AINFER_PROFILE=1` · `AINFER_STEPLOG=1`
· `AINFER_MTP=1` (speculative decoding) · `AINFER_MTP2=0/1` (depth-2 A/B control)
· `AINFER_ATTN=hybrid` (**experimental**, deferred as default: +25% at decode T)
· `CHUNK_M` / `CHUNK_N` / `CHUNK_STREAM=1` · `CHUNK_DUMP_CACHES` / `AINFER_PREFILL_DUMP`

## 3. Quality snapshot (limited — not a production claim)

| Gate | Result | Evidence |
|---|---|---|
| INT4-vs-BF16 teacher-forced batch (T1.4) | 53/60 top-1 | `reference/fwd_T14_batch.json` |
| Greedy agreement, 6 prompts × 48 pos (T1.5) | 47/48 (sole diff: narrow-margin reorder) | `reference/greedy_prompts_t14.json` |
| INT8-KV top-1 preservation | 53/53 | T6.3 |
| 64K needle retrieval, 5 depths | 5/5 HIT (file handoff) + v2 in-process HIT | `tools/t74/report_needle.json`, `report_merge.json` |
| Full CTest suite | 48/48, 1197.9 s (fast tier 40/40, 86.0 s) | Stamp 15 |
| 200-case benchmark, 3 configs (T8.5) | factual 40/40/40, coding 30/30/30, summ 20/20/20, longgen 20/20/20, arith 32/40 (shared fails), biling 27/28 (shared paraphrase) | `tools/quality/report_t85.json` |
| Retrieval 20/20 (T8.5) | 15/15 merged-path (4K–32K, multi, distractors) + 5/5 legacy 64K | `/mnt/usb/retr/*/check.json`, `report_needle.json` |
| Margin divergence, 21 BF16-backed (T8.6) | INT4-in-BF16-top5 100%; ZERO quant-attributable grade flips; INT8-KV delta none | `tools/quality/report_t86.json` |

200-case corpus covers coding/multilingual/long-generation/retrieval; remaining limits: no adversarial/long-free-form-instruction-following scale, semantic judgments recorded not scored.

## 4. Performance snapshot (AInfer leads llama.cpp)

| Metric | Value |
|---|---|
| Device roof | ~437 GB/s |
| ChunkGemm | 2.54 TFLOPS |
| Baseline decode (`AINFER_TOP5=0`) | 15.02 tok/s (66.58 ms/tok) vs llama.cpp SYCL ~14.86 tok/s (+1.1%) |
| Speculative decode (`--mtp` / `AINFER_MTP=1`) | **21.24–25.56 tok/s** (47.09–39.12 ms/tok, adaptive M2/M3) vs llama.cpp SYCL 14.86 tok/s (**+43% to +73% throughput lead**) |
| Speculative greedy determinism | **100% bitwise token identical** to non-speculative baseline (zero quality loss) |
| 64K chunked prefill (M=256) | ~80 min (`prefill_ms 4793617.1`), O(W) 11.9→26.8 s/chunk |
| Decode at 65K | ~1.65–2 s/step |

## 5. Experimental and deferred features

| Feature | State | Reopen condition |
|---|---|---|
| MTP speculation (`--mtp` / `AINFER_MTP=1`) | **Production.** Closed 2026-09-19 (T7.2) | Int4GemvM2 & Int4GemvM3 dual/triple-vector ESIMD + zero-rollback specular states + adaptive controller; 21.24–25.56 tok/s (+43% to +73% over llama.cpp) |
| Hybrid decode (`AINFER_ATTN=hybrid`) | Validated, env-gated, **deferred as default** | Launch-count reduction proven first |
| Batching / sliding window / KV-blocking | Dropped (batch=1 explicit) | — |
| Persistent HTTP worker | Open (**T8.8**) | — |
| 200+ case quality suite | Done (**T8.5/T8.6**, 2026-09-17) | Gate C evidence complete |
| Memory-safety hardening (typed spans, sanitizers, fault injection) | Open (**T8.9/T8.10**) | — |

## 6. Known issues

- All 102 tracked JSON reports strict-parse (permanent `reports_json` ctest gate, T8.2).
- Tokenizer assets pinned: manifest sha256 verified for all 5 files; `golden_t44.json` + `tokenizer_golden` gate (T8.4). crc32.txt is hub metadata, not a content gate.
- Weight-stats anomalies classified with documented methodology (`weight_stats_v2.json`, T8.3); 0 suspected corruption.
- No GPU power telemetry on this box (`xpu-smi` no device, `intel_gpu_top` i915-only).
- Documented numerics class: near-tie last-ulp flips (e.g. step-25 0.9%, step-65 0.00-margin).

## 7. Last verification

**Stamp 15 (2026-09-19):** T7.2 MTP speculative decoding closed (depth-1 $M=2$ and depth-2 chained $M=3$ with adaptive depth controller). Phase 8 P0 & P1 quality closed (200-case suite + 21 BF16 margin analyses, Gate C).
Int4GemvM2 (dual-vector, 310 us) & Int4GemvM3 (triple-vector, 308 us) landed in production (`gemvm2.spv`, `gemvm3.spv`).
Two-level zero-rollback specular state management and recorded 64 dual/triple-verification layer lists integrated into `decode_l0` with dynamic adaptive dispatch.
Verified on Arc Pro B60: 100% bitwise token determinism with baseline greedy generation, sustained throughput 21.24 tok/s typical, up to 25.56 tok/s high-alpha (+43% to +73% advantage over llama.cpp SYCL 14.86 tok/s).
CTest suite 48/48 green (1197.9 s), fast tier 40/40 green (85.95 s), 102/102 reports strict-parse.
