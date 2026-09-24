# Performance Claim Correction — Long-Context Comparison

**Date:** 2026-09-24  
**Branch:** `258v`  
**Related release evidence:** `tools/bench_258v/report_llama_comparison.json`

## Correction

The statement that **“AInfer outperforms llama.cpp”** is too broad and must be
qualified by context length and benchmark methodology.

The committed comparison proves a short-context result only:

| Runtime | Reported decode | Scope |
|---|---:|---|
| AInfer Level Zero | 35.06 tok/s | short-context T7.1/T7.5 benchmark |
| llama.cpp Vulkan | 29.33 tok/s | short-context reference benchmark |
| Relative result | 1.20× | valid only for that controlled scope |

The corrected claim is:

> **AInfer exceeds the llama.cpp Vulkan reference on the validated
> short-context benchmark. This result must not be generalized to long
> prompts. Long-context performance requires a separate same-model,
> same-prompt comparison.**

## Long-context observation

The following live server logs compare approximately the same prompt length:

| Runtime | Prompt tokens | Prefill | Decode |
|---|---:|---:|---:|
| AInfer HTTP server | 6,729 | 84.3 tok/s | 9.4 tok/s |
| llama.cpp server | 6,776 | 183.2 tok/s | 23.0 tok/s |

On these observations, llama.cpp is approximately **2.17× faster in prefill**
and **2.45× faster in decode**. The AInfer request took approximately 80 s in
prefill, which also explains the client disconnect after prefill when the
client-side HTTP timeout was shorter than the request latency.

These live observations are diagnostic evidence, not yet a release-grade
apples-to-apples benchmark. The comparison is valid only if both runs use the
same checkpoint, quantization quality, prompt, context settings, GPU backend
conditions, and MTP configuration. The llama.cpp log also reports 100% draft
acceptance, while the AInfer server log does not expose acceptance for that
request.

## Why a visible “Hello” can be 6–7K tokens

The server reports the fully rendered prompt, not only the latest visible user
message. A new OpenCode session can still include a large system/developer
prompt and all registered tool schemas. Those tokens are processed during
prefill and remain in the KV cache for decode.

## Release wording changes

Do not use:

> AInfer is faster than llama.cpp.

Use:

> AInfer is faster than the recorded llama.cpp Vulkan reference in the
> validated short-context T7.5 benchmark. Long-context performance is not
> currently claimed as superior; a same-model context sweep is required.

## Follow-up benchmark required

Run both runtimes on the same Core Ultra 7 258V with the same checkpoint and
prompt at **512, 2K, 4K, and 8K tokens**, recording separately:

- prompt/prefill throughput (`pp`),
- generation throughput (`tg`),
- MTP enabled and disabled,
- BF16 KV and KV8 where supported,
- exact model/quantization/backend identifiers,
- total latency and client timeout behavior.

Until that sweep is complete, the short-context 1.20× claim and the observed
long-context deficit must remain separate claims.
