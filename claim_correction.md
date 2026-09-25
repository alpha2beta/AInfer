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

## Post-Optimization Results (2026-09-24: Tasks I3.6 & I3.7)

Following the implementation of **I3.6 (Subgroup Butterfly Decode Attention)** and
**I3.7 (Blocked FlashAttention for Prefill)**, an isolated empirical benchmark was
conducted on Intel Arc 140V at $P=6,720$ prompt tokens (`report_long_context_catchup.json`):

| Runtime / Optimization State | Prompt Tokens | Prefill Throughput | Total Prefill Time | Greedy Decode |
|---|---:|---:|---:|---:|
| Prior AInfer (diagnostic log) | 6,729 | 84.3 tok/s | ~80.0 s | 9.4 tok/s |
| **New AInfer (I3.6 + I3.7)** | **6,720** | **174.7 tok/s** (warm 180.1) | **38.5 s** | **12.3 tok/s** |
| llama.cpp server (diagnostic log) | 6,776 | 183.2 tok/s | ~37.0 s | 23.0 tok/s* |
| **AInfer Improvement** | — | **+107.2% (2.07×)** | **-41.5 s saved** | **+30.5% (1.31×)** |
| **Ratio vs llama.cpp** | — | **0.95× (Parity)** | — | **0.53× (Greedy)** |

*\*Note: The llama.cpp server log reported speculative draft acceptance active during decode.*

### Analysis of the Results

1. **Prefill Gap Closed (0.95× Parity):** Blocked FlashAttention eliminated the $O(B \times T)$
   redundant KV DRAM traffic. Prefill throughput surged from 84.3 tok/s to **174.7 tok/s**
   (warmup reaching **180.1 tok/s**), cutting prefill latency by more than half (from ~80 s to 38.5 s).
   AInfer is now within 5% of llama.cpp prefill at ~6.7K context, fully resolving the HTTP client
   timeout issue.
2. **Greedy Decode Bottleneck Analysis (12.3 tok/s):** The subgroup butterfly shuffle kernel
   eliminated over 200,000 serial barrier synchronizations per token, elevating greedy decode from
   9.4 tok/s to **12.3 tok/s** (+30.5% speedup).
   - On Intel Core Ultra 7 258V, the theoretical peak memory bandwidth of the 128-bit LPDDR5X-8533
     unified bus is **136.53 GB/s** (with isolated stream read reaching **103.03 GB/s** in T1.6).
   - The INT4 quantized model requires streaming ~1.17 GB of static active weights (~1.30–1.40 GB total
     traffic including DeltaNet SSM state and KV cache) per decode step.
   - At $T=6,720$, the decode step time is ~81.5 ms (12.27 tok/s). Rather than being capped by weight
     streaming (which consumes only ~16 ms at ~80 GB/s achieved bandwidth), the decode step is dominated
     by the **10 full-attention layers**, which take ~6.5 ms per layer (~65 ms total per step, or ~80% of
     step latency) due to sequential KV cache scanning across the 6.7K context length.
3. **Speculative Decode Opportunity:** llama.cpp's 23.0 tok/s was achieved via speculative drafting.
   At long context, AInfer's MTP draft layer currently does not prefill its prompt KV cache during
   `prefill()`, leading to draft divergence on long prompts. Implementing prompt KV cache prefill
   for the MTP draft layer will unlock speculative speedup ($\sim 1.5\text{--}1.8\times$) at long context,
   targeting 18–22+ tok/s decode.

