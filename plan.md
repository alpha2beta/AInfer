# AInfer: Specialized Qwen3.8-27B Runtime for Intel Arc Pro B60

## 1. Objective

Build a lightweight, batch-one inference runtime for the pinned Qwen/Qwen3.8-27B checkpoint on Ubuntu and an Intel Arc Pro B60 (24 GB, Xe2). Optimize first for single-user autoregressive decode while keeping prefill correct and usable.

Pinned source (2026-09-07): `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, mirrored locally at `models/Qwen3.8-27B` (18 SafeTensors shards, BF16, ~55.6 GB declared / ~51.75 GiB on disk, headers validated). Architecture is `qwen3_5` multimodal hybrid: 64 text layers (48 linear-attention + 16 full-attention, interval 4), 1 MTP hidden layer, 27-layer vision encoder. First release is text-only; vision is deferred (see scope).

The runtime should use:

- oneAPI Level Zero for device discovery, memory management, synchronization, and dispatch;
- DPC++ ESIMD kernels where they provide a measured advantage;
- statically allocated inference buffers after initialization;
- an offline, hardware-oriented quantized model format;
- reusable execution structures with minimal host work per generated token.

This is an optimization project, not a general-purpose framework. The initial scope is one model family, one GPU architecture, batch size 1, and a fixed maximum context length.

## 2. Scope and Success Criteria

### Initial scope

- Ubuntu with a pinned, documented Intel compute software stack.
- Intel Arc Pro B60 with 24 GB VRAM.
- Pinned checkpoint `Qwen/Qwen3.8-27B@1d4bf0f` + tokenizer/chat template in `models/Qwen3.8-27B` (SafeTensors BF16 source; the `Dirk-Qwen3.8-27B-UD-Q4_K_S.gguf` file is baseline-only, not a quantizer source).
- Text-only inference first; vision encoder (27 layers, 1152 hidden) explicitly deferred to a later milestone.
- Batch size 1.
- Maximum context length of 4,096 tokens for the first release (native max is 262,144; full context is out of scope for 24 GB v1).
- Greedy decoding first; temperature, top-k, and top-p sampling second (defaults per `generation_config.json`: temp 1.0, top-k 20, top-p 0.95).
- INT4 weight-only quantization with BF16 activations/accumulation as supported by the best-performing kernel path (source dtype is BF16; SSM state dtype is FP32 and stays unquantized initially).
- CLI text generation; an HTTP API is optional and must not block runtime development.

### Definition of done

- The complete model loads without inference-time device allocation.
- Tokenization and greedy output match a trusted reference within the numerical tolerance established for the quantized model.
- The runtime completes prompt prefill and streams decode tokens without hangs, memory errors, or per-token command-list construction.
- Peak device memory stays below a measured safe limit on the 24 GB card.
- Kernel and end-to-end benchmarks are reproducible and report latency, throughput, memory use, model revision, driver version, and test parameters.
- Performance is reported from measurements, not theoretical bandwidth alone.

## 3. Validation Gates

Resolve these gates before committing to low-level kernel and file-format decisions.

### Gate A: Model identity and architecture

Pinned 2026-09-07: `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, local mirror `models/Qwen3.8-27B`. Verified from `config.json` + SafeTensors headers (18/18 shards open, ~1206 tensors indexed):

- parameters: ~27.78B BF16 (`safetensors.total`); declared total 55,562,855,904 bytes;
- text trunk: 64 layers — 48 `linear_attention` + 16 `full_attention` (interval 4), hidden 5120, intermediate 17408, vocab 248320;
- full attention: 24 query heads, 4 KV heads (GQA), head_dim 256;
- linear attention: 16 key heads x 128, 48 value heads x 128, conv kernel 4, FP32 SSM state;
- RoPE: M-RoPE interleaved, sections [11, 11, 10], theta 10,000,000, partial rotary factor 0.25;
- norm: RMSNorm eps 1e-6; MLP act SiLU (`hidden_act`), output gate swish, `attn_output_gate` true;
- embeddings untied (`tie_word_embeddings` false); `mtp_num_hidden_layers` 1, `mtp_use_dedicated_embeddings` false;
- source dtype BF16; tokenizer `tokenizer.json` (12.8 MB) + `vocab.json` + `merges.txt` + `chat_template.jinja` present locally;
- vision encoder present (depth 27) but deferred — text-only manifest is the v1 target.

MTP is confirmed present (1 hidden layer), so T7.2 stays in scope as optional work gated on acceptance measurements — not on existence. Remaining Gate A work: emit the machine-readable manifest file (T1.2) and record tokenizer/revision checksums immutably.

### Gate B: Software and hardware capabilities

Create a small Level Zero probe that records:

- GPU device and revision IDs;
- driver and loader versions;
- available memory heaps and allocation limits;
- supported module formats, floating-point modes, subgroup widths, and synchronization features;
- ESIMD compilation and execution support on the pinned toolchain;
- the practical XMX data-type combinations available on this Xe2 device.

Do not assume native INT4-to-FP16 XMX execution. Benchmark at least unpack-and-dot, unpack-to-INT8, and any supported matrix path before selecting the kernel design.

### Gate C: Memory fit

Generate the memory budget from checkpoint metadata and runtime layouts rather than fixed estimates. This is a hybrid model, so the naive full-transformer formula overestimates. Compute two pools:

```text
Full-attn KV bytes = 2 * 16 * kv_heads(4) * head_dim(256) * max_context * bytes_per_element
Linear-attn state bytes = 48 layers * per-layer SSM/conv state (FP32) + conv buffers
```

Include alignment, quantization scales and zero points, unquantized tensors, logits, temporary reductions, command resources, driver overhead, and allocator granularity. Verify the budget with actual Level Zero allocation tests before model integration.

### Gate D: Numerical baseline

Capture reference outputs from a trusted implementation for:

- individual operators;
- one transformer block;
- short prompt logits;
- greedy token sequences;
- perplexity or task accuracy on a small fixed validation corpus.

Store test prompts, expected tokens or logits, tolerances, checkpoint revision, and reference software version.

## 4. Runtime Architecture

### Host components

- **Model converter:** Reads SafeTensors, validates tensor metadata, quantizes eligible matrices, applies the selected device layout, and writes the runtime container.
- **Container loader:** Validates format version, model identity, tensor shapes, offsets, alignment, quantization metadata, and checksums before allocating GPU memory.
- **Tokenizer:** Uses the pinned tokenizer vocabulary and exact normalization, special-token, and chat-template behavior. Validate it against the reference tokenizer before generation tests.
- **Level Zero backend:** Owns the driver, device, context, queue, modules, kernels, events, command lists, and allocations through explicit RAII lifetimes.
- **Execution planner:** Builds separate prefill and decode paths from fixed tensor and workspace addresses.
- **Sampler:** Runs greedy selection initially. Add configurable sampling only after logits are validated.
- **CLI:** Accepts a prompt, exposes generation settings, streams decoded text, and prints optional timing statistics.

### Device execution paths

Prefill and decode have different shapes and must not share a single assumed GEMV strategy:

- **Prefill:** Process prompt tokens with tiled GEMM and causal attention. Chunk long prompts if required by the workspace budget.
- **Decode:** Process one token at a time with optimized weight-only GEMV or narrow GEMM, append K/V data, attend over the active context, and produce logits.

Use fixed addresses for weights, KV cache, activations, logits, token state, and position state. Small mutable values such as token ID, position, and active sequence length should live in shared or device-visible control buffers so reusable dispatches do not require changing kernel arguments.

## 5. Model Container and Quantization

### Container requirements

Use a versioned `.binfer` container with:

- magic value, format version, endianness, and alignment requirements;
- source model and revision identifiers;
- complete architecture metadata;
- tokenizer assets or an explicit tokenizer identity and checksum;
- tensor directory containing names, shapes, logical data types, storage data types, offsets, lengths, and layout IDs;
- quantization group size, scale type, zero-point policy, and padding metadata;
- per-section checksums;
- independently aligned tensor payloads.

A single file is useful, but tensors should remain independently addressable. Do not hard-code one physical swizzle until benchmarks select it; encode layout IDs so the exporter and runtime can reject incompatible files.

### Quantization workflow

1. Load and validate the pinned SafeTensors checkpoint.
2. Establish an FP16 or BF16 reference and representative calibration/evaluation corpus.
3. Implement a simple, deterministic group-wise INT4 baseline, initially group size 128.
4. Keep numerically sensitive tensors at higher precision where validation shows it is necessary.
5. Compare symmetric and asymmetric quantization only if the extra decode cost is justified by quality.
6. Select the physical packing and swizzle from measured kernel performance.
7. Emit a conversion report with tensor sizes, quantization error, final file size, and checksums.

Acceptance criteria:

- deterministic conversion from identical inputs;
- round-trip metadata and tensor-layout tests;
- dequantized tensor error within documented thresholds;
- model-level quality loss within an explicitly chosen tolerance.

## 6. Memory Plan

Allocate all model-lifetime device memory during initialization. Use a small number of large arenas or allocations to avoid allocation-count and fragmentation limits.

Budget these categories from exact dimensions:

| Category | Derivation |
| --- | --- |
| Quantized weights | Packed values + scales/zero points + higher-precision tensors + alignment |
| KV cache | Formula in Gate C for the selected KV type and 4,096-token limit |
| Activations | Maximum of prefill and decode liveness plans, not the sum of every intermediate |
| Logits and sampling | Vocabulary-sized outputs and reduction workspace |
| Runtime resources | Modules, command lists, events, descriptors, and measured driver overhead |
| Safety margin | Unallocated VRAM retained for the driver and measurement variance |

Start with a contiguous, append-only KV layout for contexts up to the fixed limit. A circular buffer changes context semantics when old tokens are overwritten and is unnecessary for the initial 4K cap. Add sliding-window or ring-buffer behavior only if the model semantics and product requirements call for it.

Candidate KV formats must be validated separately:

- FP16 or BF16 as the correctness baseline;
- FP8 or INT8 only after measuring attention accuracy and conversion overhead;
- per-head or per-block scales included in both memory and bandwidth calculations.

Fail model loading with a clear diagnostic if the computed allocation plan does not fit the selected device and safety margin.

## 7. Kernel Roadmap

Build each operator as an independently testable kernel before assembling a transformer block.

### 7.1 Measurement foundation

- Measure copy and read bandwidth over representative allocation sizes.
- Measure dispatch, barrier, event, and command-list replay overhead.
- Record warm and cold timings with Level Zero timestamp events where supported.
- Establish simple reference kernels before ESIMD specialization.
- Inspect compiler output and occupancy/resource reports rather than inferring register behavior from source alone.

### 7.2 Decode linear kernels

Prioritize INT4 weight-only GEMV and narrow GEMM for QKV, output projection, gate/up projection, down projection, and LM head.

Benchmark:

- group size and scale-loading strategy;
- symmetric versus asymmetric unpacking if both are retained;
- subgroup width and work distribution;
- cache-line-aligned packing and several swizzles;
- fused bias where the architecture requires it;
- FP16/BF16 accumulation behavior and reduction accuracy;
- persistent versus conventional dispatch where supported and beneficial.

Report effective weight bandwidth using bytes actually read, including scales and metadata. Compare against a measured sustainable bandwidth baseline, not only the advertised peak.

### 7.3 Prefill linear kernels

Implement tiled GEMM suited to multiple prompt tokens. Reusing the decode GEMV in a loop is acceptable only as an initial correctness path, not as the performance target.

### 7.4 Normalization and residual operations

Implement RMSNorm with stable reduction and validate it across realistic magnitudes. Evaluate fusions that preserve graph dependencies, such as residual-add plus RMSNorm or quantization/packing of the normalized activation for the following projection.

Do not directly fuse RMSNorm with RoPE as if they were adjacent operations: Q/K projection lies between them in the standard transformer graph. A larger QKV-projection epilogue may apply RoPE to Q and K if measurements justify the complexity.

### 7.5 RoPE and KV writes

- Apply the exact checkpoint-specific RoPE transformation.
- Fuse RoPE with Q/K post-processing and KV-cache writes when profitable.
- Keep position and active-context state in device-visible control memory.
- Test boundary positions and the maximum context length.

### 7.6 Attention (16 full layers) + linear-attention (48 layers)

Full attention (only 16 of 64 layers) follows the prefill/decode split:

- causal tiled attention for prefill;
- vectorized QK dot products, stable softmax, and weighted V reduction for decode;
- GQA indexing required (24 Q heads over 4 KV heads, head_dim 256);
- append-only contiguous KV addressing for the initial fixed context;
- online softmax to avoid materializing a full attention matrix where practical.

Linear attention (48 layers) is separate new work, not covered by the transformer attention kernels:

- 1D conv (kernel 4) + recurrent SSM state update (FP32) per step for decode;
- chunked/cached prefix handling for prefill;
- per-layer persistent SSM state buffers in the static memory plan.

Compare higher-precision and quantized KV variants before choosing the release default. Keep SSM state at FP32 until quality-gated experiments say otherwise.

### 7.7 MLP and elementwise fusion

- Implement the checkpoint's exact gated MLP (confirmed: SiLU `hidden_act`, intermediate 17408, swish output gate, `attn_output_gate` true).
- Evaluate fusing gate activation and elementwise multiplication.
- Reuse activation buffers according to the static liveness plan.
- Fuse only when profiling shows a meaningful end-to-end benefit and tests remain diagnosable.

### 7.8 Sampling

- Start with deterministic argmax.
- Add temperature, repetition penalties, top-k, and top-p as separate validated work.
- Avoid transferring the full logits vector to the host when a device reduction can return only the selected token and required statistics.

## 8. Level Zero Scheduling

Use regular reusable command lists as the initial scheduling design. Record stable kernel launches and barriers after all buffers and kernel arguments are finalized, close the lists, and submit them repeatedly with events or fences according to verified Level Zero semantics.

Separate command lists where control flow or shapes differ:

- prefill chunks;
- one decode transformer pass;
- logits and sampling;
- optional maintenance or debug readback.

An immediate command list is a submission mode, not automatically a reusable pre-recorded graph. Benchmark regular and immediate paths before choosing. Do not rely on mutating arguments of a closed or in-flight command list; use fixed argument addresses and mutable control buffers.

The token loop should perform only the necessary work:

1. Update token/position state through a coherent shared allocation or explicit copy.
2. Signal or submit the decode work after prior use has completed.
3. Wait for the selected-token event or fence without a hot, unbounded CPU spin loop.
4. Read the selected token, stream decoded text, and update stopping state.

Prototype host-visible shared memory and explicit host/device copies. Keep the faster measured option; "zero-copy" should not be assumed to be fastest on a discrete GPU.

## 9. Verification Strategy

### Unit tests

- container parsing, bounds checks, malformed inputs, checksums, and version rejection;
- tokenizer encode/decode and special-token behavior;
- INT4 pack/unpack and scale application;
- tensor swizzles against an inverse/reference layout;
- memory-budget arithmetic and alignment;
- sampling and stopping rules.

### Kernel tests

For every kernel, compare device output to a CPU or trusted framework reference across:

- exact model dimensions and representative reduced dimensions;
- random and adversarial values;
- boundary sequence lengths and positions;
- non-multiple dimensions requiring padding;
- repeated executions to expose synchronization errors;
- tolerances appropriate to each data type and reduction.

### Integration tests

- one transformer layer against the reference;
- complete short-prompt logits;
- deterministic greedy token sequences;
- empty, one-token, and maximum-length prompts;
- out-of-memory and incompatible-model diagnostics;
- repeated generation runs to detect leaks and stale state.

### Quality tests

Use a fixed validation corpus to compare the quantized runtime with the higher-precision reference. Record perplexity or task-metric deltas and inspect generation regressions. Kernel-level numerical tolerance alone is not sufficient evidence of model quality.

## 10. Benchmarking and Performance Targets

The rough bandwidth-only decode ceiling is:

```text
tokens/s <= measured sustainable bandwidth / bytes read per generated token
```

The denominator must include quantized weights, scales, higher-precision tensors, KV reads/writes, activations, and other material traffic. The 456 GB/s advertised bandwidth divided by an estimated 15.2 GB model footprint gives approximately 30 tokens/s, but this is an optimistic upper bound rather than a delivery target.

Track:

- time to first token, split into tokenization, model load, and prefill;
- prefill throughput at several prompt lengths;
- inter-token latency and decode tokens/s at several active context lengths;
- p50, p95, minimum, and maximum latency over sustained runs;
- effective bandwidth for dominant linear kernels;
- peak device and host memory;
- power and temperature if stable tooling is available;
- output quality relative to the reference.

Use staged targets:

1. **Correctness:** Full model generates validated output with no dynamic inference allocations.
2. **Baseline:** Custom runtime is stable and all major kernels have measured profiles.
3. **Optimization:** Decode reaches at least 70% of the measured roofline implied by actual bytes moved.
4. **Stretch:** Pursue 25-28 tokens/s only if the measured sustainable bandwidth and complete traffic model show it is attainable.
5. **Optional speculation:** Set an MTP target only after checkpoint support, acceptance rate, verification cost, and quality are measured.

All benchmark reports must include warm-up policy, run duration, prompt and generation lengths, context occupancy, sampling settings, clocks/power mode where known, and software revisions.

## 11. Implementation Milestones

### Milestone 0: Reproducible environment

- Pin Ubuntu, kernel, Intel compute runtime, Level Zero loader, DPC++ compiler, and build-tool versions.
- Add device-capability and allocation probes.
- Create CMake presets and a smoke-test executable.
- Capture a baseline from an available reference runtime on the same hardware.

Exit criteria: the device is identified correctly, a compiled kernel runs, timestamps work, and the environment can be reproduced from documentation.

### Milestone 1: Model specification and reference suite

- Pin the exact model/tokenizer revision. DONE 2026-09-07 (`1d4bf0f`, local mirror validated).
- Confirm MTP presence. DONE (`mtp_num_hidden_layers` 1).
- Generate a machine-readable architecture manifest (text-only v1 + deferred vision section).
- Capture operator, logits, token, memory, and quality baselines.
- Finalize the calculated memory budget with the hybrid formula (16-layer KV + 48-layer SSM state).

Exit criteria: all architecture assumptions are replaced with verified values and the projected model fits with a safety margin.

### Milestone 2: Container and exporter

- Implement the versioned `.binfer` format and strict loader validation.
- Implement deterministic INT4 quantization and initial packing.
- Add conversion reports and round-trip tests.
- Load tensors into static Level Zero allocations.

Exit criteria: the entire converted model loads reproducibly and every tensor can be checked against exporter output.

### Milestone 3: Kernel microbenchmarks

- Build bandwidth and dispatch baselines.
- Implement reference and optimized decode linear kernels.
- Implement and validate normalization, RoPE, attention, MLP, and sampling primitives.
- Select packing, swizzle, workgroup, subgroup, and accumulation strategies from data.

Exit criteria: each operator passes numerical tests and has a benchmark against its measured roofline or reference.

### Milestone 4: Correct end-to-end runtime

- Assemble one transformer block, then the full model.
- Implement a correctness-first prefill path and optimized decode path.
- Add tokenizer, greedy sampling, stopping, and CLI streaming.
- Compare logits and generated sequences with the reference.

Exit criteria: representative prompts produce accepted outputs and repeated runs are stable.

### Milestone 5: Static scheduling and memory optimization

- Finalize activation liveness and buffer reuse.
- Remove inference-time allocation.
- Record and reuse Level Zero command lists with fixed addresses.
- Move token selection to the device if beneficial.
- Profile and eliminate avoidable synchronization and transfers.

Exit criteria: steady-state decode performs no allocation or command-list construction and passes synchronization stress tests.

### Milestone 6: Performance tuning

- Optimize the top end-to-end bottlenecks in profile order.
- Tune decode across multiple context lengths and prefill across multiple prompt lengths.
- Evaluate justified fusions and KV-cache quantization.
- Publish reproducible benchmark results and compare them with the staged targets.

Exit criteria: performance is stable, explained by profiles, and does not compromise the agreed quality threshold.

### Milestone 7: Optional features

- Add configurable sampling.
- Add MTP/speculative verification only if supported by the pinned model.
- Add an OpenAI-compatible HTTP endpoint after CLI correctness and performance are stable.
- Consider larger contexts, sliding windows, or limited batching as separately scoped work.

## 12. Key Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Model name or architecture assumptions are incorrect | Pinned 2026-09-07 (`Qwen3.8-27B@1d4bf0f`, hybrid verified); manifest file still to be emitted |
| Hybrid linear-attention complexity understated by transformer-only plan | Added §7.6 linear-attention kernels + SSM-state memory pool; text-only v1, vision deferred |
| GGUF used as quantizer source causing double quantization | SafeTensors BF16 mirror now local; GGUF is baseline-only |
| 24 GB capacity is insufficient after overhead | Calculate from metadata, test real allocations, retain a safety margin, and define fallback precision/layout choices |
| Xe2 lacks an efficient assumed INT4 XMX path | Probe capabilities and benchmark unpack/vector/matrix alternatives before fixing the format |
| Offline swizzle overfits one kernel | Version layout IDs and retain a simple baseline layout |
| Quantized KV cache harms quality | Keep FP16/BF16 baseline and gate lower precision on model-level evaluation |
| Command-list reuse is incompatible with mutable state | Use fixed kernel arguments and device-visible control buffers; verify lifecycle semantics in isolation |
| Prefill is unusably slow | Give prefill its own tiled GEMM and attention path rather than looping decode kernels |
| Claimed bandwidth utilization is unrealistic | Use measured sustainable bandwidth and complete byte accounting |
| Aggressive fusion delays correctness | Integrate unfused validated kernels first, then fuse profile-selected paths |
| Tokenizer mismatch invalidates model comparisons | Pin assets and validate token IDs, special tokens, and chat templates against the reference |

## 13. Immediate Next Actions

1. ~~Confirm and pin the exact Qwen model and tokenizer revisions.~~ DONE 2026-09-07.
2. ~~Generate the architecture manifest file and exact hybrid memory budget (T1.2/T1.6).~~ DONE 2026-09-07.
3. ~~Pin the Intel software stack and run the Level Zero/ESIMD capability probe.~~ DONE 2026-09-07 (T0.1-T0.3).
4. ~~Capture reference logits, greedy outputs, and quality samples against the pinned SafeTensors.~~ DONE box-native 2026-09-11/12 (T1.4/T1.5: streamed-CPU BF16 greedy vs loop, 47/48 top-1; no bigger host per platform rule).
5. ~~Measure sustainable bandwidth and dispatch overhead on the B60.~~ DONE 2026-09-07 (T3.1: ~437 GB/s roof).
6. ~~Use those results to select the first INT4 layout and decode GEMV design.~~ DONE 2026-09-08 (T3.2/T3.3: layout-0, dp4a-i8).
