# AInfer Migration Plan

**Migration target:** Intel Core Ultra 7 258V under CachyOS  
**Target model:** `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized` (Qwen3.5-MoE architecture fine-tune; no official `Qwen/...` release exists for this checkpoint)  
**Source baseline:** AInfer on Intel Arc Pro B60 with Qwen3.8-27B  
**Plan date:** 2026-09-17  
**Status:** Superseded in part — target model identity corrected post-verification (see `target_model_identity.json`); this document's "Qwen3.6-35B-A3B" placeholder name is retained below only where historically descriptive, replaced by the verified checkpoint elsewhere

## 1. Objective

Migrate AInfer from its current discrete-GPU implementation for dense Qwen3.8-27B on Intel Arc Pro B60 to a model-specific runtime for sparse MoE Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE architecture) on the integrated Intel Arc 140V GPU in Intel Core Ultra 7 258V.

The goal is not to mechanically port the existing runtime. The migration must revalidate every assumption affected by:

- The shift from a dense hybrid model to a sparse MoE hybrid model.
- The shift from dedicated VRAM to CPU/iGPU shared memory.
- The reduced GPU execution resources and memory bandwidth.
- The shift from pinned Ubuntu packages to the CachyOS rolling environment.
- The need for dynamic expert selection and expert-weight access.

The first release should remain text-only, batch size 1, and model-specific. It should preserve AInfer's existing strengths in deterministic conversion, strict container validation, static allocation, recorded Level Zero execution, and evidence-based performance decisions.

## 2. Migration principles

1. **Verify before specializing.** Model topology, tensor inventory, active-expert count, memory behavior, and Level Zero capabilities must come from the actual checkpoint and target machine.
2. **Separate measured values from estimates.** Bandwidth ceilings, active-weight traffic, prefill throughput, and decode throughput are planning hypotheses until measured.
3. **Use one primary runtime path.** Prefill and decode must share in-memory state. File-based cache handoff may remain only as a diagnostic option.
4. **Preserve release gates.** Correctness and quality gates must pass before performance results are accepted.
5. **Do not inherit B60 tuning blindly.** The row-major INT4 DP4A design is the starting candidate, not the predetermined winner on Arc 140V.
6. **Treat system memory as a shared resource.** CPU activity, page migration, and iGPU weight access compete for the same memory subsystem.

## 3. Source and target differences

| Dimension | Existing AInfer baseline | Migration target | Required response |
|---|---|---|---|
| Model | Qwen3.8-27B dense hybrid | Tiel-Coder-35B-A3B-Genesis-Hermes (Qwen3.5-MoE) sparse MoE hybrid | Regenerate manifest, routing model, container metadata, and kernels |
| Text layers | 64, reportedly 48 linear plus 16 full attention | Reported 40, reportedly 30 DeltaNet-MoE plus 10 full-attention-MoE | Verify from checkpoint before implementation |
| Hidden size | 5120 | Reported 2048 | Retune all exact-shape kernels |
| Active computation | Dense weights per token | Sparse selected experts per token | Add deterministic top-k routing and expert execution |
| GPU | Arc Pro B60, discrete | Arc 140V, integrated | Reprobe capabilities and retune work distribution |
| Memory | Dedicated device memory | Shared system memory | Re-evaluate allocation type, page placement, and CPU contention |
| OS/toolchain | Pinned Ubuntu and oneAPI stack | CachyOS rolling environment | Pin exact package and kernel versions |
| Long-context path | Chunked prefill plus diagnostic file handoff | Unified in-memory prefill and decode | Implement single-process state ownership from the start |

All target-model figures above are provisional until Phase 1 emits a verified architecture manifest. Each provisional figure must be traceable to a specific source (model card, `config.json` field, or prior AInfer report) so that Phase 2's manifest generation can diff against a known origin and distinguish a documentation error from a genuine architecture surprise.

## 4. Baseline interpretation from current AInfer reports

The current AInfer evidence provides useful engineering patterns but not direct target performance predictions.

### 4.1 Decode baseline

Current reports describe approximately:

- 14.7 tokens/s at short context.
- 10.6 tokens/s for the reported P64/G64 sustained run.
- A llama.cpp SYCL baseline near 14.86 tokens/s on the B60 configuration.

These are decode or sustained-generation measurements for a different model and GPU. They should be used to validate benchmark methodology, not as migration acceptance thresholds.

### 4.2 Prefill baseline

The available report does not contain a clean standalone prefill-throughput field. A prompt of approximately 64 tokens and an estimated prefill interval of approximately 6 seconds imply:

```text
64 tokens / 6 seconds = approximately 10.7 tokens/s
```

This is a derived estimate, not a directly reported benchmark. It may include timing that is not purely model prefill. It must not be treated as the official AInfer prefill rate.

The migrated runtime must report prefill independently using explicit fields:

```json
{
  "prompt_tokens": 64,
  "model_load_seconds": 0.0,
  "tokenization_seconds": 0.0,
  "prefill_seconds": 0.0,
  "prefill_tokens_per_second": 0.0,
  "first_decode_seconds": 0.0,
  "cold_ttft_seconds": 0.0,
  "warm_ttft_seconds": 0.0
}
```

### 4.3 Long-context baseline

The existing reports establish functional 64K prefill and 5/5 needle retrieval, but they do not provide one reliable, isolated 64K prefill tokens/s metric suitable for target comparison. The migration must therefore establish a new long-context baseline rather than extrapolating from the B60 results.

## 5. Phase 0: Scope, identifiers, and acceptance definitions

### Objectives

- Pin the exact `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized` repository and revision.
- Define the exact Core Ultra 7 258V machine, memory size, firmware, and power profile.
- Define supported context lengths for the first migration release.
- Define correctness, quality, memory, and performance acceptance gates.
- Produce a rough worst-case memory feasibility estimate before committing further phases.

### Deliverables

- `migration_scope.md`
- `target_model_identity.json`
- `target_machine_identity.json`
- Initial `STATUS.md`
- `memory_feasibility_estimate.md`: a coarse, provisional-figure-based worst-case tally of INT4 expert weights (all experts, not only active), shared/dense weights, KV and DeltaNet state, workspaces, and estimated OS/CPU-process/page-cache overhead against the target machine's total unified memory.

### Exit criteria

- Model and tokenizer revisions are immutable.
- Target hardware and OS identifiers are recorded.
- Assumptions are explicitly labeled as provisional or verified.
- Batch size 1 and text-only scope are confirmed.
- The worst-case memory feasibility estimate shows the model plausibly fits the target machine with a stated safety margin, using provisional figures. If it does not plausibly fit, quantization policy, context-tier ambition, or hardware scope must be revisited before Phase 1 begins. This is a coarse go/no-go check, not a substitute for the measured budget in Phase 2.

## 6. Phase 1: CachyOS and Lunar Lake foundation

### 6.1 Pin the target software stack

Record the exact versions of:

- CachyOS release and kernel.
- Active Intel kernel driver.
- Intel compute runtime.
- Level Zero loader and headers.
- Intel Graphics Compiler.
- oneAPI DPC++/C++ compiler.
- CMake, Ninja, GCC, Python, and project packages.

Because CachyOS is rolling, package names alone are insufficient. Store package versions and repository snapshots or package-cache references needed to reproduce the environment.

Recording versions is not the same as being able to restore them weeks later, since kernel and Mesa/compute-runtime updates can silently change Level Zero behavior between pinning and the Phase 7 benchmark run. Adopt an explicit reproduction mechanism rather than relying on notes alone:

- Retain the local pacman package cache (or an equivalent snapshot of downloaded packages) for every version actually used, not just a version-number log.
- Perform toolchain-affecting work inside a pinned container or chroot built from the recorded snapshot, so the benchmark environment can be reconstructed exactly rather than approximately.
- Keep a tested downgrade/rollback procedure, and re-verify it works before Phase 7 rather than assuming the cache alone is sufficient.

### 6.2 Verify device selection and driver binding

Port the existing device probe and verify:

- Exact PCI device ID.
- Device name and revision.
- Active kernel-mode driver.
- Execution-unit or vector-engine topology as reported by the runtime.
- Subgroup widths.
- FP16, BF16, INT8, DP4A, DPAS, and module-format support.
- Timestamp and profiling support.
- Maximum allocation and practical allocatable memory.
- User access through the required render and video groups.

Do not hard-code the proposed PCI ID or execution-resource count until the probe confirms it.

### 6.3 Characterize unified-memory behavior

Benchmark at minimum:

- `zeMemAllocDevice` buffers.
- `zeMemAllocShared` buffers.
- Host-visible buffers where supported.
- First-touch versus warm-access behavior.
- CPU-to-iGPU and iGPU-to-CPU synchronization cost.
- Page migration and oversubscription behavior.

The B60 result that explicit device-memory control updates outperformed shared memory must not be assumed to apply to Lunar Lake. Select the target policy from measurement.

#### 6.3.1 Concurrent CPU and GPU memory contention (dedicated benchmark)

This is treated as its own gated measurement rather than a background bullet, because it is the most likely dominant bottleneck for a sparse MoE model on shared memory: expert weights generally do not fit in cache, so per-token expert selection can stream a different weight subset while the CPU is simultaneously running tokenization, host orchestration, and page-cache activity on the same DRAM channels.

- Measure iGPU weight-streaming bandwidth in isolation, then again under representative concurrent CPU load (tokenizer, host process, filesystem cache pressure).
- Report the delta between isolated and concurrent bandwidth, not just the isolated figure.
- Repeat under thermally steady conditions, since contention effects can compound with throttling.

### 6.4 Measure bandwidth and dispatch

Measure:

- Sequential and randomized device reads.
- Device copy bandwidth.
- Representative model-weight streaming.
- Launch, barrier, event, immediate-list, and regular-list replay cost.
- Performance under different power profiles and steady-state temperatures.

The provisional 90 to 115 GB/s sustained range is a planning estimate only. The measured sustainable bandwidth becomes the roofline input.

### Exit criteria

- A reproducible toolchain document exists, backed by a retained package snapshot and a tested rollback procedure.
- Device capability and allocation reports pass.
- Unified-memory policy is selected from data.
- Warm and thermally steady bandwidth and dispatch baselines are stored.
- Concurrent CPU/GPU contention bandwidth is measured and stored alongside the isolated figures, with the delta reported explicitly.

## 7. Phase 2: Verified model manifest and memory plan

### 7.1 Generate the architecture manifest

Stream the pinned SafeTensors shards and record:

- Layer count and exact layer pattern.
- Hidden and intermediate dimensions.
- Attention and DeltaNet head counts and dimensions.
- Router tensor shapes.
- Total expert count and active experts per token.
- Shared and routed expert structure.
- Vocabulary and embedding configuration.
- MTP or auxiliary heads.
- RoPE and position-encoding parameters.
- Tokenizer assets, added tokens, and chat template.
- Complete tensor inventory, dtype, shape, byte size, and source shard.

No kernel or container design should depend on the provisional topology until this manifest is complete.

### 7.2 Build a measured memory budget

Account separately for:

- All expert weights, not only active expert traffic.
- Scales and alignment.
- Embeddings, routers, norms, attention, DeltaNet, output head, and auxiliary tensors.
- KV cache for full-attention layers only.
- DeltaNet recurrent and convolution state.
- Prefill and decode workspaces.
- Command lists, modules, runtime allocations, and driver overhead.
- Host process, filesystem cache, and other system-memory users.

The proposed 20.5 GB footprint and 10 GB margin are preliminary. On a 32 GB unified-memory machine, the OS, CPU process, GPU allocations, and page cache share the same physical pool. The plan must prove fit under realistic runtime load and retain a defined safety reserve.

### 7.3 Define context tiers

Use staged context targets:

1. 4K correctness tier.
2. 16K integration tier.
3. 32K memory and performance tier.
4. 64K stretch tier, enabled only if measured memory and latency are acceptable.

### Exit criteria

- The manifest replaces every architectural assumption.
- Exact model-memory scenarios are published.
- Allocation tests prove the release context tier fits without swap or destructive memory pressure.

## 8. Phase 3: `.binfer` MoE extension

### 8.1 Prefer metadata extension over incompatible entry growth

Do not immediately add MoE fields to every existing 192-byte tensor-directory entry. That would create a broad binary-format change and may waste space for non-expert tensors.

Preferred design:

- Keep the existing independently addressable tensor directory where possible.
- Introduce an architecture-specific MoE metadata section containing expert groups, expert IDs, shared experts, routing dimensions, and tensor-to-expert mappings.
- Add explicit expert-group layout identifiers.
- Bump the container major version only if backward-compatible use of reserved fields is insufficient.

### 8.2 Quantization policy

Start with the proven candidate:

- INT4 symmetric group size 128.
- BF16 scales.
- Higher precision for routers, norms, DeltaNet recurrent parameters, sensitive attention parameters, and any tensors rejected by validation.

Do not assume all expert matrices tolerate the dense-model quantization policy. Run per-expert and per-layer error analysis, including rarely selected experts.

### 8.3 Loader validation

Add rejection tests for:

- Missing expert metadata.
- Duplicate or out-of-range expert IDs.
- Invalid tensor-to-expert mappings.
- Router dimension mismatch.
- Unsupported active-expert count.
- Expert payload overlap or truncation.
- Wrong model revision or architecture.

### Exit criteria

- Conversion is deterministic.
- A full model round-trip validates.
- Malformed MoE metadata is rejected before allocation.
- Every tensor and expert mapping is checksum-verifiable.

## 9. Phase 4: Kernel and routing adaptation

### 9.1 Deterministic router

Implement and validate:

- Router logits.
- Numerically stable selection.
- Top-k expert IDs and weights.
- Tie-breaking.
- Shared-expert behavior, if present.
- Expert-output weighted accumulation.

Keep a host reference and a device implementation. Router results must be integer-exact for expert IDs under the approved tolerance policy.

### 9.2 Expert execution strategy

Benchmark at least three strategies:

1. Host-readback of selected experts followed by expert-list submission.
2. Device-side indirect dispatch.
3. A fixed recorded list containing guarded expert kernels.

Select based on end-to-end token latency, synchronization cost, command-list complexity, and determinism. Do not assume indirect dispatch is fastest or universally supported.

This decision is not independent of the Phase 1 concurrent-contention results (6.3.1) and the Phase 2 measured memory budget (7.2). If successive tokens tend to reselect overlapping experts, keeping "hot" experts resident becomes attractive, but only to the extent the measured memory budget leaves headroom beyond the worst-case allocation. Revisit the dispatch-strategy choice if either input changes materially after this phase begins.

### 9.3 INT4 expert GEMV

Use the B60 DP4A row-major kernel as the initial reference candidate, then retune:

- SIMD 16 versus SIMD 32.
- Rows or tiles per work item.
- Work-group size.
- Scale-loading strategy.
- Accumulator reduction.
- Expert-size-dependent occupancy.
- Weight locality when successive tokens choose the same experts.

Because all expert weights must reside in shared memory even when only a subset is active, distinguish capacity traffic from per-token active traffic in reports.

### 9.4 DeltaNet adaptation

Implement the verified target equations rather than adapting by head-count substitution alone. Validate:

- Convolution history.
- Gating and normalization.
- State update order.
- FP32 recurrent state.
- Chunk-prefill equivalence to recurrent decode.
- Cross-chunk continuity and reset.

### 9.5 Full attention and KV cache

Implement the verified full-attention layer count and dimensions, with:

- BF16 KV correctness baseline.
- INT8 KV as a separately quality-gated option.
- Boundary and maximum-context tests.
- Explicit context-dependent timing.

### Exit criteria

- Router, expert MLP, DeltaNet, attention, and sampling kernels match trusted references.
- Selected dispatch strategy is justified by measurement.
- Exact model shapes are covered by automated tests.

## 10. Phase 5: Unified single-process runtime

### 10.1 Static state and ownership

Create one process that:

1. Loads the model once.
2. Allocates model, KV, DeltaNet, activation, control, and sampling memory once.
3. Runs chunked prefill directly into decode-layout state.
4. Starts recorded decode at the final prefill position.
5. Resets state safely for the next request.

### 10.2 Command-list design

Record stable per-layer or layer-class structures only after routing and buffer addresses are finalized. MoE control values should live in fixed device-visible control buffers where possible.

Potential designs must be benchmarked. A nominal target of 40 layer lists is not a requirement if another grouping reduces routing or submission overhead.

### 10.3 Diagnostic cache format

Retain cache export/import only for debugging and long-running validation. The format must include:

- Version.
- Model and tokenizer identity.
- Context length and final position.
- KV and DeltaNet layout identifiers.
- Exact section lengths.
- Checksums.
- Atomic-write behavior and free-space precheck.

### Exit criteria

- No file handoff is used in the normal path.
- Short, multi-chunk, and maximum-tier prompts transition directly to decode.
- Ten repeated runs show deterministic reset and stable memory use.

## 11. Phase 6: Verification and quality qualification

### 11.1 Reference hierarchy

Capture:

- Operator references.
- One DeltaNet-MoE block.
- One full-attention-MoE block.
- Full-model short-prompt logits.
- Greedy completions.
- Router decisions and expert contributions.

The BF16 reference may require streamed shard loading because the full checkpoint may not fit comfortably in host memory.

### 11.2 Quality corpus

Use at least 200 deterministic cases:

| Category | Minimum cases | Metric |
|---|---:|---|
| Factual and instruction following | 40 | Exact or rubric score |
| Arithmetic and reasoning | 40 | Final-answer accuracy |
| Coding | 30 | Unit-test pass rate |
| Summarization and rewriting | 20 | Format and rubric score |
| English and Chinese | 30 | Quality and tokenizer parity |
| Long free generation | 20 | Repetition, EOS, finite logits |
| Long-context retrieval | 20 | Exact retrieval and distractor resistance |

Do not require token-for-token equality for every free-running 256-token completion. Small numerical differences can legitimately alter greedy trajectories. Instead, report:

- Teacher-forced top-1 agreement.
- BF16 margin at divergences.
- Top-k overlap.
- First free-generation divergence.
- Reconvergence.
- Final-answer or task-level acceptability.

Rubric-scored categories (factual/instruction-following, summarization and rewriting) are the ones most likely to drift in interpretation over a multi-month migration. Fix the rubric wording and scoring criteria once at corpus creation time, and re-score a small fixed subset from an earlier run whenever the corpus is reused, to confirm scores remain comparable across phases rather than reflecting a quietly shifted standard.

### 11.3 Tokenizer regression

Validate configured tokenization and templates for:

- Added special tokens.
- English and Chinese.
- Unicode and whitespace.
- Empty and system messages.
- Thinking modes.
- Encode/decode with and without skipped special tokens.

### 11.4 Long-context tests

At every enabled context tier, test:

- Multiple needle depths.
- Multiple needles.
- Distractors.
- Chunk-boundary needles.
- Beginning and end boundaries.
- DeltaNet continuity.
- BF16 and INT8 KV separately.

### Exit criteria

- All kernel and integration suites pass.
- The 200-case quality report is published.
- No unexplained high-margin regression remains.
- Long-context capability is enabled only for tiers that pass memory, correctness, quality, and latency gates.

## 12. Phase 7: Performance characterization and optimization

### 12.1 Benchmark contract

Report independently:

- Model load time.
- Tokenization time.
- Prefill time and tokens/s.
- First decode latency.
- Cold TTFT.
- Warm TTFT.
- Decode tokens/s.
- p50 and p95 inter-token latency.
- Peak total system-memory use.
- GPU allocation size.
- CPU utilization.
- Temperature, power mode, and throttling indicators.
- Router distribution and expert-cache locality.

### 12.2 Required prompt sizes

Benchmark prefill at 1, 16, 64, 256, 1K, 4K, 16K, and every enabled higher context tier. Benchmark decode at representative active contexts rather than publishing one context-free rate.

### 12.3 Roofline model

Compute the roofline from measured traffic:

```text
weights actually read for selected experts
+ shared dense weights
+ scales and routing metadata
+ KV traffic
+ DeltaNet state traffic
+ activation and reduction traffic
```

A ceiling based only on approximately 3B active parameters is optimistic if routing, shared layers, poor expert locality, cache-line overfetch, or redundant reads increase actual traffic. Publish both logical and measured bytes per token.

### 12.4 Thermal methodology

Report cold, warm, and thermally steady results. A thin Lunar Lake system may change clocks during sustained prefill or generation. Performance acceptance must use the thermally steady condition.

### Exit criteria

- A controlled comparison is published against an aligned llama.cpp or other trusted runtime.
- Quality-approved configurations are attached to every result.
- Measured values replace speculative 50 to 60 tokens/s planning estimates.

## 13. Persistent HTTP service

Implement only after the unified runtime is stable.

Required behavior:

- Resident model and command lists.
- `/healthz` and `/readyz`.
- Single active generation with bounded queue.
- Cancellation and timeout.
- Deterministic state reset.
- Device-failure detection and process restart policy.
- Structured timing and error logs.

### Exit criteria

- Ten sequential requests complete without reload, state leakage, or monotonic memory growth.
- Cancellation and timeout leave the next request functional.
- Cold and warm TTFT are reported separately.

## 14. Memory-safety and operational hardening

Adopt the lessons from the B60 implementation from the beginning:

- Use typed arena spans with base, byte size, storage type, shape, alignment, and identity.
- Centralize checked offset arithmetic.
- Validate runtime kernel argument counts.
- Use guards for partial tiles, head counts, expert IDs, and context limits.
- Add allocation canaries in validation builds.
- Run host-side ASan and UBSan tests.
- Fuzz `.binfer` metadata, MoE metadata, diagnostic cache headers, and CLI parsing.
- Inject full-disk, short-write, truncated-file, bad-checksum, invalid-context, and device-loss failures.
- Stop long-running jobs immediately after unrecoverable Level Zero failure.

## 15. MTP and speculative decoding

MTP remains deferred. Reopen only if:

1. The target checkpoint contains and exposes usable MTP weights.
2. Real usage includes enough long-context decode to justify it.
3. Chained drafting is implemented.
4. Acceptance is measured on the target model at target contexts.
5. Verification cost includes real MoE routing and attention traffic.
6. The expanded quality suite can detect speculative-verification regressions.

## 16. Milestones

| Milestone | Scope | Completion condition |
|---|---|---|
| M0: Migration contract | Phase 0 | Target identity and gates pinned |
| M1: Target platform ready | Phase 1 | Reproducible CachyOS/Lunar Lake baseline |
| M2: Model fits and loads | Phases 2 and 3 | Manifest, memory budget, conversion, and loader pass |
| M3: Kernels correct | Phase 4 | Router and both layer classes match references |
| M4: Unified runtime | Phase 5 | In-memory prefill-to-decode path passes |
| M5: Quality-qualified | Phase 6 | 200-case and context-tier gates pass |
| M6: Performance characterized | Phase 7 | Measured steady-state benchmark published |
| M7: Service candidate | Phase 13 | Persistent service and recovery tests pass |

## 17. Immediate next actions

1. Pin the exact target-model revision and download only metadata and tokenizer assets first.
2. Run the target hardware and Level Zero probes on CachyOS.
3. Generate the architecture and tensor manifest before designing MoE directory fields.
4. Measure practical allocatable unified memory and sustained bandwidth under thermal steady state.
5. Decide the first supported context tier from measured fit.
6. Define `.binfer` MoE metadata without unnecessarily expanding every tensor entry.
7. Build router and one-block CPU references before porting the B60 kernels.
8. Add explicit prefill timing fields to the benchmark schema.

## 18. Definition of done

The migration is complete when:

- The exact model and target platform are reproducible.
- The full model loads without unsafe memory pressure or inference-time allocation.
- Router decisions and both block classes match trusted references.
- Prefill and decode share in-memory state.
- The expanded quality suite passes.
- Supported context tiers pass memory, quality, and performance gates.
- Thermally steady prefill and decode performance are reported independently.
- The persistent service passes request-isolation and recovery tests.
