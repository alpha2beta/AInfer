# AInfer B60 Branch Review and Investigation Plan

**Review date:** 2026-09-20  
**Branch:** `B60`  
**Target model:** Qwen/Qwen3.8-27B, text-only  
**Target hardware:** Intel Arc Pro B60, 24 GB VRAM  
**Review purpose:** Determine whether further B60 development is justified, identify the remaining performance gap, and define a focused investigation program with measurable continuation and stop criteria.

## 1. Executive conclusion

The B60 branch should **continue selectively**, but it should not expand into a broad general-purpose serving framework until the prefill gap is understood and materially reduced.

The branch is already competitive in its strongest area:

- Default INT4 greedy decode is approximately 14.9 tok/s.
- Adaptive depth-2 MTP reaches approximately 19 to 25.6 tok/s.
- The reported default output is token-identical to the streamed CPU BF16 reference.
- The MTP path is reported as bitwise identical.
- The 200-case quality suite found no quantization-attributable grade flips.
- INT8 KV produced no reported quality delta.
- The 64K path completes in one binary and passes 20/20 needle-retrieval cases.
- The B60 CTest preset is green.

The decisive weakness is prefill:

- AInfer reports approximately 30 tok/s for its current prefill path.
- The README cites an external OpenVINO GenAI INT4 result around 1046 tok/s.
- The 64K workflow takes approximately 80 minutes, corresponding to roughly 13.7 effective input tok/s if the full 65,536-token interval is included.

Therefore, the project is not generally uncompetitive. It is **decode-competitive but prefill-limited**. The next B60 phase should concentrate on controlled prefill measurement, DPAS/XMX GEMM feasibility, attention and DeltaNet profiling, and long-context scaling. Generic serving work should remain secondary.

## 2. Current verified branch status

### 2.1 Functional status

The current branch implements:

- Deterministic symmetric INT4 group-128 conversion.
- A versioned `.binfer` model container.
- Static Level Zero allocations.
- Recorded-loop decode using 66 command lists.
- The 48 linear-attention plus 16 full-attention text-layer topology.
- Greedy decode and configurable sampling.
- Adaptive MTP speculative decoding.
- BF16 and INT8 KV configurations.
- Single-binary chunked prefill-to-decode execution.
- A 64K context path.
- Streamed CPU BF16/INT4 references.
- A 200-case quality suite.
- Tokenizer handling with registered added tokens and a pinned golden reference.

### 2.2 Current headline performance

| Path | Reported result | Interpretation |
|---|---:|---|
| Default INT4 greedy decode | ~14.9 tok/s | Near the cited llama.cpp SYCL range |
| Adaptive depth-2 MTP | 19 to 25.6 tok/s | Competitive with the cited OpenVINO decode result; below the cited llama.cpp draft-MTP peak |
| General prefill | ~30 tok/s | Main unresolved performance gap |
| 64K single-binary path | ~80 minutes | Functionally successful but operationally slow |
| 64K retrieval | 20/20 | Strong correctness evidence |
| Quality suite | 200 cases | No reported quantization-attributable grade flips |
| INT8 KV | No reported quality delta | Promising, but context-stratified evidence should remain visible |
| Full test suite | Green | Current regression baseline |

### 2.3 External comparisons currently cited by the branch

The README cites:

- llama.cpp SYCL at approximately 14.9 to 17 tok/s for non-speculative decode.
- llama.cpp FP16 plus draft-MTP at approximately 29.6 tok/s.
- OpenVINO GenAI INT4 at approximately 23.3 tok/s decode.
- OpenVINO GenAI INT4 at approximately 1046 tok/s prefill.

These comparisons are useful indicators, but they are not yet sufficient for a final performance verdict. The branch needs one locally reproducible comparison protocol controlling model revision, quantization quality, KV precision, actual context occupancy, prompt template, MTP settings, warm state, power conditions, and timing boundaries.

## 3. Main findings

## 3.1 Decode is no longer the primary reason to question continuation

Default decode approximately matches the cited llama.cpp SYCL result, while MTP brings AInfer into the cited OpenVINO decode range. The remaining decode gap to the cited llama.cpp draft-MTP result is meaningful but not large enough to justify abandoning the project.

Further decode work should be limited to evidence-driven opportunities:

- Improve adaptive MTP acceptance and depth selection.
- Reduce verification overhead.
- Confirm whether the 19 to 25.6 tok/s spread is explained by acceptance rate, prompt class, context length, or kernel timing.
- Measure decode at actual occupied contexts, including 4K, 16K, and 64K.
- Separate target-model evaluations per second from accepted output tokens per second.

Decode optimization should not displace the prefill investigation unless profiling shows a newly dominant decode bottleneck.

## 3.2 Prefill is the critical technical and strategic gap

The current approximately 30 tok/s prefill result is far below the cited OpenVINO result. Before assuming a kernel deficiency of more than 30 times, the measurement must be decomposed.

Potential contributors include:

1. The current AInfer result may include setup, cache initialization, command recording, transfers, validation, or first-decode work.
2. The external result may use a longer prompt and therefore obtain much greater GEMM efficiency.
3. OpenVINO may use XMX/DPAS GEMM, fused attention, optimized quantized layouts, graph fusion, and larger prefill batches.
4. AInfer may still execute projection work using GEMV-like or insufficiently tiled kernels.
5. DeltaNet recurrence imposes a sequential component that limits the achievable end-to-end gain from GEMM alone.
6. Full-attention layers become increasingly important at long context.
7. The 64K path may use a different chunk size or kernel mix from the approximately 30 tok/s headline result.

The immediate objective is therefore not “match 1046 tok/s.” It is to build a reliable phase-level profile and determine the achievable ceiling for this exact model and B60 implementation.

## 3.3 The 64K result proves correctness, not practical long-context performance

Completing 64K prefill in approximately 80 minutes while retrieving 20/20 needles is a strong functional accomplishment. It demonstrates state continuity, position handling, cache management, and retrieval correctness.

However, an 80-minute prefill is not practical for interactive serving. If the full 65,536 tokens are included, the effective rate is approximately:

```text
65,536 tokens / 4,800 seconds = 13.65 tokens/s
```

The project should report short/medium prefill and 64K effective prefill separately. It should also identify whether the 80-minute figure includes:

- Model loading
- Tokenization
- Cache allocation or initialization
- Prompt encoding
- Prefill only
- Cache validation or export
- First decode
- Needle evaluation

## 3.4 Quality evidence is a major reason to preserve the branch

The 200-case suite, BF16 reference path, MTP bitwise identity, INT8-KV comparison, and 20/20 needle result make the B60 branch valuable even if a mature framework remains faster.

The branch is a useful controlled platform for:

- INT4 quantization analysis
- Static Level Zero execution
- Hybrid linear/full-attention models
- MTP acceptance experiments
- Long-context state validation
- Intel GPU kernel prototyping

This supports continuation as a research and verification runtime even if the project does not become the preferred production server.

## 3.5 Generic serving development has lower expected return

The README indicates that the persistent HTTP worker remains open. Building it would improve usability and eliminate repeated model loading, but it would not solve the core prefill gap.

OpenVINO and llama.cpp already provide mature serving functionality. Until AInfer demonstrates a distinct performance, memory, determinism, or research advantage, B60 development should avoid duplicating broad infrastructure such as:

- Continuous batching
- Multi-user scheduling
- Full OpenAI API parity
- General model support
- General-purpose prompt caching
- Broad deployment orchestration

A minimal resident worker may still be justified for accurate warm benchmarking and internal use. It should be treated as a measurement and integration tool, not as the main development objective.

## 4. Investigation program

## Phase B60-I0: Evidence normalization

**Objective:** establish one reproducible performance and quality baseline.

### Actions

- Create a B60 benchmark manifest containing:
  - Git commit
  - `.binfer` SHA-256
  - tokenizer hashes
  - driver, Level Zero, IGC, and oneAPI versions
  - GPU power and clock settings
  - model revision
  - KV precision
  - MTP settings
  - prompt length and actual context position
  - chunk size
  - warm/cold state
  - exact timing boundaries
- Classify every external comparison as:
  - reproduced locally on the same B60
  - same GPU/model class but externally reported
  - different quantization or checkpoint
  - unverified reference
- Add strict machine-readable benchmark output.

### Required timing fields

```json
{
  "model_load_seconds": 0.0,
  "tokenization_seconds": 0.0,
  "cache_initialization_seconds": 0.0,
  "prefill_seconds": 0.0,
  "prompt_tokens": 0,
  "prefill_tokens_per_second": 0.0,
  "first_decode_seconds": 0.0,
  "cold_ttft_seconds": 0.0,
  "warm_ttft_seconds": 0.0,
  "decode_tokens_per_second": 0.0,
  "accepted_tokens_per_second": 0.0,
  "mtp_acceptance_rate": 0.0
}
```

### Exit criteria

- Every headline number has a reproducible command and evidence file.
- Prefill excludes model loading and tokenization unless explicitly labeled end-to-end.
- MTP accepted-token throughput is not confused with target-model evaluation rate.

## Phase B60-I1: Prefill scaling sweep

**Objective:** determine where AInfer transitions from fixed-overhead-limited to kernel-limited execution.

### Prompt lengths

Run at least:

```text
P = 1, 16, 64, 128, 256, 512, 1024, 4096, 16384, 65536
```

For chunk-sensitive paths, also include boundaries around the active chunk size `C`:

```text
C - 1, C, C + 1, 2C - 1, 2C, 2C + 1
```

### Report

For each prompt length, capture:

- Median and p95 prefill time
- Tokens/s
- Chunk count
- Peak VRAM
- Kernel-time breakdown
- Command submission and synchronization time
- Full-attention versus linear-attention time
- Projection versus recurrent-state time
- Effective bandwidth
- Temperature and steady-state clock

### Exit criteria

- The approximately 30 tok/s figure is tied to a defined prompt size and path.
- The approximately 80-minute 64K result is decomposed.
- The dominant bottleneck is known at short, medium, and long contexts.

## Phase B60-I2: Projection and DPAS/XMX feasibility

**Objective:** determine whether projection kernels can move from the current path to XMX/DPAS without unacceptable packing, memory, or quality costs.

### Investigation tasks

1. Inventory all dominant projection shapes by layer class.
2. Record their share of total prefill time.
3. Prototype DPAS tiling for the highest-contribution shape.
4. Confirm emitted DPAS instructions through compiler output or disassembly.
5. Compare:
   - current kernel
   - DPAS with offline packed weights
   - any required fallback for small or misaligned shapes
6. Measure:
   - kernel speedup
   - end-to-end prefill improvement
   - packing expansion
   - scale and layout overhead
   - register pressure and spills
   - occupancy
   - numerical error

### Go criteria

Proceed to broader integration if the prototype achieves:

- At least 1.8× speedup on the dominant projection shape
- At least 15% end-to-end prefill improvement
- No runtime repacking of static weights
- No quality-gate regression
- Acceptable container-size and VRAM overhead

### No-go criteria

Defer broad DPAS adoption if:

- End-to-end gain is below 10%
- Packing and layout conversion dominate
- Kernel gains disappear at real prompt sizes
- Compilation is unreliable
- Memory overhead harms the 64K path

## Phase B60-I3: DeltaNet recurrence optimization

**Objective:** quantify and reduce the sequential floor that projection acceleration cannot remove.

### Actions

- Measure recurrence share at every prompt length.
- Verify whether state updates serialize across tokens, heads, or channels more than required.
- Investigate:
  - head/channel parallelism
  - vector width
  - state-layout locality
  - convolution-state fusion
  - reduced launch count
  - chunk-level scan or associative reformulation where mathematically valid
  - FP32 state traffic
- Preserve exact state continuity tests across chunk boundaries.

### Exit criteria

- A recurrence roofline and sequential-time floor are documented.
- Any retained optimization passes block, chunk-boundary, 64K, and quality tests.

## Phase B60-I4: Full-attention and long-context optimization

**Objective:** reduce context-dependent cost in the 16 full-attention layers.

### Actions

- Establish separate BF16-KV and INT8-KV profiles.
- Evaluate the experimental fused-attention path behind `AINFER_FLASH=1`.
- Measure context-dependent scaling at 4K, 16K, 32K, and 64K.
- Investigate:
  - quantized-KV XMX/DPAS processing
  - fused QK, masking, softmax, and V accumulation
  - tiled cache reads
  - reduced intermediate materialization
  - improved command-list grouping
- Keep exact context-boundary, maximum-context, and retrieval tests.

### Exit criteria

- Long-context attention cost is isolated from DeltaNet and projection cost.
- The selected path provides a stable end-to-end gain at 16K or above.
- INT8-KV quality remains qualified by context tier.

## Phase B60-I5: MTP optimization

**Objective:** reduce the gap to the cited 29.6 tok/s llama.cpp draft-MTP result while preserving deterministic verification.

### Actions

- Report MTP results by:
  - prompt category
  - active context
  - depth
  - acceptance rate
  - verification cost
- Tune adaptive depth policy.
- Identify prompts where depth-2 loses to depth-1.
- Separate raw target evaluations/s from accepted output tok/s.
- Investigate whether prefill or cache layout changes affect MTP verification cost.

### Exit criteria

- The source of the 19 to 25.6 tok/s range is explained.
- Adaptive MTP never materially regresses below the default path on the qualified corpus.
- Output identity remains preserved under the approved deterministic configuration.

## Phase B60-I6: Controlled framework comparison

**Objective:** determine whether AInfer has a defensible advantage or should remain research-only.

### Required configurations

Run AInfer, current llama.cpp SYCL, and OpenVINO GenAI on the same B60 where possible.

Control:

- Exact model revision
- Prompt template and thinking mode
- Quantization quality
- KV precision
- Actual occupied context
- Prompt and generation lengths
- Speculative decoding state
- Warm runtime state
- Power limit and thermal state
- Output-quality acceptance

### Required benchmark matrix

| Area | Cases |
|---|---|
| Prefill | P64, P256, P1K, P4K, P16K, P64K |
| Decode | Active contexts 128, 4K, 16K, 64K |
| Generation | At least 128 output tokens |
| MTP | Off, depth-1, adaptive depth-2 |
| KV | BF16 and INT8 where supported |
| Runs | At least five; median and p95 |

### Exit criteria

- The comparison is reproducible.
- Quality and precision differences are disclosed.
- Performance gaps are attributed to measured phases rather than headline throughput alone.

## 5. Development priorities

| Priority | Work item | Rationale |
|---|---|---|
| P0 | Normalize benchmark evidence | Current comparisons are not sufficiently aligned |
| P0 | Prefill scaling and phase profile | Determines whether the gap is measurement, GEMM, recurrence, or attention |
| P1 | DPAS/XMX projection prototype | Largest likely prefill acceleration opportunity |
| P1 | DeltaNet recurrence study | Establishes the sequential floor |
| P1 | Long-context attention optimization | Necessary to reduce the 80-minute 64K path |
| P1 | Controlled OpenVINO/llama.cpp comparison | Required for the continuation decision |
| P2 | MTP policy tuning | Decode is already competitive |
| P2 | Minimal resident worker | Useful for warm benchmarks and repeated requests |
| P2 | Typed spans and checked offsets | Important reliability hardening |
| P3 | Broad serving features | Low return until core performance is resolved |

## 6. Continuation and stop criteria

## 6.1 Continue as a performance-oriented runtime if

AInfer achieves at least one of the following after the focused investigation:

- Prefill improves by at least 2× over the current approximately 30 tok/s path.
- 64K end-to-end prefill time decreases by at least 30%.
- Decode remains within 15% of the best aligned non-speculative comparison.
- MTP remains within 20% of the best aligned speculative comparison.
- AInfer demonstrates a measurable advantage in memory use, determinism, quality traceability, long-context stability, or batch-one tail latency.

## 6.2 Continue as a research runtime if

Performance remains behind but the branch continues to provide unique value for:

- Level Zero recorded execution
- Intel kernel experimentation
- DPAS and quantized-layout research
- MTP verification
- DeltaNet state analysis
- Long-context correctness and cache investigation

In this case, explicitly describe the branch as:

> A model-specific Intel GPU kernel and runtime research platform, not a general-purpose production inference server.

## 6.3 Freeze standalone development if

After the time-boxed optimization phase:

- Prefill remains more than 2× slower than an aligned OpenVINO or llama.cpp result.
- Decode becomes more than 2× slower under the same quality and generation mode.
- No unique memory, correctness, determinism, or long-context advantage is demonstrated.
- Most new work duplicates generic serving features already available elsewhere.
- DPAS, attention, and recurrence investigations all fail their go criteria.

A frozen branch should retain reproducible builds, quality evidence, benchmark reports, and research documentation.

## 7. Recommended next execution batch

Start the following work before full serving development:

1. **B60-R1:** Create the normalized benchmark manifest and timing schema.
2. **B60-R2:** Run the P1-to-P64K prefill scaling sweep.
3. **B60-R3:** Generate a phase-level prefill profile at P256, P4K, and P64K.
4. **B60-R4:** Reproduce current llama.cpp SYCL and OpenVINO results locally on the same B60.
5. **B60-R5:** Prototype one DPAS/XMX projection shape using offline packed weights.
6. **B60-R6:** Profile DeltaNet recurrence and establish the sequential floor.
7. **B60-R7:** Profile and qualify `AINFER_FLASH=1` at long context.
8. **B60-R8:** Publish a continuation decision report.

These tasks can partially overlap, but B60-R1 must define the reporting contract before benchmark results are accepted.

## 8. Suggested task definitions

### B60-R1: Benchmark evidence normalization

- **Priority:** P0
- **Deliverables:** benchmark schema, environment manifest, commands
- **Done when:** every headline metric is reproducible and phase boundaries are explicit

### B60-R2: Prefill scaling sweep

- **Priority:** P0
- **Dependencies:** B60-R1
- **Deliverables:** scaling JSON and Markdown summary
- **Done when:** P1 through P64K are measured and the 30 tok/s figure is fully defined

### B60-R3: Phase-level prefill profile

- **Priority:** P0
- **Dependencies:** B60-R1, B60-R2
- **Deliverables:** kernel and phase timing report
- **Done when:** projection, recurrence, attention, synchronization, and other costs sum to end-to-end time

### B60-R4: Controlled baseline reproduction

- **Priority:** P1
- **Dependencies:** B60-R1
- **Deliverables:** AInfer/llama.cpp/OpenVINO comparison
- **Done when:** the same machine, model semantics, context, and quality controls are used

### B60-R5: DPAS projection feasibility

- **Priority:** P1
- **Dependencies:** B60-R3
- **Deliverables:** prototype kernel, packed layout, A/B report
- **Done when:** go/no-go criteria are evaluated on a dominant real shape

### B60-R6: DeltaNet recurrence study

- **Priority:** P1
- **Dependencies:** B60-R3
- **Deliverables:** recurrence roofline and candidate experiments
- **Done when:** the sequential floor and retained improvements are documented

### B60-R7: Long-context attention investigation

- **Priority:** P1
- **Dependencies:** B60-R2, B60-R3
- **Deliverables:** BF16/INT8 KV and fused-attention report
- **Done when:** 4K-to-64K scaling and the experimental path's decision are documented

### B60-R8: Continuation decision

- **Priority:** P0
- **Dependencies:** B60-R4, B60-R5, B60-R6, B60-R7
- **Deliverables:** `B60_decision.md`
- **Done when:** continue, research-only, or freeze is selected using the criteria in Section 6

## 9. Documentation improvements

Update `README_B60.md` so each headline result includes or links to:

- Prompt length
- Actual occupied context
- Generated-token count
- Chunk size
- KV format
- MTP mode and acceptance
- Warm/cold status
- Inclusion or exclusion of model loading
- Median/best/p95 label
- Power and thermal conditions
- Evidence filename

Recommended presentation:

```markdown
| Metric | Result | Configuration | Evidence |
|---|---:|---|---|
| Warm prefill | ... tok/s | P=..., CHUNK_M=..., KV=..., MTP=off | reports/...json |
| Decode | ... tok/s | active_ctx=..., G=..., KV=... | reports/...json |
| MTP accepted output | ... tok/s | depth=..., acceptance=... | reports/...json |
```

Also distinguish clearly between:

- Current verified truth in `STATUS.md`
- Historical findings in `progress.md`
- Planned work in `tasks.md`
- This investigation recommendation in `review_B60.md`

## 10. Final recommendation

Continue the B60 branch for a bounded investigation phase focused on prefill and long-context performance. Do not treat the branch as generally behind: its default decode is near the cited llama.cpp range, its MTP path overlaps the cited OpenVINO decode result, and its quality and 64K correctness evidence are strong.

The critical question is narrower:

> Can DPAS/XMX projection work, DeltaNet recurrence optimization, and improved long-context attention reduce the prefill gap enough to justify AInfer as more than a research runtime?

Answer that question before investing significantly in broad serving infrastructure. If the focused program delivers a substantial prefill or long-context gain, continue toward a minimal hardened resident service. If it does not, preserve B60 as a reproducible Intel GPU kernel and runtime research platform and use OpenVINO or llama.cpp for production serving.
