# AInfer Migration Scope & Acceptance Specification

**Document Version:** 1.0  
**Target Platform:** Intel Core Ultra 7 258V (Intel Arc 140V integrated GPU, 32 GB LPDDR5X-8533)  
**Target OS:** CachyOS (Linux rolling release, optimized kernel)  
**Target Model:** `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized` (Qwen3.5-MoE architecture fine-tune; GGUF reference `LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF`)  
**Date:** 2026-09-17  

---

## 1. Project Scope & Operational Boundaries

AInfer is a hardware- and model-specialized inference engine designed to achieve maximum efficiency and predictability on pinned hardware. It is intentionally not a generic multi-model runtime.

### 1.1 In-Scope Capabilities
- Autoregressive text generation with batch size 1.
- Deterministic offline quantization: INT4 symmetric group-128 weights with BF16 scales.
- Extended `.binfer` container format supporting sparse MoE metadata and expert tensor addressing.
- Single-process runtime architecture: chunked prefill directly initializes in-memory KV cache and DeltaNet recurrent state without intermediate serialization.
- Recorded Level Zero command lists for steady-state decode execution.
- Dynamic per-step parameter passing via pinned host-visible / shared device control buffers.
- Sampling modes: greedy decode (deterministic baseline), temperature, top-$k$, and top-$p$.
- Staged context lengths up to 32K (and 64K stretch).
- Independent, reproducible performance profiling reporting isolated prefill, TTFT, and decode latencies under thermally steady conditions.

### 1.2 Out-of-Scope (Explicitly Deferred or Dropped)
- **Vision Encoder:** Qwen multimodal vision modules are explicitly deferred. The v1 runtime is strictly text-only.
- **Multi-Token Prediction (MTP):** Speculative decoding via MTP heads is deferred until verified checkpoint assets and positive traffic-vs-latency benchmarks justify it.
- **Continuous / Multi-Batching:** The runtime is strictly specialized for single-user batch 1 generation.
- **General-Purpose GPU Portability:** Code paths are specialized for Intel Xe2 architecture and oneAPI Level Zero.
- **Arbitrary Dynamic Topologies:** Tensor shapes, layer sequences, and memory layouts are statically bound at initialization.

---

## 2. Staged Context Tiers

Memory capacity on a 32 GB unified-memory platform is shared across the operating system, CPU background processes, iGPU weight storage, activation workspaces, and state caches. Release qualification is structured in staged tiers:

| Tier | Context Length | Role & Gate Purpose | Memory Allowance | Latency / TTFT Target |
|---|---:|---|---|---|
| **Tier 1** | 4,096 tokens (4K) | Correctness baseline, unit verification, and teacher-forced regression testing | $\le 21.0$ GB total | Cold TTFT $\le 1.5$ s; Warm TTFT $\le 0.5$ s |
| **Tier 2** | 16,384 tokens (16K) | Multi-chunk prefill validation and state continuity testing | $\le 22.5$ GB total | TTFT $\le 4.0$ s |
| **Tier 3** | 32,768 tokens (32K) | Performance & memory qualification tier for long conversations | $\le 24.5$ GB total | TTFT $\le 10.0$ s; zero page-thrashing |
| **Tier 4** | 65,536 tokens (64K) | Stretch tier. Enabled *only* if measured memory margins and latency pass qualification | $\le 26.0$ GB total | TTFT $\le 25.0$ s; strictly gated on $\ge 6$ GB RAM reserve |

---

## 3. Acceptance Gates & Verification Criteria

The migration must satisfy four rigorous acceptance gates before any production release:

### 3.1 Numerical & Operator Correctness Gate
- **Router Parity:** Device router top-$k$ expert selection must be integer-exact compared to the CPU reference implementation.
- **Single-Block Parity:** Complete forward pass of single DeltaNet-MoE and Full-Attention-MoE blocks on Arc 140V must agree with CPU BF16 forward reference within relative error $\le 10^{-4}$ for INT4-quantized layers and $\le 10^{-6}$ for BF16 norm/attention layers.
- **Teacher-Forced Agreement:** On a 10-prompt test set (64 positions each), INT4 device execution must achieve $\ge 85\%$ teacher-forced top-1 logit agreement against unquantized BF16 reference.

### 3.2 Broad Quality Qualification Gate
- **200-Case Evaluation Corpus:** A deterministic test suite spanning 7 categories:
  1. Factual & instruction following (40 cases)
  2. Arithmetic & multi-step reasoning (40 cases)
  3. Coding & syntax generation (30 cases)
  4. Summarization & extraction (20 cases)
  5. Multilingual English & Chinese (30 cases)
  6. Long free generation (20 cases)
  7. Long-context needle retrieval (20 cases)
- **Acceptance Criterion:** The 200-case suite must show zero catastrophic failures (no repetitive looping, no NaN/Inf logits, no premature EOS on reasoning tasks) and achieve task-level accuracy within $3\%$ of reference INT4 baseline models.
- **Long-Context Retrieval:** Needle retrieval across 5 depths ($0\%, 25\%, 50\%, 75\%, 100\%$) must achieve $100\%$ accuracy on all qualified context tiers.

### 3.3 Memory & System Safety Gate
- **Physical Headroom:** Under peak steady-state allocation for the qualified context tier, total system memory usage (OS + runtime + allocations + buffers) must leave at least **6.0 GB of free/available RAM** on the 32 GB machine.
- **No Swap Thrashing:** Execution must run with zero pages swapped out to disk and zero memory-pressure kernel interventions.
- **Leak-Free Certification:** Ten consecutive generation cycles must demonstrate zero net memory growth and deterministic state cleanup.

### 3.4 Performance Characterization Gate
- **Independent Benchmark Metrics:** Performance reports must separate:
  - Model load time (seconds)
  - Tokenization time (milliseconds)
  - Prefill time (seconds) and prefill throughput (tokens/second)
  - First-decode latency / Cold and Warm TTFT (milliseconds)
  - Sustained decode throughput (tokens/second) and inter-token jitter ($p50, p95$)
- **Thermal Steady State:** Accepted performance metrics must be collected under steady-state operating temperatures ($\ge 5$ minutes of sustained workload) to prevent transient clock-boost skew.
- **Measured Traffic Roofline:** Reported hardware efficiency must account for all actual bytes read over system memory buses (active expert weights, shared weights, routing metadata, KV traffic, and activations).
