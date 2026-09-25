# AInfer (258v branch) — Qwen3.5-MoE Inference Runtime on Intel Core Ultra 7 258V

A custom, bare-metal Level Zero inference runtime engineered specifically for Intel Lunar Lake architecture (**Intel Core Ultra 7 258V**, integrated Arc 140V Xe2 GPU, 32 GB unified LPDDR5X-8533 memory) running **CachyOS**.

Pinned target model: **[`symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized`](https://huggingface.co/symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized)** (dequantized SafeTensors release of `LuffyTheFox/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF` fine-tune; base architecture `Qwen3.5-MoE` / `model_type: qwen3_5_moe`). Sparse hybrid MoE: 40 text layers (30 DeltaNet linear attention + 10 full attention), 256 routed experts / 8 active per token, ~35.95B total / ~3B active parameters per token, batch size 1, text-only.

Quantization format: INT4 symmetric group-128 weights, BF16 scales, high-precision router gates and RMSNorm (`.binfer` container).

---

## 1. Measured Performance & Status (Release Truth)

Authoritative metrics on physical Intel Arc 140V Xe2 hardware (see [`STATUS.md`](STATUS.md) and [`improvement.md`](improvement.md)):

| Workload / Metric | AInfer Measured Result | Notes / Comparison |
|---|---|---|
| **Greedy Decode (Short Context)** | **35.91 tok/s** (up to 36.08 tok/s) | +22.4% vs llama.cpp Vulkan (29.33 tok/s); 3.60× vs 8-thread CPU |
| **MTP Speculative Decode** | **45.62 tok/s** mean (up to **51.72 tok/s**) | 1.44× peak speedup (93.8% acceptance), 100% bit-exact parity (160/160 tokens) |
| **Warm TTFT ($P=21$)** | **178.43 ms** (cold 185.03 ms) | Jitter p50 = 27.75 ms, stddev = 0.46 ms |
| **Prefill Scaling ($P=128$)** | **347.30 tok/s** | Chunked parallel DeltaNet associative scan ($C=16$) |
| **Prefill Scaling ($P=256$)** | **414.37 tok/s** | Macro-chunk batched DPAS prefill |
| **Prefill Scaling ($P=512$)** | **424.49 tok/s** | High arithmetic intensity saturation |
| **Prefill Scaling ($P=1024$)** | **388.43 tok/s** | Blocked FlashAttention query tiling ($B=8, T=16$) |
| **Long-Context Prefill ($P=6,720$)** | **174.68 tok/s** (warmup **180.07 tok/s**) | **0.95× parity with llama.cpp** (183.2 tok/s); total time 38.5 s (cut from ~80 s) |
| **Long-Context Decode ($P=6,720$)** | **18.00 tok/s** (KV8 split-T) / **17.70 tok/s** (BF16) | **1.49× speedup** over 12.27 tok/s baseline via multi-head EU saturation (`AINFER_ATTN_SPLIT=4`) |
| **Fast Cold Startup** | **10.32 s** (`--fast-load`) | 4.65× faster than baseline 48 s; verified stamp bypasses redundant CRC scan |
| **Unified Memory Footprint** | **18.18–18.22 GiB committed** | Leaves **>13.8 GiB headroom** on 32 GB RAM; 0 KB RSS leak over 10 runs |
| **Quality & Parity Gates** | **Gates M0–M8 Signed Off** | Gate M4: 7/7 test suites passed bit-exact; CTest: 6/6 green (2.02 s) |

---

## 2. Core Architecture & Key Innovations

- **Unified Single-Process Resident Runtime**: Entire 35.95B model weights reside permanently in unified memory. In-memory prefill seamlessly transitions into decode without disk reloads or PCIe transfers.
- **Zero-Allocation Recorded Command Lists**: 40 layer command lists, embed list, and tail list are recorded once at startup into Level Zero command lists. A 128-byte `RuntimeControl` buffer on device drives positions, expert routing, and dynamic loop control without command list rebuilding.
- **Chunked Parallel DeltaNet Scan (I2.3)**: Replaces serial recurrent loops with a chunk-parallel associative scan ($C=16$ chunk tile). Intra-chunk triangular solve $M D = V - V_{init}$ executes in SLM across 128 threads, cutting recurrent dependency barriers by 16× and lifting prefill throughput across all prompt lengths.
- **Blocked FlashAttention for Prefill (I3.7)**: Resolves the $O(B \times T)$ memory bandwidth bottleneck when the KV cache exceeds the 8 MB GPU L2 cache. Processes $B_{\text{tile}}=8$ queries concurrently per workgroup, sharing an SLM tile of $T_{\text{tile}}=16$ key/value tokens with aligned 16-byte vector loads (`ushort8`) and in-register online softmax.
- **SIMD16 Subgroup Butterfly Decode Attention (I3.6)**: Replaces 3-barrier SLM tree reductions with register-only butterfly shuffle reductions (`intel_sub_group_shuffle`) and double-buffered SLM staging, reducing synchronization barrier frequency by 48×.
- **Split-T Multi-Head Decode Attention (I3.8)**: Solves the hardware under-saturation bottleneck where 16 query heads utilized only 16 of Arc 140V's 64 EUs. Partitions each head's context across $S=4$ workgroups (64 workgroups total), computing partial online softmax states in parallel and combining them with `gqa_attn_combine`. Cuts 6.7K attention latency by ~2× (6.78 ms → 3.41 ms) and lifts decode to **18.00 tok/s** (KV8) / **17.70 tok/s** (BF16) (`AINFER_ATTN_SPLIT=4`).
- **Pure-FP32 MTP Dual-Token Verification (I3.1)**: Coalesced 16-byte vector GEMV (`int4_gemv_m2`) processes dual-token speculative verification in a single weight pass, maintaining 100% bit-exact mathematical parity with autoregressive greedy decode while reaching up to 51.7 tok/s.
- **Consolidated MoE Micro-GEMM Characterization (I3.2)**: Evaluated active-expert compaction vs fixed-grid dispatch across 256 routed experts (8 active). Proved 100% bit-exact mathematical parity (`0.00e+00` diff) and an 85.2% reduction in dispatch workgroups (2048 → 304). Characterized Intel Xe2 hardware thread dispatch behavior: empty workgroups terminate in < 1 clock cycle (< 0.05 ms overhead across 1,744 idle workgroups), proving fixed-grid dispatch optimal for production.
- **Dynamic Auto-KV8 Policy**: Production default is high-precision BF16 KV cache for short/medium context; Auto-KV8 automatically engages when `max_ctx >= 16384` to halve KV read traffic where attention memory bandwidth dominates.
- **Instant Client Disconnect Abort (I2.1)**: Socket status polling via non-blocking `select` and peek `recv` terminates GPU execution within a single decode step of client disconnect and resets Level Zero state.
- **Native FIM & Streaming Stop Buffering (I1.1, I1.2)**: Full OpenAI API compatibility for IDE tab autocomplete (`/v1/completions` with suffix) using native Qwen FIM tokens and prefix-buffering multi-token stop sequences.

---

## 3. Repository Structure

```
AInfer/
├── models/                     # Model metadata, tokenizers, and .binfer containers
├── tools/
│   ├── kernels_258v/           # OpenCL SPIR-V kernels (all_kernels.cl), microbenchmarks
│   ├── decode/                 # C++ Level Zero runtime (runtime_258v.cpp), C API, Gate M4 tests
│   ├── bench_258v/             # Standardized T7.1 benchmark, prefill sweeps, shootout harnesses
│   ├── http/                   # Resident HTTP server (server_258v.py), LAN launcher, systemd unit
│   ├── mtp/                    # Multi-token prediction speculative decoding harnesses and tests
│   ├── quality_258v/           # Tokenizer parity, 200-case quality corpus, teacher-forced eval
│   ├── membench/               # Memory allocation policy & CPU/GPU contention benchmarks
│   ├── toolchain/              # Pinned CachyOS package cache, sysroot, rollback automation (T1.2)
│   └── fuzz/                   # Robustness, fault-injection, and differential fuzzing suites
├── STATUS.md                   # Single-source authoritative release truth
├── improvement.md              # Optimization roadmap & completed P0/P1/P2 task catalog
├── claim_correction.md         # Long-context benchmark methodology & comparison with llama.cpp
├── progress.md                 # Engineering trajectory and detailed changelog
├── plan.md / tasks.md          # Architectural plan and stable task definitions (T0.1–X2)
└── memory_feasibility_estimate.md # 32 GB unified memory budget analysis
```

---

## 4. Building and Testing

### Prerequisites
- OS: CachyOS (or Arch Linux) on Intel Core Ultra 7 258V.
- Dependencies: Pinned Level Zero loader (`libze_loader.so.1`), Intel Compute Runtime, GCC 15+, CMake 3.25+, Python 3.11+ venv.
- Sysroot: Local packages are cached in `tools/toolchain/cache/` and extracted to `tools/toolchain/sysroot/`.

### Build
```bash
# Configure using the 258v CMake preset
cmake --preset 258v

# Build runtime components and tests
cmake --build --preset 258v
```

### Automated Regression Suite
```bash
# Run the 6 automated unit & regression tests
ctest --preset 258v
```
*Current status: 6/6 tests passing (100% green) in ~2.0 seconds.*

### Gate M4 On-Device Runtime Qualification
```bash
# Rebuild and run the full on-device Gate M4 verification suite
g++ -O3 -std=c++17 tools/decode/runtime_258v.cpp tools/decode/test_runtime_258v.cpp \
  -Itools/l0probe/include -Ltools/toolchain/sysroot/usr/lib -lze_loader \
  -o tools/decode/test_runtime_258v

LD_LIBRARY_PATH=tools/toolchain/sysroot/usr/lib ./tools/decode/test_runtime_258v
```

---

## 5. Running the Resident Inference Server

A persistent resident inference daemon serves OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/healthz`, `/readyz`).

### Start the Server (LAN accessible)
```bash
# Launch server with fast startup and speculative decoding enabled
./tools/http/serve_lan.sh --port 8080 --fast-load --speculative
```

### Systemd User Service (Optional)
To run AInfer as a persistent background daemon:
```bash
# Install user service
./tools/http/install_service.sh install

# Start or check status
systemctl --user start ainfer
systemctl --user status ainfer
```

### Client Request Examples

**Chat Completion (SSE Streaming):**
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tiel-coder-35b",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}
    ],
    "max_tokens": 128,
    "stream": true
  }'
```

**Fill-In-The-Middle (FIM) Autocomplete:**
```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tiel-coder-35b",
    "prompt": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n",
    "suffix": "    return quicksort(left) + middle + quicksort(right)\n",
    "max_tokens": 64
  }'
```

---

## 6. Documentation Reference

- **[`STATUS.md`](STATUS.md)**: **Single-source release truth** (hardware specs, phase gates, benchmark records).
- **[`improvement.md`](improvement.md)**: Detailed technical reports for all completed optimizations (I1.1–I3.7).
- **[`claim_correction.md`](claim_correction.md)**: Transparent benchmark analysis comparing AInfer against llama.cpp across context lengths.
- **[`progress.md`](progress.md)**: Chronological engineering log and comprehensive change history.
- **[`memory_feasibility_estimate.md`](memory_feasibility_estimate.md)**: Detailed memory breakdown across context tiers (4K to 128K).
