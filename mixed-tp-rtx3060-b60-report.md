# Mixed-GPU Tensor Parallelism Test Report: RTX 3060 + Arc Pro B60

Date: 2026-09-18
Host: Intel i5-12400F, 30 GB RAM, Ubuntu (NVMe, 52 GB free at test time)
Binary source: `~/llama.cpp` @ `44be98f05` (b11037, `0.4.1-dev`), pulled 2026-09-18

## 1. Hardware and toolchains

| Device | Detail |
|---|---|
| NVIDIA RTX 3060 LHR (GA106) | 12 GB (11911 MiB), driver 595.84, CUDA 12.4 (`nvcc` V12.4.131), compute capability 8.6, VMM yes |
| Intel Arc Pro B60 (Battlemage G21, `xe` driver) | 24 GB (23256 MiB reported), Level-Zero 20.1.0 / driver 1.14.37020, OpenCL NEO 26.05.037020, 160 compute units |
| Intel oneAPI | 2026.1 (`icx`/`icpx`, MKL, oneDNN 2026.0, Level-Zero). `sycl-ls` sees the B60 as `[level_zero:gpu]` |
| Build tools | CMake 4.2.3, Ninja 1.13.2, 12 cores |

`llama-cli --list-devices` on the mixed build reports both:
`CUDA0: NVIDIA GeForce RTX 3060 (11911 MiB)`, `SYCL0: Intel Arc Pro B60 (23256 MiB)`.

## 2. Starting point: existing builds and release b9788

Seven checkouts under `~` (`llama.cpp`, `llama.arc`, `llama.vk`, `llama-dflash2`,
`llama-k2horizon`, `llama.spk`, `beellama.cpp`) with ten build dirs. Before this
work, none was at b9788; all ggml-org checkouts were 213-710 commits newer.

b9788 = `e9fb3b3` (2026-06-25, PR #24152, "sycl: support --split-mode tensor"):
dual-GPU (N=2) SYCL all-reduce with an FP32 direct path (small tensors) and a
BF16-compressed path (large tensors). Verified merged into `master` in every
ggml-org clone (`merge-base --is-ancestor e9fb3b3 origin/master` = YES;
`master`/`origin/master` in `branch -a --contains`).

## 3. Feasibility of mixed NVIDIA + Intel tensor parallelism

- Backend-agnostic TP (#19378) plus the b9788 SYCL path make it possible. A Reddit
  report (B580 + 5060 Ti, Qwen 27B Q4_K_M, ~14 -> ~35 t/s) confirms mixed-vendor
  works in practice.
- Code: `ggml_backend_cuda_comm_init` (`ggml-cuda.cu`) and
  `ggml_backend_sycl_comm_init` (`ggml-sycl.cpp`) both return `nullptr` for
  heterogeneous backend lists, so CUDA+SYCL falls back to the meta-backend's
  generic butterfly all-reduce. Functional, but without the NCCL / SYCL-ring
  fast paths. The SYCL fast path additionally requires pure-SYCL N=2.
- `docs/build.md` confirms multi-backend single binaries (`-DGGML_CUDA=ON
  -DGGML_SYCL=ON`); probe configure with `icx`/`icpx` + `nvcc` succeeded.

## 4. Builds made in this work

| Build dir | Config | Notes |
|---|---|---|
| `~/llama.cpp/build-mixed` | `-DGGML_CUDA=ON -DGGML_SYCL=ON`, `icx`/`icpx`, Ninja, Release, SYCL FP32 | `llama-cli --version`: `0.4.1-dev (build 11037, 44be98f05)` |
| `~/llama.cpp/build-mixed-f16` | same + `-DGGML_SYCL_F16=ON` | Same version; ~2.8x SYCL prefill (see 6.2) |

Gotcha (hit once, then avoided): the build shell MUST have
`source /opt/intel/oneapi/setvars.sh` in its environment, otherwise the SYCL
compile fails with `fatal error: 'dnnl.hpp' file not found` (include comes via
`CPLUS_INCLUDE_PATH`, not CMake flags). Same sourcing required at runtime
(`libsvml.so`, `libdnnl`).

## 5. Repro scripts (`~/llama.cpp/`)

- `qwen38-b60-only.sh` — B60 only, `-sm layer -dev SYCL0 -ngl 99 -fitt 1000`
- `qwen38-mixed-tp.sh` — mixed, `-sm tensor -dev CUDA0/SYCL0 -ts 1/2 -fa on -ctk/-ctv f16`
- `qwen38-b60-only-f16.sh`, `qwen38-mixed-tp-f16.sh` — same pair on `build-mixed-f16`
- All run `llama-bench -m <27B model> -p 512 -n 128 -r 1` from `~/llama.cpp`.

Bench syntax notes: `-ngl` needs a number (`all` rejected); inside one `-dev` /
`-ts` value entries are `/`-separated for a SINGLE multi-device run
(`-dev CUDA0/SYCL0 -ts 1/2`); commas run each combo separately. `-fitt` was
required for the 18 GB load on the 24 GB B60 (without it the load stalled).

## 6. Results (Qwen3.8-27B-Uncensored Q4_K_P, 16.7 GiB, pp512 / tg128)

### 6.1 FP32 SYCL build, f16 KV (baseline 2x2)

| | f16 KV | q8_0 KV |
|---|---|---|
| B60 only, layer | 193.50 / 16.82 | 190.38 / 16.91 |
| Mixed 3060+B60, tensor | 144.07 / 18.80 | 147.28 / 18.80 |

Mixed TP: -26% prefill, +12% decode. q8_0 KV changes speed ~0% (value is ~half
KV memory -> room for larger `-c` / more slots). Tensor+q8 works (b9455 fix holds).

### 6.2 FP16 SYCL build, f16 KV

| | FP32 build | FP16 build |
|---|---|---|
| B60 only, layer | 193.50 / 16.82 | 539.14 / 17.01 |
| Mixed, tensor | 144.07 / 18.80 | 244.74 / 19.02 |

Prefill jumps 2.8x single / 1.7x mixed; decode flat (bandwidth-bound, same bytes).
Mixed-vs-single verdict unchanged: about -55% pp / +12% tg on the FP16 build.

### 6.3 MTP (MAX-MTP file, 18.5 GB, `blk.64.nextn.*` heads present)

llama-server, 128 tokens, temp 0, same prompt, f16 KV, `-c 8192`:

| Setup | tg128 |
|---|---|
| B60 only, no spec | 19.00 |
| B60 only, `--spec-type draft-mtp --spec-draft-n-max 2` | 29.64 (+56%) |
| Mixed TP + draft-mtp | 27.57 (+45% vs no-spec; -7% vs B60+MTP, with "CPU sampler fallback" warnings on the draft path) |

MTP is the biggest decode lever here and stacks best on the single B60.
(The plain Qwen3.8 file has no MTP tensors, so MTP was never an option there;
`llama-bench` has no spec flags, hence server-based measurement.)

### 6.4 Negative result

`gemma4-v2-Q4_K_M` aborts in `-sm tensor` with
`ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`.
Reproduces single-CUDA too -> model-arch vs current-master tensor path issue,
NOT mixed-backend specific.

## 7. Max context size: single B60 vs mixed (FP16 build, q8_0 KV)

Model is a hybrid: 65 blocks, but only 25 carry `attn_k`/`attn_v`
(48 blocks carry `ssm_*`, fixed-size state). KV/token = 2 x 25 x 4 heads x 256
= 51,200 elements (~100 KB f16, ~50 KB q8_0). Probed with `llama-completion`
(`-ngl 999 -fa on`, `-p "Hi" -n 1`):

| Setup | Fits | OOMs | Verified max |
|---|---|---|---|
| B60 only, layer | 164k | 176k (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) | ~164k |
| Mixed 3060+B60, tensor `-ts 1,2` | 352k | 368k (`ggml-backend-meta.cpp:1704` buffer assert) | ~352k (~2.1x single) |

Perf at max ctx (long repetitive prompt + summary instruction, temp default):

| Setup | Prompt tokens | pp | tg (n=63 / 47) |
|---|---|---|---|
| B60 only, c=167936 | 89,701 | 457.3 t/s | 6.22 t/s |
| Mixed, c=352256 | 191,911 | 191.9 t/s | 4.35 t/s |

Notes: (a) prompts differ in length (each ~55% of its max), so this is not a
same-ctx comparison — attention cost grows with ctx; (b) decode collapses at
long ctx on both (17->6.2 single, 19->4.4 mixed) since each step attends the
full KV; (c) an early mixed run EOS'd after 1 token on pure repetition and was
re-run with a trailing summary instruction. Raw logs: `probe-b60-*.log`,
`probe-mx-*.log`, `perf-b60-max.log`, `perf-mx-max2.log` in `/tmp/opencode/`.

## 7b. Max context WITH draft-mtp (MAX-MTP file, 18.5 GB, FP16 build, q8_0 KV)

Speculation is server-only (`llama-completion` has no `--spec-type`), so these
went through `llama-server --spec-type draft-mtp --spec-draft-n-max 2 -np 1`.

| Setup | Loads | Fails | Verified max |
|---|---|---|---|
| B60 only | 148k | 152k (OOM), 160k (DEVICE_LOST) | ~148k |
| Mixed tensor `-ts 1,2` | 288k | 304k, 320k (meta-buf assert `:1704`/`:1760`) | ~288k (~1.9x single) |

Important nuance: load-ceiling is NOT run-ceiling with MTP. The 148k-ctx
single server loaded fine but OOM'd (`common.hpp:145` map failure) during a
90k-token prefill — the draft path needs extra prefill scratch. Rerun at
`-c 98304` worked.

Long-context perf with draft-mtp (varied list prompt, temp 0.7 — temp 0.0 EOS'd
after 1 token on both setups):

| Setup | Prompt tokens | pp | tg (n=48) |
|---|---|---|---|
| B60 only, c=98304 | 69,831 | 453.9 t/s | 18.30 t/s |
| Mixed, c=294912 | 164,391 | 182.3 t/s | 12.22 t/s |

So with MTP at long ctx: single B60 does 454/18.3 at 70k prompt; mixed does
182/12.2 at 164k prompt. MTP's relative gain shrinks at long ctx (draft
verification itself attends the full KV), and the single-GPU setup again wins
on speed while mixed wins on reachable ctx (288k vs 148k). For pure decode
throughput at fittable ctx, B60+MTP remains the best combo measured (29.6 t/s
at 8k ctx).

## 8. Conclusions and recommendations

1. Mixed 3060 + B60 tensor parallelism works (single CUDA+SYCL binary), but on
   PCIe with the generic fallback it trades -25..-55% prefill for +12% decode.
   Use it when a model needs >24 GB or serving is decode-bound; otherwise the
   B60 alone (especially the FP16 build at 539 t/s pp) is faster.
2. `build-mixed-f16` is now the default binary for anything touching the B60.
3. q8_0 KV: no speed change, take the memory headroom for larger context/slots.
4. MTP (+56% tg on B60) beats adding the second GPU (+12%); combine as
   B60 + MTP + q8 KV. Mixed TP + MTP works but is slower than B60 + MTP.
5. Replacing the 3060 with a B580 is NOT recommended: same 12 GB (no capacity
   gain), lose the mature CUDA path, dual-Intel fast-path gains are
   workload-dependent. A second B60 would be the real step up.
6. Max context (q8 KV) is where the second GPU wins decisively: ~352k mixed
   vs ~164k single (2.1x). At those maxes decode is KV-bound on both
   (6.2 vs 4.4 t/s).

## 9. Raw logs

- `/tmp/opencode/bench-b60only.log`, `bench-mixedtp.log` (FP32, f16 KV)
- `/tmp/opencode/bench-mixed-q8.log`, `bench-b60-q8.log` (q8 KV)
- `/tmp/opencode/bench-f16-b60.log`, `bench-f16-mixed.log` (FP16 build)
- `/tmp/opencode/mtp-base.json`, `mtp-draft.json`, `mtp-tp.json` (server timings)
- `/tmp/opencode/f16-build.log`, `mixed-build2.log` (build logs)
- `/tmp/opencode/srv-base.log`, `srv-mtp.log`, `srv-tp-mtp.log` (server logs)
- `/tmp/opencode/probe-b60-*.log`, `probe-mx-*.log` (max-ctx probes)
- `/tmp/opencode/perf-b60-max.log`, `perf-mx-max2.log` (max-ctx perf)
