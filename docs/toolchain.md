# Pinned toolchain — AInfer B60 box (T0.1)

Recorded 2026-09-07 on the Ubuntu build/inference host. Reproduce with these exact
versions before running any kernel or performance work.

| Component | Version | Source |
| --- | --- | --- |
| OS | Ubuntu 26.04 LTS (`resolute`) | `lsb_release` |
| Kernel | 7.0.0-31-generic | `uname -r` |
| GPU (target) | Intel Arc Pro B60, `03:00.0`, PCI `8086:e211` (Battlemage G21) | `lspci -nn` |
| GPU (ignore) | NVIDIA RTX 3060 (present, never a target) | `lspci -nn` |
| Intel GPU driver | 26.05.37020.3 (`libze-intel-gpu1`, `intel-opencl-icd`) | dpkg |
| Level Zero loader | 1.28.2 (`libze_loader.so.1.28.2`) | `ls /usr/lib/x86_64-linux-gnu` |
| L0 headers | `/usr/include/level_zero/` (`libze-intel-gpu-dev`) | fs |
| IGC | 1.0.17791.18 (`libigc1`) | dpkg |
| oneAPI toolkit | 2026.1 (`/opt/intel/oneapi`) | fs |
| DPC++/C++ compiler | 2026.1.1 (`icpx`, `icx`, `dpcpp`) | `icpx --version` |
| CMake / Ninja | 4.2.3 / 1.13.2 | binaries |
| Host GCC | 15.2.0 (Ubuntu 15.2.0-16ubuntu1) | `gcc --version` |
| Python (project) | `~/.venvs/ainfer` (uv): torch 2.14 CPU, safetensors 0.8.0, numpy 2.5.3, tokenizers 0.23.2 | venv |

Notes:

- OpenCL already enumerates `Intel(R) Arc(TM) Pro B60 Graphics` (`clinfo`) — driver path works.
- No sudo on this box; system `python3` (3.14) has no pip — always use the venv.
- Repo is on a USB mount (`/mnt/usb/AInfer`): no symlinks, slow large-file I/O.
- Legacy driver `24.35.30872.45` is also installed — the loader must pick the 26.05
  Battlemage driver; the T0.2 probe asserts this by PCI/device ID.

## T5.3 findings (2026-09-09, oneAPI 2026.1.1)

- SYCL Graph (`sycl::ext::oneapi::experimental`) is present in headers but
  UNUSABLE on this stack: recording throws `opencl backend is not supported by
  SYCL Graph extension`. Record/replay must use raw L0 lists + SPIR-V modules.
- Do NOT mix allocators across APIs: SYCL calls reject L0-allocated pointers
  (`urEnqueueUSMMemcpy` -> `UR_RESULT_ERROR_OUT_OF_RESOURCES`), while the L0
  driver segfaults (in `libze_intel_gpu`) on SYCL-allocated pointers from a
  foreign context. L0 paths use `zeMemAllocDevice` buffers; SYCL paths use
  SYCL-USM buffers. Inside-kernel references to L0 arenas are fine (decode).
- Working SPIR-V flow for named SYCL functors: `icpx -fsycl
  -fsycl-targets=spir64 -c` (TU must actually launch the kernel or no device
  image is emitted) -> `objcopy --dump-section
  __CLANG_OFFLOAD_BUNDLE__sycl-spir64-unknown-unknown=` (`clang-offload-bundler`
  cannot see the bundle) -> `llvm-spirv` translator (`.../bin/compiler/`).
  Entry name = functor mangling (e.g. `_ZTS7SiluMul`, 3 params, no hidden args
  — verified by parsing `OpFunctionParameter` count).

## T5.3 GEMV-port build flow (2026-09-10)

- Plain `llvm-spirv` on the spir64 bundle DIES on ESIMD code (`LLVM ERROR:
  Unsupported vector type with 32 elements` — SPIR-V cannot express wide LLVM
  vectors). The working flow replicates the icpx driver link (captured via
  `-v`): `sycl-post-link -split=auto -split-esimd -lower-esimd` first (lowers
  simd ops to `llvm.genx.*` intrinsics), then `llvm-spirv` with
  `--spirv-allow-unknown-intrinsics=llvm.genx.` (twice, verbatim) plus the
  driver's extension set (notably `+SPV_INTEL_vector_compute`). Encapsulated in
 `tools/cmdlist/extract_spv.py`, which selects one module per kernel entry BY
 CONTENT (post-link image numbering is unstable). ESIMD kernels launch under
 raw L0 with 1 WI per group. SHADOW TRAP (2026-09-11): a vectorized entry
 exists in BOTH its `spl_esimd_*` image (real code) and a scalar image
 (outline) — naive first-match silently runs scalar (ChunkSsmRecur at
 193 ms/chunk despite vector source). The extractor prefers ESIMD images;
 after ANY vectorization verify `genx` markers in the output module AND
 re-time. Related: `esimd::convert<uint16_t→float>` is a NUMERIC conversion
 (16000.0f), not a BF16 reinterpret — decode BF16 rows via zero-extend +
 `<<16` + `bit_cast_view<float>` (exact); lambdas doing simd ops need
 `SYCL_ESIMD_FUNCTION`, and `simd[i]` yields `simd<T,1>` (cast to scalar).

## T5.4 finding (2026-09-10)

- Control-block policy is settled: a 16 B `DecodeControl` in **device** memory,
  updated by immediate H2D copy (7.35 us), beats `zeMemAllocShared` + direct
  host writes (48.30 us update, 76.67 us replay — migration cost on write plus
  uncached device reads on every work-item). Confirms the plan.md rule: never
  assume zero-copy is fastest on this discrete GPU. One `kernels.o` bundle
  feeds multiple SPIR-V extractions (silu-mul + control-add so far).

## Chunk-GEMM finding (2026-09-13)

- ARG-STRIP TRAP (dead-strip family, third instance): an UNREFERENCED functor
  member is silently stripped from the kernel signature — remaining arg
  indices shift down, and `zeKernelSetArgumentValue` fails with
  INVALID_ARGUMENT (0x78000004) on the now-out-of-range index, far from the
  cause. `ChunkQkGemm` reported 9 args instead of 10 because `M` was never
  read in the body (row index came from grid decomposition). Caught by a
  30-line `numKernelArgs` runtime probe (`zeKernelGetProperties`), not the
  compiler. Rule: EVERY functor member gets referenced — a bounds guard
  (`if (m >= M) return;`) doubles as the liveness guard. The TMAX lesson
  (unused whole kernel vanishes) and Concat2 guard are the same family.
- FILL-PATTERN TRAP (2026-09-13, chunk64): `zeCommandListAppendMemoryFill`
  takes the pattern as a POINTER — passing integer `0` compiles (null
  conversion) and segfaults inside the driver with no diagnostic. Always
  pass a real pattern word (`uint8_t zpat = 0; ... &zpat ...`). Sibling
  rule of the address-capture family: recorded lists bake addresses, so
  shared weight buffers must be uploaded per layer BEFORE that layer
  executes (never bulk up front — every list would read the last layer's
  weights); every tensor kind gets its own buffer pair (in_proj_z has V6
  rows, not KVW — mis-binning overruns the device allocation).
