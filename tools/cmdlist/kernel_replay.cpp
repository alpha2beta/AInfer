// T5.3: record-once / replay-many of a REAL kernel in a regular L0 list.
// Kernel = decode-exact fused silu-mul (I=17408), compiled to SPIR-V at build
// time (icpx spir64 -> objcopy section -> llvm-spirv), launched from a closed
// regular list with fixed argument addresses; inputs vary per replay via an
// immediate upload list. Outputs must match the host reference bit-exactly
// every replay (no stale capture), with per-replay fence timing.
// Usage: kernel_replay <silumul.spv> [report.json]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: kernel_replay <silumul.spv> [report.json]\n");
    return 2;
  }
  const int N = 17408, GS = 256, NG = N / GS, IT = 50;
  FILE *sf = std::fopen(argv[1], "rb");
  if (!sf) {
    std::fprintf(stderr, "no spv\n");
    return 2;
  }
  std::fseek(sf, 0, SEEK_END);
  size_t spvN = std::ftell(sf);
  std::fseek(sf, 0, SEEK_SET);
  std::vector<uint8_t> spv(spvN);
  if (std::fread(spv.data(), 1, spvN, sf) != spvN)
    return 2;
  std::fclose(sf);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));
  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS)
      continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS &&
          pr.vendorId == 0x8086 && pr.deviceId == 0xe211) {
        dev = v;
        drv = d;
      }
    }
  }
  if (!dev)
    return 1;
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *dG = nullptr, *dU = nullptr, *dZ = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * 4, 4096, dev, &dG));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * 4, 4096, dev, &dU));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * 4, 4096, dev, &dZ));

  ze_module_handle_t mod = nullptr;
  ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spvN,
                             spv.data(),
                             nullptr,
                             nullptr};
  CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
  ze_kernel_handle_t ker = nullptr;
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                            "_ZTS7SiluMul"};
  ze_result_t kr = zeKernelCreate(mod, &kdesc, &ker);
  if (kr != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS7SiluMul not found (%d)\n", (int)kr);
    return 1;
  }
  CHECK(zeKernelSetGroupSize(ker, GS, 1, 1));
  CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dG));
  CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dU));
  CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dZ));

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_command_list_handle_t reg = nullptr;
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                  nullptr, 0, 0};
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &reg));
  ze_group_count_t gc = {(uint32_t)NG, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, ker, &gc, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(reg));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));

  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));

  // NOTE: device sycl::exp is single-precision native: host reference uses
  // expf and a 2e-6 tolerance (bitwise vs libm-double would be meaningless).
  // Replay determinism (same input twice -> bitwise identical) is checked
  // exactly: that is the property T5.3 needs.
  std::vector<float> hG(N), hU(N), hZ(N), hZ2(N), hRef(N);
  uint64_t s = 0xABCDEF;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (2.0f / 32768.0f);
  };
  std::vector<double> tRep;
  bool ok = true;
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hG)
      v = rnd();
    for (auto &v : hU)
      v = rnd();
    for (int j = 0; j < N; ++j) {
      float g = hG[j];
      hRef[j] = (g / (1.0f + expf(-g))) * hU[j];
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dG, hG.data(), (size_t)N * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dU, hU.data(), (size_t)N * 4,
                                        nullptr, 0, nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hZ.data(), dZ, (size_t)N * 4,
                                        nullptr, 0, nullptr));
    // replay determinism: same input, second execute, must be bitwise equal
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hZ2.data(), dZ, (size_t)N * 4,
                                        nullptr, 0, nullptr));
    for (int j = 0; j < N; ++j) {
      uint32_t a, b;
      __builtin_memcpy(&a, &hZ[j], 4);
      __builtin_memcpy(&b, &hZ2[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d j %d\n", it, j);
        break;
      }
      if (std::fabs(hZ[j] - hRef[j]) > 2e-6f * (1.0f + std::fabs(hRef[j]))) {
        ok = false;
        std::fprintf(stderr, "mismatch it %d j %d: got %g ref %g\n", it, j,
                     hZ[j], hRef[j]);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  std::printf("kernel-replay med %.2f us/iter %s\n", mR / 1e3,
              ok ? "BITWISE-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS7SiluMul\",\"N\":%d,"
                "\"iters\":%d,\"replay_us\":%.2f,"
                "\"replay_deterministic_and_accurate\":%s}",
                N, IT, mR / 1e3, ok ? "true" : "false");
  if (argc > 2) {
    FILE *o = std::fopen(argv[2], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
