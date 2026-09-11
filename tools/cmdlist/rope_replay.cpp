// T5.3 RoPE/KV port: record ONCE a closed regular L0 list of the decode-exact
// NeoX half-rotation (24Q+4K heads, leading 64 dims) followed by the KV-cache
// append, replay 20x over varying Q/K/V and non-monotonic positions driven by
// the control block (explicit 16 B update, no arg mutation). Checks:
// tight-tolerance match vs order-mirrored host ref (fma contraction can move
// the last ulp) + bitwise replay determinism.
// Usage: rope_replay <rope.spv> <kvappend.spv> [report.json]
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

struct DecodeControl {
  int token_id, position, active_length, selected_token;
};

static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
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
  if (argc < 3) {
    std::fprintf(stderr, "usage: rope_replay <rope.spv> <kvappend.spv> "
                         "[report.json]\n");
    return 2;
  }
  const int NQ = 24, NK = 4, D = 256, TMAX = 64, IT = 20;
  auto load_spv = [&](const char *path) {
    FILE *sf = std::fopen(path, "rb");
    if (!sf) {
      std::fprintf(stderr, "no spv: %s\n", path);
      std::exit(2);
    }
    std::fseek(sf, 0, SEEK_END);
    size_t n = std::ftell(sf);
    std::fseek(sf, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (std::fread(v.data(), 1, n, sf) != n)
      std::exit(2);
    std::fclose(sf);
    return v;
  };
  std::vector<uint8_t> spvA = load_spv(argv[1]);
  std::vector<uint8_t> spvB = load_spv(argv[2]);

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
  void *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dCos = nullptr,
       *dSin = nullptr, *dKc = nullptr, *dVc = nullptr, *dCtrl = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NQ * D * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * D * 4, 4096, dev, &dK));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * D * 4, 4096, dev, &dV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)64 * TMAX * 4, 4096, dev, &dCos));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)64 * TMAX * 4, 4096, dev, &dSin));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * TMAX * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * TMAX * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));

  auto mkmod = [&](const std::vector<uint8_t> &spv) -> ze_module_handle_t {
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spv.size(),
                               spv.data(),
                               nullptr,
                               nullptr};
    if (zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "L0 module create failed\n");
      std::exit(1);
    }
    return mod;
  };
  ze_module_handle_t modA = mkmod(spvA), modB = mkmod(spvB);
  ze_kernel_handle_t kRope = nullptr, kKv = nullptr;
  ze_kernel_desc_t kd1 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS9RopeApply"};
  ze_kernel_desc_t kd2 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS8KvAppend"};
  if (zeKernelCreate(modA, &kd1, &kRope) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modB, &kd2, &kKv) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ropekv entry point not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kRope, 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kKv, 1, 1, 1));
  int tmaxArg = TMAX;
  CHECK(zeKernelSetArgumentValue(kRope, 0, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(kRope, 1, sizeof(void *), &dK));
  CHECK(zeKernelSetArgumentValue(kRope, 2, sizeof(void *), &dCos));
  CHECK(zeKernelSetArgumentValue(kRope, 3, sizeof(void *), &dSin));
  CHECK(zeKernelSetArgumentValue(kRope, 4, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kKv, 0, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(kKv, 1, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(kKv, 2, sizeof(void *), &dK));
  CHECK(zeKernelSetArgumentValue(kKv, 3, sizeof(void *), &dV));
  CHECK(zeKernelSetArgumentValue(kKv, 4, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kKv, 5, sizeof(int), &tmaxArg));

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
  ze_group_count_t gA = {28, 1, 1};
  ze_group_count_t gB = {(uint32_t)(NK * D), 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, kRope, &gA, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(reg, kKv, &gB, nullptr, 0, nullptr));
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

  // RoPE tables, decode-identical formula.
  std::vector<float> hCos(64 * TMAX), hSin(64 * TMAX);
  for (int t = 0; t < TMAX; ++t)
    for (int i = 0; i < 64; ++i) {
      double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
      double ang = (double)t * inv;
      hCos[t * 64 + i] = (float)std::cos(ang);
      hSin[t * 64 + i] = (float)std::sin(ang);
    }
  CHECK(zeCommandListAppendMemoryCopy(up, dCos, hCos.data(), hCos.size() * 4,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dSin, hSin.data(), hSin.size() * 4,
                                      nullptr, 0, nullptr));

  uint64_t s = 0x20FE;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(NQ * D), hK(NK * D), hV(NK * D), hQ2(NQ * D),
      hSlotK(NK * D), hSlotK2(NK * D), hSlotV(NK * D);
  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hQ)
      v = rnd() * 2.0f;
    for (auto &v : hK)
      v = rnd() * 2.0f;
    for (auto &v : hV)
      v = rnd();
    int pos = (it * 41 + 3) % TMAX;
    DecodeControl c{7000 + it, pos, pos + 1, -1};
    // Host ref: rotate copies, then expected cache slot.
    std::vector<float> rQ = hQ, rK = hK;
    for (int i = 0; i < NQ + NK; ++i) {
      float *X = i < NQ ? rQ.data() + (size_t)i * D
                        : rK.data() + (size_t)(i - NQ) * D;
      for (int d = 0; d < 32; ++d) {
        float x0 = X[d], x1 = X[d + 32];
        float cc = hCos[pos * 64 + d], ss = hSin[pos * 64 + d];
        X[d] = x0 * cc - x1 * ss;
        X[d + 32] = x0 * ss + x1 * cc;
      }
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dQ, hQ.data(), hQ.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dK, hK.data(), hK.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dV, hV.data(), hV.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hQ2.data(), dQ, hQ2.size() * 4,
                                        nullptr, 0, nullptr));
    // Read back the appended cache slot (BF16; offset pos*4*D uint16s).
    // Device RNE rounding == host RNE, so slot bits must match exactly.
    std::vector<uint16_t> slot((size_t)NK * D);
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dKc + (size_t)pos * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      hSlotK[j] = bf16_to_f32(slot[j]);
      if (slot[j] != f32_to_bf16(rK[j])) {
        ok = false;
        std::fprintf(stderr, "kvslot-K bits mismatch it %d j %zu\n", it, j);
        break;
      }
    }
    if (!ok)
      break;
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dVc + (size_t)pos * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      hSlotV[j] = bf16_to_f32(slot[j]);
      if (slot[j] != f32_to_bf16(hV[j])) {
        ok = false;
        std::fprintf(stderr, "kvslot-V bits mismatch it %d j %zu\n", it, j);
        break;
      }
    }
    if (!ok)
      break;
    // Determinism: replay same state, expect identical Q.
    CHECK(zeCommandListAppendMemoryCopy(up, dQ, hQ.data(), hQ.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dK, hK.data(), hK.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dV, hV.data(), hV.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    std::vector<float> hQ3(hQ.size());
    CHECK(zeCommandListAppendMemoryCopy(up, hQ3.data(), dQ, hQ3.size() * 4,
                                        nullptr, 0, nullptr));
    double refmax = 0;
    for (float v : rQ)
      refmax = std::max(refmax, (double)std::fabs(v));
    if (refmax == 0)
      refmax = 1;
    for (size_t j = 0; j < rQ.size(); ++j) {
      double rel = std::fabs((double)hQ2[j] - (double)rQ[j]) / refmax;
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-6) {
        ok = false;
        std::fprintf(stderr, "rope mismatch it %d j %zu: got %g want %g\n", it,
                     j, hQ2[j], rQ[j]);
        break;
      }
      uint32_t a, b;
      __builtin_memcpy(&a, &hQ2[j], 4);
      __builtin_memcpy(&b, &hQ3[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d j %zu\n", it, j);
        break;
      }
    }
    if (!ok)
      break;
    // Slots already bitwise-verified above; here they only need BF16
    // agreement with the rotated values (quantization, not kernel error).
    for (size_t j = 0; j < hSlotK.size(); ++j) {
      double rel =
          std::fabs((double)hSlotK[j] - (double)bf16_to_f32(f32_to_bf16(rK[j])));
      rel /= (refmax > 0 ? refmax : 1);
      if (rel > worstRel)
        worstRel = rel;
      double relV =
          std::fabs((double)hSlotV[j] - (double)bf16_to_f32(f32_to_bf16(hV[j])));
      relV /= (refmax > 0 ? refmax : 1);
      if (rel > 1e-6 || relV > 1e-6) {
        ok = false;
        std::fprintf(stderr, "kvslot mismatch it %d j %zu\n", it, j);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  std::printf("rope-replay med %.2f us/iter worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "ROPE-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernels\":[\"_ZTS9RopeApply\","
                "\"_ZTS8KvAppend\"],\"heads\":\"24Q/4KV\",\"tmax\":%d,"
                "\"iters\":%d,\"replay_us\":%.2f,\"worst_rel\":%.2e,"
                "\"ref_tol\":1e-6,\"bitwise_deterministic\":%s,"
                "\"rope_ok\":%s}",
                TMAX, IT, mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false");
  if (argc > 3) {
    FILE *o = std::fopen(argv[3], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
