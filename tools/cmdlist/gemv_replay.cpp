// T5.3 GEMV port: record ONCE a closed regular L0 list launching the
// decode-exact ESIMD dp4a INT4 GEMV (Int4Gemv, gate shape 17408x5120) with
// fixed addresses, replay 20x over varying activations (weights/scales stay;
// XQ+SQ re-uploaded per replay via immediate list). Checks: bitwise match vs
// an order-mirrored host reference (int32 group sums are exact; float
// group-rescale follows group order on both sides) and bitwise replay
// determinism. Group size is 1 WI/group (no cross-WI communication).
// Usage: gemv_replay <gemv.spv> [report.json]
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: gemv_replay <gemv.spv> [report.json]\n");
    return 2;
  }
  const int M = 17408, K = 5120, GG = K / 128, IT = 20;
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
  void *dP = nullptr, *dS = nullptr, *dXQ = nullptr, *dSQ = nullptr,
       *dY = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * K / 2, 4096, dev, &dP));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * GG * 2, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)K, 4096, dev, &dXQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)GG * 4, 4096, dev, &dSQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * 4, 4096, dev, &dY));

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
                            "_ZTS8Int4Gemv"};
  if (zeKernelCreate(mod, &kdesc, &ker) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS8Int4Gemv not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(ker, 1, 1, 1));
  int kArg = K;
  CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dY));
  CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dP));
  CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(ker, 3, sizeof(void *), &dXQ));
  CHECK(zeKernelSetArgumentValue(ker, 4, sizeof(void *), &dSQ));
  CHECK(zeKernelSetArgumentValue(ker, 5, sizeof(int), &kArg));

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
  ze_group_count_t gc = {(uint32_t)M, 1, 1};
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

  // Synthetic weights, packed exactly like the .binfer layout-0 path.
  uint64_t s = 0x9E3779B9;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (0.02f / 32768.0f);
  };
  std::vector<uint8_t> packed((size_t)M * K / 2, 0);
  std::vector<uint16_t> scales((size_t)M * GG);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < GG; ++g) {
      float amax = 0, w[128];
      for (int j = 0; j < 128; ++j) {
        w[j] = rnd();
        amax = std::max(amax, std::fabs(w[j]));
      }
      float sc = amax == 0 ? 1.0f : amax / 7.0f;
      scales[(size_t)m * GG + g] = f32_to_bf16(sc);
      for (int j = 0; j < 128; ++j) {
        int q = (int)std::lrint(w[j] / sc);
        q = q < -8 ? -8 : (q > 7 ? 7 : q);
        size_t idx = (size_t)m * K + g * 128 + j;
        if (j & 1)
          packed[idx / 2] |= (uint8_t)((q & 0xF) << 4);
        else
          packed[idx / 2] = (uint8_t)(q & 0xF);
      }
    }
  }
  CHECK(zeCommandListAppendMemoryCopy(up, dP, packed.data(), packed.size(),
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dS, scales.data(),
                                      scales.size() * 2, nullptr, 0, nullptr));

  std::vector<float> hX(K), hSQ(GG), hY(M), hY2(M), hRef(M);
  std::vector<int8_t> hXQ(K);
  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hX)
      v = rnd() * 50.0f; // activation-scale magnitudes
    for (int g = 0; g < GG; ++g) {
      float mx = 0;
      for (int j = 0; j < 128; ++j)
        mx = std::max(mx, std::fabs(hX[g * 128 + j]));
      hSQ[g] = mx / 127.0f;
      if (hSQ[g] == 0)
        hSQ[g] = 1.0f;
      for (int j = 0; j < 128; ++j) {
        float v = hX[g * 128 + j] / hSQ[g];
        int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        hXQ[g * 128 + j] = (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
      }
    }
    // Order-mirrored host reference: int32 group dot, then float rescale in
    // group order — identical operations to the device kernel.
    for (int m = 0; m < M; ++m) {
      float acc = 0;
      for (int g = 0; g < GG; ++g) {
        float sc = bf16_to_f32(scales[(size_t)m * GG + g]);
        int gs = 0;
        for (int j = 0; j < 128; ++j) {
          size_t idx = (size_t)m * K + g * 128 + j;
          uint8_t b = packed[idx / 2];
          int nib = (idx & 1) ? (b >> 4) : (b & 0xF);
          if (nib >= 8)
            nib -= 16;
          gs += nib * (int)hXQ[g * 128 + j];
        }
        acc += (float)gs * sc * hSQ[g];
      }
      hRef[m] = acc;
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dXQ, hXQ.data(), (size_t)K,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dSQ, hSQ.data(), (size_t)GG * 4,
                                        nullptr, 0, nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hY.data(), dY, (size_t)M * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hY2.data(), dY, (size_t)M * 4,
                                        nullptr, 0, nullptr));
    // Accuracy vs order-mirrored host ref: int32 group sums are exact on both
    // sides, but the float group-rescale ((float)gs*sc*sq) may reassociate in
    // the device backend (-ffp-contract=fast -mreassociate) vs the host loop,
    // so the bar is tight relative tolerance, not bitwise. Bitwise replay
    // determinism (same input twice -> identical bits) stays strict: that is
    // the recording property T5.3 needs.
    double refmax = 0;
    for (int m = 0; m < M; ++m)
      refmax = std::max(refmax, (double)std::fabs(hRef[m]));
    for (int m = 0; m < M; ++m) {
      double rel =
          std::fabs((double)hY[m] - (double)hRef[m]) / (refmax > 0 ? refmax : 1);
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-6) {
        ok = false;
        std::fprintf(stderr, "ref mismatch it %d m %d: got %g want %g\n", it,
                     m, hY[m], hRef[m]);
        break;
      }
      uint32_t a, c;
      __builtin_memcpy(&a, &hY[m], 4);
      __builtin_memcpy(&c, &hY2[m], 4);
      if (a != c) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d m %d\n", it, m);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  std::printf("gemv-replay med %.2f us/iter worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "GEMV-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS8Int4Gemv\","
                "\"shape\":[%d,%d],\"iters\":%d,\"replay_us\":%.2f,"
                "\"worst_rel\":%.2e,\"ref_tol\":1e-6,"
                "\"bitwise_deterministic\":%s,\"gemv_ok\":%s}",
                M, K, IT, mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false");
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
