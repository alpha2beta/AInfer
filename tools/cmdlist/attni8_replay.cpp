// T6.3 INT8 KV port: KvAppendI8 (per-token row scales + int8 stores) then
// AttnCoreI8 (dequant on the fly) in one recorded list, replayed over
// varying K/V and non-monotonic positions. Checks: appended int8 slots
// BITWISE vs host round-half-away ref (same formula as the Quantize kernel),
// row scales bitwise-exact (max is rounding-free), attention out within
// float-tail tol vs dequantized ref + bitwise replay determinism.
// Usage: attni8_replay <kvappendi8.spv> <attni8.spv> [report.json]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
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

static int8_t q8one(float v) {
  int qi = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
  return (int8_t)(qi < -127 ? -127 : (qi > 127 ? 127 : qi));
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
    std::fprintf(stderr, "usage: attni8_replay <kvappendi8.spv> <attni8.spv> "
                         "[report.json]\n");
    return 2;
  }
  const int NH = 24, NKV = 4, D = 256, TMAX = 64, IT = 20;
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
  void *dQ = nullptr, *dG = nullptr, *dAtt = nullptr, *dK = nullptr,
       *dV = nullptr, *dKc = nullptr, *dVc = nullptr, *dKs = nullptr,
       *dVs = nullptr, *dCtrl = nullptr, *dWts = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dG));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dAtt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * D * 4, 4096, dev, &dK));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * D * 4, 4096, dev, &dV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAX * D, 4096, dev, &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAX * D, 4096, dev, &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)TMAX * NKV * 4, 4096, dev, &dKs));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)TMAX * NKV * 4, 4096, dev, &dVs));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)24 * TMAX * 4, 4096, dev, &dWts));

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
  ze_kernel_handle_t kKv = nullptr, kAtt = nullptr;
  ze_kernel_desc_t kd1 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS10KvAppendI8"};
  ze_kernel_desc_t kd2 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS10AttnCoreI8"};
  if (zeKernelCreate(modA, &kd1, &kKv) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modB, &kd2, &kAtt) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "i8 entry point not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kKv, 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kAtt, 1, 1, 1));
  int tmaxArg = TMAX;
  CHECK(zeKernelSetArgumentValue(kKv, 0, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(kKv, 1, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(kKv, 2, sizeof(void *), &dKs));
  CHECK(zeKernelSetArgumentValue(kKv, 3, sizeof(void *), &dVs));
  CHECK(zeKernelSetArgumentValue(kKv, 4, sizeof(void *), &dK));
  CHECK(zeKernelSetArgumentValue(kKv, 5, sizeof(void *), &dV));
  CHECK(zeKernelSetArgumentValue(kKv, 6, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kKv, 7, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kAtt, 0, sizeof(void *), &dAtt));
  CHECK(zeKernelSetArgumentValue(kAtt, 1, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(kAtt, 2, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(kAtt, 3, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(kAtt, 4, sizeof(void *), &dKs));
  CHECK(zeKernelSetArgumentValue(kAtt, 5, sizeof(void *), &dVs));
  CHECK(zeKernelSetArgumentValue(kAtt, 6, sizeof(void *), &dG));
  CHECK(zeKernelSetArgumentValue(kAtt, 7, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(kAtt, 8, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(kAtt, 9, sizeof(void *), &dWts));

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
  ze_group_count_t gA = {(uint32_t)NKV, 1, 1};
  ze_group_count_t gB = {(uint32_t)NH, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, kKv, &gA, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(reg, kAtt, &gB, nullptr, 0, nullptr));
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

  uint64_t s = 0x1a78;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(NH * D), hG(NH * D), hK(NKV * D), hV(NKV * D),
      hAtt(NH * D), hAtt2(NH * D), hRef(NH * D),
      hKc(NKV * TMAX * D, 0.0f), hVc(NKV * TMAX * D, 0.0f),
      hKs(TMAX * NKV, 1.0f), hVs(TMAX * NKV, 1.0f);
  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> wts(TMAX);
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hQ)
      v = rnd() * 2.0f;
    for (auto &v : hG)
      v = rnd();
    for (auto &v : hK)
      v = rnd() * 4.0f; // wide range exercises scale dynamics
    for (auto &v : hV)
      v = rnd() * 8.0f;
    int pos = (it * 41 + 3) % TMAX;
    DecodeControl c{8000 + it, pos, pos + 1, -1};
    // Host ref: quantize rows exactly like the device, then float attention.
    for (int hh = 0; hh < NKV; ++hh) {
      float mk = 0, mv = 0;
      for (int d = 0; d < D; ++d) {
        float a = hK[(size_t)hh * D + d];
        mk = std::max(mk, a >= 0 ? a : -a);
        float b = hV[(size_t)hh * D + d];
        mv = std::max(mv, b >= 0 ? b : -b);
      }
      float sk = mk == 0 ? 1.0f : mk / 127.0f;
      float sv = mv == 0 ? 1.0f : mv / 127.0f;
      hKs[(size_t)pos * NKV + hh] = sk;
      hVs[(size_t)pos * NKV + hh] = sv;
      for (int d = 0; d < D; ++d) {
        hKc[((size_t)pos * NKV + hh) * D + d] =
            q8one(hK[(size_t)hh * D + d] / sk) * sk;
        hVc[((size_t)pos * NKV + hh) * D + d] =
            q8one(hV[(size_t)hh * D + d] / sv) * sv;
      }
    }
    int T = pos + 1;
    for (int hh = 0; hh < NH; ++hh) {
      int kv = hh / 6;
      float mx = -1e30f;
      for (int t = 0; t < T; ++t) {
        float sc = 0;
        for (int d = 0; d < D; ++d)
          sc += hQ[(size_t)hh * D + d] *
                hKc[((size_t)t * NKV + kv) * D + d];
        sc /= 16.0f;
        wts[t] = sc;
        mx = sc > mx ? sc : mx;
      }
      float se = 0;
      for (int t = 0; t < T; ++t) {
        float w = expf(wts[t] - mx);
        wts[t] = w;
        se += w;
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0;
        for (int t = 0; t < T; ++t)
          acc += wts[t] / se * hVc[((size_t)t * NKV + kv) * D + d];
        float g = hG[(size_t)hh * D + d];
        hRef[(size_t)hh * D + d] = acc / (1.0f + expf(-g));
      }
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dQ, hQ.data(), hQ.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dG, hG.data(), hG.size() * 4,
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
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt.data(), dAtt, hAtt.size() * 4,
                                        nullptr, 0, nullptr));
    // Bitwise slot check (int8 + scales).
    std::vector<int8_t> slot((size_t)NKV * D);
    std::vector<float> ss(NKV);
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dKc + (size_t)pos * NKV * D,
                                        slot.size(), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, ss.data(),
                                        (char *)dKs + (size_t)pos * NKV * 4,
                                        ss.size() * 4, nullptr, 0, nullptr));
    for (int hh = 0; hh < NKV && ok; ++hh) {
      uint32_t a, b;
      float want = hKs[(size_t)pos * NKV + hh];
      __builtin_memcpy(&a, &ss[hh], 4);
      __builtin_memcpy(&b, &want, 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "scale bits mismatch it %d hh %d\n", it, hh);
        break;
      }
      for (int d = 0; d < D; ++d) {
        float sk = hKs[(size_t)pos * NKV + hh];
        int8_t wantq = q8one(hK[(size_t)hh * D + d] / sk);
        if (slot[(size_t)hh * D + d] != wantq) {
          ok = false;
          std::fprintf(stderr, "slot bits mismatch it %d h%d d%d\n", it, hh,
                       d);
          break;
        }
      }
    }
    if (!ok)
      break;
    // Determinism replay.
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt2.data(), dAtt,
                                        hAtt2.size() * 4, nullptr, 0,
                                        nullptr));
    double refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (size_t j = 0; j < hRef.size(); ++j) {
      double rel = std::fabs((double)hAtt[j] - (double)hRef[j]) /
                   (refmax > 0 ? refmax : 1);
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-5) {
        ok = false;
        std::fprintf(stderr, "ref mismatch it %d j %zu: got %g want %g\n", it,
                     j, hAtt[j], hRef[j]);
        break;
      }
      uint32_t a, b;
      __builtin_memcpy(&a, &hAtt[j], 4);
      __builtin_memcpy(&b, &hAtt2[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "nondeterministic replay it %d j %zu\n", it, j);
        break;
      }
    }
    if (!ok)
      break;
  }
  double mR = med(tRep);
  std::printf("attni8 med %.2f us/iter worst-rel %.2e %s\n", mR / 1e3, worstRel,
              ok ? "ATTNI8-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernels\":[\"_ZTS10KvAppendI8\","
                "\"_ZTS10AttnCoreI8\"],\"heads\":\"24Q/4KV\",\"tmax\":%d,"
                "\"iters\":%d,\"replay_us\":%.2f,\"worst_rel\":%.2e,"
                "\"ref_tol\":1e-5,\"bitwise_slots\":%s,"
                "\"bitwise_deterministic\":%s,\"attni8_ok\":%s}",
                TMAX, IT, mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false", ok ? "true" : "false");
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
