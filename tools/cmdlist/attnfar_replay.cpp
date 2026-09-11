// T7.4 long-context scan point: AttnCore at T={1024,2048,4096} in a recorded
// list (same attn.spv, TMAXF=4096 sizing). Correctness vs full host ref +
// timing grounds the long-context decode projection (T6.2 sweep stopped at
// P=256; a full 64K scalar scan is hours/token, so 4K is the honest probe).
// Usage: attnfar_replay <attn.spv> [report.json]
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
static inline float qb(float v) { return bf16_to_f32(f32_to_bf16(v)); }
static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: attnfar_replay <attn.spv> [report.json]\n");
    return 2;
  }
  const int NH = 24, NKV = 4, D = 256, TMAXF = 4096;
  const int TS[] = {1024, 2048, 4096};
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
  void *dQ = nullptr, *dG = nullptr, *dAtt = nullptr, *dKc = nullptr,
       *dVc = nullptr, *dCtrl = nullptr, *dWts = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dG));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dAtt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAXF * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TMAXF * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)24 * TMAXF * 4, 4096, dev,
                         &dWts));

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
                            "_ZTS8AttnCore"};
  CHECK(zeKernelCreate(mod, &kdesc, &ker));
  CHECK(zeKernelSetGroupSize(ker, 1, 1, 1));
  int tmaxArg = TMAXF;
  CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dAtt));
  CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dKc));
  CHECK(zeKernelSetArgumentValue(ker, 3, sizeof(void *), &dVc));
  CHECK(zeKernelSetArgumentValue(ker, 4, sizeof(void *), &dG));
  CHECK(zeKernelSetArgumentValue(ker, 5, sizeof(void *), &dCtrl));
  CHECK(zeKernelSetArgumentValue(ker, 6, sizeof(int), &tmaxArg));
  CHECK(zeKernelSetArgumentValue(ker, 7, sizeof(void *), &dWts));

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
  ze_group_count_t gc = {(uint32_t)NH, 1, 1};
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

  uint64_t s = 0xA77FA4;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(NH * D), hG(NH * D), hAtt(NH * D), hRef(NH * D),
      hKc((size_t)NKV * TMAXF * D), hVc((size_t)NKV * TMAXF * D);
  for (auto &v : hKc)
    v = qb(rnd());
  for (auto &v : hVc)
    v = qb(rnd());
  std::vector<uint16_t> hKcB(hKc.size()), hVcB(hVc.size());
  for (size_t i = 0; i < hKc.size(); ++i)
    hKcB[i] = f32_to_bf16(hKc[i]);
  for (size_t i = 0; i < hVc.size(); ++i)
    hVcB[i] = f32_to_bf16(hVc[i]);
  CHECK(zeCommandListAppendMemoryCopy(up, dKc, hKcB.data(), hKcB.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dVc, hVcB.data(), hVcB.size() * 2,
                                      nullptr, 0, nullptr));

  bool ok = true;
  double worstRel = 0, us4096 = 0;
  std::vector<float> wts(TMAXF);
  for (int T : TS) {
    for (auto &v : hQ)
      v = rnd() * 2.0f;
    for (auto &v : hG)
      v = rnd();
    DecodeControl c{9000 + T, T - 1, T, -1};
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
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    double us = (now_ns() - t0) / 1e3;
    if (T == 4096)
      us4096 = us;
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt.data(), dAtt, hAtt.size() * 4,
                                        nullptr, 0, nullptr));
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
        std::fprintf(stderr, "ref mismatch T=%d j %zu: got %g want %g\n", T, j,
                     hAtt[j], hRef[j]);
        break;
      }
    }
    if (!ok)
      break;
    std::printf("attnfar T=%d %.2f ms\n", T, us / 1e3);
  }
  std::printf("attnfar worst-rel %.2e %s\n", worstRel,
              ok ? "ATTNFAR-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS8AttnCore\","
                "\"t_points\":[1024,2048,4096],\"tmax\":%d,"
                "\"us_at_4096\":%.1f,\"worst_rel\":%.2e,\"ref_tol\":1e-5,"
                "\"attnfar_ok\":%s}",
                TMAXF, us4096, worstRel, ok ? "true" : "false");
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
