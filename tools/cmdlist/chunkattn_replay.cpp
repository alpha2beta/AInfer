// T7.4 chunked prefill: record ONCE a closed regular L0 list launching the
// causal chunk-attention kernel (ChunkAttn: 256-token chunk over a 128-deep
// cache prefix, 24Q/4KV GQA + gate, control-driven chunk_start), replay 8x
// over varying Q/gate chunks. Checks: tight-tolerance match vs order-mirrored
// host float ref (softmax exp last-ulp) + bitwise replay determinism.
// Caches stay float here (chunking math isolated from the BF16-KV decision).
// Usage: chunkattn_replay <chunkattn.spv> [report.json]
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

static double now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
// BF16 caches (production alignment with decode): device stores RNE, so the
// host reference sees exactly the quantized values (attn_replay pattern).
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: chunkattn_replay <chunkattn.spv> [report]\n");
    return 2;
  }
  const int M = 256, P0 = 128, TC = P0 + M, D = 256, NH = 24, NKV = 4, IT = 8;
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
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * 6144 * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * 6144 * 4, 4096, dev, &dG));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * 6144 * 4, 4096, dev, &dAtt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TC * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NKV * TC * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * 24 * TC * 4, 4096, dev,
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
                            "_ZTS9ChunkAttn"};
  if (zeKernelCreate(mod, &kdesc, &ker) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS9ChunkAttn not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(ker, 1, 1, 1));
  int tmaxArg = TC;
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
  ze_group_count_t gc = {(uint32_t)(M * 24), 1, 1};
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

  uint64_t s = 0xCA7C;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hQ(M * 6144), hG(M * 6144), hAtt(M * 6144),
      hAtt2(M * 6144), hRef(M * 6144), hKc(NKV * TC * D), hVc(NKV * TC * D);
  for (auto &v : hKc)
    v = rnd() * 0.5f;
  for (auto &v : hVc)
    v = rnd() * 0.5f;
  for (auto &v : hKc)
    v = qb(v); // reference sees exactly what the BF16 cache stores
  for (auto &v : hVc)
    v = qb(v);
  std::vector<uint16_t> hKcB(hKc.size()), hVcB(hVc.size());
  for (size_t i = 0; i < hKc.size(); ++i)
    hKcB[i] = f32_to_bf16(hKc[i]);
  for (size_t i = 0; i < hVc.size(); ++i)
    hVcB[i] = f32_to_bf16(hVc[i]);
  CHECK(zeCommandListAppendMemoryCopy(up, dKc, hKcB.data(), hKcB.size() * 2,
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dVc, hVcB.data(), hVcB.size() * 2,
                                      nullptr, 0, nullptr));
  DecodeControl c{0, P0, P0 + 1, -1};
  CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                      nullptr));

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> wts(TC);
  for (int it = 0; it < IT; ++it) {
    for (auto &v : hQ)
      v = rnd() * 2.0f;
    for (auto &v : hG)
      v = rnd();
    for (int m = 0; m < M; ++m) {
      int T = P0 + m + 1;
      for (int hh = 0; hh < NH; ++hh) {
        int kv = hh / 6;
        float mx = -1e30f;
        for (int t = 0; t < T; ++t) {
          float sc = 0;
          for (int d = 0; d < D; ++d)
            sc += hQ[((size_t)m * 6144) + hh * D + d] *
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
          float g = hG[((size_t)m * 6144) + hh * D + d];
          hRef[((size_t)m * 6144) + hh * D + d] =
              acc / (1.0f + expf(-g));
        }
      }
    }
    CHECK(zeCommandListAppendMemoryCopy(up, dQ, hQ.data(), hQ.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dG, hG.data(), hG.size() * 4,
                                        nullptr, 0, nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt.data(), dAtt, hAtt.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListAppendMemoryCopy(up, hAtt2.data(), dAtt,
                                        hAtt2.size() * 4, nullptr, 0, nullptr));
    double refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (size_t j = 0; j < hRef.size(); ++j) {
      double rel = std::fabs((double)hAtt[j] - (double)hRef[j]) / refmax;
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
  std::printf("chunkattn med %.2f us/iter worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "CHUNKATTN-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS9ChunkAttn\","
                "\"chunk\":%d,\"prefix\":%d,\"heads\":\"24Q/4KV\","
                "\"iters\":%d,\"replay_us\":%.2f,\"worst_rel\":%.2e,"
                "\"ref_tol\":1e-5,\"bitwise_deterministic\":%s,"
                "\"chunkattn_ok\":%s}",
                M, P0, IT, mR / 1e3, worstRel, ok ? "true" : "false",
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
