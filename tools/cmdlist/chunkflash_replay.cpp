// T7.4 fused FlashAttention probe: ChunkFlashAttn (online-softmax QK+PV,
// no score matrix) vs host fp32 FA reference on synthetic decode-layout
// KV cache. Tolerance gate worst-rel 1e-5 (online summation order differs
// from row-softmax at 1-ulp level — NOT bitwise by design). Two configs:
// exact tiles (M=32) + partial tail tile (M=20).
// Usage: chunkflash_replay <chunkflashattn.spv> [report.json]
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

static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
static uint16_t f32_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
// True IEEE fp16 encode/decode (Q buffer is read as sycl::half on device;
// BF16-rounded bits would decode to wrong values).
static uint16_t f32_to_f16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t s = (u >> 16) & 0x8000u;
  int e = (int)((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t m = u & 0x7FFFFFu;
  if (e <= 0)
    return (uint16_t)s; // underflow -> signed zero
  if (e >= 31)
    return (uint16_t)(s | 0x7BFFu); // overflow -> max finite
  uint32_t mm = m >> 13;
  uint32_t r = (m >> 12) & 1u, rest = m & 0xFFFu;
  if (r && (rest > 0 || (mm & 1u)))
    ++mm;
  if (mm >= 0x400u) {
    ++e;
    mm = 0;
    if (e >= 31)
      return (uint16_t)(s | 0x7BFFu);
  }
  return (uint16_t)(s | ((uint32_t)e << 10) | mm);
}
static float f16_to_f32(uint16_t h) {
  uint32_t s = ((uint32_t)h & 0x8000u) << 16;
  int e = (int)(((uint32_t)h >> 10) & 0x1Fu);
  uint32_t m = (uint32_t)h & 0x3FFu;
  uint32_t u;
  if (e == 0)
    u = s | (m << 13); // subnormal -> flush toward zero exponent form
  else if (e == 31)
    u = s | 0x7F800000u | (m << 13);
  else
    u = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
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

static uint64_t prng = 0x51ab3f019c7d2e44ull;
static float frnd(float s) {
  prng = prng * 6364136223846793005ull + 1442695040888963407ull;
  return (float)((int)((prng >> 33) & 0xFFFF) - 32768) * (s / 32768.0f);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: chunkflash_replay <chunkflashattn.spv> [report]\n");
    return 2;
  }
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
  auto alloc = [&](size_t n) -> void * {
    void *p = nullptr;
    if (zeMemAllocDevice(ctx, &mdesc, n, 4096, dev, &p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "alloc %zu failed\n", n);
      std::exit(1);
    }
    return p;
  };

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
                            "_ZTS14ChunkFlashAttn"};
  if (zeKernelCreate(mod, &kdesc, &ker) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS14ChunkFlashAttn not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(ker, 16, 1, 1));

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_command_list_handle_t up = nullptr;
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                  0, 0};
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &up));
  auto upload = [&](void *d, const void *h, size_t n) {
    CHECK(zeCommandListAppendMemoryCopy(up, d, h, n, nullptr, 0, nullptr));
  };

  bool all_ok = true;
  double worst_all = 0, t_med = 0;
  // Configs: {M rows, W slots, P prefix, magnitude}. Causal: t < P+m+1.
  const int cfgs[][4] = {{32, 1024, 512, 1}, {20, 768, 384, 1}, {256, 4096, 3840, 1}};
  for (int ci = 0; ci < 3; ++ci) {
    const int M = cfgs[ci][0], W = cfgs[ci][1], P = cfgs[ci][2];
    const float MAG = (float)cfgs[ci][3];
    const int KB = 512, NQ = M * 24;
    // Host data: Q fp16 (M*24 x 256), BF16 decode-layout KV cache.
    std::vector<uint16_t> hQ((size_t)NQ * 256), hKc((size_t)W * 4 * 256),
        hVc((size_t)W * 4 * 256);
    std::vector<float> hRef((size_t)NQ * 256, 0), hGot((size_t)NQ * 256, 0);
    // Outlier channels (every 32nd dim x40) mimic real activation outliers.
    for (size_t i = 0; i < hQ.size(); ++i) {
      float v = frnd(MAG);
      if (MAG > 1.5f && (i % 256) % 32 == 0)
        v *= 40.0f;
      hQ[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < hKc.size(); ++i) {
      float vk = frnd(MAG), vv = frnd(MAG);
      if (MAG > 1.5f && (i % 256) % 32 == 0) {
        vk *= 40.0f;
        vv *= 40.0f;
      }
      hKc[i] = f32_to_bf16(vk);
      hVc[i] = f32_to_bf16(vv);
    }
    // Host fp32 reference: per (m,hh), online softmax over valid keys.
    auto bf = [&](uint16_t b) { return bf16_to_f32(b); };
    auto hf = [&](uint16_t b) { return f16_to_f32(b); };
    for (int m = 0; m < M; ++m) {
      for (int hh = 0; hh < 24; ++hh) {
        int k2 = hh / 6, tlim = P + m + 1;
        double mx = -1e30, ls = 0;
        std::vector<double> o(256, 0.0);
        for (int t = 0; t < tlim && t < W; ++t) {
          double s = 0;
          for (int d = 0; d < 256; ++d)
            s += (double)hf(hQ[((size_t)m * 24 + hh) * 256 + d]) *
                 (double)bf(hKc[((size_t)t * 4 + k2) * 256 + d]);
          s /= 16.0;
          double m_new = s > mx ? s : mx;
          double resc = (mx <= -1e29) ? 0.0 : std::exp(mx - m_new);
          ls = ls * resc + std::exp(s - m_new);
          for (int d = 0; d < 256; ++d)
            o[d] = o[d] * resc +
                   std::exp(s - m_new) *
                       (double)bf(hVc[((size_t)t * 4 + k2) * 256 + d]);
          mx = m_new;
        }
        for (int d = 0; d < 256; ++d)
          hRef[((size_t)m * 24 + hh) * 256 + d] =
              (float)(ls > 0 ? o[d] / ls : 0.0);
      }
    }
    void *dQ = alloc(hQ.size() * 2), *dKc = alloc(hKc.size() * 2),
         *dVc = alloc(hVc.size() * 2), *dO = alloc(hGot.size() * 4);
    upload(dQ, hQ.data(), hQ.size() * 2);
    upload(dKc, hKc.data(), hKc.size() * 2);
    upload(dVc, hVc.data(), hVc.size() * 2);
    // Flush uploads: close/execute/sync, then reset list for reuse.
    CHECK(zeCommandListClose(up));
    {
      ze_fence_handle_t uf = nullptr;
      ze_fence_desc_t ufd = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
      CHECK(zeFenceCreate(qq, &ufd, &uf));
      CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &up, uf));
      CHECK(zeFenceHostSynchronize(uf, UINT64_MAX));
      CHECK(zeFenceDestroy(uf));
    }
    CHECK(zeCommandListReset(up));
    int iP = P, iM = M, iW = W, iKB = KB;
    CHECK(zeKernelSetArgumentValue(ker, 0, sizeof(void *), &dQ));
    CHECK(zeKernelSetArgumentValue(ker, 1, sizeof(void *), &dKc));
    CHECK(zeKernelSetArgumentValue(ker, 2, sizeof(void *), &dVc));
    CHECK(zeKernelSetArgumentValue(ker, 3, sizeof(void *), &dO));
    CHECK(zeKernelSetArgumentValue(ker, 4, sizeof(int), &iP));
    CHECK(zeKernelSetArgumentValue(ker, 5, sizeof(int), &iM));
    CHECK(zeKernelSetArgumentValue(ker, 6, sizeof(int), &iW));
    CHECK(zeKernelSetArgumentValue(ker, 7, sizeof(int), &iKB));
    CHECK(zeKernelSetArgumentValue(ker, 8, (size_t)8 * 256 * 2, nullptr));
    CHECK(zeKernelSetArgumentValue(ker, 9, (size_t)256 * 16 * 2, nullptr));
    CHECK(zeKernelSetArgumentValue(ker, 10, (size_t)16 * 256 * 2, nullptr));
    CHECK(zeKernelSetArgumentValue(ker, 11, (size_t)8 * 16 * 4, nullptr));
    CHECK(zeKernelSetArgumentValue(ker, 12, (size_t)8 * 256 * 4, nullptr));
    int nG = (M + 7) / 8;
    ze_command_list_handle_t ex = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &ldesc, &ex));
    ze_group_count_t gc = {(uint32_t)nG * 24, 1, 1};
    CHECK(zeCommandListAppendLaunchKernel(ex, ker, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(ex, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(ex));
    ze_fence_handle_t fence = nullptr;
    ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    CHECK(zeFenceCreate(qq, &fdesc, &fence));
    std::vector<double> ts;
    for (int r = 0; r < 5; ++r) {
      double t0 = now_ns();
      CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &ex, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      ts.push_back(now_ns() - t0);
    }
    double tm = med(ts);
    if (ci == 0)
      t_med = tm;
    CHECK(zeCommandListAppendMemoryCopy(up, hGot.data(), dO, hGot.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListClose(up));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &up, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListReset(up));
    double worst = 0, refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    long nmis = 0;
    for (size_t i = 0; i < hRef.size(); ++i) {
      double rel = std::fabs((double)hGot[i] - (double)hRef[i]) / refmax;
      if (rel > worst)
        worst = rel;
      if (rel > 5e-6) { // online order: 1-ulp class (258V review same)
        if (nmis < 5)
          std::printf("DIFF cfg%d i %zu: got %g want %g\n", ci, i, hGot[i],
                      hRef[i]);
        ++nmis;
      }
    }
    if (worst > worst_all)
      worst_all = worst;
    std::printf("cfg M=%d W=%d P=%d: med %.1f us worst-rel %.2e mis=%ld %s\n",
                M, W, P, tm / 1000.0, worst, nmis,
                nmis == 0 ? "FLASH-OK" : "FAIL");
    if (nmis != 0)
      all_ok = false;
  }
  if (argc >= 3) {
    FILE *rf = std::fopen(argv[2], "w");
    if (rf) {
      std::fprintf(rf,
                   "{\"device\":\"B60\",\"kernel\":\"_ZTS14ChunkFlashAttn\","
                   "\"worst_rel\":%.3g,\"med_us\":%.1f,\"ok\":%s}\n",
                   worst_all, t_med / 1000.0, all_ok ? "true" : "false");
      std::fclose(rf);
    }
  }
  std::printf(all_ok ? "FLASH-ALL-OK\n" : "FLASH-FAIL\n");
  return all_ok ? 0 : 1;
}
