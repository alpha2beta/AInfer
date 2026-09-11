// T7.4 chunked prefill: SSM chunk orchestration. Record ONCE a closed list
// of ChunkSsmConv (k4+silu, history persists) + ChunkSsmRecur (48-head delta
// rule, state persists), replay a 2-chunk sequence with states CARRYING from
// chunk A to chunk B (cross-chunk continuity). Checks per chunk: match vs
// order-mirrored host float ref (tol 1e-4, recurrence compounds) + full
// reset-rerun bitwise identical. Extra proof: chunk B replayed alone from
// zeroed states must DIFFER (state genuinely carried, not recomputed).
// Usage: chunkssm_replay <chunkssmconv.spv> <chunkssmrecur.spv> [report.json]
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
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: chunkssm_replay <conv.spv> <recur.spv> [report]\n");
    return 2;
  }
  const int C = 10240, NH = 48, D = 128, M = 32;
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
  std::vector<uint8_t> spvC = load_spv(argv[1]);
  std::vector<uint8_t> spvR = load_spv(argv[2]);

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
  void *dQKV = nullptr, *dMx = nullptr, *dCS = nullptr, *dCW = nullptr;
  void *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dS = nullptr,
       *dBt = nullptr, *dGt = nullptr, *dOut = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * C * 4, 4096, dev, &dQKV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * C * 4, 4096, dev, &dMx));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 3 * 4, 4096, dev, &dCS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 4 * 4, 4096, dev, &dCW));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * D * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * D * 4, 4096, dev, &dK));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * D * 4, 4096, dev, &dV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * D * 4, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * 4, 4096, dev, &dBt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * 4, 4096, dev, &dGt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)M * NH * D * 4, 4096, dev,
                         &dOut));

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
  ze_module_handle_t modC = mkmod(spvC), modR = mkmod(spvR);
  ze_kernel_handle_t kConv = nullptr, kRec = nullptr;
  ze_kernel_desc_t kd1 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS12ChunkSsmConv"};
  ze_kernel_desc_t kd2 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS13ChunkSsmRecur"};
  if (zeKernelCreate(modC, &kd1, &kConv) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modR, &kd2, &kRec) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "chunkssm entry point not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kConv, 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kRec, 1, 1, 1));
  int cArg = C, mArg = M;
  CHECK(zeKernelSetArgumentValue(kConv, 0, sizeof(void *), &dMx));
  CHECK(zeKernelSetArgumentValue(kConv, 1, sizeof(void *), &dQKV));
  CHECK(zeKernelSetArgumentValue(kConv, 2, sizeof(void *), &dCS));
  CHECK(zeKernelSetArgumentValue(kConv, 3, sizeof(void *), &dCW));
  CHECK(zeKernelSetArgumentValue(kConv, 4, sizeof(int), &cArg));
  CHECK(zeKernelSetArgumentValue(kConv, 5, sizeof(int), &mArg));
  CHECK(zeKernelSetArgumentValue(kRec, 0, sizeof(void *), &dOut));
  CHECK(zeKernelSetArgumentValue(kRec, 1, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(kRec, 2, sizeof(void *), &dK));
  CHECK(zeKernelSetArgumentValue(kRec, 3, sizeof(void *), &dV));
  CHECK(zeKernelSetArgumentValue(kRec, 4, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kRec, 5, sizeof(void *), &dBt));
  CHECK(zeKernelSetArgumentValue(kRec, 6, sizeof(void *), &dGt));
  CHECK(zeKernelSetArgumentValue(kRec, 7, sizeof(int), &mArg));

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
  ze_group_count_t gC = {(uint32_t)C, 1, 1};
  ze_group_count_t gR = {(uint32_t)NH, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(reg, kConv, &gC, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendLaunchKernel(reg, kRec, &gR, nullptr, 0, nullptr));
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

  uint64_t s = 0x55AA55;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hCW(C * 4), hCS(C * 3, 0.0f), hS(NH * D * D, 0.0f);
  for (auto &v : hCW)
    v = rnd() * 0.5f;
  CHECK(zeCommandListAppendMemoryCopy(up, dCW, hCW.data(), hCW.size() * 4,
                                      nullptr, 0, nullptr));
  struct Chunk {
    std::vector<float> qkv, q, k, v, bt, gt;
  };
  auto mkchunk = [&]() {
    Chunk c;
    c.qkv.assign(M * C, 0);
    c.q.assign(M * NH * D, 0);
    c.k.assign(M * NH * D, 0);
    c.v.assign(M * NH * D, 0);
    c.bt.assign(M * NH, 0);
    c.gt.assign(M * NH, 0);
    for (auto &v : c.qkv)
      v = rnd();
    for (auto &v : c.q)
      v = rnd() * 0.2f;
    for (auto &v : c.k)
      v = rnd() * 0.2f;
    for (auto &v : c.v)
      v = rnd();
    for (auto &v : c.bt)
      v = std::fabs(rnd()) * 0.5f;
    for (auto &v : c.gt)
      v = -std::fabs(rnd()) * 0.1f;
    return c;
  };
  Chunk A = mkchunk(), B = mkchunk();
  std::vector<float> hOut(M * NH * D), hRef(M * NH * D), hMx(M * C);

  // Host reference mirrors device order (float, expf), persistent states.
  auto host_chunk = [&](const Chunk &c) {
    for (int cc = 0; cc < C; ++cc)
      for (int m = 0; m < M; ++m) {
        float acc = hCS[cc * 3 + 0] * hCW[(size_t)cc * 4 + 0] +
                    hCS[cc * 3 + 1] * hCW[(size_t)cc * 4 + 1] +
                    hCS[cc * 3 + 2] * hCW[(size_t)cc * 4 + 2] +
                    c.qkv[(size_t)m * C + cc] * hCW[(size_t)cc * 4 + 3];
        hMx[(size_t)m * C + cc] = acc / (1.0f + expf(-acc));
        hCS[cc * 3 + 0] = hCS[cc * 3 + 1];
        hCS[cc * 3 + 1] = hCS[cc * 3 + 2];
        hCS[cc * 3 + 2] = c.qkv[(size_t)m * C + cc];
      }
    for (int hh = 0; hh < NH; ++hh) {
      float *S0 = hS.data() + (size_t)hh * D * D;
      for (int m = 0; m < M; ++m) {
        const float *qh = c.q.data() + ((size_t)m * NH + hh) * D;
        const float *kh = c.k.data() + ((size_t)m * NH + hh) * D;
        const float *vh = c.v.data() + ((size_t)m * NH + hh) * D;
        float gt = expf(c.gt[(size_t)m * NH + hh]);
        float bt = c.bt[(size_t)m * NH + hh];
        for (int i = 0; i < D * D; ++i)
          S0[i] *= gt;
        float kv[128];
        for (int v = 0; v < D; ++v) {
          float a = 0;
          for (int k = 0; k < D; ++k)
            a += S0[k * D + v] * kh[k];
          kv[v] = a;
        }
        for (int v = 0; v < D; ++v)
          kv[v] = (vh[v] - kv[v]) * bt;
        for (int k = 0; k < D; ++k)
          for (int v = 0; v < D; ++v)
            S0[k * D + v] += kh[k] * kv[v];
        float *oh = hRef.data() + ((size_t)m * NH + hh) * D;
        for (int v = 0; v < D; ++v) {
          float a = 0;
          for (int k = 0; k < D; ++k)
            a += S0[k * D + v] * qh[k];
          oh[v] = a;
        }
      }
    }
  };
  auto upload_chunk = [&](const Chunk &c) {
    CHECK(zeCommandListAppendMemoryCopy(up, dQKV, c.qkv.data(),
                                        c.qkv.size() * 4, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dQ, c.q.data(), c.q.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dK, c.k.data(), c.k.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dV, c.v.data(), c.v.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dBt, c.bt.data(), c.bt.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dGt, c.gt.data(), c.gt.size() * 4,
                                        nullptr, 0, nullptr));
  };
  auto zero_states = [&]() {
    std::fill(hCS.begin(), hCS.end(), 0.0f);
    std::fill(hS.begin(), hS.end(), 0.0f);
    CHECK(zeCommandListAppendMemoryCopy(up, dCS, hCS.data(), hCS.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dS, hS.data(), hS.size() * 4,
                                        nullptr, 0, nullptr));
  };
  auto run_chunk = [&](const Chunk &c, std::vector<float> &out,
                       std::vector<double> *timing) {
    upload_chunk(c);
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    if (timing)
      timing->push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, out.data(), dOut, out.size() * 4,
                                        nullptr, 0, nullptr));
  };
  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> outA(M * NH * D), outB(M * NH * D);
  auto check_full = [&](const char *tag) {
    // Device output (hOut) vs current host ref (hRef), tol 1e-4.
    double refmax = 0;
    for (float v : hRef)
      refmax = std::max(refmax, (double)std::fabs(v));
    for (size_t j = 0; j < hRef.size(); ++j) {
      double rel = std::fabs((double)hOut[j] - (double)hRef[j]) / refmax;
      if (rel > worstRel)
        worstRel = rel;
      if (rel > 1e-4) {
        ok = false;
        std::fprintf(stderr, "ref mismatch %s j %zu: got %g want %g\n", tag, j,
                     hOut[j], hRef[j]);
        break;
      }
    }
  };
  auto check_bits = [&](const std::vector<float> &want, const char *tag) {
    for (size_t j = 0; j < want.size(); ++j) {
      uint32_t a, b;
      __builtin_memcpy(&a, &hOut[j], 4);
      __builtin_memcpy(&b, &want[j], 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "reset mismatch %s j %zu\n", tag, j);
        break;
      }
    }
  };
  // Run 1: A then B (states carry A->B: cross-chunk continuity).
  zero_states();
  host_chunk(A);
  run_chunk(A, hOut, &tRep);
  check_full("A");
  if (ok)
    outA = hOut;
  host_chunk(B);
  run_chunk(B, hOut, &tRep);
  check_full("B-after-A");
  if (ok)
    outB = hOut;
  // Continuity proof: B alone from zeroed states must DIFFER (state carried).
  bool differs = false;
  if (ok) {
    zero_states();
    run_chunk(B, hOut, nullptr);
    for (size_t j = 0; j < hOut.size(); j += 97)
      if (hOut[j] != outB[j]) {
        differs = true;
        break;
      }
    if (!differs) {
      ok = false;
      std::fprintf(stderr, "continuity failure: B-alone equals B-after-A\n");
    }
    if (ok)
      std::printf("continuity: B-after-A differs from B-alone (state live)\n");
  }
  // Run 2: reset + identical A,B -> bitwise identical (determinism; no host
  // involvement needed — pure replay equality).
  if (ok) {
    zero_states();
    run_chunk(A, hOut, nullptr);
    check_bits(outA, "A");
  }
  if (ok) {
    run_chunk(B, hOut, nullptr);
    check_bits(outB, "B");
  }

  double mR = med(tRep);
  std::printf("chunkssm med %.2f us/chunk worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "CHUNKSSM-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernels\":[\"_ZTS12ChunkSsmConv\","
                "\"_ZTS13ChunkSsmRecur\"],\"chunk\":%d,\"chunks\":2,"
                "\"replay_us\":%.2f,\"worst_rel\":%.2e,\"ref_tol\":1e-4,"
                "\"cross_chunk_continuity\":%s,\"reset_deterministic\":%s,"
                "\"chunkssm_ok\":%s}",
                M, mR / 1e3, worstRel, differs ? "true" : "false",
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
