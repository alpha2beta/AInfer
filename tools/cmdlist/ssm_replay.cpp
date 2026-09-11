// T5.3 SSM port: record ONCE a closed regular L0 list of the stateful
// linear-attention core (SsmConv k4+silu with persistent history, then
// SsmRecur 48-head FP32 delta-rule with persistent 128x128 state), replay an
// 8-step input sequence with states PERSISTING across replays (zero-init at
// start, like decode). Checks per step: tight-tolerance match vs an
// order-mirrored host float reference (device exp last-ulp; recurrence
// compounds over steps). Then states are zero-reset and the identical sequence
// reruns: outputs must be bitwise identical (determinism + proof the state is
// live device memory, not baked record-time data).
// Usage: ssm_replay <ssmconv.spv> <ssmrecur.spv> [report.json]
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
    std::fprintf(stderr, "usage: ssm_replay <ssmconv.spv> <ssmrecur.spv> "
                         "[report.json]\n");
    return 2;
  }
  const int C = 10240, NH = 48, D = 128, STEPS = 8;
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
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 4, 4096, dev, &dQKV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 4, 4096, dev, &dMx));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 3 * 4, 4096, dev, &dCS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)C * 4 * 4, 4096, dev, &dCW));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dQ));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dK));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * D * 4, 4096, dev, &dS));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * 4, 4096, dev, &dBt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * 4, 4096, dev, &dGt));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NH * D * 4, 4096, dev, &dOut));

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
                          "_ZTS7SsmConv"};
  ze_kernel_desc_t kd2 = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                          "_ZTS8SsmRecur"};
  if (zeKernelCreate(modC, &kd1, &kConv) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(modR, &kd2, &kRec) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "ssm entry point not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kConv, 1, 1, 1));
  CHECK(zeKernelSetGroupSize(kRec, 1, 1, 1));
  int dArg = D;
  CHECK(zeKernelSetArgumentValue(kConv, 0, sizeof(void *), &dMx));
  CHECK(zeKernelSetArgumentValue(kConv, 1, sizeof(void *), &dQKV));
  CHECK(zeKernelSetArgumentValue(kConv, 2, sizeof(void *), &dCS));
  CHECK(zeKernelSetArgumentValue(kConv, 3, sizeof(void *), &dCW));
  CHECK(zeKernelSetArgumentValue(kRec, 0, sizeof(void *), &dOut));
  CHECK(zeKernelSetArgumentValue(kRec, 1, sizeof(void *), &dQ));
  CHECK(zeKernelSetArgumentValue(kRec, 2, sizeof(void *), &dK));
  CHECK(zeKernelSetArgumentValue(kRec, 3, sizeof(void *), &dV));
  CHECK(zeKernelSetArgumentValue(kRec, 4, sizeof(void *), &dS));
  CHECK(zeKernelSetArgumentValue(kRec, 5, sizeof(void *), &dBt));
  CHECK(zeKernelSetArgumentValue(kRec, 6, sizeof(void *), &dGt));
  CHECK(zeKernelSetArgumentValue(kRec, 7, sizeof(int), &dArg));

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

  uint64_t s = 0x55AA;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hCW(C * 4), hCS(C * 3, 0.0f);
  std::vector<float> hS(NH * D * D, 0.0f);
  std::vector<float> hQKV(C), hQ(NH * D), hK(NH * D), hV(NH * D),
      hBt(NH), hGt(NH), hOut(NH * D), hRef(NH * D), hMx(C);
  std::vector<std::vector<float>> seqOut; // per-step outputs, run 1
  for (auto &v : hCW)
    v = rnd() * 0.5f;
  CHECK(zeCommandListAppendMemoryCopy(up, dCW, hCW.data(), hCW.size() * 4,
                                      nullptr, 0, nullptr));
  // Fixed input sequence (same both runs; states reset between runs).
  struct Step {
    std::vector<float> qkv, q, k, v, bt, gt;
  };
  std::vector<Step> seq(STEPS);
  for (auto &st : seq) {
    st.qkv.assign(C, 0);
    st.q.assign(NH * D, 0);
    st.k.assign(NH * D, 0);
    st.v.assign(NH * D, 0);
    st.bt.assign(NH, 0);
    st.gt.assign(NH, 0);
    for (auto &v : st.qkv)
      v = rnd();
    for (auto &v : st.q)
      v = rnd() * 0.2f;
    for (auto &v : st.k)
      v = rnd() * 0.2f;
    for (auto &v : st.v)
      v = rnd();
    for (auto &v : st.bt)
      v = std::fabs(rnd()) * 0.5f;
    for (auto &v : st.gt)
      v = -std::fabs(rnd()) * 0.1f;
  }

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  auto zero_states = [&]() {
    std::fill(hCS.begin(), hCS.end(), 0.0f);
    std::fill(hS.begin(), hS.end(), 0.0f);
    CHECK(zeCommandListAppendMemoryCopy(up, dCS, hCS.data(), hCS.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dS, hS.data(), hS.size() * 4,
                                        nullptr, 0, nullptr));
  };
  // Host reference mirrors the device order in float (expf for exp).
  auto host_step = [&](const Step &st) {
    for (int c = 0; c < C; ++c) {
      float acc = hCS[c * 3 + 0] * hCW[(size_t)c * 4 + 0] +
                  hCS[c * 3 + 1] * hCW[(size_t)c * 4 + 1] +
                  hCS[c * 3 + 2] * hCW[(size_t)c * 4 + 2] +
                  st.qkv[c] * hCW[(size_t)c * 4 + 3];
      hMx[c] = acc / (1.0f + expf(-acc));
      hCS[c * 3 + 0] = hCS[c * 3 + 1];
      hCS[c * 3 + 1] = hCS[c * 3 + 2];
      hCS[c * 3 + 2] = st.qkv[c];
    }
    for (int hh = 0; hh < NH; ++hh) {
      float *S0 = hS.data() + (size_t)hh * D * D;
      const float *qh = st.q.data() + (size_t)hh * D,
                  *kh = st.k.data() + (size_t)hh * D,
                  *vh = st.v.data() + (size_t)hh * D;
      float gt = expf(st.gt[hh]), bt = st.bt[hh];
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
      for (int v = 0; v < D; ++v) {
        float a = 0;
        for (int k = 0; k < D; ++k)
          a += S0[k * D + v] * qh[k];
        hRef[(size_t)hh * D + v] = a;
      }
    }
  };

  for (int run = 0; run < 2 && ok; ++run) {
    zero_states();
    for (int step = 0; step < STEPS; ++step) {
      const Step &st = seq[step];
      host_step(st); // advances host states identically
      CHECK(zeCommandListAppendMemoryCopy(up, dQKV, st.qkv.data(), (size_t)C * 4,
                                          nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dQ, st.q.data(), st.q.size() * 4,
                                          nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dK, st.k.data(), st.k.size() * 4,
                                          nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dV, st.v.data(), st.v.size() * 4,
                                          nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dBt, st.bt.data(), (size_t)NH * 4,
                                          nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(up, dGt, st.gt.data(), (size_t)NH * 4,
                                          nullptr, 0, nullptr));
      double t0 = now_ns();
      CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      if (run == 0)
        tRep.push_back(now_ns() - t0);
      CHECK(zeCommandListAppendMemoryCopy(up, hOut.data(), dOut,
                                          hOut.size() * 4, nullptr, 0,
                                          nullptr));
      double refmax = 0;
      for (float v : hRef)
        refmax = std::max(refmax, (double)std::fabs(v));
      for (size_t j = 0; j < hRef.size(); ++j) {
        double rel = std::fabs((double)hOut[j] - (double)hRef[j]) /
                     (refmax > 0 ? refmax : 1);
        if (rel > worstRel)
          worstRel = rel;
        if (rel > 1e-4) {
          ok = false;
          std::fprintf(stderr, "ref mismatch run %d step %d j %zu: got %g "
                               "want %g\n", run, step, j, hOut[j], hRef[j]);
          break;
        }
      }
      if (!ok)
        break;
      if (run == 0)
        seqOut.push_back(hOut);
      else {
        // Reset determinism: identical sequence from zeroed states must be
        // bitwise identical (live state, no baked capture).
        for (size_t j = 0; j < hOut.size(); ++j) {
          uint32_t a, b;
          __builtin_memcpy(&a, &hOut[j], 4);
          __builtin_memcpy(&b, &seqOut[step][j], 4);
          if (a != b) {
            ok = false;
            std::fprintf(stderr, "reset mismatch step %d j %zu\n", step, j);
            break;
          }
        }
        if (!ok)
          break;
      }
    }
  }
  double mR = med(tRep);
  std::printf("ssm-replay med %.2f us/step worst-rel %.2e %s\n", mR / 1e3,
              worstRel, ok ? "SSM-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernels\":[\"_ZTS7SsmConv\","
                "\"_ZTS8SsmRecur\"],\"channels\":%d,\"heads\":%d,\"dim\":%d,"
                "\"steps\":%d,\"runs\":2,\"replay_us\":%.2f,"
                "\"worst_rel\":%.2e,\"ref_tol\":1e-4,"
                "\"reset_deterministic\":%s,\"ssm_ok\":%s}",
                C, NH, D, STEPS, mR / 1e3, worstRel, ok ? "true" : "false",
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
