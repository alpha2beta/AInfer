// T5.4: device-visible control buffer proof (Stamp 7 plan).
// Records a position-dependent kernel (ControlAdd: Out[i] = In[i] + Ctrl[1])
// ONCE in a closed regular L0 list with FIXED argument addresses, then replays
// it 50x with only the control-block CONTENTS changing (no kernel-arg
// mutation, no list rebuild). Two update paths compete:
//   explicit: control lives in device memory, updated per replay by a 16 B
//             immediate H2D copy;
//   shared:   control lives in zeMemAllocShared memory, updated by direct host
//             write (no copy command).
// Every replay is read back and checked bitwise against Out[i] = In[i] + pos:
// a stale capture (record-time position baked in) fails loudly.
// Usage: control_replay <control.spv> [report.json]
#include <level_zero/ze_api.h>

#include <algorithm>
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

// DecodeControl layout (Stamp 7): token_id, position, active_length,
// selected_token. The kernel reads word 1 (position).
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: control_replay <control.spv> [report.json]\n");
    return 2;
  }
  const int N = 5120, GS = 256, NG = N / GS, IT = 50;
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
  void *dIn = nullptr, *dOut = nullptr, *dCtrlDev = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * 4, 4096, dev, &dIn));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)N * 4, 4096, dev, &dOut));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrlDev));
  ze_host_mem_alloc_desc_t hdesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                    nullptr, 0};
  void *dCtrlShr = nullptr;
  CHECK(zeMemAllocShared(ctx, &mdesc, &hdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrlShr));

  ze_module_handle_t mod = nullptr;
  ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spvN,
                             spv.data(),
                             nullptr,
                             nullptr};
  CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
  // One kernel handle per path (L0 bakes args at append; each recorded list
  // pins its own control address — mutating args post-close is NOT the path).
  ze_kernel_handle_t kDev = nullptr, kShr = nullptr;
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                            "_ZTS10ControlAdd"};
  if (zeKernelCreate(mod, &kdesc, &kDev) != ZE_RESULT_SUCCESS ||
      zeKernelCreate(mod, &kdesc, &kShr) != ZE_RESULT_SUCCESS) {
    std::fprintf(stderr, "kernel _ZTS10ControlAdd not found\n");
    return 1;
  }
  CHECK(zeKernelSetGroupSize(kDev, GS, 1, 1));
  CHECK(zeKernelSetGroupSize(kShr, GS, 1, 1));
  CHECK(zeKernelSetArgumentValue(kDev, 0, sizeof(void *), &dOut));
  CHECK(zeKernelSetArgumentValue(kDev, 1, sizeof(void *), &dIn));
  CHECK(zeKernelSetArgumentValue(kDev, 2, sizeof(void *), &dCtrlDev));
  CHECK(zeKernelSetArgumentValue(kShr, 0, sizeof(void *), &dOut));
  CHECK(zeKernelSetArgumentValue(kShr, 1, sizeof(void *), &dIn));
  CHECK(zeKernelSetArgumentValue(kShr, 2, sizeof(void *), &dCtrlShr));

  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  ze_group_count_t gc = {(uint32_t)NG, 1, 1};
  ze_command_list_handle_t regDev = nullptr, regShr = nullptr;
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                  nullptr, 0, 0};
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &regDev));
  CHECK(zeCommandListAppendLaunchKernel(regDev, kDev, &gc, nullptr, 0,
                                        nullptr));
  CHECK(zeCommandListAppendBarrier(regDev, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(regDev));
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &regShr));
  CHECK(zeCommandListAppendLaunchKernel(regShr, kShr, &gc, nullptr, 0,
                                        nullptr));
  CHECK(zeCommandListAppendBarrier(regShr, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(regShr));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));

  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));

  // Fixed input; positions jump non-monotonically to prove no staleness.
  std::vector<float> hIn(N), hOut(N);
  uint64_t s = 0x5EED;
  for (auto &v : hIn) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    v = (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  }
  CHECK(zeCommandListAppendMemoryCopy(up, dIn, hIn.data(), (size_t)N * 4,
                                      nullptr, 0, nullptr));

  std::vector<double> tUpdDev, tRepDev, tUpdShr, tRepShr;
  bool ok = true;
  for (int it = 0; it < IT; ++it) {
    int pos = (it * 37 + 11) % 128; // non-monotonic jumps in [0,128)
    DecodeControl c{1000 + it, pos, pos + 1, -1};
    // --- explicit path ---
    double t0 = now_ns();
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrlDev, &c, sizeof(c), nullptr,
                                        0, nullptr));
    double t1 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &regDev, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    double t2 = now_ns();
    tUpdDev.push_back(t1 - t0);
    tRepDev.push_back(t2 - t1);
    CHECK(zeCommandListAppendMemoryCopy(up, hOut.data(), dOut, (size_t)N * 4,
                                        nullptr, 0, nullptr));
    for (int j = 0; j < N; ++j) {
      float want = hIn[j] + (float)pos;
      uint32_t a, b;
      __builtin_memcpy(&a, &hOut[j], 4);
      __builtin_memcpy(&b, &want, 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "explicit mismatch it %d j %d: got %g want %g\n",
                     it, j, hOut[j], want);
        break;
      }
    }
    if (!ok)
      break;
    // --- shared path: direct host write, no copy command ---
    t0 = now_ns();
    std::memcpy(dCtrlShr, &c, sizeof(c));
    t1 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &regShr, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    t2 = now_ns();
    tUpdShr.push_back(t1 - t0);
    tRepShr.push_back(t2 - t1);
    CHECK(zeCommandListAppendMemoryCopy(up, hOut.data(), dOut, (size_t)N * 4,
                                        nullptr, 0, nullptr));
    for (int j = 0; j < N; ++j) {
      float want = hIn[j] + (float)pos;
      uint32_t a, b;
      __builtin_memcpy(&a, &hOut[j], 4);
      __builtin_memcpy(&b, &want, 4);
      if (a != b) {
        ok = false;
        std::fprintf(stderr, "shared mismatch it %d j %d: got %g want %g\n",
                     it, j, hOut[j], want);
        break;
      }
    }
    if (!ok)
      break;
  }
  double uD = med(tUpdDev), rD = med(tRepDev);
  double uS = med(tUpdShr), rS = med(tRepShr);
  std::printf("explicit: update %.2f us replay %.2f us\n", uD / 1e3, rD / 1e3);
  std::printf("shared:   update %.2f us replay %.2f us\n", uS / 1e3, rS / 1e3);
  std::printf("%s\n", ok ? "CONTROL-OK (no arg mutation, no rebuild)" : "FAIL");
  const char *winner = (uS + rS <= uD + rD) ? "shared" : "explicit";
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS10ControlAdd\","
                "\"N\":%d,\"iters\":%d,\"control_bytes\":16,"
                "\"explicit_update_us\":%.2f,\"explicit_replay_us\":%.2f,"
                "\"shared_update_us\":%.2f,\"shared_replay_us\":%.2f,"
                "\"winner\":\"%s\",\"control_ok\":%s}",
                N, IT, uD / 1e3, rD / 1e3, uS / 1e3, rS / 1e3, winner,
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
