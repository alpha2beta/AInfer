// T7.4 far-slot KV validation: KvAppend into a 64K-deep single-slot BF16
// cache at positions {0, 32767, 65533} (TMAXF=65544, the decode_l0 64K
// sizing). Verifies each appended slot bitwise vs host RNE quantize, and
// guard slots stay zero (stride math / cross-talk check on the pos axis —
// the addressing dimension the 64K-sizing loop run never touches, since it
// generates at low positions).
// Usage: kvfar_replay <kvappend.spv> [report.json]
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
    std::fprintf(stderr, "usage: kvfar_replay <kvappend.spv> [report.json]\n");
    return 2;
  }
  const int NK = 4, D = 256, TMAXF = 65544;
  const int PPOS[] = {0, 32767, 65533};
  const int GUARD[] = {1, 32766, 32768, 65532, 65543};
  FILE *sf = std::fopen(argv[1], "rb");
  if (!sf) {
    std::fprintf(stderr, "no spv: %s\n", argv[1]);
    return 2;
  }
  std::fseek(sf, 0, SEEK_END);
  size_t nn = std::ftell(sf);
  std::fseek(sf, 0, SEEK_SET);
  std::vector<uint8_t> spv(nn);
  if (std::fread(spv.data(), 1, nn, sf) != nn)
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
  void *dK = nullptr, *dV = nullptr, *dKc = nullptr, *dVc = nullptr,
       *dCtrl = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * D * 4, 4096, dev, &dK));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * D * 4, 4096, dev, &dV));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * TMAXF * D * 2, 4096, dev,
                         &dKc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)NK * TMAXF * D * 2, 4096, dev,
                         &dVc));
  CHECK(zeMemAllocDevice(ctx, &mdesc, sizeof(DecodeControl), 4096, dev,
                         &dCtrl));

  ze_module_handle_t mod = nullptr;
  ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spv.size(),
                             spv.data(),
                             nullptr,
                             nullptr};
  CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
  ze_kernel_handle_t kKv = nullptr;
  ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                         "_ZTS8KvAppend"};
  CHECK(zeKernelCreate(mod, &kd, &kKv));
  CHECK(zeKernelSetGroupSize(kKv, 1, 1, 1));
  int tmaxArg = TMAXF;
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
  ze_group_count_t gB = {(uint32_t)(NK * D), 1, 1};
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
  // Device-side zero fill (128 MB per cache; never crosses PCIe as upload).
  const uint8_t z = 0;
  CHECK(zeCommandListAppendMemoryFill(up, dKc, &z, 1,
                                      (size_t)NK * TMAXF * D * 2, nullptr, 0,
                                      nullptr));
  CHECK(zeCommandListAppendMemoryFill(up, dVc, &z, 1,
                                      (size_t)NK * TMAXF * D * 2, nullptr, 0,
                                      nullptr));

  uint64_t s = 0x5FA84544;
  auto rnd = [&]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((s >> 33) & 0xFFFF) - 32768) * (1.0f / 32768.0f);
  };
  std::vector<float> hK(NK * D), hV(NK * D);
  std::vector<std::vector<float>> savedK, savedV;
  std::vector<double> tRep;
  bool ok = true;
  for (int p : PPOS) {
    for (auto &v : hK)
      v = rnd() * 2.0f;
    for (auto &v : hV)
      v = rnd();
    savedK.push_back(hK);
    savedV.push_back(hV);
    DecodeControl c{7000 + p, p, p + 1, -1};
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
  }
  // Verify appended slots bitwise + guard slots zero.
  std::vector<uint16_t> slot((size_t)NK * D);
  for (size_t a = 0; a < 3 && ok; ++a) {
    int p = PPOS[a];
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dKc + (size_t)p * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      if (slot[j] != f32_to_bf16(savedK[a][j])) {
        ok = false;
        std::fprintf(stderr, "far-K bits mismatch pos %d j %zu\n", p, j);
        break;
      }
    }
    if (!ok)
      break;
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dVc + (size_t)p * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      if (slot[j] != f32_to_bf16(savedV[a][j])) {
        ok = false;
        std::fprintf(stderr, "far-V bits mismatch pos %d j %zu\n", p, j);
        break;
      }
    }
    if (!ok)
      break;
  }
  for (int g : GUARD) {
    if (!ok)
      break;
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dKc + (size_t)g * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      if (slot[j] != 0) {
        ok = false;
        std::fprintf(stderr, "guard-K nonzero pos %d j %zu\n", g, j);
        break;
      }
    }
    if (!ok)
      break;
    CHECK(zeCommandListAppendMemoryCopy(up, slot.data(),
                                        (char *)dVc + (size_t)g * NK * D * 2,
                                        slot.size() * 2, nullptr, 0, nullptr));
    for (size_t j = 0; j < slot.size(); ++j) {
      if (slot[j] != 0) {
        ok = false;
        std::fprintf(stderr, "guard-V nonzero pos %d j %zu\n", g, j);
        break;
      }
    }
  }
  double mR = med(tRep);
  std::printf("kvfar med %.2f us/append %s\n", mR / 1e3,
              ok ? "KVFAR-OK" : "MISMATCH");
  char js[512];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"kernel\":\"_ZTS8KvAppend\","
                "\"tmax\":%d,\"positions\":[0,32767,65533],"
                "\"guards\":[1,32766,32768,65532,65543],"
                "\"append_us\":%.2f,\"bitwise_slots\":%s,"
                "\"guards_zero\":%s,\"kvfar_ok\":%s}",
                TMAXF, mR / 1e3, ok ? "true" : "false", ok ? "true" : "false",
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
