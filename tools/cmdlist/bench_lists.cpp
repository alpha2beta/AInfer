// T5.3: regular vs immediate Level Zero command lists (+ SYCL baseline).
// Records one D2D copy (+ barrier) once, replays it; measures host-visible
// resubmit cost — the overhead steady-state decode would shed per token.
// Usage: bench_lists [report.json]
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
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
  const size_t N = 1u << 20; // 1 MiB copy (dispatch-sized, not BW-sized)
  const int IT = 200;
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
  if (!dev) {
    std::fprintf(stderr, "no B60\n");
    return 1;
  }
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));
  // NOTE (toolchain quarantine, verified 2026-09-09): do NOT mix allocators
  // across APIs. SYCL API calls reject L0-allocated pointers
  // (urEnqueueUSMMemcpy -> OUT_OF_RESOURCES), and the L0 driver segfaults on
  // SYCL-allocated pointers from a foreign context. So: L0 buffers for the
  // L0 paths, separate SYCL buffers for the SYCL baseline. (Inside-kernel
  // references to L0 arenas are fine, as decode proves.)
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *dA = nullptr, *dB = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, N, 4096, dev, &dA));
  CHECK(zeMemAllocDevice(ctx, &mdesc, N, 4096, dev, &dB));
  sycl::device sdev0;
  for (auto p : sycl::platform::get_platforms())
    for (auto d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::name>().find("B60") !=
          std::string::npos)
        sdev0 = d;
  sycl::queue qq0(sdev0);
  void *sA = sycl::malloc_device(N, qq0);
  void *sB = sycl::malloc_device(N, qq0);

  std::vector<uint8_t> pat(N);
  for (size_t i = 0; i < N; ++i)
    pat[i] = (uint8_t)(i * 2654435761u >> 16);

  // ---- regular list: record once, execute N times via queue+fence ----
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
  CHECK(zeCommandListAppendMemoryCopy(reg, dB, dA, N, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(reg, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(reg));
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));

  // warm + upload pattern via temp immediate list
  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));
  CHECK(zeCommandListAppendMemoryCopy(up, dA, pat.data(), N, nullptr, 0,
                                      nullptr));

  std::vector<double> tReg;
  for (int i = 0; i < IT; ++i) {
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tReg.push_back(now_ns() - t0);
  }
  // verify copy happened
  std::vector<uint8_t> rb(N);
  CHECK(zeCommandListAppendMemoryCopy(up, rb.data(), dB, N, nullptr, 0,
                                      nullptr));
  bool okReg = (rb == pat);

  // ---- immediate list: append+sync each iteration (today's SYCL-adjacent) --
  std::vector<double> tImm;
  for (int i = 0; i < IT; ++i) {
    double t0 = now_ns();
    CHECK(zeCommandListAppendMemoryCopy(up, dB, dA, N, nullptr, 0, nullptr));
    tImm.push_back(now_ns() - t0);
  }
  CHECK(zeCommandListAppendMemoryCopy(up, rb.data(), dB, N, nullptr, 0,
                                      nullptr));
  bool okImm = (rb == pat);

  // ---- SYCL baseline: q.memcpy().wait() per iteration ----
  sycl::queue q(sdev0);
  qq0.memcpy(sA, pat.data(), N).wait();
  std::vector<double> tSycl;
  for (int i = 0; i < IT; ++i) {
    double t0 = now_ns();
    q.memcpy(sB, sA, N).wait();
    tSycl.push_back(now_ns() - t0);
  }
  q.memcpy(rb.data(), sB, N).wait();
  bool okSycl = (rb == pat);

  double mReg = med(tReg), mImm = med(tImm), mSycl = med(tSycl);
  std::printf("regular-replay  med %.1f us %s\n", mReg / 1e3,
              okReg ? "OK" : "MISMATCH");
  std::printf("immediate-append med %.1f us %s\n", mImm / 1e3,
              okImm ? "OK" : "MISMATCH");
  std::printf("sycl-memcpy     med %.1f us %s\n", mSycl / 1e3,
              okSycl ? "OK" : "MISMATCH");
  char js[1024];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"bytes\":%zu,\"iters\":%d,"
                "\"regular_replay_us\":%.2f,\"immediate_append_us\":%.2f,"
                "\"sycl_memcpy_us\":%.2f,\"ok\":%s}",
                N, IT, mReg / 1e3, mImm / 1e3, mSycl / 1e3,
                (okReg && okImm && okSycl) ? "true" : "false");
  if (argc > 1) {
    FILE *o = std::fopen(argv[1], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return (okReg && okImm && okSycl) ? 0 : 1;
}
