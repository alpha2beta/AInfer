// T1.5: unified-memory allocation policy comparison on Lunar Lake (Arc 140V).
// Compares zeMemAllocDevice vs zeMemAllocShared vs zeMemAllocHost on:
//   1. allocation latency (host-timed)
//   2. first-touch (cold) vs warm streaming-read bandwidth (stream_read kernel)
//   3. host-produce -> GPU-consume round-trip (shared: host write + GPU stream;
//      device: host staging write + H2D copy + GPU stream)
//   4. D2H 4-byte token readback latency (per-token host-overhead proxy, T5.4)
// Prints one JSON object to stdout.
// Usage: alloc_policy <stream_read.spv> [label]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

static const size_t TEST_BYTES = 1ULL << 30; // 1 GiB per allocation
static const int RUNS = 7;

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/membench/stream_read.spv";
  const char *label = (argc > 2) ? argv[2] : "alloc-policy";

  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open %s\n", spv_path);
    return 1;
  }
  spv_f.seekg(0, std::ios::end);
  size_t spv_size = spv_f.tellg();
  spv_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> spv(spv_size);
  spv_f.read((char *)spv.data(), spv_size);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));
  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  for (auto d : drvs) {
    uint32_t nDev = 0;
    CHECK(zeDeviceGet(d, &nDev, nullptr));
    std::vector<ze_device_handle_t> devs(nDev);
    CHECK(zeDeviceGet(d, &nDev, devs.data()));
    for (auto dv : devs) {
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      CHECK(zeDeviceGetProperties(dv, &props));
      if (props.deviceId == 0x64a0 || props.deviceId == 0xe211) {
        dev = dv;
        drv = d;
        break;
      }
    }
    if (dev) break;
  }
  if (!dev) {
    std::fprintf(stderr, "Target device not found\n");
    return 1;
  }

  ze_context_desc_t ctx_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t ctx;
  CHECK(zeContextCreate(drv, &ctx_desc, &ctx));
  ze_module_desc_t mod_desc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                            ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(),
                            "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod;
  CHECK(zeModuleCreate(ctx, dev, &mod_desc, &mod, nullptr));

  uint32_t nQgroups = 0;
  CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &nQgroups, nullptr));
  std::vector<ze_command_queue_group_properties_t> qprops(nQgroups);
  for (auto &qp : qprops) qp.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  CHECK(zeDeviceGetCommandQueueGroupProperties(dev, &nQgroups, qprops.data()));
  uint32_t compute_ord = 0;
  for (uint32_t i = 0; i < nQgroups; ++i) {
    if (qprops[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      compute_ord = i;
      break;
    }
  }
  ze_command_list_desc_t clist_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));
  ze_fence_desc_t fdesc{ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  ze_fence_handle_t fence;
  CHECK(zeFenceCreate(queue, &fdesc, &fence));

  auto now_ms = [] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };

  // 1. allocation latency per type
  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  void *d_dev = nullptr, *d_shr = nullptr, *d_host = nullptr;
  double t0 = now_ms();
  CHECK(zeMemAllocDevice(ctx, &dmem, TEST_BYTES, 4096, dev, &d_dev));
  double alloc_device_ms = now_ms() - t0;
  t0 = now_ms();
  CHECK(zeMemAllocShared(ctx, &dmem, &hmem, TEST_BYTES, 4096, dev, &d_shr));
  double alloc_shared_ms = now_ms() - t0;
  bool have_host_alloc = true;
  t0 = now_ms();
  if (zeMemAllocHost(ctx, &hmem, TEST_BYTES, 4096, &d_host) != ZE_RESULT_SUCCESS) {
    have_host_alloc = false;
  }
  double alloc_host_ms = now_ms() - t0;

  // Stream kernel setup (shared across buffer types)
  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "stream_read"};
  ze_kernel_handle_t kern;
  CHECK(zeKernelCreate(mod, &kd, &kern));
  const uint32_t NWORDS = (uint32_t)(TEST_BYTES / 16);
  const uint32_t N_GROUPS = 64;
  CHECK(zeKernelSetGroupSize(kern, 256, 1, 1));
  ze_group_count_t gc{N_GROUPS, 1, 1};

  // Scratch partials buffer (device).
  void *d_part = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dmem, N_GROUPS * sizeof(float), 64, dev, &d_part));

  auto stream_once = [&](void *buf, double &ms) -> bool {
    CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &buf));
    CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_part));
    CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(uint32_t), &NWORDS));
    ze_command_list_handle_t list;
    CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));
    CHECK(zeCommandListAppendLaunchKernel(list, kern, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));
    auto a = std::chrono::steady_clock::now();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    auto b = std::chrono::steady_clock::now();
    CHECK(zeCommandListDestroy(list));
    ms = std::chrono::duration<double, std::milli>(b - a).count();
    return true;
  };

  auto med_gbs = [&](void *buf, double &cold_ms) -> double {
    double m = 0;
    if (!stream_once(buf, m)) return -1.0;
    cold_ms = m; // first touch on fresh allocation
    std::vector<double> v;
    for (int r = 0; r < RUNS; ++r) {
      if (!stream_once(buf, m)) return -1.0;
      v.push_back(m);
    }
    std::sort(v.begin(), v.end());
    return (TEST_BYTES / 1e9) / (v[RUNS / 2] / 1e3);
  };

  // Prefault shared/host buffers on CPU so first-touch measures migration,
  // not zero-page faults (documented choice; device alloc needs no prefault).
  std::memset(d_shr, 0x11, TEST_BYTES);
  if (have_host_alloc) std::memset(d_host, 0x11, TEST_BYTES);

  double cold_dev = 0, cold_shr = 0, cold_host = 0;
  double gbs_dev = med_gbs(d_dev, cold_dev);
  double gbs_shr = med_gbs(d_shr, cold_shr);
  double gbs_host = have_host_alloc ? med_gbs(d_host, cold_host) : -1.0;
  double cold_dev_gbs = (TEST_BYTES / 1e9) / (cold_dev / 1e3);
  double cold_shr_gbs = (TEST_BYTES / 1e9) / (cold_shr / 1e3);

  // 3. host-produce -> GPU-consume round trip on 64 MiB.
  const size_t RT_BYTES = 64ULL << 20;
  // shared path: host memset + GPU stream of the same region
  t0 = now_ms();
  std::memset(d_shr, 0x22, RT_BYTES);
  double rt_host_write_ms = now_ms() - t0;
  // device path: staging write + H2D copy, then stream (copy timed below)
  std::vector<char> staging(RT_BYTES, 0x22);
  ze_command_list_handle_t cpl;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &cpl));
  CHECK(zeCommandListAppendMemoryCopy(cpl, d_dev, staging.data(), RT_BYTES, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(cpl));
  t0 = now_ms();
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cpl, fence));
  CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
  CHECK(zeFenceReset(fence));
  double h2d_ms = (std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count() -
                   t0);
  CHECK(zeCommandListDestroy(cpl));
  double h2d_gbs = (RT_BYTES / 1e9) / (h2d_ms / 1e3);

  // 4. D2H 4-byte token readback (per-token host-overhead proxy).
  uint32_t host_tok = 0;
  std::vector<double> tok_us;
  for (int r = 0; r < RUNS; ++r) {
    ze_command_list_handle_t tl;
    CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &tl));
    CHECK(zeCommandListAppendMemoryCopy(tl, &host_tok, d_dev, 4, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(tl));
    auto a = std::chrono::steady_clock::now();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &tl, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    auto b = std::chrono::steady_clock::now();
    CHECK(zeCommandListDestroy(tl));
    tok_us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
  }
  std::sort(tok_us.begin(), tok_us.end());

  std::printf("{"
              "\"label\": \"%s\", "
              "\"alloc_device_ms\": %.2f, \"alloc_shared_ms\": %.2f, \"alloc_host_ms\": %.2f, \"have_host_alloc\": %s, "
              "\"warm_device_gbs\": %.2f, \"warm_shared_gbs\": %.2f, \"warm_host_gbs\": %.2f, "
              "\"cold_device_gbs\": %.2f, \"cold_shared_gbs\": %.2f, "
              "\"host_write_64mib_ms\": %.2f, \"h2d_64mib_ms\": %.2f, \"h2d_gbs\": %.2f, "
              "\"d2h_token_us\": %.2f"
              "}\n",
              label, alloc_device_ms, alloc_shared_ms, alloc_host_ms,
              have_host_alloc ? "true" : "false", gbs_dev, gbs_shr, gbs_host,
              cold_dev_gbs, cold_shr_gbs, rt_host_write_ms, h2d_ms, h2d_gbs,
              tok_us[RUNS / 2]);
  return 0;
}
