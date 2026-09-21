// T1.6: iGPU weight-streaming bandwidth probe for Arc 140V (Xe2).
// Measures (a) sequential read-stream bandwidth over a 2 GiB device buffer
// via stream_read.cl, and (b) device-to-device copy bandwidth, using
// host fence timing (median of measured runs after warmup). Intended to run
// standalone (isolated) and under concurrent CPU stress driven externally
// by run_contention.py. Prints one JSON object to stdout.
// Usage: contention <stream_read.spv> [label]
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

static const size_t BUF_BYTES = 2ULL << 30; // 2 GiB stream buffer
static const int WARMUP = 2;
static const int RUNS = 7;

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/membench/stream_read.spv";
  const char *label = (argc > 2) ? argv[2] : "isolated";

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

  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_buf = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dmem, BUF_BYTES, 4096, dev, &d_buf));
  void *d_part = nullptr;
  const uint32_t N_GROUPS = 64;
  CHECK(zeMemAllocDevice(ctx, &dmem, N_GROUPS * sizeof(float), 64, dev, &d_part));
  // Second buffer for the D2D copy leg.
  void *d_buf2 = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dmem, BUF_BYTES, 4096, dev, &d_buf2));

  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "stream_read"};
  ze_kernel_handle_t kern;
  CHECK(zeKernelCreate(mod, &kd, &kern));

  const uint32_t NWORDS = (uint32_t)(BUF_BYTES / 16);
  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &d_buf));
  CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_part));
  CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(uint32_t), &NWORDS));
  CHECK(zeKernelSetGroupSize(kern, 256, 1, 1));
  ze_group_count_t gc{N_GROUPS, 1, 1};

  auto exec_once = [&](ze_command_list_handle_t list) -> bool {
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    return true;
  };

  // Recorded read-stream list (built once, replayed per run).
  ze_command_list_handle_t read_list;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &read_list));
  CHECK(zeCommandListAppendLaunchKernel(read_list, kern, &gc, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(read_list));

  // (a) streaming-read bandwidth
  std::vector<double> read_ms;
  for (int r = 0; r < WARMUP + RUNS; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    if (!exec_once(read_list)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    if (r >= WARMUP) {
      read_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(read_ms.begin(), read_ms.end());
  double read_med = read_ms[RUNS / 2];
  double read_gbs = (BUF_BYTES / 1e9) / (read_med / 1e3);

  // (b) device-to-device copy bandwidth (read+write traffic)
  std::vector<double> copy_ms;
  for (int r = 0; r < WARMUP + RUNS; ++r) {
    ze_command_list_handle_t cl;
    CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &cl));
    CHECK(zeCommandListAppendMemoryCopy(cl, d_buf2, d_buf, BUF_BYTES, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cl));
    auto t0 = std::chrono::steady_clock::now();
    if (!exec_once(cl)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    CHECK(zeCommandListDestroy(cl));
    if (r >= WARMUP) {
      copy_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(copy_ms.begin(), copy_ms.end());
  double copy_med = copy_ms[RUNS / 2];
  // Copy moves 2x bytes (read + write).
  double copy_gbs = (2.0 * BUF_BYTES / 1e9) / (copy_med / 1e3);

  std::printf("{\"label\": \"%s\", \"buffer_gib\": %.1f, "
              "\"stream_read_ms\": %.3f, \"stream_read_gbs\": %.2f, "
              "\"d2d_copy_ms\": %.3f, \"d2d_copy_gbs\": %.2f}\n",
              label, BUF_BYTES / 1073741824.0, read_med, read_gbs,
              copy_med, copy_gbs);
  return 0;
}
