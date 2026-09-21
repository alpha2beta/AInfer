// T1.7: dispatch + bandwidth profile on Arc 140V (Xe2).
// Measures, all host-fence-timed (median of 7 after 2 warmup unless noted):
//   1. sequential vs strided (stride 64 float4 words) read bandwidth, 1 GiB
//   2. empty command-list launch latency (submit+fence round trip)
//   3. barrier cost delta: kernel alone vs kernel+barrier per launch
//   4. cold (first exec on fresh list) vs warm vs sustained (60 s loop drift)
// Cookbook A/B (env-driven, same binary):
//   ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE, and immediate vs regular lists
//   (AINFER_IMM=1 selects zeCommandListCreateImmediate + event sync).
// Prints one JSON object to stdout.
// Usage: dispatch_profile <stream_strided.spv> [label]
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

static const size_t TEST_BYTES = 1ULL << 30; // 1 GiB
static const int WARMUP = 2;
static const int RUNS = 7;

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/membench/stream_strided.spv";
  const char *label = (argc > 2) ? argv[2] : "dispatch";
  // AINFER_IMM=1 selects immediate lists + event sync. FINDING (T1.7):
  // functional for short bursts (~36 submissions) but hangs deterministically
  // on sustained resubmission with this compute-runtime (26.31) — repeated
  // across runs, unaffected by host pacing. Production MUST use regular
  // recorded lists (which is also the T5.3 architecture — no change needed).
  const bool use_imm = std::getenv("AINFER_IMM") != nullptr;

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

  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_buf = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dmem, TEST_BYTES, 4096, dev, &d_buf));
  const uint32_t N_GROUPS = 64;
  void *d_part = nullptr;
  CHECK(zeMemAllocDevice(ctx, &dmem, N_GROUPS * sizeof(float), 64, dev, &d_part));

  ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "stream_strided"};
  ze_kernel_handle_t kern;
  CHECK(zeKernelCreate(mod, &kd, &kern));
  const uint32_t NWORDS = (uint32_t)(TEST_BYTES / 16);
  const uint32_t STRIDE = 64;
  const uint32_t STRIDE1 = 1;
  CHECK(zeKernelSetGroupSize(kern, 256, 1, 1));
  ze_group_count_t gc{N_GROUPS, 1, 1};

  ze_command_queue_handle_t queue = nullptr;
  ze_fence_handle_t fence = nullptr;
  ze_command_list_handle_t imm = nullptr;
  ze_event_handle_t imm_ev = nullptr;
  ze_event_pool_handle_t imm_pool = nullptr;
  if (!use_imm) {
    ze_command_list_desc_t clist_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
    (void)clist_desc;
    ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));
    ze_fence_desc_t fdesc{ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
    CHECK(zeFenceCreate(queue, &fdesc, &fence));
  } else {
    ze_event_pool_desc_t pdesc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                               ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
    CHECK(zeEventPoolCreate(ctx, &pdesc, 1, &dev, &imm_pool));
    ze_event_desc_t edesc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0,
                          ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    CHECK(zeEventCreate(imm_pool, &edesc, &imm_ev));
    ze_command_queue_desc_t idesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                    ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &imm));
  }

  // Submit + host-side completion wait, abstracting regular vs immediate.
  auto submit_wait = [&](ze_command_list_handle_t list) -> bool {
    if (!use_imm) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
    } else {
      (void)list; // immediate list executes on append; sync below on event
      CHECK(zeEventHostSynchronize(imm_ev, UINT64_MAX));
      CHECK(zeEventHostReset(imm_ev));
    }
    return true;
  };

  auto make_list = [&](bool with_kernel, bool with_barrier, uint32_t stride,
                       ze_command_list_handle_t &out) -> bool {
    if (!use_imm) {
      ze_command_list_desc_t clist_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
      CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &out));
    } else {
      out = imm;
      CHECK(zeEventHostReset(imm_ev));
    }
    if (with_kernel) {
      CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &d_buf));
      CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_part));
      CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(uint32_t), &NWORDS));
      CHECK(zeKernelSetArgumentValue(kern, 3, sizeof(uint32_t), &stride));
      if (!use_imm) {
        CHECK(zeCommandListAppendLaunchKernel(out, kern, &gc, nullptr, 0, nullptr));
      } else {
        CHECK(zeCommandListAppendLaunchKernel(imm, kern, &gc, imm_ev, 0, nullptr));
      }
    } else if (use_imm) {
      // Immediate lists have no fence: the completion event must be signaled
      // by *something*, or the host wait below hangs forever. A barrier is
      // the cheapest signalable no-op — this measures barrier round-trip.
      CHECK(zeCommandListAppendBarrier(imm, imm_ev, 0, nullptr));
    }
    if (with_barrier && !use_imm) {
      CHECK(zeCommandListAppendBarrier(out, nullptr, 0, nullptr));
    }
    if (!use_imm) {
      CHECK(zeCommandListClose(out));
    }
    return true;
  };

  auto time_once = [&](ze_command_list_handle_t list, double &ms) {
    auto a = std::chrono::steady_clock::now();
    if (!submit_wait(list)) return false;
    auto b = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(b - a).count();
    return true;
  };

  auto med_runs = [&](bool with_kernel, bool with_barrier, uint32_t stride,
                      double &cold_ms, bool record_cold) -> double {
    std::vector<double> v;
    for (int r = 0; r < WARMUP + RUNS; ++r) {
      ze_command_list_handle_t list = nullptr;
      if (!make_list(with_kernel, with_barrier, stride, list)) return -1.0;
      double m = 0;
      if (!time_once(list, m)) return -1.0;
      if (!use_imm) CHECK(zeCommandListDestroy(list));
      if (r == 0 && record_cold) cold_ms = m;
      if (r >= WARMUP) v.push_back(m);
    }
    std::sort(v.begin(), v.end());
    return v[RUNS / 2];
  };

  double cold_seq = 0, cold_str = 0, unused = 0;
  // 1. sequential (stride 1) vs strided read bandwidth
  double seq_ms = med_runs(true, false, STRIDE1, cold_seq, true);
  double seq_gbs = (TEST_BYTES / 1e9) / (seq_ms / 1e3);
  double str_ms = med_runs(true, false, STRIDE, cold_str, true);
  double str_gbs = (TEST_BYTES / 1e9) / (str_ms / 1e3);

  // 2. empty-list launch latency
  double empty_ms = med_runs(false, false, STRIDE1, unused, false);

  // 3. barrier delta: kernel+barrier vs kernel alone
  double kb_ms = med_runs(true, true, STRIDE1, unused, false);
  double barrier_delta_us = (kb_ms - seq_ms) * 1000.0;

  // 4. sustained drift: back-to-back sequential kernel for ~60 s, batches
  // of 20 timed as a unit; compare first-10s vs last-10s batch medians.
  std::vector<double> early, late;
  {
    ze_command_list_handle_t sus;
    if (!make_list(true, false, STRIDE1, sus)) return 1;
    auto t_start = std::chrono::steady_clock::now();
    int batch = 0;
    while (true) {
      auto a = std::chrono::steady_clock::now();
      for (int k = 0; k < 20; ++k) {
        if (!submit_wait(sus)) return 1;
      }
      auto b = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(b - a).count() / 20.0;
      double el = std::chrono::duration<double>(b - t_start).count();
      if (el < 10.0) {
        early.push_back(ms);
      } else if (el >= 50.0) {
        late.push_back(ms);
      }
      ++batch;
      if (el >= 60.0) break;
    }
    if (!use_imm) CHECK(zeCommandListDestroy(sus));
  }
  std::sort(early.begin(), early.end());
  std::sort(late.begin(), late.end());
  double early_med = early.empty() ? -1.0 : early[early.size() / 2];
  double late_med = late.empty() ? -1.0 : late[late.size() / 2];
  double drift_pct = (early_med > 0) ? 100.0 * (late_med - early_med) / early_med : 0.0;

  std::printf("{"
              "\"label\": \"%s\", \"immediate_lists\": %s, "
              "\"seq_gbs\": %.2f, \"seq_ms\": %.3f, \"cold_seq_ms\": %.3f, "
              "\"strided_gbs\": %.2f, \"strided_ms\": %.3f, \"cold_strided_ms\": %.3f, "
              "\"empty_launch_ms\": %.4f, \"barrier_delta_us\": %.2f, "
              "\"sustained_early_ms\": %.3f, \"sustained_late_ms\": %.3f, \"sustained_drift_pct\": %.2f"
              "}\n",
              label, use_imm ? "true" : "false", seq_gbs, seq_ms, cold_seq,
              str_gbs, str_ms, cold_str, empty_ms, barrier_delta_us,
              early_med, late_med, drift_pct);
  return 0;
}
