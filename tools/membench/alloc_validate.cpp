// T2.5: empirical full-budget allocation validation on Arc 140V (Xe2).
// Allocates the Phase-2 memory-budget arenas via Level Zero under realistic
// system load (light CPU triad workers + ambient box load), first-touches
// every page (MemoryFill + sampled readback), then checks for OOM-killer
// activity (/proc/vmstat oom_kill counter), zram/swap growth, and remaining
// headroom. Prints one JSON object to stdout.
// Budget (memory_budget.json, 64K tier upper bound): payload 17.32 GiB +
// scales 521 MiB + KV 1.28 GiB + SSM 62.8 MiB + workspace 256 MiB.
// Usage: alloc_validate [label]
#include <level_zero/ze_api.h>

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

// Budget arenas (bytes), matching memory_budget.json 64K tier.
static const uint64_t ARENA_PAYLOAD = 18594016768ULL; // 17.32 GiB INT4 weights
static const uint64_t ARENA_SCALES = 546467840ULL;    // 521 MiB BF16 scales
static const uint64_t ARENA_KV = 1374389536ULL;       // 1.28 GiB KV @64K
static const uint64_t ARENA_SSM = 65863680ULL;        // 62.8 MiB DeltaNet state
static const uint64_t ARENA_WORK = 268435456ULL;      // 256 MiB workspace

static long read_vmstat_oom() {
  std::ifstream f("/proc/vmstat");
  std::string k;
  long v = 0;
  while (f >> k >> v) {
    if (k == "oom_kill") return v;
  }
  return -1;
}

static void read_meminfo(long &avail_kb, long &swap_free_kb) {
  avail_kb = swap_free_kb = -1;
  std::ifstream f("/proc/meminfo");
  std::string k, unit;
  long v = 0;
  while (f >> k >> v >> unit) {
    if (k == "MemAvailable:") avail_kb = v;
    if (k == "SwapFree:") swap_free_kb = v;
  }
}

static long read_zram_used_bytes() {
  // mm_stat: orig_data_size compr_data_size mem_used_total ...
  std::ifstream f("/sys/block/zram0/mm_stat");
  long a, b, used = -1;
  if (f >> a >> b >> used) return used;
  return -1;
}

static std::string load_avg() {
  std::ifstream f("/proc/loadavg");
  std::string a, b, c;
  if (f >> a >> b >> c) return a + " " + b + " " + c;
  return "?";
}

int main(int argc, char **argv) {
  const char *label = (argc > 1) ? argv[1] : "alloc-validate";

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
  ze_command_list_desc_t clist_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));
  ze_fence_desc_t fdesc{ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  ze_fence_handle_t fence;
  CHECK(zeFenceCreate(queue, &fdesc, &fence));

  long oom0 = read_vmstat_oom();
  long avail0 = 0, swap0 = 0;
  read_meminfo(avail0, swap0);
  long zram0 = read_zram_used_bytes();

  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  struct Arena {
    const char *name;
    uint64_t bytes;
    void *ptr = nullptr;
  };
  Arena arenas[] = {
      {"payload", ARENA_PAYLOAD}, {"scales", ARENA_SCALES}, {"kv_64k", ARENA_KV},
      {"ssm", ARENA_SSM},         {"workspace", ARENA_WORK},
  };
  bool alloc_ok = true;
  auto t_alloc0 = std::chrono::steady_clock::now();
  for (auto &a : arenas) {
    if (zeMemAllocDevice(ctx, &dmem, (size_t)a.bytes, 4096, dev, &a.ptr) != ZE_RESULT_SUCCESS ||
        !a.ptr) {
      std::fprintf(stderr, "arena alloc FAILED: %s (%llu bytes)\n", a.name,
                   (unsigned long long)a.bytes);
      alloc_ok = false;
      break;
    }
  }
  auto t_alloc1 = std::chrono::steady_clock::now();
  double alloc_ms =
      std::chrono::duration<double, std::milli>(t_alloc1 - t_alloc0).count();

  // First-touch every page via MemoryFill (forces commitment), then verify
  // a sampled readback per arena through a copy list.
  bool touch_ok = alloc_ok;
  uint32_t zero = 0;
  uint64_t total_touched = 0;
  auto t_touch0 = std::chrono::steady_clock::now();
  if (alloc_ok) {
    for (auto &a : arenas) {
      ze_command_list_handle_t cl;
      CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &cl));
      CHECK(zeCommandListAppendMemoryFill(cl, a.ptr, &zero, sizeof(zero),
                                          (size_t)a.bytes, nullptr, 0, nullptr));
      CHECK(zeCommandListClose(cl));
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cl, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      CHECK(zeCommandListDestroy(cl));
      total_touched += a.bytes;
    }
    // Sampled readback: 1 MiB from the middle of each arena.
    std::vector<char> host(1 << 20);
    for (auto &a : arenas) {
      ze_command_list_handle_t cl;
      CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &cl));
      CHECK(zeCommandListAppendMemoryCopy(
          cl, host.data(), (char *)a.ptr + a.bytes / 2, host.size(), nullptr, 0, nullptr));
      CHECK(zeCommandListClose(cl));
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cl, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      CHECK(zeCommandListDestroy(cl));
      for (char c : host) {
        if (c != 0) {
          touch_ok = false;
          break;
        }
      }
      if (!touch_ok) break;
    }
  }
  auto t_touch1 = std::chrono::steady_clock::now();
  double touch_ms =
      std::chrono::duration<double, std::milli>(t_touch1 - t_touch0).count();

  long oom1 = read_vmstat_oom();
  long avail1 = 0, swap1 = 0;
  read_meminfo(avail1, swap1);
  long zram1 = read_zram_used_bytes();

  for (auto &a : arenas) {
    if (a.ptr) CHECK(zeMemFree(ctx, a.ptr));
  }

  uint64_t total = ARENA_PAYLOAD + ARENA_SCALES + ARENA_KV + ARENA_SSM + ARENA_WORK;
  bool pass = alloc_ok && touch_ok && (oom0 >= 0 && oom1 == oom0);
  std::printf("{"
              "\"label\": \"%s\", \"status\": \"%s\", "
              "\"total_arena_gib\": %.2f, \"alloc_ok\": %s, \"alloc_ms\": %.1f, "
              "\"first_touch_ok\": %s, \"first_touch_ms\": %.1f, "
              "\"oom_kill_before\": %ld, \"oom_kill_after\": %ld, "
              "\"mem_available_before_kb\": %ld, \"mem_available_after_kb\": %ld, "
              "\"swap_free_before_kb\": %ld, \"swap_free_after_kb\": %ld, "
              "\"zram_used_before_b\": %ld, \"zram_used_after_b\": %ld, "
              "\"host_loadavg\": \"%s\""
              "}\n",
              label, pass ? "PASSED" : "FAILED", total / 1073741824.0,
              alloc_ok ? "true" : "false", alloc_ms,
              touch_ok ? "true" : "false", touch_ms, oom0, oom1, avail0,
              avail1, swap0, swap1, zram0, zram1, load_avg().c_str());
  return pass ? 0 : 1;
}
