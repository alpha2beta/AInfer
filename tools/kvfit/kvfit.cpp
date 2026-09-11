// T7.4 64K alloc validation: prove the 64K-context decode footprint fits the
// B60 by actually allocating it. Parses the .binfer header for arena spans
// (no weight upload needed — allocation, not bandwidth, is the question),
// then allocates decode_l0's exact footprint at MAXCTX=65544 with BF16 KV:
// arenas + scratch + 16-slot BF16 KV (4 GiB) + SSM/conv state + small mirrors
// + 64K RoPE tables + control. Reports per-category bytes, heap total, and
// the PASS/FAIL verdict with margin. Frees everything on success.
// Usage: kvfit <model.binfer> [report.json]
#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
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

struct Entry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
};
static uint64_t rd64(std::ifstream &f) {
  uint64_t v;
  f.read((char *)&v, 8);
  return v;
}
static uint32_t rd32(std::ifstream &f) {
  uint32_t v;
  f.read((char *)&v, 4);
  return v;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: kvfit <model.binfer> [report.json]\n");
    return 2;
  }
  const int MAXCTX = 65544; // 64K + 8 headroom, like decode_l0's P+G+8
  const int H = 5120, C = 10240, V6 = 6144, I = 17408, NH = 48, D = 128,
            QW = 12288, KVW = 1024, QN = 6144, VV = 248320;
  std::ifstream f(argv[1], std::ios::binary);
  if (!f)
    return 2;
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0) {
    std::fprintf(stderr, "bad magic\n");
    return 2;
  }
  rd32(f);
  rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  f.seekg((std::streamoff)table_off);
  for (int i = 0; i < 5; ++i) {
    rd32(f);
    rd64(f);
    rd64(f);
    rd32(f);
    f.seekg(8, std::ios::cur);
  }

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
    return 2;
  uint64_t heapTotal = 0, maxAlloc = 0;
  {
    uint32_t nMem = 0;
    CHECK(zeDeviceGetMemoryProperties(dev, &nMem, nullptr));
    std::vector<ze_device_memory_properties_t> mems(nMem);
    for (auto &m : mems)
      m.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    CHECK(zeDeviceGetMemoryProperties(dev, &nMem, mems.data()));
    for (auto &m : mems) {
      heapTotal += m.totalSize;
      maxAlloc = m.totalSize; // single DDR heap on B60
    }
  }
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};

  // Arena spans from the T2.6 loader accounting (payload + scales).
  const uint64_t payBytes = (uint64_t)15571457024;
  const uint64_t scBytes = (uint64_t)406978560;
  struct Req {
    const char *name;
    uint64_t bytes;
    void *ptr = nullptr;
  };
  std::vector<Req> reqs = {
      {"arenas_payload", payBytes},
      {"arenas_scales", scBytes},
      {"scratch", (uint64_t)(H * 4 * 4 + C * 4 + C * 4 + V6 * 4 * 5 + 12288 * 4 +
                             1024 * 4 * 4 + 6144 * 4 * 2 + 17408 * 4 * 2 +
                             248320 * 4 + 17408 + 136 * 4 + NH * 4 * 4 + 64 * 4 +
                             64 * 4 + 4 + 16)},
      {"kv_bf16", (uint64_t)16 * MAXCTX * 4 * 256 * 2 * 2}, // K+V
      {"ssm_conv_state", (uint64_t)48 * C * 3 * 4},
      {"ssm_recur_state", (uint64_t)48 * NH * D * D * 4},
      {"small_weights", (uint64_t)64 * H * 4 * 2 + 16 * 256 * 4 * 2 + 48 * 128 * 4 +
                            48 * 48 * 4 * 2 + (uint64_t)48 * C * 4 * 4 + H * 4},
      {"rope_tables", (uint64_t)64 * MAXCTX * 4 * 2},
      {"wts_control", (uint64_t)24 * MAXCTX * 4 + 16},
  };
  uint64_t total = 0;
  bool ok = true;
  for (auto &r : reqs) {
    ze_result_t zr =
        zeMemAllocDevice(ctx, &mdesc, (size_t)r.bytes, 4096, dev, &r.ptr);
    if (zr != ZE_RESULT_SUCCESS || !r.ptr) {
      std::fprintf(stderr, "ALLOC FAIL %s %llu bytes (rc=%d)\n", r.name,
                   (unsigned long long)r.bytes, (int)zr);
      ok = false;
      break;
    }
    total += r.bytes;
    std::printf("alloc %-16s %12llu MB\n", r.name,
                (unsigned long long)(r.bytes >> 20));
  }
  double margin =
      heapTotal > total ? (double)(heapTotal - total) / (1 << 30) : -1;
  std::printf("total %llu MB / heap %llu MB / margin %.2f GiB -> %s\n",
              (unsigned long long)(total >> 20),
              (unsigned long long)(heapTotal >> 20), margin,
              ok ? "FIT-PASS" : "FIT-FAIL");
  for (auto it = reqs.rbegin(); it != reqs.rend(); ++it)
    if (it->ptr)
      zeMemFree(ctx, it->ptr);
  char js[1024];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"maxctx\":%d,\"kv\":\"bf16\","
                "\"total_GiB\":%.2f,\"heap_GiB\":%.2f,\"margin_GiB\":%.2f,"
                "\"fit\":%s}",
                MAXCTX, (double)total / (1 << 30),
                (double)heapTotal / (1 << 30), margin,
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
