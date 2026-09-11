// AInfer static L0 loader (T2.6; also T2.2 C++ port + T1.6 allocation proof).
// Parses the .binfer container, allocates static device arenas on the B60,
// streams all payloads host->device, then reads everything back and checks
// per-tensor CRC32 against the directory. Emits JSON report.
// Usage: l0load <model.binfer> [report.json]
#include <level_zero/ze_api.h>

#include <chrono>
#include <cinttypes>
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

static uint32_t crc32_update(uint32_t crc, const uint8_t *p, size_t n) {
  static uint32_t tab[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      tab[i] = c;
    }
    init = true;
  }
  crc ^= 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i)
    crc = tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

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
    std::fprintf(stderr, "usage: l0load <model.binfer> [report.json]\n");
    return 2;
  }
  const char *path = argv[1];
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0) {
    std::fprintf(stderr, "bad magic\n");
    return 2;
  }
  uint32_t ver = rd32(f), flags = rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  (void)ver;
  (void)flags;
  // section 5 = tensor dir
  f.seekg((std::streamoff)(table_off + 4 * 32));
  f.seekg(4 + 8 + 8, std::ios::cur); // sid, off, bytes
  uint32_t dir_crc;
  f.read((char *)&dir_crc, 4);
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f), c = rd32(f);
    (void)c;
    f.seekg(8, std::ios::cur);
    if (sid == 5) {
      dir_off = off;
      dir_bytes = nb;
    }
  }
  if (dir_bytes != n * 192) {
    std::fprintf(stderr, "dir size mismatch\n");
    return 2;
  }
  std::vector<Entry> entries(n);
  f.seekg((std::streamoff)dir_off);
  uint32_t dir_check = 0;
  for (uint64_t i = 0; i < n; ++i) {
    Entry &e = entries[i];
    f.read(e.name, 64);
    e.name[63] = 0;
    uint8_t ndim;
    f.read((char *)&ndim, 1);
    f.seekg(7, std::ios::cur);
    uint64_t shape[8];
    f.read((char *)shape, 64);
    uint8_t lt, st;
    uint16_t layout;
    uint32_t group;
    f.read((char *)&lt, 1);
    f.read((char *)&st, 1);
    f.read((char *)&layout, 2);
    f.read((char *)&group, 4);
    if (layout != 0) {
      std::fprintf(stderr, "unsupported layout %s\n", e.name);
      return 2;
    }
    e.sc_off = rd64(f);
    e.sc_bytes = rd64(f);
    e.d_off = rd64(f);
    e.d_bytes = rd64(f);
    e.crc = rd32(f);
    f.seekg(12, std::ios::cur);
    (void)ndim;
    (void)shape;
    (void)lt;
    (void)st;
    (void)group;
  }
  // verify dir CRC over raw bytes
  {
    f.clear();
    f.seekg((std::streamoff)dir_off);
    std::vector<char> buf(dir_bytes);
    f.read(buf.data(), dir_bytes);
    if (crc32_update(0, (uint8_t *)buf.data(), dir_bytes) != dir_crc) {
      std::fprintf(stderr, "dir CRC mismatch\n");
      return 2;
    }
  }

  // L0 init: B60 by PCI ID
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
    std::fprintf(stderr, "B60 not found\n");
    return 2;
  }
  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // Static arenas: one device allocation each for payloads and scales.
  uint64_t pay_lo = UINT64_MAX, pay_hi = 0, sc_lo = UINT64_MAX, sc_hi = 0;
  for (auto &e : entries) {
    pay_lo = e.d_off < pay_lo ? e.d_off : pay_lo;
    pay_hi = e.d_off + e.d_bytes > pay_hi ? e.d_off + e.d_bytes : pay_hi;
    if (e.sc_bytes) {
      sc_lo = e.sc_off < sc_lo ? e.sc_off : sc_lo;
      sc_hi = e.sc_off + e.sc_bytes > sc_hi ? e.sc_off + e.sc_bytes : sc_hi;
    }
  }
  auto t_alloc0 = std::chrono::steady_clock::now();
  ze_device_mem_alloc_desc_t mdesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                      nullptr, 0, 0};
  void *payArena = nullptr, *scArena = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)(pay_hi - pay_lo), 4096, dev,
                         &payArena));
  uint64_t scSize = (sc_hi > sc_lo) ? (sc_hi - sc_lo) : 0;
  if (scSize)
    CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)scSize, 4096, dev, &scArena));
  auto t_alloc1 = std::chrono::steady_clock::now();

  // Synchronous immediate list: copies complete on return, no events needed.
  ze_command_list_handle_t list = nullptr;
  ze_command_queue_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &ldesc, &list));

  const size_t CH = 1u << 28; // 256 MiB staging
  std::vector<char> staging(CH);
  auto copy_span = [&](uint64_t foff, void *arena, uint64_t alo, uint64_t len,
                       bool toDevice) {
    uint64_t done = 0;
    while (done < len) {
      size_t c = (size_t)((len - done > CH) ? CH : (len - done));
      if (toDevice) {
        f.clear();
        f.seekg((std::streamoff)(foff + done));
        f.read(staging.data(), c);
        CHECK(zeCommandListAppendMemoryCopy(
            list, (char *)arena + (foff + done - alo), staging.data(), c,
            nullptr, 0, nullptr));
      } else {
        CHECK(zeCommandListAppendMemoryCopy(
            list, staging.data(), (char *)arena + (foff + done - alo), c,
            nullptr, 0, nullptr));
      }
      done += c;
    }
  };

  auto t_h2d0 = std::chrono::steady_clock::now();
  copy_span(pay_lo, payArena, pay_lo, pay_hi - pay_lo, true);
  if (scSize)
    copy_span(sc_lo, scArena, sc_lo, scSize, true);
  auto t_h2d1 = std::chrono::steady_clock::now();

  // Full readback verify against directory CRCs: stream whole arenas back,
  // then CRC per-tensor slices on the host.
  uint64_t verified = 0;
  std::vector<std::string> failed;
  std::vector<char> rbPay(pay_hi - pay_lo), rbSc(scSize ? scSize : 1);
  auto t_d2h0 = std::chrono::steady_clock::now();
  {
    uint64_t done = 0, len = pay_hi - pay_lo;
    while (done < len) {
      size_t c = (size_t)((len - done > CH) ? CH : (len - done));
      CHECK(zeCommandListAppendMemoryCopy(list, rbPay.data() + done,
                                          (char *)payArena + done, c, nullptr,
                                          0, nullptr));
      done += c;
    }
  }
  if (scSize) {
    uint64_t done = 0;
    while (done < scSize) {
      size_t c = (size_t)((scSize - done > CH) ? CH : (scSize - done));
      CHECK(zeCommandListAppendMemoryCopy(list, rbSc.data() + done,
                                          (char *)scArena + done, c, nullptr,
                                          0, nullptr));
      done += c;
    }
  }
  auto t_d2h1 = std::chrono::steady_clock::now();
  for (auto &e : entries) {
    uint32_t got =
        crc32_update(0, (uint8_t *)(rbPay.data() + (e.d_off - pay_lo)),
                     (size_t)e.d_bytes);
    if (got == e.crc)
      ++verified;
    else
      failed.push_back(e.name);
  }
  double h2d_s =
      std::chrono::duration<double>(t_h2d1 - t_h2d0).count();
  double d2h_s =
      std::chrono::duration<double>(t_d2h1 - t_d2h0).count();
  double total_gb =
      ((pay_hi - pay_lo) + scSize) / 1e9;

  CHECK(zeCommandListDestroy(list));
  // NOTE: arenas intentionally left allocated until process exit (static
  // lifetime); freed here for the loader-test path.
  CHECK(zeMemFree(ctx, payArena));
  if (scArena)
    CHECK(zeMemFree(ctx, scArena));
  CHECK(zeContextDestroy(ctx));

  FILE *o = stdout;
  if (argc > 2) {
    o = std::fopen(argv[2], "w");
    if (!o)
      return 1;
  }
  std::fprintf(o,
               "{\"device\":\"B60\",\"tensors\":%" PRIu64
               ",\"verified\":%" PRIu64 ",\"failed\":%zu,"
               "\"payload_arena\":%" PRIu64 ",\"scale_arena\":%" PRIu64
               ",\"h2d_gbs\":%.2f,\"d2h_gbs\":%.2f,\"alloc_ms\":%.1f}\n",
               n, verified, failed.size(), pay_hi - pay_lo, scSize,
               total_gb / h2d_s, total_gb / d2h_s,
               std::chrono::duration<double>(t_alloc1 - t_alloc0).count() *
                   1e3);
  for (auto &nm : failed)
    std::fprintf(stderr, "MISMATCH: %s\n", nm.c_str());
  if (o != stdout)
    std::fclose(o);
  return failed.empty() ? 0 : 3;
}
