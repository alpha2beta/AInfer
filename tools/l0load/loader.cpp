// AInfer Level Zero MoE loader (T3.5, T3.6, T1.5).
// Parses the .binfer container (supporting v1.1 MoE Section 6),
// validates all headers/CRCs/descriptors before allocation,
// allocates static device arenas on Intel Arc 140V (Core Ultra 7 258V) or Arc Pro B60,
// streams all payloads host->device, then reads back and verifies
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
  uint64_t v = 0;
  f.read((char *)&v, 8);
  return v;
}

static uint32_t rd32(std::ifstream &f) {
  uint32_t v = 0;
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
  uint64_t n = rd64(f);
  uint64_t table_off = rd64(f);
  uint32_t scount = rd32(f);
  uint32_t align = rd32(f);
  uint64_t total = rd64(f);
  (void)align;

  bool is_moe = bool(flags & 2);

  // Version gate
  if (ver != 1) {
    std::fprintf(stderr, "unsupported version %u\n", ver);
    return 2;
  }
  if (flags & ~3) {
    std::fprintf(stderr, "reserved flags set\n");
    return 2;
  }

  // File size bounds checking
  f.seekg(0, std::ios::end);
  uint64_t fsize = (uint64_t)f.tellg();
  f.clear();

  if (total != fsize) {
    std::fprintf(stderr, "file size mismatch header=%" PRIu64 " actual=%" PRIu64 "\n", total, fsize);
    return 2;
  }
  if (table_off > fsize || (uint64_t)scount * 32 > fsize - table_off) {
    std::fprintf(stderr, "section table out of range\n");
    return 2;
  }
  if (n > fsize / 192 + 1) {
    std::fprintf(stderr, "tensor count out of range\n");
    return 2;
  }

  // Parse Section Table
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  uint32_t dir_crc = 0;
  uint64_t moe_off = 0, moe_bytes = 0;
  uint32_t moe_crc = 0;
  bool has_dir = false, has_moe = false;

  for (uint32_t i = 0; i < scount; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    uint32_t c = rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5) {
      dir_off = off;
      dir_bytes = nb;
      dir_crc = c;
      has_dir = true;
    } else if (sid == 6) {
      moe_off = off;
      moe_bytes = nb;
      moe_crc = c;
      has_moe = true;
    }
  }

  if (!has_dir || dir_bytes != n * 192) {
    std::fprintf(stderr, "dir size mismatch\n");
    return 2;
  }
  if (dir_off > fsize || dir_bytes > fsize - 32 - dir_off) {
    std::fprintf(stderr, "dir span out of range\n");
    return 2;
  }

  // Validate MoE Section 6 if enabled
  if (is_moe) {
    if (!has_moe || moe_off > fsize || moe_bytes > fsize - 32 - moe_off || moe_bytes < 8 + 64 + 40 * 32) {
      std::fprintf(stderr, "moe section missing or truncated\n");
      return 2;
    }
    f.seekg((std::streamoff)moe_off);
    uint64_t mlen = rd64(f);
    if (mlen != moe_bytes - 8) {
      std::fprintf(stderr, "moe section length mismatch\n");
      return 2;
    }
    std::vector<uint8_t> mbuf(mlen);
    f.read((char *)mbuf.data(), mlen);
    uint32_t chk_crc = crc32_update(0, (const uint8_t *)&mlen, 8);
    chk_crc = crc32_update(chk_crc, mbuf.data(), mlen);
    if (chk_crc != moe_crc) {
      std::fprintf(stderr, "moe section CRC mismatch\n");
      return 2;
    }

    uint32_t num_experts = *(const uint32_t *)(mbuf.data() + 0);
    uint32_t num_experts_per_tok = *(const uint32_t *)(mbuf.data() + 4);
    uint32_t layer_count = *(const uint32_t *)(mbuf.data() + 24);
    if (num_experts != 256 || num_experts_per_tok != 8 || layer_count != 40) {
      std::fprintf(stderr, "moe architecture mismatch: exp=%u, per_tok=%u, layers=%u\n",
                   num_experts, num_experts_per_tok, layer_count);
      return 2;
    }

    const uint8_t *desc_ptr = mbuf.data() + 64;
    for (uint32_t l = 0; l < layer_count; ++l) {
      const uint32_t *d = (const uint32_t *)(desc_ptr + l * 32);
      uint32_t l_idx = d[0];
      uint32_t gate_id = d[1];
      uint32_t shared_gate_id = d[2];
      uint32_t exp_gu_id = d[3];
      uint32_t exp_dn_id = d[4];
      uint32_t sh_dn_id = d[5];
      uint32_t sh_gp_id = d[6];
      uint32_t sh_up_id = d[7];
      if (l_idx != l || gate_id >= n || shared_gate_id >= n || exp_gu_id >= n ||
          exp_dn_id >= n || sh_dn_id >= n || sh_gp_id >= n || sh_up_id >= n) {
        std::fprintf(stderr, "invalid MoE descriptor index on layer %u\n", l);
        return 2;
      }
    }
  }

  // Parse Directory
  std::vector<Entry> entries(n);
  f.seekg((std::streamoff)dir_off);
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

    if (e.d_bytes == 0 || e.d_off > fsize || e.d_bytes > fsize - 32 - e.d_off) {
      std::fprintf(stderr, "payload span out of range: %s\n", e.name);
      return 2;
    }
    if (e.sc_bytes && (e.sc_off > fsize || e.sc_bytes > fsize - 32 - e.sc_off)) {
      std::fprintf(stderr, "scale span out of range: %s\n", e.name);
      return 2;
    }
  }

  // Verify Directory CRC
  {
    f.clear();
    f.seekg((std::streamoff)dir_off);
    std::vector<char> buf(dir_bytes);
    f.read(buf.data(), dir_bytes);
    if (crc32_update(0, (const uint8_t *)buf.data(), dir_bytes) != dir_crc) {
      std::fprintf(stderr, "dir CRC mismatch\n");
      return 2;
    }
  }

  // Initialize Level Zero & select device (Arc 140V 0x64a0 or B60 0xe211)
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));

  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  char devName[256] = "Unknown Intel GPU";

  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS)
      continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS &&
          pr.vendorId == 0x8086) {
        if (pr.deviceId == 0x64a0 || pr.deviceId == 0xe211 || dev == nullptr) {
          dev = v;
          drv = d;
          std::strncpy(devName, pr.name, sizeof(devName) - 1);
        }
      }
    }
  }
  if (!dev) {
    std::fprintf(stderr, "No supported Intel GPU found\n");
    return 2;
  }

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // Static arenas: one device allocation each for payloads and scales
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

  // Synchronous immediate list
  ze_command_list_handle_t list = nullptr;
  ze_command_queue_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &ldesc, &list));

  const size_t CH = 64u << 20; // 64 MiB staging buffer
  std::vector<char> staging(CH);

  auto copy_to_device = [&](uint64_t foff, void *arena, uint64_t alo, uint64_t len) -> bool {
    uint64_t done = 0;
    while (done < len) {
      size_t c = (size_t)((len - done > CH) ? CH : (len - done));
      f.clear();
      f.seekg((std::streamoff)(foff + done));
      f.read(staging.data(), c);
      ze_result_t res = zeCommandListAppendMemoryCopy(
          list, (char *)arena + (foff + done - alo), staging.data(), c,
          nullptr, 0, nullptr);
      if (res != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "copy failed with L0 error %d\n", (int)res);
        return false;
      }
      done += c;
    }
    return true;
  };

  auto t_h2d0 = std::chrono::steady_clock::now();
  copy_to_device(pay_lo, payArena, pay_lo, pay_hi - pay_lo);
  if (scSize)
    copy_to_device(sc_lo, scArena, sc_lo, scSize);
  auto t_h2d1 = std::chrono::steady_clock::now();

  // Readback verification per-tensor using 64 MiB staging buffer (preserves host RAM)
  uint64_t verified = 0;
  std::vector<std::string> failed;
  auto t_d2h0 = std::chrono::steady_clock::now();

  for (auto &e : entries) {
    uint32_t calc_crc = 0;
    uint64_t done = 0;
    while (done < e.d_bytes) {
      size_t c = (size_t)((e.d_bytes - done > CH) ? CH : (e.d_bytes - done));
      CHECK(zeCommandListAppendMemoryCopy(
          list, staging.data(), (char *)payArena + (e.d_off - pay_lo + done), c,
          nullptr, 0, nullptr));
      calc_crc = crc32_update(calc_crc, (const uint8_t *)staging.data(), c);
      done += c;
    }
    if (calc_crc == e.crc)
      ++verified;
    else
      failed.push_back(e.name);
  }
  auto t_d2h1 = std::chrono::steady_clock::now();

  double h2d_s = std::chrono::duration<double>(t_h2d1 - t_h2d0).count();
  double d2h_s = std::chrono::duration<double>(t_d2h1 - t_d2h0).count();
  double total_gb = ((pay_hi - pay_lo) + scSize) / 1e9;

  CHECK(zeCommandListDestroy(list));
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
               "{\"device\":\"%s\",\"is_moe\":%s,\"tensors\":%" PRIu64
               ",\"verified\":%" PRIu64 ",\"failed\":%zu,"
               "\"payload_arena\":%" PRIu64 ",\"scale_arena\":%" PRIu64
               ",\"h2d_gbs\":%.2f,\"d2h_gbs\":%.2f,\"alloc_ms\":%.1f}\n",
               devName, is_moe ? "true" : "false", n, verified, failed.size(),
               pay_hi - pay_lo, scSize, total_gb / h2d_s, total_gb / d2h_s,
               std::chrono::duration<double>(t_alloc1 - t_alloc0).count() * 1e3);
  for (auto &nm : failed)
    std::fprintf(stderr, "MISMATCH: %s\n", nm.c_str());
  if (o != stdout)
    std::fclose(o);

  return failed.empty() ? 0 : 3;
}
