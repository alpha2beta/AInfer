// T7.2 chained drafts: Int4GemvM3 (triple-lane GEMV) vs M2+M1 reference.
//
// Runs gemvm3 (12 args) on trunk layer-0 q_proj real INT4 weights with 3
// deterministic INT8 activation vectors, and compares bitwise against
// gemvm2 (lanes 0-1) + gemv (lane 2) from the SAME .spv image. All three
// lanes must match exactly (same unpack + dp4a math, extra lane only).
// Reports medians + (M2+M1)/M3 speedup. Bitwise gate, like all T7.x proofs.
// Usage: gemvm3_replay <gemvm3.spv> <model.binfer> [report.json]
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
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

struct Entry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
};
static uint32_t rd32(std::ifstream &f) {
  uint32_t v;
  f.read((char *)&v, 4);
  return v;
}
static uint64_t rd64(std::ifstream &f) {
  uint64_t v;
  f.read((char *)&v, 8);
  return v;
}

static uint64_t prng = 0x51ab3f019c7d2e44ull;
static int irnd(int n) {
  prng = prng * 6364136223846793005ull + 1442695040888963407ull;
  return (int)((prng >> 33) % (uint64_t)n);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: gemvm3_replay <gemvm3.spv> <model.binfer> "
                         "[report.json]\n");
    return 2;
  }
  const int M = 12288, K = 5120, GG = 40, REPS = 11;
  // Layer 0 is linear-attention (no q_proj); layer 11 is the first full
  // layer whose q_proj is stored (M=12288=q heads*512? verified by scales).
  const char *WN = "model.language_model.layers.11.self_attn.q_proj.weight";

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
  auto alloc = [&](size_t n) -> void * {
    void *p = nullptr;
    if (zeMemAllocDevice(ctx, &mdesc, n, 4096, dev, &p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "alloc %zu failed\n", n);
      std::exit(1);
    }
    return p;
  };

  // Module with all three entries (one image by content selection).
  ze_module_handle_t mod = nullptr;
  ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                             nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV,
                             spvN,
                             spv.data(),
                             nullptr,
                             nullptr};
  CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
  auto mkker = [&](const char *entry) -> ze_kernel_handle_t {
    ze_kernel_handle_t ker = nullptr;
    ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                              entry};
    if (zeKernelCreate(mod, &kdesc, &ker) != ZE_RESULT_SUCCESS ||
        zeKernelSetGroupSize(ker, 1, 1, 1) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "kernel %s setup failed\n", entry);
      std::exit(1);
    }
    return ker;
  };
  ze_kernel_handle_t k3 = mkker("_ZTS10Int4GemvM3");
  ze_kernel_handle_t k2 = mkker("_ZTS10Int4GemvM2");
  ze_kernel_handle_t k1 = mkker("_ZTS8Int4Gemv");
  // .binfer dir (§5: no length prefix; table 32 B, dir entries 192 B).
  std::ifstream f(argv[2], std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "no model\n");
    return 2;
  }
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
  uint64_t dir_off = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5)
      dir_off = off;
    (void)nb;
  }
  f.seekg((std::streamoff)dir_off);
  std::vector<Entry> ents(n);
  for (uint64_t i = 0; i < n; ++i) {
    Entry &e = ents[i];
    f.read(e.name, 64);
    e.name[63] = 0;
    f.seekg(1 + 7 + 64 + 1 + 1 + 2 + 4, std::ios::cur);
    e.sc_off = rd64(f);
    e.sc_bytes = rd64(f);
    e.d_off = rd64(f);
    e.d_bytes = rd64(f);
    e.crc = rd32(f);
    f.seekg(12, std::ios::cur);
  }
  const Entry *we = nullptr;
  for (auto &e : ents)
    if (WN == std::string(e.name))
      we = &e;
  if (!we) {
    std::fprintf(stderr, "missing %s\n", WN);
    return 2;
  }
  std::vector<uint8_t> packed(we->d_bytes);
  f.clear();
  f.seekg((std::streamoff)we->d_off);
  f.read((char *)packed.data(), we->d_bytes);
  size_t nsc = we->sc_bytes / 2;
  std::vector<uint16_t> scales(nsc);
  f.clear();
  f.seekg((std::streamoff)we->sc_off);
  f.read((char *)scales.data(), we->sc_bytes);
  if ((int)nsc != M * GG) {
    std::fprintf(stderr, "shape surprise: scales %zu vs %d\n", nsc, M * GG);
    return 2;
  }

  // Deterministic INT8 activations + fp32 scales (3 lanes).
  std::vector<int8_t> xq[3];
  std::vector<float> sq[3];
  for (int l = 0; l < 3; ++l) {
    xq[l].resize(K);
    sq[l].resize(GG);
    for (int i = 0; i < K; ++i)
      xq[l][i] = (int8_t)(irnd(17) - 8);
    for (int g = 0; g < GG; ++g)
      sq[l][g] = 0.02f + 0.001f * (float)irnd(20);
  }

  // Upload via one copy list.
  ze_command_list_handle_t up = nullptr;
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                  0, 0};
  CHECK(zeCommandListCreate(ctx, dev, &ldesc, &up));
  void *dP = alloc(packed.size()), *dS = alloc(scales.size() * 2);
  void *dXQ[3], *dSQ[3];
  for (int l = 0; l < 3; ++l) {
    dXQ[l] = alloc(K);
    dSQ[l] = alloc((size_t)GG * 4);
  }
  void *dY3 = alloc((size_t)M * 3 * 4), *dY2 = alloc((size_t)M * 2 * 4),
       *dY1 = alloc((size_t)M * 4);
  CHECK(zeCommandListAppendMemoryCopy(up, dP, packed.data(), packed.size(),
                                      nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(up, dS, scales.data(), scales.size() * 2,
                                      nullptr, 0, nullptr));
  for (int l = 0; l < 3; ++l) {
    CHECK(zeCommandListAppendMemoryCopy(up, dXQ[l], xq[l].data(), K, nullptr,
                                        0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dSQ[l], sq[l].data(),
                                        (size_t)GG * 4, nullptr, 0, nullptr));
  }
  CHECK(zeCommandListClose(up));
  ze_command_queue_handle_t qq = nullptr;
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandQueueCreate(ctx, dev, &qdesc, &qq));
  CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &up, nullptr));
  CHECK(zeCommandQueueSynchronize(qq, UINT64_MAX));

  // Timed exec: regular list, reset + re-record per rep, queue sync.
  auto seta = [&](ze_kernel_handle_t k, int i, const void *v, size_t nb) {
    CHECK(zeKernelSetArgumentValue(k, (uint32_t)i, nb, v));
  };
  auto setp = [&](ze_kernel_handle_t k, int i, void *p) {
    seta(k, i, &p, sizeof(void *));
  };
  int kk = K;
  auto run = [&](ze_kernel_handle_t k, const char *tag) -> double {
    ze_group_count_t gc = {(uint32_t)M, 1, 1};
    std::vector<double> ts;
    for (int r = 0; r < REPS; ++r) {
      ze_command_list_handle_t ex = nullptr;
      CHECK(zeCommandListCreate(ctx, dev, &ldesc, &ex));
      CHECK(zeCommandListAppendLaunchKernel(ex, k, &gc, nullptr, 0, nullptr));
      CHECK(zeCommandListAppendBarrier(ex, nullptr, 0, nullptr));
      CHECK(zeCommandListClose(ex));
      double t0 = now_ns();
      CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &ex, nullptr));
      CHECK(zeCommandQueueSynchronize(qq, UINT64_MAX));
      ts.push_back(now_ns() - t0);
      CHECK(zeCommandListDestroy(ex));
    }
    double m = med(ts);
    std::printf("%s med %.1f us\n", tag, m / 1000.0);
    return m;
  };
  // M3: 12 args (Y0,Y1,Y2,P,S,XQ0,SQ0,XQ1,SQ1,XQ2,SQ2,K).
  void *y3[3] = {(char *)dY3, (char *)dY3 + (size_t)M * 4,
                 (char *)dY3 + (size_t)M * 8};
  setp(k3, 0, y3[0]);
  setp(k3, 1, y3[1]);
  setp(k3, 2, y3[2]);
  setp(k3, 3, dP);
  setp(k3, 4, dS);
  setp(k3, 5, dXQ[0]);
  setp(k3, 6, dSQ[0]);
  setp(k3, 7, dXQ[1]);
  setp(k3, 8, dSQ[1]);
  setp(k3, 9, dXQ[2]);
  setp(k3, 10, dSQ[2]);
  seta(k3, 11, &kk, sizeof(int));
  // M2: 9 args (lanes 0-1).
  void *y2[2] = {dY2, (char *)dY2 + (size_t)M * 4};
  setp(k2, 0, y2[0]);
  setp(k2, 1, y2[1]);
  setp(k2, 2, dP);
  setp(k2, 3, dS);
  setp(k2, 4, dXQ[0]);
  setp(k2, 5, dSQ[0]);
  setp(k2, 6, dXQ[1]);
  setp(k2, 7, dSQ[1]);
  seta(k2, 8, &kk, sizeof(int));
  // M1: 6 args (lane 2).
  setp(k1, 0, dY1);
  setp(k1, 1, dP);
  setp(k1, 2, dS);
  setp(k1, 3, dXQ[2]);
  setp(k1, 4, dSQ[2]);
  seta(k1, 5, &kk, sizeof(int));

  double t3 = run(k3, "gemvm3");
  double t2 = run(k2, "gemvm2");
  double t1 = run(k1, "gemv  ");
  std::vector<float> h3((size_t)M * 3), h2((size_t)M * 2), h1(M);
  auto dl = [&](void *d, void *h, size_t n) {
    ze_command_list_handle_t cp = nullptr;
    CHECK(zeCommandListCreate(ctx, dev, &ldesc, &cp));
    CHECK(zeCommandListAppendMemoryCopy(cp, h, d, n, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cp));
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &cp, nullptr));
    CHECK(zeCommandQueueSynchronize(qq, UINT64_MAX));
  };
  dl(dY3, h3.data(), h3.size() * 4);
  dl(dY2, h2.data(), h2.size() * 4);
  dl(dY1, h1.data(), h1.size() * 4);
  int bad = 0;
  double worst = 0;
  for (int m = 0; m < M; ++m) {
    float a[3] = {h3[(size_t)m], h3[(size_t)M + m], h3[(size_t)M * 2 + m]};
    float b[3] = {h2[(size_t)m], h2[(size_t)M + m], h1[m]};
    for (int l = 0; l < 3; ++l) {
      double d = std::fabs((double)a[l] - (double)b[l]);
      // Bitwise gate: exact bit equality (same math, extra lane only).
      uint32_t ua, ub;
      std::memcpy(&ua, &a[l], 4);
      std::memcpy(&ub, &b[l], 4);
      if (ua != ub) {
        if (bad < 5)
          std::printf("DIFF m=%d lane=%d m3=%a m2m1=%a\n", m, l, a[l], b[l]);
        ++bad;
      }
      if (d > worst)
        worst = d;
    }
  }
  double speed = (t2 + t1) / t3;
  std::printf("MISMATCH=%d/%d worstabs=%.3g med3=%.1fus med2=%.1fus "
              "med1=%.1fus speedup=%.2f\n",
              bad, M * 3, worst, t3 / 1000.0, t2 / 1000.0, t1 / 1000.0,
              speed);
  bool ok = (bad == 0);
  if (argc >= 4) {
    FILE *rf = std::fopen(argv[3], "w");
    if (rf) {
      std::fprintf(rf,
                   "{\"task\":\"T7.2-gemvm3\",\"weight\":\"%s\","
                   "\"M\":%d,\"K\":%d,\"mismatch\":%d,\"worstabs\":%.3g,"
                   "\"med_us\":{\"m3\":%.1f,\"m2\":%.1f,\"m1\":%.1f},"
                   "\"speedup_m2m1_over_m3\":%.3f,\"ok\":%s}\n",
                   WN, M, K, bad, worst, t3 / 1000.0, t2 / 1000.0,
                   t1 / 1000.0, speed, ok ? "true" : "false");
      std::fclose(rf);
    }
  }
  std::printf(ok ? "GEMVM3-OK\n" : "GEMVM3-FAIL\n");
  return ok ? 0 : 1;
}
