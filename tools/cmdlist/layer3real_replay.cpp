// T5.3 adoption on REAL weights (full-attention): linear... full layer 3 as a
// SINGLE recorded raw-L0 list from the real .binfer arenas (same L0 context as
// the SYCL loop) plus L0-context scratch/caches/small-weights/RoPE tables.
// Input = certified dump slot 2 (post-layer-2 dX at step 0); reference = dump
// slot 3. KV caches zeroed (pos 0 -> single-slot T=1); position/active-length
// ride the T5.4 control block (host 16 B update, never baked).
// Checks: recorded-list output vs dump (tol 1e-4; flip physics per layerlin)
// + reset rerun bitwise identical.
// Usage: layer3real <11 spv> <model.binfer> <dump.bin> [report.json]
//   spv order: norm gemv splitqk batchnorm rope kvappend attn scalesmax
//              quantize silu resaddf
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

struct Entry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
};
struct DecodeControl {
  int token_id, position, active_length, selected_token;
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
static float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
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
  if (argc < 14) {
    std::fprintf(stderr, "usage: layer3real <11 spv> <model> <dump> [rep]\n");
    return 2;
  }
  const int H = 5120, QW = 12288, KVW = 1024, QN = 6144, I = 17408;
  const int TMAX = 8;
  const char *entries[] = {
      "_ZTS8RMSNormW", "_ZTS8Int4Gemv", "_ZTS7SplitQK", "_ZTS9BatchNorm",
      "_ZTS9RopeApply", "_ZTS8KvAppend", "_ZTS8AttnCore", "_ZTS9ScalesMax",
      "_ZTS8Quantize", "_ZTS7SiluMul", "_ZTS7ResAddF"};
  auto load_spv = [&](const char *path) {
    FILE *sf = std::fopen(path, "rb");
    if (!sf) {
      std::fprintf(stderr, "no spv: %s\n", path);
      std::exit(2);
    }
    std::fseek(sf, 0, SEEK_END);
    size_t n = std::ftell(sf);
    std::fseek(sf, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (std::fread(v.data(), 1, n, sf) != n)
      std::exit(2);
    std::fclose(sf);
    return v;
  };

  const char *modelPath = argv[12];
  std::ifstream f(modelPath, std::ios::binary);
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
  uint32_t ver = rd32(f), flags = rd32(f);
  uint64_t n;
  f.read((char *)&n, 8);
  uint64_t table_off = rd64(f);
  (void)ver;
  (void)flags;
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  for (int i = 0; i < 5; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5) {
      dir_off = off;
      dir_bytes = nb;
    }
  }
  (void)dir_bytes;
  std::vector<Entry> ents(n);
  f.seekg((std::streamoff)dir_off);
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
  auto find = [&](const std::string &nm) -> const Entry * {
    for (auto &e : ents)
      if (nm == e.name)
        return &e;
    return nullptr;
  };

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
  uint64_t pay_lo = UINT64_MAX, pay_hi = 0, sc_lo = UINT64_MAX, sc_hi = 0;
  for (auto &e : ents) {
    pay_lo = e.d_off < pay_lo ? e.d_off : pay_lo;
    pay_hi = e.d_off + e.d_bytes > pay_hi ? e.d_off + e.d_bytes : pay_hi;
    if (e.sc_bytes) {
      sc_lo = e.sc_off < sc_lo ? e.sc_off : sc_lo;
      sc_hi = e.sc_off + e.sc_bytes > sc_hi ? e.sc_off + e.sc_bytes : sc_hi;
    }
  }
  void *payArena = nullptr, *scArena = nullptr;
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)(pay_hi - pay_lo), 4096, dev,
                         &payArena));
  CHECK(zeMemAllocDevice(ctx, &mdesc, (size_t)(sc_hi - sc_lo), 4096, dev,
                         &scArena));
  auto alloc = [&](size_t sz) -> void * {
    void *p = nullptr;
    if (zeMemAllocDevice(ctx, &mdesc, sz, 4096, dev, &p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "alloc %zu failed\n", sz);
      std::exit(1);
    }
    return p;
  };
  ze_command_list_handle_t up = nullptr;
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK(zeCommandListCreateImmediate(ctx, dev, &idesc, &up));
  {
    const size_t CH = 1u << 28;
    std::vector<char> staging(CH);
    auto upspan = [&](uint64_t foff, void *arena, uint64_t alo, uint64_t len) {
      uint64_t done = 0;
      while (done < len) {
        size_t c = (size_t)((len - done > CH) ? CH : (len - done));
        f.clear();
        f.seekg((std::streamoff)(foff + done));
        f.read(staging.data(), c);
        ze_result_t r = zeCommandListAppendMemoryCopy(
            up, (char *)arena + (foff + done - alo), staging.data(), c,
            nullptr, 0, nullptr);
        if (r != ZE_RESULT_SUCCESS) {
          std::fprintf(stderr, "upload failed\n");
          std::exit(1);
        }
        done += c;
      }
    };
    upspan(pay_lo, payArena, pay_lo, pay_hi - pay_lo);
    upspan(sc_lo, scArena, sc_lo, sc_hi - sc_lo);
  }
  void *dX = alloc(H * 4), *dH = alloc(H * 4), *dTmp = alloc(H * 4),
       *dMix = alloc(H * 4);
  void *dQ16 = alloc(QW * 4), *dK16 = alloc(KVW * 4), *dV16 = alloc(KVW * 4),
       *dKn = alloc(KVW * 4);
  void *dQn = alloc(QN * 4), *dGate = alloc(QN * 4), *dAtt = alloc(QN * 4);
  void *dKc = alloc((size_t)4 * TMAX * 256 * 2),
       *dVc = alloc((size_t)4 * TMAX * 256 * 2);
  void *dG17 = alloc(I * 4), *dU17 = alloc(I * 4), *dQ8 = alloc(I),
       *dSq = alloc(136 * 4), *dCtrl = alloc(sizeof(DecodeControl)),
       *dWts = alloc((size_t)24 * TMAX * 4);
  void *dInN = alloc(H * 4), *dPostN = alloc(H * 4), *dQNW = alloc(256 * 4),
       *dKNW = alloc(256 * 4), *dCos = alloc(64 * TMAX * 4),
       *dSin = alloc(64 * TMAX * 4);
  auto load_small = [&](const std::string &nm, void *dst, size_t cnt) {
    const Entry *e = find(nm);
    if (!e) {
      std::fprintf(stderr, "missing %s\n", nm.c_str());
      std::exit(1);
    }
    std::vector<uint16_t> raw(cnt);
    f.clear();
    f.seekg((std::streamoff)e->d_off);
    f.read((char *)raw.data(), cnt * 2);
    std::vector<float> out(cnt);
    for (size_t i = 0; i < cnt; ++i)
      out[i] = bf16_to_f32(raw[i]);
    CHECK(zeCommandListAppendMemoryCopy(up, dst, out.data(), cnt * 4, nullptr,
                                        0, nullptr));
  };
  const std::string LP = "model.language_model.layers.3.";
  load_small(LP + "input_layernorm.weight", dInN, H);
  load_small(LP + "post_attention_layernorm.weight", dPostN, H);
  load_small(LP + "self_attn.q_norm.weight", dQNW, 256);
  load_small(LP + "self_attn.k_norm.weight", dKNW, 256);
  {
    std::vector<float> cs(64 * TMAX), sn(64 * TMAX);
    for (int t = 0; t < TMAX; ++t)
      for (int i = 0; i < 64; ++i) {
        double inv = 1.0 / std::pow(10000000.0, (double)(2 * (i % 32)) / 64.0);
        double ang = (double)t * inv;
        cs[t * 64 + i] = (float)std::cos(ang);
        sn[t * 64 + i] = (float)std::sin(ang);
      }
    CHECK(zeCommandListAppendMemoryCopy(up, dCos, cs.data(), cs.size() * 4,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dSin, sn.data(), sn.size() * 4,
                                        nullptr, 0, nullptr));
  }
  // Input = dump slot 2 (post-layer-2 dX at step 0).
  std::ifstream rf(argv[13], std::ios::binary);
  if (!rf) {
    std::fprintf(stderr, "no dump\n");
    return 2;
  }
  std::vector<float> xin(H);
  rf.seekg((std::streamoff)(2 * H * 4));
  rf.read((char *)xin.data(), H * 4);
  if (!rf) {
    std::fprintf(stderr, "short dump\n");
    return 2;
  }
  std::vector<float> ref(H);
  rf.seekg((std::streamoff)(3 * H * 4));
  rf.read((char *)ref.data(), H * 4);
  if (!rf) {
    std::fprintf(stderr, "short dump ref\n");
    return 2;
  }
  std::vector<uint16_t> zKc(4 * TMAX * 256, 0), zVc(4 * TMAX * 256, 0);

  ze_kernel_handle_t kh[11] = {nullptr};
  std::vector<std::vector<uint8_t>> spvs;
  for (int i = 0; i < 11; ++i) {
    FILE *sf = std::fopen(argv[1 + i], "rb");
    if (!sf) {
      std::fprintf(stderr, "no spv\n");
      return 2;
    }
    std::fseek(sf, 0, SEEK_END);
    size_t nn = std::ftell(sf);
    std::fseek(sf, 0, SEEK_SET);
    spvs.emplace_back(nn);
    if (std::fread(spvs.back().data(), 1, nn, sf) != nn)
      return 2;
    std::fclose(sf);
  }
  enum K {
    NORM,
    GEMV,
    SPLIT,
    BNORM,
    ROPE,
    KV,
    ATTN,
    SCALES,
    QUANT,
    SILU,
    RES
  };
  for (int i = 0; i < 11; ++i) {
    ze_module_handle_t mod = nullptr;
    ze_module_desc_t mddesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                               nullptr,
                               ZE_MODULE_FORMAT_IL_SPIRV,
                               spvs[i].size(),
                               spvs[i].data(),
                               nullptr,
                               nullptr};
    CHECK(zeModuleCreate(ctx, dev, &mddesc, &mod, nullptr));
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                           entries[i]};
    if (zeKernelCreate(mod, &kd, &kh[i]) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "entry %s not found\n", entries[i]);
      return 1;
    }
    CHECK(zeKernelSetGroupSize(kh[i], 1, 1, 1));
  }
  CHECK(zeKernelSetGroupSize(kh[NORM], 256, 1, 1)); // T6.1 parallel norm

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
  auto setarg = [&](ze_kernel_handle_t k, uint32_t idx, size_t sz,
                    const void *p) {
    if (zeKernelSetArgumentValue(k, idx, sz, p) != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "setArg %u failed\n", idx);
      std::exit(1);
    }
  };
  auto launch = [&](ze_kernel_handle_t k, uint32_t count) {
    ze_group_count_t gc = {count, 1, 1};
    ze_result_t r1 =
        zeCommandListAppendLaunchKernel(reg, k, &gc, nullptr, 0, nullptr);
    ze_result_t r2 = zeCommandListAppendBarrier(reg, nullptr, 0, nullptr);
    if (r1 != ZE_RESULT_SUCCESS || r2 != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "append failed\n");
      std::exit(1);
    }
  };
  auto gemvE = [&](const std::string &nm, int M, int KK, void *Y) {
    const Entry *e = find(nm);
    const void *dP = (const uint8_t *)payArena + (e->d_off - pay_lo);
    const void *dSc = (const char *)scArena + (e->sc_off - sc_lo);
    setarg(kh[GEMV], 0, sizeof(void *), &Y);
    setarg(kh[GEMV], 1, sizeof(void *), &dP);
    setarg(kh[GEMV], 2, sizeof(void *), &dSc);
    setarg(kh[GEMV], 3, sizeof(void *), &dQ8);
    setarg(kh[GEMV], 4, sizeof(void *), &dSq);
    setarg(kh[GEMV], 5, sizeof(int), &KK);
    launch(kh[GEMV], M);
  };
  auto xq = [&](void *X, int nn) {
    int gg = nn / 128;
    setarg(kh[SCALES], 0, sizeof(void *), &X);
    setarg(kh[SCALES], 1, sizeof(void *), &dSq);
    setarg(kh[SCALES], 2, sizeof(int), &gg);
    launch(kh[SCALES], gg);
    setarg(kh[QUANT], 0, sizeof(void *), &X);
    setarg(kh[QUANT], 1, sizeof(void *), &dSq);
    setarg(kh[QUANT], 2, sizeof(void *), &dQ8);
    launch(kh[QUANT], nn);
  };
  auto norm = [&](void *Y, void *X, void *W) {
    int nn = H;
    setarg(kh[NORM], 0, sizeof(void *), &Y);
    setarg(kh[NORM], 1, sizeof(void *), &X);
    setarg(kh[NORM], 2, sizeof(void *), &W);
    setarg(kh[NORM], 3, sizeof(int), &nn);
    setarg(kh[NORM], 4, (size_t)256 * 8, nullptr);
    launch(kh[NORM], 1);
  };
  auto res = [&](void *Y, void *A, void *B) {
    setarg(kh[RES], 0, sizeof(void *), &Y);
    setarg(kh[RES], 1, sizeof(void *), &A);
    setarg(kh[RES], 2, sizeof(void *), &B);
    launch(kh[RES], H);
  };
  int n256 = 256, tmax = TMAX;
  norm(dH, dX, dInN);
  xq(dH, H);
  gemvE(LP + "self_attn.q_proj.weight", QW, H, dQ16);
  gemvE(LP + "self_attn.k_proj.weight", KVW, H, dK16);
  gemvE(LP + "self_attn.v_proj.weight", KVW, H, dV16);
  setarg(kh[SPLIT], 0, sizeof(void *), &dQ16);
  setarg(kh[SPLIT], 1, sizeof(void *), &dQn);
  setarg(kh[SPLIT], 2, sizeof(void *), &dGate);
  launch(kh[SPLIT], QN);
  setarg(kh[BNORM], 0, sizeof(void *), &dQn);
  setarg(kh[BNORM], 1, sizeof(void *), &dQn);
  setarg(kh[BNORM], 2, sizeof(void *), &dQNW);
  setarg(kh[BNORM], 3, sizeof(int), &n256);
  launch(kh[BNORM], 24);
  setarg(kh[BNORM], 0, sizeof(void *), &dKn);
  setarg(kh[BNORM], 1, sizeof(void *), &dK16);
  setarg(kh[BNORM], 2, sizeof(void *), &dKNW);
  setarg(kh[BNORM], 3, sizeof(int), &n256);
  launch(kh[BNORM], 4);
  setarg(kh[ROPE], 0, sizeof(void *), &dQn);
  setarg(kh[ROPE], 1, sizeof(void *), &dKn);
  setarg(kh[ROPE], 2, sizeof(void *), &dCos);
  setarg(kh[ROPE], 3, sizeof(void *), &dSin);
  setarg(kh[ROPE], 4, sizeof(void *), &dCtrl);
  launch(kh[ROPE], 28);
  setarg(kh[KV], 0, sizeof(void *), &dKc);
  setarg(kh[KV], 1, sizeof(void *), &dVc);
  setarg(kh[KV], 2, sizeof(void *), &dKn);
  setarg(kh[KV], 3, sizeof(void *), &dV16);
  setarg(kh[KV], 4, sizeof(void *), &dCtrl);
  setarg(kh[KV], 5, sizeof(int), &tmax);
  launch(kh[KV], KVW);
  setarg(kh[ATTN], 0, sizeof(void *), &dAtt);
  setarg(kh[ATTN], 1, sizeof(void *), &dQn);
  setarg(kh[ATTN], 2, sizeof(void *), &dKc);
  setarg(kh[ATTN], 3, sizeof(void *), &dVc);
  setarg(kh[ATTN], 4, sizeof(void *), &dGate);
  setarg(kh[ATTN], 5, sizeof(void *), &dCtrl);
  setarg(kh[ATTN], 6, sizeof(int), &tmax);
  setarg(kh[ATTN], 7, sizeof(void *), &dWts);
  launch(kh[ATTN], 24);
  xq(dAtt, QN);
  gemvE(LP + "self_attn.o_proj.weight", H, QN, dMix);
  res(dTmp, dX, dMix);
  norm(dH, dTmp, dPostN);
  xq(dH, H);
  gemvE(LP + "mlp.gate_proj.weight", I, H, dG17);
  gemvE(LP + "mlp.up_proj.weight", I, H, dU17);
  setarg(kh[SILU], 0, sizeof(void *), &dG17);
  setarg(kh[SILU], 1, sizeof(void *), &dU17);
  setarg(kh[SILU], 2, sizeof(void *), &dG17);
  launch(kh[SILU], I);
  xq(dG17, I);
  gemvE(LP + "mlp.down_proj.weight", H, I, dMix);
  res(dX, dTmp, dMix);
  ze_fence_handle_t fence = nullptr;
  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK(zeFenceCreate(qq, &fdesc, &fence));
  CHECK(zeCommandListClose(reg));

  std::vector<double> tRep;
  bool ok = true;
  double worstRel = 0;
  std::vector<float> hOut(H), hOut2(H);
  for (int run = 0; run < 2 && ok; ++run) {
    CHECK(zeCommandListAppendMemoryCopy(up, dX, xin.data(), H * 4, nullptr, 0,
                                        nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dKc, zKc.data(), zKc.size() * 2,
                                        nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(up, dVc, zVc.data(), zVc.size() * 2,
                                        nullptr, 0, nullptr));
    DecodeControl c{8100, 0, 1, -1}; // pos 0, single-slot T=1
    CHECK(zeCommandListAppendMemoryCopy(up, dCtrl, &c, sizeof(c), nullptr, 0,
                                        nullptr));
    double t0 = now_ns();
    CHECK(zeCommandQueueExecuteCommandLists(qq, 1, &reg, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    tRep.push_back(now_ns() - t0);
    CHECK(zeCommandListAppendMemoryCopy(up, hOut.data(), dX, H * 4, nullptr, 0,
                                        nullptr));
    if (run == 0) {
      double refmax = 0;
      for (float v : ref)
        refmax = std::max(refmax, (double)std::fabs(v));
      for (int j = 0; j < H; ++j) {
        double rel = std::fabs((double)hOut[j] - (double)ref[j]) /
                     (refmax > 0 ? refmax : 1);
        if (rel > worstRel)
          worstRel = rel;
        if (rel > 1e-4) {
          ok = false;
          std::fprintf(stderr, "ref mismatch j %d: got %g want %g\n", j,
                       hOut[j], ref[j]);
          break;
        }
      }
      hOut2 = hOut;
    } else {
      for (int j = 0; j < H; ++j) {
        uint32_t a, b;
        __builtin_memcpy(&a, &hOut[j], 4);
        __builtin_memcpy(&b, &hOut2[j], 4);
        if (a != b) {
          ok = false;
          std::fprintf(stderr, "reset mismatch j %d\n", j);
          break;
        }
      }
    }
  }
  double mR = med(tRep);
  std::printf("layer3real med %.2f us worst-rel %.2e %s\n", mR / 1e3, worstRel,
              ok ? "LAYER3REAL-OK" : "MISMATCH");
  char js[640];
  std::snprintf(js, sizeof js,
                "{\"device\":\"B60\",\"layer\":3,\"type\":\"full-attn "
                "(real weights)\",\"launches\":24,\"runs\":2,"
                "\"replay_us\":%.2f,\"worst_rel\":%.2e,\"ref_tol\":1e-4,"
                "\"ref\":\"certified SYCL dump step-0 slot 3 (in: slot 2)\","
                "\"reset_deterministic\":%s,\"layer3real_ok\":%s}",
                mR / 1e3, worstRel, ok ? "true" : "false",
                ok ? "true" : "false");
  if (argc > 14) {
    FILE *o = std::fopen(argv[14], "w");
    if (o) {
      std::fprintf(o, "%s\n", js);
      std::fclose(o);
    }
  } else {
    std::printf("%s\n", js);
  }
  return ok ? 0 : 1;
}
