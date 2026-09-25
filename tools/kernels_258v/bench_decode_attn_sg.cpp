#include <level_zero/ze_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

const int HEAD_DIM = 256;
const int ROTARY_DIM = 64;
const int ROTARY_HALF = 32;
const int NUM_Q_HEADS = 16;
const int NUM_KV_HEADS = 2;
const int GQA_GROUP_SIZE = 8;
const float ATTN_SCALE = 0.0625f;

static inline float bf16_to_float(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static inline uint16_t float_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

void cpu_gqa_attn_decode(float *out, const float *q, const float *gate,
                         const uint16_t *k_cache, const uint16_t *v_cache,
                         uint32_t pos, uint32_t max_ctx) {
  uint32_t total_tokens = pos + 1;
  std::vector<float> scores(total_tokens);

  for (int qh = 0; qh < NUM_Q_HEADS; ++qh) {
    int kv_h = qh / GQA_GROUP_SIZE;
    const float *q_head = q + qh * HEAD_DIM;

    float max_s = -1e30f;
    for (uint32_t t = 0; t < total_tokens; ++t) {
      const uint16_t *k_slot = k_cache + (kv_h * max_ctx + t) * HEAD_DIM;
      float dot = 0.0f;
      for (int d = 0; d < HEAD_DIM; ++d) {
        dot += q_head[d] * bf16_to_float(k_slot[d]);
      }
      scores[t] = dot * ATTN_SCALE;
      if (scores[t] > max_s) max_s = scores[t];
    }

    float sum_exp = 0.0f;
    for (uint32_t t = 0; t < total_tokens; ++t) {
      scores[t] = std::exp(scores[t] - max_s);
      sum_exp += scores[t];
    }

    for (int d = 0; d < HEAD_DIM; ++d) {
      float acc = 0.0f;
      for (uint32_t t = 0; t < total_tokens; ++t) {
        const uint16_t *v_slot = v_cache + (kv_h * max_ctx + t) * HEAD_DIM;
        acc += scores[t] * bf16_to_float(v_slot[d]);
      }
      float attn_val = acc / sum_exp;
      float g_val = gate[qh * HEAD_DIM + d];
      float sig_g = 1.0f / (1.0f + std::exp(-g_val));
      out[qh * HEAD_DIM + d] = attn_val * sig_g;
    }
  }
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/kernels_258v/attn_decode_sg.spv";
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
        std::printf("Device selected: %s (devId=0x%04x)\n", props.name, props.deviceId);
        break;
      }
    }
    if (dev) break;
  }
  if (!dev) return 1;

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
  ze_command_list_handle_t list;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  auto get_kernel = [&](const char *name) -> ze_kernel_handle_t {
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t kh;
    if (zeKernelCreate(mod, &kd, &kh) != ZE_RESULT_SUCCESS) return nullptr;
    return kh;
  };

  ze_kernel_handle_t k_base = get_kernel("gqa_attn_decode_baseline");
  ze_kernel_handle_t k_u4   = get_kernel("gqa_attn_decode_sg_u4");
  ze_kernel_handle_t k_u8   = get_kernel("gqa_attn_decode_sg_u8");
  ze_kernel_handle_t k_u16  = get_kernel("gqa_attn_decode_sg_u16");

  if (!k_base || !k_u4 || !k_u8 || !k_u16) {
    std::fprintf(stderr, "Failed to load kernels: base=%p, u4=%p, u8=%p, u16=%p\n",
                 (void*)k_base, (void*)k_u4, (void*)k_u8, (void*)k_u16);
    return 1;
  }

  const uint32_t MAX_CTX = 8192;
  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

  float *d_q, *d_gate, *d_out;
  uint16_t *d_k_cache, *d_v_cache;

  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_gate));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_out));

  size_t kv_cache_bytes = (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t);
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, kv_cache_bytes, 64, dev, (void **)&d_k_cache));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, kv_cache_bytes, 64, dev, (void **)&d_v_cache));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

  std::vector<float> h_q(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_gate(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_out_ref(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_out_gpu(NUM_Q_HEADS * HEAD_DIM);
  std::vector<uint16_t> h_k_cache(NUM_KV_HEADS * MAX_CTX * HEAD_DIM);
  std::vector<uint16_t> h_v_cache(NUM_KV_HEADS * MAX_CTX * HEAD_DIM);

  for (auto &x : h_q) x = fdist(rng);
  for (auto &x : h_gate) x = fdist(rng);
  for (auto &x : h_k_cache) x = float_to_bf16(fdist(rng));
  for (auto &x : h_v_cache) x = float_to_bf16(fdist(rng));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_q, h_q.data(), h_q.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_gate, h_gate.data(), h_gate.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_k_cache, h_k_cache.data(), h_k_cache.size() * sizeof(uint16_t), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_v_cache, h_v_cache.data(), h_v_cache.size() * sizeof(uint16_t), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  ze_group_count_t gcnt_attn{(uint32_t)NUM_Q_HEADS, 1, 1};

  auto test_kernel = [&](ze_kernel_handle_t k, const char *kname, uint32_t pos) -> float {
    uint32_t max_c = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_k_cache));
    CHECK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_v_cache));
    CHECK(zeKernelSetArgumentValue(k, 5, sizeof(uint32_t), &pos));
    CHECK(zeKernelSetArgumentValue(k, 6, sizeof(uint32_t), &max_c));
    CHECK(zeKernelSetGroupSize(k, 256, 1, 1));

    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendLaunchKernel(list, k, &gcnt_attn, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, h_out_gpu.data(), d_out, NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));

    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    cpu_gqa_attn_decode(h_out_ref.data(), h_q.data(), h_gate.data(), h_k_cache.data(), h_v_cache.data(), pos, MAX_CTX);

    float max_d = 0.0f;
    for (int i = 0; i < NUM_Q_HEADS * HEAD_DIM; ++i) {
      float d = std::fabs(h_out_gpu[i] - h_out_ref[i]);
      if (d > max_d) max_d = d;
    }
    return max_d;
  };

  std::printf("\n=== Numerical Parity Verification against CPU Reference ===\n");
  for (uint32_t test_pos : {0u, 7u, 15u, 31u, 63u, 127u, 255u}) {
    float diff_base = test_kernel(k_base, "Baseline", test_pos);
    float diff_u4   = test_kernel(k_u4, "Subgroup U4", test_pos);
    float diff_u8   = test_kernel(k_u8, "Subgroup U8", test_pos);
    float diff_u16  = test_kernel(k_u16, "Subgroup U16", test_pos);
    std::printf("  Pos = %3u (ctx=%3u) | Base: %.2e | U4: %.2e [%s] | U8: %.2e [%s] | U16: %.2e [%s]\n",
                test_pos, test_pos + 1, diff_base,
                diff_u4, (diff_u4 < 5e-4f ? "PASS" : "FAIL"),
                diff_u8, (diff_u8 < 5e-4f ? "PASS" : "FAIL"),
                diff_u16, (diff_u16 < 5e-4f ? "PASS" : "FAIL"));
  }

  auto time_kernel = [&](ze_kernel_handle_t k, uint32_t pos, int iters) -> double {
    uint32_t max_c = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_k_cache));
    CHECK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_v_cache));
    CHECK(zeKernelSetArgumentValue(k, 5, sizeof(uint32_t), &pos));
    CHECK(zeKernelSetArgumentValue(k, 6, sizeof(uint32_t), &max_c));
    CHECK(zeKernelSetGroupSize(k, 256, 1, 1));

    CHECK(zeCommandListReset(list));
    for (int i = 0; i < iters; ++i) {
      CHECK(zeCommandListAppendLaunchKernel(list, k, &gcnt_attn, nullptr, 0, nullptr));
      CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    }
    CHECK(zeCommandListClose(list));

    // Warmup
    for (int w = 0; w < 3; ++w) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    auto t0 = std::chrono::steady_clock::now();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  };

  std::printf("\n===============================================================================================================================\n");
  std::printf("%-10s | %-16s | %-16s | %-16s | %-16s | %-10s | %-14s\n",
              "Context", "Baseline (1 lyr)", "SG U4 (1 lyr)", "SG U8 (1 lyr)", "SG U16 (1 lyr)", "Speedup", "10-Layer SG Time");
  std::printf("===============================================================================================================================\n");

  std::vector<uint32_t> bench_ctxs = {128, 512, 1024, 2048, 4096, 6720};
  for (uint32_t ctx_len : bench_ctxs) {
    uint32_t pos = ctx_len - 1;
    int iters = (ctx_len >= 4096) ? 50 : 200;

    double lat_base = time_kernel(k_base, pos, iters);
    double lat_u4   = time_kernel(k_u4, pos, iters);
    double lat_u8   = time_kernel(k_u8, pos, iters);
    double lat_u16  = time_kernel(k_u16, pos, iters);

    double best_sg = std::min({lat_u4, lat_u8, lat_u16});
    double speedup = lat_base / best_sg;
    double time_10lyr_ms = best_sg * 10.0 / 1000.0;

    std::printf("%-10u | %10.2f us     | %10.2f us     | %10.2f us     | %10.2f us     | %7.2fx   | %8.2f ms\n",
                ctx_len, lat_base, lat_u4, lat_u8, lat_u16, speedup, time_10lyr_ms);
  }
  std::printf("===============================================================================================================================\n");

  // Cleanup
  zeMemFree(ctx, d_q);
  zeMemFree(ctx, d_gate);
  zeMemFree(ctx, d_out);
  zeMemFree(ctx, d_k_cache);
  zeMemFree(ctx, d_v_cache);

  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeKernelDestroy(k_base);
  zeKernelDestroy(k_u4);
  zeKernelDestroy(k_u8);
  zeModuleDestroy(mod);
  zeContextDestroy(ctx);

  return 0;
}
