// AInfer Full-Attention (GQA) & RoPE Verification & Benchmark on Arc 140V (T4.5)
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
const float ROPE_THETA = 10000000.0f;
const float ATTN_SCALE = 0.0625f;

static inline uint16_t float_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static inline float bf16_to_float(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// CPU References
void cpu_rope_and_kv_append(float *q, float *k, const float *v,
                            uint16_t *k_cache, uint16_t *v_cache,
                            uint32_t pos, uint32_t max_ctx) {
  // Q RoPE
  for (int h = 0; h < NUM_Q_HEADS; ++h) {
    float *q_head = q + h * HEAD_DIM;
    for (int i = 0; i < ROTARY_HALF; ++i) {
      float inv_freq = std::pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
      float theta = (float)pos * inv_freq;
      float cos_t = std::cos(theta);
      float sin_t = std::sin(theta);

      float x0 = q_head[i];
      float x1 = q_head[i + ROTARY_HALF];
      q_head[i]               = x0 * cos_t - x1 * sin_t;
      q_head[i + ROTARY_HALF] = x0 * sin_t + x1 * cos_t;
    }
  }

  // K RoPE and Cache Write
  for (int h = 0; h < NUM_KV_HEADS; ++h) {
    float *k_head = k + h * HEAD_DIM;
    for (int i = 0; i < ROTARY_HALF; ++i) {
      float inv_freq = std::pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
      float theta = (float)pos * inv_freq;
      float cos_t = std::cos(theta);
      float sin_t = std::sin(theta);

      float x0 = k_head[i];
      float x1 = k_head[i + ROTARY_HALF];
      k_head[i]               = x0 * cos_t - x1 * sin_t;
      k_head[i + ROTARY_HALF] = x0 * sin_t + x1 * cos_t;
    }

    uint16_t *k_slot = k_cache + (h * max_ctx + pos) * HEAD_DIM;
    uint16_t *v_slot = v_cache + (h * max_ctx + pos) * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; ++d) {
      k_slot[d] = float_to_bf16(k_head[d]);
      v_slot[d] = float_to_bf16(v[h * HEAD_DIM + d]);
    }
  }
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
  const char *spv_path = (argc > 1) ? argv[1] : "tools/kernels_258v/attention.spv";
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
  ze_command_list_handle_t list;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  ze_kernel_desc_t k_rope_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "rope_and_kv_append_bf16"};
  ze_kernel_handle_t k_rope;
  CHECK(zeKernelCreate(mod, &k_rope_desc, &k_rope));

  ze_kernel_desc_t k_attn_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "gqa_attn_decode_bf16"};
  ze_kernel_handle_t k_attn;
  CHECK(zeKernelCreate(mod, &k_attn_desc, &k_attn));

  const uint32_t MAX_CTX = 4096;
  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

  // Device buffers
  float *d_q, *d_k, *d_v, *d_gate, *d_out;
  uint16_t *d_k_cache, *d_v_cache;

  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_KV_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_k));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_KV_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_v));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_gate));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_out));

  size_t kv_cache_bytes = (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t);
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, kv_cache_bytes, 64, dev, (void **)&d_k_cache));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, kv_cache_bytes, 64, dev, (void **)&d_v_cache));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

  // Host buffers
  std::vector<float> h_q(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_k(NUM_KV_HEADS * HEAD_DIM);
  std::vector<float> h_v(NUM_KV_HEADS * HEAD_DIM);
  std::vector<float> h_gate(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_out_ref(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_out_gpu(NUM_Q_HEADS * HEAD_DIM);

  std::vector<uint16_t> h_k_cache_ref(NUM_KV_HEADS * MAX_CTX * HEAD_DIM, 0);
  std::vector<uint16_t> h_v_cache_ref(NUM_KV_HEADS * MAX_CTX * HEAD_DIM, 0);

  std::printf("--- Testing Full Attention GQA & RoPE (32 sequential steps) ---\n");
  float max_diff_attn = 0.0f;
  bool all_steps_pass = true;

  ze_group_count_t gcnt_rope{1, 1, 1};
  ze_group_count_t gcnt_attn{(uint32_t)NUM_Q_HEADS, 1, 1};

  for (uint32_t step = 0; step < 32; ++step) {
    for (auto &x : h_q) x = fdist(rng);
    for (auto &x : h_k) x = fdist(rng);
    for (auto &x : h_v) x = fdist(rng);
    for (auto &x : h_gate) x = fdist(rng);

    // CPU step
    std::vector<float> h_q_cpu = h_q;
    std::vector<float> h_k_cpu = h_k;
    cpu_rope_and_kv_append(h_q_cpu.data(), h_k_cpu.data(), h_v.data(),
                           h_k_cache_ref.data(), h_v_cache_ref.data(), step, MAX_CTX);
    cpu_gqa_attn_decode(h_out_ref.data(), h_q_cpu.data(), h_gate.data(),
                        h_k_cache_ref.data(), h_v_cache_ref.data(), step, MAX_CTX);

    // GPU step
    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendMemoryCopy(list, d_q, h_q.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_k, h_k.data(), NUM_KV_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_v, h_v.data(), NUM_KV_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_gate, h_gate.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    uint32_t cur_pos = step;
    uint32_t max_ctx_arg = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k_rope, 0, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_rope, 1, sizeof(void *), &d_k));
    CHECK(zeKernelSetArgumentValue(k_rope, 2, sizeof(void *), &d_v));
    CHECK(zeKernelSetArgumentValue(k_rope, 3, sizeof(void *), &d_k_cache));
    CHECK(zeKernelSetArgumentValue(k_rope, 4, sizeof(void *), &d_v_cache));
    CHECK(zeKernelSetArgumentValue(k_rope, 5, sizeof(uint32_t), &cur_pos));
    CHECK(zeKernelSetArgumentValue(k_rope, 6, sizeof(uint32_t), &max_ctx_arg));
    CHECK(zeKernelSetGroupSize(k_rope, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list, k_rope, &gcnt_rope, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    CHECK(zeKernelSetArgumentValue(k_attn, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k_attn, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_attn, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k_attn, 3, sizeof(void *), &d_k_cache));
    CHECK(zeKernelSetArgumentValue(k_attn, 4, sizeof(void *), &d_v_cache));
    CHECK(zeKernelSetArgumentValue(k_attn, 5, sizeof(uint32_t), &cur_pos));
    CHECK(zeKernelSetArgumentValue(k_attn, 6, sizeof(uint32_t), &max_ctx_arg));
    CHECK(zeKernelSetGroupSize(k_attn, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list, k_attn, &gcnt_attn, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    CHECK(zeCommandListAppendMemoryCopy(list, h_out_gpu.data(), d_out, NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));

    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    float cur_diff = 0.0f;
    for (int i = 0; i < NUM_Q_HEADS * HEAD_DIM; ++i) {
      float diff = std::fabs(h_out_gpu[i] - h_out_ref[i]);
      if (diff > cur_diff) cur_diff = diff;
    }
    if (cur_diff > max_diff_attn) max_diff_attn = cur_diff;
    bool pass = (cur_diff < 5e-4f);
    if (!pass) all_steps_pass = false;
    std::printf("  Step %2d (ctx_len=%2d): max_diff = %.2e [%s]\n", step, step + 1,
                cur_diff, pass ? "PASS" : "FAIL");
  }

  // Benchmark at different context lengths
  std::printf("\n--- Benchmarking Attention Decode at Various Context Lengths ---\n");
  std::vector<uint32_t> bench_ctxs = {128, 512, 1024, 2048, 4096};
  std::vector<double> latencies;

  const int BENCH_ITERS = 500;
  for (uint32_t ctx_len : bench_ctxs) {
    uint32_t pos = ctx_len - 1;
    uint32_t max_ctx_arg = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k_attn, 5, sizeof(uint32_t), &pos));

    CHECK(zeCommandListReset(list));
    for (int i = 0; i < BENCH_ITERS; ++i) {
      CHECK(zeCommandListAppendLaunchKernel(list, k_attn, &gcnt_attn, nullptr, 0, nullptr));
      CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    }
    CHECK(zeCommandListClose(list));

    auto t0 = std::chrono::steady_clock::now();
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    auto t1 = std::chrono::steady_clock::now();

    double lat = std::chrono::duration<double, std::micro>(t1 - t0).count() / BENCH_ITERS;
    latencies.push_back(lat);
    std::printf("  Ctx = %4u tokens: %.2f us per layer (10 layers = %.2f us)\n",
                ctx_len, lat, lat * 10);
  }

  // Write report
  std::ofstream rpt("tools/kernels_258v/report_attention.json");
  rpt << "{\n";
  rpt << "  \"task\": \"T4.5\",\n";
  rpt << "  \"device\": \"Arc 140V (Xe2)\",\n";
  rpt << "  \"all_passed\": " << (all_steps_pass ? "true" : "false") << ",\n";
  rpt << "  \"max_attn_output_diff\": " << max_diff_attn << ",\n";
  rpt << "  \"latencies_us\": {\n";
  for (size_t i = 0; i < bench_ctxs.size(); ++i) {
    rpt << "    \"ctx_" << bench_ctxs[i] << "\": " << latencies[i]
        << (i + 1 < bench_ctxs.size() ? ",\n" : "\n");
  }
  rpt << "  },\n";
  rpt << "  \"status\": \"" << (all_steps_pass ? "PASS" : "FAIL") << "\"\n";
  rpt << "}\n";
  rpt.close();

  std::printf("\nReport written to tools/kernels_258v/report_attention.json\n");

  // Cleanup
  zeMemFree(ctx, d_q);
  zeMemFree(ctx, d_k);
  zeMemFree(ctx, d_v);
  zeMemFree(ctx, d_gate);
  zeMemFree(ctx, d_out);
  zeMemFree(ctx, d_k_cache);
  zeMemFree(ctx, d_v_cache);

  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeKernelDestroy(k_rope);
  zeKernelDestroy(k_attn);
  zeModuleDestroy(mod);
  zeContextDestroy(ctx);

  return all_steps_pass ? 0 : 1;
}
