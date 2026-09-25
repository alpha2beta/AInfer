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
const int NUM_Q_HEADS = 16;
const int NUM_KV_HEADS = 2;
const int GQA_GROUP_SIZE = 8;
const float ATTN_SCALE = 0.0625f;
const uint32_t MAX_CTX = 8192;

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

// CPU reference for causal attention batch
void cpu_attn_batch(std::vector<float> &out, const std::vector<float> &q,
                    const std::vector<float> &gate,
                    const std::vector<uint16_t> &kc,
                    const std::vector<uint16_t> &vc, uint32_t base_pos,
                    int B) {
  for (int b = 0; b < B; ++b) {
    uint32_t total = base_pos + (uint32_t)b + 1;
    for (int qh = 0; qh < NUM_Q_HEADS; ++qh) {
      int kv_h = qh / GQA_GROUP_SIZE;
      const float *q_head = &q[(size_t)b * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM];
      float max_s = -1e30f;
      std::vector<float> scores(total);
      for (uint32_t t = 0; t < total; ++t) {
        const uint16_t *k_slot = &kc[((size_t)kv_h * MAX_CTX + t) * HEAD_DIM];
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) dot += q_head[d] * bf16_to_float(k_slot[d]);
        scores[t] = dot * ATTN_SCALE;
        if (scores[t] > max_s) max_s = scores[t];
      }
      float sum = 0.0f;
      for (uint32_t t = 0; t < total; ++t) {
        scores[t] = std::exp(scores[t] - max_s);
        sum += scores[t];
      }
      float *o_head = &out[(size_t)b * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM];
      const float *g_head = &gate[(size_t)b * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM];
      for (int d = 0; d < HEAD_DIM; ++d) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < total; ++t) {
          const uint16_t *v_slot = &vc[((size_t)kv_h * MAX_CTX + t) * HEAD_DIM];
          acc += scores[t] * bf16_to_float(v_slot[d]);
        }
        float g = 1.0f / (1.0f + std::exp(-g_head[d]));
        o_head[d] = acc / sum * g;
      }
    }
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char *spv_all = (argc > 1) ? argv[1] : "tools/kernels_258v/all_kernels.spv";
  const char *spv_flash = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  auto load_spv = [](const char *path) -> std::vector<uint8_t> {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> b(sz);
    f.read((char *)b.data(), sz);
    return b;
  };

  auto spv_all_b = load_spv(spv_all);
  auto spv_flash_b = load_spv(spv_flash);
  if (spv_all_b.empty() || spv_flash_b.empty()) {
    std::fprintf(stderr, "Cannot open SPIR-V files\n");
    return 1;
  }

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
      ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
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

  ze_module_desc_t md_all{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV, spv_all_b.size(), spv_all_b.data(), "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod_all;
  CHECK(zeModuleCreate(ctx, dev, &md_all, &mod_all, nullptr));

  ze_module_desc_t md_flash{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV, spv_flash_b.size(), spv_flash_b.data(), "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod_flash;
  CHECK(zeModuleCreate(ctx, dev, &md_flash, &mod_flash, nullptr));

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

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  auto get_k = [](ze_module_handle_t m, const char *name) -> ze_kernel_handle_t {
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t kh;
    if (zeKernelCreate(m, &kd, &kh) != ZE_RESULT_SUCCESS) return nullptr;
    return kh;
  };

  ze_kernel_handle_t k_v2    = get_k(mod_all, "gqa_attn_prefill_batch_v2");
  ze_kernel_handle_t k_flash = get_k(mod_flash, "flash_attn_prefill_b8_t16");

  if (!k_v2 || !k_flash) {
    std::fprintf(stderr, "Failed to load kernels: v2=%p, flash=%p\n", (void*)k_v2, (void*)k_flash);
    return 1;
  }

  const int B_MAX = 256;
  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  float *d_q, *d_gate, *d_out_v2, *d_out_flash;
  uint16_t *d_kc, *d_vc;
  int *d_ctrl;

  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_gate));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_out_v2));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_out_flash));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t), 64, dev, (void **)&d_kc));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t), 64, dev, (void **)&d_vc));
  CHECK(zeMemAllocShared(ctx, &dmem, &hmem, 16 * sizeof(int), 64, dev, (void **)&d_ctrl));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
  std::vector<float> h_q((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_gate((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);
  std::vector<uint16_t> h_kc((size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM);
  std::vector<uint16_t> h_vc((size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM);
  std::vector<float> h_out_v2((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_out_flash((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);

  for (auto &x : h_q) x = fdist(rng);
  for (auto &x : h_gate) x = fdist(rng) * 0.5f;
  for (auto &x : h_kc) x = float_to_bf16(fdist(rng));
  for (auto &x : h_vc) x = float_to_bf16(fdist(rng));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_q, h_q.data(), h_q.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_gate, h_gate.data(), h_gate.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_kc, h_kc.data(), h_kc.size() * sizeof(uint16_t), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_vc, h_vc.data(), h_vc.size() * sizeof(uint16_t), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  auto run_v2 = [&](float *d_out, int B, uint32_t base_pos) {
    d_ctrl[1] = (int)base_pos;
    uint32_t mc = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k_v2, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k_v2, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_v2, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k_v2, 3, sizeof(void *), &d_kc));
    CHECK(zeKernelSetArgumentValue(k_v2, 4, sizeof(void *), &d_vc));
    CHECK(zeKernelSetArgumentValue(k_v2, 5, sizeof(void *), &d_ctrl));
    CHECK(zeKernelSetArgumentValue(k_v2, 6, sizeof(uint32_t), &mc));
    CHECK(zeKernelSetArgumentValue(k_v2, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k_v2, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)(B * NUM_Q_HEADS), 1, 1};

    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendLaunchKernel(list, k_v2, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  };

  auto run_flash = [&](float *d_out, int B, uint32_t base_pos) {
    d_ctrl[1] = (int)base_pos;
    uint32_t mc = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k_flash, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k_flash, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_flash, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k_flash, 3, sizeof(void *), &d_kc));
    CHECK(zeKernelSetArgumentValue(k_flash, 4, sizeof(void *), &d_vc));
    CHECK(zeKernelSetArgumentValue(k_flash, 5, sizeof(void *), &d_ctrl));
    CHECK(zeKernelSetArgumentValue(k_flash, 6, sizeof(uint32_t), &mc));
    CHECK(zeKernelSetArgumentValue(k_flash, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k_flash, 256, 1, 1));
    uint32_t num_b_blks = (uint32_t)((B + 7) / 8);
    ze_group_count_t gc{(uint32_t)NUM_Q_HEADS, num_b_blks, 1};

    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendLaunchKernel(list, k_flash, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  };

  std::printf("\n=== Numerical Parity Verification against CPU Reference ===\n");
  for (int B_test : {1, 3, 7, 8, 13, 16, 27, 32, 35}) {
    for (uint32_t base_pos_test : {0u, 5u, 128u}) {
      run_v2(d_out_v2, B_test, base_pos_test);
      run_flash(d_out_flash, B_test, base_pos_test);

      CHECK(zeCommandListReset(list));
      CHECK(zeCommandListAppendMemoryCopy(list, h_out_v2.data(), d_out_v2, (size_t)B_test * NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
      CHECK(zeCommandListAppendMemoryCopy(list, h_out_flash.data(), d_out_flash, (size_t)B_test * NUM_Q_HEADS * HEAD_DIM * sizeof(float), nullptr, 0, nullptr));
      CHECK(zeCommandListClose(list));
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

      std::vector<float> h_ref((size_t)B_test * NUM_Q_HEADS * HEAD_DIM);
      cpu_attn_batch(h_ref, h_q, h_gate, h_kc, h_vc, base_pos_test, B_test);

      float max_diff_v2 = 0.0f, max_diff_flash = 0.0f;
      for (size_t i = 0; i < h_ref.size(); ++i) {
        max_diff_v2 = std::max(max_diff_v2, std::fabs(h_out_v2[i] - h_ref[i]));
        max_diff_flash = std::max(max_diff_flash, std::fabs(h_out_flash[i] - h_ref[i]));
      }
      bool pass = max_diff_flash < 1e-3f;
      std::printf("  B=%2d, base_pos=%4u | V2 diff: %.2e | FlashAttn diff: %.2e [%s]\n",
                  B_test, base_pos_test, max_diff_v2, max_diff_flash, pass ? "PASS" : "FAIL");
      if (!pass) {
        std::fprintf(stderr, "FAILED parity check at B=%d, base_pos=%u!\n", B_test, base_pos_test);
        return 1;
      }
    }
  }

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--parity-only") {
      std::printf("Parity check finished successfully. Exiting (--parity-only).\n");
      return 0;
    }
  }

  auto time_kernel = [&](bool is_flash, int B, uint32_t base_pos, int iters) -> double {
    d_ctrl[1] = (int)base_pos;
    uint32_t mc = MAX_CTX;
    ze_kernel_handle_t k = is_flash ? k_flash : k_v2;
    float *d_out = is_flash ? d_out_flash : d_out_v2;

    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_kc));
    CHECK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_vc));
    CHECK(zeKernelSetArgumentValue(k, 5, sizeof(void *), &d_ctrl));
    CHECK(zeKernelSetArgumentValue(k, 6, sizeof(uint32_t), &mc));
    CHECK(zeKernelSetArgumentValue(k, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k, 256, 1, 1));

    ze_group_count_t gc;
    if (is_flash) {
      uint32_t num_b_blks = (uint32_t)((B + 7) / 8);
      gc = {(uint32_t)NUM_Q_HEADS, num_b_blks, 1};
    } else {
      gc = {(uint32_t)(B * NUM_Q_HEADS), 1, 1};
    }

    CHECK(zeCommandListReset(list));
    for (int i = 0; i < iters; ++i) {
      CHECK(zeCommandListAppendLaunchKernel(list, k, &gc, nullptr, 0, nullptr));
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

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  std::printf("\n===================================================================================================\n");
  std::printf("%-6s | %-12s | %-16s | %-16s | %-10s | %-16s\n",
              "B", "Context (P)", "V2 Prefill (ms)", "FlashAttn (ms)", "Speedup", "10-Lyr Flash (ms)");
  std::printf("===================================================================================================\n");

  struct TestCase { int B; uint32_t base_pos; };
  std::vector<TestCase> cases = {
    {32,   0},
    {32, 512},
    {32, 2048},
    {32, 4096},
    {32, 6656},
    {128,   0},
    {128, 1024},
    {128, 4096},
    {256,   0},
    {256, 1024},
    {256, 2048},
    {256, 4096},
    {256, 6400}
  };

  for (auto &tc : cases) {
    int iters = (tc.base_pos >= 4096) ? 10 : 25;
    double t_v2    = time_kernel(false, tc.B, tc.base_pos, iters);
    double t_flash = time_kernel(true,  tc.B, tc.base_pos, iters);
    double speedup = t_v2 / t_flash;
    double lyr10_ms = t_flash * 10.0;

    std::printf("%-6d | %-12u | %14.3f ms | %14.3f ms | %7.2fx   | %13.2f ms\n",
                tc.B, tc.base_pos + tc.B, t_v2, t_flash, speedup, lyr10_ms);
  }
  std::printf("===================================================================================================\n");

  // Cleanup
  zeMemFree(ctx, d_q);
  zeMemFree(ctx, d_gate);
  zeMemFree(ctx, d_out_v2);
  zeMemFree(ctx, d_out_flash);
  zeMemFree(ctx, d_kc);
  zeMemFree(ctx, d_vc);
  zeMemFree(ctx, d_ctrl);

  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeKernelDestroy(k_v2);
  zeKernelDestroy(k_flash);
  zeModuleDestroy(mod_all);
  zeModuleDestroy(mod_flash);
  zeContextDestroy(ctx);

  return 0;
}
