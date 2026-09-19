// AInfer Elementwise Kernels Verification & Benchmark on Arc 140V (T4.6)
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

// CPU References
void cpu_rmsnorm(float *y, const float *x, const float *w, int dim, float eps, bool zero_centered) {
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) {
    ss += x[i] * x[i];
  }
  float inv_rms = 1.0f / std::sqrt(ss / (float)dim + eps);
  for (int i = 0; i < dim; ++i) {
    float scale = zero_centered ? (1.0f + w[i]) : w[i];
    y[i] = x[i] * inv_rms * scale;
  }
}

void cpu_silu_mul(float *z, const float *g, const float *u, int dim) {
  for (int i = 0; i < dim; ++i) {
    float silu_g = g[i] / (1.0f + std::exp(-g[i]));
    z[i] = silu_g * u[i];
  }
}

void cpu_residual_add(float *y, const float *x1, const float *x2, int dim) {
  for (int i = 0; i < dim; ++i) {
    y[i] = x1[i] + x2[i];
  }
}

void cpu_moe_accum(float *out, const float *exp_out, float weight, int dim) {
  for (int i = 0; i < dim; ++i) {
    out[i] += exp_out[i] * weight;
  }
}

void cpu_moe_shared(float *out, const float *shared_out, float gate_val, int dim) {
  float g = 1.0f / (1.0f + std::exp(-gate_val));
  for (int i = 0; i < dim; ++i) {
    out[i] += shared_out[i] * g;
  }
}

int cpu_argmax(const float *logits, int N) {
  float best_val = logits[0];
  int best_idx = 0;
  for (int i = 1; i < N; ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return best_idx;
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/kernels_258v/elementwise.spv";
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

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  bool all_pass = true;
  const int ITERS = 1000;

  // -------------------------------------------------------------
  // Test 1: rmsnorm_2048
  // -------------------------------------------------------------
  std::printf("--- Testing rmsnorm_2048 ---\n");
  ze_kernel_desc_t k_rmsnorm_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "rmsnorm_2048"};
  ze_kernel_handle_t k_rmsnorm;
  CHECK(zeKernelCreate(mod, &k_rmsnorm_desc, &k_rmsnorm));

  std::vector<float> h_x_2048(2048), h_w_2048(2048), h_y_ref_2048(2048), h_y_gpu_2048(2048);
  for (int i = 0; i < 2048; ++i) {
    h_x_2048[i] = fdist(rng);
    h_w_2048[i] = fdist(rng) * 0.1f; // zero-centered weights
  }
  cpu_rmsnorm(h_y_ref_2048.data(), h_x_2048.data(), h_w_2048.data(), 2048, 1e-6f, true);

  float *d_y_2048, *d_x_2048, *d_w_2048;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_y_2048));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_x_2048));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_w_2048));

  CHECK(zeCommandListAppendMemoryCopy(list, d_x_2048, h_x_2048.data(), 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_w_2048, h_w_2048.data(), 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_rmsnorm, 0, sizeof(void *), &d_y_2048));
  CHECK(zeKernelSetArgumentValue(k_rmsnorm, 1, sizeof(void *), &d_x_2048));
  CHECK(zeKernelSetArgumentValue(k_rmsnorm, 2, sizeof(void *), &d_w_2048));
  CHECK(zeKernelSetGroupSize(k_rmsnorm, 256, 1, 1));
  ze_group_count_t gcnt_rmsnorm{1, 1, 1};

  CHECK(zeCommandListAppendLaunchKernel(list, k_rmsnorm, &gcnt_rmsnorm, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, h_y_gpu_2048.data(), d_y_2048, 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_norm2048 = 0.0f;
  for (int i = 0; i < 2048; ++i) {
    float diff = std::fabs(h_y_gpu_2048[i] - h_y_ref_2048[i]);
    if (diff > max_diff_norm2048) max_diff_norm2048 = diff;
  }
  bool pass_norm2048 = (max_diff_norm2048 < 1e-4f);
  if (!pass_norm2048) all_pass = false;
  std::printf("  rmsnorm_2048: max_diff = %.2e [%s]\n", max_diff_norm2048, pass_norm2048 ? "PASS" : "FAIL");

  // Benchmark rmsnorm_2048
  CHECK(zeCommandListReset(list));
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, k_rmsnorm, &gcnt_rmsnorm, nullptr, 0, nullptr));
  }
  CHECK(zeCommandListClose(list));
  auto t0 = std::chrono::steady_clock::now();
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  auto t1 = std::chrono::steady_clock::now();
  double lat_norm2048 = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
  std::printf("  rmsnorm_2048 latency: %.2f us\n", lat_norm2048);

  // -------------------------------------------------------------
  // Test 2: rmsnorm_head_256 (16 heads for Q)
  // -------------------------------------------------------------
  std::printf("\n--- Testing rmsnorm_head_256 (16 heads) ---\n");
  ze_kernel_desc_t k_norm256_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "rmsnorm_head_256"};
  ze_kernel_handle_t k_norm256;
  CHECK(zeKernelCreate(mod, &k_norm256_desc, &k_norm256));

  int num_q_heads = 16;
  int total_q_elems = num_q_heads * 256;
  std::vector<float> h_x_q(total_q_elems), h_w_q(256), h_y_ref_q(total_q_elems), h_y_gpu_q(total_q_elems);
  for (int i = 0; i < total_q_elems; ++i) h_x_q[i] = fdist(rng);
  for (int i = 0; i < 256; ++i) h_w_q[i] = fdist(rng) * 0.1f;
  for (int h = 0; h < num_q_heads; ++h) {
    cpu_rmsnorm(h_y_ref_q.data() + h * 256, h_x_q.data() + h * 256, h_w_q.data(), 256, 1e-6f, true);
  }

  float *d_y_q, *d_x_q, *d_w_q;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, total_q_elems * sizeof(float), 64, dev, (void **)&d_y_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, total_q_elems * sizeof(float), 64, dev, (void **)&d_x_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 256 * sizeof(float), 64, dev, (void **)&d_w_q));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_x_q, h_x_q.data(), total_q_elems * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_w_q, h_w_q.data(), 256 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_norm256, 0, sizeof(void *), &d_y_q));
  CHECK(zeKernelSetArgumentValue(k_norm256, 1, sizeof(void *), &d_x_q));
  CHECK(zeKernelSetArgumentValue(k_norm256, 2, sizeof(void *), &d_w_q));
  CHECK(zeKernelSetArgumentValue(k_norm256, 3, sizeof(int), &num_q_heads));
  CHECK(zeKernelSetGroupSize(k_norm256, 64, 1, 1));
  ze_group_count_t gcnt_norm256{(uint32_t)num_q_heads, 1, 1};

  CHECK(zeCommandListAppendLaunchKernel(list, k_norm256, &gcnt_norm256, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, h_y_gpu_q.data(), d_y_q, total_q_elems * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_norm256 = 0.0f;
  for (int i = 0; i < total_q_elems; ++i) {
    float diff = std::fabs(h_y_gpu_q[i] - h_y_ref_q[i]);
    if (diff > max_diff_norm256) max_diff_norm256 = diff;
  }
  bool pass_norm256 = (max_diff_norm256 < 1e-4f);
  if (!pass_norm256) all_pass = false;
  std::printf("  rmsnorm_head_256: max_diff = %.2e [%s]\n", max_diff_norm256, pass_norm256 ? "PASS" : "FAIL");

  // -------------------------------------------------------------
  // Test 3: silu_mul_512
  // -------------------------------------------------------------
  std::printf("\n--- Testing silu_mul_512 ---\n");
  ze_kernel_desc_t k_silu_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "silu_mul_512"};
  ze_kernel_handle_t k_silu;
  CHECK(zeKernelCreate(mod, &k_silu_desc, &k_silu));

  std::vector<float> h_g_512(512), h_u_512(512), h_z_ref_512(512), h_z_gpu_512(512);
  for (int i = 0; i < 512; ++i) {
    h_g_512[i] = fdist(rng);
    h_u_512[i] = fdist(rng);
  }
  cpu_silu_mul(h_z_ref_512.data(), h_g_512.data(), h_u_512.data(), 512);

  float *d_z_512, *d_g_512, *d_u_512;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 512 * sizeof(float), 64, dev, (void **)&d_z_512));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 512 * sizeof(float), 64, dev, (void **)&d_g_512));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 512 * sizeof(float), 64, dev, (void **)&d_u_512));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_g_512, h_g_512.data(), 512 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_u_512, h_u_512.data(), 512 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_silu, 0, sizeof(void *), &d_z_512));
  CHECK(zeKernelSetArgumentValue(k_silu, 1, sizeof(void *), &d_g_512));
  CHECK(zeKernelSetArgumentValue(k_silu, 2, sizeof(void *), &d_u_512));
  CHECK(zeKernelSetGroupSize(k_silu, 256, 1, 1));
  ze_group_count_t gcnt_silu{2, 1, 1};

  CHECK(zeCommandListAppendLaunchKernel(list, k_silu, &gcnt_silu, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, h_z_gpu_512.data(), d_z_512, 512 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_silu = 0.0f;
  for (int i = 0; i < 512; ++i) {
    float diff = std::fabs(h_z_gpu_512[i] - h_z_ref_512[i]);
    if (diff > max_diff_silu) max_diff_silu = diff;
  }
  bool pass_silu = (max_diff_silu < 1e-5f);
  if (!pass_silu) all_pass = false;
  std::printf("  silu_mul_512: max_diff = %.2e [%s]\n", max_diff_silu, pass_silu ? "PASS" : "FAIL");

  // -------------------------------------------------------------
  // Test 4: residual_add_2048
  // -------------------------------------------------------------
  std::printf("\n--- Testing residual_add_2048 ---\n");
  ze_kernel_desc_t k_res_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "residual_add_2048"};
  ze_kernel_handle_t k_res;
  CHECK(zeKernelCreate(mod, &k_res_desc, &k_res));

  std::vector<float> h_x1(2048), h_x2(2048), h_y_res_ref(2048), h_y_res_gpu(2048);
  for (int i = 0; i < 2048; ++i) {
    h_x1[i] = fdist(rng);
    h_x2[i] = fdist(rng);
  }
  cpu_residual_add(h_y_res_ref.data(), h_x1.data(), h_x2.data(), 2048);

  float *d_y_res, *d_x1, *d_x2;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_y_res));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_x1));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2048 * sizeof(float), 64, dev, (void **)&d_x2));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_x1, h_x1.data(), 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_x2, h_x2.data(), 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_res, 0, sizeof(void *), &d_y_res));
  CHECK(zeKernelSetArgumentValue(k_res, 1, sizeof(void *), &d_x1));
  CHECK(zeKernelSetArgumentValue(k_res, 2, sizeof(void *), &d_x2));
  CHECK(zeKernelSetGroupSize(k_res, 256, 1, 1));
  ze_group_count_t gcnt_res{8, 1, 1};

  CHECK(zeCommandListAppendLaunchKernel(list, k_res, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, h_y_res_gpu.data(), d_y_res, 2048 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_res = 0.0f;
  for (int i = 0; i < 2048; ++i) {
    float diff = std::fabs(h_y_res_gpu[i] - h_y_res_ref[i]);
    if (diff > max_diff_res) max_diff_res = diff;
  }
  bool pass_res = (max_diff_res == 0.0f);
  if (!pass_res) all_pass = false;
  std::printf("  residual_add_2048: max_diff = %.2e [%s]\n", max_diff_res, pass_res ? "PASS" : "FAIL");

  // -------------------------------------------------------------
  // Test 5: argmax_stage1 & argmax_stage2 (Vocab = 248320)
  // -------------------------------------------------------------
  std::printf("\n--- Testing argmax_stage1 & argmax_stage2 (Vocab = 248,320) ---\n");
  ze_kernel_desc_t k_arg1_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "argmax_stage1"};
  ze_kernel_handle_t k_arg1;
  CHECK(zeKernelCreate(mod, &k_arg1_desc, &k_arg1));

  ze_kernel_desc_t k_arg2_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "argmax_stage2"};
  ze_kernel_handle_t k_arg2;
  CHECK(zeKernelCreate(mod, &k_arg2_desc, &k_arg2));

  const uint32_t VOCAB = 248320;
  const uint32_t NUM_GROUPS = (VOCAB + 255) / 256; // 970 groups
  std::vector<float> h_logits(VOCAB);
  for (uint32_t i = 0; i < VOCAB; ++i) {
    h_logits[i] = fdist(rng);
  }
  // Plant an unmistakable max token ID
  const uint32_t TARGET_TOKEN = 123456;
  h_logits[TARGET_TOKEN] = 100.0f;

  int cpu_tok = cpu_argmax(h_logits.data(), VOCAB);

  float *d_logits, *d_group_val;
  uint32_t *d_group_idx, *d_out_tok;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, VOCAB * sizeof(float), 64, dev, (void **)&d_logits));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_GROUPS * sizeof(float), 64, dev, (void **)&d_group_val));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_GROUPS * sizeof(uint32_t), 64, dev, (void **)&d_group_idx));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sizeof(uint32_t), 64, dev, (void **)&d_out_tok));

  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_logits, h_logits.data(), VOCAB * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  uint32_t vocab_arg = VOCAB;
  CHECK(zeKernelSetArgumentValue(k_arg1, 0, sizeof(void *), &d_logits));
  CHECK(zeKernelSetArgumentValue(k_arg1, 1, sizeof(void *), &d_group_val));
  CHECK(zeKernelSetArgumentValue(k_arg1, 2, sizeof(void *), &d_group_idx));
  CHECK(zeKernelSetArgumentValue(k_arg1, 3, sizeof(uint32_t), &vocab_arg));
  CHECK(zeKernelSetGroupSize(k_arg1, 256, 1, 1));
  ze_group_count_t gcnt_arg1{NUM_GROUPS, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list, k_arg1, &gcnt_arg1, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  uint32_t ngroups_arg = NUM_GROUPS;
  CHECK(zeKernelSetArgumentValue(k_arg2, 0, sizeof(void *), &d_group_val));
  CHECK(zeKernelSetArgumentValue(k_arg2, 1, sizeof(void *), &d_group_idx));
  CHECK(zeKernelSetArgumentValue(k_arg2, 2, sizeof(void *), &d_out_tok));
  CHECK(zeKernelSetArgumentValue(k_arg2, 3, sizeof(uint32_t), &ngroups_arg));
  CHECK(zeKernelSetGroupSize(k_arg2, 256, 1, 1));
  ze_group_count_t gcnt_arg2{1, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list, k_arg2, &gcnt_arg2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  uint32_t gpu_tok = 0;
  CHECK(zeCommandListAppendMemoryCopy(list, &gpu_tok, d_out_tok, sizeof(uint32_t), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  bool pass_argmax = (gpu_tok == TARGET_TOKEN && gpu_tok == (uint32_t)cpu_tok);
  if (!pass_argmax) all_pass = false;
  std::printf("  argmax: CPU=%d, GPU=%u (expected %u) [%s]\n", cpu_tok, gpu_tok, TARGET_TOKEN,
              pass_argmax ? "PASS" : "FAIL");

  // Benchmark argmax
  CHECK(zeCommandListReset(list));
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, k_arg1, &gcnt_arg1, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_arg2, &gcnt_arg2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  }
  CHECK(zeCommandListClose(list));
  t0 = std::chrono::steady_clock::now();
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  t1 = std::chrono::steady_clock::now();
  double lat_argmax = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
  std::printf("  argmax total latency (2 stages): %.2f us\n", lat_argmax);

  // Write JSON report
  std::ofstream rpt("tools/kernels_258v/report_elementwise.json");
  rpt << "{\n";
  rpt << "  \"task\": \"T4.6\",\n";
  rpt << "  \"device\": \"Arc 140V (Xe2)\",\n";
  rpt << "  \"all_passed\": " << (all_pass ? "true" : "false") << ",\n";
  rpt << "  \"rmsnorm_2048\": {\n";
  rpt << "    \"max_diff\": " << max_diff_norm2048 << ",\n";
  rpt << "    \"latency_us\": " << lat_norm2048 << ",\n";
  rpt << "    \"status\": \"" << (pass_norm2048 ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"rmsnorm_head_256\": {\n";
  rpt << "    \"max_diff\": " << max_diff_norm256 << ",\n";
  rpt << "    \"status\": \"" << (pass_norm256 ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"silu_mul_512\": {\n";
  rpt << "    \"max_diff\": " << max_diff_silu << ",\n";
  rpt << "    \"status\": \"" << (pass_silu ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"residual_add_2048\": {\n";
  rpt << "    \"max_diff\": " << max_diff_res << ",\n";
  rpt << "    \"status\": \"" << (pass_res ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"argmax_vocab_248320\": {\n";
  rpt << "    \"cpu_token\": " << cpu_tok << ",\n";
  rpt << "    \"gpu_token\": " << gpu_tok << ",\n";
  rpt << "    \"latency_us\": " << lat_argmax << ",\n";
  rpt << "    \"status\": \"" << (pass_argmax ? "PASS" : "FAIL") << "\"\n";
  rpt << "  }\n";
  rpt << "}\n";
  rpt.close();

  std::printf("\nReport written to tools/kernels_258v/report_elementwise.json\n");
  std::printf("Overall Status: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");

  // Clean up
  zeMemFree(ctx, d_logits);
  zeMemFree(ctx, d_group_val);
  zeMemFree(ctx, d_group_idx);
  zeMemFree(ctx, d_out_tok);
  zeMemFree(ctx, d_y_res);
  zeMemFree(ctx, d_x1);
  zeMemFree(ctx, d_x2);
  zeMemFree(ctx, d_z_512);
  zeMemFree(ctx, d_g_512);
  zeMemFree(ctx, d_u_512);
  zeMemFree(ctx, d_y_q);
  zeMemFree(ctx, d_x_q);
  zeMemFree(ctx, d_w_q);
  zeMemFree(ctx, d_y_2048);
  zeMemFree(ctx, d_x_2048);
  zeMemFree(ctx, d_w_2048);

  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeKernelDestroy(k_rmsnorm);
  zeKernelDestroy(k_norm256);
  zeKernelDestroy(k_silu);
  zeKernelDestroy(k_res);
  zeKernelDestroy(k_arg1);
  zeKernelDestroy(k_arg2);
  zeModuleDestroy(mod);
  zeContextDestroy(ctx);

  return all_pass ? 0 : 1;
}
