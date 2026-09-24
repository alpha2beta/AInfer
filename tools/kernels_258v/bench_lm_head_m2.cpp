// bench_lm_head_m2.cpp — Shootout between Baseline and DPAS LM Head Argmax on Arc 140V (Xe2)

#include <level_zero/ze_api.h>

#include <algorithm>
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
                   __LINE__, #expr);                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static inline uint16_t float_to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return (uint16_t)((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

std::vector<uint8_t> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  size_t sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> b(sz);
  f.read((char *)b.data(), sz);
  return b;
}

int main(int argc, char **argv) {
  std::vector<uint8_t> spv_all = read_file("tools/kernels_258v/all_kernels.spv");
  std::vector<uint8_t> spv_opt = read_file("tools/kernels_258v/test_gemv_m2_lnl.spv");

  if (spv_all.empty() || spv_opt.empty()) {
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
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      CHECK(zeDeviceGetProperties(dv, &props));
      if (props.deviceId == 0x64a0 || props.deviceId == 0xe211) {
        dev = dv;
        drv = d;
        std::printf("Device: %s (devId=0x%04x)\n", props.name, props.deviceId);
        std::fflush(stdout);
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

  ze_module_desc_t mod_desc1{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV, spv_all.size(), spv_all.data(),
                             "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod_all;
  CHECK(zeModuleCreate(ctx, dev, &mod_desc1, &mod_all, nullptr));

  ze_module_desc_t mod_desc2{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
                             ZE_MODULE_FORMAT_IL_SPIRV, spv_opt.size(), spv_opt.data(),
                             "-cl-std=CL2.0", nullptr};
  ze_module_handle_t mod_opt;
  CHECK(zeModuleCreate(ctx, dev, &mod_desc2, &mod_opt, nullptr));

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

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  ze_command_list_desc_t cl_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
  ze_command_list_handle_t cmd;
  CHECK(zeCommandListCreate(ctx, dev, &cl_desc, &cmd));

  ze_fence_desc_t f_desc{ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  ze_fence_handle_t fence;
  CHECK(zeFenceCreate(queue, &f_desc, &fence));

  auto get_kernel = [&](ze_module_handle_t m, const char *name) -> ze_kernel_handle_t {
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t kh;
    if (zeKernelCreate(m, &kd, &kh) != ZE_RESULT_SUCCESS) return nullptr;
    return kh;
  };

  ze_kernel_handle_t k_base = get_kernel(mod_all, "int4_gemv_m2_lm_head_argmax1");
  ze_kernel_handle_t k_dpas = get_kernel(mod_opt, "int4_gemv_m2_lm_head_dpas_argmax1");

  if (!k_base || !k_dpas) {
    std::fprintf(stderr, "Failed to create LM head kernels!\n");
    return 1;
  }

  const int VOCAB_SIZE = 248320;
  const int HIDDEN_DIM = 2048;
  const int NUM_GROUPS = (VOCAB_SIZE + 255) / 256; // 970

  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  float *d_x;
  uint8_t *d_w;
  uint16_t *d_s;
  float *d_stg1_v0, *d_stg1_v1;
  uint32_t *d_stg1_i0, *d_stg1_i1;

  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, 2 * HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_x));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, (size_t)VOCAB_SIZE * (HIDDEN_DIM / 2), 64, dev, (void **)&d_w));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, (size_t)VOCAB_SIZE * (HIDDEN_DIM / 128) * sizeof(uint16_t), 64, dev, (void **)&d_s));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, NUM_GROUPS * sizeof(float), 64, dev, (void **)&d_stg1_v0));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, NUM_GROUPS * sizeof(float), 64, dev, (void **)&d_stg1_v1));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, NUM_GROUPS * sizeof(uint32_t), 64, dev, (void **)&d_stg1_i0));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, NUM_GROUPS * sizeof(uint32_t), 64, dev, (void **)&d_stg1_i1));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> f_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> u_dist(0, 255);

  for (int i = 0; i < 2 * HIDDEN_DIM; ++i) d_x[i] = f_dist(rng);
  for (size_t i = 0; i < (size_t)VOCAB_SIZE * (HIDDEN_DIM / 2); ++i) d_w[i] = (uint8_t)u_dist(rng);
  for (size_t i = 0; i < (size_t)VOCAB_SIZE * (HIDDEN_DIM / 128); ++i) d_s[i] = float_to_bf16(0.05f + 0.05f * f_dist(rng));

  void *null_logits = nullptr;
  int M = VOCAB_SIZE;
  int K = HIDDEN_DIM;

  auto run_lm_kernel = [&](ze_kernel_handle_t k, int iters) -> double {
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &null_logits));
    CHECK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_w));
    CHECK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_s));
    CHECK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_x));
    CHECK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_stg1_v0));
    CHECK(zeKernelSetArgumentValue(k, 5, sizeof(void *), &d_stg1_i0));
    CHECK(zeKernelSetArgumentValue(k, 6, sizeof(void *), &d_stg1_v1));
    CHECK(zeKernelSetArgumentValue(k, 7, sizeof(void *), &d_stg1_i1));
    CHECK(zeKernelSetArgumentValue(k, 8, sizeof(int), &M));
    CHECK(zeKernelSetArgumentValue(k, 9, sizeof(int), &K));
    CHECK(zeKernelSetGroupSize(k, 256, 1, 1));

    ze_group_count_t gc{(uint32_t)NUM_GROUPS, 1, 1};

    CHECK(zeCommandListReset(cmd));
    CHECK(zeCommandListAppendLaunchKernel(cmd, k, &gc, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(cmd));

    for (int i = 0; i < 3; ++i) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &cmd, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  };

  std::printf("=================================================================\n");
  std::printf("LM Head Dual-Token Evaluation [248320, 2048] on Arc 140V\n");
  std::printf("=================================================================\n");
  std::fflush(stdout);

  double t_base_ms = run_lm_kernel(k_base, 20);
  std::printf("Baseline Scalar LM Head:  %6.2f ms\n", t_base_ms);
  std::fflush(stdout);

  double t_dpas_ms = run_lm_kernel(k_dpas, 20);
  std::printf("DPAS Systolic LM Head:    %6.2f ms  (Speedup: %.2fx)\n",
              t_dpas_ms, t_base_ms / t_dpas_ms);
  std::fflush(stdout);
  std::printf("=================================================================\n");

  CHECK(zeMemFree(ctx, d_x));
  CHECK(zeMemFree(ctx, d_w));
  CHECK(zeMemFree(ctx, d_s));
  CHECK(zeMemFree(ctx, d_stg1_v0));
  CHECK(zeMemFree(ctx, d_stg1_v1));
  CHECK(zeMemFree(ctx, d_stg1_i0));
  CHECK(zeMemFree(ctx, d_stg1_i1));

  CHECK(zeKernelDestroy(k_base));
  CHECK(zeKernelDestroy(k_dpas));
  CHECK(zeFenceDestroy(fence));
  CHECK(zeCommandListDestroy(cmd));
  CHECK(zeCommandQueueDestroy(queue));
  CHECK(zeModuleDestroy(mod_all));
  CHECK(zeModuleDestroy(mod_opt));
  CHECK(zeContextDestroy(ctx));

  return 0;
}
