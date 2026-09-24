// bench_deltanet_chunk.cpp — A/B benchmark and numerical parity test between
// deltanet_recurrent_batch_v2 (serial baseline) and deltanet_chunked_batch (I2.3).

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

const int S_V = 128;
const int H_V = 32;
const int H_K = 16;
const int C_QKV = 8192;

int main(int argc, char **argv) {
  const char *spv_path = "tools/kernels_258v/deltanet_chunked_lnl.spv";
  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "Cannot open %s\n", spv_path);
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

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  ze_command_list_desc_t clist_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, compute_ord, 0};
  ze_command_list_handle_t list;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));

  ze_kernel_desc_t k_recr_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "deltanet_recurrent_batch_v2"};
  ze_kernel_handle_t k_recr;
  CHECK(zeKernelCreate(mod, &k_recr_desc, &k_recr));

  ze_kernel_desc_t k_chunk_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "deltanet_chunked_batch"};
  ze_kernel_handle_t k_chunk;
  CHECK(zeKernelCreate(mod, &k_chunk_desc, &k_chunk));

  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  const int MAX_B = 512;
  float *d_q, *d_k, *d_v, *d_g, *d_beta;
  float *d_out_recr, *d_out_chunk;
  float *d_state_recr, *d_state_chunk, *d_state_init;

  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_K * S_V * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_K * S_V * sizeof(float), 64, dev, (void **)&d_k));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * C_QKV * sizeof(float), 64, dev, (void **)&d_v));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_V * sizeof(float), 64, dev, (void **)&d_g));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_V * sizeof(float), 64, dev, (void **)&d_beta));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_V * S_V * sizeof(float), 64, dev, (void **)&d_out_recr));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, MAX_B * H_V * S_V * sizeof(float), 64, dev, (void **)&d_out_chunk));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, H_V * S_V * S_V * sizeof(float), 64, dev, (void **)&d_state_recr));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, H_V * S_V * S_V * sizeof(float), 64, dev, (void **)&d_state_chunk));
  CHECK(zeMemAllocShared(ctx, &dmem_desc, &hmem_desc, H_V * S_V * S_V * sizeof(float), 64, dev, (void **)&d_state_init));

  // Generate test inputs
  std::mt19937 rng(42);
  std::normal_distribution<float> norm_dist(0.0f, 1.0f);
  std::uniform_real_distribution<float> g_dist(0.85f, 0.99f);
  std::uniform_real_distribution<float> beta_dist(0.1f, 0.9f);

  for (size_t i = 0; i < (size_t)MAX_B * H_K * S_V; ++i) d_q[i] = norm_dist(rng);
  for (size_t i = 0; i < (size_t)MAX_B * H_K * S_V; ++i) d_k[i] = norm_dist(rng);

  // L2-normalize q and k per head as in head_l2_norm_qk_batch
  for (int b = 0; b < MAX_B; ++b) {
    for (int kh = 0; kh < H_K; ++kh) {
      float *q_head = d_q + (b * H_K + kh) * S_V;
      float *k_head = d_k + (b * H_K + kh) * S_V;
      float sq = 0.0f, sk = 0.0f;
      for (int i = 0; i < S_V; ++i) {
        sq += q_head[i] * q_head[i];
        sk += k_head[i] * k_head[i];
      }
      float inv_q = 1.0f / std::sqrt(sq + 1e-6f);
      float inv_k = 1.0f / std::sqrt(sk + 1e-6f);
      for (int i = 0; i < S_V; ++i) {
        q_head[i] *= inv_q;
        k_head[i] *= inv_k;
      }
    }
  }

  for (size_t i = 0; i < (size_t)MAX_B * C_QKV; ++i) d_v[i] = norm_dist(rng);
  for (size_t i = 0; i < (size_t)MAX_B * H_V; ++i) d_g[i] = g_dist(rng);
  for (size_t i = 0; i < (size_t)MAX_B * H_V; ++i) d_beta[i] = beta_dist(rng);
  for (size_t i = 0; i < (size_t)H_V * S_V * S_V; ++i) d_state_init[i] = norm_dist(rng) * 0.05f;

  std::vector<int> test_batches = {8, 16, 32, 64, 128, 256, 441};

  for (int B : test_batches) {
    // Copy initial state
    std::memcpy(d_state_recr, d_state_init, H_V * S_V * S_V * sizeof(float));
    std::memcpy(d_state_chunk, d_state_init, H_V * S_V * S_V * sizeof(float));
    std::memset(d_out_recr, 0, MAX_B * H_V * S_V * sizeof(float));
    std::memset(d_out_chunk, 0, MAX_B * H_V * S_V * sizeof(float));

    // 1. Run Serial Recurrence v2
    CHECK(zeCommandListReset(list));
    CHECK(zeKernelSetArgumentValue(k_recr, 0, sizeof(void *), &d_out_recr));
    CHECK(zeKernelSetArgumentValue(k_recr, 1, sizeof(void *), &d_state_recr));
    CHECK(zeKernelSetArgumentValue(k_recr, 2, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_recr, 3, sizeof(void *), &d_k));
    CHECK(zeKernelSetArgumentValue(k_recr, 4, sizeof(void *), &d_v));
    CHECK(zeKernelSetArgumentValue(k_recr, 5, sizeof(void *), &d_g));
    CHECK(zeKernelSetArgumentValue(k_recr, 6, sizeof(void *), &d_beta));
    CHECK(zeKernelSetArgumentValue(k_recr, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k_recr, 128, 1, 1));
    ze_group_count_t gcnt{(uint32_t)H_V, 1, 1};
    CHECK(zeCommandListAppendLaunchKernel(list, k_recr, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));

    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    // 2. Run Chunked Parallel Scan
    CHECK(zeCommandListReset(list));
    CHECK(zeKernelSetArgumentValue(k_chunk, 0, sizeof(void *), &d_out_chunk));
    CHECK(zeKernelSetArgumentValue(k_chunk, 1, sizeof(void *), &d_state_chunk));
    CHECK(zeKernelSetArgumentValue(k_chunk, 2, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k_chunk, 3, sizeof(void *), &d_k));
    CHECK(zeKernelSetArgumentValue(k_chunk, 4, sizeof(void *), &d_v));
    CHECK(zeKernelSetArgumentValue(k_chunk, 5, sizeof(void *), &d_g));
    CHECK(zeKernelSetArgumentValue(k_chunk, 6, sizeof(void *), &d_beta));
    CHECK(zeKernelSetArgumentValue(k_chunk, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k_chunk, 128, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list, k_chunk, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));

    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    std::printf("[B=%d] recr[0..3]: %f %f %f %f | chunk[0..3]: %f %f %f %f\n",
                B, d_out_recr[0], d_out_recr[1], d_out_recr[2], d_out_recr[3],
                d_out_chunk[0], d_out_chunk[1], d_out_chunk[2], d_out_chunk[3]);
    std::printf("       s_recr[0..3]: %f %f %f %f | s_chunk[0..3]: %f %f %f %f\n",
                d_state_recr[0], d_state_recr[1], d_state_recr[2], d_state_recr[3],
                d_state_chunk[0], d_state_chunk[1], d_state_chunk[2], d_state_chunk[3]);

    // 3. Parity checks
    float max_out_diff = 0.0f;
    for (size_t i = 0; i < (size_t)B * H_V * S_V; ++i) {
      float d = std::abs(d_out_recr[i] - d_out_chunk[i]);
      if (d > max_out_diff) max_out_diff = d;
    }

    float max_state_diff = 0.0f;
    for (size_t i = 0; i < (size_t)H_V * S_V * S_V; ++i) {
      float d = std::abs(d_state_recr[i] - d_state_chunk[i]);
      if (d > max_state_diff) max_state_diff = d;
    }

    // 4. Benchmarking
    const int WARMUP = 5;
    const int ITERS = 20;

    // Benchmark Serial
    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendLaunchKernel(list, k_recr, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));
    for (int it = 0; it < WARMUP; ++it) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    }
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    }
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    auto t1 = std::chrono::steady_clock::now();
    double ms_recr = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    // Benchmark Chunked
    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendLaunchKernel(list, k_chunk, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));
    for (int it = 0; it < WARMUP; ++it) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    }
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    }
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    t1 = std::chrono::steady_clock::now();
    double ms_chunk = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    double speedup = ms_recr / ms_chunk;
    const char *status = (max_out_diff < 1e-4f && max_state_diff < 1e-4f) ? "PASS" : "FAIL";

    std::printf("[B=%3d] %s | Out Diff: %8.2e | State Diff: %8.2e | Serial: %6.3f ms | Chunked: %6.3f ms | Speedup: %5.2fx\n",
                B, status, max_out_diff, max_state_diff, ms_recr, ms_chunk, speedup);
    if (std::strcmp(status, "PASS") != 0) {
      std::fprintf(stderr, "Parity check FAILED for B=%d\n", B);
      return 1;
    }
  }

  std::printf("=================================================================\n");
  std::printf("All DeltaNet Chunked Parallel Scan tests PASSED!\n");

  zeMemFree(ctx, d_q);
  zeMemFree(ctx, d_k);
  zeMemFree(ctx, d_v);
  zeMemFree(ctx, d_g);
  zeMemFree(ctx, d_beta);
  zeMemFree(ctx, d_out_recr);
  zeMemFree(ctx, d_out_chunk);
  zeMemFree(ctx, d_state_recr);
  zeMemFree(ctx, d_state_chunk);
  zeMemFree(ctx, d_state_init);

  zeKernelDestroy(k_recr);
  zeKernelDestroy(k_chunk);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeModuleDestroy(mod);
  zeContextDestroy(ctx);

  return 0;
}
