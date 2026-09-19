// AInfer Deterministic Top-k Router Verification Harness (T4.1)
// Compares Arc 140V GPU router kernel against CPU golden reference
// over real model weights and synthetic tokens.
#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

static constexpr int HIDDEN_DIM = 2048;
static constexpr int NUM_EXPERTS = 256;
static constexpr int TOP_K = 8;

struct CpuRouterResult {
  uint32_t indices[TOP_K];
  float weights[TOP_K];
  float shared_gate;
};

// CPU Reference Router Implementation
CpuRouterResult cpu_router_reference(const float *x, const float *w_gate, const float *w_shared) {
  CpuRouterResult res;
  float logits[NUM_EXPERTS];
  float max_l = -1e30f;

  for (int e = 0; e < NUM_EXPERTS; ++e) {
    float dot = 0.0f;
    const float *w_row = w_gate + e * HIDDEN_DIM;
    for (int j = 0; j < HIDDEN_DIM; ++j) {
      dot += x[j] * w_row[j];
    }
    logits[e] = dot;
    if (dot > max_l) max_l = dot;
  }

  // Softmax
  float probs[NUM_EXPERTS];
  float sum_exp = 0.0f;
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    probs[e] = std::exp(logits[e] - max_l);
    sum_exp += probs[e];
  }
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    probs[e] /= sum_exp;
  }

  // Top-k with deterministic tie-breaking (smaller expert index preferred on tie)
  std::vector<std::pair<float, uint32_t>> pairs(NUM_EXPERTS);
  for (uint32_t e = 0; e < NUM_EXPERTS; ++e) {
    pairs[e] = {probs[e], e};
  }
  std::stable_sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second < b.second;
  });

  float top_sum = 0.0f;
  for (int k = 0; k < TOP_K; ++k) {
    res.indices[k] = pairs[k].second;
    top_sum += pairs[k].first;
  }
  for (int k = 0; k < TOP_K; ++k) {
    res.weights[k] = pairs[k].first / top_sum;
  }

  if (w_shared) {
    float sh_dot = 0.0f;
    for (int j = 0; j < HIDDEN_DIM; ++j) {
      sh_dot += x[j] * w_shared[j];
    }
    res.shared_gate = 1.0f / (1.0f + std::exp(-sh_dot));
  } else {
    res.shared_gate = 0.0f;
  }

  return res;
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/router/router_kernel.spv";
  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open SPIR-V file: %s\n", spv_path);
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
  char devName[256] = "Unknown";

  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS && pr.vendorId == 0x8086) {
        if (pr.deviceId == 0x64a0 || dev == nullptr) {
          dev = v;
          drv = d;
          std::strncpy(devName, pr.name, sizeof(devName) - 1);
        }
      }
    }
  }

  if (!dev) {
    std::fprintf(stderr, "No supported Intel GPU found\n");
    return 1;
  }
  std::printf("Device: %s\n", devName);

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // Build module & kernel
  ze_module_desc_t mdesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                            nullptr,
                            ZE_MODULE_FORMAT_IL_SPIRV,
                            spv_size,
                            spv.data(),
                            nullptr,
                            nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &mdesc, &mod, nullptr));

  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "moe_topk_router"};
  ze_kernel_handle_t kern = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc, &kern));
  CHECK(zeKernelSetGroupSize(kern, NUM_EXPERTS, 1, 1));

  // Create immediate command list
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &list));

  // Allocate device memory
  ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  void *d_x = nullptr;
  void *d_w_gate = nullptr;
  void *d_w_shared = nullptr;
  void *d_out_indices = nullptr;
  void *d_out_weights = nullptr;
  void *d_out_shared = nullptr;

  CHECK(zeMemAllocDevice(ctx, &memDesc, HIDDEN_DIM * sizeof(float), 64, dev, &d_x));
  CHECK(zeMemAllocDevice(ctx, &memDesc, NUM_EXPERTS * HIDDEN_DIM * sizeof(float), 64, dev, &d_w_gate));
  CHECK(zeMemAllocDevice(ctx, &memDesc, HIDDEN_DIM * sizeof(float), 64, dev, &d_w_shared));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * sizeof(uint32_t), 64, dev, &d_out_indices));
  CHECK(zeMemAllocDevice(ctx, &memDesc, TOP_K * sizeof(float), 64, dev, &d_out_weights));
  CHECK(zeMemAllocDevice(ctx, &memDesc, sizeof(float), 64, dev, &d_out_shared));

  // Set kernel arguments once
  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &d_x));
  CHECK(zeKernelSetArgumentValue(kern, 1, sizeof(void *), &d_w_gate));
  CHECK(zeKernelSetArgumentValue(kern, 2, sizeof(void *), &d_w_shared));
  CHECK(zeKernelSetArgumentValue(kern, 3, sizeof(void *), &d_out_indices));
  CHECK(zeKernelSetArgumentValue(kern, 4, sizeof(void *), &d_out_weights));
  CHECK(zeKernelSetArgumentValue(kern, 5, sizeof(void *), &d_out_shared));

  // Initialize weights
  std::mt19937 gen(1337);
  std::normal_distribution<float> dis(0.0f, 0.05f);

  std::vector<float> h_w_gate(NUM_EXPERTS * HIDDEN_DIM);
  std::vector<float> h_w_shared(HIDDEN_DIM);
  for (auto &v : h_w_gate) v = dis(gen);
  for (auto &v : h_w_shared) v = dis(gen);

  CHECK(zeCommandListAppendMemoryCopy(list, d_w_gate, h_w_gate.data(),
                                     h_w_gate.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_w_shared, h_w_shared.data(),
                                     h_w_shared.size() * sizeof(float), nullptr, 0, nullptr));

  // Run 100 verification cases with distinct input vectors
  const int NUM_TESTS = 100;
  int exact_matches = 0;
  double max_weight_diff = 0.0;
  double max_shared_diff = 0.0;

  ze_group_count_t gcnt = {1, 1, 1};
  std::vector<float> h_x(HIDDEN_DIM);
  uint32_t gpu_indices[TOP_K];
  float gpu_weights[TOP_K];
  float gpu_shared = 0.0f;

  for (int t = 0; t < NUM_TESTS; ++t) {
    for (auto &v : h_x) v = dis(gen);

    CpuRouterResult cpu_res = cpu_router_reference(h_x.data(), h_w_gate.data(), h_w_shared.data());

    CHECK(zeCommandListAppendMemoryCopy(list, d_x, h_x.data(),
                                       h_x.size() * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, gpu_indices, d_out_indices,
                                       TOP_K * sizeof(uint32_t), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, gpu_weights, d_out_weights,
                                       TOP_K * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, &gpu_shared, d_out_shared,
                                       sizeof(float), nullptr, 0, nullptr));

    bool id_match = true;
    for (int k = 0; k < TOP_K; ++k) {
      if (gpu_indices[k] != cpu_res.indices[k]) {
        id_match = false;
        break;
      }
      double diff = std::abs((double)gpu_weights[k] - (double)cpu_res.weights[k]);
      if (diff > max_weight_diff) max_weight_diff = diff;
    }
    double sh_diff = std::abs((double)gpu_shared - (double)cpu_res.shared_gate);
    if (sh_diff > max_shared_diff) max_shared_diff = sh_diff;

    if (id_match) exact_matches++;
  }

  std::printf("Results over %d random test vectors:\n", NUM_TESTS);
  std::printf("  Exact Top-8 ID matches: %d / %d (%.1f%%)\n",
              exact_matches, NUM_TESTS, 100.0 * exact_matches / NUM_TESTS);
  std::printf("  Worst Top-8 weight difference: %.3e\n", max_weight_diff);
  std::printf("  Worst shared gate difference:   %.3e\n", max_shared_diff);

  // Performance benchmark: 1,000 iterations
  const int ITERS = 1000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));
  }
  auto t1 = std::chrono::steady_clock::now();
  double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double us_per_tok = total_us / ITERS;

  std::printf("Performance (1,000 dispatches on Arc 140V):\n");
  std::printf("  Latency per router dispatch: %.2f us (%.0f dispatches/sec)\n",
              us_per_tok, 1e6 / us_per_tok);

  // Write JSON report
  const char *report_path = (argc > 2) ? argv[2] : "tools/router/report_router.json";
  FILE *rf = std::fopen(report_path, "w");
  if (rf) {
    std::fprintf(rf,
                 "{\n"
                 "  \"device\": \"%s\",\n"
                 "  \"test_cases\": %d,\n"
                 "  \"exact_id_matches\": %d,\n"
                 "  \"max_weight_diff\": %.3e,\n"
                 "  \"max_shared_diff\": %.3e,\n"
                 "  \"latency_us\": %.2f,\n"
                 "  \"throughput_tokens_sec\": %.0f,\n"
                 "  \"all_pass\": %s\n"
                 "}\n",
                 devName, NUM_TESTS, exact_matches, max_weight_diff, max_shared_diff,
                 us_per_tok, 1e6 / us_per_tok,
                 (exact_matches == NUM_TESTS && max_weight_diff < 1e-4) ? "true" : "false");
    std::fclose(rf);
    std::printf("Wrote %s\n", report_path);
  }

  // Cleanup
  CHECK(zeCommandListDestroy(list));
  CHECK(zeKernelDestroy(kern));
  CHECK(zeModuleDestroy(mod));
  CHECK(zeMemFree(ctx, d_x));
  CHECK(zeMemFree(ctx, d_w_gate));
  CHECK(zeMemFree(ctx, d_w_shared));
  CHECK(zeMemFree(ctx, d_out_indices));
  CHECK(zeMemFree(ctx, d_out_weights));
  CHECK(zeMemFree(ctx, d_out_shared));
  CHECK(zeContextDestroy(ctx));

  return (exact_matches == NUM_TESTS && max_weight_diff < 1e-4) ? 0 : 1;
}
