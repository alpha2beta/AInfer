// A/B microbench: gqa_attn_prefill_batch (v1, SLM tree) vs
// gqa_attn_prefill_batch_v2 (subgroup-shuffle reduction) on Arc 140V.
// Synthetic buffers in runtime layout; CPU reference for parity check.
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
const int NUM_Q_HEADS = 16;
const int NUM_KV_HEADS = 2;
const int GQA_GROUP_SIZE = 8;
const float ATTN_SCALE = 0.0625f;
const uint32_t MAX_CTX = 2048;

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

// Batched CPU reference: query b attends cache slots 0..base_pos+b.
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
  const char *spv_path = (argc > 1) ? argv[1] : "tools/kernels_258v/all_kernels.spv";

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
        std::printf("Device: %s (0x%04x)\n", props.name, props.deviceId);
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
  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));
  ze_fence_desc_t fdesc{ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  ze_fence_handle_t fence;
  CHECK(zeFenceCreate(queue, &fdesc, &fence));

  ze_kernel_desc_t kd1{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "gqa_attn_prefill_batch"};
  ze_kernel_handle_t k_v1;
  CHECK(zeKernelCreate(mod, &kd1, &k_v1));
  ze_kernel_desc_t kd2{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "gqa_attn_prefill_batch_v2"};
  ze_kernel_handle_t k_v2;
  CHECK(zeKernelCreate(mod, &kd2, &k_v2));

  ze_device_mem_alloc_desc_t dmem{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hmem{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  const int B_MAX = 256;
  float *d_q, *d_gate, *d_out;
  uint16_t *d_kc, *d_vc;
  int *d_ctrl;
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_gate));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d_out));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t), 64, dev, (void **)&d_kc));
  CHECK(zeMemAllocDevice(ctx, &dmem, (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t), 64, dev, (void **)&d_vc));
  CHECK(zeMemAllocShared(ctx, &dmem, &hmem, 16 * sizeof(int), 64, dev, (void **)&d_ctrl));

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
  std::vector<float> h_q((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> h_gate((size_t)B_MAX * NUM_Q_HEADS * HEAD_DIM);
  std::vector<uint16_t> h_kc((size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM);
  std::vector<uint16_t> h_vc((size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM);
  for (auto &x : h_q) x = fdist(rng);
  for (auto &x : h_gate) x = fdist(rng) * 0.5f;
  for (auto &x : h_kc) x = float_to_bf16(fdist(rng));
  for (auto &x : h_vc) x = float_to_bf16(fdist(rng));

  auto upload = [&](void *dst, const void *src, size_t n) -> int {
    ze_command_list_handle_t ul;
    CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &ul));
    CHECK(zeCommandListAppendMemoryCopy(ul, dst, src, n, nullptr, 0, nullptr));
    CHECK(zeCommandListClose(ul));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &ul, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListDestroy(ul));
    return 0;
  };
  upload(d_q, h_q.data(), h_q.size() * sizeof(float));
  upload(d_gate, h_gate.data(), h_gate.size() * sizeof(float));
  upload(d_kc, h_kc.data(), h_kc.size() * sizeof(uint16_t));
  upload(d_vc, h_vc.data(), h_vc.size() * sizeof(uint16_t));

  auto run_kernel = [&](ze_kernel_handle_t k, int B, uint32_t base_pos, int runs) -> double {
    d_ctrl[1] = (int)base_pos;
    uint32_t mc = MAX_CTX;
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void *), &d_out));
    CHECK(zeKernelSetArgumentValue(k, 1, sizeof(void *), &d_q));
    CHECK(zeKernelSetArgumentValue(k, 2, sizeof(void *), &d_gate));
    CHECK(zeKernelSetArgumentValue(k, 3, sizeof(void *), &d_kc));
    CHECK(zeKernelSetArgumentValue(k, 4, sizeof(void *), &d_vc));
    CHECK(zeKernelSetArgumentValue(k, 5, sizeof(void *), &d_ctrl));
    CHECK(zeKernelSetArgumentValue(k, 6, sizeof(uint32_t), &mc));
    CHECK(zeKernelSetArgumentValue(k, 7, sizeof(int), &B));
    CHECK(zeKernelSetGroupSize(k, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)(B * NUM_Q_HEADS), 1, 1};
    // warmup
    for (int w = 0; w < 2; ++w) {
      ze_command_list_handle_t list;
      CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));
      CHECK(zeCommandListAppendLaunchKernel(list, k, &gc, nullptr, 0, nullptr));
      CHECK(zeCommandListClose(list));
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      CHECK(zeCommandListDestroy(list));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
      ze_command_list_handle_t list;
      CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list));
      CHECK(zeCommandListAppendLaunchKernel(list, k, &gc, nullptr, 0, nullptr));
      CHECK(zeCommandListClose(list));
      CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence));
      CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
      CHECK(zeFenceReset(fence));
      CHECK(zeCommandListDestroy(list));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
  };

  auto download = [&](std::vector<float> &h, int B) -> int {
    h.resize((size_t)B * NUM_Q_HEADS * HEAD_DIM);
    ze_command_list_handle_t dl;
    CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &dl));
    CHECK(zeCommandListAppendMemoryCopy(dl, h.data(), d_out, h.size() * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListClose(dl));
    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &dl, fence));
    CHECK(zeFenceHostSynchronize(fence, UINT64_MAX));
    CHECK(zeFenceReset(fence));
    CHECK(zeCommandListDestroy(dl));
    return 0;
  };

  const int test_B[] = {32, 128, 256};
  std::printf("%-8s %-14s %-14s %-10s %-12s %-12s\n", "B", "v1_ms", "v2_ms", "speedup", "v1_vs_cpu", "v2_vs_cpu");
  bool all_ok = true;
  for (int B : test_B) {
    double t1 = run_kernel(k_v1, B, 0, 5);
    std::vector<float> h_v1;
    download(h_v1, B);
    double t2 = run_kernel(k_v2, B, 0, 5);
    std::vector<float> h_v2;
    download(h_v2, B);

    // CPU reference on first min(B,32) tokens only (reference is O(B^2*D), slow)
    int Br = B > 32 ? 32 : B;
    std::vector<float> h_ref((size_t)Br * NUM_Q_HEADS * HEAD_DIM);
    cpu_attn_batch(h_ref, h_q, h_gate, h_kc, h_vc, 0, Br);
    float d1 = 0.0f, d2 = 0.0f, d12 = 0.0f;
    for (size_t i = 0; i < h_ref.size(); ++i) {
      d1 = std::max(d1, std::fabs(h_v1[i] - h_ref[i]));
      d2 = std::max(d2, std::fabs(h_v2[i] - h_ref[i]));
      d12 = std::max(d12, std::fabs(h_v1[i] - h_v2[i]));
    }
    bool ok = d2 < 1e-3f;
    all_ok = all_ok && ok;
    std::printf("%-8d %-14.3f %-14.3f %-10.2fx %-12.2e %-12.2e %s (v1v2=%e)\n", B, t1, t2,
                t1 / t2, d1, d2, ok ? "PASS" : "FAIL", d12);
  }
  std::printf(all_ok ? "ALL PASS\n" : "FAILURES PRESENT\n");
  return all_ok ? 0 : 1;
}
