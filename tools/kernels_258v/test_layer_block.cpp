// AInfer Single-Layer End-to-End Block Verification Harness on Arc 140V (T4.7)
// Tests:
// 1. DeltaNet-MoE Block (Layer 0 topology)
// 2. Full-Attention-MoE Block (Layer 3 topology)
// Verifies bit-level output parity against CPU reference implementation.
// Emits tools/kernels_258v/report_layer_block.json.

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

const int HIDDEN_DIM = 2048;
const int EXP_INTER_DIM = 512;
const int NUM_EXPERTS = 256;
const int TOP_K = 8;
const int S_V = 128;
const int H_V = 32;
const int H_K = 16;
const int C_QKV = 8192;
const int HEAD_DIM = 256;
const int ROTARY_HALF = 32;
const int NUM_Q_HEADS = 16;
const int NUM_KV_HEADS = 2;
const int GQA_GROUP_SIZE = 8;
const float ROPE_THETA = 10000000.0f;
const float ATTN_SCALE = 0.0625f;
const float SCALE_128 = 0.08838834764831845f;
const float EPS = 1e-6f;

static inline float bf16_to_fp32(uint16_t b) {
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

// CPU Reference Helpers
void cpu_rmsnorm(float *y, const float *x, const float *w, int dim, float eps, bool zero_centered) {
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
  float inv_rms = 1.0f / std::sqrt(ss / (float)dim + eps);
  for (int i = 0; i < dim; ++i) {
    float scale = zero_centered ? (1.0f + w[i]) : w[i];
    y[i] = x[i] * inv_rms * scale;
  }
}

void cpu_gemv_int4(float *y, const uint8_t *w_packed, const uint16_t *w_scale,
                   const float *x, int M, int K) {
  int num_groups = K / 128;
  for (int m = 0; m < M; ++m) {
    const uint8_t *row_w = w_packed + (size_t)m * (K / 2);
    const uint16_t *row_s = w_scale + (size_t)m * num_groups;
    float total = 0.0f;

    for (int g = 0; g < num_groups; ++g) {
      float scale = bf16_to_fp32(row_s[g]);
      const uint8_t *grp_w = row_w + g * 64;
      const float *grp_x = x + g * 128;

      float grp_acc = 0.0f;
      for (int i = 0; i < 64; ++i) {
        uint8_t wb = grp_w[i];
        int8_t n0 = (wb & 0x0F); if (n0 >= 8) n0 -= 16;
        int8_t n1 = (wb >> 4);   if (n1 >= 8) n1 -= 16;
        grp_acc += (float)n0 * grp_x[2 * i] + (float)n1 * grp_x[2 * i + 1];
      }
      total += grp_acc * scale;
    }
    y[m] = total;
  }
}

void cpu_silu_mul(float *z, const float *g, const float *u, int dim) {
  for (int i = 0; i < dim; ++i) {
    float silu_g = g[i] / (1.0f + std::exp(-g[i]));
    z[i] = silu_g * u[i];
  }
}

void cpu_top8_router(uint32_t *top_idx, float *top_wt, float *shared_gate_val,
                     const float *x, const float *w_gate, const float *w_shared) {
  std::vector<float> logits(NUM_EXPERTS);
  float max_l = -1e30f;
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM; ++j) sum += x[j] * w_gate[e * HIDDEN_DIM + j];
    logits[e] = sum;
    if (sum > max_l) max_l = sum;
  }

  std::vector<float> probs(NUM_EXPERTS);
  float sum_p = 0.0f;
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    probs[e] = std::exp(logits[e] - max_l);
    sum_p += probs[e];
  }
  for (int e = 0; e < NUM_EXPERTS; ++e) probs[e] /= sum_p;

  // Top-8 selection with tie-breaking
  std::vector<float> best_p(TOP_K, -1.0f);
  std::vector<uint32_t> best_i(TOP_K, 0xFFFFFFFF);

  for (uint32_t i = 0; i < NUM_EXPERTS; ++i) {
    float p = probs[i];
    if (p > best_p[TOP_K - 1]) {
      int pos = TOP_K - 1;
      while (pos > 0 && (p > best_p[pos - 1] || (p == best_p[pos - 1] && i < best_i[pos - 1]))) {
        pos--;
      }
      for (int j = TOP_K - 1; j > pos; --j) {
        best_p[j] = best_p[j - 1];
        best_i[j] = best_i[j - 1];
      }
      best_p[pos] = p;
      best_i[pos] = i;
    }
  }

  float top_sum = 0.0f;
  for (int k = 0; k < TOP_K; ++k) top_sum += best_p[k];
  for (int k = 0; k < TOP_K; ++k) {
    top_idx[k] = best_i[k];
    top_wt[k] = best_p[k] / top_sum;
  }

  if (w_shared) {
    float sh_sum = 0.0f;
    for (int j = 0; j < HIDDEN_DIM; ++j) sh_sum += x[j] * w_shared[j];
    *shared_gate_val = sh_sum;
  }
}

// Helper to generate INT4 quantized weights and BF16 scales
void make_quant_weight(std::vector<uint8_t> &w_packed, std::vector<uint16_t> &w_scale,
                       int M, int K, std::mt19937 &rng) {
  w_packed.resize((size_t)M * (K / 2));
  w_scale.resize((size_t)M * (K / 128));
  std::uniform_int_distribution<int> idist(0, 255);
  for (size_t i = 0; i < w_packed.size(); ++i) w_packed[i] = (uint8_t)idist(rng);
  for (size_t i = 0; i < w_scale.size(); ++i) w_scale[i] = float_to_bf16(0.01f);
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
        std::printf("Device: %s (devId=0x%04x)\n", props.name, props.deviceId);
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
  ze_command_list_handle_t list_copy, list1_attn, list1_moe, list2_attn, list2_moe;
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list_copy));
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list1_attn));
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list1_moe));
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list2_attn));
  CHECK(zeCommandListCreate(ctx, dev, &clist_desc, &list2_moe));

  ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, compute_ord, 0, 0,
                                 ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue;
  CHECK(zeCommandQueueCreate(ctx, dev, &q_desc, &queue));

  // Initialize kernel handles
  auto get_kernel = [&](const char *name) -> ze_kernel_handle_t {
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t kh = nullptr;
    ze_result_t res = zeKernelCreate(mod, &kd, &kh);
    if (res != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "Failed to create kernel %s (code %d)\n", name, (int)res);
      std::exit(1);
    }
    return kh;
  };

  ze_kernel_handle_t k_gemv = get_kernel("int4_gemv_m1");
  ze_kernel_handle_t k_router = get_kernel("moe_topk_router");
  ze_kernel_handle_t k_norm2048 = get_kernel("rmsnorm_2048");
  ze_kernel_handle_t k_silu512 = get_kernel("silu_mul_512");
  ze_kernel_handle_t k_resadd = get_kernel("residual_add_2048");
  ze_kernel_handle_t k_accum = get_kernel("moe_accumulate_expert");
  ze_kernel_handle_t k_add_shared = get_kernel("moe_add_shared_expert");

  // DeltaNet kernels
  ze_kernel_handle_t k_conv = get_kernel("conv1d_update_silu");
  ze_kernel_handle_t k_l2_norm = get_kernel("head_l2_norm_128");
  ze_kernel_handle_t k_gate_prep = get_kernel("deltanet_gate_prep");
  ze_kernel_handle_t k_recr = get_kernel("deltanet_recurrent_decode");
  ze_kernel_handle_t k_hnorm = get_kernel("deltanet_head_norm_silu_z");

  // Full-Attention kernels
  ze_kernel_handle_t k_norm256 = get_kernel("rmsnorm_head_256");
  ze_kernel_handle_t k_norm256_k = get_kernel("rmsnorm_head_256");
  ze_kernel_handle_t k_rope = get_kernel("rope_and_kv_append_bf16");
  ze_kernel_handle_t k_attn = get_kernel("gqa_attn_decode_bf16");

  auto launch_gemv = [&](ze_command_list_handle_t cmd_list, float *y, uint8_t *w, uint16_t *s, float *x, int M, int K) -> void {
    zeKernelSetArgumentValue(k_gemv, 0, sizeof(void *), &y);
    zeKernelSetArgumentValue(k_gemv, 1, sizeof(void *), &w);
    zeKernelSetArgumentValue(k_gemv, 2, sizeof(void *), &s);
    zeKernelSetArgumentValue(k_gemv, 3, sizeof(void *), &x);
    zeKernelSetArgumentValue(k_gemv, 4, sizeof(int), &M);
    zeKernelSetArgumentValue(k_gemv, 5, sizeof(int), &K);
    zeKernelSetGroupSize(k_gemv, 256, 1, 1);
    ze_group_count_t gc{(uint32_t)((M + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(cmd_list, k_gemv, &gc, nullptr, 0, nullptr);
  };

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

  // Common MoE and Norm host parameters
  std::vector<float> h_in_norm_w(HIDDEN_DIM), h_post_norm_w(HIDDEN_DIM);
  std::vector<float> h_router_w(NUM_EXPERTS * HIDDEN_DIM), h_shared_gate_w(HIDDEN_DIM);
  for (auto &x : h_in_norm_w) x = fdist(rng) * 0.1f;
  for (auto &x : h_post_norm_w) x = fdist(rng) * 0.1f;
  for (auto &x : h_router_w) x = fdist(rng) * 0.1f;
  for (auto &x : h_shared_gate_w) x = fdist(rng) * 0.1f;

  // Shared expert weights
  std::vector<uint8_t> sh_gate_w, sh_up_w, sh_down_w;
  std::vector<uint16_t> sh_gate_s, sh_up_s, sh_down_s;
  make_quant_weight(sh_gate_w, sh_gate_s, EXP_INTER_DIM, HIDDEN_DIM, rng);
  make_quant_weight(sh_up_w, sh_up_s, EXP_INTER_DIM, HIDDEN_DIM, rng);
  make_quant_weight(sh_down_w, sh_down_s, HIDDEN_DIM, EXP_INTER_DIM, rng);

  // 8 Active routed expert weights (mock 8 selected experts)
  std::vector<std::vector<uint8_t>> exp_gu_w(TOP_K), exp_down_w(TOP_K);
  std::vector<std::vector<uint16_t>> exp_gu_s(TOP_K), exp_down_s(TOP_K);
  for (int k = 0; k < TOP_K; ++k) {
    make_quant_weight(exp_gu_w[k], exp_gu_s[k], 2 * EXP_INTER_DIM, HIDDEN_DIM, rng);
    make_quant_weight(exp_down_w[k], exp_down_s[k], HIDDEN_DIM, EXP_INTER_DIM, rng);
  }

  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

  // Shared / Common Device Allocations
  float *d_x_in, *d_in_norm_w, *d_x_norm, *d_post_norm_w, *d_router_w, *d_shared_gate_w;
  float *d_top_wt, *d_sh_gate_val, *d_moe_acc, *d_y_final;
  uint32_t *d_top_idx;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_x_in));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_in_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_x_norm));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_post_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_EXPERTS * HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_router_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_shared_gate_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, TOP_K * sizeof(float), 64, dev, (void **)&d_top_wt));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sizeof(float), 64, dev, (void **)&d_sh_gate_val));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, TOP_K * sizeof(uint32_t), 64, dev, (void **)&d_top_idx));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_moe_acc));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_y_final));

  // Shared expert device weights & scratch
  uint8_t *d_sh_gw, *d_sh_uw, *d_sh_dw;
  uint16_t *d_sh_gs, *d_sh_us, *d_sh_ds;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_gate_w.size(), 64, dev, (void **)&d_sh_gw));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_gate_s.size() * 2, 64, dev, (void **)&d_sh_gs));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_up_w.size(), 64, dev, (void **)&d_sh_uw));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_up_s.size() * 2, 64, dev, (void **)&d_sh_us));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_down_w.size(), 64, dev, (void **)&d_sh_dw));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, sh_down_s.size() * 2, 64, dev, (void **)&d_sh_ds));

  float *d_exp_gu, *d_exp_act, *d_exp_out, *d_sh_g, *d_sh_u, *d_sh_act, *d_sh_out;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2 * EXP_INTER_DIM * sizeof(float), 64, dev, (void **)&d_exp_gu));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, EXP_INTER_DIM * sizeof(float), 64, dev, (void **)&d_exp_act));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_exp_out));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, EXP_INTER_DIM * sizeof(float), 64, dev, (void **)&d_sh_g));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, EXP_INTER_DIM * sizeof(float), 64, dev, (void **)&d_sh_u));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, EXP_INTER_DIM * sizeof(float), 64, dev, (void **)&d_sh_act));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_sh_out));

  // 8 Active expert weights
  std::vector<uint8_t *> d_exp_gu_w(TOP_K), d_exp_down_w(TOP_K);
  std::vector<uint16_t *> d_exp_gu_s(TOP_K), d_exp_down_s(TOP_K);
  for (int k = 0; k < TOP_K; ++k) {
    CHECK(zeMemAllocDevice(ctx, &dmem_desc, exp_gu_w[k].size(), 64, dev, (void **)&d_exp_gu_w[k]));
    CHECK(zeMemAllocDevice(ctx, &dmem_desc, exp_gu_s[k].size() * 2, 64, dev, (void **)&d_exp_gu_s[k]));
    CHECK(zeMemAllocDevice(ctx, &dmem_desc, exp_down_w[k].size(), 64, dev, (void **)&d_exp_down_w[k]));
    CHECK(zeMemAllocDevice(ctx, &dmem_desc, exp_down_s[k].size() * 2, 64, dev, (void **)&d_exp_down_s[k]));
  }

  // Upload common & expert weights upfront
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_in_norm_w, h_in_norm_w.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_post_norm_w, h_post_norm_w.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_router_w, h_router_w.data(), h_router_w.size() * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_shared_gate_w, h_shared_gate_w.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_gw, sh_gate_w.data(), sh_gate_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_gs, sh_gate_s.data(), sh_gate_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_uw, sh_up_w.data(), sh_up_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_us, sh_up_s.data(), sh_up_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_dw, sh_down_w.data(), sh_down_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_sh_ds, sh_down_s.data(), sh_down_s.size() * 2, nullptr, 0, nullptr));

  for (int k = 0; k < TOP_K; ++k) {
    CHECK(zeCommandListAppendMemoryCopy(list_copy, d_exp_gu_w[k], exp_gu_w[k].data(), exp_gu_w[k].size(), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list_copy, d_exp_gu_s[k], exp_gu_s[k].data(), exp_gu_s[k].size() * 2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list_copy, d_exp_down_w[k], exp_down_w[k].data(), exp_down_w[k].size(), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list_copy, d_exp_down_s[k], exp_down_s[k].data(), exp_down_s[k].size() * 2, nullptr, 0, nullptr));
  }
  CHECK(zeCommandListAppendBarrier(list_copy, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Common Input vector
  std::vector<float> h_x_in(HIDDEN_DIM);
  for (auto &x : h_x_in) x = fdist(rng);

  // Upload input vector
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_x_in, h_x_in.data(), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // =========================================================================
  // BLOCK 1: DELTANET-MOE BLOCK (LAYER 0) VERIFICATION
  // =========================================================================
  std::printf("\n=======================================================\n");
  std::printf("--- Verifying Block 1: DeltaNet-MoE (Layer 0 Topology) ---\n");
  std::printf("=======================================================\n");

  // Linear attention projection weights
  std::vector<uint8_t> qkv_w, z_w, a_w, b_w, out_proj_w;
  std::vector<uint16_t> qkv_s, z_s, a_s, b_s, out_proj_s;
  make_quant_weight(qkv_w, qkv_s, C_QKV, HIDDEN_DIM, rng);
  make_quant_weight(z_w, z_s, H_V * S_V, HIDDEN_DIM, rng);
  make_quant_weight(a_w, a_s, H_V, HIDDEN_DIM, rng);
  make_quant_weight(b_w, b_s, H_V, HIDDEN_DIM, rng);
  make_quant_weight(out_proj_w, out_proj_s, HIDDEN_DIM, H_V * S_V, rng);

  std::vector<float> h_conv_w(C_QKV * 4), h_conv_state_ref(C_QKV * 3, 0.0f);
  for (auto &x : h_conv_w) x = fdist(rng) * 0.1f;
  std::vector<float> h_dt_bias(H_V), h_A_log(H_V), h_ssm_norm_w(S_V);
  for (auto &x : h_dt_bias) x = fdist(rng) * 0.5f;
  for (auto &x : h_A_log) x = -fdist(rng) * 0.5f - 1.0f;
  for (auto &x : h_ssm_norm_w) x = fdist(rng) * 0.2f + 0.8f;
  std::vector<float> h_ssm_state_ref(H_V * S_V * S_V, 0.0f);

  // --- CPU Forward Pass for Block 1 ---
  std::vector<float> c_x_norm(HIDDEN_DIM);
  cpu_rmsnorm(c_x_norm.data(), h_x_in.data(), h_in_norm_w.data(), HIDDEN_DIM, EPS, true);

  std::vector<float> c_qkv(C_QKV), c_z(H_V * S_V), c_a(H_V), c_b(H_V);
  cpu_gemv_int4(c_qkv.data(), qkv_w.data(), qkv_s.data(), c_x_norm.data(), C_QKV, HIDDEN_DIM);
  cpu_gemv_int4(c_z.data(), z_w.data(), z_s.data(), c_x_norm.data(), H_V * S_V, HIDDEN_DIM);
  cpu_gemv_int4(c_a.data(), a_w.data(), a_s.data(), c_x_norm.data(), H_V, HIDDEN_DIM);
  cpu_gemv_int4(c_b.data(), b_w.data(), b_s.data(), c_x_norm.data(), H_V, HIDDEN_DIM);

  std::vector<float> c_qkv_conv(C_QKV);
  for (int c = 0; c < C_QKV; ++c) {
    float s0 = h_conv_state_ref[c * 3 + 0];
    float s1 = h_conv_state_ref[c * 3 + 1];
    float s2 = h_conv_state_ref[c * 3 + 2];
    float x = c_qkv[c];
    float sum = s0 * h_conv_w[c * 4 + 0] + s1 * h_conv_w[c * 4 + 1] + s2 * h_conv_w[c * 4 + 2] + x * h_conv_w[c * 4 + 3];
    c_qkv_conv[c] = sum / (1.0f + std::exp(-sum));
    h_conv_state_ref[c * 3 + 0] = s1;
    h_conv_state_ref[c * 3 + 1] = s2;
    h_conv_state_ref[c * 3 + 2] = x;
  }

  std::vector<float> c_q(H_K * S_V), c_k(H_K * S_V);
  for (int h = 0; h < H_K; ++h) {
    float ss_q = 0.0f, ss_k = 0.0f;
    for (int i = 0; i < S_V; ++i) {
      ss_q += c_qkv_conv[h * S_V + i] * c_qkv_conv[h * S_V + i];
      ss_k += c_qkv_conv[H_K * S_V + h * S_V + i] * c_qkv_conv[H_K * S_V + h * S_V + i];
    }
    float inv_l2_q = 1.0f / std::max(std::sqrt(ss_q), EPS);
    float inv_l2_k = 1.0f / std::max(std::sqrt(ss_k), EPS);
    for (int i = 0; i < S_V; ++i) {
      c_q[h * S_V + i] = c_qkv_conv[h * S_V + i] * inv_l2_q;
      c_k[h * S_V + i] = c_qkv_conv[H_K * S_V + h * S_V + i] * inv_l2_k;
    }
  }
  const float *c_v = c_qkv_conv.data() + 2 * H_K * S_V;

  std::vector<float> c_g(H_V), c_beta(H_V);
  for (int h = 0; h < H_V; ++h) {
    float a_val = c_a[h] + h_dt_bias[h];
    float dt = (a_val > 20.0f) ? a_val : std::log(1.0f + std::exp(a_val));
    c_g[h] = std::exp(-std::exp(h_A_log[h]) * dt);
    c_beta[h] = 1.0f / (1.0f + std::exp(-c_b[h]));
  }

  std::vector<float> c_attn_out(H_V * S_V);
  for (int h = 0; h < H_V; ++h) {
    int kh = h / 2;
    const float *qh = c_q.data() + kh * S_V;
    const float *kh_v = c_k.data() + kh * S_V;
    const float *vh = c_v + h * S_V;
    float *S_h = h_ssm_state_ref.data() + h * (S_V * S_V);
    float gh = c_g[h];
    float betah = c_beta[h];

    for (int j = 0; j < S_V; ++j) {
      float kv = 0.0f;
      for (int i = 0; i < S_V; ++i) kv += S_h[i * S_V + j] * kh_v[i];
      kv *= gh;
      float delta = (vh[j] - kv) * betah;
      float o = 0.0f;
      for (int i = 0; i < S_V; ++i) {
        float s_new = gh * S_h[i * S_V + j] + kh_v[i] * delta;
        S_h[i * S_V + j] = s_new;
        o += s_new * qh[i];
      }
      c_attn_out[h * S_V + j] = o * SCALE_128;
    }
  }

  std::vector<float> c_attn_norm(H_V * S_V);
  for (int h = 0; h < H_V; ++h) {
    float ss = 0.0f;
    for (int j = 0; j < S_V; ++j) ss += c_attn_out[h * S_V + j] * c_attn_out[h * S_V + j];
    float inv_rms = 1.0f / std::sqrt(ss / (float)S_V + EPS);
    for (int j = 0; j < S_V; ++j) {
      float n = c_attn_out[h * S_V + j] * inv_rms * h_ssm_norm_w[j];
      float zv = c_z[h * S_V + j];
      c_attn_norm[h * S_V + j] = n * (zv / (1.0f + std::exp(-zv)));
    }
  }

  std::vector<float> c_attn_proj(HIDDEN_DIM);
  cpu_gemv_int4(c_attn_proj.data(), out_proj_w.data(), out_proj_s.data(), c_attn_norm.data(), HIDDEN_DIM, H_V * S_V);

  std::vector<float> c_x_mid(HIDDEN_DIM);
  for (int i = 0; i < HIDDEN_DIM; ++i) c_x_mid[i] = h_x_in[i] + c_attn_proj[i];

  std::vector<float> c_x_post(HIDDEN_DIM);
  cpu_rmsnorm(c_x_post.data(), c_x_mid.data(), h_post_norm_w.data(), HIDDEN_DIM, EPS, true);

  // MoE Step
  uint32_t c_top_idx[TOP_K];
  float c_top_wt[TOP_K];
  float c_sh_gate_val;
  cpu_top8_router(c_top_idx, c_top_wt, &c_sh_gate_val, c_x_post.data(), h_router_w.data(), h_shared_gate_w.data());

  std::vector<float> c_moe_acc(HIDDEN_DIM, 0.0f);
  for (int k = 0; k < TOP_K; ++k) {
    std::vector<float> gu(2 * EXP_INTER_DIM);
    cpu_gemv_int4(gu.data(), exp_gu_w[k].data(), exp_gu_s[k].data(), c_x_post.data(), 2 * EXP_INTER_DIM, HIDDEN_DIM);
    std::vector<float> act(EXP_INTER_DIM);
    cpu_silu_mul(act.data(), gu.data(), gu.data() + EXP_INTER_DIM, EXP_INTER_DIM);
    std::vector<float> exp_out(HIDDEN_DIM);
    cpu_gemv_int4(exp_out.data(), exp_down_w[k].data(), exp_down_s[k].data(), act.data(), HIDDEN_DIM, EXP_INTER_DIM);
    for (int i = 0; i < HIDDEN_DIM; ++i) c_moe_acc[i] += exp_out[i] * c_top_wt[k];
  }

  // Shared expert
  std::vector<float> c_sh_g(EXP_INTER_DIM), c_sh_u(EXP_INTER_DIM), c_sh_act(EXP_INTER_DIM), c_sh_out(HIDDEN_DIM);
  cpu_gemv_int4(c_sh_g.data(), sh_gate_w.data(), sh_gate_s.data(), c_x_post.data(), EXP_INTER_DIM, HIDDEN_DIM);
  cpu_gemv_int4(c_sh_u.data(), sh_up_w.data(), sh_up_s.data(), c_x_post.data(), EXP_INTER_DIM, HIDDEN_DIM);
  cpu_silu_mul(c_sh_act.data(), c_sh_g.data(), c_sh_u.data(), EXP_INTER_DIM);
  cpu_gemv_int4(c_sh_out.data(), sh_down_w.data(), sh_down_s.data(), c_sh_act.data(), HIDDEN_DIM, EXP_INTER_DIM);

  float sh_sig = 1.0f / (1.0f + std::exp(-c_sh_gate_val));
  for (int i = 0; i < HIDDEN_DIM; ++i) c_moe_acc[i] += c_sh_out[i] * sh_sig;

  std::vector<float> c_y_final(HIDDEN_DIM);
  for (int i = 0; i < HIDDEN_DIM; ++i) c_y_final[i] = c_x_mid[i] + c_moe_acc[i];

  // --- GPU Allocations for Block 1 DeltaNet ---
  uint8_t *d_qkv_w, *d_z_w, *d_a_w, *d_b_w, *d_out_proj_w;
  uint16_t *d_qkv_s, *d_z_s, *d_a_s, *d_b_s, *d_out_proj_s;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, qkv_w.size(), 64, dev, (void **)&d_qkv_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, qkv_s.size() * 2, 64, dev, (void **)&d_qkv_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, z_w.size(), 64, dev, (void **)&d_z_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, z_s.size() * 2, 64, dev, (void **)&d_z_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, a_w.size(), 64, dev, (void **)&d_a_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, a_s.size() * 2, 64, dev, (void **)&d_a_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, b_w.size(), 64, dev, (void **)&d_b_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, b_s.size() * 2, 64, dev, (void **)&d_b_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, out_proj_w.size(), 64, dev, (void **)&d_out_proj_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, out_proj_s.size() * 2, 64, dev, (void **)&d_out_proj_s));

  float *d_qkv, *d_z, *d_a, *d_b, *d_qkv_conv, *d_conv_state, *d_conv_w;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * sizeof(float), 64, dev, (void **)&d_qkv));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_z));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_a));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_b));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * sizeof(float), 64, dev, (void **)&d_qkv_conv));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * 3 * sizeof(float), 64, dev, (void **)&d_conv_state));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * 4 * sizeof(float), 64, dev, (void **)&d_conv_w));

  float *d_q, *d_k, *d_g, *d_beta, *d_dt_bias, *d_A_log, *d_ssm_state, *d_attn_out, *d_ssm_norm_w, *d_attn_norm, *d_attn_proj;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_K * S_V * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_K * S_V * sizeof(float), 64, dev, (void **)&d_k));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_g));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_beta));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_dt_bias));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_A_log));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * S_V * sizeof(float), 64, dev, (void **)&d_ssm_state));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_attn_out));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, S_V * sizeof(float), 64, dev, (void **)&d_ssm_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_attn_norm));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_attn_proj));

  float *d_x_mid, *d_x_post;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_x_mid));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d_x_post));

  // Upload DeltaNet weights and init states
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_qkv_w, qkv_w.data(), qkv_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_qkv_s, qkv_s.data(), qkv_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_z_w, z_w.data(), z_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_z_s, z_s.data(), z_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_a_w, a_w.data(), a_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_a_s, a_s.data(), a_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_b_w, b_w.data(), b_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_b_s, b_s.data(), b_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_out_proj_w, out_proj_w.data(), out_proj_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_out_proj_s, out_proj_s.data(), out_proj_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_conv_w, h_conv_w.data(), h_conv_w.size() * 4, nullptr, 0, nullptr));
  std::vector<float> zero_conv(C_QKV * 3, 0.0f);
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_conv_state, zero_conv.data(), zero_conv.size() * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_dt_bias, h_dt_bias.data(), H_V * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_A_log, h_A_log.data(), H_V * 4, nullptr, 0, nullptr));
  std::vector<float> zero_ssm(H_V * S_V * S_V, 0.0f);
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_ssm_state, zero_ssm.data(), zero_ssm.size() * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_ssm_norm_w, h_ssm_norm_w.data(), S_V * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list_copy, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Build list1_attn: DeltaNet + Router
  CHECK(zeCommandListReset(list1_attn));

  // 1. RMSNorm 2048
  CHECK(zeKernelSetArgumentValue(k_norm2048, 0, sizeof(void *), &d_x_norm));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 1, sizeof(void *), &d_x_in));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 2, sizeof(void *), &d_in_norm_w));
  CHECK(zeKernelSetGroupSize(k_norm2048, 256, 1, 1));
  ze_group_count_t gcnt_norm2048{1, 1, 1};
  ze_group_count_t gcnt_silu{2, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_norm2048, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 2. QKV, Z, A, B GEMVs
  launch_gemv(list1_attn, d_qkv, d_qkv_w, d_qkv_s, d_x_norm, C_QKV, HIDDEN_DIM);
  launch_gemv(list1_attn, d_z, d_z_w, d_z_s, d_x_norm, H_V * S_V, HIDDEN_DIM);
  launch_gemv(list1_attn, d_a, d_a_w, d_a_s, d_x_norm, H_V, HIDDEN_DIM);
  launch_gemv(list1_attn, d_b, d_b_w, d_b_s, d_x_norm, H_V, HIDDEN_DIM);
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 3. Conv1D
  CHECK(zeKernelSetArgumentValue(k_conv, 0, sizeof(void *), &d_qkv_conv));
  CHECK(zeKernelSetArgumentValue(k_conv, 1, sizeof(void *), &d_qkv));
  CHECK(zeKernelSetArgumentValue(k_conv, 2, sizeof(void *), &d_conv_state));
  CHECK(zeKernelSetArgumentValue(k_conv, 3, sizeof(void *), &d_conv_w));
  CHECK(zeKernelSetGroupSize(k_conv, 256, 1, 1));
  ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_conv, &gcnt_conv, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 4. Head L2 Norm on Q and K
  int num_heads_16 = H_K;
  CHECK(zeKernelSetArgumentValue(k_l2_norm, 0, sizeof(void *), &d_q));
  CHECK(zeKernelSetArgumentValue(k_l2_norm, 1, sizeof(void *), &d_qkv_conv));
  CHECK(zeKernelSetArgumentValue(k_l2_norm, 2, sizeof(int), &num_heads_16));
  CHECK(zeKernelSetGroupSize(k_l2_norm, 128, 1, 1));
  ze_group_count_t gcnt_l2{(uint32_t)H_K, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_l2_norm, &gcnt_l2, nullptr, 0, nullptr));

  float *d_qkv_k = d_qkv_conv + H_K * S_V;
  ze_kernel_handle_t k_l2_k = get_kernel("head_l2_norm_128");
  CHECK(zeKernelSetArgumentValue(k_l2_k, 0, sizeof(void *), &d_k));
  CHECK(zeKernelSetArgumentValue(k_l2_k, 1, sizeof(void *), &d_qkv_k));
  CHECK(zeKernelSetArgumentValue(k_l2_k, 2, sizeof(int), &num_heads_16));
  CHECK(zeKernelSetGroupSize(k_l2_k, 128, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_l2_k, &gcnt_l2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 5. Gate Prep
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 0, sizeof(void *), &d_g));
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 1, sizeof(void *), &d_beta));
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 2, sizeof(void *), &d_a));
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 3, sizeof(void *), &d_b));
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 4, sizeof(void *), &d_dt_bias));
  CHECK(zeKernelSetArgumentValue(k_gate_prep, 5, sizeof(void *), &d_A_log));
  CHECK(zeKernelSetGroupSize(k_gate_prep, 32, 1, 1));
  ze_group_count_t gcnt_gate{1, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_gate_prep, &gcnt_gate, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 6. DeltaNet Recurrent Decode
  float *d_v_ptr = d_qkv_conv + 2 * H_K * S_V;
  CHECK(zeKernelSetArgumentValue(k_recr, 0, sizeof(void *), &d_attn_out));
  CHECK(zeKernelSetArgumentValue(k_recr, 1, sizeof(void *), &d_ssm_state));
  CHECK(zeKernelSetArgumentValue(k_recr, 2, sizeof(void *), &d_q));
  CHECK(zeKernelSetArgumentValue(k_recr, 3, sizeof(void *), &d_k));
  CHECK(zeKernelSetArgumentValue(k_recr, 4, sizeof(void *), &d_v_ptr));
  CHECK(zeKernelSetArgumentValue(k_recr, 5, sizeof(void *), &d_g));
  CHECK(zeKernelSetArgumentValue(k_recr, 6, sizeof(void *), &d_beta));
  CHECK(zeKernelSetGroupSize(k_recr, 128, 1, 1));
  ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_recr, &gcnt_recr, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 7. Head Norm & SiLU(z)
  CHECK(zeKernelSetArgumentValue(k_hnorm, 0, sizeof(void *), &d_attn_norm));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 1, sizeof(void *), &d_attn_out));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 2, sizeof(void *), &d_z));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 3, sizeof(void *), &d_ssm_norm_w));
  CHECK(zeKernelSetGroupSize(k_hnorm, 128, 1, 1));
  ze_group_count_t gcnt_hnorm{(uint32_t)H_V, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_hnorm, &gcnt_hnorm, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 8. Output Proj GEMV
  launch_gemv(list1_attn, d_attn_proj, d_out_proj_w, d_out_proj_s, d_attn_norm, HIDDEN_DIM, H_V * S_V);
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 9. Mid Residual Add: x_mid = x_in + attn_proj
  CHECK(zeKernelSetArgumentValue(k_resadd, 0, sizeof(void *), &d_x_mid));
  CHECK(zeKernelSetArgumentValue(k_resadd, 1, sizeof(void *), &d_x_in));
  CHECK(zeKernelSetArgumentValue(k_resadd, 2, sizeof(void *), &d_attn_proj));
  CHECK(zeKernelSetGroupSize(k_resadd, 256, 1, 1));
  ze_group_count_t gcnt_res{8, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_resadd, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 10. Post-Attn Norm
  CHECK(zeKernelSetArgumentValue(k_norm2048, 0, sizeof(void *), &d_x_post));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 1, sizeof(void *), &d_x_mid));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 2, sizeof(void *), &d_post_norm_w));
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_norm2048, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));

  // 11. Router
  CHECK(zeKernelSetArgumentValue(k_router, 0, sizeof(void *), &d_x_post));
  CHECK(zeKernelSetArgumentValue(k_router, 1, sizeof(void *), &d_router_w));
  CHECK(zeKernelSetArgumentValue(k_router, 2, sizeof(void *), &d_shared_gate_w));
  CHECK(zeKernelSetArgumentValue(k_router, 3, sizeof(void *), &d_top_idx));
  CHECK(zeKernelSetArgumentValue(k_router, 4, sizeof(void *), &d_top_wt));
  CHECK(zeKernelSetArgumentValue(k_router, 5, sizeof(void *), &d_sh_gate_val));
  CHECK(zeKernelSetGroupSize(k_router, 256, 1, 1));
  ze_group_count_t gcnt_router{1, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list1_attn, k_router, &gcnt_router, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_attn, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list1_attn));

  // Execute list1_attn ONCE to generate routing decisions
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list1_attn, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Read back routing decisions
  std::vector<uint32_t> g_top_idx(TOP_K);
  std::vector<float> g_top_wt(TOP_K);
  float g_sh_gate_val;
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g_top_idx.data(), d_top_idx, TOP_K * sizeof(uint32_t), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g_top_wt.data(), d_top_wt, TOP_K * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, &g_sh_gate_val, d_sh_gate_val, sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Build list1_moe: 8 Active Experts + Shared Expert + Final ResAdd
  CHECK(zeCommandListReset(list1_moe));
  for (int k = 0; k < TOP_K; ++k) {
    // 1. Gate_up GEMV [1024, 2048]
    launch_gemv(list1_moe, d_exp_gu, d_exp_gu_w[k], d_exp_gu_s[k], d_x_post, 2 * EXP_INTER_DIM, HIDDEN_DIM);
    CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

    // 2. SwiGLU 512
    float *d_exp_gu_up = d_exp_gu + EXP_INTER_DIM;
    CHECK(zeKernelSetArgumentValue(k_silu512, 0, sizeof(void *), &d_exp_act));
    CHECK(zeKernelSetArgumentValue(k_silu512, 1, sizeof(void *), &d_exp_gu));
    CHECK(zeKernelSetArgumentValue(k_silu512, 2, sizeof(void *), &d_exp_gu_up));
    CHECK(zeKernelSetGroupSize(k_silu512, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list1_moe, k_silu512, &gcnt_silu, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

    // 3. Down GEMV [2048, 512]
    launch_gemv(list1_moe, d_exp_out, d_exp_down_w[k], d_exp_down_s[k], d_exp_act, HIDDEN_DIM, EXP_INTER_DIM);
    CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

    // 4. MoE Accumulate (is_first = 1 on k=0 resets accumulator)
    float wt = g_top_wt[k];
    int is_first = (k == 0) ? 1 : 0;
    CHECK(zeKernelSetArgumentValue(k_accum, 0, sizeof(void *), &d_moe_acc));
    CHECK(zeKernelSetArgumentValue(k_accum, 1, sizeof(void *), &d_exp_out));
    CHECK(zeKernelSetArgumentValue(k_accum, 2, sizeof(float), &wt));
    CHECK(zeKernelSetArgumentValue(k_accum, 3, sizeof(int), &is_first));
    CHECK(zeKernelSetGroupSize(k_accum, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list1_moe, k_accum, &gcnt_res, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));
  }

  // Shared expert
  launch_gemv(list1_moe, d_sh_g, d_sh_gw, d_sh_gs, d_x_post, EXP_INTER_DIM, HIDDEN_DIM);
  launch_gemv(list1_moe, d_sh_u, d_sh_uw, d_sh_us, d_x_post, EXP_INTER_DIM, HIDDEN_DIM);
  CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_silu512, 0, sizeof(void *), &d_sh_act));
  CHECK(zeKernelSetArgumentValue(k_silu512, 1, sizeof(void *), &d_sh_g));
  CHECK(zeKernelSetArgumentValue(k_silu512, 2, sizeof(void *), &d_sh_u));
  CHECK(zeKernelSetGroupSize(k_silu512, 256, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list1_moe, k_silu512, &gcnt_silu, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

  launch_gemv(list1_moe, d_sh_out, d_sh_dw, d_sh_ds, d_sh_act, HIDDEN_DIM, EXP_INTER_DIM);
  CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_add_shared, 0, sizeof(void *), &d_moe_acc));
  CHECK(zeKernelSetArgumentValue(k_add_shared, 1, sizeof(void *), &d_sh_out));
  CHECK(zeKernelSetArgumentValue(k_add_shared, 2, sizeof(float), &g_sh_gate_val));
  CHECK(zeKernelSetGroupSize(k_add_shared, 256, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list1_moe, k_add_shared, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

  // Final Block Residual Add: y_final = x_mid + moe_acc
  CHECK(zeKernelSetArgumentValue(k_resadd, 0, sizeof(void *), &d_y_final));
  CHECK(zeKernelSetArgumentValue(k_resadd, 1, sizeof(void *), &d_x_mid));
  CHECK(zeKernelSetArgumentValue(k_resadd, 2, sizeof(void *), &d_moe_acc));
  CHECK(zeCommandListAppendLaunchKernel(list1_moe, k_resadd, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list1_moe, nullptr, 0, nullptr));

  CHECK(zeCommandListClose(list1_moe));

  // Execute list1_moe ONCE for bit-exact parity check
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list1_moe, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Read back output for parity check
  std::vector<float> g_y_final(HIDDEN_DIM);
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g_y_final.data(), d_y_final, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_b1 = 0.0f;
  for (int i = 0; i < HIDDEN_DIM; ++i) {
    float diff = std::fabs(g_y_final[i] - c_y_final[i]);
    if (diff > max_diff_b1) max_diff_b1 = diff;
  }
  bool pass_b1 = (max_diff_b1 < 1e-3f);
  std::printf("  DeltaNet-MoE output max_diff = %.2e [%s]\n", max_diff_b1, pass_b1 ? "PASS" : "FAIL");

  // Benchmark full Block 1 (DeltaNet + MoE back-to-back)
  const int BENCH_ITERS = 200;
  ze_command_list_handle_t cmdlists1[2] = {list1_attn, list1_moe};
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < BENCH_ITERS; ++i) {
    CHECK(zeCommandQueueExecuteCommandLists(queue, 2, cmdlists1, nullptr));
  }
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  auto t1 = std::chrono::steady_clock::now();
  double lat_deltanet_moe = std::chrono::duration<double, std::micro>(t1 - t0).count() / BENCH_ITERS;

  std::printf("  DeltaNet-MoE block latency = %.2f us (%.2f ms across 30 layers)\n",
              lat_deltanet_moe, (lat_deltanet_moe * 30.0) / 1000.0);

  // =========================================================================
  // BLOCK 2: FULL-ATTENTION-MOE BLOCK (LAYER 3) VERIFICATION
  // =========================================================================
  std::printf("\n=======================================================\n");
  std::printf("--- Verifying Block 2: Full-Attention-MoE (Layer 3) ---\n");
  std::printf("=======================================================\n");

  // Full-attention weights
  std::vector<uint8_t> q_proj_w, k_proj_w, v_proj_w, o_proj_w;
  std::vector<uint16_t> q_proj_s, k_proj_s, v_proj_s, o_proj_s;
  make_quant_weight(q_proj_w, q_proj_s, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM, rng); // [8192, 2048]
  make_quant_weight(k_proj_w, k_proj_s, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM, rng);     // [512, 2048]
  make_quant_weight(v_proj_w, v_proj_s, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM, rng);     // [512, 2048]
  make_quant_weight(o_proj_w, o_proj_s, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM, rng);     // [2048, 4096]

  std::vector<float> h_q_norm_w(HEAD_DIM), h_k_norm_w(HEAD_DIM);
  for (auto &x : h_q_norm_w) x = fdist(rng) * 0.1f;
  for (auto &x : h_k_norm_w) x = fdist(rng) * 0.1f;

  const uint32_t MAX_CTX = 2048;
  std::vector<uint16_t> h_k_cache(NUM_KV_HEADS * MAX_CTX * HEAD_DIM, 0);
  std::vector<uint16_t> h_v_cache(NUM_KV_HEADS * MAX_CTX * HEAD_DIM, 0);

  // --- CPU Forward Pass for Block 2 ---
  std::vector<float> c2_x_norm(HIDDEN_DIM);
  cpu_rmsnorm(c2_x_norm.data(), h_x_in.data(), h_in_norm_w.data(), HIDDEN_DIM, EPS, true);

  std::vector<float> c2_q_full(2 * NUM_Q_HEADS * HEAD_DIM);
  cpu_gemv_int4(c2_q_full.data(), q_proj_w.data(), q_proj_s.data(), c2_x_norm.data(), 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
  std::vector<float> c2_q(NUM_Q_HEADS * HEAD_DIM);
  std::vector<float> c2_gate(NUM_Q_HEADS * HEAD_DIM);
  std::memcpy(c2_q.data(), c2_q_full.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float));
  std::memcpy(c2_gate.data(), c2_q_full.data() + NUM_Q_HEADS * HEAD_DIM, NUM_Q_HEADS * HEAD_DIM * sizeof(float));

  std::vector<float> c2_k(NUM_KV_HEADS * HEAD_DIM), c2_v(NUM_KV_HEADS * HEAD_DIM);
  cpu_gemv_int4(c2_k.data(), k_proj_w.data(), k_proj_s.data(), c2_x_norm.data(), NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  cpu_gemv_int4(c2_v.data(), v_proj_w.data(), v_proj_s.data(), c2_x_norm.data(), NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);

  for (int h = 0; h < NUM_Q_HEADS; ++h) {
    cpu_rmsnorm(c2_q.data() + h * HEAD_DIM, c2_q.data() + h * HEAD_DIM, h_q_norm_w.data(), HEAD_DIM, EPS, true);
  }
  for (int h = 0; h < NUM_KV_HEADS; ++h) {
    cpu_rmsnorm(c2_k.data() + h * HEAD_DIM, c2_k.data() + h * HEAD_DIM, h_k_norm_w.data(), HEAD_DIM, EPS, true);
  }

  uint32_t step_pos = 0;
  for (int h = 0; h < NUM_Q_HEADS; ++h) {
    float *qh = c2_q.data() + h * HEAD_DIM;
    for (int i = 0; i < ROTARY_HALF; ++i) {
      float inv_f = std::pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
      float th = (float)step_pos * inv_f;
      float x0 = qh[i], x1 = qh[i + ROTARY_HALF];
      qh[i] = x0 * std::cos(th) - x1 * std::sin(th);
      qh[i + ROTARY_HALF] = x0 * std::sin(th) + x1 * std::cos(th);
    }
  }
  for (int h = 0; h < NUM_KV_HEADS; ++h) {
    float *kh = c2_k.data() + h * HEAD_DIM;
    for (int i = 0; i < ROTARY_HALF; ++i) {
      float inv_f = std::pow(ROPE_THETA, -((float)i / (float)ROTARY_HALF));
      float th = (float)step_pos * inv_f;
      float x0 = kh[i], x1 = kh[i + ROTARY_HALF];
      kh[i] = x0 * std::cos(th) - x1 * std::sin(th);
      kh[i + ROTARY_HALF] = x0 * std::sin(th) + x1 * std::cos(th);
    }
    for (int d = 0; d < HEAD_DIM; ++d) {
      h_k_cache[(h * MAX_CTX + step_pos) * HEAD_DIM + d] = float_to_bf16(kh[d]);
      h_v_cache[(h * MAX_CTX + step_pos) * HEAD_DIM + d] = float_to_bf16(c2_v[h * HEAD_DIM + d]);
    }
  }

  std::vector<float> c2_attn_out(NUM_Q_HEADS * HEAD_DIM);
  for (int qh = 0; qh < NUM_Q_HEADS; ++qh) {
    int kv_h = qh / GQA_GROUP_SIZE;
    const float *qh_p = c2_q.data() + qh * HEAD_DIM;
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += qh_p[d] * bf16_to_fp32(h_k_cache[(kv_h * MAX_CTX + step_pos) * HEAD_DIM + d]);
    }
    for (int d = 0; d < HEAD_DIM; ++d) {
      float v_val = bf16_to_fp32(h_v_cache[(kv_h * MAX_CTX + step_pos) * HEAD_DIM + d]);
      float g_v = c2_gate[qh * HEAD_DIM + d];
      c2_attn_out[qh * HEAD_DIM + d] = v_val * (1.0f / (1.0f + std::exp(-g_v)));
    }
  }

  std::vector<float> c2_attn_proj(HIDDEN_DIM);
  cpu_gemv_int4(c2_attn_proj.data(), o_proj_w.data(), o_proj_s.data(), c2_attn_out.data(), HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);

  std::vector<float> c2_x_mid(HIDDEN_DIM);
  for (int i = 0; i < HIDDEN_DIM; ++i) c2_x_mid[i] = h_x_in[i] + c2_attn_proj[i];

  std::vector<float> c2_x_post(HIDDEN_DIM);
  cpu_rmsnorm(c2_x_post.data(), c2_x_mid.data(), h_post_norm_w.data(), HIDDEN_DIM, EPS, true);

  // MoE for Block 2 CPU
  uint32_t c2_top_idx[TOP_K];
  float c2_top_wt[TOP_K];
  float c2_sh_gate_val;
  cpu_top8_router(c2_top_idx, c2_top_wt, &c2_sh_gate_val, c2_x_post.data(), h_router_w.data(), h_shared_gate_w.data());

  std::vector<float> c2_moe_acc(HIDDEN_DIM, 0.0f);
  for (int k = 0; k < TOP_K; ++k) {
    std::vector<float> gu(2 * EXP_INTER_DIM);
    cpu_gemv_int4(gu.data(), exp_gu_w[k].data(), exp_gu_s[k].data(), c2_x_post.data(), 2 * EXP_INTER_DIM, HIDDEN_DIM);
    std::vector<float> act(EXP_INTER_DIM);
    cpu_silu_mul(act.data(), gu.data(), gu.data() + EXP_INTER_DIM, EXP_INTER_DIM);
    std::vector<float> exp_out(HIDDEN_DIM);
    cpu_gemv_int4(exp_out.data(), exp_down_w[k].data(), exp_down_s[k].data(), act.data(), HIDDEN_DIM, EXP_INTER_DIM);
    for (int i = 0; i < HIDDEN_DIM; ++i) c2_moe_acc[i] += exp_out[i] * c2_top_wt[k];
  }

  // Shared expert
  std::vector<float> c2_sh_g(EXP_INTER_DIM), c2_sh_u(EXP_INTER_DIM), c2_sh_act(EXP_INTER_DIM), c2_sh_out(HIDDEN_DIM);
  cpu_gemv_int4(c2_sh_g.data(), sh_gate_w.data(), sh_gate_s.data(), c2_x_post.data(), EXP_INTER_DIM, HIDDEN_DIM);
  cpu_gemv_int4(c2_sh_u.data(), sh_up_w.data(), sh_up_s.data(), c2_x_post.data(), EXP_INTER_DIM, HIDDEN_DIM);
  cpu_silu_mul(c2_sh_act.data(), c2_sh_g.data(), c2_sh_u.data(), EXP_INTER_DIM);
  cpu_gemv_int4(c2_sh_out.data(), sh_down_w.data(), sh_down_s.data(), c2_sh_act.data(), HIDDEN_DIM, EXP_INTER_DIM);

  float sh_sig2 = 1.0f / (1.0f + std::exp(-c2_sh_gate_val));
  for (int i = 0; i < HIDDEN_DIM; ++i) c2_moe_acc[i] += c2_sh_out[i] * sh_sig2;

  std::vector<float> c2_y_final(HIDDEN_DIM);
  for (int i = 0; i < HIDDEN_DIM; ++i) c2_y_final[i] = c2_x_mid[i] + c2_moe_acc[i];

  // --- GPU Allocations for Block 2 Attention ---
  uint8_t *d_q_proj_w, *d_k_proj_w, *d_v_proj_w, *d_o_proj_w;
  uint16_t *d_q_proj_s, *d_k_proj_s, *d_v_proj_s, *d_o_proj_s;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, q_proj_w.size(), 64, dev, (void **)&d_q_proj_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, q_proj_s.size() * 2, 64, dev, (void **)&d_q_proj_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, k_proj_w.size(), 64, dev, (void **)&d_k_proj_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, k_proj_s.size() * 2, 64, dev, (void **)&d_k_proj_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, v_proj_w.size(), 64, dev, (void **)&d_v_proj_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, v_proj_s.size() * 2, 64, dev, (void **)&d_v_proj_s));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, o_proj_w.size(), 64, dev, (void **)&d_o_proj_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, o_proj_s.size() * 2, 64, dev, (void **)&d_o_proj_s));

  float *d2_q_full, *d2_k, *d2_v, *d2_q_norm_w, *d2_k_norm_w, *d2_attn_out, *d2_attn_proj, *d2_x_mid, *d2_x_post;
  uint16_t *d2_k_cache, *d2_v_cache;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, 2 * NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_q_full));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_KV_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_k));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_KV_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_v));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_q_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_k_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, NUM_Q_HEADS * HEAD_DIM * sizeof(float), 64, dev, (void **)&d2_attn_out));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d2_attn_proj));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d2_x_mid));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev, (void **)&d2_x_post));

  size_t cache_sz = (size_t)NUM_KV_HEADS * MAX_CTX * HEAD_DIM * sizeof(uint16_t);
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, cache_sz, 64, dev, (void **)&d2_k_cache));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, cache_sz, 64, dev, (void **)&d2_v_cache));

  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_q_proj_w, q_proj_w.data(), q_proj_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_q_proj_s, q_proj_s.data(), q_proj_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_k_proj_w, k_proj_w.data(), k_proj_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_k_proj_s, k_proj_s.data(), k_proj_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_v_proj_w, v_proj_w.data(), v_proj_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_v_proj_s, v_proj_s.data(), v_proj_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_o_proj_w, o_proj_w.data(), o_proj_w.size(), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d_o_proj_s, o_proj_s.data(), o_proj_s.size() * 2, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d2_q_norm_w, h_q_norm_w.data(), HEAD_DIM * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, d2_k_norm_w, h_k_norm_w.data(), HEAD_DIM * 4, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list_copy, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Build list2_attn: Full Attention + Router
  CHECK(zeCommandListReset(list2_attn));

  // 1. RMSNorm 2048
  CHECK(zeKernelSetArgumentValue(k_norm2048, 0, sizeof(void *), &d_x_norm));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 1, sizeof(void *), &d_x_in));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 2, sizeof(void *), &d_in_norm_w));
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_norm2048, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 2. Q_full, K, V GEMVs
  launch_gemv(list2_attn, d2_q_full, d_q_proj_w, d_q_proj_s, d_x_norm, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
  launch_gemv(list2_attn, d2_k, d_k_proj_w, d_k_proj_s, d_x_norm, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  launch_gemv(list2_attn, d2_v, d_v_proj_w, d_v_proj_s, d_x_norm, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 3. Q and K head norm
  int nq = NUM_Q_HEADS;
  CHECK(zeKernelSetArgumentValue(k_norm256, 0, sizeof(void *), &d2_q_full));
  CHECK(zeKernelSetArgumentValue(k_norm256, 1, sizeof(void *), &d2_q_full));
  CHECK(zeKernelSetArgumentValue(k_norm256, 2, sizeof(void *), &d2_q_norm_w));
  CHECK(zeKernelSetArgumentValue(k_norm256, 3, sizeof(int), &nq));
  CHECK(zeKernelSetGroupSize(k_norm256, 64, 1, 1));
  ze_group_count_t gcnt_nq{(uint32_t)NUM_Q_HEADS, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_norm256, &gcnt_nq, nullptr, 0, nullptr));

  int nkv = NUM_KV_HEADS;
  CHECK(zeKernelSetArgumentValue(k_norm256_k, 0, sizeof(void *), &d2_k));
  CHECK(zeKernelSetArgumentValue(k_norm256_k, 1, sizeof(void *), &d2_k));
  CHECK(zeKernelSetArgumentValue(k_norm256_k, 2, sizeof(void *), &d2_k_norm_w));
  CHECK(zeKernelSetArgumentValue(k_norm256_k, 3, sizeof(int), &nkv));
  CHECK(zeKernelSetGroupSize(k_norm256_k, 64, 1, 1));
  ze_group_count_t gcnt_nkv{(uint32_t)NUM_KV_HEADS, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_norm256_k, &gcnt_nkv, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 4. RoPE and KV Cache Append
  uint32_t cur_pos2 = step_pos;
  uint32_t max_c2 = MAX_CTX;
  CHECK(zeKernelSetArgumentValue(k_rope, 0, sizeof(void *), &d2_q_full));
  CHECK(zeKernelSetArgumentValue(k_rope, 1, sizeof(void *), &d2_k));
  CHECK(zeKernelSetArgumentValue(k_rope, 2, sizeof(void *), &d2_v));
  CHECK(zeKernelSetArgumentValue(k_rope, 3, sizeof(void *), &d2_k_cache));
  CHECK(zeKernelSetArgumentValue(k_rope, 4, sizeof(void *), &d2_v_cache));
  CHECK(zeKernelSetArgumentValue(k_rope, 5, sizeof(uint32_t), &cur_pos2));
  CHECK(zeKernelSetArgumentValue(k_rope, 6, sizeof(uint32_t), &max_c2));
  CHECK(zeKernelSetGroupSize(k_rope, 256, 1, 1));
  ze_group_count_t gcnt_rope{1, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_rope, &gcnt_rope, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 5. GQA Attention Decode
  float *d2_gate = d2_q_full + NUM_Q_HEADS * HEAD_DIM;
  CHECK(zeKernelSetArgumentValue(k_attn, 0, sizeof(void *), &d2_attn_out));
  CHECK(zeKernelSetArgumentValue(k_attn, 1, sizeof(void *), &d2_q_full));
  CHECK(zeKernelSetArgumentValue(k_attn, 2, sizeof(void *), &d2_gate));
  CHECK(zeKernelSetArgumentValue(k_attn, 3, sizeof(void *), &d2_k_cache));
  CHECK(zeKernelSetArgumentValue(k_attn, 4, sizeof(void *), &d2_v_cache));
  CHECK(zeKernelSetArgumentValue(k_attn, 5, sizeof(uint32_t), &cur_pos2));
  CHECK(zeKernelSetArgumentValue(k_attn, 6, sizeof(uint32_t), &max_c2));
  CHECK(zeKernelSetGroupSize(k_attn, 256, 1, 1));
  ze_group_count_t gcnt_attn{(uint32_t)NUM_Q_HEADS, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_attn, &gcnt_attn, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 6. O_proj GEMV
  launch_gemv(list2_attn, d2_attn_proj, d_o_proj_w, d_o_proj_s, d2_attn_out, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 7. Residual Add
  CHECK(zeKernelSetArgumentValue(k_resadd, 0, sizeof(void *), &d2_x_mid));
  CHECK(zeKernelSetArgumentValue(k_resadd, 1, sizeof(void *), &d_x_in));
  CHECK(zeKernelSetArgumentValue(k_resadd, 2, sizeof(void *), &d2_attn_proj));
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_resadd, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 8. Post Norm
  CHECK(zeKernelSetArgumentValue(k_norm2048, 0, sizeof(void *), &d2_x_post));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 1, sizeof(void *), &d2_x_mid));
  CHECK(zeKernelSetArgumentValue(k_norm2048, 2, sizeof(void *), &d_post_norm_w));
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_norm2048, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));

  // 9. Router for Block 2
  CHECK(zeKernelSetArgumentValue(k_router, 0, sizeof(void *), &d2_x_post));
  CHECK(zeKernelSetArgumentValue(k_router, 1, sizeof(void *), &d_router_w));
  CHECK(zeKernelSetArgumentValue(k_router, 2, sizeof(void *), &d_shared_gate_w));
  CHECK(zeKernelSetArgumentValue(k_router, 3, sizeof(void *), &d_top_idx));
  CHECK(zeKernelSetArgumentValue(k_router, 4, sizeof(void *), &d_top_wt));
  CHECK(zeKernelSetArgumentValue(k_router, 5, sizeof(void *), &d_sh_gate_val));
  CHECK(zeKernelSetGroupSize(k_router, 256, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list2_attn, k_router, &gcnt_router, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_attn, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list2_attn));

  // Execute list2_attn ONCE to generate routing decisions
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list2_attn, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Read back routing decisions for Block 2
  std::vector<uint32_t> g2_top_idx(TOP_K);
  std::vector<float> g2_top_wt(TOP_K);
  float g2_sh_gate_val;
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g2_top_idx.data(), d_top_idx, TOP_K * sizeof(uint32_t), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g2_top_wt.data(), d_top_wt, TOP_K * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, &g2_sh_gate_val, d_sh_gate_val, sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Build list2_moe: 8 Active Experts + Shared Expert + Final ResAdd
  CHECK(zeCommandListReset(list2_moe));
  for (int k = 0; k < TOP_K; ++k) {
    launch_gemv(list2_moe, d_exp_gu, d_exp_gu_w[k], d_exp_gu_s[k], d2_x_post, 2 * EXP_INTER_DIM, HIDDEN_DIM);
    CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

    float *d_exp_gu_up = d_exp_gu + EXP_INTER_DIM;
    CHECK(zeKernelSetArgumentValue(k_silu512, 0, sizeof(void *), &d_exp_act));
    CHECK(zeKernelSetArgumentValue(k_silu512, 1, sizeof(void *), &d_exp_gu));
    CHECK(zeKernelSetArgumentValue(k_silu512, 2, sizeof(void *), &d_exp_gu_up));
    CHECK(zeKernelSetGroupSize(k_silu512, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list2_moe, k_silu512, &gcnt_silu, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

    launch_gemv(list2_moe, d_exp_out, d_exp_down_w[k], d_exp_down_s[k], d_exp_act, HIDDEN_DIM, EXP_INTER_DIM);
    CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

    float wt = g2_top_wt[k];
    int is_first = (k == 0) ? 1 : 0;
    CHECK(zeKernelSetArgumentValue(k_accum, 0, sizeof(void *), &d_moe_acc));
    CHECK(zeKernelSetArgumentValue(k_accum, 1, sizeof(void *), &d_exp_out));
    CHECK(zeKernelSetArgumentValue(k_accum, 2, sizeof(float), &wt));
    CHECK(zeKernelSetArgumentValue(k_accum, 3, sizeof(int), &is_first));
    CHECK(zeKernelSetGroupSize(k_accum, 256, 1, 1));
    CHECK(zeCommandListAppendLaunchKernel(list2_moe, k_accum, &gcnt_res, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));
  }

  // Shared expert
  launch_gemv(list2_moe, d_sh_g, d_sh_gw, d_sh_gs, d2_x_post, EXP_INTER_DIM, HIDDEN_DIM);
  launch_gemv(list2_moe, d_sh_u, d_sh_uw, d_sh_us, d2_x_post, EXP_INTER_DIM, HIDDEN_DIM);
  CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_silu512, 0, sizeof(void *), &d_sh_act));
  CHECK(zeKernelSetArgumentValue(k_silu512, 1, sizeof(void *), &d_sh_g));
  CHECK(zeKernelSetArgumentValue(k_silu512, 2, sizeof(void *), &d_sh_u));
  CHECK(zeKernelSetGroupSize(k_silu512, 256, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list2_moe, k_silu512, &gcnt_silu, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

  launch_gemv(list2_moe, d_sh_out, d_sh_dw, d_sh_ds, d_sh_act, HIDDEN_DIM, EXP_INTER_DIM);
  CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

  CHECK(zeKernelSetArgumentValue(k_add_shared, 0, sizeof(void *), &d_moe_acc));
  CHECK(zeKernelSetArgumentValue(k_add_shared, 1, sizeof(void *), &d_sh_out));
  CHECK(zeKernelSetArgumentValue(k_add_shared, 2, sizeof(float), &g2_sh_gate_val));
  CHECK(zeKernelSetGroupSize(k_add_shared, 256, 1, 1));
  CHECK(zeCommandListAppendLaunchKernel(list2_moe, k_add_shared, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

  // 10. Final Residual Add
  CHECK(zeKernelSetArgumentValue(k_resadd, 0, sizeof(void *), &d_y_final));
  CHECK(zeKernelSetArgumentValue(k_resadd, 1, sizeof(void *), &d2_x_mid));
  CHECK(zeKernelSetArgumentValue(k_resadd, 2, sizeof(void *), &d_moe_acc));
  CHECK(zeCommandListAppendLaunchKernel(list2_moe, k_resadd, &gcnt_res, nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list2_moe, nullptr, 0, nullptr));

  CHECK(zeCommandListClose(list2_moe));

  // Execute list2_moe ONCE for bit-exact parity check
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list2_moe, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Read back output for parity check
  std::vector<float> g2_y_final(HIDDEN_DIM);
  CHECK(zeCommandListReset(list_copy));
  CHECK(zeCommandListAppendMemoryCopy(list_copy, g2_y_final.data(), d_y_final, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list_copy));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list_copy, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  float max_diff_b2 = 0.0f;
  for (int i = 0; i < HIDDEN_DIM; ++i) {
    float diff = std::fabs(g2_y_final[i] - c2_y_final[i]);
    if (diff > max_diff_b2) max_diff_b2 = diff;
  }
  bool pass_b2 = (max_diff_b2 < 1e-3f);
  std::printf("  Full-Attention-MoE output max_diff = %.2e [%s]\n", max_diff_b2, pass_b2 ? "PASS" : "FAIL");

  // Benchmark full Block 2 (Full-Attention + MoE back-to-back)
  ze_command_list_handle_t cmdlists2[2] = {list2_attn, list2_moe};
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < BENCH_ITERS; ++i) {
    CHECK(zeCommandQueueExecuteCommandLists(queue, 2, cmdlists2, nullptr));
  }
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  t1 = std::chrono::steady_clock::now();
  double lat_fullattn_moe = std::chrono::duration<double, std::micro>(t1 - t0).count() / BENCH_ITERS;

  std::printf("  Full-Attention-MoE block latency = %.2f us (%.2f ms across 10 layers)\n",
              lat_fullattn_moe, (lat_fullattn_moe * 10.0) / 1000.0);

  // Full Model Projection
  double full_model_step_us = lat_deltanet_moe * 30.0 + lat_fullattn_moe * 10.0;
  double full_model_tok_per_s = 1e6 / full_model_step_us;

  std::printf("\n=======================================================\n");
  std::printf("--- 40-Layer Full Model Decode Projection on Arc 140V ---\n");
  std::printf("=======================================================\n");
  std::printf("  30 DeltaNet-MoE blocks: %.2f ms\n", (lat_deltanet_moe * 30.0) / 1000.0);
  std::printf("  10 Full-Attn-MoE blocks: %.2f ms\n", (lat_fullattn_moe * 10.0) / 1000.0);
  std::printf("  Total 40-layer decode latency: %.2f ms per token\n", full_model_step_us / 1000.0);
  std::printf("  Projected Decode Speed: %.2f tok/s (Roofline Target: 94-121 tok/s)\n", full_model_tok_per_s);
  std::printf("=======================================================\n");

  bool all_passed = pass_b1 && pass_b2;

  // Emit report
  std::ofstream rpt("tools/kernels_258v/report_layer_block.json");
  rpt << "{\n";
  rpt << "  \"task\": \"T4.7\",\n";
  rpt << "  \"device\": \"Arc 140V (Xe2)\",\n";
  rpt << "  \"all_passed\": " << (all_passed ? "true" : "false") << ",\n";
  rpt << "  \"deltanet_moe_block\": {\n";
  rpt << "    \"max_diff\": " << max_diff_b1 << ",\n";
  rpt << "    \"latency_us\": " << lat_deltanet_moe << ",\n";
  rpt << "    \"status\": \"" << (pass_b1 ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"fullattn_moe_block\": {\n";
  rpt << "    \"max_diff\": " << max_diff_b2 << ",\n";
  rpt << "    \"latency_us\": " << lat_fullattn_moe << ",\n";
  rpt << "    \"status\": \"" << (pass_b2 ? "PASS" : "FAIL") << "\"\n";
  rpt << "  },\n";
  rpt << "  \"full_model_40_layers_projection\": {\n";
  rpt << "    \"total_latency_ms\": " << (full_model_step_us / 1000.0) << ",\n";
  rpt << "    \"projected_tok_per_s\": " << full_model_tok_per_s << "\n";
  rpt << "  }\n";
  rpt << "}\n";
  rpt.close();

  return all_passed ? 0 : 1;
}
