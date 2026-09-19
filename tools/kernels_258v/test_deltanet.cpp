// AInfer DeltaNet Operators Verification & Benchmark on Arc 140V (T4.4)
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
const float SCALE_128 = 0.08838834764831845f;
const float EPS = 1e-6f;

// CPU Reference Implementations
void cpu_conv1d_update_silu(float *qkv_out, const float *qkv_in, float *conv_state,
                            const float *conv_weight) {
  for (int c = 0; c < C_QKV; ++c) {
    float s0 = conv_state[c * 3 + 0];
    float s1 = conv_state[c * 3 + 1];
    float s2 = conv_state[c * 3 + 2];
    float x  = qkv_in[c];

    float w0 = conv_weight[c * 4 + 0];
    float w1 = conv_weight[c * 4 + 1];
    float w2 = conv_weight[c * 4 + 2];
    float w3 = conv_weight[c * 4 + 3];

    float sum = s0 * w0 + s1 * w1 + s2 * w2 + x * w3;
    float silu_val = sum / (1.0f + std::exp(-sum));
    qkv_out[c] = silu_val;

    conv_state[c * 3 + 0] = s1;
    conv_state[c * 3 + 1] = s2;
    conv_state[c * 3 + 2] = x;
  }
}

void cpu_head_l2_norm_128(float *y, const float *x, int num_heads) {
  for (int h = 0; h < num_heads; ++h) {
    const float *x_head = x + h * S_V;
    float *y_head = y + h * S_V;
    float ss = 0.0f;
    for (int i = 0; i < S_V; ++i) {
      ss += x_head[i] * x_head[i];
    }
    float inv_l2 = 1.0f / std::max(std::sqrt(ss), EPS);
    for (int i = 0; i < S_V; ++i) {
      y_head[i] = x_head[i] * inv_l2;
    }
  }
}

void cpu_deltanet_gate_prep(float *g_out, float *beta_out, const float *a_in,
                            const float *b_in, const float *dt_bias,
                            const float *A_log) {
  for (int h = 0; h < H_V; ++h) {
    float a_val = a_in[h] + dt_bias[h];
    float dt = (a_val > 20.0f) ? a_val : std::log(1.0f + std::exp(a_val));
    float gate = -std::exp(A_log[h]) * dt;
    g_out[h] = std::exp(gate);
    beta_out[h] = 1.0f / (1.0f + std::exp(-b_in[h]));
  }
}

void cpu_deltanet_recurrent_decode(float *out, float *state, const float *q,
                                   const float *k, const float *v,
                                   const float *g, const float *beta) {
  for (int h = 0; h < H_V; ++h) {
    int kh = h / 2;
    const float *q_h = q + kh * S_V;
    const float *k_h = k + kh * S_V;
    const float *v_h = v + h * S_V;
    float g_val = g[h];
    float b_val = beta[h];

    float *S_h = state + h * (S_V * S_V);

    for (int j = 0; j < S_V; ++j) {
      float kv_acc = 0.0f;
      for (int i = 0; i < S_V; ++i) {
        kv_acc += S_h[i * S_V + j] * k_h[i];
      }
      float kv_j = kv_acc * g_val;
      float delta_j = (v_h[j] - kv_j) * b_val;

      float o_acc = 0.0f;
      for (int i = 0; i < S_V; ++i) {
        float s_old = S_h[i * S_V + j];
        float s_new = g_val * s_old + k_h[i] * delta_j;
        S_h[i * S_V + j] = s_new;
        o_acc += s_new * q_h[i];
      }
      out[h * S_V + j] = o_acc * SCALE_128;
    }
  }
}

void cpu_deltanet_head_norm_silu_z(float *final_out, const float *attn_out,
                                   const float *z_gate, const float *ssm_norm_w) {
  for (int h = 0; h < H_V; ++h) {
    const float *x_head = attn_out + h * S_V;
    const float *z_head = z_gate + h * S_V;
    float *y_head = final_out + h * S_V;

    float ss = 0.0f;
    for (int j = 0; j < S_V; ++j) {
      ss += x_head[j] * x_head[j];
    }
    float inv_rms = 1.0f / std::sqrt(ss / (float)S_V + EPS);

    for (int j = 0; j < S_V; ++j) {
      float norm_val = x_head[j] * inv_rms * ssm_norm_w[j];
      float zv = z_head[j];
      float silu_z = zv / (1.0f + std::exp(-zv));
      y_head[j] = norm_val * silu_z;
    }
  }
}

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "tools/kernels_258v/deltanet.spv";
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

  // Create kernels
  ze_kernel_desc_t k_conv_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "conv1d_update_silu"};
  ze_kernel_handle_t k_conv;
  CHECK(zeKernelCreate(mod, &k_conv_desc, &k_conv));

  ze_kernel_desc_t k_l2_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "head_l2_norm_128"};
  ze_kernel_handle_t k_l2;
  CHECK(zeKernelCreate(mod, &k_l2_desc, &k_l2));

  ze_kernel_desc_t k_gate_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "deltanet_gate_prep"};
  ze_kernel_handle_t k_gate;
  CHECK(zeKernelCreate(mod, &k_gate_desc, &k_gate));

  ze_kernel_desc_t k_recr_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "deltanet_recurrent_decode"};
  ze_kernel_handle_t k_recr;
  CHECK(zeKernelCreate(mod, &k_recr_desc, &k_recr));

  ze_kernel_desc_t k_hnorm_desc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "deltanet_head_norm_silu_z"};
  ze_kernel_handle_t k_hnorm;
  CHECK(zeKernelCreate(mod, &k_hnorm_desc, &k_hnorm));

  ze_device_mem_alloc_desc_t dmem_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

  // Allocate host buffers
  std::vector<float> h_qkv_in(C_QKV), h_qkv_conv_ref(C_QKV), h_qkv_conv_gpu(C_QKV);
  std::vector<float> h_conv_state_ref(C_QKV * 3, 0.0f), h_conv_state_gpu(C_QKV * 3, 0.0f);
  std::vector<float> h_conv_w(C_QKV * 4);
  for (int i = 0; i < C_QKV * 4; ++i) h_conv_w[i] = fdist(rng) * 0.1f;

  std::vector<float> h_a(H_V), h_b(H_V), h_dt_bias(H_V), h_A_log(H_V);
  std::vector<float> h_g_ref(H_V), h_beta_ref(H_V), h_g_gpu(H_V), h_beta_gpu(H_V);
  for (int i = 0; i < H_V; ++i) {
    h_a[i] = fdist(rng);
    h_b[i] = fdist(rng);
    h_dt_bias[i] = fdist(rng) * 0.5f;
    h_A_log[i] = -fdist(rng) * 0.5f - 1.0f;
  }

  const int STATE_ELEMS = H_V * S_V * S_V; // 32 * 128 * 128 = 524,288 floats (2 MiB)
  std::vector<float> h_state_ref(STATE_ELEMS, 0.0f), h_state_gpu(STATE_ELEMS, 0.0f);

  std::vector<float> h_attn_out_ref(H_V * S_V), h_attn_out_gpu(H_V * S_V);
  std::vector<float> h_z(H_V * S_V), h_ssm_norm_w(S_V);
  std::vector<float> h_final_ref(H_V * S_V), h_final_gpu(H_V * S_V);
  for (int i = 0; i < H_V * S_V; ++i) h_z[i] = fdist(rng);
  for (int i = 0; i < S_V; ++i) h_ssm_norm_w[i] = fdist(rng) * 0.2f + 0.8f; // ~0.8-1.0

  // Allocate device buffers
  float *d_qkv_in, *d_qkv_conv, *d_conv_state, *d_conv_w;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * sizeof(float), 64, dev, (void **)&d_qkv_in));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * sizeof(float), 64, dev, (void **)&d_qkv_conv));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * 3 * sizeof(float), 64, dev, (void **)&d_conv_state));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, C_QKV * 4 * sizeof(float), 64, dev, (void **)&d_conv_w));

  float *d_q, *d_k, *d_v;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_K * S_V * sizeof(float), 64, dev, (void **)&d_q));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_K * S_V * sizeof(float), 64, dev, (void **)&d_k));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_v));

  float *d_a, *d_b, *d_dt_bias, *d_A_log, *d_g, *d_beta;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_a));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_b));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_dt_bias));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_A_log));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_g));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * sizeof(float), 64, dev, (void **)&d_beta));

  float *d_state, *d_attn_out, *d_z, *d_ssm_norm_w, *d_final_out;
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, STATE_ELEMS * sizeof(float), 64, dev, (void **)&d_state));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_attn_out));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_z));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, S_V * sizeof(float), 64, dev, (void **)&d_ssm_norm_w));
  CHECK(zeMemAllocDevice(ctx, &dmem_desc, H_V * S_V * sizeof(float), 64, dev, (void **)&d_final_out));

  // Initialize device memory
  CHECK(zeCommandListReset(list));
  CHECK(zeCommandListAppendMemoryCopy(list, d_conv_state, h_conv_state_ref.data(), C_QKV * 3 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_conv_w, h_conv_w.data(), C_QKV * 4 * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_dt_bias, h_dt_bias.data(), H_V * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_A_log, h_A_log.data(), H_V * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_state, h_state_ref.data(), STATE_ELEMS * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendMemoryCopy(list, d_ssm_norm_w, h_ssm_norm_w.data(), S_V * sizeof(float), nullptr, 0, nullptr));
  CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

  // Configure kernel parameters
  CHECK(zeKernelSetArgumentValue(k_conv, 0, sizeof(void *), &d_qkv_conv));
  CHECK(zeKernelSetArgumentValue(k_conv, 1, sizeof(void *), &d_qkv_in));
  CHECK(zeKernelSetArgumentValue(k_conv, 2, sizeof(void *), &d_conv_state));
  CHECK(zeKernelSetArgumentValue(k_conv, 3, sizeof(void *), &d_conv_w));
  CHECK(zeKernelSetGroupSize(k_conv, 256, 1, 1));
  ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};

  int num_heads_q = H_K;
  CHECK(zeKernelSetArgumentValue(k_l2, 0, sizeof(void *), &d_q));
  CHECK(zeKernelSetArgumentValue(k_l2, 1, sizeof(void *), &d_qkv_conv)); // reads from first 2048
  CHECK(zeKernelSetArgumentValue(k_l2, 2, sizeof(int), &num_heads_q));
  CHECK(zeKernelSetGroupSize(k_l2, 128, 1, 1));
  ze_group_count_t gcnt_l2{(uint32_t)H_K, 1, 1};

  // For K: offset pointer by 2048 floats
  float *d_qkv_conv_k = d_qkv_conv + H_K * S_V;
  ze_kernel_handle_t k_l2_k;
  CHECK(zeKernelCreate(mod, &k_l2_desc, &k_l2_k));
  CHECK(zeKernelSetArgumentValue(k_l2_k, 0, sizeof(void *), &d_k));
  CHECK(zeKernelSetArgumentValue(k_l2_k, 1, sizeof(void *), &d_qkv_conv_k));
  CHECK(zeKernelSetArgumentValue(k_l2_k, 2, sizeof(int), &num_heads_q));
  CHECK(zeKernelSetGroupSize(k_l2_k, 128, 1, 1));

  CHECK(zeKernelSetArgumentValue(k_gate, 0, sizeof(void *), &d_g));
  CHECK(zeKernelSetArgumentValue(k_gate, 1, sizeof(void *), &d_beta));
  CHECK(zeKernelSetArgumentValue(k_gate, 2, sizeof(void *), &d_a));
  CHECK(zeKernelSetArgumentValue(k_gate, 3, sizeof(void *), &d_b));
  CHECK(zeKernelSetArgumentValue(k_gate, 4, sizeof(void *), &d_dt_bias));
  CHECK(zeKernelSetArgumentValue(k_gate, 5, sizeof(void *), &d_A_log));
  CHECK(zeKernelSetGroupSize(k_gate, 32, 1, 1));
  ze_group_count_t gcnt_gate{1, 1, 1};

  // V is the remaining 4096 elements of qkv_conv
  float *d_qkv_conv_v = d_qkv_conv + 2 * H_K * S_V;
  CHECK(zeKernelSetArgumentValue(k_recr, 0, sizeof(void *), &d_attn_out));
  CHECK(zeKernelSetArgumentValue(k_recr, 1, sizeof(void *), &d_state));
  CHECK(zeKernelSetArgumentValue(k_recr, 2, sizeof(void *), &d_q));
  CHECK(zeKernelSetArgumentValue(k_recr, 3, sizeof(void *), &d_k));
  CHECK(zeKernelSetArgumentValue(k_recr, 4, sizeof(void *), &d_qkv_conv_v));
  CHECK(zeKernelSetArgumentValue(k_recr, 5, sizeof(void *), &d_g));
  CHECK(zeKernelSetArgumentValue(k_recr, 6, sizeof(void *), &d_beta));
  CHECK(zeKernelSetGroupSize(k_recr, 128, 1, 1));
  ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};

  CHECK(zeKernelSetArgumentValue(k_hnorm, 0, sizeof(void *), &d_final_out));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 1, sizeof(void *), &d_attn_out));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 2, sizeof(void *), &d_z));
  CHECK(zeKernelSetArgumentValue(k_hnorm, 3, sizeof(void *), &d_ssm_norm_w));
  CHECK(zeKernelSetGroupSize(k_hnorm, 128, 1, 1));
  ze_group_count_t gcnt_hnorm{(uint32_t)H_V, 1, 1};

  // Run 10 sequential decode steps to test exact state update parity
  std::printf("--- Testing DeltaNet Multi-Step Sequential Equivalence (10 steps) ---\n");
  float max_err_conv = 0.0f;
  float max_err_gate = 0.0f;
  float max_err_attn = 0.0f;
  float max_err_state = 0.0f;
  float max_err_final = 0.0f;
  bool all_steps_pass = true;

  for (int step = 0; step < 10; ++step) {
    // Generate new random inputs for this step
    for (int i = 0; i < C_QKV; ++i) h_qkv_in[i] = fdist(rng);
    for (int i = 0; i < H_V; ++i) {
      h_a[i] = fdist(rng);
      h_b[i] = fdist(rng);
    }
    for (int i = 0; i < H_V * S_V; ++i) h_z[i] = fdist(rng);

    // CPU Step
    cpu_conv1d_update_silu(h_qkv_conv_ref.data(), h_qkv_in.data(), h_conv_state_ref.data(), h_conv_w.data());
    std::vector<float> h_q_ref(H_K * S_V), h_k_ref(H_K * S_V);
    cpu_head_l2_norm_128(h_q_ref.data(), h_qkv_conv_ref.data(), H_K);
    cpu_head_l2_norm_128(h_k_ref.data(), h_qkv_conv_ref.data() + H_K * S_V, H_K);
    const float *h_v_ref = h_qkv_conv_ref.data() + 2 * H_K * S_V;

    cpu_deltanet_gate_prep(h_g_ref.data(), h_beta_ref.data(), h_a.data(), h_b.data(), h_dt_bias.data(), h_A_log.data());
    cpu_deltanet_recurrent_decode(h_attn_out_ref.data(), h_state_ref.data(), h_q_ref.data(), h_k_ref.data(), h_v_ref, h_g_ref.data(), h_beta_ref.data());
    cpu_deltanet_head_norm_silu_z(h_final_ref.data(), h_attn_out_ref.data(), h_z.data(), h_ssm_norm_w.data());

    // GPU Step
    CHECK(zeCommandListReset(list));
    CHECK(zeCommandListAppendMemoryCopy(list, d_qkv_in, h_qkv_in.data(), C_QKV * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_a, h_a.data(), H_V * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_b, h_b.data(), H_V * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, d_z, h_z.data(), H_V * S_V * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // 1. Conv1D
    CHECK(zeCommandListAppendLaunchKernel(list, k_conv, &gcnt_conv, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // 2. Q & K L2 norm
    CHECK(zeCommandListAppendLaunchKernel(list, k_l2, &gcnt_l2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_l2_k, &gcnt_l2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // 3. Gate prep
    CHECK(zeCommandListAppendLaunchKernel(list, k_gate, &gcnt_gate, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // 4. Recurrent decode
    CHECK(zeCommandListAppendLaunchKernel(list, k_recr, &gcnt_recr, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // 5. Head norm & SiLU(z)
    CHECK(zeCommandListAppendLaunchKernel(list, k_hnorm, &gcnt_hnorm, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // Read back final output and state
    CHECK(zeCommandListAppendMemoryCopy(list, h_final_gpu.data(), d_final_out, H_V * S_V * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListAppendMemoryCopy(list, h_state_gpu.data(), d_state, STATE_ELEMS * sizeof(float), nullptr, 0, nullptr));
    CHECK(zeCommandListClose(list));

    CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));

    // Compare
    float cur_final_diff = 0.0f;
    for (int i = 0; i < H_V * S_V; ++i) {
      float diff = std::fabs(h_final_gpu[i] - h_final_ref[i]);
      if (diff > cur_final_diff) cur_final_diff = diff;
    }
    float cur_state_diff = 0.0f;
    for (int i = 0; i < STATE_ELEMS; ++i) {
      float diff = std::fabs(h_state_gpu[i] - h_state_ref[i]);
      if (diff > cur_state_diff) cur_state_diff = diff;
    }

    if (cur_final_diff > max_err_final) max_err_final = cur_final_diff;
    if (cur_state_diff > max_err_state) max_err_state = cur_state_diff;

    bool step_pass = (cur_final_diff < 1e-4f && cur_state_diff < 1e-4f);
    if (!step_pass) all_steps_pass = false;
    std::printf("  Step %d: out_diff = %.2e, state_diff = %.2e [%s]\n", step,
                cur_final_diff, cur_state_diff, step_pass ? "PASS" : "FAIL");
  }

  // Benchmark full DeltaNet decode step (1000 iterations)
  const int ITERS = 1000;
  CHECK(zeCommandListReset(list));
  for (int i = 0; i < ITERS; ++i) {
    CHECK(zeCommandListAppendLaunchKernel(list, k_conv, &gcnt_conv, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_l2, &gcnt_l2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_l2_k, &gcnt_l2, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_gate, &gcnt_gate, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_recr, &gcnt_recr, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendLaunchKernel(list, k_hnorm, &gcnt_hnorm, nullptr, 0, nullptr));
    CHECK(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  }
  CHECK(zeCommandListClose(list));

  auto t0 = std::chrono::steady_clock::now();
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  auto t1 = std::chrono::steady_clock::now();

  double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double lat_deltanet = total_us / ITERS;
  std::printf("\nDeltaNet full step latency: %.2f us (%.2f kTokens/s kernel roofline)\n",
              lat_deltanet, 1e3 / lat_deltanet);

  // Write report
  std::ofstream rpt("tools/kernels_258v/report_deltanet.json");
  rpt << "{\n";
  rpt << "  \"task\": \"T4.4\",\n";
  rpt << "  \"device\": \"Arc 140V (Xe2)\",\n";
  rpt << "  \"all_passed\": " << (all_steps_pass ? "true" : "false") << ",\n";
  rpt << "  \"max_final_output_diff\": " << max_err_final << ",\n";
  rpt << "  \"max_ssm_state_diff\": " << max_err_state << ",\n";
  rpt << "  \"latency_us\": " << lat_deltanet << ",\n";
  rpt << "  \"status\": \"" << (all_steps_pass ? "PASS" : "FAIL") << "\"\n";
  rpt << "}\n";
  rpt.close();

  std::printf("Report written to tools/kernels_258v/report_deltanet.json\n");

  // Cleanup
  zeMemFree(ctx, d_qkv_in);
  zeMemFree(ctx, d_qkv_conv);
  zeMemFree(ctx, d_conv_state);
  zeMemFree(ctx, d_conv_w);
  zeMemFree(ctx, d_q);
  zeMemFree(ctx, d_k);
  zeMemFree(ctx, d_v);
  zeMemFree(ctx, d_a);
  zeMemFree(ctx, d_b);
  zeMemFree(ctx, d_dt_bias);
  zeMemFree(ctx, d_A_log);
  zeMemFree(ctx, d_g);
  zeMemFree(ctx, d_beta);
  zeMemFree(ctx, d_state);
  zeMemFree(ctx, d_attn_out);
  zeMemFree(ctx, d_z);
  zeMemFree(ctx, d_ssm_norm_w);
  zeMemFree(ctx, d_final_out);

  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeKernelDestroy(k_conv);
  zeKernelDestroy(k_l2);
  zeKernelDestroy(k_l2_k);
  zeKernelDestroy(k_gate);
  zeKernelDestroy(k_recr);
  zeKernelDestroy(k_hnorm);
  zeModuleDestroy(mod);
  zeContextDestroy(ctx);

  return all_steps_pass ? 0 : 1;
}
