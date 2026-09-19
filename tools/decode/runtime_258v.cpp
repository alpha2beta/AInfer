// AInfer Unified Single-Process Runtime Engine Implementation (Phase 5)
// Target: Intel Core Ultra 7 258V (Arc 140V, Xe2)
#include "runtime_258v.h"

#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iomanip>

namespace ainfer {

static uint64_t rd64(std::ifstream &f) {
  uint64_t v = 0;
  f.read((char *)&v, 8);
  return v;
}

static uint32_t rd32(std::ifstream &f) {
  uint32_t v = 0;
  f.read((char *)&v, 4);
  return v;
}

bool AInferRuntime258V::load_model_metadata(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "Cannot open model container %s\n", path.c_str());
    return false;
  }

  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "BINFER\x00\x01", 8) != 0) {
    std::fprintf(stderr, "Invalid .binfer magic\n");
    return false;
  }

  uint32_t ver = rd32(f), flags = rd32(f);
  uint64_t n = rd64(f);
  uint64_t table_off = rd64(f);
  uint32_t scount = rd32(f);
  uint32_t align = rd32(f);
  uint64_t total = rd64(f);
  (void)align;

  if (ver != 1 || !(flags & 2)) {
    std::fprintf(stderr, "Unsupported container version %u or non-MoE flag %u\n", ver, flags);
    return false;
  }

  f.seekg(0, std::ios::end);
  uint64_t fsize = (uint64_t)f.tellg();
  f.clear();

  if (total != fsize) {
    std::fprintf(stderr, "File size mismatch: header=%" PRIu64 " actual=%" PRIu64 "\n", total, fsize);
    return false;
  }

  // Parse Section Table
  f.seekg((std::streamoff)table_off);
  uint64_t dir_off = 0, dir_bytes = 0;
  uint32_t dir_crc = 0;
  uint64_t moe_off = 0, moe_bytes = 0;
  uint32_t moe_crc = 0;
  bool has_dir = false, has_moe = false;

  for (uint32_t i = 0; i < scount; ++i) {
    uint32_t sid = rd32(f);
    uint64_t off = rd64(f), nb = rd64(f);
    uint32_t c = rd32(f);
    f.seekg(8, std::ios::cur);
    if (sid == 5) {
      dir_off = off;
      dir_bytes = nb;
      dir_crc = c;
      has_dir = true;
    } else if (sid == 6) {
      moe_off = off;
      moe_bytes = nb;
      moe_crc = c;
      has_moe = true;
    }
  }

  if (!has_dir || !has_moe) {
    std::fprintf(stderr, "Missing Directory or MoE Section\n");
    return false;
  }

  // Validate MoE Section 6
  f.seekg((std::streamoff)moe_off);
  uint64_t mlen = rd64(f);
  std::vector<uint8_t> mbuf(mlen);
  f.read((char *)mbuf.data(), mlen);
  uint32_t chk_crc = crc32_compute(0, (const uint8_t *)&mlen, 8);
  chk_crc = crc32_compute(chk_crc, mbuf.data(), mlen);
  if (chk_crc != moe_crc) {
    std::fprintf(stderr, "MoE Section CRC mismatch\n");
    return false;
  }

  uint32_t num_exp = *(const uint32_t *)(mbuf.data() + 0);
  uint32_t per_tok = *(const uint32_t *)(mbuf.data() + 4);
  uint32_t nlayers = *(const uint32_t *)(mbuf.data() + 24);
  if (num_exp != 256 || per_tok != 8 || nlayers != 40) {
    std::fprintf(stderr, "MoE metadata mismatch: exp=%u, per_tok=%u, layers=%u\n", num_exp, per_tok, nlayers);
    return false;
  }

  // Parse Directory
  f.seekg((std::streamoff)dir_off);
  std::vector<uint8_t> dir_buf(dir_bytes);
  f.read((char *)dir_buf.data(), dir_bytes);
  if (crc32_compute(0, dir_buf.data(), dir_bytes) != dir_crc) {
    std::fprintf(stderr, "Directory CRC mismatch\n");
    return false;
  }

  entries_.clear();
  pay_lo_ = UINT64_MAX;
  pay_hi_ = 0;
  sc_lo_ = UINT64_MAX;
  sc_hi_ = 0;

  for (uint64_t i = 0; i < n; ++i) {
    const uint8_t *entry_ptr = dir_buf.data() + i * 192;
    BinferEntry e;
    std::memcpy(e.name, entry_ptr, 64);
    e.name[63] = 0;
    e.ndim = entry_ptr[64];
    std::memcpy(e.shape, entry_ptr + 72, 64);
    e.lt = entry_ptr[136];
    e.st = entry_ptr[137];
    e.sc_off = *(const uint64_t *)(entry_ptr + 144);
    e.sc_bytes = *(const uint64_t *)(entry_ptr + 152);
    e.d_off = *(const uint64_t *)(entry_ptr + 160);
    e.d_bytes = *(const uint64_t *)(entry_ptr + 168);
    e.crc = *(const uint32_t *)(entry_ptr + 176);

    pay_lo_ = std::min(pay_lo_, e.d_off);
    pay_hi_ = std::max(pay_hi_, e.d_off + e.d_bytes);
    if (e.sc_bytes) {
      sc_lo_ = std::min(sc_lo_, e.sc_off);
      sc_hi_ = std::max(sc_hi_, e.sc_off + e.sc_bytes);
    }
    entries_[e.name] = e;
  }

  return true;
}

bool AInferRuntime258V::init_level_zero() {
  CHECK_L0(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrv = 0;
  CHECK_L0(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK_L0(zeDriverGet(&nDrv, drvs.data()));

  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS) continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS && pr.vendorId == 0x8086) {
        if (pr.deviceId == 0x64a0 || dev_ == nullptr) {
          dev_ = v;
          drv_ = d;
          std::strncpy(dev_name_, pr.name, sizeof(dev_name_) - 1);
        }
      }
    }
  }

  if (!dev_) {
    std::fprintf(stderr, "No supported Intel GPU found\n");
    return false;
  }

  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK_L0(zeContextCreate(drv_, &cdesc, &ctx_));

  // Asynchronous queue for recorded command list replay
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK_L0(zeCommandQueueCreate(ctx_, dev_, &qdesc, &queue_));

  // Synchronous immediate list for copy/upload/reset
  ze_command_queue_desc_t idesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  CHECK_L0(zeCommandListCreateImmediate(ctx_, dev_, &idesc, &cmd_copy_));

  ze_fence_desc_t fdesc = {ZE_STRUCTURE_TYPE_FENCE_DESC, nullptr, 0};
  CHECK_L0(zeFenceCreate(queue_, &fdesc, &fence_));

  return true;
}

bool AInferRuntime258V::allocate_static_arenas() {
  ze_device_mem_alloc_desc_t dmem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

  // 1. Static Weight Payload Arena (T5.1)
  size_t pay_size = (size_t)(pay_hi_ - pay_lo_);
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, pay_size, 4096, dev_, &pay_arena_));

  // 2. Static Weight Scale Arena (T5.1)
  size_t sc_size = (size_t)(sc_hi_ - sc_lo_);
  if (sc_size > 0) {
    CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, sc_size, 4096, dev_, &sc_arena_));
  }

  // 3. Static KV Cache Arena (10 layers, 2 heads, max_ctx, 256 dim, BF16)
  // Cache per layer: K [2, max_ctx, 256] + V [2, max_ctx, 256]
  size_t layer_kv_bytes = (size_t)NUM_KV_HEADS * max_ctx_ * HEAD_DIM * sizeof(uint16_t);
  kv_cache_bytes_ = (size_t)NUM_FULL_ATTN_LAYERS * 2 * layer_kv_bytes;
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, kv_cache_bytes_, 4096, dev_, &kv_cache_arena_));

  // 4. Static SSM Recurrent State Arena (30 layers, 32 heads, 128 x 128, FP32)
  size_t layer_ssm_recr_bytes = (size_t)H_V * S_V * S_V * sizeof(float);
  size_t total_ssm_recr_bytes = (size_t)NUM_DELTANET_LAYERS * layer_ssm_recr_bytes;
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, total_ssm_recr_bytes, 4096, dev_, &ssm_recr_arena_));

  // 5. Static SSM Conv State Arena (30 layers, 8192 channels, 3 steps, FP32)
  size_t layer_ssm_conv_bytes = (size_t)C_QKV * 3 * sizeof(float);
  size_t total_ssm_conv_bytes = (size_t)NUM_DELTANET_LAYERS * layer_ssm_conv_bytes;
  ssm_state_bytes_ = total_ssm_recr_bytes + total_ssm_conv_bytes;
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, total_ssm_conv_bytes, 4096, dev_, &ssm_conv_arena_));

  // 6. Activation Workspace Arena (~128 MiB)
  workspace_bytes_ = 128ULL << 20; // 128 MiB
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, workspace_bytes_, 4096, dev_, &workspace_arena_));

  // 7. Pinned Host-Visible / Device Control Block (T5.4)
  ze_host_mem_alloc_desc_t hmem_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  CHECK_L0(zeMemAllocShared(ctx_, &dmem_desc, &hmem_desc, sizeof(RuntimeControl), 64, dev_, (void **)&d_ctrl_));
  std::memset(&h_ctrl_, 0, sizeof(h_ctrl_));
  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  // Sub-allocate workspace buffers
  uint8_t *w_ptr = (uint8_t *)workspace_arena_;
  auto bump_f32 = [&](size_t count) -> float * {
    float *p = (float *)w_ptr;
    size_t sz = (count * sizeof(float) + 63) & ~63ULL;
    w_ptr += sz;
    return p;
  };

  d_x_ = bump_f32(HIDDEN_DIM);
  d_x_norm_ = bump_f32(HIDDEN_DIM);
  d_qkv_ = bump_f32(C_QKV);
  d_z_ = bump_f32(H_V * S_V);
  d_a_ = bump_f32(H_V);
  d_b_ = bump_f32(H_V);
  d_qkv_conv_ = bump_f32(C_QKV);
  d_q_ = bump_f32(H_K * S_V);
  d_k_ = bump_f32(H_K * S_V);
  d_g_ = bump_f32(H_V);
  d_beta_ = bump_f32(H_V);
  d_attn_out_ = bump_f32(H_V * S_V);
  d_attn_norm_ = bump_f32(H_V * S_V);
  d_attn_proj_ = bump_f32(HIDDEN_DIM);
  d_x_mid_ = bump_f32(HIDDEN_DIM);
  d_x_post_ = bump_f32(HIDDEN_DIM);

  d_q_proj_raw_ = bump_f32(2 * NUM_Q_HEADS * HEAD_DIM);
  d_q_full_ = bump_f32(NUM_Q_HEADS * HEAD_DIM);
  d_gate_full_ = bump_f32(NUM_Q_HEADS * HEAD_DIM);
  d_k_full_ = bump_f32(NUM_KV_HEADS * HEAD_DIM);
  d_v_full_ = bump_f32(NUM_KV_HEADS * HEAD_DIM);

  d_exp_gu_ = bump_f32(2 * EXP_INTER_DIM);
  d_exp_act_ = bump_f32(EXP_INTER_DIM);
  d_exp_out_ = bump_f32(HIDDEN_DIM);
  d_moe_acc_ = bump_f32(HIDDEN_DIM);
  d_exp_gu_all_ = bump_f32(TOP_K * 2 * EXP_INTER_DIM);
  d_exp_act_all_ = bump_f32(TOP_K * EXP_INTER_DIM);
  d_sh_g_ = bump_f32(EXP_INTER_DIM);
  d_sh_u_ = bump_f32(EXP_INTER_DIM);
  d_sh_act_ = bump_f32(EXP_INTER_DIM);
  d_sh_out_ = bump_f32(HIDDEN_DIM);

  d_logits_ = bump_f32(VOCAB_SIZE);
  d_stage1_vals_ = bump_f32(1024);
  d_stage1_idxs_ = (uint32_t *)bump_f32(1024);

  // Sub-allocate chunked prefill workspaces (MAX_PREFILL_CHUNK = 32)
  d_x_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);
  d_x_norm_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);
  d_x_mid_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);
  d_x_post_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);

  d_qkv_chunk_ = bump_f32(MAX_PREFILL_CHUNK * C_QKV);
  d_z_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (H_V * S_V));
  d_a_chunk_ = bump_f32(MAX_PREFILL_CHUNK * H_V);
  d_b_chunk_ = bump_f32(MAX_PREFILL_CHUNK * H_V);
  d_qkv_conv_chunk_ = bump_f32(MAX_PREFILL_CHUNK * C_QKV);
  d_q_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (H_K * S_V));
  d_k_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (H_K * S_V));
  d_g_chunk_ = bump_f32(MAX_PREFILL_CHUNK * H_V);
  d_beta_chunk_ = bump_f32(MAX_PREFILL_CHUNK * H_V);
  d_attn_out_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (H_V * S_V));
  d_attn_norm_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (H_V * S_V));
  d_attn_proj_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);

  d_q_proj_raw_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (2 * NUM_Q_HEADS * HEAD_DIM));
  d_q_full_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (NUM_Q_HEADS * HEAD_DIM));
  d_gate_full_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (NUM_Q_HEADS * HEAD_DIM));
  d_k_full_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (NUM_KV_HEADS * HEAD_DIM));
  d_v_full_chunk_ = bump_f32(MAX_PREFILL_CHUNK * (NUM_KV_HEADS * HEAD_DIM));

  d_sh_g_chunk_ = bump_f32(MAX_PREFILL_CHUNK * EXP_INTER_DIM);
  d_sh_u_chunk_ = bump_f32(MAX_PREFILL_CHUNK * EXP_INTER_DIM);
  d_sh_act_chunk_ = bump_f32(MAX_PREFILL_CHUNK * EXP_INTER_DIM);
  d_sh_down_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);

  d_top_idx_chunk_ = (uint32_t *)bump_f32(MAX_PREFILL_CHUNK * TOP_K);
  d_top_wt_chunk_ = bump_f32(MAX_PREFILL_CHUNK * TOP_K);
  d_sh_gate_chunk_ = bump_f32(MAX_PREFILL_CHUNK);

  d_exp_gu_chunk_ = bump_f32(MAX_PREFILL_CHUNK * TOP_K * 2 * EXP_INTER_DIM);
  d_exp_act_chunk_ = bump_f32(MAX_PREFILL_CHUNK * TOP_K * EXP_INTER_DIM);
  d_exp_down_chunk_ = bump_f32(MAX_PREFILL_CHUNK * TOP_K * HIDDEN_DIM);
  d_moe_acc_chunk_ = bump_f32(MAX_PREFILL_CHUNK * HIDDEN_DIM);

  d_expert_counts_ = (int *)bump_f32(256);
  d_expert_offsets_ = (int *)bump_f32(256);
  d_sorted_tokens_ = (int *)bump_f32(MAX_PREFILL_CHUNK * TOP_K);
  d_sorted_slots_ = (int *)bump_f32(MAX_PREFILL_CHUNK * TOP_K);

  size_t used_workspace = (size_t)(w_ptr - (uint8_t *)workspace_arena_);
  if (used_workspace > workspace_bytes_) {
    std::fprintf(stderr, "FATAL: Workspace overflow: used %zu > allocated %zu\n", used_workspace, (size_t)workspace_bytes_);
    return false;
  }

  CHECK_L0(zeMemAllocShared(ctx_, &dmem_desc, &hmem_desc, MAX_PREFILL_CHUNK * sizeof(int), 64, dev_, (void **)&d_tokens_chunk_));

  // Map root tensor device addresses
  auto get_pay = [&](const std::string &nm) -> void * {
    auto it = entries_.find(nm);
    if (it == entries_.end()) return nullptr;
    return (char *)pay_arena_ + (it->second.d_off - pay_lo_);
  };
  auto get_sc = [&](const std::string &nm) -> void * {
    auto it = entries_.find(nm);
    if (it == entries_.end() || it->second.sc_bytes == 0) return nullptr;
    return (char *)sc_arena_ + (it->second.sc_off - sc_lo_);
  };

  d_embed_tokens_ = get_pay("embed_tokens.weight");
  d_final_norm_w_ = get_pay("norm.weight");
  d_lm_head_w_ = get_pay("lm_head.weight");
  d_lm_head_s_ = get_sc("lm_head.weight");

  // Map 40-layer bindings
  int full_slot_cnt = 0;
  int linear_slot_cnt = 0;

  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    LayerBinding &lb = layers_[l];
    lb.layer_idx = l;
    lb.is_full_attn = (l % FULL_ATTN_INTERVAL == 3);

    std::string prefix = "layers." + std::to_string(l) + ".";
    lb.in_norm_w = get_pay(prefix + "input_layernorm.weight");
    lb.post_norm_w = get_pay(prefix + "post_attention_layernorm.weight");

    if (lb.is_full_attn) {
      lb.full_slot = full_slot_cnt++;
      lb.linear_slot = -1;
      lb.q_proj_w = get_pay(prefix + "self_attn.q_proj.weight");
      lb.q_proj_s = get_sc(prefix + "self_attn.q_proj.weight");
      lb.k_proj_w = get_pay(prefix + "self_attn.k_proj.weight");
      lb.k_proj_s = get_sc(prefix + "self_attn.k_proj.weight");
      lb.v_proj_w = get_pay(prefix + "self_attn.v_proj.weight");
      lb.v_proj_s = get_sc(prefix + "self_attn.v_proj.weight");
      lb.o_proj_w = get_pay(prefix + "self_attn.o_proj.weight");
      lb.o_proj_s = get_sc(prefix + "self_attn.o_proj.weight");
      lb.q_norm_w = get_pay(prefix + "self_attn.q_norm.weight");
      lb.k_norm_w = get_pay(prefix + "self_attn.k_norm.weight");

      size_t slot_offset = (size_t)lb.full_slot * (2 * layer_kv_bytes);
      lb.k_cache = (char *)kv_cache_arena_ + slot_offset;
      lb.v_cache = (char *)kv_cache_arena_ + slot_offset + layer_kv_bytes;
    } else {
      lb.linear_slot = linear_slot_cnt++;
      lb.full_slot = -1;
      lb.qkv_w = get_pay(prefix + "linear_attn.in_proj_qkv.weight");
      lb.qkv_s = get_sc(prefix + "linear_attn.in_proj_qkv.weight");
      lb.z_w = get_pay(prefix + "linear_attn.in_proj_z.weight");
      lb.z_s = get_sc(prefix + "linear_attn.in_proj_z.weight");
      lb.a_w = get_pay(prefix + "linear_attn.in_proj_a.weight");
      lb.a_s = get_sc(prefix + "linear_attn.in_proj_a.weight");
      lb.b_w = get_pay(prefix + "linear_attn.in_proj_b.weight");
      lb.b_s = get_sc(prefix + "linear_attn.in_proj_b.weight");
      lb.conv_w = get_pay(prefix + "linear_attn.conv1d.weight");
      lb.dt_bias = get_pay(prefix + "linear_attn.dt_bias");
      lb.A_log = get_pay(prefix + "linear_attn.A_log");
      lb.ssm_norm_w = get_pay(prefix + "linear_attn.norm.weight");
      lb.out_proj_w = get_pay(prefix + "linear_attn.out_proj.weight");
      lb.out_proj_s = get_sc(prefix + "linear_attn.out_proj.weight");

      lb.ssm_state = (char *)ssm_recr_arena_ + (size_t)lb.linear_slot * layer_ssm_recr_bytes;
      lb.conv_state = (char *)ssm_conv_arena_ + (size_t)lb.linear_slot * layer_ssm_conv_bytes;
    }

    // MoE weights
    lb.router_w = get_pay(prefix + "mlp.gate.weight");
    lb.shared_gate_w = get_pay(prefix + "mlp.shared_expert_gate.weight");
    lb.exp_gu_w = get_pay(prefix + "mlp.experts.gate_up_proj");
    lb.exp_gu_s = get_sc(prefix + "mlp.experts.gate_up_proj");
    lb.exp_dn_w = get_pay(prefix + "mlp.experts.down_proj");
    lb.exp_dn_s = get_sc(prefix + "mlp.experts.down_proj");

    lb.sh_gate_w = get_pay(prefix + "mlp.shared_expert.gate_proj.weight");
    lb.sh_gate_s = get_sc(prefix + "mlp.shared_expert.gate_proj.weight");
    lb.sh_up_w = get_pay(prefix + "mlp.shared_expert.up_proj.weight");
    lb.sh_up_s = get_sc(prefix + "mlp.shared_expert.up_proj.weight");
    lb.sh_down_w = get_pay(prefix + "mlp.shared_expert.down_proj.weight");
    lb.sh_down_s = get_sc(prefix + "mlp.shared_expert.down_proj.weight");
  }

  // Clear states
  reset_state();

  return true;
}

bool AInferRuntime258V::upload_weights(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;

  const size_t CH = 64u << 20; // 64 MiB staging buffer (preserves host RAM)
  std::vector<char> staging(CH);

  auto stream_to_device = [&](uint64_t foff, void *dst_arena, uint64_t alo, uint64_t len) -> bool {
    uint64_t done = 0;
    while (done < len) {
      size_t c = (size_t)((len - done > CH) ? CH : (len - done));
      f.clear();
      f.seekg((std::streamoff)(foff + done));
      f.read(staging.data(), c);
      CHECK_L0(zeCommandListAppendMemoryCopy(
          cmd_copy_, (char *)dst_arena + (foff + done - alo), staging.data(), c,
          nullptr, 0, nullptr));
      done += c;
    }
    return true;
  };

  // Upload payload arena
  if (!stream_to_device(pay_lo_, pay_arena_, pay_lo_, pay_hi_ - pay_lo_)) return false;

  // Upload scale arena
  if (sc_hi_ > sc_lo_) {
    if (!stream_to_device(sc_lo_, sc_arena_, sc_lo_, sc_hi_ - sc_lo_)) return false;
  }

  return true;
}

bool AInferRuntime258V::compile_kernels(const std::string &spv_path) {
  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "Cannot open SPIR-V module %s\n", spv_path.c_str());
    return false;
  }
  spv_f.seekg(0, std::ios::end);
  size_t spv_size = spv_f.tellg();
  spv_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> spv(spv_size);
  spv_f.read((char *)spv.data(), spv_size);

  ze_module_desc_t mdesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV,
                            spv_size, spv.data(), nullptr, nullptr};
  CHECK_L0(zeModuleCreate(ctx_, dev_, &mdesc, &mod_, nullptr));

  auto get_k = [&](const char *name) -> ze_kernel_handle_t {
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ze_kernel_handle_t kh = nullptr;
    ze_result_t res = zeKernelCreate(mod_, &kd, &kh);
    if (res != ZE_RESULT_SUCCESS) {
      std::fprintf(stderr, "Failed to create kernel %s (code %d)\n", name, (int)res);
      return nullptr;
    }
    return kh;
  };

  k_gemv_ = get_k("int4_gemv_m1");
  k_router_ = get_k("moe_topk_router");
  k_norm2048_ = get_k("rmsnorm_2048");
  k_norm256_ = get_k("rmsnorm_head_256");
  k_silu512_ = get_k("silu_mul_512");
  k_resadd_ = get_k("residual_add_2048");
  k_conv_ = get_k("conv1d_update_silu");
  k_l2_norm_ = get_k("head_l2_norm_128");
  k_gate_prep_ = get_k("deltanet_gate_prep");
  k_recr_ = get_k("deltanet_recurrent_decode");
  k_hnorm_ = get_k("deltanet_head_norm_silu_z");
  k_argmax1_ = get_k("argmax_stage1");

  // Control-block driven kernels
  k_embed_ = get_k("embed_gather");
  k_exp_gemv_ = get_k("moe_expert_gemv_ctrl");
  k_exp_dn_accum_ = get_k("moe_expert_down_accum_ctrl");
  k_gemv_add_scaled_ = get_k("int4_gemv_m1_add_scaled");
  k_accum_ctrl_ = get_k("moe_accumulate_expert_ctrl");
  k_add_shared_ctrl_ = get_k("moe_add_shared_expert_ctrl");
  k_rope_ctrl_ = get_k("rope_and_kv_append_ctrl");
  k_attn_ctrl_ = get_k("gqa_attn_decode_ctrl");
  k_deinterleave_qg_ = get_k("deinterleave_q_gate");
  k_argmax2_ctrl_ = get_k("argmax_stage2_ctrl");
  k_concat2_ = get_k("concat2");
  k_argmax2_mtp_ = get_k("argmax_stage2_ptr");

  // Batched MoE and Fused Tail Kernels
  k_exp_gu_all_ = get_k("moe_gateup_all8_ctrl");
  k_silu_all_ = get_k("silu_mul_all8");
  k_exp_dn_accum_all_ = get_k("moe_down_accum_all8_ctrl");
  k_lm_head_argmax1_ = get_k("int4_gemv_lm_head_argmax1");

  // Batch prefill kernels
  k_gemm_prefill_ = get_k("int4_gemm_prefill");
  k_embed_batch_ = get_k("embed_gather_batch");
  k_norm2048_batch_ = get_k("rmsnorm_2048_batch");
  k_conv_batch_ = get_k("conv1d_update_silu_batch");
  k_l2_norm_qk_batch_ = get_k("head_l2_norm_qk_batch");
  k_gate_prep_batch_ = get_k("gate_prep_batch");
  // v2 double-buffers 4 recurrence steps per workgroup barrier (1 B/4
  // barriers vs B); AINFER_RECR_V1=1 restores the v1 kernel.
  k_recr_batch_ = std::getenv("AINFER_RECR_V1")
                      ? get_k("deltanet_recurrent_batch")
                      : get_k("deltanet_recurrent_batch_v2");
  k_hnorm_batch_ = get_k("deltanet_head_norm_silu_z_batch");
  k_deinterleave_qg_batch_ = get_k("deinterleave_q_gate_batch");
  k_rope_batch_ = get_k("rope_and_kv_append_batch");
  // v2 uses in-register subgroup-shuffle reduction (0.25 barriers/position
  // vs ~10 in v1); AINFER_ATTN_V1=1 restores the v1 SLM-tree kernel.
  k_attn_batch_ = std::getenv("AINFER_ATTN_V1")
                      ? get_k("gqa_attn_prefill_batch")
                      : get_k("gqa_attn_prefill_batch_v2");
  k_router_batch_ = get_k("moe_topk_router_batch");
  k_moe_build_expert_bins_ = get_k("moe_build_expert_bins");
  k_moe_gateup_grouped_batch_ = get_k("moe_gateup_grouped_batch");
  k_moe_down_grouped_batch_ = get_k("moe_down_grouped_batch");
  k_moe_accum_down_batch_ = get_k("moe_accum_down_batch");
  k_exp_gu_all_batch_ = get_k("moe_gateup_all8_batch");
  k_silu_all_batch_ = get_k("silu_mul_all8_batch");
  k_exp_dn_accum_all_batch_ = get_k("moe_down_accum_all8_batch");
  k_silu_mul_batch_ = get_k("silu_mul_batch");
  k_block_resadd_moe_batch_ = get_k("block_resadd_moe_batch");
  k_resadd_batch_ = get_k("resadd_batch");

  // Speculative verification kernels (T10.1)
  k_conv_m2_spec_ = get_k("conv1d_update_silu_m2_spec");
  k_recr_m2_spec_ = get_k("deltanet_recurrent_m2_spec");
  k_lm_head_m2_argmax1_ = get_k("int4_gemv_m2_lm_head_argmax1");
  k_gemv_m2_ = get_k("int4_gemv_m2");

  if (!k_gemv_ || !k_router_ || !k_norm2048_ || !k_norm256_ || !k_silu512_ || !k_resadd_ ||
      !k_conv_ || !k_recr_ || !k_hnorm_ || !k_embed_ || !k_gemv_add_scaled_ ||
      !k_exp_gu_all_ || !k_silu_all_ || !k_exp_dn_accum_all_ || !k_lm_head_argmax1_ ||
      !k_rope_ctrl_ || !k_attn_ctrl_ || !k_deinterleave_qg_ || !k_argmax2_ctrl_ ||
      !k_gemm_prefill_ || !k_embed_batch_ || !k_norm2048_batch_ || !k_conv_batch_ ||
      !k_l2_norm_qk_batch_ || !k_gate_prep_batch_ || !k_recr_batch_ || !k_hnorm_batch_ ||
      !k_deinterleave_qg_batch_ || !k_rope_batch_ || !k_attn_batch_ || !k_router_batch_ ||
      !k_moe_build_expert_bins_ || !k_moe_gateup_grouped_batch_ ||
      !k_moe_down_grouped_batch_ || !k_moe_accum_down_batch_ ||
      !k_exp_gu_all_batch_ || !k_silu_all_batch_ || !k_exp_dn_accum_all_batch_ ||
      !k_silu_mul_batch_ || !k_block_resadd_moe_batch_ || !k_resadd_batch_ ||
      !k_conv_m2_spec_ || !k_recr_m2_spec_ || !k_lm_head_m2_argmax1_ || !k_gemv_m2_) {
    std::fprintf(stderr, "One or more required kernels could not be created\n");
    return false;
  }

  return true;
}

bool AInferRuntime258V::record_command_lists() {
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};

  // 0. Create unified single recorded command lists for prefill and decode
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_step_));
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_prefill_step_));

#define APPEND_L0_K(cl, kern, gc) do { \
    CHECK_L0(zeCommandListAppendLaunchKernel((cl), (kern), (gc), nullptr, 0, nullptr)); \
    if (cmd_step_) CHECK_L0(zeCommandListAppendLaunchKernel(cmd_step_, (kern), (gc), nullptr, 0, nullptr)); \
    if (cmd_prefill_step_ && (cl) != cmd_tail_) CHECK_L0(zeCommandListAppendLaunchKernel(cmd_prefill_step_, (kern), (gc), nullptr, 0, nullptr)); \
  } while (0)

#define APPEND_L0_B(cl) do { \
    CHECK_L0(zeCommandListAppendBarrier((cl), nullptr, 0, nullptr)); \
    if (cmd_step_) CHECK_L0(zeCommandListAppendBarrier(cmd_step_, nullptr, 0, nullptr)); \
    if (cmd_prefill_step_ && (cl) != cmd_tail_) CHECK_L0(zeCommandListAppendBarrier(cmd_prefill_step_, nullptr, 0, nullptr)); \
  } while (0)

  // 1. Record cmd_embed_ list (T5.3)
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_embed_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 0, sizeof(void *), &d_x_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 1, sizeof(void *), &d_embed_tokens_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 2, sizeof(void *), &d_ctrl_));
  CHECK_L0(zeKernelSetGroupSize(k_embed_, 256, 1, 1));
  ze_group_count_t gcnt_embed{HIDDEN_DIM / 256, 1, 1};
  APPEND_L0_K(cmd_embed_, k_embed_, &gcnt_embed);
  APPEND_L0_B(cmd_embed_);
  CHECK_L0(zeCommandListClose(cmd_embed_));

  // Helper lambda for GEMV append
  auto append_gemv = [&](ze_command_list_handle_t list, float *y, void *w, void *s, float *x, int M, int K) {
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 0, sizeof(void *), &y));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 1, sizeof(void *), &w));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 2, sizeof(void *), &s));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 3, sizeof(void *), &x));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 4, sizeof(int), &M));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 5, sizeof(int), &K));
    CHECK_L0_VOID(zeKernelSetGroupSize(k_gemv_, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)((M + 255) / 256), 1, 1};
    CHECK_L0_VOID(zeCommandListAppendLaunchKernel(list, k_gemv_, &gc, nullptr, 0, nullptr));
    if (cmd_step_) CHECK_L0_VOID(zeCommandListAppendLaunchKernel(cmd_step_, k_gemv_, &gc, nullptr, 0, nullptr));
    if (cmd_prefill_step_ && list != cmd_tail_) CHECK_L0_VOID(zeCommandListAppendLaunchKernel(cmd_prefill_step_, k_gemv_, &gc, nullptr, 0, nullptr));
  };

  ze_group_count_t gcnt_norm2048{1, 1, 1};
  ze_group_count_t gcnt_silu{2, 1, 1};
  ze_group_count_t gcnt_res{8, 1, 1};
  ze_group_count_t gcnt_router{1, 1, 1};

  // 2. Record 40 layer command lists (T5.3)
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    LayerBinding &lb = layers_[l];
    ze_command_list_handle_t list = nullptr;
    CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &list));

    // A. Input RMSNorm: d_x_norm = norm(d_x_, in_norm_w)
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_norm_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &lb.in_norm_w));
    CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
    APPEND_L0_K(list, k_norm2048_, &gcnt_norm2048);
    APPEND_L0_B(list);

    if (lb.is_full_attn) {
      // Full-Attention Layer (10 layers)
      append_gemv(list, d_q_proj_raw_, lb.q_proj_w, lb.q_proj_s, d_x_norm_, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemv(list, d_k_full_, lb.k_proj_w, lb.k_proj_s, d_x_norm_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemv(list, d_v_full_, lb.v_proj_w, lb.v_proj_s, d_x_norm_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      APPEND_L0_B(list);

      // Deinterleave Q and Gate: [16, 512] -> Q [16, 256] and Gate [16, 256]
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 0, sizeof(void *), &d_q_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 1, sizeof(void *), &d_gate_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 2, sizeof(void *), &d_q_proj_raw_));
      CHECK_L0(zeKernelSetGroupSize(k_deinterleave_qg_, 256, 1, 1));
      ze_group_count_t gcnt_deint{(uint32_t)((NUM_Q_HEADS * HEAD_DIM) / 256), 1, 1}; // 16 groups
      APPEND_L0_K(list, k_deinterleave_qg_, &gcnt_deint);
      APPEND_L0_B(list);

      // Q head norm
      int nq = NUM_Q_HEADS;
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_q_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_q_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.q_norm_w));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &nq));
      CHECK_L0(zeKernelSetGroupSize(k_norm256_, 64, 1, 1));
      ze_group_count_t gcnt_nq{(uint32_t)NUM_Q_HEADS, 1, 1};
      APPEND_L0_K(list, k_norm256_, &gcnt_nq);

      // K head norm
      int nkv = NUM_KV_HEADS;
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_k_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_k_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.k_norm_w));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &nkv));
      ze_group_count_t gcnt_nkv{(uint32_t)NUM_KV_HEADS, 1, 1};
      APPEND_L0_K(list, k_norm256_, &gcnt_nkv);
      APPEND_L0_B(list);

      // RoPE and KV Cache Append (Device control block drives position) (T5.4)
      uint32_t max_c = max_ctx_;
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 0, sizeof(void *), &d_q_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 1, sizeof(void *), &d_k_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 2, sizeof(void *), &d_v_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0(zeKernelSetGroupSize(k_rope_ctrl_, 256, 1, 1));
      ze_group_count_t gcnt_rope{1, 1, 1};
      APPEND_L0_K(list, k_rope_ctrl_, &gcnt_rope);
      APPEND_L0_B(list);

      // GQA Attention Decode (Device control block drives position) (T5.4)
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 0, sizeof(void *), &d_attn_out_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 1, sizeof(void *), &d_q_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 2, sizeof(void *), &d_gate_full_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0(zeKernelSetGroupSize(k_attn_ctrl_, 256, 1, 1));
      ze_group_count_t gcnt_attn{(uint32_t)NUM_Q_HEADS, 1, 1};
      APPEND_L0_K(list, k_attn_ctrl_, &gcnt_attn);
      APPEND_L0_B(list);

      // O Proj GEMV
      append_gemv(list, d_attn_proj_, lb.o_proj_w, lb.o_proj_s, d_attn_out_, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
      APPEND_L0_B(list);
    } else {
      // DeltaNet Layer (30 layers)
      append_gemv(list, d_qkv_, lb.qkv_w, lb.qkv_s, d_x_norm_, C_QKV, HIDDEN_DIM);
      append_gemv(list, d_z_, lb.z_w, lb.z_s, d_x_norm_, H_V * S_V, HIDDEN_DIM);
      append_gemv(list, d_a_, lb.a_w, lb.a_s, d_x_norm_, H_V, HIDDEN_DIM);
      append_gemv(list, d_b_, lb.b_w, lb.b_s, d_x_norm_, H_V, HIDDEN_DIM);
      APPEND_L0_B(list);

      // Conv1D
      CHECK_L0(zeKernelSetArgumentValue(k_conv_, 0, sizeof(void *), &d_qkv_conv_));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_, 1, sizeof(void *), &d_qkv_));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_, 2, sizeof(void *), &lb.conv_state));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_, 3, sizeof(void *), &lb.conv_w));
      CHECK_L0(zeKernelSetGroupSize(k_conv_, 256, 1, 1));
      ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};
      APPEND_L0_K(list, k_conv_, &gcnt_conv);
      APPEND_L0_B(list);

      // Head L2 Norm on Q and K
      int num_heads_16 = H_K;
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 0, sizeof(void *), &d_q_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 1, sizeof(void *), &d_qkv_conv_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 2, sizeof(int), &num_heads_16));
      CHECK_L0(zeKernelSetGroupSize(k_l2_norm_, 128, 1, 1));
      ze_group_count_t gcnt_l2{(uint32_t)H_K, 1, 1};
      APPEND_L0_K(list, k_l2_norm_, &gcnt_l2);

      float *d_qkv_k = d_qkv_conv_ + H_K * S_V;
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 0, sizeof(void *), &d_k_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 1, sizeof(void *), &d_qkv_k));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_, 2, sizeof(int), &num_heads_16));
      APPEND_L0_K(list, k_l2_norm_, &gcnt_l2);
      APPEND_L0_B(list);

      // Gate Prep
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 0, sizeof(void *), &d_g_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 1, sizeof(void *), &d_beta_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 2, sizeof(void *), &d_a_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 3, sizeof(void *), &d_b_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 4, sizeof(void *), &lb.dt_bias));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_, 5, sizeof(void *), &lb.A_log));
      CHECK_L0(zeKernelSetGroupSize(k_gate_prep_, 32, 1, 1));
      ze_group_count_t gcnt_gate{1, 1, 1};
      APPEND_L0_K(list, k_gate_prep_, &gcnt_gate);
      APPEND_L0_B(list);

      // DeltaNet Recurrent Decode
      float *d_v_ptr = d_qkv_conv_ + 2 * H_K * S_V;
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 0, sizeof(void *), &d_attn_out_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 1, sizeof(void *), &lb.ssm_state));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 2, sizeof(void *), &d_q_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 3, sizeof(void *), &d_k_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 4, sizeof(void *), &d_v_ptr));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 5, sizeof(void *), &d_g_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_, 6, sizeof(void *), &d_beta_));
      CHECK_L0(zeKernelSetGroupSize(k_recr_, 128, 1, 1));
      ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};
      APPEND_L0_K(list, k_recr_, &gcnt_recr);
      APPEND_L0_B(list);

      // Head Norm & SiLU(z)
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_, 0, sizeof(void *), &d_attn_norm_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_, 1, sizeof(void *), &d_attn_out_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_, 2, sizeof(void *), &d_z_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_, 3, sizeof(void *), &lb.ssm_norm_w));
      CHECK_L0(zeKernelSetGroupSize(k_hnorm_, 128, 1, 1));
      ze_group_count_t gcnt_hnorm{(uint32_t)H_V, 1, 1};
      APPEND_L0_K(list, k_hnorm_, &gcnt_hnorm);
      APPEND_L0_B(list);

      // Output Proj GEMV
      append_gemv(list, d_attn_proj_, lb.out_proj_w, lb.out_proj_s, d_attn_norm_, HIDDEN_DIM, H_V * S_V);
      APPEND_L0_B(list);
    }

    // B. Mid Residual Add: d_x_mid = d_x + d_attn_proj
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 0, sizeof(void *), &d_x_mid_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 1, sizeof(void *), &d_x_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 2, sizeof(void *), &d_attn_proj_));
    CHECK_L0(zeKernelSetGroupSize(k_resadd_, 256, 1, 1));
    APPEND_L0_K(list, k_resadd_, &gcnt_res);
    APPEND_L0_B(list);

    // C. Post-Attn Norm: d_x_post = norm(d_x_mid, post_norm_w)
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_post_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_mid_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &lb.post_norm_w));
    APPEND_L0_K(list, k_norm2048_, &gcnt_norm2048);
    APPEND_L0_B(list);

    // D. MoE Router (Strategy 3: Writes top_idx, top_wt, sh_gate_val into d_ctrl_)
    uint32_t *d_top_idx_ptr = d_ctrl_->top_idx;
    float *d_top_wt_ptr = d_ctrl_->top_wt;
    float *d_sh_gate_ptr = &d_ctrl_->sh_gate_val;

    CHECK_L0(zeKernelSetArgumentValue(k_router_, 0, sizeof(void *), &d_x_post_));
    CHECK_L0(zeKernelSetArgumentValue(k_router_, 1, sizeof(void *), &lb.router_w));
    CHECK_L0(zeKernelSetArgumentValue(k_router_, 2, sizeof(void *), &lb.shared_gate_w));
    CHECK_L0(zeKernelSetArgumentValue(k_router_, 3, sizeof(void *), &d_top_idx_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_router_, 4, sizeof(void *), &d_top_wt_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_router_, 5, sizeof(void *), &d_sh_gate_ptr));
    CHECK_L0(zeKernelSetGroupSize(k_router_, 256, 1, 1));
    APPEND_L0_K(list, k_router_, &gcnt_router);
    APPEND_L0_B(list);

    // E. Batched MoE 8 Active Experts (3 kernels total instead of 24) (T4.2/T5.4)
    int M_gu = 2 * EXP_INTER_DIM, K_gu = HIDDEN_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 0, sizeof(void *), &d_exp_gu_all_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 1, sizeof(void *), &lb.exp_gu_w));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 2, sizeof(void *), &lb.exp_gu_s));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 3, sizeof(void *), &d_x_post_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 4, sizeof(void *), &d_top_idx_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 5, sizeof(int), &K_gu));
    CHECK_L0(zeKernelSetGroupSize(k_exp_gu_all_, 256, 1, 1));
    ze_group_count_t gc_gu_all{(uint32_t)((TOP_K * M_gu + 255) / 256), 1, 1};
    APPEND_L0_K(list, k_exp_gu_all_, &gc_gu_all);
    APPEND_L0_B(list);

    CHECK_L0(zeKernelSetArgumentValue(k_silu_all_, 0, sizeof(void *), &d_exp_act_all_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_all_, 1, sizeof(void *), &d_exp_gu_all_));
    CHECK_L0(zeKernelSetGroupSize(k_silu_all_, 256, 1, 1));
    ze_group_count_t gc_silu_all{(uint32_t)((TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
    APPEND_L0_K(list, k_silu_all_, &gc_silu_all);
    APPEND_L0_B(list);

    int M_dn = HIDDEN_DIM, K_dn = EXP_INTER_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 0, sizeof(void *), &d_moe_acc_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 1, sizeof(void *), &lb.exp_dn_w));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 2, sizeof(void *), &lb.exp_dn_s));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 3, sizeof(void *), &d_exp_act_all_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 4, sizeof(void *), &d_top_idx_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 5, sizeof(void *), &d_top_wt_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 6, sizeof(int), &M_dn));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 7, sizeof(int), &K_dn));
    CHECK_L0(zeKernelSetGroupSize(k_exp_dn_accum_all_, 64, 1, 1));
    ze_group_count_t gc_dn_all{(uint32_t)((M_dn + 63) / 64), 1, 1};
    APPEND_L0_K(list, k_exp_dn_accum_all_, &gc_dn_all);
    APPEND_L0_B(list);

    // F. Shared Expert
    append_gemv(list, d_sh_g_, lb.sh_gate_w, lb.sh_gate_s, d_x_post_, EXP_INTER_DIM, HIDDEN_DIM);
    append_gemv(list, d_sh_u_, lb.sh_up_w, lb.sh_up_s, d_x_post_, EXP_INTER_DIM, HIDDEN_DIM);
    APPEND_L0_B(list);

    CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 0, sizeof(void *), &d_sh_act_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 1, sizeof(void *), &d_sh_g_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 2, sizeof(void *), &d_sh_u_));
    CHECK_L0(zeKernelSetGroupSize(k_silu512_, 256, 1, 1));
    APPEND_L0_K(list, k_silu512_, &gcnt_silu);
    APPEND_L0_B(list);

    // Fused Shared Down GEMV + Add into d_moe_acc_ [2048, 512]
    int M_sh_dn = HIDDEN_DIM, K_sh_dn = EXP_INTER_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 0, sizeof(void *), &d_moe_acc_));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 1, sizeof(void *), &lb.sh_down_w));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 2, sizeof(void *), &lb.sh_down_s));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 3, sizeof(void *), &d_sh_act_));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 4, sizeof(void *), &d_sh_gate_ptr));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 5, sizeof(int), &M_sh_dn));
    CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 6, sizeof(int), &K_sh_dn));
    CHECK_L0(zeKernelSetGroupSize(k_gemv_add_scaled_, 256, 1, 1));
    ze_group_count_t gc_sh_dn{(uint32_t)((M_sh_dn + 255) / 256), 1, 1};
    APPEND_L0_K(list, k_gemv_add_scaled_, &gc_sh_dn);
    APPEND_L0_B(list);

    // G. Final Block Residual Add: d_x = d_x_mid + d_moe_acc
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 0, sizeof(void *), &d_x_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 1, sizeof(void *), &d_x_mid_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 2, sizeof(void *), &d_moe_acc_));
    APPEND_L0_K(list, k_resadd_, &gcnt_res);
    APPEND_L0_B(list);

    CHECK_L0(zeCommandListClose(list));
    cmd_layers_[l] = list;
  }

  // Close unified prefill command list (embed + all 40 layers)
  CHECK_L0(zeCommandListClose(cmd_prefill_step_));

  // 3. Record cmd_tail_ list (T5.3)
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_tail_));

  // Final RMSNorm
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_norm_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &d_final_norm_w_));
  APPEND_L0_K(cmd_tail_, k_norm2048_, &gcnt_norm2048);
  APPEND_L0_B(cmd_tail_);

  // LM Head GEMV + Argmax Stage 1 fused [248320, 2048]
  int M_lm = VOCAB_SIZE;
  int K_lm = HIDDEN_DIM;
  void *null_logits = nullptr;
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 0, sizeof(void *), &null_logits));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 1, sizeof(void *), &d_lm_head_w_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 2, sizeof(void *), &d_lm_head_s_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 3, sizeof(void *), &d_x_norm_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 4, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 5, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 6, sizeof(int), &M_lm));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 7, sizeof(int), &K_lm));
  CHECK_L0(zeKernelSetGroupSize(k_lm_head_argmax1_, 256, 1, 1));
  ze_group_count_t gcnt_lm{(VOCAB_SIZE + 255) / 256, 1, 1}; // 970 groups
  APPEND_L0_K(cmd_tail_, k_lm_head_argmax1_, &gcnt_lm);
  APPEND_L0_B(cmd_tail_);

  // Argmax Stage 2 (Writes selected_token directly into d_ctrl_->selected_token) (T5.4)
  uint32_t num_stage1_groups = gcnt_lm.groupCountX;
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_ctrl_, 0, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_ctrl_, 1, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_ctrl_, 2, sizeof(void *), &d_ctrl_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_ctrl_, 3, sizeof(uint32_t), &num_stage1_groups));
  CHECK_L0(zeKernelSetGroupSize(k_argmax2_ctrl_, 256, 1, 1));
  ze_group_count_t gcnt_arg2{1, 1, 1};
  APPEND_L0_K(cmd_tail_, k_argmax2_ctrl_, &gcnt_arg2);
  APPEND_L0_B(cmd_tail_);

  CHECK_L0(zeCommandListClose(cmd_tail_));

  // Close unified decode command list (embed + all 40 layers + tail)
  CHECK_L0(zeCommandListClose(cmd_step_));

  // Build full step lists array for unified single-call dispatch
  all_step_lists_.clear();
  all_step_lists_.push_back(cmd_embed_);
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    all_step_lists_.push_back(cmd_layers_[l]);
  }
  all_step_lists_.push_back(cmd_tail_);

  return true;
}

bool AInferRuntime258V::init(const std::string &binfer_path, const std::string &spv_path, uint32_t max_ctx) {
  max_ctx_ = max_ctx;

  std::printf("[AInfer 258V] 1. Parsing model manifest & directory: %s\n", binfer_path.c_str());
  if (!load_model_metadata(binfer_path)) return false;

  std::printf("[AInfer 258V] 2. Initializing Level Zero on Intel Arc 140V (Xe2)\n");
  if (!init_level_zero()) return false;

  std::printf("[AInfer 258V] 3. Allocating static unified memory arenas (T5.1)\n");
  if (!allocate_static_arenas()) return false;

  std::printf("[AInfer 258V]    Payload Arena:   %.2f GiB\n", (pay_hi_ - pay_lo_) / (1024.0 * 1024.0 * 1024.0));
  std::printf("[AInfer 258V]    Scale Arena:     %.2f MiB\n", (sc_hi_ - sc_lo_) / (1024.0 * 1024.0));
  std::printf("[AInfer 258V]    KV Cache Arena:  %.2f MiB (10 full layers @ %u ctx)\n", kv_cache_bytes_ / (1024.0 * 1024.0), max_ctx_);
  std::printf("[AInfer 258V]    SSM State Arena: %.2f MiB (30 DeltaNet layers)\n", ssm_state_bytes_ / (1024.0 * 1024.0));
  std::printf("[AInfer 258V]    Workspace:       %.2f MiB\n", workspace_bytes_ / (1024.0 * 1024.0));

  std::printf("[AInfer 258V] 4. Streaming weights into static GPU arenas\n");
  auto t0 = std::chrono::steady_clock::now();
  if (!upload_weights(binfer_path)) return false;
  auto t1 = std::chrono::steady_clock::now();
  double upload_s = std::chrono::duration<double>(t1 - t0).count();
  double total_gb = ((pay_hi_ - pay_lo_) + (sc_hi_ - sc_lo_)) / 1e9;
  std::printf("[AInfer 258V]    Loaded %.2f GB in %.2f s (%.2f GB/s)\n", total_gb, upload_s, total_gb / upload_s);

  std::printf("[AInfer 258V] 5. Compiling SPIR-V kernels: %s\n", spv_path.c_str());
  if (!compile_kernels(spv_path)) return false;

  std::printf("[AInfer 258V] 6. Recording 40 layer + embed + tail Level Zero command lists (T5.3)\n");
  if (!record_command_lists()) return false;

  std::printf("[AInfer 258V] Unified Single-Process Runtime initialized successfully!\n");
  return true;
}

#define CHECK_L0_RET_NULL(expr) do { \
    ze_result_t _r = (expr); \
    if (_r != ZE_RESULT_SUCCESS) { \
      std::fprintf(stderr, "Level Zero error %d at %s:%d (%s)\n", (int)_r, __FILE__, __LINE__, #expr); \
      return nullptr; \
    } \
  } while (0)

ze_command_list_handle_t AInferRuntime258V::get_or_record_prefill_chunk_list(int B) {
  if (B < 1 || B > MAX_PREFILL_CHUNK) return nullptr;
  if (cmd_prefill_chunk_[B] != nullptr) {
    return cmd_prefill_chunk_[B];
  }

  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  ze_command_list_handle_t list = nullptr;
  CHECK_L0_RET_NULL(zeCommandListCreate(ctx_, dev_, &ldesc, &list));

  // 1. Embed gather batch: d_x_chunk_ = embed(d_tokens_chunk_, d_embed_tokens_)
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_embed_batch_, 0, sizeof(void *), &d_x_chunk_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_embed_batch_, 1, sizeof(void *), &d_embed_tokens_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_embed_batch_, 2, sizeof(void *), &d_tokens_chunk_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_embed_batch_, 3, sizeof(int), &B));
  CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_embed_batch_, 256, 1, 1));
  ze_group_count_t gcnt_embed{(uint32_t)((B * HIDDEN_DIM + 255) / 256), 1, 1};
  CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_embed_batch_, &gcnt_embed, nullptr, 0, nullptr));
  CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  // Helper lambda for GEMM append
  auto append_gemm = [&](float *Y, void *w, void *s, float *X, int M, int K) {
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 0, sizeof(void *), &Y));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 1, sizeof(void *), &w));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 2, sizeof(void *), &s));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 3, sizeof(void *), &X));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 4, sizeof(int), &M));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 5, sizeof(int), &K));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemm_prefill_, 6, sizeof(int), &B));
    CHECK_L0_VOID(zeKernelSetGroupSize(k_gemm_prefill_, 128, 1, 1));
    ze_group_count_t gc{(uint32_t)((M + 127) / 128), (uint32_t)((B + 31) / 32), 1};
    CHECK_L0_VOID(zeCommandListAppendLaunchKernel(list, k_gemm_prefill_, &gc, nullptr, 0, nullptr));
  };

  // 2. Iterate through 40 layers
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    LayerBinding &lb = layers_[l];

    // A. Input RMSNorm: d_x_norm_chunk_ = norm(d_x_chunk_, in_norm_w)
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_norm_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &lb.in_norm_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1));
    ze_group_count_t gcnt_norm{(uint32_t)B, 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_norm2048_batch_, &gcnt_norm, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    if (lb.is_full_attn) {
      // Full-Attention Layer (10 layers)
      append_gemm(d_q_proj_raw_chunk_, lb.q_proj_w, lb.q_proj_s, d_x_norm_chunk_, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemm(d_k_full_chunk_, lb.k_proj_w, lb.k_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemm(d_v_full_chunk_, lb.v_proj_w, lb.v_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Deinterleave Q and Gate
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 1, sizeof(void *), &d_gate_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 2, sizeof(void *), &d_q_proj_raw_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 3, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_deinterleave_qg_batch_, 256, 1, 1));
      ze_group_count_t gcnt_deint{(uint32_t)((B * NUM_Q_HEADS * HEAD_DIM + 255) / 256), 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_deinterleave_qg_batch_, &gcnt_deint, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Q Head Norm
      int total_nq = B * NUM_Q_HEADS;
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.q_norm_w));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nq));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_norm256_, 64, 1, 1));
      ze_group_count_t gcnt_nq{(uint32_t)total_nq, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_norm256_, &gcnt_nq, nullptr, 0, nullptr));

      // K Head Norm
      int total_nkv = B * NUM_KV_HEADS;
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.k_norm_w));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nkv));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_norm256_, 64, 1, 1));
      ze_group_count_t gcnt_nkv{(uint32_t)total_nkv, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_norm256_, &gcnt_nkv, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // RoPE & KV Cache Append
      uint32_t max_c = max_ctx_;
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 1, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 2, sizeof(void *), &d_v_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_rope_batch_, 7, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_rope_batch_, 256, 1, 1));
      ze_group_count_t gcnt_rope{(uint32_t)B, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_rope_batch_, &gcnt_rope, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // GQA Attention Prefill
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 0, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 1, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 2, sizeof(void *), &d_gate_full_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_attn_batch_, 7, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_attn_batch_, 256, 1, 1));
      ze_group_count_t gcnt_attn{(uint32_t)(B * NUM_Q_HEADS), 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_attn_batch_, &gcnt_attn, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Output Proj GEMM
      append_gemm(d_attn_proj_chunk_, lb.o_proj_w, lb.o_proj_s, d_attn_out_chunk_, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    } else {
      // DeltaNet Layer (30 layers)
      append_gemm(d_qkv_chunk_, lb.qkv_w, lb.qkv_s, d_x_norm_chunk_, C_QKV, HIDDEN_DIM);
      append_gemm(d_z_chunk_, lb.z_w, lb.z_s, d_x_norm_chunk_, H_V * S_V, HIDDEN_DIM);
      append_gemm(d_a_chunk_, lb.a_w, lb.a_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
      append_gemm(d_b_chunk_, lb.b_w, lb.b_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Conv1D Batch
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_conv_batch_, 0, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_conv_batch_, 1, sizeof(void *), &d_qkv_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_conv_batch_, 2, sizeof(void *), &lb.conv_state));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_conv_batch_, 3, sizeof(void *), &lb.conv_w));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_conv_batch_, 4, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_conv_batch_, 256, 1, 1));
      ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_conv_batch_, &gcnt_conv, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Head L2 Norm on Q and K Batch
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 0, sizeof(void *), &d_q_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 1, sizeof(void *), &d_k_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 2, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 3, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_l2_norm_qk_batch_, 128, 1, 1));
      ze_group_count_t gcnt_l2{(uint32_t)(B * H_K), 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_l2_norm_qk_batch_, &gcnt_l2, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Gate Prep Batch
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 0, sizeof(void *), &d_g_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 1, sizeof(void *), &d_beta_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 2, sizeof(void *), &d_a_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 3, sizeof(void *), &d_b_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 4, sizeof(void *), &lb.dt_bias));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 5, sizeof(void *), &lb.A_log));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_gate_prep_batch_, 6, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_gate_prep_batch_, 32, 1, 1));
      ze_group_count_t gcnt_gate{(uint32_t)B, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_gate_prep_batch_, &gcnt_gate, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // DeltaNet Recurrence Batch
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 0, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 1, sizeof(void *), &lb.ssm_state));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 2, sizeof(void *), &d_q_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 3, sizeof(void *), &d_k_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 4, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 5, sizeof(void *), &d_g_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 6, sizeof(void *), &d_beta_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_recr_batch_, 7, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_recr_batch_, 128, 1, 1));
      ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_recr_batch_, &gcnt_recr, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Head Norm & SiLU(Z) Batch
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_hnorm_batch_, 0, sizeof(void *), &d_attn_norm_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_hnorm_batch_, 1, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_hnorm_batch_, 2, sizeof(void *), &d_z_chunk_));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_hnorm_batch_, 3, sizeof(void *), &lb.ssm_norm_w));
      CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_hnorm_batch_, 4, sizeof(int), &B));
      CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_hnorm_batch_, 128, 1, 1));
      ze_group_count_t gcnt_hnorm{(uint32_t)(B * H_V), 1, 1};
      CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_hnorm_batch_, &gcnt_hnorm, nullptr, 0, nullptr));
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

      // Output Proj GEMM
      append_gemm(d_attn_proj_chunk_, lb.out_proj_w, lb.out_proj_s, d_attn_norm_chunk_, HIDDEN_DIM, H_V * S_V);
      CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    }

    // B. Mid Residual Add: d_x_mid_chunk_ = d_x_chunk_ + d_attn_proj_chunk_
    int total_res_mid = B * HIDDEN_DIM;
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_resadd_batch_, 0, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_resadd_batch_, 1, sizeof(void *), &d_x_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_resadd_batch_, 2, sizeof(void *), &d_attn_proj_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_resadd_batch_, 3, sizeof(int), &total_res_mid));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_resadd_batch_, 256, 1, 1));
    ze_group_count_t gcnt_res_mid{(uint32_t)((total_res_mid + 255) / 256), 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_resadd_batch_, &gcnt_res_mid, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // C. Post-Attn Norm: d_x_post_chunk_ = norm(d_x_mid_chunk_, post_norm_w)
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &lb.post_norm_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1));
    ze_group_count_t gcnt_norm_post{(uint32_t)B, 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_norm2048_batch_, &gcnt_norm_post, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // D. MoE Router Batch
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 0, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 1, sizeof(void *), &lb.router_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 2, sizeof(void *), &lb.shared_gate_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 3, sizeof(void *), &d_top_idx_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 4, sizeof(void *), &d_top_wt_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 5, sizeof(void *), &d_sh_gate_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_router_batch_, 6, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_router_batch_, 256, 1, 1));
    ze_group_count_t gcnt_router{(uint32_t)B, 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_router_batch_, &gcnt_router, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // D2. MoE Build Expert Bins
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 0, sizeof(void *), &d_expert_counts_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 1, sizeof(void *), &d_expert_offsets_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 2, sizeof(void *), &d_sorted_tokens_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 3, sizeof(void *), &d_sorted_slots_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 4, sizeof(void *), &d_top_idx_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_build_expert_bins_, 5, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_moe_build_expert_bins_, 256, 1, 1));
    ze_group_count_t gc_bins{1, 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_moe_build_expert_bins_, &gc_bins, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // E. MoE Active Experts Grouped Micro-GEMM (DPAS systolic acceleration)
    int K_gu = HIDDEN_DIM;
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 0, sizeof(void *), &d_exp_gu_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 1, sizeof(void *), &lb.exp_gu_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 2, sizeof(void *), &lb.exp_gu_s));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 3, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 4, sizeof(void *), &d_expert_counts_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 5, sizeof(void *), &d_expert_offsets_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 6, sizeof(void *), &d_sorted_tokens_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 7, sizeof(void *), &d_sorted_slots_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 8, sizeof(int), &K_gu));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 9, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_moe_gateup_grouped_batch_, 128, 1, 1));
    ze_group_count_t gc_gu_grouped{2048, 1, 1}; // 256 experts * 8 workgroups = 2048 workgroups
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_moe_gateup_grouped_batch_, &gc_gu_grouped, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_all_batch_, 0, sizeof(void *), &d_exp_act_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_all_batch_, 1, sizeof(void *), &d_exp_gu_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_all_batch_, 2, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_silu_all_batch_, 256, 1, 1));
    ze_group_count_t gc_silu_all{(uint32_t)((B * TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_silu_all_batch_, &gc_silu_all, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // F. MoE Active Experts Down Grouped Micro-GEMM (DPAS systolic acceleration)
    int K_dn = EXP_INTER_DIM; // 512
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 0, sizeof(void *), &d_exp_down_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 1, sizeof(void *), &lb.exp_dn_w));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 2, sizeof(void *), &lb.exp_dn_s));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 3, sizeof(void *), &d_exp_act_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 4, sizeof(void *), &d_expert_counts_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 5, sizeof(void *), &d_expert_offsets_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 6, sizeof(void *), &d_sorted_tokens_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 7, sizeof(void *), &d_sorted_slots_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 8, sizeof(int), &K_dn));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 9, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_moe_down_grouped_batch_, 128, 1, 1));
    ze_group_count_t gc_dn_grouped{4096, 1, 1}; // 256 experts * 16 workgroups = 4096 workgroups
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_moe_down_grouped_batch_, &gc_dn_grouped, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // F2. MoE Down Accumulation across active experts
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_accum_down_batch_, 0, sizeof(void *), &d_moe_acc_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_accum_down_batch_, 1, sizeof(void *), &d_exp_down_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_accum_down_batch_, 2, sizeof(void *), &d_top_wt_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_moe_accum_down_batch_, 3, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_moe_accum_down_batch_, 256, 1, 1));
    ze_group_count_t gc_accum_dn{(uint32_t)((B * HIDDEN_DIM + 255) / 256), 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_moe_accum_down_batch_, &gc_accum_dn, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // F. Shared Expert Batch
    append_gemm(d_sh_g_chunk_, lb.sh_gate_w, lb.sh_gate_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    append_gemm(d_sh_u_chunk_, lb.sh_up_w, lb.sh_up_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    int total_sh_act = B * EXP_INTER_DIM;
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_mul_batch_, 0, sizeof(void *), &d_sh_act_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_mul_batch_, 1, sizeof(void *), &d_sh_g_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_mul_batch_, 2, sizeof(void *), &d_sh_u_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_silu_mul_batch_, 3, sizeof(int), &total_sh_act));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_silu_mul_batch_, 256, 1, 1));
    ze_group_count_t gc_sh_silu{(uint32_t)((total_sh_act + 255) / 256), 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_silu_mul_batch_, &gc_sh_silu, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    append_gemm(d_sh_down_chunk_, lb.sh_down_w, lb.sh_down_s, d_sh_act_chunk_, HIDDEN_DIM, EXP_INTER_DIM);
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

    // G. Block Residual Add: d_x_chunk_ = d_x_mid_chunk_ + d_moe_acc_chunk_ + d_sh_down_chunk_ * d_sh_gate_chunk_
    int M_res = HIDDEN_DIM;
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 0, sizeof(void *), &d_x_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 1, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 2, sizeof(void *), &d_moe_acc_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 3, sizeof(void *), &d_sh_down_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 4, sizeof(void *), &d_sh_gate_chunk_));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 5, sizeof(int), &B));
    CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 6, sizeof(int), &M_res));
    CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_block_resadd_moe_batch_, 256, 1, 1));
    ze_group_count_t gc_block_res{(uint32_t)((B * M_res + 255) / 256), 1, 1};
    CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_block_resadd_moe_batch_, &gc_block_res, nullptr, 0, nullptr));
    CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
  }

  CHECK_L0_RET_NULL(zeCommandListClose(list));
  cmd_prefill_chunk_[B] = list;
  return list;
}

ze_command_list_handle_t AInferRuntime258V::get_or_record_prefill_tail_list(int B) {
  if (B < 1 || B > MAX_PREFILL_CHUNK) return nullptr;
  if (cmd_prefill_tail_[B] != nullptr) {
    return cmd_prefill_tail_[B];
  }

  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  ze_command_list_handle_t list = nullptr;
  CHECK_L0_RET_NULL(zeCommandListCreate(ctx_, dev_, &ldesc, &list));

  float *d_x_last = d_x_chunk_ + (size_t)(B - 1) * HIDDEN_DIM;
  // Ensure d_x_ holds the terminal hidden state for subsequent decode/MTP drafting
  CHECK_L0_RET_NULL(zeCommandListAppendMemoryCopy(list, d_x_, d_x_last, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  // Final RMSNorm directly on the last token of the chunk:
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_norm_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_last));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &d_final_norm_w_));
  CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  ze_group_count_t gcnt_norm2048{1, 1, 1};
  CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  // LM Head GEMV + Argmax Stage 1 fused [248320, 2048]
  int M_lm = VOCAB_SIZE;
  int K_lm = HIDDEN_DIM;
  void *null_logits = nullptr;
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 0, sizeof(void *), &null_logits));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 1, sizeof(void *), &d_lm_head_w_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 2, sizeof(void *), &d_lm_head_s_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 3, sizeof(void *), &d_x_norm_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 4, sizeof(void *), &d_stage1_vals_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 5, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 6, sizeof(int), &M_lm));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_lm_head_argmax1_, 7, sizeof(int), &K_lm));
  CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_lm_head_argmax1_, 256, 1, 1));
  ze_group_count_t gcnt_lm{(VOCAB_SIZE + 255) / 256, 1, 1};
  CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_lm_head_argmax1_, &gcnt_lm, nullptr, 0, nullptr));
  CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  // Argmax Stage 2 (Writes selected_token directly into d_ctrl_->selected_token)
  uint32_t num_stage1_groups = gcnt_lm.groupCountX;
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_argmax2_ctrl_, 0, sizeof(void *), &d_stage1_vals_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_argmax2_ctrl_, 1, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_argmax2_ctrl_, 2, sizeof(void *), &d_ctrl_));
  CHECK_L0_RET_NULL(zeKernelSetArgumentValue(k_argmax2_ctrl_, 3, sizeof(uint32_t), &num_stage1_groups));
  CHECK_L0_RET_NULL(zeKernelSetGroupSize(k_argmax2_ctrl_, 256, 1, 1));
  ze_group_count_t gcnt_arg2{1, 1, 1};
  CHECK_L0_RET_NULL(zeCommandListAppendLaunchKernel(list, k_argmax2_ctrl_, &gcnt_arg2, nullptr, 0, nullptr));
  CHECK_L0_RET_NULL(zeCommandListAppendBarrier(list, nullptr, 0, nullptr));

  CHECK_L0_RET_NULL(zeCommandListClose(list));
  cmd_prefill_tail_[B] = list;
  return list;
}

bool AInferRuntime258V::prefill(const std::vector<int> &prompt_ids, int *out_first_token) {
  if (prompt_ids.empty()) return false;
  int P = (int)prompt_ids.size();
  if (P > (int)max_ctx_) {
    std::fprintf(stderr, "Prompt size %d exceeds max context %u\n", P, max_ctx_);
    return false;
  }

  // Chunked in-memory prefill using batched GEMM & state updates (T3.1 / T5.2)
  int pos = 0;
  while (pos < P) {
    int B = std::min(P - pos, MAX_PREFILL_CHUNK);
    bool is_terminal = (pos + B == P);

    // Copy prompt chunk tokens to shared device memory
    std::memcpy(d_tokens_chunk_, &prompt_ids[pos], B * sizeof(int));

    // Update control block position (chunk start) and active length
    h_ctrl_.position = pos;
    h_ctrl_.active_length = pos + B;
    h_ctrl_.token_id = prompt_ids[pos];
    std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

    // Execute chunk forward pass across all 40 layers in a single recorded command list
    ze_command_list_handle_t cmd_chunk = get_or_record_prefill_chunk_list(B);
    if (!cmd_chunk) return false;

    CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_chunk, fence_));
    CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
    CHECK_L0(zeFenceReset(fence_));

    if (is_terminal) {
      // Execute terminal tail list (final norm + LM head + argmax)
      ze_command_list_handle_t cmd_tail = get_or_record_prefill_tail_list(B);
      if (!cmd_tail) return false;

      CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_tail, fence_));
      CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
      CHECK_L0(zeFenceReset(fence_));

      // Read back terminal sampled / argmax token directly from control block (T5.4)
      int next_tok = d_ctrl_->selected_token;
      h_ctrl_.selected_token = next_tok;
      h_ctrl_.token_id = next_tok;
      h_ctrl_.position = P - 1;
      h_ctrl_.active_length = P;
      std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

      if (out_first_token) *out_first_token = next_tok;
    }

    pos += B;
  }

  // Queue synchronization to guarantee all prefill state updates are committed to device memory
  CHECK_L0(zeCommandQueueSynchronize(queue_, UINT64_MAX));
  return true;
}

bool AInferRuntime258V::decode_step(int *out_next_token) {
  // Update control block position and input token (T5.4)
  h_ctrl_.token_id = h_ctrl_.selected_token;
  h_ctrl_.position += 1;
  h_ctrl_.active_length += 1;

  if (h_ctrl_.position >= (int)max_ctx_) {
    std::fprintf(stderr, "Decode reached max context limit %u\n", max_ctx_);
    return false;
  }

  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  // Execute unified pre-recorded command list (embed + 40 layers + tail) with zero allocations (T5.3/T5.4)
  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_step_, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));

  int next_tok = d_ctrl_->selected_token;
  h_ctrl_.selected_token = next_tok;
  if (out_next_token) *out_next_token = next_tok;

  return true;
}

bool AInferRuntime258V::teacher_forced_eval(const std::vector<int> &prompt_ids, std::vector<int> &out_predicted_tokens) {
  if (prompt_ids.empty()) return false;
  int P = (int)prompt_ids.size();
  if (P > (int)max_ctx_) {
    std::fprintf(stderr, "Prompt size %d exceeds max context %u\n", P, max_ctx_);
    return false;
  }
  out_predicted_tokens.clear();
  out_predicted_tokens.reserve(P);

  for (int p = 0; p < P; ++p) {
    h_ctrl_.token_id = prompt_ids[p];
    h_ctrl_.position = p;
    h_ctrl_.active_length = p + 1;
    std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

    CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_step_, fence_));
    CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
    CHECK_L0(zeFenceReset(fence_));

    out_predicted_tokens.push_back(d_ctrl_->selected_token);
  }
  return true;
}

bool AInferRuntime258V::generate(const std::vector<int> &prompt_ids, int max_new_tokens,
                                 std::vector<int> &generated_ids, double *out_prefill_ms,
                                 double *out_decode_tok_per_s) {
  generated_ids.clear();
  if (prompt_ids.empty()) return false;

  // 1. In-memory Prefill (T5.2)
  auto t_pref0 = std::chrono::steady_clock::now();
  int first_tok = 0;
  if (!prefill(prompt_ids, &first_tok)) return false;
  auto t_pref1 = std::chrono::steady_clock::now();
  double pref_ms = std::chrono::duration<double, std::milli>(t_pref1 - t_pref0).count();
  if (out_prefill_ms) *out_prefill_ms = pref_ms;

  generated_ids.push_back(first_tok);

  // 2. In-memory Steady-State Decode (T5.3/T5.4)
  auto t_dec0 = std::chrono::steady_clock::now();
  for (int g = 1; g < max_new_tokens; ++g) {
    int next_tok = 0;
    if (!decode_step(&next_tok)) break;
    generated_ids.push_back(next_tok);
    // Break on standard EOS (248044 / 248046)
    if (next_tok == 248044 || next_tok == 248046) break;
  }
  auto t_dec1 = std::chrono::steady_clock::now();
  double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
  if (out_decode_tok_per_s && dec_s > 0 && generated_ids.size() > 1) {
    *out_decode_tok_per_s = (generated_ids.size() - 1) / dec_s;
  }

  return true;
}

bool AInferRuntime258V::profile_step_breakdown(double &embed_ms, double &layers_ms, double &tail_ms,
                                              double &step_ms, std::vector<double> &layer_times_ms) {
  // Update control block for 1 decode step
  h_ctrl_.token_id = h_ctrl_.selected_token;
  h_ctrl_.position += 1;
  h_ctrl_.active_length += 1;
  if (h_ctrl_.position >= (int)max_ctx_) return false;
  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  // 1. Time cmd_embed_
  auto t_start = std::chrono::steady_clock::now();
  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_embed_, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));
  auto t_embed = std::chrono::steady_clock::now();
  embed_ms = std::chrono::duration<double, std::milli>(t_embed - t_start).count();

  // 2. Time layers individually
  layer_times_ms.resize(TOTAL_LAYERS);
  auto t_layers_start = std::chrono::steady_clock::now();
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    auto t_l0 = std::chrono::steady_clock::now();
    CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_layers_[l], fence_));
    CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
    CHECK_L0(zeFenceReset(fence_));
    auto t_l1 = std::chrono::steady_clock::now();
    layer_times_ms[l] = std::chrono::duration<double, std::milli>(t_l1 - t_l0).count();
  }
  auto t_layers_end = std::chrono::steady_clock::now();
  layers_ms = std::chrono::duration<double, std::milli>(t_layers_end - t_layers_start).count();

  // 3. Time cmd_tail_
  auto t_tail_start = std::chrono::steady_clock::now();
  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_tail_, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));
  auto t_tail_end = std::chrono::steady_clock::now();
  tail_ms = std::chrono::duration<double, std::milli>(t_tail_end - t_tail_start).count();

  h_ctrl_.selected_token = d_ctrl_->selected_token;

  // 4. Time unified cmd_step_
  h_ctrl_.token_id = h_ctrl_.selected_token;
  h_ctrl_.position += 1;
  h_ctrl_.active_length += 1;
  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  auto t_step0 = std::chrono::steady_clock::now();
  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_step_, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));
  auto t_step1 = std::chrono::steady_clock::now();
  step_ms = std::chrono::duration<double, std::milli>(t_step1 - t_step0).count();
  h_ctrl_.selected_token = d_ctrl_->selected_token;

  return true;
}

bool AInferRuntime258V::profile_prefill_breakdown(int B) {
  if (B < 1 || B > MAX_PREFILL_CHUNK) return false;

  auto append_gemm_to_list = [&](ze_command_list_handle_t list, float *Y, void *w, void *s, float *X, int M, int K) {
    zeKernelSetArgumentValue(k_gemm_prefill_, 0, sizeof(void *), &Y);
    zeKernelSetArgumentValue(k_gemm_prefill_, 1, sizeof(void *), &w);
    zeKernelSetArgumentValue(k_gemm_prefill_, 2, sizeof(void *), &s);
    zeKernelSetArgumentValue(k_gemm_prefill_, 3, sizeof(void *), &X);
    zeKernelSetArgumentValue(k_gemm_prefill_, 4, sizeof(int), &M);
    zeKernelSetArgumentValue(k_gemm_prefill_, 5, sizeof(int), &K);
    zeKernelSetArgumentValue(k_gemm_prefill_, 6, sizeof(int), &B);
    zeKernelSetGroupSize(k_gemm_prefill_, 128, 1, 1);
    ze_group_count_t gc{(uint32_t)((M + 127) / 128), (uint32_t)((B + 31) / 32), 1};
    zeCommandListAppendLaunchKernel(list, k_gemm_prefill_, &gc, nullptr, 0, nullptr);
  };

  auto time_cmd = [&](auto record_fn) -> double {
    ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
    ze_command_list_handle_t list = nullptr;
    if (zeCommandListCreate(ctx_, dev_, &ldesc, &list) != ZE_RESULT_SUCCESS) return 0.0;
    record_fn(list);
    zeCommandListClose(list);

    // Warmup
    zeCommandQueueExecuteCommandLists(queue_, 1, &list, fence_);
    zeFenceHostSynchronize(fence_, UINT64_MAX);
    zeFenceReset(fence_);

    const int RUNS = 5;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
      zeCommandQueueExecuteCommandLists(queue_, 1, &list, fence_);
      zeFenceHostSynchronize(fence_, UINT64_MAX);
      zeFenceReset(fence_);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
    zeCommandListDestroy(list);
    return ms;
  };

  const LayerBinding &l0 = layers_[0];
  const LayerBinding &l3 = layers_[3];

  std::printf("\n=================================================================\n");
  std::printf("--- Prefill Component Profiling Breakdown (B = %d) ---------------\n", B);
  std::printf("=================================================================\n");

  // 1. Embed Gather
  double t_embed = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_embed_batch_, 0, sizeof(void *), &d_x_chunk_);
    zeKernelSetArgumentValue(k_embed_batch_, 1, sizeof(void *), &d_embed_tokens_);
    zeKernelSetArgumentValue(k_embed_batch_, 2, sizeof(void *), &d_tokens_chunk_);
    zeKernelSetArgumentValue(k_embed_batch_, 3, sizeof(int), &B);
    zeKernelSetGroupSize(k_embed_batch_, 256, 1, 1);
    ze_group_count_t gc{(uint32_t)((B * HIDDEN_DIM + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_embed_batch_, &gc, nullptr, 0, nullptr);
  });
  std::printf("[Embed]   Token Gather:                   %6.3f ms\n", t_embed);

  // 2. DeltaNet Dense Projections (Layer 0)
  double t_qkv = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_qkv_chunk_, l0.qkv_w, l0.qkv_s, d_x_norm_chunk_, C_QKV, HIDDEN_DIM);
  });
  double t_z_ba = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_z_chunk_, l0.z_w, l0.z_s, d_x_norm_chunk_, H_V * S_V, HIDDEN_DIM);
    append_gemm_to_list(list, d_a_chunk_, l0.a_w, l0.a_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
    append_gemm_to_list(list, d_b_chunk_, l0.b_w, l0.b_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
  });
  double t_dn_out = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_attn_proj_chunk_, l0.out_proj_w, l0.out_proj_s, d_attn_norm_chunk_, HIDDEN_DIM, H_V * S_V);
  });

  // 3. DeltaNet Recurrence Core
  double t_conv = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_conv_batch_, 0, sizeof(void *), &d_qkv_conv_chunk_);
    zeKernelSetArgumentValue(k_conv_batch_, 1, sizeof(void *), &d_qkv_chunk_);
    zeKernelSetArgumentValue(k_conv_batch_, 2, sizeof(void *), &l0.conv_state);
    zeKernelSetArgumentValue(k_conv_batch_, 3, sizeof(void *), &l0.conv_w);
    zeKernelSetArgumentValue(k_conv_batch_, 4, sizeof(int), &B);
    zeKernelSetGroupSize(k_conv_batch_, 256, 1, 1);
    ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_conv_batch_, &gcnt_conv, nullptr, 0, nullptr);
  });

  double t_l2 = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 0, sizeof(void *), &d_q_chunk_);
    zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 1, sizeof(void *), &d_k_chunk_);
    zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 2, sizeof(void *), &d_qkv_conv_chunk_);
    zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 3, sizeof(int), &B);
    zeKernelSetGroupSize(k_l2_norm_qk_batch_, 128, 1, 1);
    ze_group_count_t gcnt_l2{(uint32_t)(B * H_K), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_l2_norm_qk_batch_, &gcnt_l2, nullptr, 0, nullptr);
  });

  double t_gate = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_gate_prep_batch_, 0, sizeof(void *), &d_g_chunk_);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 1, sizeof(void *), &d_beta_chunk_);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 2, sizeof(void *), &d_a_chunk_);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 3, sizeof(void *), &d_b_chunk_);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 4, sizeof(void *), &l0.dt_bias);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 5, sizeof(void *), &l0.A_log);
    zeKernelSetArgumentValue(k_gate_prep_batch_, 6, sizeof(int), &B);
    zeKernelSetGroupSize(k_gate_prep_batch_, 32, 1, 1);
    ze_group_count_t gcnt_gate{(uint32_t)B, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_gate_prep_batch_, &gcnt_gate, nullptr, 0, nullptr);
  });

  double t_recr_loop = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_recr_batch_, 0, sizeof(void *), &d_attn_out_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 1, sizeof(void *), &l0.ssm_state);
    zeKernelSetArgumentValue(k_recr_batch_, 2, sizeof(void *), &d_q_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 3, sizeof(void *), &d_k_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 4, sizeof(void *), &d_qkv_conv_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 5, sizeof(void *), &d_g_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 6, sizeof(void *), &d_beta_chunk_);
    zeKernelSetArgumentValue(k_recr_batch_, 7, sizeof(int), &B);
    zeKernelSetGroupSize(k_recr_batch_, 128, 1, 1);
    ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_recr_batch_, &gcnt_recr, nullptr, 0, nullptr);
  });

  double t_hnorm = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_hnorm_batch_, 0, sizeof(void *), &d_attn_norm_chunk_);
    zeKernelSetArgumentValue(k_hnorm_batch_, 1, sizeof(void *), &d_attn_out_chunk_);
    zeKernelSetArgumentValue(k_hnorm_batch_, 2, sizeof(void *), &d_z_chunk_);
    zeKernelSetArgumentValue(k_hnorm_batch_, 3, sizeof(void *), &l0.ssm_norm_w);
    zeKernelSetArgumentValue(k_hnorm_batch_, 4, sizeof(int), &B);
    zeKernelSetGroupSize(k_hnorm_batch_, 128, 1, 1);
    ze_group_count_t gcnt_hnorm{(uint32_t)(B * H_V), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_hnorm_batch_, &gcnt_hnorm, nullptr, 0, nullptr);
  });

  std::printf("[DeltaNet] QKV Proj (8192x2048):           %6.3f ms\n", t_qkv);
  std::printf("[DeltaNet] Z+A+B Projs (2304x2048):         %6.3f ms\n", t_z_ba);
  std::printf("[DeltaNet] Conv1D Update SiLU:              %6.3f ms\n", t_conv);
  std::printf("[DeltaNet] Head L2 Norm (Q, K):             %6.3f ms\n", t_l2);
  std::printf("[DeltaNet] Gate Prep:                       %6.3f ms\n", t_gate);
  std::printf("[DeltaNet] Recurrence Loop:                 %6.3f ms\n", t_recr_loop);
  std::printf("[DeltaNet] Head Norm & SiLU(Z):             %6.3f ms\n", t_hnorm);
  std::printf("[DeltaNet] Out Proj (2048x2048):            %6.3f ms\n", t_dn_out);

  // 4. MoE Pipeline (Layer 0)
  double t_router = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_post_chunk_);
    zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_mid_chunk_);
    zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &l0.post_norm_w);
    zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B);
    zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1);
    ze_group_count_t gcnt_norm{(uint32_t)B, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_norm2048_batch_, &gcnt_norm, nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);

    zeKernelSetArgumentValue(k_router_batch_, 0, sizeof(void *), &d_x_post_chunk_);
    zeKernelSetArgumentValue(k_router_batch_, 1, sizeof(void *), &l0.router_w);
    zeKernelSetArgumentValue(k_router_batch_, 2, sizeof(void *), &l0.shared_gate_w);
    zeKernelSetArgumentValue(k_router_batch_, 3, sizeof(void *), &d_top_idx_chunk_);
    zeKernelSetArgumentValue(k_router_batch_, 4, sizeof(void *), &d_top_wt_chunk_);
    zeKernelSetArgumentValue(k_router_batch_, 5, sizeof(void *), &d_sh_gate_chunk_);
    zeKernelSetArgumentValue(k_router_batch_, 6, sizeof(int), &B);
    zeKernelSetGroupSize(k_router_batch_, 256, 1, 1);
    ze_group_count_t gc_router{(uint32_t)B, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_router_batch_, &gc_router, nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);

    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 0, sizeof(void *), &d_expert_counts_);
    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 1, sizeof(void *), &d_expert_offsets_);
    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 2, sizeof(void *), &d_sorted_tokens_);
    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 3, sizeof(void *), &d_sorted_slots_);
    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 4, sizeof(void *), &d_top_idx_chunk_);
    zeKernelSetArgumentValue(k_moe_build_expert_bins_, 5, sizeof(int), &B);
    zeKernelSetGroupSize(k_moe_build_expert_bins_, 256, 1, 1);
    ze_group_count_t gc_bins{1, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_moe_build_expert_bins_, &gc_bins, nullptr, 0, nullptr);
  });

  int K_gu = HIDDEN_DIM;
  double t_moe_gu = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 0, sizeof(void *), &d_exp_gu_chunk_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 1, sizeof(void *), &l0.exp_gu_w);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 2, sizeof(void *), &l0.exp_gu_s);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 3, sizeof(void *), &d_x_post_chunk_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 4, sizeof(void *), &d_expert_counts_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 5, sizeof(void *), &d_expert_offsets_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 6, sizeof(void *), &d_sorted_tokens_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 7, sizeof(void *), &d_sorted_slots_);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 8, sizeof(int), &K_gu);
    zeKernelSetArgumentValue(k_moe_gateup_grouped_batch_, 9, sizeof(int), &B);
    zeKernelSetGroupSize(k_moe_gateup_grouped_batch_, 128, 1, 1);
    ze_group_count_t gc_gu_grouped{2048, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_moe_gateup_grouped_batch_, &gc_gu_grouped, nullptr, 0, nullptr);
  });

  double t_moe_silu = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_silu_all_batch_, 0, sizeof(void *), &d_exp_act_chunk_);
    zeKernelSetArgumentValue(k_silu_all_batch_, 1, sizeof(void *), &d_exp_gu_chunk_);
    zeKernelSetArgumentValue(k_silu_all_batch_, 2, sizeof(int), &B);
    zeKernelSetGroupSize(k_silu_all_batch_, 256, 1, 1);
    ze_group_count_t gc_silu_all{(uint32_t)((B * TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_silu_all_batch_, &gc_silu_all, nullptr, 0, nullptr);
  });

  int K_dn = EXP_INTER_DIM;
  double t_moe_dn = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 0, sizeof(void *), &d_exp_down_chunk_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 1, sizeof(void *), &l0.exp_dn_w);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 2, sizeof(void *), &l0.exp_dn_s);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 3, sizeof(void *), &d_exp_act_chunk_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 4, sizeof(void *), &d_expert_counts_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 5, sizeof(void *), &d_expert_offsets_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 6, sizeof(void *), &d_sorted_tokens_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 7, sizeof(void *), &d_sorted_slots_);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 8, sizeof(int), &K_dn);
    zeKernelSetArgumentValue(k_moe_down_grouped_batch_, 9, sizeof(int), &B);
    zeKernelSetGroupSize(k_moe_down_grouped_batch_, 128, 1, 1);
    ze_group_count_t gc_dn_grouped{4096, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_moe_down_grouped_batch_, &gc_dn_grouped, nullptr, 0, nullptr);
  });

  double t_moe_accum = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_moe_accum_down_batch_, 0, sizeof(void *), &d_moe_acc_chunk_);
    zeKernelSetArgumentValue(k_moe_accum_down_batch_, 1, sizeof(void *), &d_exp_down_chunk_);
    zeKernelSetArgumentValue(k_moe_accum_down_batch_, 2, sizeof(void *), &d_top_wt_chunk_);
    zeKernelSetArgumentValue(k_moe_accum_down_batch_, 3, sizeof(int), &B);
    zeKernelSetGroupSize(k_moe_accum_down_batch_, 256, 1, 1);
    ze_group_count_t gc_accum_dn{(uint32_t)((B * HIDDEN_DIM + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_moe_accum_down_batch_, &gc_accum_dn, nullptr, 0, nullptr);
  });

  double t_sh_exp = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_sh_g_chunk_, l0.sh_gate_w, l0.sh_gate_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    append_gemm_to_list(list, d_sh_u_chunk_, l0.sh_up_w, l0.sh_up_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);

    int total_sh_act = B * EXP_INTER_DIM;
    zeKernelSetArgumentValue(k_silu_mul_batch_, 0, sizeof(void *), &d_sh_act_chunk_);
    zeKernelSetArgumentValue(k_silu_mul_batch_, 1, sizeof(void *), &d_sh_g_chunk_);
    zeKernelSetArgumentValue(k_silu_mul_batch_, 2, sizeof(void *), &d_sh_u_chunk_);
    zeKernelSetArgumentValue(k_silu_mul_batch_, 3, sizeof(int), &total_sh_act);
    zeKernelSetGroupSize(k_silu_mul_batch_, 256, 1, 1);
    ze_group_count_t gc_sh_silu{(uint32_t)((total_sh_act + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_silu_mul_batch_, &gc_sh_silu, nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);

    append_gemm_to_list(list, d_sh_down_chunk_, l0.sh_down_w, l0.sh_down_s, d_sh_act_chunk_, HIDDEN_DIM, EXP_INTER_DIM);
  });

  std::printf("[MoE]     Router + Binning:               %6.3f ms\n", t_router);
  std::printf("[MoE]     GateUp Grouped (DPAS):          %6.3f ms\n", t_moe_gu);
  std::printf("[MoE]     SiLU Activation:                %6.3f ms\n", t_moe_silu);
  std::printf("[MoE]     Down Grouped (DPAS):            %6.3f ms\n", t_moe_dn);
  std::printf("[MoE]     Down Accumulation:              %6.3f ms\n", t_moe_accum);
  std::printf("[MoE]     Shared Expert (3 GEMMs):        %6.3f ms\n", t_sh_exp);

  double per_dn_layer = t_qkv + t_z_ba + t_conv + t_l2 + t_gate + t_recr_loop + t_hnorm + t_dn_out + t_router + t_moe_gu + t_moe_silu + t_moe_dn + t_moe_accum + t_sh_exp;
  std::printf("-----------------------------------------------------------------\n");
  std::printf("  1 DeltaNet Layer Total:                 %6.3f ms\n", per_dn_layer);
  std::printf("  Extrapolated 30 DeltaNet Layers:        %6.3f ms\n", per_dn_layer * 30.0);

  // 5. Full-Attention Layer (Layer 3 bindings; MoE tail shared with DeltaNet)
  const LayerBinding &l3b = layers_[3];
  double t_fa_qkv = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_q_proj_raw_chunk_, l3b.q_proj_w, l3b.q_proj_s, d_x_norm_chunk_, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
    append_gemm_to_list(list, d_k_full_chunk_, l3b.k_proj_w, l3b.k_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
    append_gemm_to_list(list, d_v_full_chunk_, l3b.v_proj_w, l3b.v_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  });
  double t_fa_prep = time_cmd([&](ze_command_list_handle_t list) {
    zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 0, sizeof(void *), &d_q_full_chunk_);
    zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 1, sizeof(void *), &d_gate_full_chunk_);
    zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 2, sizeof(void *), &d_q_proj_raw_chunk_);
    zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 3, sizeof(int), &B);
    zeKernelSetGroupSize(k_deinterleave_qg_batch_, 256, 1, 1);
    ze_group_count_t gcnt_deint{(uint32_t)((B * NUM_Q_HEADS * HEAD_DIM + 255) / 256), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_deinterleave_qg_batch_, &gcnt_deint, nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    int total_nq = B * NUM_Q_HEADS;
    zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_q_full_chunk_);
    zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_q_full_chunk_);
    zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &l3b.q_norm_w);
    zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nq);
    zeKernelSetGroupSize(k_norm256_, 64, 1, 1);
    ze_group_count_t gcnt_nq{(uint32_t)total_nq, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_norm256_, &gcnt_nq, nullptr, 0, nullptr);
    int total_nkv = B * NUM_KV_HEADS;
    zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_k_full_chunk_);
    zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_k_full_chunk_);
    zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &l3b.k_norm_w);
    zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nkv);
    zeKernelSetGroupSize(k_norm256_, 64, 1, 1);
    ze_group_count_t gcnt_nkv{(uint32_t)total_nkv, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_norm256_, &gcnt_nkv, nullptr, 0, nullptr);
    zeCommandListAppendBarrier(list, nullptr, 0, nullptr);
    uint32_t max_c = max_ctx_;
    zeKernelSetArgumentValue(k_rope_batch_, 0, sizeof(void *), &d_q_full_chunk_);
    zeKernelSetArgumentValue(k_rope_batch_, 1, sizeof(void *), &d_k_full_chunk_);
    zeKernelSetArgumentValue(k_rope_batch_, 2, sizeof(void *), &d_v_full_chunk_);
    zeKernelSetArgumentValue(k_rope_batch_, 3, sizeof(void *), &l3b.k_cache);
    zeKernelSetArgumentValue(k_rope_batch_, 4, sizeof(void *), &l3b.v_cache);
    zeKernelSetArgumentValue(k_rope_batch_, 5, sizeof(void *), &d_ctrl_);
    zeKernelSetArgumentValue(k_rope_batch_, 6, sizeof(uint32_t), &max_c);
    zeKernelSetArgumentValue(k_rope_batch_, 7, sizeof(int), &B);
    zeKernelSetGroupSize(k_rope_batch_, 256, 1, 1);
    ze_group_count_t gcnt_rope{(uint32_t)B, 1, 1};
    zeCommandListAppendLaunchKernel(list, k_rope_batch_, &gcnt_rope, nullptr, 0, nullptr);
  });
  // d_ctrl_ is host-visible shared memory: pin base position to 0 so the
  // attention timing reflects a fresh prefill (queries attend t=0..b),
  // not whatever position a prior prefill left behind.
  d_ctrl_->position = 0;
  double t_fa_attn = time_cmd([&](ze_command_list_handle_t list) {
    uint32_t max_c = max_ctx_;
    zeKernelSetArgumentValue(k_attn_batch_, 0, sizeof(void *), &d_attn_out_chunk_);
    zeKernelSetArgumentValue(k_attn_batch_, 1, sizeof(void *), &d_q_full_chunk_);
    zeKernelSetArgumentValue(k_attn_batch_, 2, sizeof(void *), &d_gate_full_chunk_);
    zeKernelSetArgumentValue(k_attn_batch_, 3, sizeof(void *), &l3b.k_cache);
    zeKernelSetArgumentValue(k_attn_batch_, 4, sizeof(void *), &l3b.v_cache);
    zeKernelSetArgumentValue(k_attn_batch_, 5, sizeof(void *), &d_ctrl_);
    zeKernelSetArgumentValue(k_attn_batch_, 6, sizeof(uint32_t), &max_c);
    zeKernelSetArgumentValue(k_attn_batch_, 7, sizeof(int), &B);
    zeKernelSetGroupSize(k_attn_batch_, 256, 1, 1);
    ze_group_count_t gcnt_attn{(uint32_t)(B * NUM_Q_HEADS), 1, 1};
    zeCommandListAppendLaunchKernel(list, k_attn_batch_, &gcnt_attn, nullptr, 0, nullptr);
  });
  double t_fa_out = time_cmd([&](ze_command_list_handle_t list) {
    append_gemm_to_list(list, d_attn_proj_chunk_, l3b.o_proj_w, l3b.o_proj_s, d_attn_out_chunk_, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
  });
  double t_moe_tail = t_router + t_moe_gu + t_moe_silu + t_moe_dn + t_moe_accum + t_sh_exp;
  double per_fa_layer = t_fa_qkv + t_fa_prep + t_fa_attn + t_fa_out + t_moe_tail;
  std::printf("-----------------------------------------------------------------\n");
  std::printf("[FullAttn] QKV Projs:                      %6.3f ms\n", t_fa_qkv);
  std::printf("[FullAttn] Deinterleave+Norms+RoPE:        %6.3f ms\n", t_fa_prep);
  std::printf("[FullAttn] GQA Attention:                  %6.3f ms\n", t_fa_attn);
  std::printf("[FullAttn] Out Proj:                       %6.3f ms\n", t_fa_out);
  std::printf("[FullAttn] MoE Tail (shared):              %6.3f ms\n", t_moe_tail);
  std::printf("  1 Full-Attn Layer Total:                %6.3f ms\n", per_fa_layer);
  std::printf("  Extrapolated 10 Full-Attn Layers:       %6.3f ms\n", per_fa_layer * 10.0);
  std::printf("  Full-model estimate (30xDN + 10xFA):    %6.3f ms\n",
              per_dn_layer * 30.0 + per_fa_layer * 10.0);
  std::printf("=================================================================\n\n");

  return true;
}

bool AInferRuntime258V::export_diagnostic_cache(const std::string &cache_file, uint32_t pos) {
  DiagnosticCacheHeader hdr{};
  std::strncpy(hdr.magic, "AINFER_CACHE_V1", sizeof(hdr.magic) - 1);
  hdr.version = 1;
  hdr.model_crc = 0x35BA3B01;
  hdr.position = pos;
  hdr.max_ctx = max_ctx_;
  hdr.num_kv_layers = NUM_FULL_ATTN_LAYERS;
  hdr.num_ssm_layers = NUM_DELTANET_LAYERS;
  hdr.kv_bytes_total = kv_cache_bytes_;
  hdr.ssm_bytes_total = (size_t)NUM_DELTANET_LAYERS * H_V * S_V * S_V * sizeof(float);
  hdr.conv_bytes_total = (size_t)NUM_DELTANET_LAYERS * C_QKV * 3 * sizeof(float);

  // Download all device states via immediate list
  std::vector<uint8_t> kv_buf(hdr.kv_bytes_total);
  std::vector<uint8_t> ssm_buf(hdr.ssm_bytes_total);
  std::vector<uint8_t> conv_buf(hdr.conv_bytes_total);

  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, kv_buf.data(), kv_cache_arena_, hdr.kv_bytes_total, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, ssm_buf.data(), ssm_recr_arena_, hdr.ssm_bytes_total, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, conv_buf.data(), ssm_conv_arena_, hdr.conv_bytes_total, nullptr, 0, nullptr));

  uint32_t crc = crc32_compute(0, kv_buf.data(), kv_buf.size());
  crc = crc32_compute(crc, ssm_buf.data(), ssm_buf.size());
  crc = crc32_compute(crc, conv_buf.data(), conv_buf.size());
  hdr.payload_crc32 = crc;

  std::ofstream out(cache_file, std::ios::binary);
  if (!out) return false;
  out.write((const char *)&hdr, sizeof(hdr));
  out.write((const char *)kv_buf.data(), kv_buf.size());
  out.write((const char *)ssm_buf.data(), ssm_buf.size());
  out.write((const char *)conv_buf.data(), conv_buf.size());
  return true;
}

bool AInferRuntime258V::import_diagnostic_cache(const std::string &cache_file, uint32_t *out_pos) {
  std::ifstream in(cache_file, std::ios::binary);
  if (!in) return false;

  DiagnosticCacheHeader hdr{};
  in.read((char *)&hdr, sizeof(hdr));
  if (std::memcmp(hdr.magic, "AINFER_CACHE_V1", 15) != 0 || hdr.version != 1) {
    std::fprintf(stderr, "Invalid diagnostic cache header\n");
    return false;
  }

  if (hdr.kv_bytes_total != kv_cache_bytes_ ||
      hdr.ssm_bytes_total != (size_t)NUM_DELTANET_LAYERS * H_V * S_V * S_V * sizeof(float) ||
      hdr.conv_bytes_total != (size_t)NUM_DELTANET_LAYERS * C_QKV * 3 * sizeof(float)) {
    std::fprintf(stderr, "Diagnostic cache geometry mismatch\n");
    return false;
  }

  std::vector<uint8_t> kv_buf(hdr.kv_bytes_total);
  std::vector<uint8_t> ssm_buf(hdr.ssm_bytes_total);
  std::vector<uint8_t> conv_buf(hdr.conv_bytes_total);

  in.read((char *)kv_buf.data(), kv_buf.size());
  in.read((char *)ssm_buf.data(), ssm_buf.size());
  in.read((char *)conv_buf.data(), conv_buf.size());

  uint32_t crc = crc32_compute(0, kv_buf.data(), kv_buf.size());
  crc = crc32_compute(crc, ssm_buf.data(), ssm_buf.size());
  crc = crc32_compute(crc, conv_buf.data(), conv_buf.size());

  if (crc != hdr.payload_crc32) {
    std::fprintf(stderr, "Diagnostic cache checksum corrupted (calc=%08x expected=%08x)\n", crc, hdr.payload_crc32);
    return false;
  }

  // Restore states to GPU
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, kv_cache_arena_, kv_buf.data(), hdr.kv_bytes_total, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, ssm_recr_arena_, ssm_buf.data(), hdr.ssm_bytes_total, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, ssm_conv_arena_, conv_buf.data(), hdr.conv_bytes_total, nullptr, 0, nullptr));

  h_ctrl_.position = hdr.position;
  h_ctrl_.active_length = hdr.position;
  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  if (out_pos) *out_pos = hdr.position;
  return true;
}

bool AInferRuntime258V::reset_state() {
  if (!cmd_copy_) return false;

  // Ensure queue is idle before clearing memory
  if (queue_) {
    CHECK_L0(zeCommandQueueSynchronize(queue_, UINT64_MAX));
  }

  // Zero out KV cache arena (T5.6)
  uint32_t zero = 0;
  if (kv_cache_arena_ && kv_cache_bytes_ > 0) {
    CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, kv_cache_arena_, &zero, sizeof(zero), kv_cache_bytes_, nullptr, 0, nullptr));
  }

  // Zero out SSM state arenas (T5.6)
  size_t total_ssm_recr = (size_t)NUM_DELTANET_LAYERS * H_V * S_V * S_V * sizeof(float);
  size_t total_ssm_conv = (size_t)NUM_DELTANET_LAYERS * C_QKV * 3 * sizeof(float);
  if (ssm_recr_arena_) {
    CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, ssm_recr_arena_, &zero, sizeof(zero), total_ssm_recr, nullptr, 0, nullptr));
  }
  if (ssm_conv_arena_) {
    CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, ssm_conv_arena_, &zero, sizeof(zero), total_ssm_conv, nullptr, 0, nullptr));
  }

  if (d_x_) {
    CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, d_x_, &zero, sizeof(zero), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
  }

  // Reset host and device control block
  std::memset(&h_ctrl_, 0, sizeof(h_ctrl_));
  h_ctrl_.temperature = 0.0f; // Default greedy
  if (d_ctrl_) {
    std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));
  }

  // Zero out MTP KV cache and buffers if initialized
  if (mtp_.initialized) {
    if (mtp_.k_cache) {
      size_t kv_sz = (size_t)NUM_KV_HEADS * max_ctx_ * HEAD_DIM * sizeof(uint16_t);
      CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, mtp_.k_cache, &zero, sizeof(zero), kv_sz, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, mtp_.v_cache, &zero, sizeof(zero), kv_sz, nullptr, 0, nullptr));
    }
    if (mtp_.d_x) {
      CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, mtp_.d_x, &zero, sizeof(zero), HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
    }
    if (mtp_.d_draft_token) {
      *mtp_.d_draft_token = 0;
    }
  }

  pending_draft_token_ = -1;
  if (d_verify_tokens_) {
    d_verify_tokens_[0] = 0;
    d_verify_tokens_[1] = 0;
  }

  // Synchronize immediate copy list to ensure all fills have completed on GPU
  CHECK_L0(zeCommandListHostSynchronize(cmd_copy_, UINT64_MAX));

  return true;
}

bool AInferRuntime258V::init_mtp() {
  if (mtp_.initialized) return true;
  if (!is_initialized()) {
    std::fprintf(stderr, "[MTP] Base runtime must be initialized before init_mtp()\n");
    return false;
  }

  auto get_pay = [&](const std::string &nm) -> void * {
    auto it = entries_.find(nm);
    if (it == entries_.end()) return nullptr;
    return (char *)pay_arena_ + (it->second.d_off - pay_lo_);
  };
  auto get_sc = [&](const std::string &nm) -> void * {
    auto it = entries_.find(nm);
    if (it == entries_.end() || it->second.sc_bytes == 0) return nullptr;
    return (char *)sc_arena_ + (it->second.sc_off - sc_lo_);
  };

  // Map 10 INT4 MTP weights directly from payloads and scales
  mtp_.fc_w = get_pay("mtp.fc.weight");
  mtp_.fc_s = get_sc("mtp.fc.weight");
  mtp_.q_proj_w = get_pay("mtp.layers.0.self_attn.q_proj.weight");
  mtp_.q_proj_s = get_sc("mtp.layers.0.self_attn.q_proj.weight");
  mtp_.k_proj_w = get_pay("mtp.layers.0.self_attn.k_proj.weight");
  mtp_.k_proj_s = get_sc("mtp.layers.0.self_attn.k_proj.weight");
  mtp_.v_proj_w = get_pay("mtp.layers.0.self_attn.v_proj.weight");
  mtp_.v_proj_s = get_sc("mtp.layers.0.self_attn.v_proj.weight");
  mtp_.o_proj_w = get_pay("mtp.layers.0.self_attn.o_proj.weight");
  mtp_.o_proj_s = get_sc("mtp.layers.0.self_attn.o_proj.weight");
  mtp_.exp_gu_w = get_pay("mtp.layers.0.mlp.experts.gate_up_proj");
  mtp_.exp_gu_s = get_sc("mtp.layers.0.mlp.experts.gate_up_proj");
  mtp_.exp_dn_w = get_pay("mtp.layers.0.mlp.experts.down_proj");
  mtp_.exp_dn_s = get_sc("mtp.layers.0.mlp.experts.down_proj");
  mtp_.sh_gate_w = get_pay("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
  mtp_.sh_gate_s = get_sc("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
  mtp_.sh_up_w = get_pay("mtp.layers.0.mlp.shared_expert.up_proj.weight");
  mtp_.sh_up_s = get_sc("mtp.layers.0.mlp.shared_expert.up_proj.weight");
  mtp_.sh_down_w = get_pay("mtp.layers.0.mlp.shared_expert.down_proj.weight");
  mtp_.sh_down_s = get_sc("mtp.layers.0.mlp.shared_expert.down_proj.weight");

  // The 9 MTP norm and router gate weights are stored as BF16 in the checkpoint.
  // Allocate a compact 2.05 MiB GPU buffer and convert them to FP32 once at startup
  // so they match the FP32 kernel interfaces (rmsnorm_2048, rmsnorm_head_256, moe_topk_router).
  size_t total_fp32_elements = 2048 * 5 + 256 * 2 + 2048 + 256 * 2048; // 537,088 floats
  ze_device_mem_alloc_desc_t dmem_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, total_fp32_elements * sizeof(float), 4096, dev_, &mtp_.d_fp32_weights_arena));

  float *fp32_ptr = (float *)mtp_.d_fp32_weights_arena;
  auto upload_bf16_as_fp32 = [&](const std::string &nm, size_t count) -> float * {
    void *src_gpu = get_pay(nm);
    if (!src_gpu) return nullptr;
    std::vector<uint16_t> h_bf16(count);
    std::vector<float> h_fp32(count);
    CHECK_L0_RET_NULL(zeCommandListAppendMemoryCopy(cmd_copy_, h_bf16.data(), src_gpu, count * sizeof(uint16_t), nullptr, 0, nullptr));
    for (size_t i = 0; i < count; ++i) {
      uint32_t u = (uint32_t)h_bf16[i] << 16;
      float f;
      std::memcpy(&f, &u, 4);
      h_fp32[i] = f;
    }
    float *dst_ptr = fp32_ptr;
    CHECK_L0_RET_NULL(zeCommandListAppendMemoryCopy(cmd_copy_, dst_ptr, h_fp32.data(), count * sizeof(float), nullptr, 0, nullptr));
    fp32_ptr += count;
    return dst_ptr;
  };

  mtp_.pre_fc_norm_emb_w = upload_bf16_as_fp32("mtp.pre_fc_norm_embedding.weight", 2048);
  mtp_.pre_fc_norm_hid_w = upload_bf16_as_fp32("mtp.pre_fc_norm_hidden.weight", 2048);
  mtp_.in_norm_w         = upload_bf16_as_fp32("mtp.layers.0.input_layernorm.weight", 2048);
  mtp_.q_norm_w          = upload_bf16_as_fp32("mtp.layers.0.self_attn.q_norm.weight", 256);
  mtp_.k_norm_w          = upload_bf16_as_fp32("mtp.layers.0.self_attn.k_norm.weight", 256);
  mtp_.post_norm_w       = upload_bf16_as_fp32("mtp.layers.0.post_attention_layernorm.weight", 2048);
  mtp_.shared_gate_w     = upload_bf16_as_fp32("mtp.layers.0.mlp.shared_expert_gate.weight", 2048);
  mtp_.router_w          = upload_bf16_as_fp32("mtp.layers.0.mlp.gate.weight", 256 * 2048);
  mtp_.norm_w            = upload_bf16_as_fp32("mtp.norm.weight", 2048);

  if (!mtp_.pre_fc_norm_emb_w || !mtp_.pre_fc_norm_hid_w || !mtp_.fc_w || !mtp_.in_norm_w ||
      !mtp_.q_proj_w || !mtp_.k_proj_w || !mtp_.v_proj_w || !mtp_.o_proj_w ||
      !mtp_.post_norm_w || !mtp_.router_w || !mtp_.exp_gu_w || !mtp_.exp_dn_w ||
      !mtp_.norm_w) {
    std::fprintf(stderr, "[MTP] Missing required MTP weights in model container\n");
    return false;
  }

  // Allocate MTP KV cache (1 layer: 2 heads, max_ctx, 256 dim, BF16)
  size_t kv_sz = (size_t)NUM_KV_HEADS * max_ctx_ * HEAD_DIM * sizeof(uint16_t);
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, kv_sz, 4096, dev_, &mtp_.k_cache));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, kv_sz, 4096, dev_, &mtp_.v_cache));

  // Allocate MTP local activation workspaces
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_e_raw));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_e_norm));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_h_norm));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, 2 * HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_x_cat));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_fc_out));
  CHECK_L0(zeMemAllocDevice(ctx_, &dmem_desc, HIDDEN_DIM * sizeof(float), 64, dev_, (void **)&mtp_.d_x));

  // Pinned host-visible memory for draft token output
  ze_host_mem_alloc_desc_t hmem_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
  CHECK_L0(zeMemAllocShared(ctx_, &dmem_desc, &hmem_desc, sizeof(int), 64, dev_, (void **)&mtp_.d_draft_token));
  *mtp_.d_draft_token = 0;

  // Zero out MTP KV cache
  uint32_t zero = 0;
  CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, mtp_.k_cache, &zero, sizeof(zero), kv_sz, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, mtp_.v_cache, &zero, sizeof(zero), kv_sz, nullptr, 0, nullptr));

  // Record cmd_draft list
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &mtp_.cmd_draft));

  auto append_gemv = [&](ze_command_list_handle_t list, float *y, void *w, void *s, float *x, int M, int K) {
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 0, sizeof(void *), &y));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 1, sizeof(void *), &w));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 2, sizeof(void *), &s));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 3, sizeof(void *), &x));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 4, sizeof(int), &M));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_, 5, sizeof(int), &K));
    CHECK_L0_VOID(zeKernelSetGroupSize(k_gemv_, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)((M + 255) / 256), 1, 1};
    CHECK_L0_VOID(zeCommandListAppendLaunchKernel(list, k_gemv_, &gc, nullptr, 0, nullptr));
  };

  ze_group_count_t gcnt_embed{HIDDEN_DIM / 256, 1, 1};
  ze_group_count_t gcnt_norm2048{1, 1, 1};
  ze_group_count_t gcnt_silu{2, 1, 1};
  ze_group_count_t gcnt_res{8, 1, 1};
  ze_group_count_t gcnt_router{1, 1, 1};

  // 1. Embed Token Lookup for selected token: d_embed_tokens_[selected_token] -> mtp_.d_e_raw
  void *d_selected_token_ptr = (void *)((uintptr_t)d_ctrl_ + 12); // selected_token is at offset 12 in RuntimeControl
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 0, sizeof(void *), &mtp_.d_e_raw));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 1, sizeof(void *), &d_embed_tokens_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_, 2, sizeof(void *), &d_selected_token_ptr));
  CHECK_L0(zeKernelSetGroupSize(k_embed_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_embed_, &gcnt_embed, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 2. Pre-FC RMSNorm:
  //    e_norm = norm(d_e_raw, pre_fc_norm_emb_w)
  //    h_norm = norm(d_x_, pre_fc_norm_hid_w)   // d_x_ holds trunk hidden state before final norm
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &mtp_.d_e_norm));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &mtp_.d_e_raw));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &mtp_.pre_fc_norm_emb_w));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));

  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &mtp_.d_h_norm));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &mtp_.pre_fc_norm_hid_w));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 3. Concat2 [e_norm, h_norm] -> d_x_cat [4096]
  int n_hid = HIDDEN_DIM;
  CHECK_L0(zeKernelSetArgumentValue(k_concat2_, 0, sizeof(void *), &mtp_.d_x_cat));
  CHECK_L0(zeKernelSetArgumentValue(k_concat2_, 1, sizeof(void *), &mtp_.d_e_norm));
  CHECK_L0(zeKernelSetArgumentValue(k_concat2_, 2, sizeof(void *), &mtp_.d_h_norm));
  CHECK_L0(zeKernelSetArgumentValue(k_concat2_, 3, sizeof(int), &n_hid));
  CHECK_L0(zeKernelSetGroupSize(k_concat2_, 256, 1, 1));
  ze_group_count_t gcnt_concat{(uint32_t)((2 * HIDDEN_DIM + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_concat2_, &gcnt_concat, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 4. FC GEMV: d_x_cat [4096] @ fc_w [2048, 4096] -> mtp_.d_fc_out [2048]
  append_gemv(mtp_.cmd_draft, mtp_.d_fc_out, mtp_.fc_w, mtp_.fc_s, mtp_.d_x_cat, HIDDEN_DIM, 2 * HIDDEN_DIM);
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 5. MTP Full-Attention Layer:
  //    Input layernorm: norm(mtp_.d_fc_out, in_norm_w) -> d_x_norm_
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_norm_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &mtp_.d_fc_out));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &mtp_.in_norm_w));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // QKV projections
  append_gemv(mtp_.cmd_draft, d_q_proj_raw_, mtp_.q_proj_w, mtp_.q_proj_s, d_x_norm_, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
  append_gemv(mtp_.cmd_draft, d_k_full_, mtp_.k_proj_w, mtp_.k_proj_s, d_x_norm_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  append_gemv(mtp_.cmd_draft, d_v_full_, mtp_.v_proj_w, mtp_.v_proj_s, d_x_norm_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Deinterleave Q and Gate
  int num_q_elements = NUM_Q_HEADS * HEAD_DIM;
  CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 0, sizeof(void *), &d_q_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 1, sizeof(void *), &d_gate_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_, 2, sizeof(void *), &d_q_proj_raw_));
  CHECK_L0(zeKernelSetGroupSize(k_deinterleave_qg_, 256, 1, 1));
  ze_group_count_t gcnt_deint{(uint32_t)((num_q_elements + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_deinterleave_qg_, &gcnt_deint, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Head RMSNorm on Q and K
  for (int h = 0; h < NUM_Q_HEADS; ++h) {
    float *qh = d_q_full_ + h * HEAD_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &qh));
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &qh));
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &mtp_.q_norm_w));
    CHECK_L0(zeKernelSetGroupSize(k_norm256_, 256, 1, 1));
    ze_group_count_t gc_h{1, 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm256_, &gc_h, nullptr, 0, nullptr));
  }
  for (int h = 0; h < NUM_KV_HEADS; ++h) {
    float *kh = d_k_full_ + h * HEAD_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &kh));
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &kh));
    CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &mtp_.k_norm_w));
    CHECK_L0(zeKernelSetGroupSize(k_norm256_, 256, 1, 1));
    ze_group_count_t gc_h{1, 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm256_, &gc_h, nullptr, 0, nullptr));
  }
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // RoPE + KV append to MTP's dedicated KV cache
  uint32_t max_c = max_ctx_;
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 0, sizeof(void *), &d_q_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 1, sizeof(void *), &d_k_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 2, sizeof(void *), &d_v_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 3, sizeof(void *), &mtp_.k_cache));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 4, sizeof(void *), &mtp_.v_cache));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 5, sizeof(void *), &d_ctrl_));
  CHECK_L0(zeKernelSetArgumentValue(k_rope_ctrl_, 6, sizeof(uint32_t), &max_c));
  CHECK_L0(zeKernelSetGroupSize(k_rope_ctrl_, 256, 1, 1));
  ze_group_count_t gcnt_rope{1, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_rope_ctrl_, &gcnt_rope, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // GQA Attention Decode
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 0, sizeof(void *), &d_attn_out_));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 1, sizeof(void *), &d_q_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 2, sizeof(void *), &d_gate_full_));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 3, sizeof(void *), &mtp_.k_cache));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 4, sizeof(void *), &mtp_.v_cache));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 5, sizeof(void *), &d_ctrl_));
  CHECK_L0(zeKernelSetArgumentValue(k_attn_ctrl_, 6, sizeof(uint32_t), &max_c));
  CHECK_L0(zeKernelSetGroupSize(k_attn_ctrl_, 256, 1, 1));
  ze_group_count_t gcnt_attn{(uint32_t)NUM_Q_HEADS, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_attn_ctrl_, &gcnt_attn, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // O Proj GEMV: d_attn_out_ @ o_proj_w [2048, 4096] -> d_attn_proj_ [2048]
  append_gemv(mtp_.cmd_draft, d_attn_proj_, mtp_.o_proj_w, mtp_.o_proj_s, d_attn_out_, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Attention Residual Add: d_x_mid_ = mtp_.d_fc_out + d_attn_proj_
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 0, sizeof(void *), &d_x_mid_));
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 1, sizeof(void *), &mtp_.d_fc_out));
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 2, sizeof(void *), &d_attn_proj_));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_resadd_, &gcnt_res, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 6. MTP MoE Layer:
  //    Post-attention RMSNorm: d_x_post_ = norm(d_x_mid_, post_norm_w)
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_post_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &d_x_mid_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &mtp_.post_norm_w));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Router Top-8
  void *d_top_idx_ptr = (void *)((uintptr_t)d_ctrl_ + 32);
  void *d_top_wt_ptr  = (void *)((uintptr_t)d_ctrl_ + 64);
  void *d_sh_gate_ptr = (void *)((uintptr_t)d_ctrl_ + 96);

  CHECK_L0(zeKernelSetArgumentValue(k_router_, 0, sizeof(void *), &d_x_post_));
  CHECK_L0(zeKernelSetArgumentValue(k_router_, 1, sizeof(void *), &mtp_.router_w));
  CHECK_L0(zeKernelSetArgumentValue(k_router_, 2, sizeof(void *), &mtp_.shared_gate_w));
  CHECK_L0(zeKernelSetArgumentValue(k_router_, 3, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_router_, 4, sizeof(void *), &d_top_wt_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_router_, 5, sizeof(void *), &d_sh_gate_ptr));
  CHECK_L0(zeKernelSetGroupSize(k_router_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_router_, &gcnt_router, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Batched 8 Active Experts GateUp + SiLU + Down

  int M_gu = EXP_INTER_DIM * 2, K_gu = HIDDEN_DIM;
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 0, sizeof(void *), &d_exp_gu_all_));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 1, sizeof(void *), &mtp_.exp_gu_w));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 2, sizeof(void *), &mtp_.exp_gu_s));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 3, sizeof(void *), &d_x_post_));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 4, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_, 5, sizeof(int), &K_gu));
  CHECK_L0(zeKernelSetGroupSize(k_exp_gu_all_, 256, 1, 1));
  ze_group_count_t gc_gu_all{(uint32_t)((TOP_K * M_gu + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_exp_gu_all_, &gc_gu_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  CHECK_L0(zeKernelSetArgumentValue(k_silu_all_, 0, sizeof(void *), &d_exp_act_all_));
  CHECK_L0(zeKernelSetArgumentValue(k_silu_all_, 1, sizeof(void *), &d_exp_gu_all_));
  CHECK_L0(zeKernelSetGroupSize(k_silu_all_, 256, 1, 1));
  ze_group_count_t gc_silu_all{(uint32_t)((TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_silu_all_, &gc_silu_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  int M_dn = HIDDEN_DIM, K_dn = EXP_INTER_DIM;
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 0, sizeof(void *), &d_moe_acc_));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 1, sizeof(void *), &mtp_.exp_dn_w));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 2, sizeof(void *), &mtp_.exp_dn_s));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 3, sizeof(void *), &d_exp_act_all_));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 4, sizeof(void *), &d_top_idx_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 5, sizeof(void *), &d_top_wt_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 6, sizeof(int), &M_dn));
  CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_, 7, sizeof(int), &K_dn));
  CHECK_L0(zeKernelSetGroupSize(k_exp_dn_accum_all_, 64, 1, 1));
  ze_group_count_t gc_dn_all{(uint32_t)((M_dn + 63) / 64), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_exp_dn_accum_all_, &gc_dn_all, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Shared Expert
  append_gemv(mtp_.cmd_draft, d_sh_g_, mtp_.sh_gate_w, mtp_.sh_gate_s, d_x_post_, EXP_INTER_DIM, HIDDEN_DIM);
  append_gemv(mtp_.cmd_draft, d_sh_u_, mtp_.sh_up_w, mtp_.sh_up_s, d_x_post_, EXP_INTER_DIM, HIDDEN_DIM);
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 0, sizeof(void *), &d_sh_act_));
  CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 1, sizeof(void *), &d_sh_g_));
  CHECK_L0(zeKernelSetArgumentValue(k_silu512_, 2, sizeof(void *), &d_sh_u_));
  CHECK_L0(zeKernelSetGroupSize(k_silu512_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_silu512_, &gcnt_silu, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  int M_sh_dn = HIDDEN_DIM, K_sh_dn = EXP_INTER_DIM;
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 0, sizeof(void *), &d_moe_acc_));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 1, sizeof(void *), &mtp_.sh_down_w));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 2, sizeof(void *), &mtp_.sh_down_s));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 3, sizeof(void *), &d_sh_act_));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 4, sizeof(void *), &d_sh_gate_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 5, sizeof(int), &M_sh_dn));
  CHECK_L0(zeKernelSetArgumentValue(k_gemv_add_scaled_, 6, sizeof(int), &K_sh_dn));
  CHECK_L0(zeKernelSetGroupSize(k_gemv_add_scaled_, 256, 1, 1));
  ze_group_count_t gc_sh_dn{(uint32_t)((M_sh_dn + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_gemv_add_scaled_, &gc_sh_dn, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Residual add: mtp_.d_x = d_x_mid_ + d_moe_acc_
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 0, sizeof(void *), &mtp_.d_x));
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 1, sizeof(void *), &d_x_mid_));
  CHECK_L0(zeKernelSetArgumentValue(k_resadd_, 2, sizeof(void *), &d_moe_acc_));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_resadd_, &gcnt_res, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 7. Final Norm: norm(mtp_.d_x, mtp_.norm_w) -> d_x_norm_
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 0, sizeof(void *), &d_x_norm_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 1, sizeof(void *), &mtp_.d_x));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_, 2, sizeof(void *), &mtp_.norm_w));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_norm2048_, &gcnt_norm2048, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // 8. LM Head GEMV + Argmax Stage 1 + Stage 2
  int M_lm = VOCAB_SIZE;
  int K_lm = HIDDEN_DIM;
  void *null_logits = nullptr;
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 0, sizeof(void *), &null_logits));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 1, sizeof(void *), &d_lm_head_w_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 2, sizeof(void *), &d_lm_head_s_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 3, sizeof(void *), &d_x_norm_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 4, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 5, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 6, sizeof(int), &M_lm));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_argmax1_, 7, sizeof(int), &K_lm));
  CHECK_L0(zeKernelSetGroupSize(k_lm_head_argmax1_, 256, 1, 1));
  ze_group_count_t gcnt_lm{(VOCAB_SIZE + 255) / 256, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_lm_head_argmax1_, &gcnt_lm, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  // Stage 2 Argmax: writes directly into mtp_.d_draft_token
  uint32_t num_stage1_groups = gcnt_lm.groupCountX;
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 0, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 1, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 2, sizeof(void *), &mtp_.d_draft_token));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 3, sizeof(uint32_t), &num_stage1_groups));
  CHECK_L0(zeKernelSetGroupSize(k_argmax2_mtp_, 256, 1, 1));
  ze_group_count_t gcnt_arg2{1, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(mtp_.cmd_draft, k_argmax2_mtp_, &gcnt_arg2, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(mtp_.cmd_draft, nullptr, 0, nullptr));

  CHECK_L0(zeCommandListClose(mtp_.cmd_draft));

  mtp_.initialized = true;
  std::printf("[MTP 258V] MTP Layer 0 initialized and command list recorded successfully!\n");
  return true;
}

bool AInferRuntime258V::mtp_draft_step(int *out_draft_token, double *out_latency_us) {
  if (!mtp_.initialized) {
    std::fprintf(stderr, "[MTP] MTP not initialized. Call init_mtp() first.\n");
    return false;
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &mtp_.cmd_draft, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));

  auto t1 = std::chrono::high_resolution_clock::now();
  if (out_latency_us) {
    *out_latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }

  int draft_tok = *mtp_.d_draft_token;
  if (out_draft_token) {
    *out_draft_token = draft_tok;
  }


  return true;
}

bool AInferRuntime258V::init_speculative_verification() {
  if (verify_initialized_) return true;
  if (!init_mtp()) {
    std::fprintf(stderr, "[Speculative] Failed to initialize MTP base layer\n");
    return false;
  }

  ze_device_mem_alloc_desc_t ddesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  ze_host_mem_alloc_desc_t hdesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};

  // 1. Allocate snapshot buffers for 30 DeltaNet layers
  size_t total_conv_bytes = (size_t)NUM_DELTANET_LAYERS * C_QKV * 3 * sizeof(float);
  size_t total_ssm_bytes = (size_t)NUM_DELTANET_LAYERS * H_V * S_V * S_V * sizeof(float);
  CHECK_L0(zeMemAllocDevice(ctx_, &ddesc, total_conv_bytes, 4096, dev_, &d_conv_snap_));
  CHECK_L0(zeMemAllocDevice(ctx_, &ddesc, total_ssm_bytes, 4096, dev_, &d_ssm_snap_));

  // Zero out snapshots initially
  uint32_t zero = 0;
  CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, d_conv_snap_, &zero, sizeof(zero), total_conv_bytes, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryFill(cmd_copy_, d_ssm_snap_, &zero, sizeof(zero), total_ssm_bytes, nullptr, 0, nullptr));

  // 2. Allocate Stage 1 reduction buffers for Token 1 (Token 0 reuses d_stage1_vals_, d_stage1_idxs_)
  CHECK_L0(zeMemAllocDevice(ctx_, &ddesc, 1024 * sizeof(float), 64, dev_, (void **)&d_stage1_vals_1_));
  CHECK_L0(zeMemAllocDevice(ctx_, &ddesc, 1024 * sizeof(uint32_t), 64, dev_, (void **)&d_stage1_idxs_1_));

  // 3. Allocate dual verification output tokens (USM shared for zero-copy host read)
  CHECK_L0(zeMemAllocShared(ctx_, &ddesc, &hdesc, 2 * sizeof(int), 64, dev_, (void **)&d_verify_tokens_));
  d_verify_tokens_[0] = 0;
  d_verify_tokens_[1] = 0;

  // 4. Record Rollback Command List
  // On reject: restores conv and ssm states from snapshots
  ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_rollback_));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_rollback_, ssm_conv_arena_, d_conv_snap_, total_conv_bytes, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendMemoryCopy(cmd_rollback_, ssm_recr_arena_, d_ssm_snap_, total_ssm_bytes, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd_rollback_, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListClose(cmd_rollback_));

  // 5. Record Dual-Token Verification Command List (B = 2)
  CHECK_L0(zeCommandListCreate(ctx_, dev_, &ldesc, &cmd_verify_m2_));
  const int B = 2;

  // A. Embed Token 0 and Token 1: d_tokens_chunk_[0..1] -> d_x_chunk_ [2, 2048]
  CHECK_L0(zeKernelSetArgumentValue(k_embed_batch_, 0, sizeof(void *), &d_x_chunk_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_batch_, 1, sizeof(void *), &d_embed_tokens_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_batch_, 2, sizeof(void *), &d_tokens_chunk_));
  CHECK_L0(zeKernelSetArgumentValue(k_embed_batch_, 3, sizeof(int), &B));
  CHECK_L0(zeKernelSetGroupSize(k_embed_batch_, 256, 1, 1));
  ze_group_count_t gcnt_embed{(uint32_t)((B * HIDDEN_DIM + 255) / 256), 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_embed_batch_, &gcnt_embed, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

  // Helper lambda for Dual-Token GEMV append (T10.1 int4_gemv_m2)
  auto append_gemm = [&](float *Y, void *w, void *s, float *X, int M, int K) {
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 0, sizeof(void *), &Y));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 1, sizeof(void *), &w));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 2, sizeof(void *), &s));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 3, sizeof(void *), &X));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 4, sizeof(int), &M));
    CHECK_L0_VOID(zeKernelSetArgumentValue(k_gemv_m2_, 5, sizeof(int), &K));
    CHECK_L0_VOID(zeKernelSetGroupSize(k_gemv_m2_, 256, 1, 1));
    ze_group_count_t gc{(uint32_t)((M + 255) / 256), 1, 1};
    CHECK_L0_VOID(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_gemv_m2_, &gc, nullptr, 0, nullptr));
  };

  // 40 Trunk Layers
  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    LayerBinding &lb = layers_[l];

    // Input RMSNorm
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_norm_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &lb.in_norm_w));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1));
    ze_group_count_t gcnt_norm{(uint32_t)B, 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_norm2048_batch_, &gcnt_norm, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    if (lb.is_full_attn) {
      // Full Attention Block
      append_gemm(d_q_proj_raw_chunk_, lb.q_proj_w, lb.q_proj_s, d_x_norm_chunk_, 2 * NUM_Q_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemm(d_k_full_chunk_, lb.k_proj_w, lb.k_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      append_gemm(d_v_full_chunk_, lb.v_proj_w, lb.v_proj_s, d_x_norm_chunk_, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Deinterleave Q and Gate
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 1, sizeof(void *), &d_gate_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 2, sizeof(void *), &d_q_proj_raw_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_deinterleave_qg_batch_, 3, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_deinterleave_qg_batch_, 256, 1, 1));
      ze_group_count_t gcnt_deint{(uint32_t)((B * NUM_Q_HEADS * HEAD_DIM + 255) / 256), 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_deinterleave_qg_batch_, &gcnt_deint, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Q Head Norm
      int total_nq = B * NUM_Q_HEADS;
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.q_norm_w));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nq));
      CHECK_L0(zeKernelSetGroupSize(k_norm256_, 64, 1, 1));
      ze_group_count_t gcnt_nq{(uint32_t)total_nq, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_norm256_, &gcnt_nq, nullptr, 0, nullptr));

      // K Head Norm
      int total_nkv = B * NUM_KV_HEADS;
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 0, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 1, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 2, sizeof(void *), &lb.k_norm_w));
      CHECK_L0(zeKernelSetArgumentValue(k_norm256_, 3, sizeof(int), &total_nkv));
      CHECK_L0(zeKernelSetGroupSize(k_norm256_, 64, 1, 1));
      ze_group_count_t gcnt_nkv{(uint32_t)total_nkv, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_norm256_, &gcnt_nkv, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // RoPE & KV Cache Append
      uint32_t max_c = max_ctx_;
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 0, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 1, sizeof(void *), &d_k_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 2, sizeof(void *), &d_v_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0(zeKernelSetArgumentValue(k_rope_batch_, 7, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_rope_batch_, 256, 1, 1));
      ze_group_count_t gcnt_rope{(uint32_t)B, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_rope_batch_, &gcnt_rope, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // GQA Attention
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 0, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 1, sizeof(void *), &d_q_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 2, sizeof(void *), &d_gate_full_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 3, sizeof(void *), &lb.k_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 4, sizeof(void *), &lb.v_cache));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 5, sizeof(void *), &d_ctrl_));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 6, sizeof(uint32_t), &max_c));
      CHECK_L0(zeKernelSetArgumentValue(k_attn_batch_, 7, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_attn_batch_, 256, 1, 1));
      ze_group_count_t gcnt_attn{(uint32_t)(B * NUM_Q_HEADS), 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_attn_batch_, &gcnt_attn, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Output Proj GEMM
      append_gemm(d_attn_proj_chunk_, lb.o_proj_w, lb.o_proj_s, d_attn_out_chunk_, HIDDEN_DIM, NUM_Q_HEADS * HEAD_DIM);
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));
    } else {
      // DeltaNet Linear Attention Block
      append_gemm(d_qkv_chunk_, lb.qkv_w, lb.qkv_s, d_x_norm_chunk_, C_QKV, HIDDEN_DIM);
      append_gemm(d_z_chunk_, lb.z_w, lb.z_s, d_x_norm_chunk_, H_V * S_V, HIDDEN_DIM);
      append_gemm(d_a_chunk_, lb.a_w, lb.a_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
      append_gemm(d_b_chunk_, lb.b_w, lb.b_s, d_x_norm_chunk_, H_V, HIDDEN_DIM);
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Conv1D with Intermediate State Snapshot
      void *conv_snap_ptr = (char *)d_conv_snap_ + (size_t)lb.linear_slot * (C_QKV * 3 * sizeof(float));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_m2_spec_, 0, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_m2_spec_, 1, sizeof(void *), &d_qkv_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_m2_spec_, 2, sizeof(void *), &lb.conv_state));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_m2_spec_, 3, sizeof(void *), &conv_snap_ptr));
      CHECK_L0(zeKernelSetArgumentValue(k_conv_m2_spec_, 4, sizeof(void *), &lb.conv_w));
      CHECK_L0(zeKernelSetGroupSize(k_conv_m2_spec_, 256, 1, 1));
      ze_group_count_t gcnt_conv{C_QKV / 256, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_conv_m2_spec_, &gcnt_conv, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Head L2 Norm on Q and K Batch
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 0, sizeof(void *), &d_q_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 1, sizeof(void *), &d_k_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 2, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_l2_norm_qk_batch_, 3, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_l2_norm_qk_batch_, 128, 1, 1));
      ze_group_count_t gcnt_l2{(uint32_t)(B * H_K), 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_l2_norm_qk_batch_, &gcnt_l2, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Gate Prep Batch
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 0, sizeof(void *), &d_g_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 1, sizeof(void *), &d_beta_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 2, sizeof(void *), &d_a_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 3, sizeof(void *), &d_b_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 4, sizeof(void *), &lb.dt_bias));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 5, sizeof(void *), &lb.A_log));
      CHECK_L0(zeKernelSetArgumentValue(k_gate_prep_batch_, 6, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_gate_prep_batch_, 32, 1, 1));
      ze_group_count_t gcnt_gate{(uint32_t)B, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_gate_prep_batch_, &gcnt_gate, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // DeltaNet Recurrence with Intermediate SSM State Snapshot
      void *ssm_snap_ptr = (char *)d_ssm_snap_ + (size_t)lb.linear_slot * (H_V * S_V * S_V * sizeof(float));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 0, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 1, sizeof(void *), &lb.ssm_state));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 2, sizeof(void *), &ssm_snap_ptr));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 3, sizeof(void *), &d_q_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 4, sizeof(void *), &d_k_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 5, sizeof(void *), &d_qkv_conv_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 6, sizeof(void *), &d_g_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_recr_m2_spec_, 7, sizeof(void *), &d_beta_chunk_));
      CHECK_L0(zeKernelSetGroupSize(k_recr_m2_spec_, 128, 1, 1));
      ze_group_count_t gcnt_recr{(uint32_t)H_V, 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_recr_m2_spec_, &gcnt_recr, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Head Norm & SiLU(Z) Batch
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_batch_, 0, sizeof(void *), &d_attn_norm_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_batch_, 1, sizeof(void *), &d_attn_out_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_batch_, 2, sizeof(void *), &d_z_chunk_));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_batch_, 3, sizeof(void *), &lb.ssm_norm_w));
      CHECK_L0(zeKernelSetArgumentValue(k_hnorm_batch_, 4, sizeof(int), &B));
      CHECK_L0(zeKernelSetGroupSize(k_hnorm_batch_, 128, 1, 1));
      ze_group_count_t gcnt_hnorm{(uint32_t)(B * H_V), 1, 1};
      CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_hnorm_batch_, &gcnt_hnorm, nullptr, 0, nullptr));
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

      // Output Proj GEMM
      append_gemm(d_attn_proj_chunk_, lb.out_proj_w, lb.out_proj_s, d_attn_norm_chunk_, HIDDEN_DIM, H_V * S_V);
      CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));
    }

    // Mid Residual Add
    int total_res_mid = B * HIDDEN_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_batch_, 0, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_batch_, 1, sizeof(void *), &d_x_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_batch_, 2, sizeof(void *), &d_attn_proj_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_resadd_batch_, 3, sizeof(int), &total_res_mid));
    CHECK_L0(zeKernelSetGroupSize(k_resadd_batch_, 256, 1, 1));
    ze_group_count_t gcnt_res_mid{(uint32_t)((total_res_mid + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_resadd_batch_, &gcnt_res_mid, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    // Post-Attn Norm
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &lb.post_norm_w));
    CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1));
    ze_group_count_t gcnt_norm_post{(uint32_t)B, 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_norm2048_batch_, &gcnt_norm_post, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    // MoE Router Batch
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 0, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 1, sizeof(void *), &lb.router_w));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 2, sizeof(void *), &lb.shared_gate_w));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 3, sizeof(void *), &d_top_idx_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 4, sizeof(void *), &d_top_wt_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 5, sizeof(void *), &d_sh_gate_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_router_batch_, 6, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_router_batch_, 256, 1, 1));
    ze_group_count_t gcnt_router{(uint32_t)B, 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_router_batch_, &gcnt_router, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    // MoE Active Experts Batch
    int K_gu = HIDDEN_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 0, sizeof(void *), &d_exp_gu_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 1, sizeof(void *), &lb.exp_gu_w));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 2, sizeof(void *), &lb.exp_gu_s));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 3, sizeof(void *), &d_x_post_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 4, sizeof(void *), &d_top_idx_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 5, sizeof(int), &K_gu));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_gu_all_batch_, 6, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_exp_gu_all_batch_, 256, 1, 1));
    ze_group_count_t gc_gu_all{(uint32_t)((B * TOP_K * 2 * EXP_INTER_DIM + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_exp_gu_all_batch_, &gc_gu_all, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    CHECK_L0(zeKernelSetArgumentValue(k_silu_all_batch_, 0, sizeof(void *), &d_exp_act_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_all_batch_, 1, sizeof(void *), &d_exp_gu_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_all_batch_, 2, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_silu_all_batch_, 256, 1, 1));
    ze_group_count_t gc_silu_all{(uint32_t)((B * TOP_K * EXP_INTER_DIM + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_silu_all_batch_, &gc_silu_all, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    int M_dn = HIDDEN_DIM, K_dn = EXP_INTER_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 0, sizeof(void *), &d_moe_acc_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 1, sizeof(void *), &lb.exp_dn_w));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 2, sizeof(void *), &lb.exp_dn_s));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 3, sizeof(void *), &d_exp_act_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 4, sizeof(void *), &d_top_idx_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 5, sizeof(void *), &d_top_wt_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 6, sizeof(int), &M_dn));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 7, sizeof(int), &K_dn));
    CHECK_L0(zeKernelSetArgumentValue(k_exp_dn_accum_all_batch_, 8, sizeof(int), &B));
    CHECK_L0(zeKernelSetGroupSize(k_exp_dn_accum_all_batch_, 64, 1, 1));
    ze_group_count_t gc_dn_all{(uint32_t)((B * M_dn + 63) / 64), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_exp_dn_accum_all_batch_, &gc_dn_all, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    // Shared Expert
    append_gemm(d_sh_g_chunk_, lb.sh_gate_w, lb.sh_gate_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    append_gemm(d_sh_u_chunk_, lb.sh_up_w, lb.sh_up_s, d_x_post_chunk_, EXP_INTER_DIM, HIDDEN_DIM);
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    int total_sh_act = B * EXP_INTER_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_silu_mul_batch_, 0, sizeof(void *), &d_sh_act_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_mul_batch_, 1, sizeof(void *), &d_sh_g_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_mul_batch_, 2, sizeof(void *), &d_sh_u_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_silu_mul_batch_, 3, sizeof(int), &total_sh_act));
    CHECK_L0(zeKernelSetGroupSize(k_silu_mul_batch_, 256, 1, 1));
    ze_group_count_t gc_sh_silu{(uint32_t)((total_sh_act + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_silu_mul_batch_, &gc_sh_silu, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    append_gemm(d_sh_down_chunk_, lb.sh_down_w, lb.sh_down_s, d_sh_act_chunk_, HIDDEN_DIM, EXP_INTER_DIM);
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

    // Block Residual Add
    int M_res = HIDDEN_DIM;
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 0, sizeof(void *), &d_x_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 1, sizeof(void *), &d_x_mid_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 2, sizeof(void *), &d_moe_acc_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 3, sizeof(void *), &d_sh_down_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 4, sizeof(void *), &d_sh_gate_chunk_));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 5, sizeof(int), &B));
    CHECK_L0(zeKernelSetArgumentValue(k_block_resadd_moe_batch_, 6, sizeof(int), &M_res));
    CHECK_L0(zeKernelSetGroupSize(k_block_resadd_moe_batch_, 256, 1, 1));
    ze_group_count_t gc_block_res{(uint32_t)((B * M_res + 255) / 256), 1, 1};
    CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_block_resadd_moe_batch_, &gc_block_res, nullptr, 0, nullptr));
    CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));
  }

  // Final RMSNorm on both tokens: d_x_chunk_ [2, 2048] -> d_x_norm_chunk_ [2, 2048]
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 0, sizeof(void *), &d_x_norm_chunk_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 1, sizeof(void *), &d_x_chunk_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 2, sizeof(void *), &d_final_norm_w_));
  CHECK_L0(zeKernelSetArgumentValue(k_norm2048_batch_, 3, sizeof(int), &B));
  CHECK_L0(zeKernelSetGroupSize(k_norm2048_batch_, 256, 1, 1));
  ze_group_count_t gcnt_final_norm{(uint32_t)B, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_norm2048_batch_, &gcnt_final_norm, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

  // Dual-Token LM Head GEMV + Stage 1 Argmax [248320, 2048]
  int M_lm = VOCAB_SIZE;
  int K_lm = HIDDEN_DIM;
  void *null_logits = nullptr;
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 0, sizeof(void *), &null_logits));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 1, sizeof(void *), &d_lm_head_w_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 2, sizeof(void *), &d_lm_head_s_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 3, sizeof(void *), &d_x_norm_chunk_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 4, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 5, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 6, sizeof(void *), &d_stage1_vals_1_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 7, sizeof(void *), &d_stage1_idxs_1_));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 8, sizeof(int), &M_lm));
  CHECK_L0(zeKernelSetArgumentValue(k_lm_head_m2_argmax1_, 9, sizeof(int), &K_lm));
  CHECK_L0(zeKernelSetGroupSize(k_lm_head_m2_argmax1_, 256, 1, 1));
  ze_group_count_t gcnt_lm{(VOCAB_SIZE + 255) / 256, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_lm_head_m2_argmax1_, &gcnt_lm, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

  // Stage 2 Argmax for Token 0 -> d_verify_tokens_[0]
  uint32_t num_stage1_groups = gcnt_lm.groupCountX;
  void *d_tok0_ptr = (void *)((int *)d_verify_tokens_ + 0);
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 0, sizeof(void *), &d_stage1_vals_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 1, sizeof(void *), &d_stage1_idxs_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 2, sizeof(void *), &d_tok0_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 3, sizeof(uint32_t), &num_stage1_groups));
  CHECK_L0(zeKernelSetGroupSize(k_argmax2_mtp_, 256, 1, 1));
  ze_group_count_t gcnt_arg2{1, 1, 1};
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_argmax2_mtp_, &gcnt_arg2, nullptr, 0, nullptr));

  // Stage 2 Argmax for Token 1 -> d_verify_tokens_[1]
  void *d_tok1_ptr = (void *)((int *)d_verify_tokens_ + 1);
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 0, sizeof(void *), &d_stage1_vals_1_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 1, sizeof(void *), &d_stage1_idxs_1_));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 2, sizeof(void *), &d_tok1_ptr));
  CHECK_L0(zeKernelSetArgumentValue(k_argmax2_mtp_, 3, sizeof(uint32_t), &num_stage1_groups));
  CHECK_L0(zeKernelSetGroupSize(k_argmax2_mtp_, 256, 1, 1));
  CHECK_L0(zeCommandListAppendLaunchKernel(cmd_verify_m2_, k_argmax2_mtp_, &gcnt_arg2, nullptr, 0, nullptr));
  CHECK_L0(zeCommandListAppendBarrier(cmd_verify_m2_, nullptr, 0, nullptr));

  CHECK_L0(zeCommandListClose(cmd_verify_m2_));

  verify_initialized_ = true;
  std::printf("[Speculative 258V] Verification engine initialized and command lists recorded successfully!\n");
  return true;
}

bool AInferRuntime258V::speculative_step(int *out_tok1, int *out_tok2, int *out_num_emitted, bool *out_accepted, double *out_round_us) {
  if (!verify_initialized_) {
    if (!init_speculative_verification()) return false;
  }

  // Ensure we have a draft candidate
  if (pending_draft_token_ < 0) {
    if (!mtp_draft_step(&pending_draft_token_)) return false;
  }

  // Check context bounds
  if (h_ctrl_.position + 2 >= (int)max_ctx_) {
    std::fprintf(stderr, "[Speculative] Context limit reached at pos %d\n", h_ctrl_.position);
    return false;
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  int cur_tok = h_ctrl_.selected_token;
  int draft_tok = pending_draft_token_;

  // 1. Advance position by 1 for Token 0
  h_ctrl_.position += 1;
  h_ctrl_.active_length += 1;
  h_ctrl_.token_id = cur_tok;
  std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

  // 2. Load input tokens [cur_tok, draft_tok] into d_tokens_chunk_
  d_tokens_chunk_[0] = cur_tok;
  d_tokens_chunk_[1] = draft_tok;

  // 3. Execute dual-token verification forward pass (B = 2)
  auto t_v0 = std::chrono::high_resolution_clock::now();
  CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_verify_m2_, fence_));
  CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
  CHECK_L0(zeFenceReset(fence_));
  auto t_v1 = std::chrono::high_resolution_clock::now();
  double v_ms = std::chrono::duration<double, std::milli>(t_v1 - t_v0).count();

  int true_t1 = d_verify_tokens_[0];
  int true_t2 = d_verify_tokens_[1];
  bool accepted = (draft_tok == true_t1);

  double rb_ms = 0.0, dr_ms = 0.0;

  if (accepted) {
    // ACCEPTED: Both Token 1 (draft) and Token 2 are correct!
    *out_tok1 = true_t1;
    *out_tok2 = true_t2;
    *out_num_emitted = 2;
    *out_accepted = true;

    // Advance position by another 1 (now at position of Token 1)
    h_ctrl_.position += 1;
    h_ctrl_.active_length += 1;
    h_ctrl_.selected_token = true_t2;
    std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

    // Update d_x_ to point to Token 1 hidden state for subsequent drafting
    CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, d_x_, d_x_chunk_ + HIDDEN_DIM, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK_L0(zeCommandListHostSynchronize(cmd_copy_, UINT64_MAX));

    // Draft next candidate from true_t2
    pending_draft_token_ = -1;
    mtp_draft_step(&pending_draft_token_, &dr_ms);
    dr_ms /= 1000.0;
  } else {
    // REJECTED: Only Token 0's prediction is correct. Roll back SSM and Conv states!
    *out_tok1 = true_t1;
    *out_num_emitted = 1;
    *out_accepted = false;

    // Execute recorded GPU rollback copy
    auto t_rb0 = std::chrono::high_resolution_clock::now();
    CHECK_L0(zeCommandQueueExecuteCommandLists(queue_, 1, &cmd_rollback_, fence_));
    CHECK_L0(zeFenceHostSynchronize(fence_, UINT64_MAX));
    CHECK_L0(zeFenceReset(fence_));
    auto t_rb1 = std::chrono::high_resolution_clock::now();
    rb_ms = std::chrono::duration<double, std::milli>(t_rb1 - t_rb0).count();

    // Position remains at Token 0 position
    h_ctrl_.selected_token = true_t1;
    std::memcpy(d_ctrl_, &h_ctrl_, sizeof(RuntimeControl));

    // Update d_x_ to point to Token 0 hidden state for subsequent drafting
    CHECK_L0(zeCommandListAppendMemoryCopy(cmd_copy_, d_x_, d_x_chunk_, HIDDEN_DIM * sizeof(float), nullptr, 0, nullptr));
    CHECK_L0(zeCommandListHostSynchronize(cmd_copy_, UINT64_MAX));

    // Draft next candidate from true_t1
    pending_draft_token_ = -1;
    mtp_draft_step(&pending_draft_token_, &dr_ms);
    dr_ms /= 1000.0;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  if (out_round_us) {
    *out_round_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }

  static const char *prof_env = std::getenv("AINFER_SPEC_PROFILE");
  if (prof_env && prof_env[0] == '1') {
    std::printf("    [SPEC_PROF] Verify: %.2f ms | Draft: %.2f ms | Rollback: %.2f ms | Accepted: %s\n",
                v_ms, dr_ms, rb_ms, accepted ? "YES" : "NO");
  }

  return true;
}

bool AInferRuntime258V::generate_speculative(
    const std::vector<int> &prompt_ids, int max_new_tokens,
    std::vector<int> &generated_ids, double *out_prefill_ms,
    double *out_spec_tok_per_s, double *out_acceptance_rate) {
  if (!init_speculative_verification()) return false;
  generated_ids.clear();
  generated_ids.reserve(max_new_tokens);

  auto t_pref0 = std::chrono::steady_clock::now();
  int first_tok = 0;
  if (!prefill(prompt_ids, &first_tok)) return false;
  auto t_pref1 = std::chrono::steady_clock::now();
  if (out_prefill_ms) {
    *out_prefill_ms = std::chrono::duration<double, std::milli>(t_pref1 - t_pref0).count();
  }

  generated_ids.push_back(first_tok);
  if (max_new_tokens <= 1) return true;

  // Initial draft from first token
  pending_draft_token_ = -1;
  mtp_draft_step(&pending_draft_token_);

  int total_rounds = 0;
  int accepted_rounds = 0;

  auto t_gen0 = std::chrono::steady_clock::now();
  while ((int)generated_ids.size() < max_new_tokens) {
    int tok1 = 0, tok2 = 0, n_emitted = 0;
    bool accepted = false;
    if (!speculative_step(&tok1, &tok2, &n_emitted, &accepted)) {
      break;
    }
    total_rounds++;
    if (accepted) accepted_rounds++;

    generated_ids.push_back(tok1);
    if ((int)generated_ids.size() < max_new_tokens && n_emitted == 2) {
      generated_ids.push_back(tok2);
    }
    if (tok1 == 151645 || tok1 == 151643 || (n_emitted == 2 && (tok2 == 151645 || tok2 == 151643))) {
      break;
    }
  }
  auto t_gen1 = std::chrono::steady_clock::now();
  double gen_sec = std::chrono::duration<double>(t_gen1 - t_gen0).count();
  int num_new = (int)generated_ids.size() - 1;
  if (out_spec_tok_per_s && gen_sec > 0.0) {
    *out_spec_tok_per_s = (double)num_new / gen_sec;
  }
  if (out_acceptance_rate && total_rounds > 0) {
    *out_acceptance_rate = (double)accepted_rounds / (double)total_rounds;
  }
  return true;
}

void AInferRuntime258V::cleanup() {
  if (mtp_.cmd_draft) { zeCommandListDestroy(mtp_.cmd_draft); mtp_.cmd_draft = nullptr; }
  if (k_concat2_) { zeKernelDestroy(k_concat2_); k_concat2_ = nullptr; }
  if (k_argmax2_mtp_) { zeKernelDestroy(k_argmax2_mtp_); k_argmax2_mtp_ = nullptr; }
  if (mtp_.k_cache) { zeMemFree(ctx_, mtp_.k_cache); mtp_.k_cache = nullptr; }
  if (mtp_.v_cache) { zeMemFree(ctx_, mtp_.v_cache); mtp_.v_cache = nullptr; }
  if (mtp_.d_e_raw) { zeMemFree(ctx_, mtp_.d_e_raw); mtp_.d_e_raw = nullptr; }
  if (mtp_.d_e_norm) { zeMemFree(ctx_, mtp_.d_e_norm); mtp_.d_e_norm = nullptr; }
  if (mtp_.d_h_norm) { zeMemFree(ctx_, mtp_.d_h_norm); mtp_.d_h_norm = nullptr; }
  if (mtp_.d_x_cat) { zeMemFree(ctx_, mtp_.d_x_cat); mtp_.d_x_cat = nullptr; }
  if (mtp_.d_fc_out) { zeMemFree(ctx_, mtp_.d_fc_out); mtp_.d_fc_out = nullptr; }
  if (mtp_.d_x) { zeMemFree(ctx_, mtp_.d_x); mtp_.d_x = nullptr; }
  if (mtp_.d_draft_token) { zeMemFree(ctx_, mtp_.d_draft_token); mtp_.d_draft_token = nullptr; }
  if (mtp_.d_fp32_weights_arena) { zeMemFree(ctx_, mtp_.d_fp32_weights_arena); mtp_.d_fp32_weights_arena = nullptr; }
  mtp_.initialized = false;

  if (fence_) { zeFenceDestroy(fence_); fence_ = nullptr; }
  if (cmd_embed_) { zeCommandListDestroy(cmd_embed_); cmd_embed_ = nullptr; }
  for (auto &cl : cmd_layers_) {
    if (cl) { zeCommandListDestroy(cl); cl = nullptr; }
  }
  if (cmd_tail_) { zeCommandListDestroy(cmd_tail_); cmd_tail_ = nullptr; }
  if (cmd_step_) { zeCommandListDestroy(cmd_step_); cmd_step_ = nullptr; }
  if (cmd_prefill_step_) { zeCommandListDestroy(cmd_prefill_step_); cmd_prefill_step_ = nullptr; }
  if (cmd_copy_) { zeCommandListDestroy(cmd_copy_); cmd_copy_ = nullptr; }
  if (queue_) { zeCommandQueueDestroy(queue_); queue_ = nullptr; }

  // Destroy kernels
  if (k_gemv_) zeKernelDestroy(k_gemv_);
  if (k_router_) zeKernelDestroy(k_router_);
  if (k_norm2048_) zeKernelDestroy(k_norm2048_);
  if (k_norm256_) zeKernelDestroy(k_norm256_);
  if (k_silu512_) zeKernelDestroy(k_silu512_);
  if (k_resadd_) zeKernelDestroy(k_resadd_);
  if (k_conv_) zeKernelDestroy(k_conv_);
  if (k_l2_norm_) zeKernelDestroy(k_l2_norm_);
  if (k_gate_prep_) zeKernelDestroy(k_gate_prep_);
  if (k_recr_) zeKernelDestroy(k_recr_);
  if (k_hnorm_) zeKernelDestroy(k_hnorm_);
  if (k_argmax1_) zeKernelDestroy(k_argmax1_);
  if (k_embed_) zeKernelDestroy(k_embed_);
  if (k_exp_gemv_) zeKernelDestroy(k_exp_gemv_);
  if (k_exp_dn_accum_) zeKernelDestroy(k_exp_dn_accum_);
  if (k_gemv_add_scaled_) zeKernelDestroy(k_gemv_add_scaled_);
  if (k_accum_ctrl_) zeKernelDestroy(k_accum_ctrl_);
  if (k_add_shared_ctrl_) zeKernelDestroy(k_add_shared_ctrl_);
  if (k_rope_ctrl_) zeKernelDestroy(k_rope_ctrl_);
  if (k_attn_ctrl_) zeKernelDestroy(k_attn_ctrl_);
  if (k_deinterleave_qg_) zeKernelDestroy(k_deinterleave_qg_);
  if (k_argmax2_ctrl_) zeKernelDestroy(k_argmax2_ctrl_);
  if (k_exp_gu_all_) zeKernelDestroy(k_exp_gu_all_);
  if (k_silu_all_) zeKernelDestroy(k_silu_all_);
  if (k_exp_dn_accum_all_) zeKernelDestroy(k_exp_dn_accum_all_);
  if (k_lm_head_argmax1_) zeKernelDestroy(k_lm_head_argmax1_);

  // Destroy batch prefill kernels
  if (k_gemm_prefill_) zeKernelDestroy(k_gemm_prefill_);
  if (k_embed_batch_) zeKernelDestroy(k_embed_batch_);
  if (k_norm2048_batch_) zeKernelDestroy(k_norm2048_batch_);
  if (k_conv_batch_) zeKernelDestroy(k_conv_batch_);
  if (k_l2_norm_qk_batch_) zeKernelDestroy(k_l2_norm_qk_batch_);
  if (k_gate_prep_batch_) zeKernelDestroy(k_gate_prep_batch_);
  if (k_recr_batch_) zeKernelDestroy(k_recr_batch_);
  if (k_hnorm_batch_) zeKernelDestroy(k_hnorm_batch_);
  if (k_deinterleave_qg_batch_) zeKernelDestroy(k_deinterleave_qg_batch_);
  if (k_rope_batch_) zeKernelDestroy(k_rope_batch_);
  if (k_attn_batch_) zeKernelDestroy(k_attn_batch_);
  if (k_router_batch_) zeKernelDestroy(k_router_batch_);
  if (k_moe_build_expert_bins_) zeKernelDestroy(k_moe_build_expert_bins_);
  if (k_moe_gateup_grouped_batch_) zeKernelDestroy(k_moe_gateup_grouped_batch_);
  if (k_moe_down_grouped_batch_) zeKernelDestroy(k_moe_down_grouped_batch_);
  if (k_moe_accum_down_batch_) zeKernelDestroy(k_moe_accum_down_batch_);
  if (k_exp_gu_all_batch_) zeKernelDestroy(k_exp_gu_all_batch_);
  if (k_silu_all_batch_) zeKernelDestroy(k_silu_all_batch_);
  if (k_exp_dn_accum_all_batch_) zeKernelDestroy(k_exp_dn_accum_all_batch_);
  if (k_silu_mul_batch_) zeKernelDestroy(k_silu_mul_batch_);
  if (k_block_resadd_moe_batch_) zeKernelDestroy(k_block_resadd_moe_batch_);
  if (k_resadd_batch_) zeKernelDestroy(k_resadd_batch_);
  if (k_conv_m2_spec_) zeKernelDestroy(k_conv_m2_spec_);
  if (k_recr_m2_spec_) zeKernelDestroy(k_recr_m2_spec_);
  if (k_lm_head_m2_argmax1_) zeKernelDestroy(k_lm_head_m2_argmax1_);
  if (k_gemv_m2_) zeKernelDestroy(k_gemv_m2_);

  // Destroy speculative verification command lists & buffers
  if (cmd_verify_m2_) { zeCommandListDestroy(cmd_verify_m2_); cmd_verify_m2_ = nullptr; }
  if (cmd_rollback_) { zeCommandListDestroy(cmd_rollback_); cmd_rollback_ = nullptr; }
  if (d_conv_snap_) { zeMemFree(ctx_, d_conv_snap_); d_conv_snap_ = nullptr; }
  if (d_ssm_snap_) { zeMemFree(ctx_, d_ssm_snap_); d_ssm_snap_ = nullptr; }
  if (d_stage1_vals_1_) { zeMemFree(ctx_, d_stage1_vals_1_); d_stage1_vals_1_ = nullptr; }
  if (d_stage1_idxs_1_) { zeMemFree(ctx_, d_stage1_idxs_1_); d_stage1_idxs_1_ = nullptr; }
  if (d_verify_tokens_) { zeMemFree(ctx_, d_verify_tokens_); d_verify_tokens_ = nullptr; }
  verify_initialized_ = false;

  // Destroy cached prefill chunk lists
  for (int b = 1; b <= MAX_PREFILL_CHUNK; ++b) {
    if (cmd_prefill_chunk_[b]) { zeCommandListDestroy(cmd_prefill_chunk_[b]); cmd_prefill_chunk_[b] = nullptr; }
    if (cmd_prefill_tail_[b]) { zeCommandListDestroy(cmd_prefill_tail_[b]); cmd_prefill_tail_[b] = nullptr; }
  }

  if (mod_) { zeModuleDestroy(mod_); mod_ = nullptr; }

  // Free static device arenas
  if (pay_arena_) { zeMemFree(ctx_, pay_arena_); pay_arena_ = nullptr; }
  if (sc_arena_) { zeMemFree(ctx_, sc_arena_); sc_arena_ = nullptr; }
  if (kv_cache_arena_) { zeMemFree(ctx_, kv_cache_arena_); kv_cache_arena_ = nullptr; }
  if (ssm_recr_arena_) { zeMemFree(ctx_, ssm_recr_arena_); ssm_recr_arena_ = nullptr; }
  if (ssm_conv_arena_) { zeMemFree(ctx_, ssm_conv_arena_); ssm_conv_arena_ = nullptr; }
  if (workspace_arena_) { zeMemFree(ctx_, workspace_arena_); workspace_arena_ = nullptr; }
  if (d_ctrl_) { zeMemFree(ctx_, d_ctrl_); d_ctrl_ = nullptr; }
  if (d_tokens_chunk_) { zeMemFree(ctx_, d_tokens_chunk_); d_tokens_chunk_ = nullptr; }

  if (ctx_) { zeContextDestroy(ctx_); ctx_ = nullptr; }
}

} // namespace ainfer
