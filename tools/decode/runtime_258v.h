// AInfer Unified Single-Process Runtime Engine on Intel Arc 140V (Lunar Lake 258V)
// Phase 5: T5.1 (Static Arenas), T5.2 (In-Memory Prefill), T5.3 (Recorded 40-Layer Lists),
// T5.4 (Control Buffer / Zero Reallocation), T5.5 (Diagnostic Cache), T5.6 (State Reset)
#pragma once

#include <level_zero/ze_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_L0(expr)                                                         \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "Level Zero error %d at %s:%d (%s)\n", (int)_r,      \
                   __FILE__, __LINE__, #expr);                                 \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CHECK_L0_VOID(expr)                                                    \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "Level Zero error %d at %s:%d (%s)\n", (int)_r,      \
                   __FILE__, __LINE__, #expr);                                 \
      return;                                                                  \
    }                                                                          \
  } while (0)

namespace ainfer {

// Architecture Constants for pinned model Tiel-Coder-35B-A3B (Qwen3.5-MoE)
constexpr int HIDDEN_DIM = 2048;
constexpr int TOTAL_LAYERS = 40;
constexpr int FULL_ATTN_INTERVAL = 4; // Layers 3, 7, 11, ... (10 total)
constexpr int NUM_FULL_ATTN_LAYERS = 10;
constexpr int NUM_DELTANET_LAYERS = 30;

constexpr int EXP_INTER_DIM = 512;
constexpr int NUM_EXPERTS = 256;
constexpr int TOP_K = 8;

constexpr int S_V = 128;
constexpr int H_V = 32;
constexpr int H_K = 16;
constexpr int C_QKV = 8192;

constexpr int HEAD_DIM = 256;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 2;
// I3.8 split-T decode attention: max T-splits/head (partials [16][SMAX][258]).
// Must match ATTN_SPLIT_MAX in tools/kernels_258v/all_kernels.cl.
constexpr int ATTN_SPLIT_MAX = 8;
constexpr int GQA_GROUP_SIZE = 8;
constexpr int VOCAB_SIZE = 248320;

// Host / Device Shared Control Block (T5.4)
// Pinned device-allocated or host-visible buffer containing all per-step mutable parameters.
// Recorded command lists bind this buffer address once at initialization.
struct alignas(64) RuntimeControl {
  int token_id;          // [0] Current input token ID
  int position;          // [1] Current sequence position (0..max_ctx)
  int active_length;     // [2] Total sequence length
  int selected_token;    // [3] Output sampled / argmax token ID
  float temperature;     // [4] Sampling temperature (0.0 = greedy)
  uint32_t top_k;        // [5] Top-K sampling threshold
  float top_p;           // [6] Top-P nucleus threshold
  float rep_penalty;     // [7] Repetition penalty
  uint32_t top_idx[8];   // [8..15] Top-8 routed active expert indices
  float top_wt[8];       // [16..23] Top-8 routed expert normalized weights
  float sh_gate_val;     // [24] Shared expert gate activation logit
  int pad[7];            // 128 bytes total (aligned to 64 bytes)
};

// T9.1: Typed arena spans with checked bounds.
//
// Device memory is host-addressable here (unified memory / shared
// allocations), so all bounds checks run on the host at init/record time
// with zero device-side overhead. Violations emit a diagnostic to stderr
// and yield null/empty, which callers must treat as fatal init errors
// (see verify_bindings()). Runtime Gens/Decodes never do unchecked
// pointer arithmetic after this.
template <typename T>
struct ArenaSpan {
  T *data = nullptr;
  size_t count = 0; // element count (not bytes)
  const char *name = "unnamed";

  size_t size_bytes() const { return count * sizeof(T); }
  bool empty() const { return data == nullptr || count == 0; }

  // Checked element access: nullptr + diagnostic on out-of-range.
  T *at(size_t i) const {
    if (!data || i >= count) {
      std::fprintf(stderr, "[ArenaSpan:%s] OOB access index %zu of %zu\n",
                   name, i, count);
      return nullptr;
    }
    return data + i;
  }

  // Checked sub-slice [off, off+n): empty span + diagnostic on violation.
  ArenaSpan<T> slice(size_t off, size_t n, const char *sname = nullptr) const {
    ArenaSpan<T> out{nullptr, 0, sname ? sname : name};
    if (!data || n > count || off > count - n) {
      std::fprintf(stderr,
                   "[ArenaSpan:%s] OOB slice off=%zu n=%zu of count=%zu\n",
                   name, off, n, count);
      return out;
    }
    out.data = data + off;
    out.count = n;
    return out;
  }

  // Raw access for Level Zero kernel args. Callers must have validated
  // via at()/slice() or a verify pass first; null here means "invalid".
  T *get() const { return data; }
};

// Overflow-safe size arithmetic for slot/offset computations.
inline bool checked_mul_add(size_t a, size_t b, size_t c, size_t *out) {
  size_t m;
  if (__builtin_mul_overflow(a, b, &m)) return false;
  if (__builtin_add_overflow(m, c, out)) return false;
  return true;
}

// Unified end-of-sequence test. The pinned checkpoint is Qwen3.5-MoE-based;
// it emits 151643 (<|endoftext|>) / 151645 (<|im_end|>). 248044/248046 are the
// Qwen3.8-era ids kept so old prompts/harnesses still terminate. Every
// generation loop (generate, generate_speculative) and the HTTP server must
// use this set — a loop that only knows one pair runs past the other's EOS
// (long-context verify-vs-greedy LENGTH-DIFF of 2026-09-25).
inline bool is_eos_token(int id) {
  return id == 151643 || id == 151645 || id == 248044 || id == 248046;
}

// Bump allocator with per-allocation ledger and fail-fast overflow.
// Replaces ad-hoc w_ptr arithmetic: every sub-allocation is range-checked
// at allocate time (not just totaled at the end) and recorded for audits.
struct CheckedArena {
  uint8_t *base = nullptr;
  size_t total = 0;
  size_t used = 0;
  struct Entry {
    const char *name;
    size_t offset;
    size_t bytes;
  };
  std::vector<Entry> ledger;

  void reset(void *base_ptr, size_t total_bytes) {
    base = (uint8_t *)base_ptr;
    total = total_bytes;
    used = 0;
    ledger.clear();
  }

  // Allocate `count` elements of size `elem_size`, 64B-aligned.
  // Returns nullptr + diagnostic on overflow (callers treat as fatal).
  void *bump(size_t count, size_t elem_size, const char *name = "anon") {
    size_t need;
    if (__builtin_mul_overflow(count, elem_size, &need)) {
      std::fprintf(stderr, "[CheckedArena] size overflow for '%s'\n", name);
      return nullptr;
    }
    size_t aligned = (need + 63) & ~63ULL;
    size_t next;
    if (__builtin_add_overflow(used, aligned, &next) || next > total) {
      std::fprintf(stderr,
                   "[CheckedArena] overflow: '%s' needs %zu bytes, %zu/%zu used\n",
                   name, aligned, used, total);
      return nullptr;
    }
    void *p = base + used;
    ledger.push_back({name, used, aligned});
    used = next;
    return p;
  }

  float *bump_f32(size_t count, const char *name = "anon") {
    return (float *)bump(count, sizeof(float), name);
  }

  // Look up a recorded span by allocation name (for audit/validation).
  bool span_f32(const char *name, ArenaSpan<float> *out) const {
    for (const auto &e : ledger) {
      if (std::strcmp(e.name, name) == 0) {
        out->data = (float *)(base + e.offset);
        out->count = e.bytes / sizeof(float);
        out->name = e.name;
        return true;
      }
    }
    return false;
  }
};

// .binfer Directory Entry
struct BinferEntry {
  char name[64];
  uint64_t d_off, d_bytes, sc_off, sc_bytes;
  uint32_t crc;
  uint64_t shape[8];
  uint8_t ndim;
  uint8_t lt, st;
};

// Per-layer buffer descriptors
struct LayerBinding {
  int layer_idx;
  bool is_full_attn;
  int full_slot;   // 0..9 for full attn
  int linear_slot; // 0..29 for deltanet

  // Norm weights
  void *in_norm_w;
  void *post_norm_w;

  // Linear Attention weights (if deltanet)
  void *qkv_w;
  void *qkv_s;
  void *z_w;
  void *z_s;
  void *a_w;
  void *a_s;
  void *b_w;
  void *b_s;
  void *conv_w;
  void *dt_bias;
  void *A_log;
  void *ssm_norm_w;
  void *out_proj_w;
  void *out_proj_s;
  void *ssm_state; // [32, 128, 128] FP32
  void *conv_state;// [8192, 3] FP32

  // Full Attention weights (if full attn)
  void *q_proj_w;
  void *q_proj_s;
  void *k_proj_w;
  void *k_proj_s;
  void *v_proj_w;
  void *v_proj_s;
  void *o_proj_w;
  void *o_proj_s;
  void *q_norm_w;
  void *k_norm_w;
  void *k_cache; // [2, max_ctx, 256] BF16
  void *v_cache; // [2, max_ctx, 256] BF16
  void *k_cache_i8 = nullptr;
  void *v_cache_i8 = nullptr;
  void *k_scale_i8 = nullptr;
  void *v_scale_i8 = nullptr;

  // MoE weights
  void *router_w;
  void *shared_gate_w;
  void *exp_gu_w; // 3D bank [256, 1024, 2048]
  void *exp_gu_s;
  void *exp_dn_w; // 3D bank [256, 2048, 512]
  void *exp_dn_s;
  void *sh_gate_w;
  void *sh_gate_s;
  void *sh_up_w;
  void *sh_up_s;
  void *sh_down_w;
  void *sh_down_s;
};

// MTP Layer Binding & State (T10.1)
struct MtpBinding {
  void *pre_fc_norm_emb_w = nullptr;
  void *pre_fc_norm_hid_w = nullptr;
  void *fc_w = nullptr;
  void *fc_s = nullptr;

  void *in_norm_w = nullptr;
  void *q_proj_w = nullptr;
  void *q_proj_s = nullptr;
  void *k_proj_w = nullptr;
  void *k_proj_s = nullptr;
  void *v_proj_w = nullptr;
  void *v_proj_s = nullptr;
  void *o_proj_w = nullptr;
  void *o_proj_s = nullptr;
  void *q_norm_w = nullptr;
  void *k_norm_w = nullptr;

  void *post_norm_w = nullptr;
  void *router_w = nullptr;
  void *shared_gate_w = nullptr;
  void *exp_gu_w = nullptr;
  void *exp_gu_s = nullptr;
  void *exp_dn_w = nullptr;
  void *exp_dn_s = nullptr;
  void *sh_gate_w = nullptr;
  void *sh_gate_s = nullptr;
  void *sh_up_w = nullptr;
  void *sh_up_s = nullptr;
  void *sh_down_w = nullptr;
  void *sh_down_s = nullptr;

  void *norm_w = nullptr;

  void *k_cache = nullptr; // [2, max_ctx, 256] BF16
  void *v_cache = nullptr; // [2, max_ctx, 256] BF16
  void *k_cache_i8 = nullptr;
  void *v_cache_i8 = nullptr;
  void *k_scale_i8 = nullptr;
  void *v_scale_i8 = nullptr;

  float *d_e_raw = nullptr;    // [2048]
  float *d_e_norm = nullptr;   // [2048]
  float *d_h_norm = nullptr;   // [2048]
  float *d_x_cat = nullptr;    // [4096]
  float *d_fc_out = nullptr;   // [2048] FC projection output
  float *d_x = nullptr;        // [2048]
  int *d_draft_token = nullptr;// [1] output draft token
  void *d_fp32_weights_arena = nullptr; // [537088 floats = ~2.05 MiB] converted BF16->FP32 weights

  ze_command_list_handle_t cmd_draft = nullptr;
  bool initialized = false;
};


// Diagnostic Cache Header (T5.5)
struct DiagnosticCacheHeader {
  char magic[16];          // "AINFER_CACHE_V1\0"
  uint32_t version;        // 1
  uint32_t model_crc;      // Checksum or model ID
  uint32_t position;       // Current sequence context position
  uint32_t max_ctx;        // Context size
  uint32_t num_kv_layers;  // 10
  uint32_t num_ssm_layers; // 30
  uint64_t kv_bytes_total; // 10 * 2 * 2 * max_ctx * 256 * sizeof(uint16_t)
  uint64_t ssm_bytes_total;// 30 * 32 * 128 * 128 * sizeof(float)
  uint64_t conv_bytes_total;// 30 * 8192 * 3 * sizeof(float)
  uint32_t payload_crc32;  // CRC32 across all state buffers
};

class AInferRuntime258V {
public:
  AInferRuntime258V() = default;
  ~AInferRuntime258V() { cleanup(); }

  // T5.1: Initialize Level Zero, parse model container, allocate static arenas, load weights
  bool init(const std::string &binfer_path, const std::string &spv_path, uint32_t max_ctx = 2048);

  // T5.2: Chunked / token in-memory prefill writing directly into decode-layout KV & SSM states
  bool prefill(const std::vector<int> &prompt_ids, int *out_first_token = nullptr);

  // T5.4: Execute one recorded decode step with zero host allocations
  bool decode_step(int *out_next_token);

  // T6.4: Teacher-forced sequential evaluation across token trajectory
  bool teacher_forced_eval(const std::vector<int> &prompt_ids, std::vector<int> &out_predicted_tokens);

  // End-to-end generate prompt -> new tokens
  bool generate(const std::vector<int> &prompt_ids, int max_new_tokens,
                std::vector<int> &generated_ids, double *out_prefill_ms = nullptr,
                double *out_decode_tok_per_s = nullptr);

  // Performance profiling breakdown
  bool profile_step_breakdown(double &embed_ms, double &layers_ms, double &tail_ms,
                              double &step_ms, std::vector<double> &layer_times_ms);
  bool profile_prefill_breakdown(int B);
  bool profile_moe_shootout(int B);

  // T5.5: Save / restore offline diagnostic cache
  bool export_diagnostic_cache(const std::string &cache_file, uint32_t pos);
  bool import_diagnostic_cache(const std::string &cache_file, uint32_t *out_pos);

  // T5.6: Clean deterministic state reset
  bool reset_state();

  // Arena & performance getters
  uint64_t get_payload_arena_bytes() const { return pay_hi_ - pay_lo_; }
  uint64_t get_scale_arena_bytes() const { return sc_hi_ - sc_lo_; }
  uint64_t get_kv_cache_bytes() const { return kv_cache_bytes_; }
  uint64_t get_ssm_state_bytes() const { return ssm_state_bytes_; }
  uint64_t get_workspace_bytes() const { return workspace_bytes_; }
  const char *get_device_name() const { return dev_name_; }
  uint32_t get_max_ctx() const { return max_ctx_; }
  int get_current_position() const { return h_ctrl_.position; }
  bool is_initialized() const { return (dev_ != nullptr && queue_ != nullptr && cmd_step_ != nullptr); }

// T9.2: pure host-side step-parameter validator (no device access, so it
// is unit-testable without a GPU). Validates the dynamic values that flow
// into device control buffers BEFORE command-list submission:
//   position      in [0, max_ctx)          (KV-cache slot bound)
//   active_length in [position, max_ctx]   (bounded, non-decreasing vs
//     position; equality is legitimate after diagnostic-cache import,
//     which restores active_length == position. active_length is
//     host-informational — no kernel indexes by it.)
//   token_id      in [0, VOCAB_SIZE)       (embed-table bound)
//   chunk_tokens  in [1, MAX_PREFILL_CHUNK] (chunk-buffer bound)
// Returns true when valid; otherwise writes a diagnostic into err.
struct StepGuard {
  static bool check(int position, int active_length, int token_id,
                    int chunk_tokens, uint32_t max_ctx, char *err,
                    size_t errcap) {
    auto fail = [&](const char *what, long long v, long long lo, long long hi) {
      if (err && errcap) {
        std::snprintf(err, errcap, "step param %s=%lld outside [%lld,%lld)",
                      what, v, lo, hi);
      }
      return false;
    };
    if (position < 0 || (uint32_t)position >= max_ctx)
      return fail("position", position, 0, max_ctx);
    if (active_length < position || (uint32_t)active_length > max_ctx)
      return fail("active_length", active_length, position, (long long)max_ctx + 1);
    if (token_id < 0 || token_id >= VOCAB_SIZE)
      return fail("token_id", token_id, 0, VOCAB_SIZE);
    if (chunk_tokens < 1 || chunk_tokens > MAX_PREFILL_CHUNK)
      return fail("chunk_tokens", chunk_tokens, 1, MAX_PREFILL_CHUNK + 1);
    return true;
  }

  // Prompt-ID range scan (O(P) integer compares at prefill entry; device
  // embed gather clamps as last resort, but invalid IDs must not dispatch).
  static bool check_prompt_ids(const int *ids, size_t n, char *err,
                               size_t errcap) {
    if (!ids && n) {
      if (err && errcap) std::snprintf(err, errcap, "null prompt ids");
      return false;
    }
    for (size_t i = 0; i < n; ++i) {
      if (ids[i] < 0 || ids[i] >= VOCAB_SIZE) {
        if (err && errcap) {
          std::snprintf(err, errcap, "prompt_ids[%zu]=%d outside [0,%d)",
                        i, ids[i], VOCAB_SIZE);
        }
        return false;
      }
    }
    return true;
  }
};

  bool check_device_health() const {
    if (!dev_) return false;
    return (zeDeviceGetStatus(dev_) == ZE_RESULT_SUCCESS);
  }

  // T10.1: Multi-Token Prediction (MTP) Speculative Drafting & Verification
  bool init_mtp();
  bool mtp_draft_step(int *out_draft_token, double *out_latency_us = nullptr);
  bool has_mtp() const { return mtp_.initialized; }

  bool init_speculative_verification();
  bool speculative_step(int *out_tok1, int *out_tok2, int *out_num_emitted, bool *out_accepted, double *out_round_us = nullptr);
  bool generate_speculative(const std::vector<int> &prompt_ids, int max_new_tokens,
                            std::vector<int> &generated_ids, double *out_prefill_ms = nullptr,
                            double *out_spec_tok_per_s = nullptr, double *out_acceptance_rate = nullptr);
  bool has_speculative() const { return verify_initialized_; }

private:
  static uint32_t crc32_compute(uint32_t crc, const uint8_t *p, size_t n) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) {
      for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
          c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        tab[i] = c;
      }
      init = true;
    }
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i)
      crc = tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
  }

  bool load_model_metadata(const std::string &path);
  bool init_level_zero();
  bool allocate_static_arenas();
  bool verify_bindings() const; // T9.1: init-time null + arena-containment audit
  bool verify_payload_crcs(const std::string &path); // T9.5: per-tensor payload CRCs vs directory
  bool check_verified_stamp(const std::string &binfer_path) const; // I2.2: verified stamp check
  bool write_verified_stamp(const std::string &binfer_path) const; // I2.2: write verified stamp
  // T9.1: single range-checked container-offset resolvers (all call sites
  // delegate here; container offsets are untrusted input).
  void *checked_pay(const std::string &nm) const;
  void *checked_sc(const std::string &nm) const;
  bool upload_weights(const std::string &path);
  bool compile_kernels(const std::string &spv_path);
  bool record_command_lists();
  void cleanup();

  // File metadata
  std::unordered_map<std::string, BinferEntry> entries_;
  uint64_t pay_lo_ = UINT64_MAX, pay_hi_ = 0;
  uint64_t sc_lo_ = UINT64_MAX, sc_hi_ = 0;
  uint32_t dir_crc_ = 0;
  uint32_t moe_crc_ = 0;
  uint32_t max_ctx_ = 2048;
  bool kv8_enabled_ = false;

  // Level Zero Handles
  ze_driver_handle_t drv_ = nullptr;
  ze_device_handle_t dev_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
  ze_command_queue_handle_t queue_ = nullptr;
  ze_command_list_handle_t cmd_copy_ = nullptr;
  ze_fence_handle_t fence_ = nullptr;
  ze_result_t wait_fence(); // I3.5: centralized fence wait (power-mode aware)
  ze_module_handle_t mod_ = nullptr;
  ze_module_handle_t mod_kv8_ = nullptr;
  ze_module_handle_t mod_attn_ = nullptr; // T-decode-opt override module (optional)
  char dev_name_[256] = "Intel Arc 140V";

  // Static Arenas (T5.1)
  void *pay_arena_ = nullptr;
  void *sc_arena_ = nullptr;
  void *kv_cache_arena_ = nullptr;
  void *kv_scale_arena_ = nullptr;
  void *ssm_recr_arena_ = nullptr;
  void *ssm_conv_arena_ = nullptr;
  void *workspace_arena_ = nullptr;
  RuntimeControl *d_ctrl_ = nullptr; // Device control block
  RuntimeControl h_ctrl_{};          // Host shadow control block

  uint64_t kv_cache_bytes_ = 0;
  uint64_t kv_scale_bytes_ = 0;
  uint64_t ssm_state_bytes_ = 0;
  uint64_t workspace_bytes_ = 0;
  CheckedArena workspace_bump_; // T9.1: checked bump ledger for workspace_arena_

  // Root weights
  void *d_embed_tokens_ = nullptr;
  void *d_final_norm_w_ = nullptr;
  void *d_lm_head_w_ = nullptr;
  void *d_lm_head_s_ = nullptr;

  // Layer Bindings (40 layers)
  LayerBinding layers_[TOTAL_LAYERS];

  // Workspaces (within workspace_arena_)
  float *d_x_ = nullptr;
  float *d_x_norm_ = nullptr;
  float *d_qkv_ = nullptr;
  float *d_z_ = nullptr;
  float *d_a_ = nullptr;
  float *d_b_ = nullptr;
  float *d_qkv_conv_ = nullptr;
  float *d_q_ = nullptr;
  float *d_k_ = nullptr;
  float *d_g_ = nullptr;
  float *d_beta_ = nullptr;
  float *d_attn_out_ = nullptr;
  float *d_attn_split_ = nullptr; // [16, ATTN_SPLIT_MAX, 258] I3.8 partials
  float *d_attn_norm_ = nullptr;
  float *d_attn_proj_ = nullptr;
  float *d_x_mid_ = nullptr;
  float *d_x_post_ = nullptr;

  float *d_q_proj_raw_ = nullptr;
  float *d_q_full_ = nullptr;
  float *d_gate_full_ = nullptr;
  float *d_k_full_ = nullptr;
  float *d_v_full_ = nullptr;

  float *d_exp_gu_ = nullptr;
  float *d_exp_act_ = nullptr;
  float *d_exp_out_ = nullptr;
  float *d_moe_acc_ = nullptr;
  float *d_exp_gu_all_ = nullptr;  // [8 * 1024]
  float *d_exp_act_all_ = nullptr; // [8 * 512]
  float *d_sh_g_ = nullptr;
  float *d_sh_u_ = nullptr;
  float *d_sh_act_ = nullptr;
  float *d_sh_out_ = nullptr;

  float *d_logits_ = nullptr;
  float *d_stage1_vals_ = nullptr;
  uint32_t *d_stage1_idxs_ = nullptr;

  // Kernel Handles
  ze_kernel_handle_t k_gemv_ = nullptr;
  ze_kernel_handle_t k_router_ = nullptr;
  ze_kernel_handle_t k_norm2048_ = nullptr;
  ze_kernel_handle_t k_norm256_ = nullptr;
  ze_kernel_handle_t k_silu512_ = nullptr;
  ze_kernel_handle_t k_resadd_ = nullptr;
  ze_kernel_handle_t k_conv_ = nullptr;
  ze_kernel_handle_t k_l2_norm_ = nullptr;
  ze_kernel_handle_t k_gate_prep_ = nullptr;
  ze_kernel_handle_t k_recr_ = nullptr;
  ze_kernel_handle_t k_hnorm_ = nullptr;
  ze_kernel_handle_t k_argmax1_ = nullptr;

  // Batched MoE and Fused Tail Kernels
  ze_kernel_handle_t k_exp_gu_all_ = nullptr;
  ze_kernel_handle_t k_silu_all_ = nullptr;
  ze_kernel_handle_t k_exp_dn_accum_all_ = nullptr;
  ze_kernel_handle_t k_lm_head_argmax1_ = nullptr;

  // Control-buffer driven kernels (T5.3/T5.4)
  ze_kernel_handle_t k_embed_ = nullptr;
  ze_kernel_handle_t k_exp_gemv_ = nullptr;
  ze_kernel_handle_t k_exp_dn_accum_ = nullptr;
  ze_kernel_handle_t k_gemv_add_scaled_ = nullptr;
  ze_kernel_handle_t k_accum_ctrl_ = nullptr;
  ze_kernel_handle_t k_add_shared_ctrl_ = nullptr;
  ze_kernel_handle_t k_rope_ctrl_ = nullptr;
  ze_kernel_handle_t k_attn_ctrl_ = nullptr;
  ze_kernel_handle_t k_attn_split_ = nullptr;   // I3.8 split-T partials
  ze_kernel_handle_t k_attn_combine_ = nullptr; // I3.8 split-T merge
  int attn_split_s_ = 0; // I3.8: 0/1 = legacy single launch; N = T-splits
  ze_kernel_handle_t k_rope_ctrl_i8_ = nullptr;
  ze_kernel_handle_t k_attn_ctrl_i8_ = nullptr;
  ze_kernel_handle_t k_deinterleave_qg_ = nullptr;
  ze_kernel_handle_t k_argmax2_ctrl_ = nullptr;

  // Recorded Command Lists (T5.3)
  ze_command_list_handle_t cmd_embed_ = nullptr;
  std::vector<ze_command_list_handle_t> cmd_layers_{TOTAL_LAYERS, nullptr};
  ze_command_list_handle_t cmd_tail_ = nullptr;
  ze_command_list_handle_t cmd_prefill_step_ = nullptr; // Unified command list for prefill (embed + 40 layers)
  ze_command_list_handle_t cmd_step_ = nullptr;         // Unified command list for decode (embed + 40 layers + tail)
  std::vector<ze_command_list_handle_t> all_step_lists_; // embed + 40 layers + tail

  // Chunked Prefill Acceleration (T3.1 / T5.2)
  // Raised 256 -> 512 (2026-09-19): larger chunks improve MoE expert
  // grouping (16 vs 8 slots/expert avg) and DPAS tile fill; workspace
  // arena sized accordingly (256 MiB). Chunk buffers + cmd caches scale
  // automatically via this constant; overflow guard fires at init if fit
  // is wrong.
  static constexpr int MAX_PREFILL_CHUNK = 512;

  float *d_x_chunk_ = nullptr;            // [32, 2048]
  float *d_x_norm_chunk_ = nullptr;       // [32, 2048]
  float *d_x_mid_chunk_ = nullptr;        // [32, 2048]
  float *d_x_post_chunk_ = nullptr;       // [32, 2048]

  float *d_qkv_chunk_ = nullptr;          // [32, 8192]
  float *d_z_chunk_ = nullptr;            // [32, 4096]
  float *d_a_chunk_ = nullptr;            // [32, 32]
  float *d_b_chunk_ = nullptr;            // [32, 32]
  float *d_qkv_conv_chunk_ = nullptr;     // [32, 8192]
  float *d_q_chunk_ = nullptr;            // [32, 2048]
  float *d_k_chunk_ = nullptr;            // [32, 2048]
  float *d_g_chunk_ = nullptr;            // [32, 32]
  float *d_beta_chunk_ = nullptr;         // [32, 32]
  float *d_attn_out_chunk_ = nullptr;     // [32, 4096]
  float *d_attn_norm_chunk_ = nullptr;    // [32, 4096]
  float *d_attn_proj_chunk_ = nullptr;    // [32, 2048]

  float *d_q_proj_raw_chunk_ = nullptr;   // [32, 8192]
  float *d_q_full_chunk_ = nullptr;       // [32, 4096]
  float *d_gate_full_chunk_ = nullptr;    // [32, 4096]
  float *d_k_full_chunk_ = nullptr;       // [32, 512]
  float *d_v_full_chunk_ = nullptr;       // [32, 512]

  float *d_sh_g_chunk_ = nullptr;         // [32, 512]
  float *d_sh_u_chunk_ = nullptr;         // [32, 512]
  float *d_sh_act_chunk_ = nullptr;       // [32, 512]
  float *d_sh_down_chunk_ = nullptr;      // [32, 2048]

  uint32_t *d_top_idx_chunk_ = nullptr;  // [32, 8]
  float *d_top_wt_chunk_ = nullptr;      // [32, 8]
  float *d_sh_gate_chunk_ = nullptr;     // [32]

  float *d_exp_gu_chunk_ = nullptr;      // [32, 8, 1024]
  float *d_exp_act_chunk_ = nullptr;     // [32, 8, 512]
  float *d_exp_down_chunk_ = nullptr;    // [MAX_PREFILL_CHUNK, 8, 2048]
  float *d_moe_acc_chunk_ = nullptr;     // [32, 2048]
  int *d_tokens_chunk_ = nullptr;        // [32] shared memory
  int *d_expert_counts_ = nullptr;       // [256]
  int *d_expert_offsets_ = nullptr;      // [256]
  int *d_sorted_tokens_ = nullptr;       // [MAX_PREFILL_CHUNK * 8]
  int *d_sorted_slots_ = nullptr;        // [MAX_PREFILL_CHUNK * 8]
  int *d_active_expert_ids_ = nullptr;   // [256]
  int *d_num_active_experts_ = nullptr;  // [4] int + pad
  uint32_t *d_launch_args_gu_ = nullptr; // [3] groupCountX, Y, Z
  uint32_t *d_launch_args_dn_ = nullptr; // [3] groupCountX, Y, Z

  // Batch Kernel Handles
  ze_kernel_handle_t k_gemm_prefill_ = nullptr;
  int gemm_rows_per_group_ = 128; // 128: v1/v2 (M_tile=16); 256: v4 (M_tile=32)
  ze_kernel_handle_t k_embed_batch_ = nullptr;
  ze_kernel_handle_t k_norm2048_batch_ = nullptr;
  ze_kernel_handle_t k_conv_batch_ = nullptr;
  ze_kernel_handle_t k_l2_norm_qk_batch_ = nullptr;
  ze_kernel_handle_t k_gate_prep_batch_ = nullptr;
  ze_kernel_handle_t k_recr_batch_ = nullptr;
  ze_kernel_handle_t k_hnorm_batch_ = nullptr;
  ze_kernel_handle_t k_deinterleave_qg_batch_ = nullptr;
  ze_kernel_handle_t k_rope_batch_ = nullptr;
  ze_kernel_handle_t k_attn_batch_ = nullptr;
  ze_kernel_handle_t k_rope_batch_i8_ = nullptr;
  ze_kernel_handle_t k_attn_batch_i8_ = nullptr;
  bool flash_attn_prefill_ = true;
  ze_kernel_handle_t k_router_batch_ = nullptr;
  ze_kernel_handle_t k_moe_build_expert_bins_ = nullptr;
  ze_kernel_handle_t k_moe_build_expert_bins_compact_ = nullptr;
  ze_kernel_handle_t k_moe_gateup_grouped_batch_ = nullptr;
  ze_kernel_handle_t k_moe_gateup_compact_batch_ = nullptr;
  ze_kernel_handle_t k_moe_down_grouped_batch_ = nullptr;
  ze_kernel_handle_t k_moe_down_compact_batch_ = nullptr;
  bool moe_compact_ = false;
  ze_kernel_handle_t k_moe_accum_down_batch_ = nullptr;
  ze_kernel_handle_t k_exp_gu_all_batch_ = nullptr;
  ze_kernel_handle_t k_silu_all_batch_ = nullptr;
  ze_kernel_handle_t k_exp_dn_accum_all_batch_ = nullptr;
  ze_kernel_handle_t k_silu_mul_batch_ = nullptr;
  ze_kernel_handle_t k_block_resadd_moe_batch_ = nullptr;
  ze_kernel_handle_t k_moe_add_shared_batch_ = nullptr; // verify-vs-decode parity (decode order)
  ze_kernel_handle_t k_resadd_batch_ = nullptr;

  // Cached chunk command lists: cmd_prefill_chunk_[B] and cmd_prefill_tail_[B] for B in 1..32
  ze_command_list_handle_t cmd_prefill_chunk_[MAX_PREFILL_CHUNK + 1]{};
  ze_command_list_handle_t cmd_prefill_tail_[MAX_PREFILL_CHUNK + 1]{};

  ze_command_list_handle_t get_or_record_prefill_chunk_list(int B);
  ze_command_list_handle_t get_or_record_prefill_tail_list(int B);
  // MTP prompt-KV fill: per-chunk MTP QKV + RoPE append into mtp_.k/v_cache
  // so long-context drafts attend over the prompt (not zeros). Recorded
  // lazily per B; executed inside prefill() right after each trunk chunk.
  ze_command_list_handle_t get_or_record_mtp_prefill_chunk_list(int B);
  ze_command_list_handle_t cmd_mtp_prefill_chunk_[MAX_PREFILL_CHUNK + 1]{};

  // MTP Runtime State & Kernels (T10.1)
  MtpBinding mtp_{};
  ze_kernel_handle_t k_concat2_ = nullptr;
  ze_kernel_handle_t k_argmax2_mtp_ = nullptr;

  // Speculative Verification (T10.1 Dual-Token Verification)
  bool verify_initialized_ = false;
  int pending_draft_token_ = -1;

  void *d_conv_snap_ = nullptr;        // [30 * 8192 * 3 * sizeof(float)]
  void *d_ssm_snap_ = nullptr;         // [30 * 32 * 128 * 128 * sizeof(float)]
  ArenaSpan<uint8_t> conv_snap_span_;  // T9.1: checked view of d_conv_snap_
  ArenaSpan<uint8_t> ssm_snap_span_;   // T9.1: checked view of d_ssm_snap_
  float *d_stage1_vals_1_ = nullptr;   // [1024]
  uint32_t *d_stage1_idxs_1_ = nullptr;// [1024]
  int *d_verify_tokens_ = nullptr;     // [2] USM shared

  ze_kernel_handle_t k_conv_m2_spec_ = nullptr;
  ze_kernel_handle_t k_recr_m2_spec_ = nullptr;
  ze_kernel_handle_t k_lm_head_m2_argmax1_ = nullptr;
  ze_kernel_handle_t k_gemv_m2_ = nullptr;

  ze_command_list_handle_t cmd_verify_m2_ = nullptr;
  ze_command_list_handle_t cmd_rollback_ = nullptr;
};

} // namespace ainfer
