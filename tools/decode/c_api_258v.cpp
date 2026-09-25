#include "c_api_258v.h"
#include "runtime_258v.h"

#include <vector>

using namespace ainfer;

extern "C" {

void *ainfer_create() {
  return new (std::nothrow) AInferRuntime258V();
}

int ainfer_init(void *handle, const char *binfer_path, const char *spv_path, uint32_t max_ctx) {
  if (!handle || !binfer_path || !spv_path) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->init(binfer_path, spv_path, max_ctx) ? 0 : -1;
}

int ainfer_prefill(void *handle, const int *prompt_ids, int count, int *out_first_token) {
  if (!handle || !prompt_ids || count <= 0) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  std::vector<int> ids(prompt_ids, prompt_ids + count);
  return rt->prefill(ids, out_first_token) ? 0 : -1;
}

// I4.4: incremental append prefill over the snapshotted prefix of exactly
// start_pos tokens. Returns nonzero when no valid snapshot exists (caller
// falls back to ainfer_prefill); device state is unchanged on that path.
int ainfer_prefill_incremental(void *handle, const int *suffix_ids, int count, int start_pos,
                               int *out_first_token) {
  if (!handle || !suffix_ids || count <= 0 || start_pos < 0) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  std::vector<int> ids(suffix_ids, suffix_ids + count);
  return rt->prefill_incremental(ids, start_pos, out_first_token) ? 0 : -1;
}

// I4.4b: rebase onto the terminal-chunk-boundary anchor. new_ids is the FULL
// new prompt; anchor_pos must equal ainfer_prefix_anchor_len(). Nonzero
// return = no valid anchor (caller falls back to ainfer_prefill).
int ainfer_prefill_anchor(void *handle, const int *new_ids, int count, int anchor_pos,
                          int *out_first_token) {
  if (!handle || !new_ids || count <= 0 || anchor_pos < 0) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  std::vector<int> ids(new_ids, new_ids + count);
  return rt->prefill_from_anchor(ids, anchor_pos, out_first_token) ? 0 : -1;
}

// I4.4: length of the prompt whose post-prefill state is snapshotted, or -1.
int ainfer_prefix_cached_len(void *handle) {
  if (!handle) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->prefix_cached_len();
}

// I4.4b: terminal-chunk-boundary anchor position of the cached prompt, or -1.
int ainfer_prefix_anchor_len(void *handle) {
  if (!handle) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->prefix_anchor_len();
}

int ainfer_decode_step(void *handle, int *out_next_token) {
  if (!handle || !out_next_token) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->decode_step(out_next_token) ? 0 : -1;
}

int ainfer_init_speculative(void *handle) {
  if (!handle) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->init_speculative_verification() ? 0 : -1;
}

int ainfer_speculative_step(void *handle, int *out_tok1, int *out_tok2, int *out_num_emitted, int *out_accepted) {
  if (!handle || !out_tok1 || !out_tok2 || !out_num_emitted || !out_accepted) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  bool acc = false;
  bool ok = rt->speculative_step(out_tok1, out_tok2, out_num_emitted, &acc);
  *out_accepted = acc ? 1 : 0;
  return ok ? 0 : -1;
}

int ainfer_has_speculative(void *handle) {
  if (!handle) return 0;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->has_speculative() ? 1 : 0;
}

int ainfer_reset_state(void *handle) {
  if (!handle) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->reset_state() ? 0 : -1;
}

int ainfer_get_position(void *handle) {
  if (!handle) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->get_current_position();
}

int ainfer_is_initialized(void *handle) {
  if (!handle) return 0;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->is_initialized() ? 1 : 0;
}

int ainfer_check_device_health(void *handle) {
  if (!handle) return 0;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->check_device_health() ? 1 : 0;
}

const char *ainfer_get_device_name(void *handle) {
  if (!handle) return "Unknown";
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->get_device_name();
}

uint64_t ainfer_get_total_memory_bytes(void *handle) {
  if (!handle) return 0;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->get_payload_arena_bytes() + rt->get_scale_arena_bytes() +
         rt->get_kv_cache_bytes() + rt->get_ssm_state_bytes() +
         rt->get_workspace_bytes();
}

void ainfer_destroy(void *handle) {
  if (handle) {
    auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
    delete rt;
  }
}

}
