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

int ainfer_decode_step(void *handle, int *out_next_token) {
  if (!handle || !out_next_token) return -1;
  auto *rt = reinterpret_cast<AInferRuntime258V *>(handle);
  return rt->decode_step(out_next_token) ? 0 : -1;
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
