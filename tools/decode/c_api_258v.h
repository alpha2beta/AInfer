#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *ainfer_create();
int ainfer_init(void *handle, const char *binfer_path, const char *spv_path, uint32_t max_ctx);
int ainfer_prefill(void *handle, const int *prompt_ids, int count, int *out_first_token);
int ainfer_prefill_incremental(void *handle, const int *suffix_ids, int count, int start_pos,
                               int *out_first_token);
int ainfer_prefix_cached_len(void *handle);
int ainfer_prefill_anchor(void *handle, const int *new_ids, int count, int anchor_pos,
                          int *out_first_token);
int ainfer_prefix_anchor_len(void *handle);
int ainfer_decode_step(void *handle, int *out_next_token);
int ainfer_init_speculative(void *handle);
int ainfer_speculative_step(void *handle, int *out_tok1, int *out_tok2, int *out_num_emitted, int *out_accepted);
int ainfer_has_speculative(void *handle);
int ainfer_reset_state(void *handle);
int ainfer_get_position(void *handle);
int ainfer_is_initialized(void *handle);
int ainfer_check_device_health(void *handle);
const char *ainfer_get_device_name(void *handle);
uint64_t ainfer_get_total_memory_bytes(void *handle);
void ainfer_destroy(void *handle);

#ifdef __cplusplus
}
#endif
