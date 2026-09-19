// Diagnostic tool to inspect MTP intermediate activations on Arc 140V
#include "../decode/runtime_258v.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ainfer;

void print_stats(const char *name, const float *data, size_t n) {
  float min_val = 1e30f, max_val = -1e30f, sum = 0.0f;
  int nans = 0, infs = 0, zeros = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = data[i];
    if (std::isnan(v)) nans++;
    else if (std::isinf(v)) infs++;
    else {
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
      sum += v;
      if (v == 0.0f) zeros++;
    }
  }
  std::printf("  %-20s [%6zu]: min=%10.4f, max=%10.4f, mean=%10.4f, zeros=%zu, nans=%d, infs=%d\n",
              name, n, (nans + infs == (int)n) ? 0.0f : min_val,
              (nans + infs == (int)n) ? 0.0f : max_val,
              sum / (n - nans - infs > 0 ? n - nans - infs : 1), zeros, nans, infs);
}

int main(int argc, char **argv) {
  const char *model_path = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path   = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V runtime;
  if (!runtime.init(model_path, spv_path, 2048)) return 1;
  if (!runtime.init_mtp()) return 1;

  std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645, 151648, 198, 936, 31920, 9467, 7, 82, 25, 1146, 1000, 1146, 512};
  runtime.reset_state();
  int first_tok = 0;
  runtime.prefill(prompt, &first_tok);
  std::printf("Prefill completed. First decoded token: %d\n", first_tok);

  int draft_tok = -999;
  double draft_us = 0.0;
  runtime.mtp_draft_step(&draft_tok, &draft_us);
  std::printf("MTP Draft returned token: %d (in %.2f us)\n", draft_tok, draft_us);

  int next_tok = 0;
  runtime.decode_step(&next_tok);
  std::printf("Trunk decode step returned token: %d\n", next_tok);

  return 0;
}
