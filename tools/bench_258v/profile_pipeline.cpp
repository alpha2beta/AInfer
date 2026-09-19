#include "../decode/runtime_258v.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  const char *model_path = (argc > 1) ? argv[1] : "models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer";
  const char *spv_path = (argc > 2) ? argv[2] : "tools/kernels_258v/all_kernels.spv";

  AInferRuntime258V rt;
  if (!rt.init(model_path, spv_path, 2048)) {
    std::fprintf(stderr, "Failed to init runtime\n");
    return 1;
  }

  std::vector<int> prompt = {1, 15043, 29892, 1024};
  int first_tok = 0;
  rt.prefill(prompt, &first_tok);

  // Warmup step
  int next_tok = 0;
  rt.decode_step(&next_tok);

  // Run profiling breakdown
  double embed_ms = 0, layers_ms = 0, tail_ms = 0, step_ms = 0;
  std::vector<double> layer_times_ms;

  const int ITERS = 5;
  double sum_embed = 0, sum_layers = 0, sum_tail = 0, sum_step = 0;
  std::vector<double> sum_layer_times(TOTAL_LAYERS, 0.0);

  for (int it = 0; it < ITERS; ++it) {
    rt.profile_step_breakdown(embed_ms, layers_ms, tail_ms, step_ms, layer_times_ms);
    sum_embed += embed_ms;
    sum_layers += layers_ms;
    sum_tail += tail_ms;
    sum_step += step_ms;
    for (int l = 0; l < TOTAL_LAYERS; ++l) {
      sum_layer_times[l] += layer_times_ms[l];
    }
  }

  double avg_embed = sum_embed / ITERS;
  double avg_layers = sum_layers / ITERS;
  double avg_tail = sum_tail / ITERS;
  double avg_step = sum_step / ITERS;

  std::printf("\n=================================================================\n");
  std::printf("--- Decode Step Profiling Breakdown (Arc 140V Xe2) ---\n");
  std::printf("=================================================================\n");
  std::printf("  Unified Single cmd_step: %.3f ms (Throughput: %.2f tok/s)\n", avg_step, 1000.0 / avg_step);
  std::printf("  Embed Latency:           %.3f ms (%.1f%%)\n", avg_embed, (avg_embed / avg_step) * 100.0);
  std::printf("  Total 40 Layers Latency: %.3f ms (%.1f%%)\n", avg_layers, (avg_layers / avg_step) * 100.0);
  std::printf("  Tail (LM Head + Argmax): %.3f ms (%.1f%%)\n", avg_tail, (avg_tail / avg_step) * 100.0);
  std::printf("-----------------------------------------------------------------\n");

  double sum_deltanet = 0.0;
  double sum_fullattn = 0.0;
  int cnt_deltanet = 0, cnt_fullattn = 0;

  for (int l = 0; l < TOTAL_LAYERS; ++l) {
    double l_ms = sum_layer_times[l] / ITERS;
    bool is_full = ((l + 1) % FULL_ATTN_INTERVAL == 0);
    if (is_full) {
      sum_fullattn += l_ms;
      cnt_fullattn++;
    } else {
      sum_deltanet += l_ms;
      cnt_deltanet++;
    }
  }

  std::printf("  DeltaNet Layer (mean of %d): %.3f ms/layer (Total: %.2f ms)\n",
              cnt_deltanet, sum_deltanet / cnt_deltanet, sum_deltanet);
  std::printf("  Full-Attn Layer (mean of %d): %.3f ms/layer (Total: %.2f ms)\n",
              cnt_fullattn, sum_fullattn / cnt_fullattn, sum_fullattn);
  std::printf("=================================================================\n\n");

  return 0;
}
