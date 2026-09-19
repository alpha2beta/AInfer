#include "../decode/runtime_258v.h"
#include <chrono>
#include <cstdio>
#include <vector>
using namespace ainfer;
int main() {
  AInferRuntime258V rt;
  if (!rt.init("models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer",
               "tools/kernels_258v/all_kernels.spv", 2048)) return 1;
  std::vector<int> sample = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
  const int P = 441;
  std::vector<int> prompt;
  while ((int)prompt.size() < P) for (int t : sample) if ((int)prompt.size() < P) prompt.push_back(t);
  int dummy = 0;
  rt.reset_state(); rt.prefill(prompt, &dummy); // warmup
  double tot = 0; const int RUNS = 5;
  for (int r = 0; r < RUNS; ++r) {
    rt.reset_state();
    auto t0 = std::chrono::steady_clock::now();
    rt.prefill(prompt, &dummy);
    auto t1 = std::chrono::steady_clock::now();
    tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  double avg = tot / RUNS;
  std::printf("P=441: %.2f ms, %.2f tok/s, %.2f ms/tok (avg of %d)\n", avg, P / (avg * 1e-3), avg / P, RUNS);
  return 0;
}
