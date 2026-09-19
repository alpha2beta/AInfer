#include "../decode/runtime_258v.h"
#include <cstdio>
using namespace ainfer;
int main() {
  AInferRuntime258V rt;
  if (!rt.init("models/Tiel-Coder-35B-A3B-Genesis-Hermes/tiel-coder-35b-text-int4g128.binfer",
               "tools/kernels_258v/all_kernels.spv", 2048)) return 1;
  rt.profile_prefill_breakdown(512);
  return 0;
}
