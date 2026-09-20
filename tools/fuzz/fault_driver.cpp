// T9.5: fault-injection driver for the AInfer runtime.
// Modes (each prints a single RESULT line; exit 0 = driver survived):
//   init   <model> <spv>                      -> RESULT init=<0|1>
//   import <model> <spv> <cache>              -> RESULT import=<0|1>
//   export <model> <spv> <outpath>            -> RESULT export=<0|1>
// Any crash (signal), hang (harness timeout), or C++ exception escaping
// main is itself a T9.5 finding. Driven by tools/fuzz/fault_inject.py.
#include "../decode/runtime_258v.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace ainfer;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s init|import|export ...\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];

  if (mode == "init") {
    if (argc < 4) return 2;
    AInferRuntime258V rt;
    bool ok = rt.init(argv[2], argv[3], 2048);
    std::printf("RESULT init=%d\n", ok ? 1 : 0);
    return 0;
  }

  if (mode == "import") {
    if (argc < 5) return 2;
    AInferRuntime258V rt;
    if (!rt.init(argv[2], argv[3], 2048)) {
      std::printf("RESULT init=0 import=skipped\n");
      return 0;
    }
    uint32_t pos = 0xDEADu;
    bool ok = rt.import_diagnostic_cache(argv[4], &pos);
    std::printf("RESULT init=1 import=%d pos=%u\n", ok ? 1 : 0, pos);
    return 0;
  }

  if (mode == "export") {
    if (argc < 5) return 2;
    AInferRuntime258V rt;
    if (!rt.init(argv[2], argv[3], 2048)) {
      std::printf("RESULT init=0 export=skipped\n");
      return 0;
    }
    std::vector<int> prompt = {151644, 8948, 198, 2610, 525, 264, 10925, 151645};
    int tok = 0;
    rt.reset_state();
    if (!rt.prefill(prompt, &tok)) {
      std::printf("RESULT init=1 export=prefill-failed\n");
      return 0;
    }
    bool ok = rt.export_diagnostic_cache(argv[4], 8);
    std::printf("RESULT init=1 export=%d\n", ok ? 1 : 0);
    return 0;
  }

  std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
  return 2;
}
