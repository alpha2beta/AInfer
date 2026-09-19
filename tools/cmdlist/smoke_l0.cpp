// AInfer Level Zero Smoke Test for Intel Arc 140V (Lunar Lake Xe2)
// Tests: Level Zero GPU initialization, SPIR-V module compilation,
// kernel execution, device memory allocation, and readback.
#include <level_zero/ze_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t _r = (expr);                                                   \
    if (_r != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)_r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  const char *spv_path = (argc > 1) ? argv[1] : "/tmp/test_spv.o";
  std::ifstream spv_f(spv_path, std::ios::binary);
  if (!spv_f) {
    std::fprintf(stderr, "cannot open SPIR-V file: %s\n", spv_path);
    return 1;
  }
  spv_f.seekg(0, std::ios::end);
  size_t spv_size = spv_f.tellg();
  spv_f.seekg(0, std::ios::beg);
  std::vector<uint8_t> spv(spv_size);
  spv_f.read((char *)spv.data(), spv_size);

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrv = 0;
  CHECK(zeDriverGet(&nDrv, nullptr));
  std::vector<ze_driver_handle_t> drvs(nDrv);
  CHECK(zeDriverGet(&nDrv, drvs.data()));

  ze_device_handle_t dev = nullptr;
  ze_driver_handle_t drv = nullptr;
  char devName[256] = "Unknown";

  for (auto d : drvs) {
    uint32_t nv = 0;
    if (zeDeviceGet(d, &nv, nullptr) != ZE_RESULT_SUCCESS)
      continue;
    std::vector<ze_device_handle_t> vs(nv);
    zeDeviceGet(d, &nv, vs.data());
    for (auto v : vs) {
      ze_device_properties_t pr = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      if (zeDeviceGetProperties(v, &pr) == ZE_RESULT_SUCCESS && pr.vendorId == 0x8086) {
        if (pr.deviceId == 0x64a0 || dev == nullptr) {
          dev = v;
          drv = d;
          std::strncpy(devName, pr.name, sizeof(devName) - 1);
        }
      }
    }
  }

  if (!dev) {
    std::fprintf(stderr, "No Intel GPU device found\n");
    return 1;
  }
  std::printf("Using device: %s\n", devName);

  ze_context_handle_t ctx = nullptr;
  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  CHECK(zeContextCreate(drv, &cdesc, &ctx));

  // Compile SPIR-V module
  ze_module_desc_t mdesc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                            nullptr,
                            ZE_MODULE_FORMAT_IL_SPIRV,
                            spv_size,
                            spv.data(),
                            nullptr,
                            nullptr};
  ze_module_handle_t mod = nullptr;
  CHECK(zeModuleCreate(ctx, dev, &mdesc, &mod, nullptr));

  // Create kernel
  ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "test_k"};
  ze_kernel_handle_t kern = nullptr;
  CHECK(zeKernelCreate(mod, &kdesc, &kern));

  // Allocate device memory
  ze_device_mem_alloc_desc_t memDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                                       nullptr, 0, 0};
  void *devPtr = nullptr;
  CHECK(zeMemAllocDevice(ctx, &memDesc, sizeof(int), 64, dev, &devPtr));

  // Set kernel arguments
  CHECK(zeKernelSetArgumentValue(kern, 0, sizeof(void *), &devPtr));
  CHECK(zeKernelSetGroupSize(kern, 1, 1, 1));

  // Immediate synchronous command list
  ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                   nullptr, 0, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
                                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &list));

  ze_group_count_t gcnt = {1, 1, 1};
  CHECK(zeCommandListAppendLaunchKernel(list, kern, &gcnt, nullptr, 0, nullptr));

  int hostVal = 0;
  CHECK(zeCommandListAppendMemoryCopy(list, &hostVal, devPtr, sizeof(int),
                                     nullptr, 0, nullptr));

  std::printf("Result from GPU: %d (expected 42)\n", hostVal);

  CHECK(zeCommandListDestroy(list));
  CHECK(zeKernelDestroy(kern));
  CHECK(zeModuleDestroy(mod));
  CHECK(zeMemFree(ctx, devPtr));
  CHECK(zeContextDestroy(ctx));

  if (hostVal == 42) {
    std::printf("SMOKE TEST PASSED!\n");
    return 0;
  } else {
    std::fprintf(stderr, "SMOKE TEST FAILED: mismatch\n");
    return 1;
  }
}
