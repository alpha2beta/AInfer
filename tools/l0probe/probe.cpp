// AInfer Level Zero device-capability probe (T0.2 / Gate B).
// Pure Level Zero C API, no SYCL. Selects the Arc Pro B60 by PCI ID
// (8086:e211) and emits a machine-readable JSON report to stdout.
// Usage: l0probe [output.json]
#include <level_zero/ze_api.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
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

static void jstr(const char *s) {
  std::putchar('"');
  for (; *s; ++s) {
    if (*s == '"' || *s == '\\')
      std::putchar('\\');
    std::putchar(*s);
  }
  std::putchar('"');
}

int main(int argc, char **argv) {
  FILE *out = stdout;
  if (argc > 1) {
    out = std::fopen(argv[1], "w");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", argv[1]);
      return 1;
    }
  }
  // Redirect JSON via out; diagnostics stay on stderr. Use freopen trick:
  // simplest is to build with stdout redirected by caller; here we dup.
  if (out != stdout) {
    std::fclose(stdout);
    stdout = out;
  }

  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t nDrivers = 0;
  CHECK(zeDriverGet(&nDrivers, nullptr));
  std::vector<ze_driver_handle_t> drivers(nDrivers);
  CHECK(zeDriverGet(&nDrivers, drivers.data()));

  std::printf("{\"drivers\":[\n");
  ze_device_handle_t b60 = nullptr;
  for (uint32_t di = 0; di < nDrivers; ++di) {
    ze_api_version_t apiVer = {};
    CHECK(zeDriverGetApiVersion(drivers[di], &apiVer));
    ze_driver_properties_t drvProps = {ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES,
                                       nullptr, {}, 0};
    CHECK(zeDriverGetProperties(drivers[di], &drvProps));
    uint32_t nExt = 0;
    CHECK(zeDriverGetExtensionProperties(drivers[di], &nExt, nullptr));
    std::vector<ze_driver_extension_properties_t> exts(nExt);
    CHECK(zeDriverGetExtensionProperties(drivers[di], &nExt, exts.data()));

    uint32_t nDev = 0;
    CHECK(zeDeviceGet(drivers[di], &nDev, nullptr));
    std::vector<ze_device_handle_t> devs(nDev);
    CHECK(zeDeviceGet(drivers[di], &nDev, devs.data()));

    std::printf(
        "%s{\"api\":\"%d.%d\",\"driver_version\":%" PRIu32
        ",\"extensions\":[",
        di ? ",\n" : "", ZE_MAJOR_VERSION(apiVer), ZE_MINOR_VERSION(apiVer),
        drvProps.driverVersion);
    for (uint32_t i = 0; i < nExt; ++i)
      std::printf("%s\"%s\"", i ? "," : "", exts[i].name);
    std::printf("],\"devices\":[\n");

    for (uint32_t vi = 0; vi < nDev; ++vi) {
      ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      CHECK(zeDeviceGetProperties(devs[vi], &p));
      ze_device_compute_properties_t c = {
          ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
      CHECK(zeDeviceGetComputeProperties(devs[vi], &c));
      ze_device_module_properties_t m = {
          ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES};
      CHECK(zeDeviceGetModuleProperties(devs[vi], &m));
      uint32_t nMem = 0;
      CHECK(zeDeviceGetMemoryProperties(devs[vi], &nMem, nullptr));
      std::vector<ze_device_memory_properties_t> mems(nMem);
      for (auto &mm : mems) {
        mm.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        mm.pNext = nullptr;
      }
      CHECK(zeDeviceGetMemoryProperties(devs[vi], &nMem, mems.data()));
      uint32_t nCache = 0;
      CHECK(zeDeviceGetCacheProperties(devs[vi], &nCache, nullptr));
      std::vector<ze_device_cache_properties_t> caches(nCache);
      for (auto &cc : caches) {
        cc.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
        cc.pNext = nullptr;
      }
      CHECK(zeDeviceGetCacheProperties(devs[vi], &nCache, caches.data()));
      uint32_t nQ = 0;
      CHECK(zeDeviceGetCommandQueueGroupProperties(devs[vi], &nQ, nullptr));
      std::vector<ze_command_queue_group_properties_t> qs(nQ);
      for (auto &q : qs) {
        q.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        q.pNext = nullptr;
      }
      CHECK(zeDeviceGetCommandQueueGroupProperties(devs[vi], &nQ, qs.data()));
      ze_device_image_properties_t img = {
          ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES};
      ze_result_t imgRes = zeDeviceGetImageProperties(devs[vi], &img);
      uint32_t nSub = 0;
      CHECK(zeDeviceGetSubDevices(devs[vi], &nSub, nullptr));

      if (p.vendorId == 0x8086 && (p.deviceId == 0x64a0 || p.deviceId == 0xe211 || b60 == nullptr))
        b60 = devs[vi];

      std::printf("%s{\"name\":", vi ? ",\n" : "\n");
      jstr(p.name);
      std::printf(
          ",\"pci\":\"%04x:%04x\",\"type\":%d,\"flags\":%u,\"subdevices\":%u,"
          "\"core_clock_mhz\":%u,\"max_mem_alloc\":%" PRIu64
          ",\"max_hw_contexts\":%u,\"topology\":{\"slices\":%u,\"subslices_"
          "per_slice\":%u,\"eus_per_subslice\":%u,\"threads_per_eu\":%u,"
          "\"simd_width\":%u},\"timer\":{\"resolution_ns\":%" PRIu64
          ",\"valid_bits\":%u,\"kernel_valid_bits\":%u},\"compute\":{"
          "\"max_total_group\":%u,\"max_group_xyz\":[%u,%u,%u],\"max_count_xyz\":"
          "[%u,%u,%u],\"max_slm\":%u,\"subgroup_sizes\":[",
          p.vendorId, p.deviceId, (int)p.type, p.flags, nSub, p.coreClockRate,
          p.maxMemAllocSize, p.maxHardwareContexts, p.numSlices,
          p.numSubslicesPerSlice, p.numEUsPerSubslice, p.numThreadsPerEU,
          p.physicalEUSimdWidth, p.timerResolution, p.timestampValidBits,
          p.kernelTimestampValidBits, c.maxTotalGroupSize, c.maxGroupSizeX,
          c.maxGroupSizeY, c.maxGroupSizeZ, c.maxGroupCountX, c.maxGroupCountY,
          c.maxGroupCountZ, c.maxSharedLocalMemory);
      for (uint32_t i = 0; i < c.numSubGroupSizes; ++i)
        std::printf("%s%u", i ? "," : "", c.subGroupSizes[i]);
      std::printf(
          "]},\"module\":{\"spirv\":\"%d.%d\",\"flags\":%u,\"fp16\":%u,\"fp32\":"
          "%u,\"fp64\":%u,\"max_args\":%u,\"printf_buf\":%u},\"memory\":[",
          ZE_MAJOR_VERSION(m.spirvVersionSupported),
          ZE_MINOR_VERSION(m.spirvVersionSupported), m.flags, m.fp16flags,
          m.fp32flags, m.fp64flags, m.maxArgumentsSize, m.printfBufferSize);
      for (uint32_t i = 0; i < nMem; ++i) {
        std::printf("%s{\"name\":", i ? "," : "");
        jstr(mems[i].name);
        std::printf(",\"flags\":%u,\"clock_mhz\":%u,\"bus_width\":%u,"
                    "\"total\":%" PRIu64 "}",
                    mems[i].flags, mems[i].maxClockRate, mems[i].maxBusWidth,
                    mems[i].totalSize);
      }
      std::printf("],\"caches\":[");
      for (uint32_t i = 0; i < nCache; ++i)
        std::printf("%s{\"flags\":%u,\"size\":%llu}", i ? "," : "",
                    caches[i].flags, (unsigned long long)caches[i].cacheSize);
      std::printf("],\"queue_groups\":[");
      for (uint32_t i = 0; i < nQ; ++i)
        std::printf("%s{\"flags\":%u,\"fill_pattern\":%llu,\"queues\":%u}",
                    i ? "," : "", qs[i].flags,
                    (unsigned long long)qs[i].maxMemoryFillPatternSize,
                    qs[i].numQueues);
      if (imgRes == ZE_RESULT_SUCCESS)
        std::printf(
            "],\"images\":{\"dims_1d\":%u,\"dims_2d\":%u,\"dims_3d\":%u,"
            "\"buf_size\":%" PRIu64 ",\"samplers\":%u}}",
            img.maxImageDims1D, img.maxImageDims2D, img.maxImageDims3D,
            img.maxImageBufferSize, img.maxSamplers);
      else
        std::printf("],\"images\":null}");
    }
    std::printf("\n]}");
  }
  std::printf("\n],\"target_gpu_found\":%s}\n", b60 ? "true" : "false");
  if (!b60) {
    std::fprintf(stderr, "WARNING: Target Intel GPU not enumerated\n");
    return 2;
  }
  return 0;
}
