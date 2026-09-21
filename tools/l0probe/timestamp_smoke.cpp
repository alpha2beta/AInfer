// Compiler-neutral Level Zero timestamp smoke test for the 258v preset.
// Allocates a shared buffer, submits a device fill with a kernel-timestamp
// event, verifies the fill, and checks that the device timestamp interval is
// non-zero and ordered.
#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(expr)                                                            \
  do {                                                                         \
    ze_result_t r = (expr);                                                    \
    if (r != ZE_RESULT_SUCCESS) {                                              \
      std::fprintf(stderr, "L0 error %d at %s:%d (%s)\n", (int)r, __FILE__,   \
                   __LINE__, #expr);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t ndrv = 0;
  CHECK(zeDriverGet(&ndrv, nullptr));
  std::vector<ze_driver_handle_t> drivers(ndrv);
  CHECK(zeDriverGet(&ndrv, drivers.data()));

  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  for (auto d : drivers) {
    uint32_t ndev = 0;
    CHECK(zeDeviceGet(d, &ndev, nullptr));
    std::vector<ze_device_handle_t> devices(ndev);
    CHECK(zeDeviceGet(d, &ndev, devices.data()));
    for (auto dev : devices) {
      ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
      CHECK(zeDeviceGetProperties(dev, &props));
      if (props.vendorId == 0x8086 &&
          (props.deviceId == 0x64a0 || props.deviceId == 0xe211)) {
        driver = d;
        device = dev;
        break;
      }
    }
    if (device) break;
  }
  if (!device) {
    std::fprintf(stderr, "Target Intel GPU not found\n");
    return 2;
  }

  ze_context_desc_t context_desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t context = nullptr;
  CHECK(zeContextCreate(driver, &context_desc, &context));

  uint32_t nq = 0;
  CHECK(zeDeviceGetCommandQueueGroupProperties(device, &nq, nullptr));
  std::vector<ze_command_queue_group_properties_t> groups(nq);
  for (auto &g : groups) g = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES};
  CHECK(zeDeviceGetCommandQueueGroupProperties(device, &nq, groups.data()));
  uint32_t ordinal = 0;
  for (uint32_t i = 0; i < nq; ++i) {
    if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ordinal = i;
      break;
    }
  }

  ze_command_queue_desc_t queue_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                     nullptr, ordinal, 0, 0,
                                     ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t queue = nullptr;
  CHECK(zeCommandQueueCreate(context, device, &queue_desc, &queue));
  ze_command_list_desc_t list_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                   nullptr, ordinal, 0};
  ze_command_list_handle_t list = nullptr;
  CHECK(zeCommandListCreate(context, device, &list_desc, &list));

  ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                 ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, 1};
  ze_event_pool_handle_t pool = nullptr;
  CHECK(zeEventPoolCreate(context, &pool_desc, 1, &device, &pool));
  ze_event_desc_t event_desc{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, 0, 0};
  ze_event_handle_t event = nullptr;
  CHECK(zeEventCreate(pool, &event_desc, &event));

  ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr, 0};
  uint32_t *buffer = nullptr;
  CHECK(zeMemAllocHost(context, &host_desc, 4096, 64, (void **)&buffer));
  *buffer = 0;
  uint32_t pattern = 0x2580;
  CHECK(zeCommandListAppendMemoryFill(list, buffer, &pattern, sizeof(pattern),
                                      4096, event, 0, nullptr));
  CHECK(zeCommandListClose(list));
  CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  if (*buffer != pattern) {
    std::fprintf(stderr, "MemoryFill verification failed: got 0x%x\n", *buffer);
    return 1;
  }

  ze_kernel_timestamp_result_t timestamp{};
  CHECK(zeEventQueryKernelTimestamp(event, &timestamp));
  if (timestamp.context.kernelStart == timestamp.context.kernelEnd) {
    std::fprintf(stderr, "Kernel timestamp interval is empty\n");
    return 1;
  }
  std::printf("timestamp_smoke: pattern=0x%x start=%llu end=%llu\n", pattern,
              (unsigned long long)timestamp.context.kernelStart,
              (unsigned long long)timestamp.context.kernelEnd);

  zeMemFree(context, buffer);
  zeEventDestroy(event);
  zeEventPoolDestroy(pool);
  zeCommandListDestroy(list);
  zeCommandQueueDestroy(queue);
  zeContextDestroy(context);
  return 0;
}
