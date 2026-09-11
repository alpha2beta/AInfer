add_executable(l0load loader.cpp)
target_include_directories(l0load PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(l0load PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)

enable_testing()
add_test(NAME l0load
  COMMAND l0load ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                 ${CMAKE_CURRENT_BINARY_DIR}/report_l0load.json)
# Full 15 GB upload + 15 GB readback; generous timeout for USB-mounted source.
set_tests_properties(l0load PROPERTIES TIMEOUT 3600)
