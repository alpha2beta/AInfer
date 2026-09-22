# B60-era SYCL benchmarks; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(blkblock block.cpp)
target_include_directories(blkblock PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(blkblock PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(blkblock PRIVATE -fsycl)
target_link_options(blkblock PRIVATE -fsycl)

enable_testing()
add_test(NAME blkblock
  COMMAND blkblock ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                   ${CMAKE_CURRENT_BINARY_DIR}/report_blkblock.json)
set_tests_properties(blkblock PROPERTIES TIMEOUT 3600)

add_executable(blklinear linear.cpp)
target_include_directories(blklinear PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(blklinear PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(blklinear PRIVATE -fsycl)
target_link_options(blklinear PRIVATE -fsycl)
add_test(NAME blklinear
  COMMAND blklinear ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                    ${CMAKE_CURRENT_BINARY_DIR}/report_blklinear.json)
set_tests_properties(blklinear PROPERTIES TIMEOUT 3600)
endif()
