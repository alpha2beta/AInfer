add_executable(blkexec blkexec.cpp)
target_include_directories(blkexec PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(blkexec PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(blkexec PRIVATE -fsycl)
target_link_options(blkexec PRIVATE -fsycl)

enable_testing()
add_test(NAME blkexec
  COMMAND blkexec ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                  ${CMAKE_CURRENT_BINARY_DIR}/report_blkexec.json)
set_tests_properties(blkexec PROPERTIES TIMEOUT 3600)
