# B60-era SYCL benchmark; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(bench_t31 bench.cpp)
target_include_directories(bench_t31 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(bench_t31 PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(bench_t31 PRIVATE -fsycl)
target_link_options(bench_t31 PRIVATE -fsycl)

enable_testing()
add_test(NAME bench_t31
  COMMAND bench_t31 ${CMAKE_CURRENT_BINARY_DIR}/report_t31.json)
set_tests_properties(bench_t31 PROPERTIES TIMEOUT 1800)
endif()
