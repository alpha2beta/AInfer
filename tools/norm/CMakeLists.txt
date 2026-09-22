# B60-era SYCL benchmark; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(norm_t35 norm.cpp)
target_compile_options(norm_t35 PRIVATE -fsycl)
target_link_options(norm_t35 PRIVATE -fsycl)

enable_testing()
add_test(NAME norm_t35
  COMMAND norm_t35 ${CMAKE_CURRENT_BINARY_DIR}/report_t35.json)
set_tests_properties(norm_t35 PROPERTIES TIMEOUT 900)
endif()
