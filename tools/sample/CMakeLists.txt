# B60-era SYCL benchmark; skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(sample_t39 sample.cpp)
target_compile_options(sample_t39 PRIVATE -fsycl)
target_link_options(sample_t39 PRIVATE -fsycl)

enable_testing()
add_test(NAME sample_t39
  COMMAND sample_t39 ${CMAKE_CURRENT_BINARY_DIR}/report_t39.json)
set_tests_properties(sample_t39 PROPERTIES TIMEOUT 900)
endif()

# T7.1 host sampler (temp/top-k/top-p) + statistical validation (host-only).
add_executable(validate_t71 validate_t71.cpp)
enable_testing()
add_test(NAME validate_t71
  COMMAND validate_t71 ${CMAKE_CURRENT_BINARY_DIR}/report_t71.json)
set_tests_properties(validate_t71 PROPERTIES TIMEOUT 900)
