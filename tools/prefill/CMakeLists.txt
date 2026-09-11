add_executable(prefill_t34 prefill.cpp)
target_compile_options(prefill_t34 PRIVATE -fsycl)
target_link_options(prefill_t34 PRIVATE -fsycl)

add_executable(dpas_roof dpas_roof.cpp)
target_compile_options(dpas_roof PRIVATE -fsycl)
target_link_options(dpas_roof PRIVATE -fsycl)

enable_testing()
add_test(NAME prefill_t34
  COMMAND prefill_t34 ${CMAKE_CURRENT_BINARY_DIR}/report_t34.json)
set_tests_properties(prefill_t34 PROPERTIES TIMEOUT 1800)
