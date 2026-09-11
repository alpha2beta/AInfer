add_executable(kvfit kvfit.cpp)
target_include_directories(kvfit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(kvfit PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)

enable_testing()
add_test(NAME kvfit
  COMMAND kvfit ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                 ${CMAKE_CURRENT_BINARY_DIR}/report_kvfit.json)
set_tests_properties(kvfit PROPERTIES TIMEOUT 600)
