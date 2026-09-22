# AINFER_ZE_LOADER mirrors tools/l0probe (Ubuntu default; 258v preset points
# at the sysroot loader since CachyOS has no system libze_loader).
set(AINFER_ZE_LOADER "/usr/lib/x86_64-linux-gnu/libze_loader.so.1"
    CACHE FILEPATH "Level Zero loader library")
add_executable(kvfit kvfit.cpp)
target_include_directories(kvfit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(kvfit PRIVATE
  ${AINFER_ZE_LOADER})

enable_testing()
add_test(NAME kvfit
  COMMAND kvfit ${CMAKE_CURRENT_SOURCE_DIR}/../../models/Qwen3.8-27B/qwen3.8-27b-text-int4g128.binfer
                 ${CMAKE_CURRENT_BINARY_DIR}/report_kvfit.json)
set_tests_properties(kvfit PROPERTIES TIMEOUT 600)
