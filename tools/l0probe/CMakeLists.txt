cmake_minimum_required(VERSION 3.16)
project(ainfer_l0probe CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AINFER_ZE_LOADER "/usr/lib/x86_64-linux-gnu/libze_loader.so.1"
    CACHE FILEPATH "Level Zero loader library")

# Vendored Level Zero core headers (v1.28.2, matches loader 1.28.2).
# System provides only Intel extension headers + versioned libze_loader
# (no .so dev symlink without sudo), so link the versioned file directly.
add_executable(l0probe probe.cpp)
target_include_directories(l0probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(l0probe PRIVATE ${AINFER_ZE_LOADER})

add_executable(l0_timestamp_smoke timestamp_smoke.cpp)
target_include_directories(l0_timestamp_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(l0_timestamp_smoke PRIVATE ${AINFER_ZE_LOADER})

enable_testing()
add_test(NAME l0probe
  COMMAND l0probe ${CMAKE_CURRENT_BINARY_DIR}/report_l0probe.json)
set_tests_properties(l0probe PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../toolchain/sysroot/usr/lib")
add_test(NAME l0_timestamp_smoke COMMAND l0_timestamp_smoke)
set_tests_properties(l0_timestamp_smoke PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../toolchain/sysroot/usr/lib")
