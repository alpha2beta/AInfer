cmake_minimum_required(VERSION 3.16)
project(ainfer_esimd_check CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Built with icpx (-fsycl). Reuses the vendored L0 headers from l0probe
# for the raw-timestamp check; links the versioned loader (no dev symlink).
# Built with icpx (-fsycl); skipped when AINFER_ENABLE_SYCL=OFF (e.g. 258v/clang++).
if(AINFER_ENABLE_SYCL)
add_executable(esimd_smoke smoke.cpp)
target_include_directories(esimd_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../l0probe/include)
target_link_libraries(esimd_smoke PRIVATE
  /usr/lib/x86_64-linux-gnu/libze_loader.so.1)
target_compile_options(esimd_smoke PRIVATE -fsycl)
target_link_options(esimd_smoke PRIVATE -fsycl)

enable_testing()
add_test(NAME esimd_smoke
  COMMAND esimd_smoke ${CMAKE_CURRENT_BINARY_DIR}/report_esimd.json)
endif()
