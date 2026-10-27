# CanaryMiner (Pearl / PRL only) - CUDA native build.
#
#   cmake --preset production-sm120 -DCMAKE_CUDA_COMPILER=/path/to/nvcc ;  cmake --build --preset production-sm120
#   cmake --preset tests ;  cmake --build --preset tests ;  ctest --preset tests          (CPU-only host tests)
#
# The profile selects the architecture and the compile definitions that the qualified production natives were built
# with (see cmake/CanaryProfile.cmake). Whole-program compilation is deliberate: separable compilation runs the device
# linker, which resolves to SASS and drops PTX.
#
# Needs CMake 3.31+ (architecture-specific "a" targets) and a CUDA 13.1 toolkit. nvcc needs MSVC as its host compiler on
# Windows; a Windows build has not been attempted in Gate 1a.

cmake_minimum_required(VERSION 3.28)
include(cmake/CanaryProfile.cmake)
if(CANARY_PROFILE STREQUAL "production-sm120" AND CMAKE_VERSION VERSION_LESS 3.31)
  message(FATAL_ERROR "production-sm120 needs CMake 3.31+ for the architecture-specific 120a target")
endif()

option(CANARY_FEE_TEST_POLICY "Build the explicitly test-only accelerated donation schedule" OFF)

if(CANARY_PROFILE STREQUAL "tests")
  # Host-side tests only: CPU, no nvcc, no GPU, no network. Built with warnings; format warnings are errors.
  project(canaryminer-tests LANGUAGES CXX)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  enable_testing()
  add_subdirectory(tests)
  return()
endif()

project(canaryminer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# run.cpp is pushed through nvcc (-x cu) like the miner translation unit that calls into it.
set(CANARY_SOURCES
  src/core/miner.cu
  src/core/registry.cu
  src/algos/pearl-pow/algo.cu
  src/core/run.cpp
)
set_source_files_properties(src/core/run.cpp PROPERTIES LANGUAGE CUDA)

add_executable(canaryminer ${CANARY_SOURCES})
set_target_properties(canaryminer PROPERTIES OUTPUT_NAME canary-miner CUDA_SEPARABLE_COMPILATION OFF)

# Stable, distinct per-unit seeds for nvcc-generated identifiers (reproducible builds).
foreach(unit IN LISTS CANARY_SOURCES)
  string(SHA256 unit_seed "canary:${unit}")
  set_property(SOURCE "${unit}" APPEND PROPERTY COMPILE_OPTIONS "--frandom-seed=${unit_seed}")
endforeach()

target_compile_definitions(canaryminer PRIVATE ${CANARY_DEFINITIONS})
if(CANARY_FEE_TEST_POLICY)
  target_compile_definitions(canaryminer PRIVATE CANARY_FEE_TEST_POLICY=1)
endif()
# Host code is compiled with -Wall -Wformat and format problems are errors (a printf format bug, e.g. a bare '%' in a
# string passed as the format, must fail the build). Device code is unaffected by host flags.
target_compile_options(canaryminer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
# MSVC spelling of the same host check: C4473/C4474/C4477 (printf-style format mismatches) are errors.
if(MSVC)
  target_compile_options(canaryminer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,/we4473,/we4474,/we4477,/utf-8>)
else()
  target_compile_options(canaryminer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wformat,-Werror=format>)
endif()
target_include_directories(canaryminer PRIVATE ${CANARY_INCLUDE_DIRS})
if(NOT CANARY_PROFILE STREQUAL "production-sm120")
  message(STATUS "CanaryMiner profile ${CANARY_PROFILE} (CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}); internal build, not a release")
endif()

# NVML is loaded at runtime (dlopen/LoadLibrary); only the loader needs linking on Unix.
if(UNIX)
  target_link_libraries(canaryminer PRIVATE dl)
endif()
if(WIN32)
  target_link_libraries(canaryminer PRIVATE ws2_32)
endif()

install(TARGETS canaryminer RUNTIME DESTINATION bin)
