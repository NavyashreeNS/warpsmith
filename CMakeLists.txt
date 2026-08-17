cmake_minimum_required(VERSION 3.24)

# This block must precede project(). CUDA toolkits refuse to compile against an
# MSVC newer than the one they shipped against, and the check fires during CMake's
# own compiler probe inside project() - long before any target flags apply. Adding
# the override afterwards leaves configuration failing on a machine that has, say,
# Visual Studio 2026 and CUDA 12.6, which is an ordinary combination.
#
# Set -DWS_ALLOW_UNSUPPORTED_COMPILER=OFF to restore the strict check.
option(WS_ALLOW_UNSUPPORTED_COMPILER
       "Pass -allow-unsupported-compiler to nvcc when MSVC is newer than the CUDA toolkit" ON)
if(CMAKE_HOST_WIN32 AND WS_ALLOW_UNSUPPORTED_COMPILER)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

# Also before project(): project() itself picks a default architecture (sm_52 on
# this toolkit) while probing the compiler, which means a `NOT DEFINED` guard
# placed after it never fires. Compiling for sm_52 silently disables the tensor
# cores and produces a much slower binary, so the default is established here
# where it still takes effect. An explicit -DCMAKE_CUDA_ARCHITECTURES=... on the
# command line arrives as a cache entry and wins.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
endif()

project(warpsmith
        VERSION 1.0.0
        DESCRIPTION "A CUDA kernel optimization lab: hand-written kernels benchmarked against cuBLAS"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Standards and architecture
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "warpsmith: building for CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(WS_LINEINFO "Embed line information for Nsight Compute source correlation" ON)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Shared kernel library
# ---------------------------------------------------------------------------
add_library(warpsmith_kernels STATIC
  src/kernels/sgemm/00_cublas.cu
  src/kernels/sgemm/01_naive.cu
  src/kernels/sgemm/02_coalesced.cu
  src/kernels/sgemm/03_smem_tiled.cu
  src/kernels/sgemm/04_blocktile_1d.cu
  src/kernels/sgemm/05_blocktile_2d.cu
  src/kernels/sgemm/06_vectorized.cu
  src/kernels/sgemm/07_warptiled.cu
  src/kernels/sgemm/08_wmma_tf32.cu
  src/kernels/sgemm/registry.cu
  src/kernels/memory/bandwidth.cu
  src/kernels/reduce/reduce.cu
  src/kernels/transpose/transpose.cu
  src/kernels/softmax/softmax.cu
  src/kernels/norm/rmsnorm.cu
  src/kernels/attention/attention.cu
)

target_include_directories(warpsmith_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(warpsmith_kernels PUBLIC CUDA::cudart CUDA::cublas)

# Relocatable device code is off on purpose: every kernel is self-contained, and
# whole-program compilation lets nvcc make better register allocation decisions.
set_target_properties(warpsmith_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

function(warpsmith_cuda_flags target)
  target_compile_options(${target} PRIVATE
    # No --use_fast_math: kernels that want the fast intrinsics ask for them by
    # name (__expf, rsqrtf), which keeps the precision tradeoff visible in the
    # source instead of hidden in a build flag.
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  if(WS_LINEINFO)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
endfunction()

warpsmith_cuda_flags(warpsmith_kernels)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(warpsmith_bench
  src/bench/main.cu
  src/bench/console.cu
  src/bench/bench_sgemm.cu)
target_link_libraries(warpsmith_bench PRIVATE warpsmith_kernels)
warpsmith_cuda_flags(warpsmith_bench)

add_executable(warpsmith_test src/tests/test_sgemm.cu)
target_link_libraries(warpsmith_test PRIVATE warpsmith_kernels)
warpsmith_cuda_flags(warpsmith_test)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(CTest)
add_test(NAME sgemm_correctness COMMAND warpsmith_test)
# A fast end-to-end run of every suite, used as a smoke test: it exits non-zero
# if any kernel produces a wrong answer.
add_test(NAME suite_smoke COMMAND warpsmith_bench --quick --iters 5 --warmup 2)
