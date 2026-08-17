// warpsmith - SGEMM variant registry.
//
// Single-precision GEMM is the reference problem of GPU performance
// engineering: it is arithmetically simple, it is the dominant cost in dense
// neural networks, and NVIDIA ships a hand-tuned closed-source implementation
// (cuBLAS) that sets an honest upper bound. Each variant below adds exactly one
// optimization to the previous one, so the measured jump between two adjacent
// stages isolates the value of that single technique.
//
// All variants compute the row-major operation
//
//     C = alpha * A @ B + beta * C,      A: M x K,  B: K x N,  C: M x N
#pragma once

#include <vector>

#include "warpsmith/bench.cuh"
#include "warpsmith/device.cuh"

namespace ws::sgemm {

using Launcher = void (*)(int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C);
using Inspector = KernelAttrs (*)(const DeviceInfo&);

struct Variant {
  const char* id;         // Stable slug used in JSON and chart labels.
  const char* name;       // Display name.
  const char* technique;  // The one optimization this stage introduces.
  int stage;              // Ordering index; stage 0 is the library baseline.
  int align;              // Required divisor of M, N and K (1 = any size).
  bool is_reference;      // True for the cuBLAS baselines.
  bool tensor_core;       // True if the variant uses tensor cores (TF32 math).
  Launcher launch;
  Inspector inspect;      // Static resource usage; null for library kernels.
};

// Declared launchers, one per stage.
void launch_cublas_fp32(int, int, int, float, const float*, const float*, float, float*);
void launch_cublas_tf32(int, int, int, float, const float*, const float*, float, float*);
void launch_naive(int, int, int, float, const float*, const float*, float, float*);
void launch_coalesced(int, int, int, float, const float*, const float*, float, float*);
void launch_smem_tiled(int, int, int, float, const float*, const float*, float, float*);
void launch_blocktile_1d(int, int, int, float, const float*, const float*, float, float*);
void launch_blocktile_2d(int, int, int, float, const float*, const float*, float, float*);
void launch_vectorized(int, int, int, float, const float*, const float*, float, float*);
void launch_warptiled(int, int, int, float, const float*, const float*, float, float*);
void launch_wmma_tf32(int, int, int, float, const float*, const float*, float, float*);

KernelAttrs inspect_naive(const DeviceInfo&);
KernelAttrs inspect_coalesced(const DeviceInfo&);
KernelAttrs inspect_smem_tiled(const DeviceInfo&);
KernelAttrs inspect_blocktile_1d(const DeviceInfo&);
KernelAttrs inspect_blocktile_2d(const DeviceInfo&);
KernelAttrs inspect_vectorized(const DeviceInfo&);
KernelAttrs inspect_warptiled(const DeviceInfo&);
KernelAttrs inspect_wmma_tf32(const DeviceInfo&);

// cuBLAS handle lifetime, owned by the benchmark driver.
void cublas_init();
void cublas_shutdown();

const std::vector<Variant>& variants();

// Autotuner: sweeps the tile-configuration space of the vectorized kernel.
struct TuneConfig {
  int bm, bn, bk, tm, tn, threads;
  std::size_t smem_bytes;
  bool valid;
};
const std::vector<TuneConfig>& tune_configs();
void launch_tuned(int cfg_index, int M, int N, int K, float alpha, const float* A, const float* B,
                  float beta, float* C);

}  // namespace ws::sgemm
