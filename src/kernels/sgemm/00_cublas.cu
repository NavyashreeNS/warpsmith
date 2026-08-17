// Stage 0 - the baseline we are chasing.
//
// cuBLAS is NVIDIA's hand-tuned, architecture-specific BLAS. It is closed
// source and it is fast, which makes it the only credible yardstick: a kernel
// that reaches 90% of cuBLAS is fast in an absolute sense, whereas a kernel
// that is "10x faster than my first attempt" may still be terrible.
//
// Two baselines are registered. The first uses classic FP32 FMA pipelines. The
// second permits cuBLAS to use TF32 tensor cores, which on Ampere doubles the
// theoretical ceiling in exchange for a 10-bit mantissa - the fair comparison
// for the WMMA kernel in stage 8.

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

cublasHandle_t g_handle = nullptr;

void cublas_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "[warpsmith] cuBLAS failure in %s: status %d\n", what,
                 static_cast<int>(s));
    std::exit(EXIT_FAILURE);
  }
}

// cuBLAS is column-major. Rather than transpose the data, we exploit the
// identity (A @ B)^T = B^T @ A^T: a row-major M x N result is bit-identical to
// the column-major N x M result of swapping the operands. So we hand cuBLAS
// (B, A) with the dimensions swapped and no explicit transposes.
void gemm(cublasMath_t math_mode, int M, int N, int K, float alpha, const float* A, const float* B,
          float beta, float* C) {
  cublas_check(cublasSetMathMode(g_handle, math_mode), "cublasSetMathMode");
  cublas_check(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta,
                           C, N),
               "cublasSgemm");
}

}  // namespace

void cublas_init() {
  if (g_handle == nullptr) cublas_check(cublasCreate(&g_handle), "cublasCreate");
}

void cublas_shutdown() {
  if (g_handle != nullptr) {
    cublas_check(cublasDestroy(g_handle), "cublasDestroy");
    g_handle = nullptr;
  }
}

void launch_cublas_fp32(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                        float* C) {
  gemm(CUBLAS_DEFAULT_MATH, M, N, K, alpha, A, B, beta, C);
}

void launch_cublas_tf32(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                        float* C) {
  gemm(CUBLAS_TF32_TENSOR_OP_MATH, M, N, K, alpha, A, B, beta, C);
}

}  // namespace ws::sgemm
