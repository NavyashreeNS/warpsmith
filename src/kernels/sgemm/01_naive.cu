// Stage 1 - one thread per output element, with a deliberately bad thread map.
//
// This is the textbook first GEMM, and it is slow for a reason worth measuring
// precisely. Note the index assignment: `threadIdx.x` selects the *row*. Within
// a warp, the 32 lanes therefore walk down a column of C and of A, so each lane
// touches a different 128-byte cache line. A single warp instruction becomes 32
// separate memory transactions and the kernel wastes ~97% of every line it
// fetches.
//
// Stage 2 changes nothing but this mapping, which turns the gap between the two
// into a clean measurement of what coalescing alone is worth.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int kBlock = 32;

__global__ void naive_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                             const float* __restrict__ B, float beta, float* __restrict__ C) {
  // threadIdx.x -> row: consecutive lanes are strided by K in A and by N in C.
  const int row = blockIdx.x * kBlock + threadIdx.x;
  const int col = blockIdx.y * kBlock + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

}  // namespace

void launch_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                  float* C) {
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(M, kBlock), ceil_div(N, kBlock));
  naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_naive(const DeviceInfo& dev) {
  return inspect_kernel(naive_kernel, kBlock * kBlock, 0, dev);
}

}  // namespace ws::sgemm
