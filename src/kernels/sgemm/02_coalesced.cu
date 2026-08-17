// Stage 2 - global memory coalescing.
//
// The arithmetic is byte-for-byte identical to stage 1. The only change is that
// `threadIdx.x` now selects the *column*, so the 32 lanes of a warp read 32
// consecutive floats of B and write 32 consecutive floats of C. Those requests
// coalesce into four 128-byte transactions instead of thirty-two, and the
// broadcast read of A[row * K + k] is served from the constant path of L1 since
// every lane in the warp wants the same address.
//
// This is the single highest-leverage line of code in the whole progression: no
// shared memory, no tiling, no register blocking - just an index swap.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int kBlock = 32;

__global__ void coalesced_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                 const float* __restrict__ B, float beta, float* __restrict__ C) {
  // threadIdx.x -> column: consecutive lanes are contiguous in B and C.
  const int col = blockIdx.x * kBlock + threadIdx.x;
  const int row = blockIdx.y * kBlock + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

}  // namespace

void launch_coalesced(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                      float* C) {
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(N, kBlock), ceil_div(M, kBlock));
  coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_coalesced(const DeviceInfo& dev) {
  return inspect_kernel(coalesced_kernel, kBlock * kBlock, 0, dev);
}

}  // namespace ws::sgemm
