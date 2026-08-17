// Stage 3 - shared memory tiling (cache blocking).
//
// Stages 1 and 2 read 2K floats from DRAM per output element. The arithmetic
// intensity is therefore ~0.25 FLOP/byte, an order of magnitude below this
// GPU's roofline ridge point, so the kernel is hopelessly bandwidth-bound no
// matter how well it coalesces.
//
// The fix is reuse. A block of 32x32 threads cooperatively stages a 32x32 tile
// of A and of B into shared memory; every element loaded is then consumed by 32
// different threads. DRAM traffic drops by 32x and intensity rises to ~8
// FLOP/byte, which is where compute starts to matter.
//
// This stage is still limited by the fact that each thread holds only *one*
// accumulator, so it must issue two shared-memory loads for every FMA. Shared
// memory is fast, but not free: that 2:1 load-to-math ratio is what stage 4
// attacks.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int kTile = 32;

__global__ void smem_tiled_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                  const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[kTile * kTile];
  __shared__ float Bs[kTile * kTile];

  const int tx = threadIdx.x % kTile;  // Column within the tile.
  const int ty = threadIdx.x / kTile;  // Row within the tile.
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    // Cooperative load. The bounds checks let this kernel handle any shape,
    // which is why it is the last stage in the progression that does so.
    As[ty * kTile + tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
    Bs[ty * kTile + tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      // As is read down a column of the tile (broadcast within the warp) and Bs
      // across a row (conflict-free: 32 lanes hit 32 distinct banks).
      acc += As[ty * kTile + k] * Bs[k * kTile + tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

}  // namespace

void launch_smem_tiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                       float* C) {
  const dim3 block(kTile * kTile);
  const dim3 grid(ceil_div(N, kTile), ceil_div(M, kTile));
  smem_tiled_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_smem_tiled(const DeviceInfo& dev) {
  return inspect_kernel(smem_tiled_kernel, kTile * kTile, 0, dev);
}

}  // namespace ws::sgemm
