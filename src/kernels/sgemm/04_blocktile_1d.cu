// Stage 4 - 1D block tiling: one thread computes a column of eight outputs.
//
// Stage 3 is bound by shared-memory bandwidth, not DRAM: two SMEM loads per
// FMA. The cure is the same trick applied one level down the hierarchy. Instead
// of caching in shared memory, we cache in *registers*.
//
// Each thread now owns TM = 8 accumulators covering eight rows of one column of
// C. For each step of the dot product it loads a single element of B into a
// register and reuses it against eight elements of A, so the load-to-math ratio
// falls from 2:1 to roughly 1.1:1. The block tile widens to 64x64 with a
// K-slice of 8, which also cuts the number of __syncthreads() barriers per unit
// of work.
//
// From this stage onward the kernels assume M, N and K are multiples of the tile
// dimensions. Boundary handling costs a predicate on every load and would blur
// the measurement; production code would add a separate ragged-edge path, and
// the benchmark declares the alignment requirement in its results.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int kThreads = (BM * BN) / TM;  // 512

__global__ __launch_bounds__(kThreads) void blocktile_1d_kernel(
    int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float beta, float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int c_row = blockIdx.y;
  const int c_col = blockIdx.x;

  // Position of this thread's output column within the block tile.
  const int thread_col = threadIdx.x % BN;
  const int thread_row = threadIdx.x / BN;

  // Loading indices: the A tile is 64x8 and the B tile is 8x64, both 512
  // elements, so with 512 threads each thread loads exactly one of each.
  const int inner_row_a = threadIdx.x / BK;
  const int inner_col_a = threadIdx.x % BK;
  const int inner_row_b = threadIdx.x / BN;
  const int inner_col_b = threadIdx.x % BN;

  // Advance the pointers to this block's tile once, then walk them along K.
  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;

  float acc[TM] = {0.0f};

  for (int k0 = 0; k0 < K; k0 += BK) {
    As[inner_row_a * BK + inner_col_a] = A[inner_row_a * K + inner_col_a];
    Bs[inner_row_b * BN + inner_col_b] = B[inner_row_b * N + inner_col_b];
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      // One SMEM load of B, reused against TM elements of A.
      const float b_val = Bs[dot * BN + thread_col];
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        acc[m] += As[(thread_row * TM + m) * BK + dot] * b_val;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    float* dst = &C[(thread_row * TM + m) * N + thread_col];
    *dst = alpha * acc[m] + beta * (*dst);
  }
}

}  // namespace

void launch_blocktile_1d(int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, float* C) {
  const dim3 block(kThreads);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  blocktile_1d_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_blocktile_1d(const DeviceInfo& dev) {
  return inspect_kernel(blocktile_1d_kernel, kThreads, 0, dev);
}

}  // namespace ws::sgemm
