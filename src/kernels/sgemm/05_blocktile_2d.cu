// Stage 5 - 2D block tiling: an 8x8 register tile per thread.
//
// Stage 4 amortized one B load over eight A loads. Going 2D amortizes in both
// directions at once: each thread loads TM = 8 elements of A and TN = 8 elements
// of B into registers, then performs 64 FMAs against them. Sixteen shared-memory
// loads now feed sixty-four multiply-adds - a 4:1 math-to-load ratio, four times
// better than stage 4 and eight times better than stage 3.
//
// This is the outer-product formulation of GEMM, and it is the structural core
// of every fast dense kernel ever written, cuBLAS and CUTLASS included. The
// block tile grows to 128x128 so that 256 threads still cover it, which means a
// single block now produces 16384 outputs and reads each input tile exactly
// once from DRAM.
//
// Register pressure becomes the binding constraint here: 64 accumulators plus
// 16 staging registers plus addressing means occupancy drops sharply. That
// trade - fewer resident warps, far more work per warp - is a deliberate and
// usually winning bet on an SM with 128 FP32 lanes.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int kThreads = (BM * BN) / (TM * TN);  // 256

__global__ __launch_bounds__(kThreads) void blocktile_2d_kernel(
    int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float beta, float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int c_row = blockIdx.y;
  const int c_col = blockIdx.x;

  // This thread's 8x8 output tile, indexed in units of tiles.
  const int thread_col = threadIdx.x % (BN / TN);
  const int thread_row = threadIdx.x / (BN / TN);

  // The A tile is 128x8 = 1024 elements and there are 256 threads, so each
  // thread performs four loads, striding by `stride_a` rows between them. The
  // stride keeps consecutive threads on consecutive addresses, preserving
  // coalescing on every one of the four loads.
  const int inner_row_a = threadIdx.x / BK;
  const int inner_col_a = threadIdx.x % BK;
  constexpr int stride_a = kThreads / BK;  // 32 rows per pass
  const int inner_row_b = threadIdx.x / BN;
  const int inner_col_b = threadIdx.x % BN;
  constexpr int stride_b = kThreads / BN;  // 2 rows per pass

  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;

  float acc[TM * TN] = {0.0f};
  float reg_m[TM];
  float reg_n[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += stride_a) {
      As[(inner_row_a + off) * BK + inner_col_a] = A[(inner_row_a + off) * K + inner_col_a];
    }
#pragma unroll
    for (int off = 0; off < BK; off += stride_b) {
      Bs[(inner_row_b + off) * BN + inner_col_b] = B[(inner_row_b + off) * N + inner_col_b];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      // Stage the operands into registers once...
#pragma unroll
      for (int m = 0; m < TM; ++m) reg_m[m] = As[(thread_row * TM + m) * BK + dot];
#pragma unroll
      for (int n = 0; n < TN; ++n) reg_n[n] = Bs[dot * BN + thread_col * TN + n];
      // ...then take their full outer product entirely out of registers.
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          acc[m * TN + n] += reg_m[m] * reg_n[n];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      float* dst = &C[(thread_row * TM + m) * N + thread_col * TN + n];
      *dst = alpha * acc[m * TN + n] + beta * (*dst);
    }
  }
}

}  // namespace

void launch_blocktile_2d(int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, float* C) {
  const dim3 block(kThreads);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  blocktile_2d_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_blocktile_2d(const DeviceInfo& dev) {
  return inspect_kernel(blocktile_2d_kernel, kThreads, 0, dev);
}

}  // namespace ws::sgemm
