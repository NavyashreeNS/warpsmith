// Stage 6 kernel, expressed as a template over its tile geometry.
//
// Two ideas are added on top of stage 5.
//
// 1. Vectorized memory access. Every global load and store, and every
//    shared-memory read in the inner loop, moves 16 bytes at a time via float4.
//    One LDG.E.128 replaces four LDG.E.32: a quarter of the instructions, a
//    quarter of the address arithmetic, and full utilisation of the 128-bit
//    load-store path.
//
// 2. A transposed shared-memory layout for A. Stage 5 read A down a column of
//    the tile (stride BK), which cannot be vectorized and invites bank
//    conflicts. Here the A tile is *stored* transposed at load time - As[k][m]
//    instead of As[m][k] - so the inner loop reads it contiguously and the
//    float4 load is legal. The transpose costs nothing extra: it happens while
//    the data is in flight to shared memory.
//
// The geometry is a template parameter rather than a constant because the
// optimum depends on the GPU: the autotuner instantiates a dozen configurations
// and measures them all. Every constraint that makes a configuration legal is
// enforced by static_assert, so an illegal tile shape is a compile error rather
// than a silently wrong result.
#pragma once

#include "warpsmith/common.cuh"

namespace ws::sgemm {

// Compile-time legality of a tile configuration, usable in host code so the
// autotuner can filter candidates before instantiating them.
__host__ __device__ constexpr bool tile_config_valid(int BM, int BN, int BK, int TM, int TN) {
  if (BM <= 0 || BN <= 0 || BK <= 0 || TM <= 0 || TN <= 0) return false;
  if (BM % TM != 0 || BN % TN != 0) return false;
  if (BK % 4 != 0 || BN % 4 != 0) return false;
  if (TM % 4 != 0 || TN % 4 != 0) return false;
  const int threads = (BM * BN) / (TM * TN);
  if (threads % 32 != 0 || threads > 1024) return false;
  // Each thread must move a whole number of float4 vectors per tile pass.
  if ((BM * BK) % (threads * 4) != 0) return false;
  if ((BK * BN) % (threads * 4) != 0) return false;
  if (threads % (BK / 4) != 0 || threads % (BN / 4) != 0) return false;
  const int stride_a = threads / (BK / 4);
  const int stride_b = threads / (BN / 4);
  if (stride_a == 0 || stride_b == 0) return false;
  if (BM % stride_a != 0 || BK % stride_b != 0) return false;
  // Shared memory budget: two tiles of floats, double-checked against the
  // 48 KB per-block default limit.
  if ((BM * BK + BK * BN) * 4 > 48 * 1024) return false;
  return true;
}

__host__ __device__ constexpr int tile_threads(int BM, int BN, int TM, int TN) {
  return (BM * BN) / (TM * TN);
}
__host__ __device__ constexpr std::size_t tile_smem(int BM, int BN, int BK) {
  return static_cast<std::size_t>(BM * BK + BK * BN) * sizeof(float);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(tile_threads(BM, BN, TM, TN)) void vectorized_kernel(
    int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float beta, float* __restrict__ C) {
  static_assert(tile_config_valid(BM, BN, BK, TM, TN), "illegal tile configuration");

  constexpr int kThreads = tile_threads(BM, BN, TM, TN);
  constexpr int kStrideA = kThreads / (BK / 4);
  constexpr int kStrideB = kThreads / (BN / 4);

  // As is stored transposed: [BK][BM]. Bs keeps its natural [BK][BN] layout.
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int tid = threadIdx.x;
  const int thread_col = tid % (BN / TN);
  const int thread_row = tid / (BN / TN);

  const int inner_row_a = tid / (BK / 4);
  const int inner_col_a = tid % (BK / 4);
  const int inner_row_b = tid / (BN / 4);
  const int inner_col_b = tid % (BN / 4);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[TM * TN] = {0.0f};
  float reg_m[TM];
  float reg_n[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // --- Stage the A tile, transposing on the way in. ---
#pragma unroll
    for (int off = 0; off < BM; off += kStrideA) {
      const float4 v =
          *reinterpret_cast<const float4*>(&A[(inner_row_a + off) * K + inner_col_a * 4]);
      const int m = inner_row_a + off;
      const int k = inner_col_a * 4;
      As[(k + 0) * BM + m] = v.x;
      As[(k + 1) * BM + m] = v.y;
      As[(k + 2) * BM + m] = v.z;
      As[(k + 3) * BM + m] = v.w;
    }
    // --- Stage the B tile, layout preserved, 16 bytes per thread per pass. ---
#pragma unroll
    for (int off = 0; off < BK; off += kStrideB) {
      *reinterpret_cast<float4*>(&Bs[(inner_row_b + off) * BN + inner_col_b * 4]) =
          *reinterpret_cast<const float4*>(&B[(inner_row_b + off) * N + inner_col_b * 4]);
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    // --- Outer product accumulation, entirely from registers. ---
#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        *reinterpret_cast<float4*>(&reg_m[i]) =
            *reinterpret_cast<const float4*>(&As[dot * BM + thread_row * TM + i]);
      }
#pragma unroll
      for (int i = 0; i < TN; i += 4) {
        *reinterpret_cast<float4*>(&reg_n[i]) =
            *reinterpret_cast<const float4*>(&Bs[dot * BN + thread_col * TN + i]);
      }
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

  // --- Epilogue: read-modify-write C in 16-byte chunks. ---
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float* dst = &C[(thread_row * TM + m) * N + thread_col * TN + n];
      float4 out = *reinterpret_cast<float4*>(dst);
      out.x = alpha * acc[m * TN + n + 0] + beta * out.x;
      out.y = alpha * acc[m * TN + n + 1] + beta * out.y;
      out.z = alpha * acc[m * TN + n + 2] + beta * out.z;
      out.w = alpha * acc[m * TN + n + 3] + beta * out.w;
      *reinterpret_cast<float4*>(dst) = out;
    }
  }
}

}  // namespace ws::sgemm
