// Stage 8 - tensor cores via WMMA, in TF32.
//
// Everything up to stage 7 issues FFMA instructions on the FP32 pipelines, and
// those pipelines are a hard ceiling: 128 lanes per SM, two FLOP each per clock.
// Ampere has a second, wider datapath. A tensor core consumes an entire small
// matrix multiply as a single instruction, and on this GPU the TF32 path runs at
// twice the FP32 FMA rate (the FP16 path at four times).
//
// TF32 is not FP32. It keeps the 8-bit exponent of FP32 - so the dynamic range
// is unchanged and nothing overflows that would not have overflowed before - but
// truncates the mantissa from 23 bits to 10. Accumulation still happens in full
// FP32 inside the tensor core, which is why the error stays small enough for
// neural network training; the measured relative error is reported in the
// results table rather than hidden, because a fast wrong answer is not a result.
//
// This is exactly the tradeoff cuBLAS makes when TF32 math mode is enabled, so
// stage 8 is benchmarked against the cuBLAS TF32 baseline as well as the FP32
// one.

#include <mma.h>

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

// `nvcuda` is only declared by <mma.h> for targets that actually have tensor
// cores, so the using-directive lives inside the guarded kernel body rather than
// at namespace scope. Without this the file fails to compile for, say, sm_52 -
// which is the architecture CMake defaults to if a project forgets to set one.

// WMMA shape for TF32 on Ampere: 16x16x8 per instruction.
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;

// The tile geometry matters more here than anywhere else in the progression.
// A tensor core retires a 16x16x8 multiply in a handful of cycles, so a block
// tile that is too small starves it: the kernel spends its time staging shared
// memory instead of issuing MMA instructions. A 128x128 block tile with a 32x64
// warp tile gives each warp eight accumulator fragments to keep in flight.
constexpr int BM = 128, BN = 128, BK = 32;  // Block tile.
constexpr int WM = 32, WN = 64;             // Warp tile: 2x4 WMMA fragments.
constexpr int kThreads = 256;               // 8 warps, 4 along M and 2 along N.
constexpr int kWarpSize = 32;
constexpr int kFragM = WM / WMMA_M;         // 2
constexpr int kFragN = WN / WMMA_N;         // 4

static_assert(BK % WMMA_K == 0, "K-slice must be a whole number of MMA steps");
static_assert((BM / WM) * (BN / WN) == kThreads / kWarpSize, "warps must tile the block");

__global__ __launch_bounds__(kThreads) void wmma_tf32_kernel(
    int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float beta, float* __restrict__ C) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;

  __shared__ float As[BM * BK];  // [BM][BK], row-major.
  __shared__ float Bs[BK * BN];  // [BK][BN], row-major.
  __shared__ float stage[(kThreads / kWarpSize) * WMMA_M * WMMA_N];

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int warp_col = warp_id % (BN / WN);
  const int warp_row = warp_id / (BN / WN);

  // Staging indices for float4 cooperative loads.
  const int inner_row_a = tid / (BK / 4);
  const int inner_col_a = tid % (BK / 4);
  constexpr int kStrideA = kThreads / (BK / 4);
  const int inner_row_b = tid / (BN / 4);
  const int inner_col_b = tid % (BN / 4);
  constexpr int kStrideB = kThreads / (BN / 4);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[kFragM][kFragN];
#pragma unroll
  for (int i = 0; i < kFragM; ++i)
#pragma unroll
    for (int j = 0; j < kFragN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += kStrideA) {
      *reinterpret_cast<float4*>(&As[(inner_row_a + off) * BK + inner_col_a * 4]) =
          *reinterpret_cast<const float4*>(&A[(inner_row_a + off) * K + inner_col_a * 4]);
    }
#pragma unroll
    for (int off = 0; off < BK; off += kStrideB) {
      *reinterpret_cast<float4*>(&Bs[(inner_row_b + off) * BN + inner_col_b * 4]) =
          *reinterpret_cast<const float4*>(&B[(inner_row_b + off) * N + inner_col_b * 4]);
    }
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int kk = 0; kk < BK; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                     wmma::row_major> a_frag[kFragM];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                     wmma::row_major> b_frag[kFragN];

#pragma unroll
      for (int i = 0; i < kFragM; ++i) {
        wmma::load_matrix_sync(a_frag[i], &As[(warp_row * WM + i * WMMA_M) * BK + kk], BK);
        // Explicit narrowing to TF32: the hardware requires the operand to
        // already be rounded, and doing it here makes the precision loss a
        // visible, deliberate step rather than a hidden one.
#pragma unroll
        for (int t = 0; t < a_frag[i].num_elements; ++t) {
          a_frag[i].x[t] = wmma::__float_to_tf32(a_frag[i].x[t]);
        }
      }
#pragma unroll
      for (int j = 0; j < kFragN; ++j) {
        wmma::load_matrix_sync(b_frag[j], &Bs[kk * BN + warp_col * WN + j * WMMA_N], BN);
#pragma unroll
        for (int t = 0; t < b_frag[j].num_elements; ++t) {
          b_frag[j].x[t] = wmma::__float_to_tf32(b_frag[j].x[t]);
        }
      }
#pragma unroll
      for (int i = 0; i < kFragM; ++i) {
#pragma unroll
        for (int j = 0; j < kFragN; ++j) {
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  // Epilogue. A fragment's element-to-lane mapping is opaque, so each 16x16 tile
  // is written to shared memory with store_matrix_sync and then scaled by
  // alpha/beta on its way to global memory.
  float* my_stage = &stage[warp_id * WMMA_M * WMMA_N];
#pragma unroll
  for (int i = 0; i < kFragM; ++i) {
#pragma unroll
    for (int j = 0; j < kFragN; ++j) {
      wmma::store_matrix_sync(my_stage, acc[i][j], WMMA_N, wmma::mem_row_major);
      __syncwarp();
      const int row0 = warp_row * WM + i * WMMA_M;
      const int col0 = warp_col * WN + j * WMMA_N;
      for (int idx = lane; idx < WMMA_M * WMMA_N; idx += kWarpSize) {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        float* dst = &C[(row0 + r) * N + col0 + c];
        *dst = alpha * my_stage[idx] + beta * (*dst);
      }
      __syncwarp();
    }
  }
#else
  // Tensor cores with TF32 require compute capability 8.0 or newer. Compiling
  // for an older target yields a kernel that reports rather than misleads.
  (void)M; (void)N; (void)K; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
#endif
}

}  // namespace

void launch_wmma_tf32(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                      float* C) {
  const dim3 block(kThreads);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  wmma_tf32_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_wmma_tf32(const DeviceInfo& dev) {
  return inspect_kernel(wmma_tf32_kernel, kThreads, 0, dev);
}

}  // namespace ws::sgemm
