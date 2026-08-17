// Stage 7 - warp tiling: a third level of blocking, between block and thread.
//
// Stage 6 has two levels of the memory hierarchy blocked (DRAM -> SMEM, SMEM ->
// registers) but ignores the fact that the SM does not execute 256 independent
// threads. It executes 8 warps, and a warp is the unit that shares an
// instruction stream, issues one shared-memory request, and occupies one of the
// four processing blocks of the SM.
//
// So we block for the warp as well. The 128x128 block tile is partitioned into
// four 64x64 warp tiles, one per warp, and each warp tile is covered by 32
// threads each owning several small sub-tiles rather than one large one. Two
// things improve:
//
//   * Shared-memory reads are amortized across the warp. The 32 lanes of a warp
//     now read the same narrow slice of As and Bs, so the values land in the
//     register file once and are reused across many more FMAs.
//   * The MMA instruction stream inside a warp becomes a long, dependency-free
//     run of FFMAs on registers that are already live, which is exactly what the
//     scheduler needs to keep all four processing blocks busy at low occupancy.
//
// Each thread accumulates 128 outputs in registers here. That is an aggressive
// register budget - the whole point is to trade occupancy for instruction-level
// parallelism, and the measured registers-per-thread figure in the results table
// shows exactly how far that trade was pushed.

#include "warpsmith/sgemm.cuh"

namespace ws::sgemm {
namespace {

constexpr int kWarpSize = 32;
constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WM = 64, WN = 64;    // Warp tile.
constexpr int WNITER = 4;          // Sub-tiles per warp along N.
constexpr int TM = 8, TN = 4;      // Per-thread sub-tile.
constexpr int kThreads = 128;      // 4 warps.

constexpr int kWarps = kThreads / kWarpSize;
// Sub-tiles per warp along M, derived so the warp tile is exactly covered.
constexpr int WMITER = (WM * WN) / (kWarpSize * TM * TN * WNITER);
constexpr int WSUBM = WM / WMITER;
constexpr int WSUBN = WN / WNITER;

static_assert(WMITER > 0, "warp tile does not divide evenly");
static_assert((BM / WM) * (BN / WN) == kWarps, "warp tiles must cover the block tile");
static_assert((WSUBM / TM) * (WSUBN / TN) == kWarpSize, "thread tiles must cover the warp sub-tile");
static_assert(TN % 4 == 0 && TM % 4 == 0, "vectorized access requires multiples of 4");

__global__ __launch_bounds__(kThreads) void warptiled_kernel(
    int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
    float beta, float* __restrict__ C) {
  __shared__ float As[BK * BM];  // Transposed: [BK][BM].
  __shared__ float Bs[BK * BN];  // Natural:    [BK][BN].

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int warp_col = warp_id % (BN / WN);
  const int warp_row = warp_id / (BN / WN);

  const int lane = tid % kWarpSize;
  const int lane_col = lane % (WSUBN / TN);
  const int lane_row = lane / (WSUBN / TN);

  // Tile-staging indices, identical in spirit to stage 6.
  const int inner_row_a = tid / (BK / 4);
  const int inner_col_a = tid % (BK / 4);
  constexpr int kStrideA = (kThreads * 4) / BK;
  const int inner_row_b = tid / (BN / 4);
  const int inner_col_b = tid % (BN / 4);
  constexpr int kStrideB = kThreads / (BN / 4);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[WMITER * TM * WNITER * TN] = {0.0f};
  float reg_m[WMITER * TM];
  float reg_n[WNITER * TN];
  constexpr int kAccStride = WNITER * TN;

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int off = 0; off + inner_row_a < BM; off += kStrideA) {
      const float4 v =
          *reinterpret_cast<const float4*>(&A[(inner_row_a + off) * K + inner_col_a * 4]);
      const int m = inner_row_a + off;
      const int k = inner_col_a * 4;
      As[(k + 0) * BM + m] = v.x;
      As[(k + 1) * BM + m] = v.y;
      As[(k + 2) * BM + m] = v.z;
      As[(k + 3) * BM + m] = v.w;
    }
#pragma unroll
    for (int off = 0; off + inner_row_b < BK; off += kStrideB) {
      *reinterpret_cast<float4*>(&Bs[(inner_row_b + off) * BN + inner_col_b * 4]) =
          *reinterpret_cast<const float4*>(&B[(inner_row_b + off) * N + inner_col_b * 4]);
    }
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int dot = 0; dot < BK; ++dot) {
      // Gather this warp's slice of both operands into registers.
#pragma unroll
      for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
        for (int i = 0; i < TM; i += 4) {
          *reinterpret_cast<float4*>(&reg_m[wm * TM + i]) = *reinterpret_cast<const float4*>(
              &As[dot * BM + warp_row * WM + wm * WSUBM + lane_row * TM + i]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WNITER; ++wn) {
#pragma unroll
        for (int i = 0; i < TN; i += 4) {
          *reinterpret_cast<float4*>(&reg_n[wn * TN + i]) = *reinterpret_cast<const float4*>(
              &Bs[dot * BN + warp_col * WN + wn * WSUBN + lane_col * TN + i]);
        }
      }
      // 128 FFMAs against 24 freshly loaded registers.
#pragma unroll
      for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
        for (int wn = 0; wn < WNITER; ++wn) {
#pragma unroll
          for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int n = 0; n < TN; ++n) {
              acc[(wm * TM + m) * kAccStride + wn * TN + n] += reg_m[wm * TM + m] * reg_n[wn * TN + n];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Epilogue, one warp sub-tile at a time so the stores stay contiguous.
  C += (warp_row * WM) * N + warp_col * WN;
#pragma unroll
  for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
    for (int wn = 0; wn < WNITER; ++wn) {
      float* c_sub = C + (wm * WSUBM) * N + wn * WSUBN;
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
          float* dst = &c_sub[(lane_row * TM + m) * N + lane_col * TN + n];
          float4 out = *reinterpret_cast<float4*>(dst);
          const int base = (wm * TM + m) * kAccStride + wn * TN + n;
          out.x = alpha * acc[base + 0] + beta * out.x;
          out.y = alpha * acc[base + 1] + beta * out.y;
          out.z = alpha * acc[base + 2] + beta * out.z;
          out.w = alpha * acc[base + 3] + beta * out.w;
          *reinterpret_cast<float4*>(dst) = out;
        }
      }
    }
  }
}

}  // namespace

void launch_warptiled(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                      float* C) {
  const dim3 block(kThreads);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  warptiled_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_warptiled(const DeviceInfo& dev) {
  return inspect_kernel(warptiled_kernel, kThreads, 0, dev);
}

}  // namespace ws::sgemm
