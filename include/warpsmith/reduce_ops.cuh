// warpsmith - reusable warp and block reduction primitives.
//
// Shared by the softmax, RMSNorm and attention kernels. Warp-level exchange uses
// __shfl_xor_sync (butterfly) rather than __shfl_down_sync so that *every* lane
// ends up holding the result, which removes a broadcast step from the callers.
#pragma once

#include <cuda_runtime.h>

#include <cfloat>

namespace ws {

constexpr int kWarpSize = 32;

struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
  __device__ __forceinline__ static float identity() { return 0.0f; }
};

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a > b ? a : b; }
  __device__ __forceinline__ static float identity() { return -FLT_MAX; }
};

template <typename Op>
__device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    v = op(v, __shfl_xor_sync(0xFFFFFFFFu, v, mask));
  }
  return v;
}

// Block-wide reduction. `scratch` must hold at least blockDim.x / 32 floats.
// Every thread receives the result.
template <typename Op>
__device__ __forceinline__ float block_reduce(float v, float* scratch, Op op) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  v = warp_reduce(v, op);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();

  // The first warp folds the per-warp results, then publishes to everyone.
  if (warp == 0) {
    float x = lane < warps ? scratch[lane] : Op::identity();
    x = warp_reduce(x, op);
    if (lane == 0) scratch[0] = x;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();  // Protects scratch against the next reduction in the caller.
  return result;
}

}  // namespace ws
