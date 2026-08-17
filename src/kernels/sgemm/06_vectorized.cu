// Stage 6 - the default instantiation of the vectorized kernel.
//
// The 128x128x8 tile with an 8x8 register block is the configuration that the
// autotuner selects on most Ampere parts, and it is registered here as the
// canonical stage 6 so the progression chart has a single line to draw. Run
// `warpsmith_bench --suite tune` to see the whole search space measured.

#include "warpsmith/sgemm.cuh"
#include "warpsmith/sgemm_vectorized.cuh"

namespace ws::sgemm {
namespace {

constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int kThreads = tile_threads(BM, BN, TM, TN);

}  // namespace

void launch_vectorized(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                       float* C) {
  const dim3 block(kThreads);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  vectorized_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

KernelAttrs inspect_vectorized(const DeviceInfo& dev) {
  return inspect_kernel(vectorized_kernel<BM, BN, BK, TM, TN>, kThreads, 0, dev);
}

}  // namespace ws::sgemm
