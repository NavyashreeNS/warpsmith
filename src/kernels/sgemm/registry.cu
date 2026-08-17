// The SGEMM variant table and the tile-geometry autotuner.
//
// The table is the spine of the benchmark: it fixes the order of the stages, the
// one-line description of what each stage adds, and the alignment each kernel
// requires. The autotuner exists because the "best" tile shape is not a fact
// about GEMM, it is a fact about a particular GPU's register file, shared memory
// and issue width - so it is measured, not assumed.

#include <vector>

#include "warpsmith/sgemm.cuh"
#include "warpsmith/sgemm_vectorized.cuh"

namespace ws::sgemm {

const std::vector<Variant>& variants() {
  static const std::vector<Variant> table = {
      {"00_cublas_fp32", "cuBLAS SGEMM (FP32)", "vendor library baseline", 0, 1, true, false,
       launch_cublas_fp32, nullptr},
      {"01_naive", "Naive, uncoalesced", "one thread per output, column-major thread map", 1, 1,
       false, false, launch_naive, inspect_naive},
      {"02_coalesced", "Global memory coalescing", "swap the thread map so warps read contiguously",
       2, 1, false, false, launch_coalesced, inspect_coalesced},
      {"03_smem_tiled", "Shared memory tiling", "stage 32x32 tiles in shared memory", 3, 1, false,
       false, launch_smem_tiled, inspect_smem_tiled},
      {"04_blocktile_1d", "1D block tiling", "8 accumulators per thread in registers", 4, 64, false,
       false, launch_blocktile_1d, inspect_blocktile_1d},
      {"05_blocktile_2d", "2D block tiling", "8x8 register tile, outer-product inner loop", 5, 128,
       false, false, launch_blocktile_2d, inspect_blocktile_2d},
      {"06_vectorized", "Vectorized + transposed SMEM", "float4 access and a transposed A tile", 6,
       128, false, false, launch_vectorized, inspect_vectorized},
      {"07_warptiled", "Warp tiling", "explicit warp-level tile between block and thread", 7, 128,
       false, false, launch_warptiled, inspect_warptiled},
      {"08_wmma_tf32", "Tensor cores (WMMA, TF32)", "16x16x8 MMA instructions on the tensor cores",
       8, 128, false, true, launch_wmma_tf32, inspect_wmma_tf32},
      {"09_cublas_tf32", "cuBLAS SGEMM (TF32 tensor op)", "vendor library, tensor cores enabled", 9,
       1, true, true, launch_cublas_tf32, nullptr},
  };
  return table;
}

// ---------------------------------------------------------------------------
// Autotuner
// ---------------------------------------------------------------------------

// Each entry is (BM, BN, BK, TM, TN). The list spans small tiles that maximize
// occupancy, wide tiles that maximize reuse, and deliberately over-subscribed
// register configurations that should spill - the spill is worth measuring
// because it is the failure mode this whole design space is navigating around.
#define WS_TUNE_CONFIGS(F)      \
  F(64, 64, 8, 8, 8)            \
  F(64, 64, 8, 4, 8)            \
  F(64, 64, 16, 4, 4)           \
  F(128, 64, 8, 8, 8)           \
  F(128, 64, 16, 8, 8)          \
  F(64, 128, 16, 8, 8)          \
  F(128, 128, 8, 8, 8)          \
  F(128, 128, 16, 8, 8)         \
  F(128, 128, 32, 8, 8)         \
  F(128, 128, 16, 8, 16)        \
  F(128, 128, 16, 16, 8)        \
  F(256, 128, 16, 8, 8)

const std::vector<TuneConfig>& tune_configs() {
  static const std::vector<TuneConfig> cfgs = {
#define WS_EMIT(BM, BN, BK, TM, TN)                                                    \
  TuneConfig{BM, BN, BK, TM, TN, tile_threads(BM, BN, TM, TN), tile_smem(BM, BN, BK),  \
             tile_config_valid(BM, BN, BK, TM, TN)},
      WS_TUNE_CONFIGS(WS_EMIT)
#undef WS_EMIT
  };
  return cfgs;
}

void launch_tuned(int cfg_index, int M, int N, int K, float alpha, const float* A, const float* B,
                  float beta, float* C) {
  int i = 0;
#define WS_DISPATCH(BM, BN, BK, TM, TN)                                                \
  if (i++ == cfg_index) {                                                              \
    constexpr int threads = tile_threads(BM, BN, TM, TN);                              \
    const dim3 block(threads);                                                         \
    const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));                                 \
    vectorized_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C); \
    return;                                                                            \
  }
  WS_TUNE_CONFIGS(WS_DISPATCH)
#undef WS_DISPATCH
}

}  // namespace ws::sgemm
