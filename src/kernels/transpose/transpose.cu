// Matrix transpose: the shortest path to understanding shared-memory banks.
//
// Transpose does no arithmetic at all. Every microsecond it costs is a memory
// system artifact, which makes it the cleanest possible demonstration of three
// separate hardware behaviours:
//
//   1. Coalescing. The naive kernel reads along rows and writes along columns.
//      One of the two is always strided, and a strided 32-lane access becomes 32
//      transactions instead of 4.
//   2. Shared memory as a staging area. Reading a tile into shared memory lets
//      *both* the global read and the global write be contiguous; the transpose
//      happens inside the SM where strided access is cheap.
//   3. Bank conflicts. Shared memory is 32 banks of 4 bytes. A 32x32 float tile
//      puts an entire column in one bank, so reading the tile column-wise
//      serializes 32 ways. Padding the row stride to 33 shifts each row by one
//      bank and the conflict disappears - one added character, measured here.
//
// A pure copy kernel is included as the upper bound: transpose can never be
// faster than moving the same bytes without rearranging them.

#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/suite.cuh"
#include "warpsmith/testing.cuh"

namespace ws {
namespace {

constexpr int kTile = 32;
constexpr int kRowsPerBlock = 8;  // For the multi-row variants: block is 32x8.

// Upper bound: same traffic, no rearrangement.
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
#pragma unroll
  for (int j = 0; j < kTile; j += kRowsPerBlock) {
    if (x < n && y + j < n) out[(y + j) * n + x] = in[(y + j) * n + x];
  }
}

// Stage 1: direct, with strided writes.
__global__ void naive_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

// Stage 2: staged through shared memory. Both global accesses are contiguous;
// the cost has moved into a 32-way shared-memory bank conflict on the read.
__global__ void tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float tile[kTile][kTile];
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();

  const int tx = blockIdx.y * kTile + threadIdx.x;
  const int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < n && ty < n) out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

// Stage 3: one word of padding per row. tile[t][32] means consecutive rows start
// in consecutive banks, so a column read touches 32 distinct banks.
__global__ void tiled_padded_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float tile[kTile][kTile + 1];
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();

  const int tx = blockIdx.y * kTile + threadIdx.x;
  const int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < n && ty < n) out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

// Stage 4: four rows per thread. A 32x8 block covers a 32x32 tile in four
// passes, which quarters the number of blocks, index computations and barriers
// while keeping every global access 128 bytes wide.
__global__ void tiled_padded_multirow_kernel(const float* __restrict__ in, float* __restrict__ out,
                                             int n) {
  __shared__ float tile[kTile][kTile + 1];
  const int x = blockIdx.x * kTile + threadIdx.x;
  const int y = blockIdx.y * kTile + threadIdx.y;
#pragma unroll
  for (int j = 0; j < kTile; j += kRowsPerBlock) {
    if (x < n && y + j < n) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
  }
  __syncthreads();

  const int tx = blockIdx.y * kTile + threadIdx.x;
  const int ty = blockIdx.x * kTile + threadIdx.y;
#pragma unroll
  for (int j = 0; j < kTile; j += kRowsPerBlock) {
    if (tx < n && ty + j < n) out[(ty + j) * n + tx] = tile[threadIdx.x][threadIdx.y + j];
  }
}

}  // namespace

void run_transpose(BenchContext& ctx) {
  print_header("Transpose", "no arithmetic at all: pure memory-system behaviour");

  const int n = ctx.quick ? 2048 : 4096;
  const std::size_t elems = static_cast<std::size_t>(n) * n;
  const double bytes = 2.0 * static_cast<double>(elems) * sizeof(float);

  const auto host = random_vector(elems, 77);
  DeviceBuffer<float> in(elems), out(elems);
  in.upload(host);

  // Host reference, computed once.
  std::vector<float> want(elems);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) want[static_cast<std::size_t>(c) * n + r] = host[static_cast<std::size_t>(r) * n + c];

  printf("\n  -> %dx%d floats (%s), %s of traffic per pass\n", n, n,
         human_bytes(static_cast<double>(elems) * sizeof(float)).c_str(),
         human_bytes(bytes).c_str());
  print_table_head(true);

  struct Variant {
    const char* id;
    const char* name;
    const char* technique;
    int stage;
    bool is_copy;
    void (*run)(const float*, float*, int);
  };

  const std::vector<Variant> table = {
      {"00_copy_bound", "Copy (upper bound)", "same traffic, no rearrangement", 0, true,
       [](const float* in, float* out, int n) {
         const dim3 block(kTile, kRowsPerBlock);
         const dim3 grid(ceil_div(n, kTile), ceil_div(n, kTile));
         copy_kernel<<<grid, block>>>(in, out, n);
       }},
      {"01_naive", "Naive", "strided global writes", 1, false,
       [](const float* in, float* out, int n) {
         const dim3 block(kTile, kTile);
         const dim3 grid(ceil_div(n, kTile), ceil_div(n, kTile));
         naive_kernel<<<grid, block>>>(in, out, n);
       }},
      {"02_tiled", "SMEM tiled", "both global accesses coalesced", 2, false,
       [](const float* in, float* out, int n) {
         const dim3 block(kTile, kTile);
         const dim3 grid(ceil_div(n, kTile), ceil_div(n, kTile));
         tiled_kernel<<<grid, block>>>(in, out, n);
       }},
      {"03_tiled_padded", "SMEM tiled + padding", "32x33 tile removes bank conflicts", 3, false,
       [](const float* in, float* out, int n) {
         const dim3 block(kTile, kTile);
         const dim3 grid(ceil_div(n, kTile), ceil_div(n, kTile));
         tiled_padded_kernel<<<grid, block>>>(in, out, n);
       }},
      {"04_tiled_multirow", "Padded + 4 rows/thread", "fewer blocks, fewer barriers", 4, false,
       [](const float* in, float* out, int n) {
         const dim3 block(kTile, kRowsPerBlock);
         const dim3 grid(ceil_div(n, kTile), ceil_div(n, kTile));
         tiled_padded_multirow_kernel<<<grid, block>>>(in, out, n);
       }},
  };

  double stage1_ms = 0.0;
  for (const auto& v : table) {
    out.zero();
    v.run(in.get(), out.get(), n);
    WS_CHECK_KERNEL();
    // The copy kernel is a bandwidth reference, not a transpose; comparing it to
    // the transposed reference would be meaningless.
    const auto err = v.is_copy ? compare(out.to_host(), host, 0.0)
                               : compare(out.to_host(), want, 0.0);

    const auto m = time_op([&] { v.run(in.get(), out.get(), n); }, ctx.warmup,
                           std::max(20, ctx.iters / 2));

    Record r;
    r.suite = "transpose";
    r.kernel = v.name;
    r.id = v.id;
    r.stage = v.stage;
    r.technique = v.technique;
    r.size = n;
    r.shape = std::to_string(n) + "x" + std::to_string(n);
    r.timing = m;
    r.bytes = bytes;
    r.max_abs_err = err.max_abs;
    r.rel_l2_err = err.rel_l2;
    r.correct = err.max_abs == 0.0;  // Transpose is exact: any error is a bug.
    if (v.stage == 1) stage1_ms = m.median_ms;
    r.speedup_vs_stage0 = stage1_ms > 0.0 ? stage1_ms / m.median_ms : 0.0;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }
}

}  // namespace ws
