// Reduction: summing 64 million floats, six ways.
//
// Reduction is the other half of GPU programming. GEMM is compute-bound and
// rewards blocking; reduction is bandwidth-bound and rewards getting out of the
// memory system's way. The theoretical best is simply "read the array once at
// full bandwidth", so here the roofline is a hard wall and the interesting
// question is how close each formulation gets to it.
//
// A second result falls out for free. Floating-point addition is not
// associative, so the summation *order* changes the answer. The tree reductions
// have depth log2(n) and accumulate error like sqrt(log n); a long sequential
// accumulation accumulates it like sqrt(n). The measured relative error against
// a Kahan-compensated double-precision sum is reported alongside the timings, and
// the difference is three orders of magnitude - a reminder that the parallel
// algorithm here is not just faster than the serial one, it is more accurate.

#include <cub/cub.cuh>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/suite.cuh"
#include "warpsmith/testing.cuh"

namespace ws {
namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;

// --- Stage 1: one atomic per element. Correct, trivially short, and a disaster:
// every one of 64 million threads serializes on the same 4 bytes. ---
__global__ void atomic_kernel(const float* __restrict__ in, float* __restrict__ out,
                              std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, in[i]);
}

// --- Stage 2: shared-memory tree with interleaved addressing.
// The classic first attempt. Its flaw is the modulo: on the first iteration only
// even lanes are active, so half of every warp idles while still being issued,
// and the stride-2 shared-memory pattern hits two-way bank conflicts. ---
__global__ void tree_interleaved_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        std::size_t n) {
  __shared__ float s[kBlock];
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x % (2 * stride) == 0) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// --- Stage 3: sequential addressing.
// Same tree, same number of additions, but the active threads are now the *first*
// half of the block. Whole warps retire together instead of every warp running
// half-empty, and the shared-memory strides are conflict-free. ---
__global__ void tree_sequential_kernel(const float* __restrict__ in, float* __restrict__ out,
                                       std::size_t n) {
  __shared__ float s[kBlock];
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? in[i] : 0.0f;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// --- Stage 4: grid-stride accumulation before the tree.
// The tree itself is pure overhead: log2(256) = 8 barriers to produce one number
// from 256. So do as much work as possible *before* entering it. Each thread
// accumulates many elements in a register with a grid-stride loop, and the block
// only pays for one tree at the end. The grid now sizes to the GPU, not the
// input. ---
__global__ void grid_stride_kernel(const float* __restrict__ in, float* __restrict__ out,
                                   std::size_t n) {
  __shared__ float s[kBlock];
  float acc = 0.0f;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc += in[i];
  }
  s[threadIdx.x] = acc;
  __syncthreads();

  for (unsigned st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// Warp-level reduction with no shared memory and no barriers: the shuffle
// instruction reads another lane's register directly.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

// --- Stage 5: shuffle-based two-level reduction.
// Shared memory is only used to combine the 8 warps of the block - the 32 lanes
// within a warp exchange through the register file instead. That removes 5 of the
// 8 barriers and all but 8 shared-memory transactions per block. ---
__global__ void shuffle_kernel(const float* __restrict__ in, float* __restrict__ out,
                               std::size_t n) {
  __shared__ float warp_sums[kBlock / kWarp];
  float acc = 0.0f;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc += in[i];
  }

  acc = warp_reduce_sum(acc);
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();

  if (warp == 0) {
    acc = lane < static_cast<int>(blockDim.x / kWarp) ? warp_sums[lane] : 0.0f;
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[blockIdx.x] = acc;
  }
}

// --- Stage 6: vectorized loads on top of the shuffle reduction.
// The reduction arithmetic is already free; what remains is the load pipeline.
// float4 accesses quarter the number of memory instructions and are what finally
// pins this kernel to the bandwidth roofline. ---
__global__ void vec4_shuffle_kernel(const float4* __restrict__ in, float* __restrict__ out,
                                    std::size_t n_vec) {
  __shared__ float warp_sums[kBlock / kWarp];
  float acc = 0.0f;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    const float4 v = in[i];
    // Pairwise, not sequential: a two-level sum of four values has half the
    // error growth of a running accumulation.
    acc += (v.x + v.y) + (v.z + v.w);
  }

  acc = warp_reduce_sum(acc);
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();

  if (warp == 0) {
    acc = lane < static_cast<int>(blockDim.x / kWarp) ? warp_sums[lane] : 0.0f;
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[blockIdx.x] = acc;
  }
}

// Second pass: fold the per-block partials into one value. Always one block, so
// its cost is a fixed ~5 microseconds regardless of the input size.
__global__ void finalize_kernel(const float* __restrict__ partials, float* __restrict__ out,
                               int count) {
  __shared__ float warp_sums[kBlock / kWarp];
  float acc = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) acc += partials[i];
  acc = warp_reduce_sum(acc);
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  if (lane == 0) warp_sums[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    acc = lane < static_cast<int>(blockDim.x / kWarp) ? warp_sums[lane] : 0.0f;
    acc = warp_reduce_sum(acc);
    if (lane == 0) *out = acc;
  }
}

// Kahan-compensated double-precision sum: the trusted answer.
double kahan_sum(const std::vector<float>& v) {
  double sum = 0.0, comp = 0.0;
  for (float x : v) {
    const double y = static_cast<double>(x) - comp;
    const double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  return sum;
}

}  // namespace

void run_reduce(BenchContext& ctx) {
  print_header("Reduction", "sum of a large float array; bandwidth-bound, so the roofline is a wall");

  const std::size_t n = ctx.quick ? (1u << 24) : (1u << 26);
  const double bytes = static_cast<double>(n) * sizeof(float);
  const auto host = random_vector(n, 42);
  const double truth = kahan_sum(host);

  DeviceBuffer<float> in(n);
  in.upload(host);

  // Enough blocks to saturate the device for the grid-stride variants.
  const int persistent_grid = ctx.dev.sm_count * 16;
  const int full_grid = static_cast<int>(ceil_div(static_cast<int>(n / 1), kBlock));
  DeviceBuffer<float> partials(static_cast<std::size_t>(full_grid) + 1);
  DeviceBuffer<float> result(1);

  printf("\n  -> n = %zu floats (%s), reference sum = %.6f (Kahan, double precision)\n", n,
         human_bytes(bytes).c_str(), truth);
  print_table_head(true);

  const int iters = std::max(20, ctx.iters / 2);

  struct Variant {
    const char* id;
    const char* name;
    const char* technique;
    int stage;
    void (*run)(const float*, float*, float*, std::size_t, int, int);
  };

  const std::vector<Variant> table = {
      {"01_atomic_global", "Global atomics", "one atomicAdd per element", 1,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         WS_CHECK(cudaMemset(out, 0, sizeof(float)));
         atomic_kernel<<<ceil_div(static_cast<int>(n), kBlock), kBlock>>>(in, out, n);
       }},
      {"02_tree_interleaved", "SMEM tree, interleaved", "divergent branch, bank conflicts", 2,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         const int grid = ceil_div(static_cast<int>(n), kBlock);
         tree_interleaved_kernel<<<grid, kBlock>>>(in, partials, n);
         finalize_kernel<<<1, kBlock>>>(partials, out, grid);
       }},
      {"03_tree_sequential", "SMEM tree, sequential", "conflict-free, whole warps retire", 3,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         const int grid = ceil_div(static_cast<int>(n), kBlock);
         tree_sequential_kernel<<<grid, kBlock>>>(in, partials, n);
         finalize_kernel<<<1, kBlock>>>(partials, out, grid);
       }},
      {"04_grid_stride", "Grid-stride accumulate", "many elements per thread before the tree", 4,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         grid_stride_kernel<<<persistent, kBlock>>>(in, partials, n);
         finalize_kernel<<<1, kBlock>>>(partials, out, persistent);
       }},
      {"05_warp_shuffle", "Warp shuffle", "register exchange instead of shared memory", 5,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         shuffle_kernel<<<persistent, kBlock>>>(in, partials, n);
         finalize_kernel<<<1, kBlock>>>(partials, out, persistent);
       }},
      {"06_vec4_shuffle", "Vectorized + shuffle", "float4 loads, pairwise accumulation", 6,
       [](const float* in, float* partials, float* out, std::size_t n, int full, int persistent) {
         vec4_shuffle_kernel<<<persistent, kBlock>>>(reinterpret_cast<const float4*>(in), partials,
                                                     n / 4);
         finalize_kernel<<<1, kBlock>>>(partials, out, persistent);
       }},
  };

  double stage1_ms = 0.0;

  // CUB's DeviceReduce is the library baseline: a production, heavily tuned
  // reduction shipped with the CUDA toolkit.
  {
    void* temp = nullptr;
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(temp, temp_bytes, in.get(), result.get(), static_cast<int>(n));
    WS_CHECK(cudaMalloc(&temp, temp_bytes));
    cub::DeviceReduce::Sum(temp, temp_bytes, in.get(), result.get(), static_cast<int>(n));
    WS_CHECK_KERNEL();
    std::vector<float> got;
    result.download(got);
    const auto m = time_op(
        [&] { cub::DeviceReduce::Sum(temp, temp_bytes, in.get(), result.get(), static_cast<int>(n)); },
        ctx.warmup, iters);
    WS_CHECK(cudaFree(temp));

    Record r;
    r.suite = "reduce";
    r.kernel = "CUB DeviceReduce::Sum";
    r.id = "00_cub";
    r.stage = 0;
    r.technique = "vendor library baseline";
    r.size = static_cast<long long>(n);
    r.shape = human_bytes(bytes);
    r.timing = m;
    r.bytes = bytes;
    r.rel_l2_err = std::fabs((got[0] - truth) / truth);
    r.max_abs_err = std::fabs(got[0] - truth);
    r.correct = r.rel_l2_err < 1.0e-4;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }

  for (const auto& v : table) {
    partials.zero();
    result.zero();
    v.run(in.get(), partials.get(), result.get(), n, full_grid, persistent_grid);
    WS_CHECK_KERNEL();
    std::vector<float> got;
    result.download(got);

    const auto m = time_op(
        [&] { v.run(in.get(), partials.get(), result.get(), n, full_grid, persistent_grid); },
        ctx.warmup, iters);

    Record r;
    r.suite = "reduce";
    r.kernel = v.name;
    r.id = v.id;
    r.stage = v.stage;
    r.technique = v.technique;
    r.size = static_cast<long long>(n);
    r.shape = human_bytes(bytes);
    r.timing = m;
    r.bytes = bytes;
    r.rel_l2_err = std::fabs((got[0] - truth) / truth);
    r.max_abs_err = std::fabs(got[0] - truth);
    // The atomic variant is non-deterministic in its summation order, so it is
    // held to a looser bar; anything above 1e-3 indicates a real bug, not
    // rounding.
    r.correct = r.rel_l2_err < (v.stage == 1 ? 1.0e-2 : 1.0e-4);
    if (v.stage == 1) stage1_ms = m.median_ms;
    r.speedup_vs_stage0 = stage1_ms > 0.0 ? stage1_ms / m.median_ms : 0.0;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }
}

}  // namespace ws
