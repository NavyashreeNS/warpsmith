// RMSNorm: the normalization every modern transformer actually uses.
//
// LLaMA, Mistral, Qwen and Gemma all replaced LayerNorm with RMSNorm, which drops
// the mean subtraction and normalizes by the root-mean-square alone:
//
//     y = x * rsqrt(mean(x^2) + eps) * w
//
// It is a strictly bandwidth-bound operation with one reduction, so the
// optimization story is short and sharp: the arithmetic is already free, and
// every gain comes from touching DRAM less and using wider accesses. The
// residual-add variant is included because that is how the operator appears in a
// real decoder block - fusing the residual saves an entire read-modify-write pass
// over the activation tensor, which is why production kernels always fuse it.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/reduce_ops.cuh"
#include "warpsmith/suite.cuh"
#include "warpsmith/testing.cuh"

namespace ws {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxWarps = kBlock / kWarpSize;
constexpr float kEps = 1.0e-6f;

// Stage 1: the definition. Two reads of x - one for the sum of squares, one to
// scale - plus one read of the weight vector and one write.
__global__ void two_pass_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ y, int cols) {
  __shared__ float scratch[kMaxWarps];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * cols;
  float* orow = y + static_cast<std::size_t>(blockIdx.x) * cols;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = row[i];
    acc += v * v;
  }
  acc = block_reduce(acc, scratch, SumOp{});
  const float scale = rsqrtf(acc / static_cast<float>(cols) + kEps);

  for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = row[i] * scale * w[i];
}

// Stage 2: identical traffic, 16-byte accesses.
__global__ void vec4_kernel(const float4* __restrict__ x, const float4* __restrict__ w,
                            float4* __restrict__ y, int cols4, int cols) {
  __shared__ float scratch[kMaxWarps];
  const float4* row = x + static_cast<std::size_t>(blockIdx.x) * cols4;
  float4* orow = y + static_cast<std::size_t>(blockIdx.x) * cols4;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  acc = block_reduce(acc, scratch, SumOp{});
  const float scale = rsqrtf(acc / static_cast<float>(cols) + kEps);

  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    const float4 g = w[i];
    orow[i] = make_float4(v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                          v.w * scale * g.w);
  }
}

// Stage 3: keep the row in shared memory across the reduction, so x is read from
// DRAM exactly once.
__global__ void smem_cached_kernel(const float4* __restrict__ x, const float4* __restrict__ w,
                                   float4* __restrict__ y, int cols4, int cols) {
  extern __shared__ float4 tile[];
  __shared__ float scratch[kMaxWarps];
  const float4* row = x + static_cast<std::size_t>(blockIdx.x) * cols4;
  float4* orow = y + static_cast<std::size_t>(blockIdx.x) * cols4;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    tile[i] = v;
    acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  acc = block_reduce(acc, scratch, SumOp{});
  const float scale = rsqrtf(acc / static_cast<float>(cols) + kEps);

  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = tile[i];
    const float4 g = w[i];
    orow[i] = make_float4(v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                          v.w * scale * g.w);
  }
}

// Stage 4: fused residual add, as it appears inside a decoder block:
//     h = x + residual;  y = rmsnorm(h) * w
// The fused form writes h once and normalizes it out of shared memory. The
// unfused form would need a separate elementwise kernel, a full extra round trip
// through DRAM for the whole tensor.
__global__ void fused_residual_kernel(const float4* __restrict__ x, const float4* __restrict__ res,
                                      const float4* __restrict__ w, float4* __restrict__ h_out,
                                      float4* __restrict__ y, int cols4, int cols) {
  extern __shared__ float4 tile[];
  __shared__ float scratch[kMaxWarps];
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * cols4;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 a = x[off + i];
    const float4 b = res[off + i];
    const float4 v = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    tile[i] = v;
    h_out[off + i] = v;  // The residual stream is needed by the next layer.
    acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  acc = block_reduce(acc, scratch, SumOp{});
  const float scale = rsqrtf(acc / static_cast<float>(cols) + kEps);

  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = tile[i];
    const float4 g = w[i];
    y[off + i] = make_float4(v.x * scale * g.x, v.y * scale * g.y, v.z * scale * g.z,
                             v.w * scale * g.w);
  }
}

std::vector<float> host_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int rows,
                                int cols) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = x.data() + static_cast<std::size_t>(r) * cols;
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * row[c];
    const double scale = 1.0 / std::sqrt(acc / cols + kEps);
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(r) * cols + c] = static_cast<float>(row[c] * scale * w[c]);
    }
  }
  return out;
}

}  // namespace

void run_rmsnorm(BenchContext& ctx) {
  print_header("RMSNorm", "the normalization used by LLaMA-family transformers");

  const int rows = ctx.quick ? 4096 : 8192;   // Tokens in the batch.
  const int cols = 4096;                      // Model dimension.
  const std::size_t elems = static_cast<std::size_t>(rows) * cols;
  const double min_bytes = 2.0 * static_cast<double>(elems) * sizeof(float);

  auto hx = random_vector(elems, 101);
  auto hw = random_vector(static_cast<std::size_t>(cols), 102);
  for (auto& v : hw) v = 1.0f + 0.1f * v;  // Weights near unity, as after training.
  auto hres = random_vector(elems, 103);
  const auto want = host_rmsnorm(hx, hw, rows, cols);

  DeviceBuffer<float> x(elems), y(elems), res(elems), h(elems), w(cols);
  x.upload(hx);
  res.upload(hres);
  w.upload(hw);

  const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const bool smem_fits = smem_bytes <= ctx.dev.smem_per_block;

  printf("\n  -> %d tokens x %d dims (%s), minimum traffic %s\n", rows, cols,
         human_bytes(static_cast<double>(elems) * sizeof(float)).c_str(),
         human_bytes(min_bytes).c_str());
  print_table_head(true);

  struct Variant {
    const char* id;
    const char* name;
    const char* technique;
    int stage;
    bool needs_smem;
    bool fused_residual;
    void (*run)(const float*, const float*, const float*, float*, float*, int, int, std::size_t);
  };

  const std::vector<Variant> table = {
      {"01_two_pass", "Two-pass scalar", "2 reads + 1 write, 4-byte access", 1, false, false,
       [](const float* x, const float* res, const float* w, float* h, float* y, int rows, int cols,
          std::size_t) { two_pass_kernel<<<rows, kBlock>>>(x, w, y, cols); }},
      {"02_vec4", "Two-pass + float4", "same traffic, 16-byte access", 2, false, false,
       [](const float* x, const float* res, const float* w, float* h, float* y, int rows, int cols,
          std::size_t) {
         vec4_kernel<<<rows, kBlock>>>(reinterpret_cast<const float4*>(x),
                                       reinterpret_cast<const float4*>(w),
                                       reinterpret_cast<float4*>(y), cols / 4, cols);
       }},
      {"03_smem_cached", "SMEM-cached row", "1 read + 1 write: the minimum", 3, true, false,
       [](const float* x, const float* res, const float* w, float* h, float* y, int rows, int cols,
          std::size_t smem) {
         smem_cached_kernel<<<rows, kBlock, smem>>>(reinterpret_cast<const float4*>(x),
                                                    reinterpret_cast<const float4*>(w),
                                                    reinterpret_cast<float4*>(y), cols / 4, cols);
       }},
      {"04_fused_residual", "Fused residual + norm", "one kernel for x+res then norm", 4, true, true,
       [](const float* x, const float* res, const float* w, float* h, float* y, int rows, int cols,
          std::size_t smem) {
         fused_residual_kernel<<<rows, kBlock, smem>>>(
             reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(res),
             reinterpret_cast<const float4*>(w), reinterpret_cast<float4*>(h),
             reinterpret_cast<float4*>(y), cols / 4, cols);
       }},
  };

  // The fused variant computes a different function (it adds the residual first),
  // so it is validated against its own reference.
  std::vector<float> hsum(elems);
  for (std::size_t i = 0; i < elems; ++i) hsum[i] = hx[i] + hres[i];
  const auto want_fused = host_rmsnorm(hsum, hw, rows, cols);

  double stage1_ms = 0.0;
  for (const auto& v : table) {
    if (v.needs_smem && !smem_fits) {
      printf("  %-28s SKIP (row needs %s of shared memory)\n", v.id,
             human_bytes(static_cast<double>(smem_bytes)).c_str());
      continue;
    }
    y.zero();
    v.run(x.get(), res.get(), w.get(), h.get(), y.get(), rows, cols, smem_bytes);
    WS_CHECK_KERNEL();
    const auto err = compare(y.to_host(), v.fused_residual ? want_fused : want, 1.0e-5);

    const auto m = time_op(
        [&] { v.run(x.get(), res.get(), w.get(), h.get(), y.get(), rows, cols, smem_bytes); },
        ctx.warmup, std::max(20, ctx.iters / 2));

    Record r;
    r.suite = "rmsnorm";
    r.kernel = v.name;
    r.id = v.id;
    r.stage = v.stage;
    r.technique = v.technique;
    r.size = rows;
    r.shape = std::to_string(rows) + "x" + std::to_string(cols);
    r.timing = m;
    // The fused variant genuinely moves more: it also writes the residual stream.
    r.bytes = v.fused_residual ? min_bytes * 2.0 : min_bytes;
    r.max_abs_err = err.max_abs;
    r.rel_l2_err = err.rel_l2;
    r.correct = !err.has_nan && err.rel_l2 < 1.0e-5;
    if (v.stage == 1) stage1_ms = m.median_ms;
    r.speedup_vs_stage0 = stage1_ms > 0.0 ? stage1_ms / m.median_ms : 0.0;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }
}

}  // namespace ws
