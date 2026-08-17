// Row-wise softmax: the memory-traffic optimization.
//
// Softmax is defined by a formula that reads its input three times - once for the
// maximum, once for the sum of exponentials, once to normalize. On a
// bandwidth-bound operation, each of those passes is a direct multiplier on the
// runtime, so the entire optimization problem is "read the input fewer times".
//
//   stage 1: 3 reads + 1 write   - the definition, transcribed
//   stage 2: 2 reads + 1 write   - online softmax: max and sum in one pass
//   stage 3: 2 reads + 1 write   - same algorithm, float4 and shuffle reductions
//   stage 4: 1 read  + 1 write   - the row is cached in shared memory
//
// Stage 2 deserves a note, because it is the same trick that makes
// FlashAttention possible. The naive algorithm needs the maximum before it can
// exponentiate anything. The online form keeps a running maximum `m` and a
// running sum `l`, and whenever a larger element appears it retroactively
// corrects the sum by the exact factor exp(m_old - m_new). One pass, and
// numerically identical to the two-pass version - no overflow, because nothing is
// ever exponentiated with a positive argument.
//
// Throughput is reported against the *minimum* traffic any correct
// implementation must move (one read plus one write), so the number for a
// multi-pass kernel is an effective bandwidth: it says how fast the operation was
// delivered, not how fast the DRAM ran.

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

// Stage 1: three separate passes over the row, exactly as the formula reads.
__global__ void three_pass_kernel(const float* __restrict__ in, float* __restrict__ out, int cols) {
  __shared__ float scratch[kMaxWarps];
  const float* row = in + static_cast<std::size_t>(blockIdx.x) * cols;
  float* orow = out + static_cast<std::size_t>(blockIdx.x) * cols;

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, row[i]);
  m = block_reduce(m, scratch, MaxOp{});

  float l = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) l += __expf(row[i] - m);
  l = block_reduce(l, scratch, SumOp{});

  const float inv = 1.0f / l;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = __expf(row[i] - m) * inv;
}

// Stage 2: online softmax. One pass computes both statistics.
__global__ void online_kernel(const float* __restrict__ in, float* __restrict__ out, int cols) {
  __shared__ float scratch[kMaxWarps];
  const float* row = in + static_cast<std::size_t>(blockIdx.x) * cols;
  float* orow = out + static_cast<std::size_t>(blockIdx.x) * cols;

  float m = -FLT_MAX;
  float l = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float x = row[i];
    const float m_new = fmaxf(m, x);
    // Correct the running sum for the new maximum, then add this term.
    l = l * __expf(m - m_new) + __expf(x - m_new);
    m = m_new;
  }
  // Combining partial (m, l) pairs across threads uses the same correction.
  const float m_block = block_reduce(m, scratch, MaxOp{});
  l *= __expf(m - m_block);
  l = block_reduce(l, scratch, SumOp{});

  const float inv = 1.0f / l;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = __expf(row[i] - m_block) * inv;
}

// Stage 3: the online algorithm with 16-byte accesses and no scalar tail
// (columns are a multiple of 4 in the benchmark).
__global__ void online_vec4_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                   int cols4) {
  __shared__ float scratch[kMaxWarps];
  const float4* row = in + static_cast<std::size_t>(blockIdx.x) * cols4;
  float4* orow = out + static_cast<std::size_t>(blockIdx.x) * cols4;

  float m = -FLT_MAX;
  float l = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    const float vmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    const float m_new = fmaxf(m, vmax);
    const float corr = __expf(m - m_new);
    l = l * corr + __expf(v.x - m_new) + __expf(v.y - m_new) + __expf(v.z - m_new) +
        __expf(v.w - m_new);
    m = m_new;
  }
  const float m_block = block_reduce(m, scratch, MaxOp{});
  l *= __expf(m - m_block);
  l = block_reduce(l, scratch, SumOp{});

  const float inv = 1.0f / l;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    orow[i] = make_float4(__expf(v.x - m_block) * inv, __expf(v.y - m_block) * inv,
                          __expf(v.z - m_block) * inv, __expf(v.w - m_block) * inv);
  }
}

// Stage 4: cache the row in shared memory, so DRAM sees the theoretical minimum
// of one read and one write. Requires cols * 4 bytes to fit in a block's shared
// memory, which is the constraint that makes this a specialization rather than
// the default.
__global__ void smem_cached_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                                   int cols4) {
  extern __shared__ float4 tile[];
  __shared__ float scratch[kMaxWarps];
  const float4* row = in + static_cast<std::size_t>(blockIdx.x) * cols4;
  float4* orow = out + static_cast<std::size_t>(blockIdx.x) * cols4;

  float m = -FLT_MAX;
  float l = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = row[i];
    tile[i] = v;
    const float vmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    const float m_new = fmaxf(m, vmax);
    const float corr = __expf(m - m_new);
    l = l * corr + __expf(v.x - m_new) + __expf(v.y - m_new) + __expf(v.z - m_new) +
        __expf(v.w - m_new);
    m = m_new;
  }
  const float m_block = block_reduce(m, scratch, MaxOp{});
  l *= __expf(m - m_block);
  l = block_reduce(l, scratch, SumOp{});

  const float inv = 1.0f / l;
  for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
    const float4 v = tile[i];  // Re-read from shared memory, not DRAM.
    orow[i] = make_float4(__expf(v.x - m_block) * inv, __expf(v.y - m_block) * inv,
                          __expf(v.z - m_block) * inv, __expf(v.w - m_block) * inv);
  }
}

// Double-precision host softmax: the trusted answer.
std::vector<float> host_softmax(const std::vector<float>& x, int rows, int cols) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = x.data() + static_cast<std::size_t>(r) * cols;
    double m = -1.0e300;
    for (int c = 0; c < cols; ++c) m = row[c] > m ? row[c] : m;
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(static_cast<double>(row[c]) - m);
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(r) * cols + c] =
          static_cast<float>(std::exp(static_cast<double>(row[c]) - m) / sum);
    }
  }
  return out;
}

}  // namespace

void run_softmax(BenchContext& ctx) {
  print_header("Softmax", "row-wise, the shape of a language model's logits");

  const int rows = ctx.quick ? 2048 : 4096;
  const int cols = 4096;
  const std::size_t elems = static_cast<std::size_t>(rows) * cols;
  // The minimum any correct implementation must move: read the input once,
  // write the output once.
  const double min_bytes = 2.0 * static_cast<double>(elems) * sizeof(float);

  auto host = random_vector(elems, 91);
  // Widen the range so the maximum-subtraction actually matters; with inputs in
  // [-1, 1] a buggy kernel that skipped it would still pass.
  for (auto& v : host) v *= 12.0f;
  const auto want = host_softmax(host, rows, cols);

  DeviceBuffer<float> in(elems), out(elems);
  in.upload(host);

  printf("\n  -> %d rows x %d cols (%s), minimum traffic %s\n", rows, cols,
         human_bytes(static_cast<double>(elems) * sizeof(float)).c_str(),
         human_bytes(min_bytes).c_str());
  print_table_head(true);

  const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const bool smem_fits = smem_bytes <= ctx.dev.smem_per_block;

  struct Variant {
    const char* id;
    const char* name;
    const char* technique;
    int stage;
    bool needs_smem;
    void (*run)(const float*, float*, int, int, std::size_t);
  };

  const std::vector<Variant> table = {
      {"01_three_pass", "Three-pass (definition)", "3 reads + 1 write", 1, false,
       [](const float* in, float* out, int rows, int cols, std::size_t) {
         three_pass_kernel<<<rows, kBlock>>>(in, out, cols);
       }},
      {"02_online", "Online softmax", "2 reads + 1 write, running max", 2, false,
       [](const float* in, float* out, int rows, int cols, std::size_t) {
         online_kernel<<<rows, kBlock>>>(in, out, cols);
       }},
      {"03_online_vec4", "Online + float4", "16-byte accesses, shuffle reductions", 3, false,
       [](const float* in, float* out, int rows, int cols, std::size_t) {
         online_vec4_kernel<<<rows, kBlock>>>(reinterpret_cast<const float4*>(in),
                                              reinterpret_cast<float4*>(out), cols / 4);
       }},
      {"04_smem_cached", "SMEM-cached row", "1 read + 1 write: the minimum", 4, true,
       [](const float* in, float* out, int rows, int cols, std::size_t smem) {
         smem_cached_kernel<<<rows, kBlock, smem>>>(reinterpret_cast<const float4*>(in),
                                                    reinterpret_cast<float4*>(out), cols / 4);
       }},
  };

  double stage1_ms = 0.0;
  for (const auto& v : table) {
    if (v.needs_smem && !smem_fits) {
      printf("  %-28s SKIP (row needs %s of shared memory)\n", v.id,
             human_bytes(static_cast<double>(smem_bytes)).c_str());
      continue;
    }
    out.zero();
    v.run(in.get(), out.get(), rows, cols, smem_bytes);
    WS_CHECK_KERNEL();
    const auto err = compare(out.to_host(), want, 1.0e-5);

    const auto m = time_op([&] { v.run(in.get(), out.get(), rows, cols, smem_bytes); }, ctx.warmup,
                           std::max(20, ctx.iters / 2));

    Record r;
    r.suite = "softmax";
    r.kernel = v.name;
    r.id = v.id;
    r.stage = v.stage;
    r.technique = v.technique;
    r.size = rows;
    r.shape = std::to_string(rows) + "x" + std::to_string(cols);
    r.timing = m;
    r.bytes = min_bytes;
    r.max_abs_err = err.max_abs;
    r.rel_l2_err = err.rel_l2;
    r.correct = !err.has_nan && err.rel_l2 < 1.0e-5;
    if (v.stage == 1) stage1_ms = m.median_ms;
    r.speedup_vs_stage0 = stage1_ms > 0.0 ? stage1_ms / m.median_ms : 0.0;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }
}

}  // namespace ws
