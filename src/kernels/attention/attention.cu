// Scaled dot-product attention, materialized versus fused (FlashAttention).
//
// This is the most consequential kernel optimization of the last decade, and the
// reason is not arithmetic. Both implementations below perform the same 4*S^2*D
// multiply-adds. The difference is that the textbook version writes the S x S
// attention matrix to DRAM, reads it back for the softmax, writes it again, and
// reads it a third time for the value projection - while the fused version never
// materializes it at all.
//
// The consequences are worth stating precisely, because they are what the
// benchmark measures:
//
//   * Memory. The baseline needs O(S^2) scratch space. At S = 4096 that is 64 MB
//     for a single head, which is why attention was the memory bottleneck of
//     early transformers. The fused kernel needs O(1) beyond its inputs.
//   * Traffic. The baseline moves ~4*S^2 floats through DRAM that carry no
//     information the kernel did not just compute. The fused kernel keeps the
//     tile in registers and shared memory and moves none of it.
//   * The enabling trick. Softmax appears to need a global maximum before
//     anything can be exponentiated, which is what forces materialization. The
//     online formulation removes that dependency: carry a running maximum `m` and
//     running denominator `l`, and when a tile contains a larger value, correct
//     the accumulator by exactly exp(m_old - m_new). The result is not an
//     approximation - it is the same number the two-pass algorithm produces.
//
// Causal masking is benchmarked separately because it is what autoregressive
// decoding actually runs, and because the fused kernel can exploit it in a way the
// materialized one cannot: an entire key tile lying above the diagonal is skipped
// before a single dot product is issued, so the work halves. The baseline must
// still compute those scores and then throw them away.

#include <cublas_v2.h>

#include <cfloat>
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

constexpr int kHeadDim = 64;   // Standard for LLaMA-class models.
constexpr int kFlashBM = 128;  // Query rows per block, one per thread.
constexpr int kFlashBN = 32;   // Key/value rows per tile.

// ---------------------------------------------------------------------------
// Materialized baseline
// ---------------------------------------------------------------------------

// Row softmax over the S x S score matrix, with optional causal masking. This is
// the strongest reasonable baseline: cuBLAS for both GEMMs and a fused,
// vectorized, single-pass softmax in between.
__global__ void score_softmax_kernel(float* __restrict__ scores, int seq, bool causal) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float* p = scores + static_cast<std::size_t>(row) * seq;
  // Under causal masking a query at position `row` may only attend to keys at
  // positions <= row.
  const int limit = causal ? row + 1 : seq;

  float m = -FLT_MAX;
  float l = 0.0f;
  for (int i = threadIdx.x; i < limit; i += blockDim.x) {
    const float x = p[i];
    const float m_new = fmaxf(m, x);
    l = l * __expf(m - m_new) + __expf(x - m_new);
    m = m_new;
  }
  const float m_block = block_reduce(m, scratch, MaxOp{});
  l *= __expf(m - m_block);
  l = block_reduce(l, scratch, SumOp{});

  const float inv = 1.0f / l;
  for (int i = threadIdx.x; i < seq; i += blockDim.x) {
    p[i] = i < limit ? __expf(p[i] - m_block) * inv : 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Fused kernel (FlashAttention-style)
// ---------------------------------------------------------------------------

template <int D, int BM, int BN, bool kCausal>
__global__ __launch_bounds__(BM) void flash_kernel(const float* __restrict__ Q,
                                                   const float* __restrict__ K,
                                                   const float* __restrict__ V,
                                                   float* __restrict__ O, int seq, float scale) {
  __shared__ float Ks[BN * D];
  __shared__ float Vs[BN * D];

  const int head = blockIdx.y;
  const int row = blockIdx.x * BM + threadIdx.x;
  const std::size_t head_off = static_cast<std::size_t>(head) * seq * D;

  // The query row lives in registers for the whole kernel: it is reused against
  // every key tile, so it must never be re-read.
  float q[D];
  {
    const float4* q4 = reinterpret_cast<const float4*>(Q + head_off + static_cast<std::size_t>(row) * D);
#pragma unroll
    for (int i = 0; i < D / 4; ++i) {
      const float4 v = q4[i];
      q[i * 4 + 0] = v.x;
      q[i * 4 + 1] = v.y;
      q[i * 4 + 2] = v.z;
      q[i * 4 + 3] = v.w;
    }
  }

  float acc[D];
#pragma unroll
  for (int d = 0; d < D; ++d) acc[d] = 0.0f;
  float m = -FLT_MAX;  // Running maximum.
  float l = 0.0f;      // Running softmax denominator.

  // Under causal masking, tiles beyond this query block's diagonal contain
  // nothing but masked entries, so the loop simply stops early.
  const int key_end = kCausal ? min(seq, (static_cast<int>(blockIdx.x) + 1) * BM) : seq;

  for (int kt = 0; kt < key_end; kt += BN) {
    // Cooperative, fully coalesced staging of the key and value tiles.
    {
      const float4* k4 = reinterpret_cast<const float4*>(K + head_off + static_cast<std::size_t>(kt) * D);
      const float4* v4 = reinterpret_cast<const float4*>(V + head_off + static_cast<std::size_t>(kt) * D);
      float4* ks4 = reinterpret_cast<float4*>(Ks);
      float4* vs4 = reinterpret_cast<float4*>(Vs);
#pragma unroll
      for (int i = threadIdx.x; i < (BN * D) / 4; i += BM) {
        ks4[i] = k4[i];
        vs4[i] = v4[i];
      }
    }
    __syncthreads();

    // Scores for this tile. Every thread reads the same Ks element at the same
    // time, so the shared-memory access is a broadcast: no bank conflicts.
    float s[BN];
    float tile_max = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < BN; ++j) {
      float dot = 0.0f;
#pragma unroll
      for (int d = 0; d < D; ++d) dot += q[d] * Ks[j * D + d];
      dot *= scale;
      if (kCausal && (kt + j) > row) dot = -FLT_MAX;  // Mask inside the diagonal tile.
      s[j] = dot;
      tile_max = fmaxf(tile_max, dot);
    }

    // Online softmax rescaling: one correction factor for the whole tile.
    const float m_new = fmaxf(m, tile_max);
    const float corr = __expf(m - m_new);
    float l_tile = 0.0f;
#pragma unroll
    for (int j = 0; j < BN; ++j) {
      s[j] = __expf(s[j] - m_new);
      l_tile += s[j];
    }
    l = l * corr + l_tile;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[d] *= corr;

    // Accumulate this tile's contribution to the output.
#pragma unroll
    for (int j = 0; j < BN; ++j) {
      const float p = s[j];
#pragma unroll
      for (int d = 0; d < D; ++d) acc[d] += p * Vs[j * D + d];
    }
    m = m_new;
    __syncthreads();
  }

  // Normalize once, at the very end - the denominator was never needed before.
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  float4* o4 = reinterpret_cast<float4*>(O + head_off + static_cast<std::size_t>(row) * D);
#pragma unroll
  for (int i = 0; i < D / 4; ++i) {
    o4[i] = make_float4(acc[i * 4 + 0] * inv, acc[i * 4 + 1] * inv, acc[i * 4 + 2] * inv,
                        acc[i * 4 + 3] * inv);
  }
}

// ---------------------------------------------------------------------------
// Host reference, in double precision
// ---------------------------------------------------------------------------

std::vector<float> host_attention(const std::vector<float>& Q, const std::vector<float>& K,
                                  const std::vector<float>& V, int heads, int seq, int dim,
                                  bool causal) {
  std::vector<float> out(Q.size(), 0.0f);
  std::vector<double> s(seq);
  for (int h = 0; h < heads; ++h) {
    const std::size_t off = static_cast<std::size_t>(h) * seq * dim;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    for (int i = 0; i < seq; ++i) {
      const int limit = causal ? i + 1 : seq;
      double m = -1.0e300;
      for (int j = 0; j < limit; ++j) {
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
          dot += static_cast<double>(Q[off + static_cast<std::size_t>(i) * dim + d]) *
                 static_cast<double>(K[off + static_cast<std::size_t>(j) * dim + d]);
        }
        s[j] = dot * scale;
        if (s[j] > m) m = s[j];
      }
      double sum = 0.0;
      for (int j = 0; j < limit; ++j) {
        s[j] = std::exp(s[j] - m);
        sum += s[j];
      }
      for (int d = 0; d < dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j < limit; ++j) {
          acc += s[j] * static_cast<double>(V[off + static_cast<std::size_t>(j) * dim + d]);
        }
        out[off + static_cast<std::size_t>(i) * dim + d] = static_cast<float>(acc / sum);
      }
    }
  }
  return out;
}

}  // namespace

void run_attention(BenchContext& ctx) {
  print_header("Attention",
               "materialized (cuBLAS + softmax) versus fused FlashAttention-style streaming");

  const int heads = 8;
  const int dim = kHeadDim;
  const std::vector<int> seqs = ctx.quick ? std::vector<int>{1024, 2048}
                                          : std::vector<int>{1024, 2048, 4096};

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  for (int seq : seqs) {
    const std::size_t tensor = static_cast<std::size_t>(heads) * seq * dim;
    const std::size_t scores = static_cast<std::size_t>(seq) * seq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    auto hQ = random_vector(tensor, 201);
    auto hK = random_vector(tensor, 202);
    auto hV = random_vector(tensor, 203);

    DeviceBuffer<float> Q(tensor), K(tensor), V(tensor), O(tensor), Oref(tensor);
    DeviceBuffer<float> S(scores);  // Only the baseline needs this.
    Q.upload(hQ);
    K.upload(hK);
    V.upload(hV);

    // The double-precision host reference is O(H*S^2*D) and becomes the slowest
    // part of the whole benchmark, so it is computed on a subset of heads at the
    // larger sequence lengths.
    const int ref_heads = seq <= 1024 ? heads : 1;

    printf("\n  -> S=%d, H=%d, D=%d   Q/K/V %s each, baseline scratch %s\n", seq, heads, dim,
           human_bytes(static_cast<double>(tensor) * sizeof(float)).c_str(),
           human_bytes(static_cast<double>(scores) * sizeof(float)).c_str());
    print_table_head(false);

    for (int causal_i = 0; causal_i < 2; ++causal_i) {
      const bool causal = causal_i == 1;
      // Causal attention performs half the work, up to the diagonal.
      const double work_fraction = causal ? 0.5 : 1.0;
      const double flops = 4.0 * static_cast<double>(seq) * seq * dim * heads * work_fraction;

      std::vector<float> want;
      {
        std::vector<float> qsub(hQ.begin(), hQ.begin() + static_cast<std::size_t>(ref_heads) * seq * dim);
        std::vector<float> ksub(hK.begin(), hK.begin() + static_cast<std::size_t>(ref_heads) * seq * dim);
        std::vector<float> vsub(hV.begin(), hV.begin() + static_cast<std::size_t>(ref_heads) * seq * dim);
        want = host_attention(qsub, ksub, vsub, ref_heads, seq, dim, causal);
      }

      // --- Baseline: three kernels per head, S x S materialized. ---
      auto run_baseline = [&] {
        const float zero = 0.0f, one = 1.0f;
        for (int h = 0; h < heads; ++h) {
          const float* q = Q.get() + static_cast<std::size_t>(h) * seq * dim;
          const float* k = K.get() + static_cast<std::size_t>(h) * seq * dim;
          const float* v = V.get() + static_cast<std::size_t>(h) * seq * dim;
          float* o = O.get() + static_cast<std::size_t>(h) * seq * dim;
          // scores = Q @ K^T * scale   (row-major via the operand-swap identity)
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, seq, seq, dim, &scale, k, dim, q, dim, &zero,
                      S.get(), seq);
          score_softmax_kernel<<<seq, 256>>>(S.get(), seq, causal);
          // out = P @ V
          cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, seq, seq, &one, v, dim, S.get(), seq,
                      &zero, o, dim);
        }
      };

      auto run_flash = [&] {
        const dim3 block(kFlashBM);
        const dim3 grid(ceil_div(seq, kFlashBM), heads);
        if (causal) {
          flash_kernel<kHeadDim, kFlashBM, kFlashBN, true>
              <<<grid, block>>>(Q.get(), K.get(), V.get(), O.get(), seq, scale);
        } else {
          flash_kernel<kHeadDim, kFlashBM, kFlashBN, false>
              <<<grid, block>>>(Q.get(), K.get(), V.get(), O.get(), seq, scale);
        }
      };

      struct Impl {
        const char* id;
        const char* name;
        const char* technique;
        int stage;
        double scratch_bytes;
        bool fused;
      };
      const Impl impls[2] = {
          {causal ? "02_materialized_causal" : "00_materialized",
           causal ? "Materialized, causal" : "Materialized",
           "cuBLAS GEMM + softmax + cuBLAS GEMM", causal ? 2 : 0,
           static_cast<double>(scores) * sizeof(float), false},
          {causal ? "03_flash_causal" : "01_flash", causal ? "Fused flash, causal" : "Fused flash",
           "one kernel, online softmax, no S x S matrix", causal ? 3 : 1, 0.0, true},
      };

      double baseline_ms = 0.0;
      for (const auto& impl : impls) {
        O.zero();
        if (impl.fused) {
          run_flash();
        } else {
          run_baseline();
        }
        WS_CHECK_KERNEL();

        // Compare only the heads the host reference covers.
        auto got_full = O.to_host();
        std::vector<float> got(got_full.begin(),
                               got_full.begin() + static_cast<std::size_t>(ref_heads) * seq * dim);
        const auto err = compare(got, want, 1.0e-3);

        const int iters = seq >= 4096 ? std::max(8, ctx.iters / 10) : std::max(15, ctx.iters / 4);
        const auto m = time_op([&] { impl.fused ? run_flash() : run_baseline(); },
                               std::max(5, ctx.warmup / 2), iters);

        Record r;
        r.suite = "attention";
        r.kernel = impl.name;
        r.id = impl.id;
        r.stage = impl.stage;
        r.technique = impl.technique;
        r.size = seq;
        r.shape = "S" + std::to_string(seq) + "xH" + std::to_string(heads) + "xD" +
                  std::to_string(dim);
        r.timing = m;
        r.flops = flops;
        // Traffic: Q, K, V and O once each, plus the baseline's round trips
        // through the S x S score matrix (write, read, write, read).
        const double io = 4.0 * static_cast<double>(tensor) * sizeof(float);
        r.bytes = impl.fused ? io
                             : io + 4.0 * static_cast<double>(scores) * sizeof(float) * heads;
        r.max_abs_err = err.max_abs;
        r.rel_l2_err = err.rel_l2;
        r.correct = !err.has_nan && err.rel_l2 < 2.0e-3;
        if (!impl.fused) baseline_ms = m.median_ms;
        r.speedup_vs_stage0 = baseline_ms > 0.0 ? baseline_ms / m.median_ms : 0.0;
        r.pct_of_reference = baseline_ms > 0.0 ? 100.0 * baseline_ms / m.median_ms : 0.0;

        print_row(ctx.add(std::move(r)), 0.0, false);
      }
    }
  }
  cublasDestroy(handle);
}

}  // namespace ws
