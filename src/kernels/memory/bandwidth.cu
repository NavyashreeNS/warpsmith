// Memory microbenchmarks: the empirical roofline.
//
// Theoretical bandwidth is a product of clock and bus width, and no real kernel
// reaches it - refresh cycles, read/write turnaround and imperfect scheduling
// cost 15-25% on every GPU ever shipped. Every bandwidth-bound kernel in this
// repository is therefore compared against *two* ceilings: the theoretical
// figure, and the best number any kernel here can actually achieve on this
// device. The second one is measured right here.
//
// The access-width sweep (float, float2, float4) also quantifies something the
// GEMM kernels rely on: how much of the memory pipeline is left idle when loads
// are 4 bytes wide instead of 16.

#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/suite.cuh"
#include "warpsmith/testing.cuh"

namespace ws {
namespace {

constexpr int kBlock = 256;

// --- Read-only: sum every element. The compiler cannot elide the loads because
// the result is written out, and the write is a single float per block. ---
template <typename V, int kWide>
__global__ void read_kernel(const V* __restrict__ in, float* __restrict__ out, std::size_t n_vec) {
  float acc = 0.0f;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const V v = in[i];
    const float* f = reinterpret_cast<const float*>(&v);
#pragma unroll
    for (int k = 0; k < kWide; ++k) acc += f[k];
  }
  // Keep the accumulator alive without paying for a full reduction.
  if (acc == 1.2345e-31f) out[blockIdx.x] = acc;
}

// --- Write-only: stream a constant out. ---
__global__ void write_kernel(float4* __restrict__ out, std::size_t n_vec, float value) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const float4 v = make_float4(value, value, value, value);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) out[i] = v;
}

// --- Copy: the canonical read+write stream. ---
template <typename V>
__global__ void copy_kernel(const V* __restrict__ in, V* __restrict__ out, std::size_t n_vec) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) out[i] = in[i];
}

// --- Triad: two reads, one write, one FMA. The STREAM benchmark's hardest case
// and the closest analogue to a real elementwise kernel. ---
__global__ void triad_kernel(const float4* __restrict__ b, const float4* __restrict__ c,
                             float4* __restrict__ a, std::size_t n_vec, float s) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const float4 bb = b[i];
    const float4 cc = c[i];
    a[i] = make_float4(bb.x + s * cc.x, bb.y + s * cc.y, bb.z + s * cc.z, bb.w + s * cc.w);
  }
}

// A grid sized to the device rather than to the problem: enough blocks to fill
// every SM several times over, then grid-stride loops cover the data. This keeps
// the launch configuration identical across problem sizes.
int grid_for(const DeviceInfo& dev, int blocks_per_sm = 8) {
  return dev.sm_count * blocks_per_sm;
}

}  // namespace

void run_memory(BenchContext& ctx) {
  print_header("Memory bandwidth",
               "grid-stride streaming kernels; establishes the achievable roofline");

  // 128 MiB per array: far larger than the 2 MiB L2, so every access is a true
  // DRAM access and the number means what it says.
  const std::size_t n = ctx.quick ? (1u << 24) : (1u << 25);
  const std::size_t bytes = n * sizeof(float);

  DeviceBuffer<float> a(n), b(n), c(n);
  DeviceBuffer<float> sink(1024);
  a.upload(random_vector(n, 5));
  b.upload(random_vector(n, 6));
  c.upload(random_vector(n, 7));

  const int grid = grid_for(ctx.dev);
  const int iters = std::max(20, ctx.iters / 2);

  printf("\n  -> %s per array, grid = %d blocks x %d threads\n", human_bytes(bytes).c_str(), grid,
         kBlock);
  print_table_head(true);

  struct Entry {
    const char* id;
    const char* name;
    const char* technique;
    int stage;
    double bytes_moved;
    void (*run)(const DeviceInfo&, int, int, DeviceBuffer<float>&, DeviceBuffer<float>&,
                DeviceBuffer<float>&, DeviceBuffer<float>&, std::size_t);
  };

  // Each lambda-free launcher below is a plain function pointer so the table
  // stays a simple data structure.
  const std::vector<Entry> entries = {
      {"read_f32", "Read, 4-byte access", "scalar loads", 1, static_cast<double>(bytes),
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>&,
          DeviceBuffer<float>&, DeviceBuffer<float>& sink, std::size_t n) {
         read_kernel<float, 1><<<grid, kBlock>>>(a.get(), sink.get(), n);
       }},
      {"read_f64x2", "Read, 8-byte access", "float2 loads", 2, static_cast<double>(bytes),
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>&,
          DeviceBuffer<float>&, DeviceBuffer<float>& sink, std::size_t n) {
         read_kernel<float2, 2><<<grid, kBlock>>>(reinterpret_cast<const float2*>(a.get()),
                                                  sink.get(), n / 2);
       }},
      {"read_f32x4", "Read, 16-byte access", "float4 loads", 3, static_cast<double>(bytes),
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>&,
          DeviceBuffer<float>&, DeviceBuffer<float>& sink, std::size_t n) {
         read_kernel<float4, 4><<<grid, kBlock>>>(reinterpret_cast<const float4*>(a.get()),
                                                  sink.get(), n / 4);
       }},
      {"write_f32x4", "Write only", "float4 stores", 4, static_cast<double>(bytes),
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>&,
          DeviceBuffer<float>&, DeviceBuffer<float>&, std::size_t n) {
         write_kernel<<<grid, kBlock>>>(reinterpret_cast<float4*>(a.get()), n / 4, 1.5f);
       }},
      {"copy_f32", "Copy, 4-byte access", "scalar load + store", 5, 2.0 * bytes,
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>& b,
          DeviceBuffer<float>&, DeviceBuffer<float>&, std::size_t n) {
         copy_kernel<float><<<grid, kBlock>>>(a.get(), b.get(), n);
       }},
      {"copy_f32x4", "Copy, 16-byte access", "float4 load + store", 6, 2.0 * bytes,
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>& b,
          DeviceBuffer<float>&, DeviceBuffer<float>&, std::size_t n) {
         copy_kernel<float4><<<grid, kBlock>>>(reinterpret_cast<const float4*>(a.get()),
                                               reinterpret_cast<float4*>(b.get()), n / 4);
       }},
      {"triad", "STREAM triad", "2 reads + 1 write + FMA", 7, 3.0 * bytes,
       [](const DeviceInfo&, int grid, int, DeviceBuffer<float>& a, DeviceBuffer<float>& b,
          DeviceBuffer<float>& c, DeviceBuffer<float>&, std::size_t n) {
         triad_kernel<<<grid, kBlock>>>(reinterpret_cast<const float4*>(b.get()),
                                        reinterpret_cast<const float4*>(c.get()),
                                        reinterpret_cast<float4*>(a.get()), n / 4, 2.5f);
       }},
  };

  double best_gbs = 0.0;
  for (const auto& e : entries) {
    e.run(ctx.dev, grid, kBlock, a, b, c, sink, n);
    WS_CHECK_KERNEL();
    const auto m = time_op([&] { e.run(ctx.dev, grid, kBlock, a, b, c, sink, n); }, ctx.warmup,
                           iters);

    Record r;
    r.suite = "memory";
    r.kernel = e.name;
    r.id = e.id;
    r.stage = e.stage;
    r.technique = e.technique;
    r.size = static_cast<long long>(n);
    r.shape = human_bytes(bytes);
    r.timing = m;
    r.bytes = e.bytes_moved;
    r.correct = true;
    const double gbs = e.bytes_moved / (m.median_ms * 1.0e6);
    if (gbs > best_gbs) best_gbs = gbs;
    print_row(ctx.add(std::move(r)), 0.0, true);
  }

  printf("\n  Achievable bandwidth on this device: %.1f GB/s (%.1f%% of the %.1f GB/s theoretical "
         "peak)\n",
         best_gbs, 100.0 * best_gbs / ctx.dev.peak_bandwidth_gbs, ctx.dev.peak_bandwidth_gbs);
}

}  // namespace ws
