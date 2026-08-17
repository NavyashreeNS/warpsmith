// warpsmith - measurement harness.
//
// Benchmarking a GPU kernel badly is easy: time one launch, include the launch
// latency, forget that the clocks were still ramping, and report the mean of a
// long-tailed distribution. This harness avoids each of those:
//
//   * timing uses CUDA events recorded on the stream, so host-side launch
//     overhead and driver scheduling jitter are excluded;
//   * every kernel gets untimed warm-up launches so the SM and memory clocks
//     have settled and the instruction cache is hot before the first sample;
//   * many samples are collected and the *median* is reported, with p95 and the
//     coefficient of variation published alongside so the reader can judge how
//     stable the measurement was.
//
// A measurement without a stability figure is an anecdote, so the CV travels
// with every number this project prints.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/device.cuh"

namespace ws {

struct Measurement {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double p95_ms = 0.0;
  double stddev_ms = 0.0;
  double cv_pct = 0.0;  // stddev / mean, as a percentage.
  int iters = 0;

  // Derived throughputs. `flops` and `bytes` describe the *algorithmic* work of
  // one kernel invocation, not the traffic actually observed at DRAM, so a
  // kernel that re-reads its inputs from L2 can legitimately exceed the DRAM
  // roofline here. That is the point: it is how cache reuse becomes visible.
  double gflops(double flops) const { return flops / (median_ms * 1.0e6); }
  double gbytes_per_s(double bytes) const { return bytes / (median_ms * 1.0e6); }
};

inline double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q * (static_cast<double>(v.size()) - 1.0);
  const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
  if (lo == hi) return v[lo];
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

inline Measurement summarize(std::vector<double> samples) {
  Measurement m;
  if (samples.empty()) return m;
  std::sort(samples.begin(), samples.end());

  m.iters = static_cast<int>(samples.size());
  m.min_ms = samples.front();
  m.median_ms = percentile(samples, 0.50);
  m.p95_ms = percentile(samples, 0.95);

  // Mean, standard deviation and the coefficient of variation are computed over
  // the middle 90% of samples rather than all of them.
  //
  // This is not cosmetic. These kernels are timed on a desktop operating system
  // that preempts the GPU for compositing, so an occasional sample is tens of
  // times the median - a scheduling artifact, not a property of the kernel. A
  // single such outlier in 100 samples moves the mean by an order of magnitude
  // and makes the coefficient of variation report several hundred percent, which
  // says nothing useful about the kernel's stability. The median and p95 above
  // are already robust to it; trimming makes the dispersion figure robust too.
  // At least one sample is trimmed from each end whenever there are enough
  // samples to afford it: the large problems are timed with only a couple of
  // dozen iterations, where a 5% fraction rounds down to zero and would leave a
  // single preemption artifact dominating the dispersion figure.
  std::size_t lo = static_cast<std::size_t>(0.05 * samples.size());
  if (lo == 0 && samples.size() >= 8) lo = 1;
  const std::size_t hi = samples.size() - lo;
  const std::size_t n = hi > lo ? hi - lo : samples.size();
  const auto begin = hi > lo ? samples.begin() + lo : samples.begin();
  const auto end = hi > lo ? samples.begin() + hi : samples.end();

  m.mean_ms = std::accumulate(begin, end, 0.0) / static_cast<double>(n);
  double acc = 0.0;
  for (auto it = begin; it != end; ++it) acc += (*it - m.mean_ms) * (*it - m.mean_ms);
  m.stddev_ms = std::sqrt(acc / static_cast<double>(n));
  m.cv_pct = m.mean_ms > 0.0 ? 100.0 * m.stddev_ms / m.mean_ms : 0.0;
  return m;
}

// Times `op` with CUDA events. `op` must enqueue work on the default stream and
// must not synchronize internally.
template <typename Fn>
Measurement time_op(Fn&& op, int warmup = 20, int iters = 100) {
  cudaEvent_t start, stop;
  WS_CHECK(cudaEventCreate(&start));
  WS_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) op();
  WS_CHECK(cudaGetLastError());
  WS_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    WS_CHECK(cudaEventRecord(start));
    op();
    WS_CHECK(cudaEventRecord(stop));
    WS_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    WS_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }
  WS_CHECK(cudaGetLastError());
  WS_CHECK(cudaEventDestroy(start));
  WS_CHECK(cudaEventDestroy(stop));
  return summarize(std::move(samples));
}

// Static resource usage of a kernel, straight from the driver. These numbers
// explain occupancy: registers per thread and shared memory per block are what
// bound how many blocks an SM can host concurrently.
struct KernelAttrs {
  int num_regs = 0;
  std::size_t static_smem = 0;
  std::size_t local_bytes = 0;
  int max_threads_per_block = 0;
  int occupancy_blocks_per_sm = 0;
  double occupancy_pct = 0.0;
};

template <typename KernelPtr>
KernelAttrs inspect_kernel(KernelPtr kernel, int block_size, std::size_t dynamic_smem,
                           const DeviceInfo& dev) {
  cudaFuncAttributes attr{};
  WS_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)));

  KernelAttrs k;
  k.num_regs = attr.numRegs;
  k.static_smem = attr.sharedSizeBytes;
  k.local_bytes = attr.localSizeBytes;
  k.max_threads_per_block = attr.maxThreadsPerBlock;

  int blocks = 0;
  WS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, reinterpret_cast<const void*>(kernel), block_size, dynamic_smem));
  k.occupancy_blocks_per_sm = blocks;
  k.occupancy_pct = dev.max_threads_per_sm > 0
                        ? 100.0 * (blocks * block_size) / dev.max_threads_per_sm
                        : 0.0;
  return k;
}

// One row of the results table: a kernel variant measured at one problem size.
struct Record {
  std::string suite;      // "sgemm", "reduce", ...
  std::string kernel;     // Human-readable variant name.
  std::string id;         // Stable slug, e.g. "05_2d_blocktile".
  int stage = 0;          // Optimization stage index, for ordering charts.
  std::string technique;  // The one optimization this stage introduces.
  long long size = 0;     // Primary problem dimension.
  std::string shape;      // Full shape description, e.g. "4096x4096x4096".

  Measurement timing;
  double flops = 0.0;
  double bytes = 0.0;
  double gflops = 0.0;
  double gbytes_per_s = 0.0;
  double pct_of_peak_compute = 0.0;
  double pct_of_peak_bandwidth = 0.0;
  double pct_of_reference = 0.0;   // Against cuBLAS / the library baseline.
  double speedup_vs_stage0 = 0.0;  // Against the naive implementation.

  // Numerics.
  double max_abs_err = 0.0;
  double rel_l2_err = 0.0;
  bool correct = true;

  KernelAttrs attrs;
};

}  // namespace ws
