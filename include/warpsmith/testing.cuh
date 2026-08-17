// warpsmith - numerical comparison utilities.
//
// A fast kernel that computes the wrong answer is worthless, so every variant in
// this repository is validated against a trusted reference before it is timed,
// and the validation numbers are published next to the performance numbers.
//
// Two metrics are used, because either alone is misleading:
//
//   * max absolute error catches a single badly indexed element that an averaged
//     metric would drown out;
//   * relative L2 error (||x - ref|| / ||ref||) measures whether the result is
//     the right answer overall, and is the metric that scales sensibly with the
//     problem size - for a K-term dot product, FP32 rounding alone accumulates
//     an error that grows like sqrt(K), so a fixed absolute tolerance would
//     wrongly fail large problems.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "warpsmith/common.cuh"

namespace ws {

// A small, fast, fully deterministic PRNG. Reproducibility matters more than
// statistical perfection here: the same seed must produce the same inputs on
// every machine so that reported error figures are comparable.
class Xorshift {
 public:
  explicit Xorshift(std::uint32_t seed = 0x9E3779B9u) : s_(seed ? seed : 1u) {}
  std::uint32_t next() {
    s_ ^= s_ << 13;
    s_ ^= s_ >> 17;
    s_ ^= s_ << 5;
    return s_;
  }
  // Uniform in [-1, 1).
  float next_float() { return static_cast<float>(next() >> 8) / 8388608.0f - 1.0f; }

 private:
  std::uint32_t s_;
};

inline std::vector<float> random_vector(std::size_t n, std::uint32_t seed) {
  Xorshift rng(seed);
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = rng.next_float();
  return v;
}

struct ErrorMetrics {
  double max_abs = 0.0;
  double rel_l2 = 0.0;
  std::size_t first_bad_index = 0;
  double first_bad_got = 0.0;
  double first_bad_want = 0.0;
  bool has_nan = false;
};

inline ErrorMetrics compare(const std::vector<float>& got, const std::vector<float>& want,
                            double per_element_tol) {
  ErrorMetrics e;
  double num = 0.0, den = 0.0;
  bool found_bad = false;
  const std::size_t n = got.size() < want.size() ? got.size() : want.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double g = got[i], w = want[i];
    if (std::isnan(g) || std::isinf(g)) e.has_nan = true;
    const double d = std::fabs(g - w);
    if (d > e.max_abs) e.max_abs = d;
    num += d * d;
    den += w * w;
    if (!found_bad && d > per_element_tol * (1.0 + std::fabs(w))) {
      found_bad = true;
      e.first_bad_index = i;
      e.first_bad_got = g;
      e.first_bad_want = w;
    }
  }
  e.rel_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  return e;
}

// Device buffer with automatic release, so a failing test cannot leak VRAM
// across the hundreds of allocations a full sweep performs.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
    o.ptr_ = nullptr;
    o.count_ = 0;
  }

  void allocate(std::size_t count) {
    release();
    count_ = count;
    WS_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }
  void release() {
    if (ptr_) {
      WS_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      count_ = 0;
    }
  }
  void upload(const std::vector<T>& host) {
    WS_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(std::vector<T>& host) const {
    host.resize(count_);
    WS_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
  std::vector<T> to_host() const {
    std::vector<T> h;
    download(h);
    return h;
  }
  void zero() { WS_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }
  void copy_from(const DeviceBuffer& other) {
    WS_CHECK(cudaMemcpy(ptr_, other.ptr_, count_ * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace ws
