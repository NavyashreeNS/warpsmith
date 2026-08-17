// warpsmith - common utilities shared by every kernel and benchmark.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ws {

// Aborts with file/line context when a CUDA call fails. Every runtime call in
// this project is wrapped: a silently ignored error turns a benchmark into a
// meaningless number.
inline void cuda_check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[warpsmith] CUDA error %s:%d -> %s (%s)\n", file, line,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define WS_CHECK(expr) ::ws::cuda_check((expr), __FILE__, __LINE__)

// Catches launch configuration failures and any asynchronous fault raised by
// the most recently launched kernel.
#define WS_CHECK_KERNEL()                       \
  do {                                          \
    WS_CHECK(cudaGetLastError());               \
    WS_CHECK(cudaDeviceSynchronize());          \
  } while (0)

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Rounds `v` up to the next multiple of `m` (m must be a power of two).
constexpr int round_up_pow2(int v, int m) { return (v + m - 1) & ~(m - 1); }

// Human-readable byte counts for benchmark banners.
inline std::string human_bytes(double bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int u = 0;
  while (bytes >= 1024.0 && u < 4) {
    bytes /= 1024.0;
    ++u;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s", bytes, units[u]);
  return std::string(buf);
}

}  // namespace ws
