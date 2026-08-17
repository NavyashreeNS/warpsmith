// warpsmith - device introspection and theoretical peak derivation.
//
// Every performance claim in this repository is expressed as a fraction of a
// hardware limit, so those limits must be derived from the device itself rather
// than pasted from a spec sheet. This header turns `cudaDeviceProp` into the
// two roofline anchors we care about: peak FP32 throughput and peak DRAM
// bandwidth.
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "warpsmith/common.cuh"

namespace ws {

// FP32 CUDA cores per SM, indexed by compute capability. These counts are an
// architectural property of the SM design and are not exposed by the driver
// API, so a lookup table is unavoidable.
inline int cores_per_sm(int major, int minor) {
  switch (major * 10 + minor) {
    case 60: return 64;    // Pascal GP100
    case 61: case 62: return 128;  // Pascal GP10x
    case 70: case 72: return 64;   // Volta
    case 75: return 64;    // Turing
    case 80: return 64;    // Ampere GA100
    case 86: case 87: case 89: return 128;  // Ampere GA10x, Ada
    case 90: return 128;   // Hopper
    case 100: case 120: return 128;  // Blackwell
    default: return 128;   // Reasonable modern default.
  }
}

// Tensor-core FP16 throughput relative to the FP32 FMA rate. Consumer Ampere
// (GA10x) and Ada run FP16 tensor ops at 4x the FP32 core rate; TF32 at 2x.
inline double tensor_fp16_ratio(int major, int minor) {
  const int cc = major * 10 + minor;
  if (cc == 86 || cc == 87 || cc == 89) return 4.0;
  if (cc == 80 || cc == 90) return 8.0;
  if (cc == 75) return 8.0;
  return 0.0;  // No tensor cores, or unknown ratio.
}

struct DeviceInfo {
  std::string name;
  int major = 0;
  int minor = 0;
  int sm_count = 0;
  int clock_khz = 0;
  int mem_clock_khz = 0;
  int mem_bus_width = 0;
  int l2_bytes = 0;
  std::size_t smem_per_block = 0;
  std::size_t smem_per_sm = 0;
  int regs_per_sm = 0;
  int max_threads_per_sm = 0;
  int warp_size = 32;
  std::size_t global_mem_bytes = 0;
  int driver_version = 0;
  int runtime_version = 0;

  // Derived roofline anchors.
  double peak_fp32_gflops = 0.0;   // Non-tensor FMA throughput.
  double peak_tf32_gflops = 0.0;   // Tensor-core TF32 throughput.
  double peak_fp16_gflops = 0.0;   // Tensor-core FP16 throughput.
  double peak_bandwidth_gbs = 0.0; // DRAM, theoretical.

  std::string arch() const { return "sm_" + std::to_string(major) + std::to_string(minor); }

  // FLOP:byte ratio at which a kernel flips from bandwidth-bound to
  // compute-bound. This is the "ridge point" of the roofline.
  double ridge_point() const { return peak_fp32_gflops / peak_bandwidth_gbs; }
};

inline DeviceInfo query_device(int ordinal = 0) {
  cudaDeviceProp p{};
  WS_CHECK(cudaGetDeviceProperties(&p, ordinal));

  DeviceInfo d;
  d.name = p.name;
  d.major = p.major;
  d.minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.clock_khz = p.clockRate;
  d.mem_clock_khz = p.memoryClockRate;
  d.mem_bus_width = p.memoryBusWidth;
  d.l2_bytes = p.l2CacheSize;
  d.smem_per_block = p.sharedMemPerBlock;
  d.smem_per_sm = p.sharedMemPerMultiprocessor;
  d.regs_per_sm = p.regsPerMultiprocessor;
  d.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
  d.warp_size = p.warpSize;
  d.global_mem_bytes = p.totalGlobalMem;
  WS_CHECK(cudaDriverGetVersion(&d.driver_version));
  WS_CHECK(cudaRuntimeGetVersion(&d.runtime_version));

  const double clock_ghz = d.clock_khz / 1.0e6;
  // 2 FLOP per FMA per core per cycle.
  d.peak_fp32_gflops = d.sm_count * cores_per_sm(d.major, d.minor) * 2.0 * clock_ghz;
  const double ratio = tensor_fp16_ratio(d.major, d.minor);
  d.peak_fp16_gflops = d.peak_fp32_gflops * ratio;
  d.peak_tf32_gflops = d.peak_fp32_gflops * (ratio > 0.0 ? ratio / 2.0 : 0.0);
  // GDDR/HBM transfers twice per clock; bus width is in bits.
  d.peak_bandwidth_gbs = (d.mem_clock_khz * 1.0e3) * 2.0 * (d.mem_bus_width / 8.0) / 1.0e9;
  return d;
}

}  // namespace ws
