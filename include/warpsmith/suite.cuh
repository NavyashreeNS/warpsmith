// warpsmith - benchmark suite interface.
//
// Each suite owns one kernel family, registers its variants, measures them and
// appends Records to the shared context. Keeping the registration local to each
// suite means adding a kernel touches exactly one file.
#pragma once

#include <string>
#include <vector>

#include "warpsmith/bench.cuh"
#include "warpsmith/device.cuh"

namespace ws {

struct BenchContext {
  DeviceInfo dev;
  int warmup = 20;
  int iters = 100;
  bool quick = false;   // Smaller problem sizes and fewer iterations.
  bool verbose = true;
  std::vector<Record> records;

  // Fills in the derived throughput and roofline fractions, then stores the
  // record and hands back a reference to it. Centralising this keeps the
  // definition of "% of peak" identical across every suite, and returning the
  // stored record means the console table and the JSON always show the same
  // derived numbers.
  const Record& add(Record r) {
    r.gflops = r.flops > 0.0 ? r.timing.gflops(r.flops) : 0.0;
    r.gbytes_per_s = r.bytes > 0.0 ? r.timing.gbytes_per_s(r.bytes) : 0.0;
    r.pct_of_peak_compute =
        dev.peak_fp32_gflops > 0.0 ? 100.0 * r.gflops / dev.peak_fp32_gflops : 0.0;
    r.pct_of_peak_bandwidth =
        dev.peak_bandwidth_gbs > 0.0 ? 100.0 * r.gbytes_per_s / dev.peak_bandwidth_gbs : 0.0;
    // TF32 tensor-core kernels are measured against the tensor-core ceiling, not
    // the FP32 one; comparing them to the FP32 peak would flatter them.
    if (dev.peak_tf32_gflops > 0.0 && r.id.find("tf32") != std::string::npos) {
      r.pct_of_peak_compute = 100.0 * r.gflops / dev.peak_tf32_gflops;
    }
    records.push_back(std::move(r));
    return records.back();
  }
};

// Suites. Each is self-contained and may be run individually.
void run_sgemm(BenchContext&);
void run_tune(BenchContext&);
void run_memory(BenchContext&);
void run_reduce(BenchContext&);
void run_transpose(BenchContext&);
void run_softmax(BenchContext&);
void run_rmsnorm(BenchContext&);
void run_attention(BenchContext&);

// Shared console table formatting, so every suite prints the same shape.
void print_header(const std::string& title, const std::string& subtitle);
void print_row(const Record& r, double reference_gflops, bool bandwidth_bound);
void print_table_head(bool bandwidth_bound);

}  // namespace ws
