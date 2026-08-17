// warpsmith_bench - the benchmark driver.
//
// Runs one or all suites, prints a table per suite, and writes every measurement
// to JSON so the reports and charts in this repository are generated from data
// rather than transcribed by hand.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/device.cuh"
#include "warpsmith/json.hpp"
#include "warpsmith/suite.cuh"

namespace {

struct SuiteEntry {
  const char* name;
  const char* description;
  void (*run)(ws::BenchContext&);
};

const std::vector<SuiteEntry>& suites() {
  static const std::vector<SuiteEntry> s = {
      {"memory", "streaming bandwidth microbenchmarks (the empirical roofline)", ws::run_memory},
      {"sgemm", "single-precision GEMM, 9 stages from naive to tensor cores", ws::run_sgemm},
      {"tune", "autotune the stage-6 GEMM tile geometry", ws::run_tune},
      {"reduce", "array sum, 6 formulations", ws::run_reduce},
      {"transpose", "matrix transpose: coalescing and bank conflicts", ws::run_transpose},
      {"softmax", "row-wise softmax, online algorithm", ws::run_softmax},
      {"rmsnorm", "RMSNorm with residual fusion", ws::run_rmsnorm},
      {"attention", "attention: materialized vs fused FlashAttention", ws::run_attention},
  };
  return s;
}

void print_usage() {
  printf("warpsmith_bench - CUDA kernel benchmark suite\n\n");
  printf("usage: warpsmith_bench [options]\n\n");
  printf("options:\n");
  printf("  --suite <name>    run one suite (default: all)\n");
  printf("  --json <path>     write results as JSON\n");
  printf("  --iters <n>       timed iterations per kernel (default 100)\n");
  printf("  --warmup <n>      untimed warm-up launches (default 20)\n");
  printf("  --quick           smaller problems and fewer iterations\n");
  printf("  --device <n>      CUDA device ordinal (default 0)\n");
  printf("  --list            list the suites and exit\n");
  printf("  --help            this message\n\n");
  printf("suites:\n");
  for (const auto& s : suites()) printf("  %-12s %s\n", s.name, s.description);
}

void print_device_banner(const ws::DeviceInfo& d) {
  printf("%s\n", std::string(112, '=').c_str());
  printf("  warpsmith - CUDA kernel optimization lab\n");
  printf("%s\n", std::string(112, '=').c_str());
  printf("  device            %s (%s)\n", d.name.c_str(), d.arch().c_str());
  printf("  SMs               %d, %d threads/SM max, %d registers/SM\n", d.sm_count,
         d.max_threads_per_sm, d.regs_per_sm);
  printf("  clocks            %d MHz SM, %d MHz memory (%d-bit bus)\n", d.clock_khz / 1000,
         d.mem_clock_khz / 1000, d.mem_bus_width);
  printf("  memory            %s global, %s L2, %s shared/block\n",
         ws::human_bytes(static_cast<double>(d.global_mem_bytes)).c_str(),
         ws::human_bytes(static_cast<double>(d.l2_bytes)).c_str(),
         ws::human_bytes(static_cast<double>(d.smem_per_block)).c_str());
  printf("  theoretical peak  %.2f TFLOP/s FP32", d.peak_fp32_gflops / 1000.0);
  if (d.peak_tf32_gflops > 0.0) {
    printf(", %.2f TFLOP/s TF32, %.2f TFLOP/s FP16 (tensor cores)",
           d.peak_tf32_gflops / 1000.0, d.peak_fp16_gflops / 1000.0);
  }
  printf("\n");
  printf("  theoretical peak  %.1f GB/s DRAM bandwidth\n", d.peak_bandwidth_gbs);
  printf("  roofline ridge    %.1f FLOP/byte (below this a kernel is bandwidth-bound)\n",
         d.ridge_point());
  printf("  toolkit           CUDA %d.%d runtime, driver %d.%d\n", d.runtime_version / 1000,
         (d.runtime_version % 1000) / 10, d.driver_version / 1000, (d.driver_version % 1000) / 10);
  printf("%s\n", std::string(112, '=').c_str());
}

void write_json(const std::string& path, const ws::BenchContext& ctx) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "[warpsmith] cannot open %s for writing\n", path.c_str());
    return;
  }
  ws::JsonWriter j(out);
  const auto& d = ctx.dev;

  j.begin_object();
  j.field("schema", "warpsmith/results/1");

  j.key("device");
  j.begin_object();
  j.field("name", d.name);
  j.field("arch", d.arch());
  j.field("sm_count", d.sm_count);
  j.field("clock_mhz", d.clock_khz / 1000);
  j.field("mem_clock_mhz", d.mem_clock_khz / 1000);
  j.field("mem_bus_bits", d.mem_bus_width);
  j.field("l2_bytes", d.l2_bytes);
  j.field("smem_per_block", d.smem_per_block);
  j.field("smem_per_sm", d.smem_per_sm);
  j.field("regs_per_sm", d.regs_per_sm);
  j.field("max_threads_per_sm", d.max_threads_per_sm);
  j.field("global_mem_bytes", d.global_mem_bytes);
  j.field("peak_fp32_gflops", d.peak_fp32_gflops);
  j.field("peak_tf32_gflops", d.peak_tf32_gflops);
  j.field("peak_fp16_gflops", d.peak_fp16_gflops);
  j.field("peak_bandwidth_gbs", d.peak_bandwidth_gbs);
  j.field("ridge_point_flop_per_byte", d.ridge_point());
  j.field("cuda_runtime", d.runtime_version);
  j.field("cuda_driver", d.driver_version);
  j.end_object();

  j.key("config");
  j.begin_object();
  j.field("warmup", ctx.warmup);
  j.field("iters", ctx.iters);
  j.field("quick", ctx.quick);
  j.end_object();

  j.key("records");
  j.begin_array();
  for (const auto& r : ctx.records) {
    j.begin_object();
    j.field("suite", r.suite);
    j.field("kernel", r.kernel);
    j.field("id", r.id);
    j.field("stage", r.stage);
    j.field("technique", r.technique);
    j.field("size", r.size);
    j.field("shape", r.shape);
    j.field("median_ms", r.timing.median_ms);
    j.field("min_ms", r.timing.min_ms);
    j.field("mean_ms", r.timing.mean_ms);
    j.field("p95_ms", r.timing.p95_ms);
    j.field("stddev_ms", r.timing.stddev_ms);
    j.field("cv_pct", r.timing.cv_pct);
    j.field("iters", r.timing.iters);
    j.field("flops", r.flops);
    j.field("bytes", r.bytes);
    j.field("gflops", r.gflops);
    j.field("gbytes_per_s", r.gbytes_per_s);
    j.field("pct_of_peak_compute", r.pct_of_peak_compute);
    j.field("pct_of_peak_bandwidth", r.pct_of_peak_bandwidth);
    j.field("pct_of_reference", r.pct_of_reference);
    j.field("speedup_vs_stage1", r.speedup_vs_stage0);
    j.field("max_abs_err", r.max_abs_err);
    j.field("rel_l2_err", r.rel_l2_err);
    j.field("correct", r.correct);
    j.field("regs_per_thread", r.attrs.num_regs);
    j.field("static_smem_bytes", r.attrs.static_smem);
    j.field("spill_bytes", r.attrs.local_bytes);
    j.field("blocks_per_sm", r.attrs.occupancy_blocks_per_sm);
    j.field("occupancy_pct", r.attrs.occupancy_pct);
    j.end_object();
  }
  j.end_array();
  j.end_object();
  out << "\n";
  printf("\nwrote %zu records to %s\n", ctx.records.size(), path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::string suite = "all";
  std::string json_path;
  int device = 0;
  ws::BenchContext ctx;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "[warpsmith] %s requires a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--suite") {
      suite = next("--suite");
    } else if (a == "--json") {
      json_path = next("--json");
    } else if (a == "--iters") {
      ctx.iters = std::atoi(next("--iters").c_str());
    } else if (a == "--warmup") {
      ctx.warmup = std::atoi(next("--warmup").c_str());
    } else if (a == "--device") {
      device = std::atoi(next("--device").c_str());
    } else if (a == "--quick") {
      ctx.quick = true;
    } else if (a == "--list") {
      for (const auto& s : suites()) printf("%-12s %s\n", s.name, s.description);
      return 0;
    } else if (a == "--help" || a == "-h") {
      print_usage();
      return 0;
    } else {
      std::fprintf(stderr, "[warpsmith] unknown option: %s\n", a.c_str());
      print_usage();
      return 2;
    }
  }

  int count = 0;
  WS_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::fprintf(stderr, "[warpsmith] no CUDA device found\n");
    return 1;
  }
  WS_CHECK(cudaSetDevice(device));
  ctx.dev = ws::query_device(device);
  print_device_banner(ctx.dev);

  bool ran = false;
  for (const auto& s : suites()) {
    if (suite == "all" || suite == s.name) {
      s.run(ctx);
      ran = true;
    }
  }
  if (!ran) {
    std::fprintf(stderr, "[warpsmith] unknown suite: %s (try --list)\n", suite.c_str());
    return 2;
  }

  int failures = 0;
  for (const auto& r : ctx.records) {
    if (!r.correct) ++failures;
  }
  printf("\n%s\n", std::string(112, '=').c_str());
  printf("  %zu measurements, %d correctness failures\n", ctx.records.size(), failures);
  printf("%s\n", std::string(112, '=').c_str());

  if (!json_path.empty()) write_json(json_path, ctx);
  return failures == 0 ? 0 : 1;
}
