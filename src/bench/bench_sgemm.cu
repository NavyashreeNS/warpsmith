// SGEMM benchmark: the full progression, measured at several problem sizes.
//
// Two ratios are reported for each variant. "% of cuBLAS" answers the practical
// question - is this kernel competitive with the vendor library. "% of peak"
// answers the architectural one - how much of the silicon is actually working.
// Both are needed: cuBLAS itself only reaches a fraction of theoretical peak, so
// a kernel at 95% of cuBLAS and 55% of peak is simultaneously excellent and
// evidence that FP32 GEMM does not saturate this GPU.

#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/sgemm.cuh"
#include "warpsmith/suite.cuh"
#include "warpsmith/testing.cuh"

namespace ws {
namespace {

// 2 FLOP per multiply-add, M*N*K of them, plus the alpha/beta epilogue.
double gemm_flops(int M, int N, int K) {
  return 2.0 * static_cast<double>(M) * N * K + 3.0 * static_cast<double>(M) * N;
}

// Minimum DRAM traffic for one GEMM, assuming perfect caching: read A, read B,
// read C, write C.
double gemm_bytes(int M, int N, int K) {
  return sizeof(float) * (static_cast<double>(M) * K + static_cast<double>(K) * N +
                          2.0 * static_cast<double>(M) * N);
}

}  // namespace

void run_sgemm(BenchContext& ctx) {
  // The sweep starts at 1024. Below that, a single GEMM finishes in tens of
  // microseconds and the measurement is dominated by launch latency, clock ramping
  // and OS-level GPU preemption rather than by the kernel - cuBLAS itself shows a
  // triple-digit coefficient of variation at 512 on this device. Measuring there
  // would produce numbers that look precise and mean nothing.
  const std::vector<int> sizes = ctx.quick ? std::vector<int>{1024, 2048}
                                           : std::vector<int>{1024, 2048, 4096};

  print_header("SGEMM", "C = alpha*A@B + beta*C, row-major FP32, square problems");

  sgemm::cublas_init();
  for (int n : sizes) {
    const int M = n, N = n, K = n;
    const std::size_t elems = static_cast<std::size_t>(n) * n;

    auto hA = random_vector(elems, 11);
    auto hB = random_vector(elems, 22);
    auto hC = random_vector(elems, 33);

    DeviceBuffer<float> dA(elems), dB(elems), dC(elems), dRef(elems), dInit(elems);
    dA.upload(hA);
    dB.upload(hB);
    dInit.upload(hC);

    const float alpha = 1.0f, beta = 0.0f;

    // Reference result and reference timing, both from cuBLAS FP32.
    dRef.copy_from(dInit);
    sgemm::launch_cublas_fp32(M, N, K, alpha, dA.get(), dB.get(), beta, dRef.get());
    WS_CHECK_KERNEL();
    const auto reference = dRef.to_host();

    // Iteration count scales down with problem size so the whole sweep stays
    // interactive while small problems still get enough samples to be stable.
    const int iters = n >= 4096 ? std::max(25, ctx.iters / 4)
                                : (n >= 2048 ? std::max(30, ctx.iters / 3) : ctx.iters);
    const int warmup = n >= 4096 ? std::max(5, ctx.warmup / 4) : ctx.warmup;

    double reference_gflops = 0.0;
    double stage0_ms = 0.0;

    printf("\n  %s  M=N=K=%d   %s per matrix\n", "->", n,
           human_bytes(static_cast<double>(elems) * sizeof(float)).c_str());
    print_table_head(false);

    for (const auto& v : sgemm::variants()) {
      if (M % v.align || N % v.align || K % v.align) continue;
      // The naive kernels take seconds at 4096; measuring them there costs more
      // wall-clock than the information is worth, and the trend is already
      // established at smaller sizes.
      if (v.stage >= 1 && v.stage <= 2 && n >= 4096) continue;

      dC.copy_from(dInit);
      v.launch(M, N, K, alpha, dA.get(), dB.get(), beta, dC.get());
      WS_CHECK_KERNEL();
      const double tol_l2 = v.tensor_core ? 4.0e-3 : 2.0e-6 * std::sqrt(static_cast<double>(K));
      const auto err = compare(dC.to_host(), reference, v.tensor_core ? 2.0e-2 : 1.0e-4);

      const auto m = time_op(
          [&] { v.launch(M, N, K, alpha, dA.get(), dB.get(), beta, dC.get()); }, warmup, iters);

      Record r;
      r.suite = "sgemm";
      r.kernel = v.name;
      r.id = v.id;
      r.stage = v.stage;
      r.technique = v.technique;
      r.size = n;
      r.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
      r.timing = m;
      r.flops = gemm_flops(M, N, K);
      r.bytes = gemm_bytes(M, N, K);
      r.max_abs_err = err.max_abs;
      r.rel_l2_err = err.rel_l2;
      r.correct = !err.has_nan && err.rel_l2 <= tol_l2;
      if (v.inspect) r.attrs = v.inspect(ctx.dev);

      if (v.stage == 0) {
        reference_gflops = r.flops / (m.median_ms * 1.0e6);
      }
      if (v.stage == 1) stage0_ms = m.median_ms;
      r.pct_of_reference =
          reference_gflops > 0.0 ? 100.0 * (r.flops / (m.median_ms * 1.0e6)) / reference_gflops : 0.0;
      r.speedup_vs_stage0 = stage0_ms > 0.0 ? stage0_ms / m.median_ms : 0.0;

      print_row(ctx.add(std::move(r)), reference_gflops, false);
    }
  }
  sgemm::cublas_shutdown();
}

void run_tune(BenchContext& ctx) {
  print_header("SGEMM autotune",
               "tile-geometry sweep of the stage-6 kernel; the winner is hardware-specific");

  const int n = ctx.quick ? 1024 : 2048;
  const std::size_t elems = static_cast<std::size_t>(n) * n;
  auto hA = random_vector(elems, 11);
  auto hB = random_vector(elems, 22);

  DeviceBuffer<float> dA(elems), dB(elems), dC(elems), dRef(elems);
  dA.upload(hA);
  dB.upload(hB);
  dC.zero();
  dRef.zero();

  sgemm::cublas_init();
  sgemm::launch_cublas_fp32(n, n, n, 1.0f, dA.get(), dB.get(), 0.0f, dRef.get());
  WS_CHECK_KERNEL();
  const auto reference = dRef.to_host();
  const auto cublas_m = time_op(
      [&] { sgemm::launch_cublas_fp32(n, n, n, 1.0f, dA.get(), dB.get(), 0.0f, dRef.get()); },
      ctx.warmup, std::max(20, ctx.iters / 3));
  const double flops = gemm_flops(n, n, n);
  const double reference_gflops = flops / (cublas_m.median_ms * 1.0e6);
  sgemm::cublas_shutdown();

  printf("\n  -> M=N=K=%d, cuBLAS FP32 reference = %.0f GFLOP/s\n", n, reference_gflops);
  printf("  %-22s %8s %8s %8s %9s %10s %8s %8s\n", "tile BMxBNxBK / TMxTN", "threads", "smem",
         "regs", "occ %", "GFLOP/s", "% cuBLAS", "check");
  printf("  %s\n", std::string(88, '-').c_str());

  const auto& cfgs = sgemm::tune_configs();
  for (std::size_t i = 0; i < cfgs.size(); ++i) {
    const auto& cfg = cfgs[i];
    char label[64];
    std::snprintf(label, sizeof(label), "%dx%dx%d / %dx%d", cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn);
    if (!cfg.valid || n % cfg.bm || n % cfg.bn || n % cfg.bk) {
      printf("  %-22s %8d %8s %8s %9s %10s %8s %8s\n", label, cfg.threads, "-", "-", "-", "-", "-",
             "skip");
      continue;
    }

    dC.zero();
    sgemm::launch_tuned(static_cast<int>(i), n, n, n, 1.0f, dA.get(), dB.get(), 0.0f, dC.get());
    WS_CHECK_KERNEL();
    const auto err = compare(dC.to_host(), reference, 1.0e-4);
    const auto m = time_op(
        [&] { sgemm::launch_tuned(static_cast<int>(i), n, n, n, 1.0f, dA.get(), dB.get(), 0.0f,
                                  dC.get()); },
        ctx.warmup, std::max(20, ctx.iters / 3));

    Record r;
    r.suite = "sgemm_tune";
    r.kernel = label;
    r.id = label;
    r.stage = static_cast<int>(i);
    r.technique = "tile geometry sweep";
    r.size = n;
    r.shape = std::to_string(n) + "x" + std::to_string(n) + "x" + std::to_string(n);
    r.timing = m;
    r.flops = flops;
    r.bytes = gemm_bytes(n, n, n);
    r.rel_l2_err = err.rel_l2;
    r.max_abs_err = err.max_abs;
    r.correct = !err.has_nan && err.rel_l2 <= 2.0e-6 * std::sqrt(static_cast<double>(n));
    r.pct_of_reference = 100.0 * (flops / (m.median_ms * 1.0e6)) / reference_gflops;

    const double gflops = flops / (m.median_ms * 1.0e6);
    printf("  %-22s %8d %8s %8s %8.1f%% %10.0f %7.1f%% %8s\n", label, cfg.threads,
           human_bytes(static_cast<double>(cfg.smem_bytes)).c_str(), "-", 0.0, gflops,
           r.pct_of_reference, r.correct ? "ok" : "FAIL");
    ctx.add(std::move(r));
  }
}

}  // namespace ws
