// Correctness suite for the SGEMM progression.
//
// Every variant is checked against cuBLAS on the same inputs, with non-trivial
// alpha and beta so that a kernel which silently ignores the epilogue cannot
// pass. Sizes include a square case, a skinny case and a case whose K is not a
// multiple of the tile depth, which is the shape that catches missing boundary
// handling in the general-purpose stages.

#include <cstdio>
#include <string>
#include <vector>

#include "warpsmith/common.cuh"
#include "warpsmith/sgemm.cuh"
#include "warpsmith/testing.cuh"

namespace {

struct Case {
  int M, N, K;
  const char* label;
};

int run_case(const Case& c, int& checked) {
  const float alpha = 1.7f, beta = -0.3f;
  auto hA = ws::random_vector(static_cast<std::size_t>(c.M) * c.K, 11);
  auto hB = ws::random_vector(static_cast<std::size_t>(c.K) * c.N, 22);
  auto hC = ws::random_vector(static_cast<std::size_t>(c.M) * c.N, 33);

  ws::DeviceBuffer<float> dA(hA.size()), dB(hB.size()), dC(hC.size()), dRef(hC.size());
  dA.upload(hA);
  dB.upload(hB);

  // Reference: cuBLAS FP32 on a pristine copy of C.
  dRef.upload(hC);
  ws::sgemm::launch_cublas_fp32(c.M, c.N, c.K, alpha, dA.get(), dB.get(), beta, dRef.get());
  WS_CHECK_KERNEL();
  const auto reference = dRef.to_host();

  int failures = 0;
  for (const auto& v : ws::sgemm::variants()) {
    const bool aligned = (c.M % v.align == 0) && (c.N % v.align == 0) && (c.K % v.align == 0);
    if (!aligned) {
      std::printf("  %-30s %-18s SKIP (needs multiples of %d)\n", v.id, c.label, v.align);
      continue;
    }
    dC.upload(hC);
    v.launch(c.M, c.N, c.K, alpha, dA.get(), dB.get(), beta, dC.get());
    WS_CHECK_KERNEL();
    const auto got = dC.to_host();

    // TF32 keeps only 10 mantissa bits, so it is held to a looser bar than the
    // FP32 kernels. Both tolerances scale with sqrt(K) to account for the
    // rounding that any correct summation order accumulates.
    const double scale = std::sqrt(static_cast<double>(c.K));
    const double tol_l2 = v.tensor_core ? 4.0e-3 : 2.0e-6 * scale;
    const double per_elem = v.tensor_core ? 2.0e-2 : 1.0e-4;

    const auto err = ws::compare(got, reference, per_elem);
    const bool ok = !err.has_nan && err.rel_l2 <= tol_l2;
    ++checked;
    if (!ok) ++failures;
    std::printf("  %-30s %-18s %s  rel_l2=%.3e max_abs=%.3e (tol %.1e)\n", v.id, c.label,
                ok ? "PASS" : "FAIL", err.rel_l2, err.max_abs, tol_l2);
    if (!ok) {
      std::printf("      first mismatch at %zu: got %.6f want %.6f\n", err.first_bad_index,
                  err.first_bad_got, err.first_bad_want);
    }
  }
  return failures;
}

}  // namespace

int main() {
  ws::sgemm::cublas_init();

  const std::vector<Case> cases = {
      {512, 512, 512, "square-512"},
      {1024, 1024, 1024, "square-1024"},
      {256, 1024, 512, "rectangular"},
      {129, 257, 61, "ragged"},
  };

  int failures = 0, checked = 0;
  for (const auto& c : cases) {
    std::printf("[case %s] M=%d N=%d K=%d\n", c.label, c.M, c.N, c.K);
    failures += run_case(c, checked);
  }

  // The autotuner's instantiations are separately validated: a mis-specified
  // tile geometry must fail loudly here rather than produce a fast wrong number
  // in the tuning table.
  std::printf("[case autotune-configs] M=N=K=512\n");
  {
    const int M = 512, N = 512, K = 512;
    const float alpha = 1.3f, beta = 0.4f;
    auto hA = ws::random_vector(static_cast<std::size_t>(M) * K, 7);
    auto hB = ws::random_vector(static_cast<std::size_t>(K) * N, 8);
    auto hC = ws::random_vector(static_cast<std::size_t>(M) * N, 9);
    ws::DeviceBuffer<float> dA(hA.size()), dB(hB.size()), dC(hC.size()), dRef(hC.size());
    dA.upload(hA);
    dB.upload(hB);
    dRef.upload(hC);
    ws::sgemm::launch_cublas_fp32(M, N, K, alpha, dA.get(), dB.get(), beta, dRef.get());
    WS_CHECK_KERNEL();
    const auto reference = dRef.to_host();

    const auto& cfgs = ws::sgemm::tune_configs();
    for (std::size_t i = 0; i < cfgs.size(); ++i) {
      const auto& cfg = cfgs[i];
      char label[64];
      std::snprintf(label, sizeof(label), "%dx%dx%d/%dx%d", cfg.bm, cfg.bn, cfg.bk, cfg.tm, cfg.tn);
      if (!cfg.valid || M % cfg.bm || N % cfg.bn || K % cfg.bk) {
        std::printf("  %-30s %-18s SKIP\n", label, "tune");
        continue;
      }
      dC.upload(hC);
      ws::sgemm::launch_tuned(static_cast<int>(i), M, N, K, alpha, dA.get(), dB.get(), beta,
                              dC.get());
      WS_CHECK_KERNEL();
      const auto err = ws::compare(dC.to_host(), reference, 1.0e-4);
      const double tol = 2.0e-6 * std::sqrt(static_cast<double>(K));
      const bool ok = !err.has_nan && err.rel_l2 <= tol;
      ++checked;
      if (!ok) ++failures;
      std::printf("  %-30s %-18s %s  rel_l2=%.3e\n", label, "tune", ok ? "PASS" : "FAIL",
                  err.rel_l2);
    }
  }

  ws::sgemm::cublas_shutdown();
  std::printf("\n%d checks, %d failures\n", checked, failures);
  return failures == 0 ? 0 : 1;
}
