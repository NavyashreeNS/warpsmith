// Console table formatting.
//
// Every suite prints through these three functions so the output has one shape.
// The columns are chosen to answer, in order: how long did it take, how stable
// was the measurement, how fast is that in useful units, how much of the machine
// did it use, how does it compare, and why - registers and occupancy being the
// two numbers that explain most surprises.

#include <cstdio>
#include <string>

#include "warpsmith/suite.cuh"

namespace ws {

void print_header(const std::string& title, const std::string& subtitle) {
  printf("\n%s\n", std::string(112, '=').c_str());
  printf("  %s\n", title.c_str());
  if (!subtitle.empty()) printf("  %s\n", subtitle.c_str());
  printf("%s\n", std::string(112, '=').c_str());
}

void print_table_head(bool bandwidth_bound) {
  if (bandwidth_bound) {
    printf("  %-26s %9s %9s %6s %11s %8s %9s %6s %7s\n", "kernel", "median", "p95", "cv",
           "GB/s", "% peak", "speedup", "regs", "check");
  } else {
    printf("  %-26s %9s %9s %6s %11s %8s %9s %9s %6s %7s\n", "kernel", "median", "p95", "cv",
           "GFLOP/s", "% peak", "% ref", "speedup", "regs", "check");
  }
  printf("  %s\n", std::string(110, '-').c_str());
}

namespace {

// Milliseconds are the wrong unit for a 6-microsecond kernel and the right one
// for a 200-millisecond kernel, so the unit follows the magnitude.
std::string format_time(double ms) {
  char buf[32];
  if (ms < 1.0) {
    std::snprintf(buf, sizeof(buf), "%.1f us", ms * 1000.0);
  } else if (ms < 1000.0) {
    std::snprintf(buf, sizeof(buf), "%.3f ms", ms);
  } else {
    std::snprintf(buf, sizeof(buf), "%.3f s", ms / 1000.0);
  }
  return std::string(buf);
}

}  // namespace

void print_row(const Record& r, double reference_gflops, bool bandwidth_bound) {
  const double gflops = r.flops > 0.0 ? r.timing.gflops(r.flops) : 0.0;
  const double gbs = r.bytes > 0.0 ? r.timing.gbytes_per_s(r.bytes) : 0.0;
  const char* check = r.correct ? "ok" : "FAIL";

  char regs[16];
  if (r.attrs.num_regs > 0) {
    std::snprintf(regs, sizeof(regs), "%d", r.attrs.num_regs);
  } else {
    std::snprintf(regs, sizeof(regs), "-");
  }
  char speedup[16];
  if (r.speedup_vs_stage0 > 0.0) {
    std::snprintf(speedup, sizeof(speedup), "%.2fx", r.speedup_vs_stage0);
  } else {
    std::snprintf(speedup, sizeof(speedup), "-");
  }

  if (bandwidth_bound) {
    printf("  %-26s %9s %9s %5.1f%% %11.1f %7.1f%% %9s %6s %7s\n", r.id.c_str(),
           format_time(r.timing.median_ms).c_str(), format_time(r.timing.p95_ms).c_str(),
           r.timing.cv_pct, gbs, r.pct_of_peak_bandwidth > 0.0 ? r.pct_of_peak_bandwidth : 0.0,
           speedup, regs, check);
  } else {
    char pct_ref[16];
    if (r.pct_of_reference > 0.0) {
      std::snprintf(pct_ref, sizeof(pct_ref), "%.1f%%", r.pct_of_reference);
    } else {
      std::snprintf(pct_ref, sizeof(pct_ref), "-");
    }
    printf("  %-26s %9s %9s %5.1f%% %11.1f %7.1f%% %9s %9s %6s %7s\n", r.id.c_str(),
           format_time(r.timing.median_ms).c_str(), format_time(r.timing.p95_ms).c_str(),
           r.timing.cv_pct, gflops, r.pct_of_peak_compute, pct_ref, speedup, regs, check);
  }
}

}  // namespace ws
