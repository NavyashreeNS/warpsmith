#!/usr/bin/env python3
"""Check that a results file is well formed and internally consistent.

This is a guard against a subtler failure than a crash: a benchmark that runs,
writes plausible-looking JSON, and reports numbers that cannot be true. Each check
below corresponds to a way that has actually gone wrong in benchmark harnesses.

usage:
    python tools/validate_results.py results/results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REQUIRED_RECORD_FIELDS = [
    "suite", "kernel", "id", "stage", "technique", "size", "shape",
    "median_ms", "min_ms", "mean_ms", "p95_ms", "cv_pct", "iters",
    "flops", "bytes", "gflops", "gbytes_per_s",
    "pct_of_peak_compute", "pct_of_peak_bandwidth",
    "rel_l2_err", "max_abs_err", "correct",
]

REQUIRED_DEVICE_FIELDS = [
    "name", "arch", "sm_count", "clock_mhz", "peak_fp32_gflops",
    "peak_bandwidth_gbs", "ridge_point_flop_per_byte",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=pathlib.Path)
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.results.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema") != "warpsmith/results/1":
        errors.append(f"unexpected schema: {payload.get('schema')!r}")

    device = payload.get("device", {})
    for field in REQUIRED_DEVICE_FIELDS:
        if field not in device:
            errors.append(f"device is missing {field!r}")

    records = payload.get("records", [])
    if not records:
        errors.append("no records")

    peak_fp32 = device.get("peak_fp32_gflops", 0.0)
    peak_bw = device.get("peak_bandwidth_gbs", 0.0)

    for i, r in enumerate(records):
        where = f"record {i} ({r.get('suite')}/{r.get('id')})"
        for field in REQUIRED_RECORD_FIELDS:
            if field not in r:
                errors.append(f"{where} is missing {field!r}")
                continue

        if r.get("median_ms", 0) <= 0:
            errors.append(f"{where} has a non-positive median time")
        # Ordering that any correct summary must satisfy.
        if not (r.get("min_ms", 0) <= r.get("median_ms", 0) <= r.get("p95_ms", 0) + 1e-9):
            errors.append(f"{where} violates min <= median <= p95")
        if r.get("iters", 0) < 1:
            errors.append(f"{where} reports fewer than one iteration")
        if not r.get("correct", False):
            errors.append(f"{where} failed its correctness check")

        # A kernel cannot exceed the arithmetic ceiling of the hardware. Tensor
        # cores have their own, higher ceiling, so TF32 kernels are exempt from
        # the FP32 comparison.
        if peak_fp32 and "tf32" not in r.get("id", "") and r.get("gflops", 0) > peak_fp32 * 1.02:
            errors.append(f"{where} claims {r['gflops']:.0f} GFLOP/s above the "
                          f"{peak_fp32:.0f} GFLOP/s FP32 ceiling")

        # Exceeding the DRAM roof is legal when a kernel's inputs fit in cache, so
        # this is a warning rather than an error - but it should be a deliberate,
        # explainable result, not a surprise.
        if peak_bw and r.get("gbytes_per_s", 0) > peak_bw * 1.05:
            warnings.append(f"{where} exceeds the DRAM roof "
                            f"({r['gbytes_per_s']:.0f} > {peak_bw:.0f} GB/s) - cache reuse?")

        if r.get("cv_pct", 0) > 25.0:
            warnings.append(f"{where} is unstable: cv {r['cv_pct']:.1f}%")

    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error:   {e}", file=sys.stderr)

    suites = sorted({r["suite"] for r in records})
    print(f"\n{len(records)} records across {len(suites)} suites: {', '.join(suites)}")
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
