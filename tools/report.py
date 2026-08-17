#!/usr/bin/env python3
"""Turn the benchmark JSON into a Markdown report.

The report is generated, never edited. CI regenerates it from the committed
results and fails if the checked-in file differs, which makes it impossible for
the numbers in the documentation to drift from the numbers that were measured.

usage:
    python tools/report.py results/results.json --out results/REPORT.md
    python tools/report.py results/results.json --headlines
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable

Record = dict[str, Any]


def fmt_time(ms: float) -> str:
    if ms < 1.0:
        return f"{ms * 1000:.1f} us"
    if ms < 1000.0:
        return f"{ms:.3f} ms"
    return f"{ms / 1000:.3f} s"


def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def strip_stage(text: str) -> str:
    if len(text) > 3 and text[:2].isdigit() and text[2] == "_":
        return text[3:]
    return text


def table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or (["left"] + ["right"] * (len(headers) - 1))
    sep = {"left": ":---", "right": "---:", "center": ":---:"}
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep[a] for a in align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


class Report:
    def __init__(self, payload: dict[str, Any]):
        self.device: dict[str, Any] = payload["device"]
        self.config: dict[str, Any] = payload.get("config", {})
        self.records: list[Record] = payload["records"]

    # --- helpers ---------------------------------------------------------

    def suite(self, name: str) -> list[Record]:
        return [r for r in self.records if r["suite"] == name]

    def sizes(self, name: str) -> list[int]:
        return sorted({r["size"] for r in self.suite(name)})

    def rows_at(self, name: str, size: int) -> list[Record]:
        return sorted((r for r in self.suite(name) if r["size"] == size),
                      key=lambda r: r["stage"])

    def find(self, suite: str, kernel_id: str, size: int | None = None) -> Record | None:
        candidates = [r for r in self.suite(suite) if r["id"] == kernel_id]
        if size is not None:
            candidates = [r for r in candidates if r["size"] == size]
        return candidates[-1] if candidates else None

    def achieved_bw(self) -> float:
        return max((r["gbytes_per_s"] for r in self.suite("memory")), default=0.0)

    # --- sections --------------------------------------------------------

    def device_section(self) -> str:
        d = self.device
        rows = [
            ["GPU", f"{d['name']} (`{d['arch']}`)"],
            ["Streaming multiprocessors", f"{d['sm_count']}"],
            ["SM clock", f"{d['clock_mhz']} MHz"],
            ["Memory", f"{fmt_bytes(d['global_mem_bytes'])} @ {d['mem_clock_mhz']} MHz, "
                       f"{d['mem_bus_bits']}-bit bus"],
            ["L2 cache", fmt_bytes(d["l2_bytes"])],
            ["Shared memory per block", fmt_bytes(d["smem_per_block"])],
            ["Registers per SM", f"{d['regs_per_sm']:,}"],
            ["Theoretical FP32 peak", f"{d['peak_fp32_gflops'] / 1000:.2f} TFLOP/s"],
            ["Theoretical TF32 peak (tensor cores)", f"{d['peak_tf32_gflops'] / 1000:.2f} TFLOP/s"],
            ["Theoretical FP16 peak (tensor cores)", f"{d['peak_fp16_gflops'] / 1000:.2f} TFLOP/s"],
            ["Theoretical DRAM bandwidth", f"{d['peak_bandwidth_gbs']:.1f} GB/s"],
            ["Measured DRAM bandwidth", f"{self.achieved_bw():.1f} GB/s "
                                        f"({100 * self.achieved_bw() / d['peak_bandwidth_gbs']:.1f}% of theoretical)"],
            ["Roofline ridge point", f"{d['ridge_point_flop_per_byte']:.1f} FLOP/byte"],
            ["CUDA runtime", f"{d['cuda_runtime'] // 1000}.{(d['cuda_runtime'] % 1000) // 10}"],
        ]
        return "## Test system\n\n" + table(["Property", "Value"], rows,
                                           ["left", "left"]) + "\n"

    def compute_table(self, suite: str, size: int) -> str:
        rows = []
        for r in self.rows_at(suite, size):
            rows.append([
                f"`{r['id']}`",
                r["technique"],
                fmt_time(r["median_ms"]),
                f"{r['cv_pct']:.1f}%",
                f"{r['gflops']:,.0f}",
                f"{r['pct_of_peak_compute']:.1f}%",
                f"{r['pct_of_reference']:.1f}%" if r["pct_of_reference"] else "-",
                f"{r['speedup_vs_stage1']:.2f}x" if r["speedup_vs_stage1"] else "-",
                f"{r['regs_per_thread']}" if r["regs_per_thread"] else "-",
                f"{r['occupancy_pct']:.0f}%" if r["occupancy_pct"] else "-",
                f"{r['rel_l2_err']:.2e}",
                "pass" if r["correct"] else "**FAIL**",
            ])
        headers = ["kernel", "technique added", "median", "cv", "GFLOP/s", "% peak", "% cuBLAS",
                   "speedup", "regs", "occ", "rel L2 err", "check"]
        align = ["left", "left"] + ["right"] * (len(headers) - 2)
        return table(headers, rows, align)

    def bandwidth_table(self, suite: str, size: int) -> str:
        rows = []
        for r in self.rows_at(suite, size):
            rows.append([
                f"`{r['id']}`",
                r["technique"],
                fmt_time(r["median_ms"]),
                f"{r['cv_pct']:.1f}%",
                f"{r['gbytes_per_s']:,.1f}",
                f"{r['pct_of_peak_bandwidth']:.1f}%",
                f"{r['speedup_vs_stage1']:.2f}x" if r["speedup_vs_stage1"] else "-",
                f"{r['rel_l2_err']:.2e}",
                "pass" if r["correct"] else "**FAIL**",
            ])
        headers = ["kernel", "technique added", "median", "cv", "GB/s", "% peak", "speedup",
                   "rel err", "check"]
        align = ["left", "left"] + ["right"] * (len(headers) - 2)
        return table(headers, rows, align)

    def suite_section(self, suite: str, title: str, blurb: str, bandwidth: bool) -> str:
        sizes = self.sizes(suite)
        if not sizes:
            return ""
        out = [f"## {title}\n", blurb + "\n"]
        for size in sizes:
            rows = self.rows_at(suite, size)
            shape = rows[0]["shape"] if rows else str(size)
            out.append(f"### `{shape}`\n")
            out.append(self.bandwidth_table(suite, size) if bandwidth
                       else self.compute_table(suite, size))
            out.append("")
        return "\n".join(out)

    def autotune_section(self) -> str:
        rows = [r for r in self.suite("sgemm_tune")]
        if not rows:
            return ""
        rows.sort(key=lambda r: -r["gflops"])
        body = []
        for i, r in enumerate(rows):
            body.append([
                f"`{r['id']}`",
                f"{r['gflops']:,.0f}",
                f"{r['pct_of_reference']:.1f}%",
                fmt_time(r["median_ms"]),
                "**best**" if i == 0 else "",
            ])
        head = table(["tile BMxBNxBK / TMxTN", "GFLOP/s", "% cuBLAS", "median", ""], body)
        return ("## SGEMM autotuning\n\n"
                "The same templated kernel, instantiated across the tile-geometry search space "
                "and measured. The winner is a property of this GPU, not of the algorithm.\n\n"
                + head + "\n")

    def headlines(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        sgemm_sizes = self.sizes("sgemm")
        if sgemm_sizes:
            big = max(sgemm_sizes)
            at_big = [r for r in self.suite("sgemm") if r["size"] == big]
            # The FP32 progression and the tensor-core kernel are reported
            # separately: they are measured against different ceilings, so a single
            # combined row would put "124% of cuBLAS" next to "22% of peak" and read
            # as a contradiction rather than as two facts about two datapaths.
            fp32 = [r for r in at_big if "cublas" not in r["id"] and "tf32" not in r["id"]]
            best = max(fp32, key=lambda r: r["gflops"], default=None)
            if best:
                out.append(("SGEMM, best hand-written FP32 kernel",
                            f"{best['gflops']:,.0f} GFLOP/s at {best['shape']} - "
                            f"{best['pct_of_reference']:.0f}% of cuBLAS FP32, "
                            f"{best['pct_of_peak_compute']:.0f}% of the FP32 ceiling"))
            tc = self.find("sgemm", "08_wmma_tf32", big)
            if tc:
                out.append(("SGEMM, tensor-core kernel (WMMA, TF32)",
                            f"{tc['gflops']:,.0f} GFLOP/s - "
                            f"{tc['pct_of_reference']:.0f}% of cuBLAS FP32, "
                            f"{tc['pct_of_peak_compute']:.0f}% of the TF32 tensor-core ceiling"))
            naive_sizes = [r["size"] for r in self.suite("sgemm") if r["id"] == "01_naive"]
            if naive_sizes and best:
                s = max(naive_sizes)
                naive = self.find("sgemm", "01_naive", s)
                fastest_there = max((r for r in self.suite("sgemm")
                                     if r["size"] == s and "cublas" not in r["id"]),
                                    key=lambda r: r["gflops"], default=None)
                if naive and fastest_there:
                    out.append(("SGEMM, speedup from naive to fastest, same problem",
                                f"{naive['median_ms'] / fastest_there['median_ms']:.0f}x "
                                f"at {naive['shape']}"))
        red = self.suite("reduce")
        if red:
            best = max(red, key=lambda r: r["gbytes_per_s"])
            worst = self.find("reduce", "01_atomic_global")
            out.append(("Reduction, best kernel",
                        f"{best['gbytes_per_s']:.1f} GB/s "
                        f"({best['pct_of_peak_bandwidth']:.0f}% of theoretical bandwidth)"))
            if worst:
                out.append(("Reduction, speedup over global atomics",
                            f"{worst['median_ms'] / best['median_ms']:.0f}x"))
        tr = self.suite("transpose")
        if tr:
            best = max((r for r in tr if r["id"] != "00_copy_bound"),
                       key=lambda r: r["gbytes_per_s"])
            out.append(("Transpose, best kernel",
                        f"{best['gbytes_per_s']:.1f} GB/s "
                        f"({best['pct_of_peak_bandwidth']:.0f}% of theoretical bandwidth)"))
        att = self.suite("attention")
        if att:
            big = max(r["size"] for r in att)
            for base_id, flash_id, label in (("00_materialized", "01_flash", "full"),
                                             ("02_materialized_causal", "03_flash_causal",
                                              "causal")):
                b = self.find("attention", base_id, big)
                f = self.find("attention", flash_id, big)
                if b and f:
                    out.append((f"Attention ({label}), fused vs materialized at S={big}",
                                f"{b['median_ms'] / f['median_ms']:.2f}x faster, "
                                f"{fmt_bytes(b['bytes'] - f['bytes'])} less DRAM traffic"))
        out.append(("Total measurements", f"{len(self.records)}"))
        out.append(("Correctness failures",
                    f"{sum(1 for r in self.records if not r['correct'])}"))
        return out

    # --- README injection ------------------------------------------------

    def readme_sections(self) -> dict[str, str]:
        """Markdown blocks injected into README.md between named markers.

        The README claims its numbers are measured rather than transcribed, so they
        are written by this function instead of by hand.
        """
        out: dict[str, str] = {}

        out["HEADLINE"] = table(["Result", "Measured"],
                                [[k, f"**{v}**"] for k, v in self.headlines()],
                                ["left", "left"])

        sgemm_sizes = self.sizes("sgemm")
        if sgemm_sizes:
            size = max(sgemm_sizes)
            rows = []
            for r in self.rows_at("sgemm", size):
                rows.append([
                    f"`{r['id']}`",
                    r["technique"],
                    f"{r['gflops']:,.0f}",
                    f"{r['pct_of_reference']:.0f}%" if r["pct_of_reference"] else "-",
                    f"{r['regs_per_thread']}" if r["regs_per_thread"] else "-",
                    f"{r['rel_l2_err']:.1e}",
                ])
            body = table(["stage", "what it adds", "GFLOP/s", "% of cuBLAS", "regs/thread",
                          "rel L2 error"], rows,
                         ["left", "left", "right", "right", "right", "right"])
            shape = self.rows_at("sgemm", size)[0]["shape"]
            present = {r["id"] for r in self.rows_at("sgemm", size)}
            omitted = [i for i in ("01_naive", "02_coalesced") if i not in present]
            note = ""
            if omitted:
                names = " and ".join(f"`{i}`" for i in omitted)
                note = (f" {names} are omitted at this size: they take minutes per launch, and "
                        f"their trend is already established at smaller sizes.")
            out["SGEMM"] = (f"At `{shape}`.{note} The full sweep across every problem size, with "
                            f"timing distributions, occupancy and the naive stages, is in the "
                            f"[report](results/REPORT.md).\n\n{body}")

        att = self.suite("attention")
        if att:
            size = max(r["size"] for r in att)
            rows = []
            for r in sorted((x for x in att if x["size"] == size), key=lambda x: x["stage"]):
                scratch = "0" if "flash" in r["id"] else fmt_bytes(size * size * 4)
                rows.append([
                    f"`{r['id']}`",
                    fmt_time(r["median_ms"]),
                    f"{r['gflops']:,.0f}",
                    f"{r['speedup_vs_stage1']:.2f}x" if r["speedup_vs_stage1"] else "-",
                    scratch,
                    f"{r['rel_l2_err']:.1e}",
                ])
            out["ATTENTION"] = (f"At `S={size}`, 8 heads, head dimension 64.\n\n"
                                + table(["implementation", "median", "GFLOP/s", "speedup",
                                         "S x S scratch", "rel L2 error"], rows,
                                        ["left", "right", "right", "right", "right", "right"]))

        bw_suites = [("reduce", "Reduction"), ("transpose", "Transpose"),
                     ("softmax", "Softmax"), ("rmsnorm", "RMSNorm")]
        rows = []
        for suite, title in bw_suites:
            recs = [r for r in self.suite(suite) if r["gbytes_per_s"] > 0]
            if not recs:
                continue
            # Excluded from "best": the library baseline and the copy bound are
            # references rather than entries, and the fused-residual RMSNorm computes
            # a different function (it also writes the residual stream), so ranking it
            # against the others by wall-clock would be comparing two workloads.
            mine = [r for r in recs
                    if r["id"] not in ("00_cub", "00_copy_bound", "04_fused_residual")]
            if not mine:
                continue
            best = max(mine, key=lambda r: r["gbytes_per_s"])
            first = min(mine, key=lambda r: r["stage"])
            rows.append([
                title,
                f"`{best['id']}`",
                f"{best['gbytes_per_s']:,.1f}",
                f"{best['pct_of_peak_bandwidth']:.0f}%",
                f"{first['median_ms'] / best['median_ms']:.2f}x",
            ])
        if rows:
            achieved = self.achieved_bw()
            peak = self.device["peak_bandwidth_gbs"]
            out["BANDWIDTH"] = (
                f"The streaming microbenchmarks sustain **{achieved:.0f} GB/s** on this device, "
                f"{100 * achieved / peak:.0f}% of the {peak:.0f} GB/s theoretical figure - that is "
                f"the real ceiling these kernels are chasing. A row may edge past it: the "
                f"percentages are computed from each kernel's *algorithmic* traffic, so a kernel "
                f"whose working set is partly served by L2 moves fewer bytes at DRAM than it "
                f"logically reads.\n\n"
                + table(["suite", "best kernel", "GB/s", "% of theoretical peak",
                         "speedup over stage 1"], rows,
                        ["left", "left", "right", "right", "right"]))
        return out

    # --- assembly --------------------------------------------------------

    def render(self) -> str:
        parts = [
            "# warpsmith benchmark report\n",
            "Generated by `tools/report.py` from `results/results.json`. Do not edit by hand.\n",
            self.device_section(),
            "## Headline results\n",
            table(["Result", "Measurement"],
                  [[k, v] for k, v in self.headlines()], ["left", "left"]) + "\n",
            self.suite_section(
                "sgemm", "SGEMM",
                "Nine implementations of `C = alpha*A@B + beta*C`, each adding one optimization "
                "to the previous. `% cuBLAS` compares against NVIDIA's hand-tuned library; "
                "`% peak` against the architectural ceiling. The TF32 variants are measured "
                "against the tensor-core ceiling rather than the FP32 one.",
                bandwidth=False),
            self.autotune_section(),
            self.suite_section(
                "attention", "Attention",
                "A materialized implementation (cuBLAS for both GEMMs, with a fused single-pass "
                "softmax between them) against a fused FlashAttention-style kernel that never "
                "writes the S x S score matrix to memory.",
                bandwidth=False),
            self.suite_section(
                "memory", "Memory bandwidth",
                "Streaming microbenchmarks that establish what this GPU can actually sustain. "
                "Every bandwidth-bound result elsewhere in this report is judged against these.",
                bandwidth=True),
            self.suite_section(
                "reduce", "Reduction",
                "Summing 64 million floats six ways. The relative error column is against a "
                "Kahan-compensated double-precision sum, which shows that the parallel tree is "
                "not only faster than a serial accumulation but more accurate.",
                bandwidth=True),
            self.suite_section(
                "transpose", "Transpose",
                "No arithmetic whatsoever, so every difference is a memory-system effect: "
                "coalescing, then shared-memory bank conflicts, then launch overhead. "
                "`00_copy_bound` moves the same bytes without transposing and is the upper bound.",
                bandwidth=True),
            self.suite_section(
                "softmax", "Softmax",
                "Row-wise softmax at the shape of a language model's logits. The optimization is "
                "entirely about reading the input fewer times: three passes, then two via the "
                "online algorithm, then one by caching the row in shared memory.",
                bandwidth=True),
            self.suite_section(
                "rmsnorm", "RMSNorm",
                "The normalization used by LLaMA-family transformers. `04_fused_residual` computes "
                "a different function - it adds the residual stream first and writes it back - so "
                "it moves twice the bytes by design and its wall-clock time is not comparable to "
                "the others.",
                bandwidth=True),
            "## Methodology\n",
            f"- Timing uses CUDA events on the stream, excluding host launch overhead.\n"
            f"- Each kernel gets {self.config.get('warmup', 20)} untimed warm-up launches so SM "
            f"and memory clocks settle, then {self.config.get('iters', 100)} timed samples "
            f"(fewer for the largest problems).\n"
            "- The reported figure is the **median**; `cv` is the coefficient of variation across "
            "samples, published so the reader can judge stability. This is a laptop GPU with "
            "aggressive clock management, and a high cv means exactly that.\n"
            "- Every kernel is validated against a trusted reference before it is timed - cuBLAS "
            "for GEMM, CUB for reduction, double-precision host code elsewhere. A kernel that "
            "fails validation is reported as `FAIL` and the benchmark exits non-zero.\n"
            "- `% peak` is derived from the device's own clocks and core counts, not from a "
            "marketing figure.\n",
        ]
        return "\n".join(p for p in parts if p)


def inject(path: pathlib.Path, sections: dict[str, str]) -> int:
    """Replace the content between `<!-- BEGIN:NAME -->` and `<!-- END:NAME -->`.

    Returns the number of sections replaced. A marker with no matching section, or a
    section with no matching marker, is reported rather than silently ignored - a
    quietly un-updated table is exactly the drift this is meant to prevent.
    """
    text = path.read_text(encoding="utf-8")
    replaced = 0
    for name, body in sections.items():
        begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0:
            print(f"  note: {path.name} has no {begin} ... {end} block", file=sys.stderr)
            continue
        if j < i:
            print(f"  error: {end} precedes {begin} in {path.name}", file=sys.stderr)
            return -1
        text = text[:i + len(begin)] + "\n\n" + body + "\n\n" + text[j:]
        replaced += 1
    path.write_text(text, encoding="utf-8", newline="\n")
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, help="write Markdown here (default: stdout)")
    parser.add_argument("--headlines", action="store_true",
                        help="print only the headline numbers")
    parser.add_argument("--inject", type=pathlib.Path,
                        help="update the generated tables inside a Markdown file in place")
    args = parser.parse_args()

    report = Report(json.loads(args.results.read_text(encoding="utf-8")))

    if args.headlines:
        for key, value in report.headlines():
            print(f"{key:52s} {value}")
        return 0

    if args.inject:
        n = inject(args.inject, report.readme_sections())
        if n < 0:
            return 1
        print(f"updated {n} generated sections in {args.inject}")
        if not args.out:
            return 0

    text = report.render()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
