#!/usr/bin/env python3
"""Render the benchmark results as SVG charts.

Every figure in this repository is generated from ``results/results.json`` by this
script. Nothing is drawn by hand and no number is transcribed, so a chart can
never drift away from the measurement it claims to show - re-run the benchmark,
re-run this, and the documentation is correct by construction.

Each chart is emitted twice, for light and dark backgrounds, so the README reads
correctly in either GitHub theme via a ``<picture>`` element.

Palette note: the categorical colors below are the validated defaults for
colour-vision deficiency separation and contrast against their surface. Charts
that need only two colours use the first two slots; scatter plots are capped at
two categories for the same reason.

usage:
    python tools/plot.py results/results.json --outdir docs/charts
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

try:
    import matplotlib
except ImportError:  # pragma: no cover - environment guidance, not logic
    sys.exit("matplotlib is required: python -m pip install matplotlib")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Deterministic output. By default matplotlib stamps the current date into every
# SVG's metadata and derives element ids from object identity, so regenerating an
# unchanged chart still produces a diff. Both are pinned here: re-running this
# script without changing the data leaves the committed charts byte-identical, and
# git history shows only real changes.
matplotlib.rcParams["svg.hashsalt"] = "warpsmith"
SAVE_METADATA = {"Date": None}

# --- Theme parameters ------------------------------------------------------

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2de",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        # Ordinal blue ramp, lightest step still above 2:1 against the surface.
        "ramp": ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                 "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
        "ceiling": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "ramp": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                 "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95"],
        "ceiling": "#c3c2b7",
    },
}


def apply_theme(theme: dict[str, Any]) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["text"],
        "axes.labelcolor": theme["muted"],
        "axes.edgecolor": theme["grid"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "axes.titlecolor": theme["text"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def style_axes(ax, theme: dict[str, Any], *, grid_axis: str = "y") -> None:
    """Recessive grid and axes: the data should be the only assertive thing."""
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, linewidth=0.8, color=theme["grid"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])


# --- Data access -----------------------------------------------------------


class Results:
    def __init__(self, payload: dict[str, Any]):
        self.device = payload["device"]
        self.records = payload["records"]

    def suite(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["suite"] == name]

    def at_size(self, name: str, size: int) -> list[dict[str, Any]]:
        return sorted((r for r in self.suite(name) if r["size"] == size),
                      key=lambda r: r["stage"])

    def sizes(self, name: str) -> list[int]:
        return sorted({r["size"] for r in self.suite(name)})

    @property
    def peak_fp32(self) -> float:
        return self.device["peak_fp32_gflops"]

    @property
    def peak_bw(self) -> float:
        return self.device["peak_bandwidth_gbs"]

    def achieved_bw(self) -> float:
        mem = self.suite("memory")
        return max((r["gbytes_per_s"] for r in mem), default=self.peak_bw)

    def gpu_label(self) -> str:
        return f"{self.device['name']} ({self.device['arch']})"


def short_label(record: dict[str, Any]) -> str:
    """`05_blocktile_2d` -> `blocktile 2d`, which reads better on an axis."""
    text = record["id"]
    if len(text) > 3 and text[:2].isdigit() and text[2] == "_":
        text = text[3:]
    return text.replace("_", " ")


# --- Charts ----------------------------------------------------------------


def chart_sgemm_progression(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """The headline: what each optimization is worth, at the largest size."""
    size = max(res.sizes("sgemm"))
    rows = [r for r in res.at_size("sgemm", size)]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [short_label(r) for r in rows]
    values = [r["gflops"] for r in rows]

    ramp = theme["ramp"]
    colors = []
    for r in rows:
        if "cublas" in r["id"]:
            colors.append(theme["series"][1])          # Vendor library: its own identity.
        else:
            # Ordinal shade by stage, so the eye reads the progression.
            idx = min(len(ramp) - 1, 2 + r["stage"])
            colors.append(ramp[idx])

    bars = ax.bar(labels, values, color=colors, width=0.66, linewidth=0)
    for bar in bars:
        bar.set_capstyle("round")

    ax.axhline(res.peak_fp32, color=theme["ceiling"], linestyle=(0, (5, 4)), linewidth=1.4)
    ax.text(len(rows) - 0.45, res.peak_fp32, f"  theoretical FP32 peak {res.peak_fp32/1000:.2f} TFLOP/s",
            va="bottom", ha="right", color=theme["muted"], fontsize=9)

    for bar, value in zip(bars, values):
        ax.annotate(f"{value:,.0f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=theme["text"])

    ax.set_ylabel("GFLOP/s (higher is better)")
    ax.set_title(f"SGEMM: nine stages of optimization, M=N=K={size}")
    ax.set_ylim(0, max(values + [res.peak_fp32]) * 1.14)
    ax.tick_params(axis="x", rotation=28)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    style_axes(ax, theme)

    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=8, color=theme["ramp"][6],
               label="hand-written kernel (this repo)"),
        Line2D([], [], marker="s", linestyle="none", markersize=8, color=theme["series"][1],
               label="cuBLAS (vendor library)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, labelcolor=theme["muted"])
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_sgemm_scaling(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """Throughput against problem size: which stages actually scale."""
    wanted = ["01_naive", "03_smem_tiled", "05_blocktile_2d", "07_warptiled", "00_cublas_fp32"]
    sizes = res.sizes("sgemm")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for i, kernel_id in enumerate(wanted):
        xs, ys = [], []
        for size in sizes:
            match = [r for r in res.at_size("sgemm", size) if r["id"] == kernel_id]
            if match:
                xs.append(size)
                ys.append(match[0]["gflops"])
        if not xs:
            continue
        color = theme["series"][i % len(theme["series"])]
        label = "cuBLAS FP32" if kernel_id == "00_cublas_fp32" else short_label({"id": kernel_id})
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=color, label=label)

    ax.axhline(res.peak_fp32, color=theme["ceiling"], linestyle=(0, (5, 4)), linewidth=1.2)
    ax.text(sizes[0], res.peak_fp32, " theoretical FP32 peak", va="bottom", ha="left",
            color=theme["muted"], fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("matrix dimension (M = N = K)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("SGEMM throughput against problem size")
    ax.legend(fontsize=9, labelcolor=theme["muted"], loc="center left")
    style_axes(ax, theme)
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_cost_of_speed(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """Throughput and register pressure side by side.

    Two panels rather than two y-axes on one plot: a dual-axis chart invites the
    reader to see a correlation in whatever way the arbitrary scaling suggests.
    """
    size = max(res.sizes("sgemm"))
    rows = [r for r in res.at_size("sgemm", size)
            if r["regs_per_thread"] > 0]
    if not rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    labels = [short_label(r) for r in rows]
    ramp = theme["ramp"]
    colors = [ramp[min(len(ramp) - 1, 2 + r["stage"])] for r in rows]

    ax1.bar(labels, [r["gflops"] for r in rows], color=colors, width=0.66, linewidth=0)
    ax1.set_ylabel("GFLOP/s")
    ax1.set_title("Throughput")

    ax2.bar(labels, [r["regs_per_thread"] for r in rows], color=colors, width=0.66, linewidth=0)
    ax2.set_ylabel("registers per thread")
    ax2.set_title("Register pressure")
    ax2.axhline(255, color=theme["ceiling"], linestyle=(0, (5, 4)), linewidth=1.2)
    ax2.text(len(rows) - 0.4, 255, " hardware limit 255 ", va="bottom", ha="right",
             color=theme["muted"], fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=32)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        style_axes(ax, theme)

    fig.suptitle(f"The cost of speed: registers bought the throughput (M=N=K={size})",
                 color=theme["text"], fontsize=12, fontweight="600")
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_roofline(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """Roofline: arithmetic intensity against achieved throughput.

    Only kernels with an unambiguous FLOP count appear (GEMM and attention). The
    bandwidth-bound suites are covered by their own chart, where GB/s is the
    meaningful axis.
    """
    points = []
    for r in res.records:
        if r["flops"] <= 0 or r["bytes"] <= 0 or r["suite"] == "sgemm_tune":
            continue
        intensity = r["flops"] / r["bytes"]
        points.append((intensity, r["gflops"], r, r["suite"]))
    if not points:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    xs = [p[0] for p in points]
    x_lo = max(0.25, min(xs) / 3)
    x_hi = max(xs) * 3
    grid = [x_lo * (x_hi / x_lo) ** (i / 200) for i in range(201)]

    peak = res.peak_fp32
    bw_theory = res.peak_bw
    bw_measured = res.achieved_bw()

    ax.plot(grid, [min(peak, bw_theory * x) for x in grid], color=theme["ceiling"],
            linewidth=1.6, linestyle=(0, (5, 4)))
    ax.plot(grid, [min(peak, bw_measured * x) for x in grid], color=theme["ceiling"],
            linewidth=1.2, linestyle=(0, (1, 3)))

    ridge = peak / bw_theory
    ax.text(grid[3], bw_theory * grid[3] * 1.08, f"theoretical DRAM roof {bw_theory:.0f} GB/s",
            color=theme["muted"], fontsize=9, rotation=34, va="bottom")
    ax.text(grid[30], bw_measured * grid[30] * 0.62, f"measured {bw_measured:.0f} GB/s",
            color=theme["muted"], fontsize=8.5, rotation=34, va="top")
    ax.text(x_hi / 1.3, peak * 1.06, f"FP32 compute roof {peak/1000:.2f} TFLOP/s",
            color=theme["muted"], fontsize=9, ha="right")
    ax.axvline(ridge, color=theme["grid"], linewidth=1.0)
    ax.text(ridge, min(p[1] for p in points) / 2.4, f" ridge {ridge:.0f} FLOP/byte",
            color=theme["muted"], fontsize=9, rotation=90, va="bottom")

    # Two categories only, which keeps the all-pairs colour separation valid.
    for suite, color, marker in (("sgemm", theme["series"][0], "o"),
                                 ("attention", theme["series"][1], "^")):
        sel = [p for p in points if p[3] == suite]
        if not sel:
            continue
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=64, marker=marker,
                   color=color, edgecolor=theme["surface"], linewidth=1.2, zorder=3,
                   label={"sgemm": "SGEMM stages", "attention": "attention"}[suite])

    # Label only the extremes: the best and worst of each family.
    for suite in ("sgemm", "attention"):
        sel = sorted((p for p in points if p[3] == suite), key=lambda p: p[1])
        for p in {0, len(sel) - 1} if len(sel) > 1 else {0}:
            if not sel:
                continue
            x, y, rec, _ = sel[p]
            ax.annotate(short_label(rec), (x, y), textcoords="offset points",
                        xytext=(9, -3), fontsize=8.5, color=theme["text"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Roofline: {res.gpu_label()}")
    ax.legend(fontsize=9, labelcolor=theme["muted"], loc="lower right")
    style_axes(ax, theme, grid_axis="both")
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_bandwidth(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """How close each bandwidth-bound kernel gets to the achievable roof."""
    suites = [("memory", "Streaming"), ("reduce", "Reduction"), ("transpose", "Transpose"),
              ("softmax", "Softmax"), ("rmsnorm", "RMSNorm")]

    labels: list[str] = []
    values: list[float] = []
    group_edges: list[tuple[str, int, int]] = []
    for suite, title in suites:
        rows = sorted(res.suite(suite), key=lambda r: r["stage"])
        rows = [r for r in rows if r["gbytes_per_s"] > 0]
        if not rows:
            continue
        start = len(labels)
        for r in rows:
            labels.append(f"{suite}: {short_label(r)}")
            values.append(100.0 * r["gbytes_per_s"] / res.peak_bw)
        group_edges.append((title, start, len(labels)))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9.0, 0.34 * len(labels) + 1.9))
    ramp = theme["ramp"]
    colors = []
    for i, (_, start, end) in enumerate(group_edges):
        shade = ramp[min(len(ramp) - 1, 3 + i)]
        colors.extend([shade] * (end - start))

    positions = list(range(len(labels)))[::-1]
    ax.barh(positions, values, color=colors, height=0.68, linewidth=0)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.5)

    achieved = 100.0 * res.achieved_bw() / res.peak_bw
    ax.axvline(achieved, color=theme["ceiling"], linestyle=(0, (5, 4)), linewidth=1.4)
    ax.text(achieved, len(labels) - 0.2, f" measured ceiling {achieved:.0f}%",
            color=theme["muted"], fontsize=9, va="top")

    for pos, value in zip(positions, values):
        ax.annotate(f"{value:.0f}%", (value, pos), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8, color=theme["text"])

    ax.set_xlim(0, 108)
    ax.set_xlabel(f"percent of theoretical DRAM bandwidth ({res.peak_bw:.0f} GB/s)")
    ax.set_title("Bandwidth-bound kernels against the memory roof")
    style_axes(ax, theme, grid_axis="x")
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_attention(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """Materialized versus fused attention, full and causal."""
    rows = res.suite("attention")
    if not rows:
        return
    sizes = sorted({r["size"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    panels = [("full attention", "00_materialized", "01_flash"),
              ("causal attention", "02_materialized_causal", "03_flash_causal")]

    width = 0.34
    for ax, (title, base_id, flash_id) in zip(axes, panels):
        base = [next((r["median_ms"] for r in rows if r["size"] == s and r["id"] == base_id), 0.0)
                for s in sizes]
        flash = [next((r["median_ms"] for r in rows if r["size"] == s and r["id"] == flash_id), 0.0)
                 for s in sizes]
        xs = list(range(len(sizes)))
        ax.bar([x - width / 2 - 0.01 for x in xs], base, width=width,
               color=theme["series"][1], linewidth=0, label="materialized (cuBLAS + softmax)")
        ax.bar([x + width / 2 + 0.01 for x in xs], flash, width=width,
               color=theme["series"][0], linewidth=0, label="fused flash (this repo)")

        for x, (b, f) in zip(xs, zip(base, flash)):
            if b > 0 and f > 0:
                ax.annotate(f"{b / f:.2f}x faster", (x + width / 2 + 0.01, f),
                            textcoords="offset points", xytext=(0, 4), ha="center",
                            fontsize=8.5, color=theme["text"])

        ax.set_xticks(xs)
        ax.set_xticklabels([f"S={s}" for s in sizes])
        ax.set_title(title)
        style_axes(ax, theme)

    axes[0].set_ylabel("milliseconds (lower is better)")
    axes[0].legend(fontsize=9, labelcolor=theme["muted"], loc="upper left")
    fig.suptitle("Attention: never materializing the S x S matrix",
                 color=theme["text"], fontsize=12, fontweight="600")
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


def chart_autotune(res: Results, theme: dict[str, Any], path: pathlib.Path) -> None:
    """The tile-geometry search space, measured."""
    rows = [r for r in res.suite("sgemm_tune") if r["gflops"] > 0]
    if not rows:
        return
    rows.sort(key=lambda r: r["gflops"])
    best = max(r["gflops"] for r in rows)

    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(rows) + 1.6))
    positions = list(range(len(rows)))
    colors = [theme["series"][1] if r["gflops"] == best else theme["ramp"][5] for r in rows]
    ax.barh(positions, [r["gflops"] for r in rows], color=colors, height=0.66, linewidth=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([r["id"] for r in rows], fontsize=8.5, family="monospace")

    for pos, r in zip(positions, rows):
        suffix = "  <- best" if r["gflops"] == best else ""
        ax.annotate(f"{r['gflops']:,.0f}{suffix}", (r["gflops"], pos),
                    textcoords="offset points", xytext=(4, 0), va="center",
                    fontsize=8, color=theme["text"])

    size = rows[0]["size"]
    ax.set_xlabel("GFLOP/s")
    ax.set_xlim(0, best * 1.22)
    ax.set_title(f"Autotuning the tile geometry (BMxBNxBK / TMxTN, M=N=K={size})")
    style_axes(ax, theme, grid_axis="x")
    fig.savefig(path, bbox_inches="tight",
                metadata=SAVE_METADATA if path.suffix == ".svg" else None)
    plt.close(fig)


CHARTS = {
    "sgemm-progression": chart_sgemm_progression,
    "sgemm-scaling": chart_sgemm_scaling,
    "cost-of-speed": chart_cost_of_speed,
    "roofline": chart_roofline,
    "bandwidth": chart_bandwidth,
    "attention": chart_attention,
    "autotune": chart_autotune,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=pathlib.Path, help="results JSON from warpsmith_bench")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("docs/charts"))
    parser.add_argument("--only", action="append", choices=sorted(CHARTS),
                        help="render only the named chart (repeatable)")
    parser.add_argument("--format", default="svg", choices=["svg", "png"],
                        help="output format; png is for previewing, svg is what ships")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    res = Results(payload)
    args.outdir.mkdir(parents=True, exist_ok=True)

    selected = args.only or sorted(CHARTS)
    written = 0
    for name in selected:
        fn = CHARTS[name]
        for mode, theme in THEMES.items():
            apply_theme(theme)
            suffix = "" if mode == "light" else "-dark"
            path = args.outdir / f"{name}{suffix}.{args.format}"
            fn(res, theme, path)
            if path.exists():
                written += 1
                print(f"  {path}")
    print(f"wrote {written} charts to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
