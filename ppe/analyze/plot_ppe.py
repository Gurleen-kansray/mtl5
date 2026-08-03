#!/usr/bin/env python3
"""Plot the PPE GEMM progression: throughput vs N, one figure per type family.

    ./plot_ppe.py data/ppe_gemm.csv --outdir data

Produces:
    ppe-gemm-integer.png   int8 / int16 / int32 / int64
    ppe-gemm-float.png     fp16 / fp32 / fp64
    ppe-gemm-progression.png   best-per-kernel across both families

Each figure is a grid of subplots, one per type, with a curve per kernel version
so the progression is readable as "did this step help, and at which N".

Standard library plus matplotlib. Analysis tooling, not part of any build.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

INT_TYPES = ["int8", "int16", "int32", "int64"]
FLT_TYPES = ["fp16", "fp32", "fp64"]

# The progression in order, with a stable colour per step so the same kernel is
# the same colour in every subplot and every figure.
KERNELS = [
    ("v0_naive",   "#9e9e9e", "o", "v0 naive ijk"),
    ("v1_ikj",     "#1f77b4", "s", "v1 loop reorder (ikj)"),
    ("v2_blocked", "#2ca02c", "^", "v2 + cache blocking"),
    ("v3_packed",  "#ff7f0e", "D", "v3 + packing"),
    ("v4_regtile", "#d62728", "v", "v4 + register tile"),
    ("v5_micro",   "#9467bd", "P", "v5 + compile-time microkernel"),
]


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "kernel": r["kernel"], "type": r["type"], "n": int(r["n"]),
                "gops": float(r["gops"]), "peak": float(r["peak_gops"]),
                "eff": float(r["efficiency"]), "correct": r["correct"] == "1",
            })
    return rows


def family_figure(rows, types, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = [t for t in types if any(r["type"] == t for r in rows)]
    if not types:
        print(f"  (no data for {title})")
        return

    ncols = min(2, len(types))
    nrows = (len(types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows), squeeze=False)

    for idx, ty in enumerate(types):
        ax = axes[idx // ncols][idx % ncols]
        peak = next((r["peak"] for r in rows if r["type"] == ty), None)
        for kern, colour, marker, label in KERNELS:
            pts = sorted((r["n"], r["gops"]) for r in rows
                         if r["type"] == ty and r["kernel"] == kern)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker=marker, color=colour, label=label, linewidth=1.6, markersize=5)
        if peak:
            ax.axhline(peak, color="black", linestyle=":", linewidth=1.2)
            ax.text(0.99, peak, f" modelled peak {peak:.0f}", ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(), fontsize=7, color="black")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N (square GEMM)")
        ax.set_ylabel("throughput (GOP/s)")
        ax.set_title(ty, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linewidth=0.5)

    # one shared legend, so each subplot keeps its area for data
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    for i in range(len(types), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


def progression_figure(rows, out):
    """Speedup of each step over v0, averaged across N >= 256, per type.

    This is the 'did the engineering work' view: one bar group per type, one bar
    per version, so a step that helps some types and hurts others is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    types = [t for t in INT_TYPES + FLT_TYPES if any(r["type"] == t for r in rows)]
    base = {}
    for ty in types:
        vals = [r["gops"] for r in rows if r["type"] == ty and r["kernel"] == "v0_naive" and r["n"] >= 256]
        base[ty] = sum(vals) / len(vals) if vals else None

    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.13
    x = np.arange(len(types))
    for ki, (kern, colour, _m, label) in enumerate(KERNELS):
        ys = []
        for ty in types:
            vals = [r["gops"] for r in rows if r["type"] == ty and r["kernel"] == kern and r["n"] >= 256]
            ys.append((sum(vals) / len(vals) / base[ty]) if vals and base.get(ty) else 0.0)
        ax.bar(x + (ki - len(KERNELS) / 2 + 0.5) * width, ys, width, label=label, color=colour)

    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(types)
    ax.set_ylabel("speedup over v0 (mean, N >= 256)")
    ax.set_title("GEMM optimization progression — speedup over the naive triple loop",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    rows = load(args.csv)
    bad = [r for r in rows if not r["correct"]]
    if bad:
        raise SystemExit(f"refusing to plot: {len(bad)} row(s) failed verification")

    os.makedirs(args.outdir, exist_ok=True)
    family_figure(rows, INT_TYPES, "GEMM progression — integer types (operand -> int32/int64 accumulator)",
                  os.path.join(args.outdir, "ppe-gemm-integer.png"))
    family_figure(rows, FLT_TYPES, "GEMM progression — floating-point types",
                  os.path.join(args.outdir, "ppe-gemm-float.png"))
    progression_figure(rows, os.path.join(args.outdir, "ppe-gemm-progression.png"))


if __name__ == "__main__":
    main()
