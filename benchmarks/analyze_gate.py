#!/usr/bin/env python3
"""Analyze the native-fast benchmark against OpenBLAS -- the epic #82 gate.

Reads bench_all CSVs (columns: operation,backend,size,...,gflops,...) and, for
each operation and size, reports the native-fast GFLOP/s as a **percentage of
OpenBLAS** and, optionally, as a **percentage of FMA peak** (pass --peak-gflops
for your machine: cores_used * freq_GHz * fma_units * lanes * 2).

It also doubles as the optional performance guard (`--gate`): exit non-zero if
the MEDIAN of native-fast/OpenBLAS across the gated sizes falls below
`--threshold`, or if any individual size falls below the harder `--floor`.
Gating on the median rather than on every size is deliberate -- per-size ratios
move by several points between runs while the median is reproducible to ~0.2
points, so the old every-size rule let noise decide the verdict (#327).

This is deliberately NOT wired into the per-push CI -- shared CI runners make
absolute perf gates flaky -- but it is handy locally and in an opt-in workflow.

Examples:
    # Comparison table (% of OpenBLAS), all ops
    ./analyze_gate.py data/blas_sweep_native-fast.csv data/blas_sweep_openblas.csv

    # Add % of FMA peak (e.g. 1 P-core, 4.0 GHz, 2 FMA units, 4 fp64 lanes)
    ./analyze_gate.py data/blas_sweep_*.csv --peak-gflops 64

    # Gate: fail if the MEDIAN GEMM ratio < 80% of OpenBLAS for N >= 256,
    # or if any single size drops below the 70% floor
    ./analyze_gate.py data/blas_sweep_native-fast.csv data/blas_sweep_openblas.csv \\
        --gate --op gemm --threshold 0.80 --floor 0.70 --min-size 256

    # Compare against BLIS instead of OpenBLAS (table or gate)
    ./analyze_gate.py data/blas_sweep_*.csv --reference blis

Standard library only (no pandas). Benchmark tooling, not part of the library.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict


def load(paths):
    """Return {operation: {backend: {size: gflops}}} merged across CSV files."""
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    op, backend = row["operation"], row["backend"]
                    size = int(row["size"])
                    g = row.get("gflops", "")
                    if g == "":
                        continue
                    data[op][backend][size] = float(g)
        except OSError as exc:
            sys.exit(f"error: cannot read {path}: {exc}")
        except (KeyError, ValueError) as exc:
            sys.exit(f"error: {path}: malformed CSV ({exc})")
    return data


def pick(backends, *candidates):
    """First backend label present, matched case-insensitively."""
    low = {b.lower(): b for b in backends}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def print_table(data, peak, reference):
    for op in sorted(data):
        backends = data[op]
        nf = pick(backends, "native-fast", "native_fast")
        ref = pick(backends, reference, "openblas", "blas")
        ref_name = ref if ref else reference
        if nf is None:
            continue
        print(f"\n== {op} ==")
        hdr = f"{'N':>6} {'native-fast':>12}"
        if ref:
            hdr += f" {ref_name:>12} {('% ' + ref_name):>11}"
        if peak:
            hdr += f" {'% FMA peak':>11}"
        print(hdr)
        for n in sorted(backends[nf]):
            g = backends[nf][n]
            line = f"{n:>6} {g:>12.2f}"
            if ref and n in backends[ref] and backends[ref][n] > 0:
                line += f" {backends[ref][n]:>12.2f} {100.0 * g / backends[ref][n]:>10.1f}%"
            elif ref:
                line += f" {'--':>12} {'--':>11}"
            if peak:
                line += f" {100.0 * g / peak:>10.1f}%"
            print(line)


def run_gate(data, op, threshold, min_size, reference, floor):
    """Gate on the MEDIAN of the per-size ratios, with a per-size floor.

    Rationale (#327). The original rule failed if any single size fell below
    the threshold -- i.e. it gated on the minimum, the noisiest statistic
    available. Two runs of the identical protocol on the same idle machine
    produced these per-size ratios for gemm at N >= 256:

        statistic                run A    run B    swing
        median of ratios         82.3%    82.5%     0.2 pt
        mean                     82.1%    82.2%     0.1 pt
        min  (the old rule)      76.3%    78.7%     2.4 pt
        worst single size          --       --      6.2 pt

    They failed at disjoint sets of sizes, so the verdict was decided by noise.
    The median is reproducible to ~0.2 points, roughly 30x more stable, and it
    is also what the epic actually claims -- "within 10-20% of OpenBLAS" is an
    aggregate statement, not a per-size one.

    Raising the iteration count does not help: within-run stddev is already
    only 0.3-2.2% of the median at 10 iterations. The variance is BETWEEN runs
    (turbo/thermal state, allocation, page layout), not within them.

    The floor keeps the rule honest. A genuine cliff at one size -- a blocking
    pathology, a bad edge case -- must still fail, so no individual size may
    drop below `floor`, set well under the threshold so ordinary +-6 point
    noise cannot trip it.
    """
    backends = data.get(op, {})
    nf = pick(backends, "native-fast", "native_fast")
    ref = pick(backends, reference, "openblas", "blas")
    ref_name = ref if ref else reference
    if nf is None or ref is None:
        sys.exit(f"gate: need both native-fast and {reference} data for '{op}' "
                 f"(have: {sorted(backends)})")

    ratios = []
    below_floor = []
    for n in sorted(backends[nf]):
        if n < min_size or n not in backends[ref] or backends[ref][n] <= 0:
            continue
        frac = backends[nf][n] / backends[ref][n]
        ratios.append((n, frac))
        # Per-size marks are diagnostic: "low" is below the aggregate
        # threshold (informational), "FLOOR" is below the hard floor (fatal).
        if frac < floor:
            status = "FLOOR"
            below_floor.append((n, frac))
        elif frac < threshold:
            status = "low"
        else:
            status = "ok"
        print(f"  N={n:>5}  native-fast={backends[nf][n]:8.2f}  "
              f"{ref_name}={backends[ref][n]:8.2f}  {100*frac:5.1f}%  [{status}]")

    if not ratios:
        sys.exit(f"gate: no comparable sizes >= {min_size} for '{op}'")

    fracs = sorted(f for _, f in ratios)
    med = statistics.median(fracs)
    mean = statistics.fmean(fracs)
    lo, hi = fracs[0], fracs[-1]
    print(f"\n  median {100*med:.1f}%   mean {100*mean:.1f}%   "
          f"min {100*lo:.1f}%   max {100*hi:.1f}%   ({len(fracs)} sizes)")

    fails = []
    if med < threshold:
        fails.append(f"median {100*med:.1f}% < {100*threshold:.0f}%")
    if below_floor:
        worst_n, worst_f = min(below_floor, key=lambda t: t[1])
        fails.append(f"{len(below_floor)} size(s) below the {100*floor:.0f}% "
                     f"floor (worst N={worst_n} at {100*worst_f:.1f}%)")
    if fails:
        print(f"\nGATE FAIL: {op} vs {ref_name} — " + "; ".join(fails) + ".")
        return 1

    note = ""
    if lo < threshold:
        note = (f" Individual sizes range down to {100*lo:.1f}%; per-size "
                f"variation of a few points between runs is expected (#327).")
    print(f"\nGATE PASS: {op} median {100*med:.1f}% >= {100*threshold:.0f}% of "
          f"{ref_name}; all {len(fracs)} sizes >= the {100*floor:.0f}% floor "
          f"(N >= {min_size}).{note}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Native-fast vs OpenBLAS gate analyzer.")
    ap.add_argument("csv", nargs="+", help="bench_all CSV file(s) to merge")
    ap.add_argument("--peak-gflops", type=float, default=None,
                    help="machine FMA peak (GFLOP/s) for the %% FMA peak column")
    ap.add_argument("--gate", action="store_true", help="run the pass/fail perf gate")
    ap.add_argument("--op", default="gemm", help="operation to gate (default: gemm)")
    ap.add_argument("--reference", default="openblas",
                    help="reference backend to compare against (default: openblas; "
                         "e.g. blis, mkl)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="min MEDIAN native-fast/reference fraction across the "
                         "gated sizes (default: 0.80)")
    ap.add_argument("--floor", type=float, default=0.70,
                    help="no individual size may fall below this fraction; "
                         "catches a genuine single-size cliff while tolerating "
                         "ordinary run-to-run spread (default: 0.70)")
    ap.add_argument("--min-size", type=int, default=256,
                    help="only gate sizes >= this (default: 256)")
    args = ap.parse_args()
    if args.peak_gflops is not None and args.peak_gflops <= 0:
        ap.error("--peak-gflops must be > 0")
    if args.floor > args.threshold:
        ap.error("--floor must be <= --threshold (the floor is the per-size "
                 "backstop under the aggregate threshold)")

    data = load(args.csv)
    if args.gate:
        print(f"Gate: {args.op} native-fast median >= {args.threshold*100:.0f}% "
              f"of {args.reference} (per-size floor {args.floor*100:.0f}%) "
              f"for N >= {args.min_size}")
        sys.exit(run_gate(data, args.op, args.threshold, args.min_size,
                          args.reference, args.floor))
    print_table(data, args.peak_gflops, args.reference)


if __name__ == "__main__":
    main()
