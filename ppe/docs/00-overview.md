# PPE — Platform Performance Engineering

A from-scratch progression of GEMM implementations, measured across seven
operand types, to make the sources of performance — and of *lost* performance —
explicit and attributable.

This is a laboratory, not a library. Nothing here includes `<mtl/...>`; the
kernels are written independently so each optimization step can be added in
isolation and its effect measured. The findings feed back into MTL5's own
kernels (see [Feeding back](#feeding-back)).

## The method

Six kernels, each adding exactly **one** idea to the previous one, so a measured
delta is attributable to that idea:

| step | idea added |
|---|---|
| `v0_naive` | none — the textbook `ijk` triple loop |
| `v1_ikj` | loop reorder so the inner loop walks contiguous memory |
| `v2_blocked` | cache blocking (`MC`/`NC`/`KC` tiles) |
| `v3_packed` | pack tiles into contiguous panels before computing |
| `v4_regtile` | accumulate an `MR × NR` microtile in locals, not in `C` |
| `v5_micro` | compile-time `MR`/`NR` in a separate function, no edge tests |

All six compute the same thing, and **every kernel is verified against `v0` for
the same type and size before its timing is recorded** — exactly for integers, to
a tight relative tolerance for floats. A fast wrong answer cannot become a
result.

## The type matrix

Two families, on the same throughput axis:

| family | operand → accumulator | why |
|---|---|---|
| integer | `int8 → int32`, `int16 → int32`, `int32 → int32`, `int64 → int64` | narrow integer GEMM accumulates wide in practice (quantized inference is int8 × int8 → int32); the widening is part of what is measured |
| floating point | `fp16 → fp32`, `fp32 → fp32`, `fp64 → fp64` | fp16 has no native arithmetic on this target, so it converts to fp32 — the cost of that is a result, not a nuisance |

Throughput is reported in **GOP/s** (giga-operations per second) rather than
GFLOP/s, because half the matrix is integer. A GEMM performs `2·m·n·k`
operations — one multiply and one add per inner step — under the usual
convention, for both families, so the two are directly comparable.

## What "efficiency" is measured against

Efficiency is throughput as a fraction of a **modelled single-core peak**, and
the model is written out in full in `include/ppe/peak.hpp` so a reader can
disagree with a specific number rather than with the word "peak".

Two earlier approaches were discarded, and the failures are instructive enough
to record:

1. **A first ISA model put `int64` at 2 ops/cycle** (scalar `imul`).
   Measurements reached **119% of it**. AVX2 has no 64-bit SIMD multiply, but
   the compiler emulates one from 32-bit pieces, so the real ceiling is higher
   than "scalar". A model that can be exceeded is not a ceiling.

2. **A probe that *measured* achievable rate** looked more honest and was worse.
   With literal operands the compiler folded the loop away and it reported
   **3.8 million GOP/s** — what a deleted loop looks like. With the operands made
   opaque it reported **17.7 GOP/s for fp64, below the ~37 GOP/s the GEMM kernels
   actually achieve**, because the accumulator array did not stay in vector
   registers. A probe beaten by the thing it is meant to bound is not a ceiling
   either.

So the model is ISA-anchored, documented, and **validated**: the driver fails
with `PEAK MODEL TOO LOW` if any measurement exceeds it. That guard is the
reason the `int64` error above was caught rather than published.

## Running it

```bash
cd ppe
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
taskset -c 4 ./build/ppe_gemm --sizes 64,128,256,512,1024,2048 --ghz 5.0 --csv data/ppe_gemm.csv
./analyze/plot_ppe.py data/ppe_gemm.csv --outdir data
```

`-march=native` is applied by the build: without it the compiler targets a
baseline ISA and the vectorizable steps cannot use FMA or the wider registers,
which is most of what `v5` is meant to measure. The binaries are therefore
non-portable, which is fine — these are measurements of *this* machine.

Pinning matters on a hybrid CPU. `taskset -c 4` puts the run on a P-core; an
unpinned run can land on an E-core and report a number that describes different
silicon.

## The write-ups

| document | what it covers |
|---|---|
| [01 — naive baseline](01-naive-baseline.md) | why `v0` is slow, and why that is the right baseline |
| [02 — loop order and locality](02-loop-order.md) | the single largest step in the whole progression |
| [03 — blocking and packing](03-blocking-packing.md) | why the classic wins are smaller here than folklore suggests |
| [04 — register tiling](04-register-tiling.md) | accumulators, and the point at which they stop fitting |
| [05 — the vectorization cliff](05-vectorization-cliff.md) | where the progression goes **backwards**, and why |
| [RESULTS](results.md) | consolidated measurements, graphs, and open hypotheses |

## Feeding back

The `v5` result — a fully-unrolled microtile that *fails* to vectorize and
spills — is the same mechanism as
[mtl5#351](https://github.com/stillwater-sc/mtl5/issues/351), which asks why
MTL5's own single-core GEMM sustains ~72–78% of peak while its register tile is
derived at exactly the minimum accumulator count. PPE is where that hypothesis
can be tested cheaply, in isolation, before touching the library.
