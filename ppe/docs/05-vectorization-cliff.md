# 05 — The vectorization cliff

The step where the progression goes **backwards**, and the most useful result in
the study.

## The experiment

`v5_micro` takes `v4_regtile` and changes one thing: the microtile becomes a
separate function with **compile-time** `MR`/`NR` and no edge tests, so the
compiler can see fixed trip counts, fully unroll, and keep the accumulators in
registers.

```cpp
template <typename TC, std::size_t MR, std::size_t NR>
inline void micro(std::size_t kc, const TC* __restrict Ap, std::size_t lda,
                  const TC* __restrict Bp, std::size_t ldb,
                  TC* __restrict C, std::size_t ldc) {
    TC acc[MR][NR] = {};
    for (std::size_t p = 0; p < kc; ++p) { ... }
}
```

This is the standard advice. It is what every GEMM tutorial does next. **It made
things slower**, in some cases by 5×.

## The measurement

At N=512, fp32: `v1_ikj` reaches 37.3 GOP/s and `v5_micro` reaches **7.3** — the
"most optimized" kernel is a fifth the speed of a loop reorder.

## Why: the compiler declined to vectorize, and then it spilled

Not a guess — the compiler says so:

```
$ g++ -O3 -march=native -fopt-info-vec-missed
kernels.hpp:160: missed: not vectorized: loop nest containing
                 two or more consecutive inner loops cannot be vectorized
kernels.hpp:172: missed: complicated access pattern
```

And the consequence is visible in stack usage:

```
$ g++ -fstack-usage
gemm_v4_regtile   832 bytes
gemm_v5_micro    1344 bytes      <- +512 bytes of spill
```

The mechanism:

1. `MR × NR = 4 × 8 = 32` accumulators, with compile-time bounds, so the
   compiler fully unrolls the `ii`/`jj` nest.
2. Full unrolling replaces the vectorizable inner loop with 32 *scalar*
   operations — GCC then reports the nest as unvectorizable.
3. 32 scalar accumulators do not fit in 16 architectural registers, so they
   spill to the stack.
4. Every `p` iteration now reloads and restores 32 values. The kernel is
   memory-bound on its own stack frame.

`v4`, which keeps **runtime** `mr`/`nr` bounds, cannot fully unroll — so the
compiler vectorizes the inner `jj` loop instead and keeps a handful of vector
accumulators live. The "less optimized" version wins because it left the
compiler something it could actually do.

## Refinement after the full sweep: the penalty is type-dependent

The measurement across all seven types sharpens this. `v5` is **not** uniformly
bad — at N=2048 it is the *best* kernel for `int8` (36.4 GOP/s) and `int32`
(35.9), and catastrophic for the floats (`fp32` 7.3 against `v3`'s 28.6).

Checking the compiler per type settles why:

```
v5 micro <int32 > : 0 vectorized loops in the microkernel
v5 micro <double> : 0 vectorized loops in the microkernel
v5 micro <float > : 0 vectorized loops in the microkernel
```

**`v5` never vectorizes, for any type.** So the spill is a secondary effect, and
the proximate mechanism is simpler than "register pressure":

> The compile-time-unrolled microkernel is always scalar, and the penalty for
> being scalar is proportional to how much that type gains from vectorization.

- `fp32`/`fp64`/`fp16` gain 4–8× from SIMD, so losing it is catastrophic.
- `int32` gains little, because `vpmulld` is throughput-limited rather than
  lane-limited — a scalar kernel is competitive, and `v5`'s better instruction
  scheduling then wins on the margin.
- `int8`/`int16` widen to `int32` to accumulate, so they inherit that behaviour.

This **answers H5 and reframes H1**: register pressure is not the first-order
cause. The first-order cause is that full unrolling removes the loop structure
the vectorizer needs, and whether that matters depends on the type's SIMD
headroom. H2 (keep the tile, keep the loop) is therefore the experiment to run
first, ahead of H1 (change the tile shape).

## What this says generally

**Unrolling and vectorization compete for the same registers.** A microtile has
to be sized in *vector registers*, not in scalars: for fp64 on AVX2 a 4×8 tile is
32 doubles = 8 ymm registers *if vectorized*, and 32 scalar registers if not.
The difference between those two outcomes is a compiler decision that source-level
"optimization" can silently flip.

It also means **`MR × NR` is not a free parameter**. It is bounded above by the
register file, and the binding constraint depends on whether the code vectorizes
— which is exactly the failure mode this step demonstrates.

## Hypotheses for the shortfall, and how to test them

| # | hypothesis | experiment | expected signal if true |
|---|---|---|---|
| H1 | The tile is too large to vectorize *and* fit registers | sweep `MR × NR` ∈ {4×4, 4×8, 6×8, 8×8, 4×16} at fixed everything else | a peak at the shape where `MR × NR / SIMD_width` ≈ 8–12 vector registers |
| H2 | The failure is unrolling, not size | keep 4×8 but force the `jj` loop to stay a loop (`#pragma GCC novector` off, or a runtime bound) | `v5` recovers to `v4` levels without changing the tile |
| H3 | It is a GCC-specific decision | build the same source with Clang 18 and compare | Clang vectorizes where GCC does not, or fails differently |
| H4 | Explicit SIMD types would remove the guesswork | reimplement `micro` with a portable vector type instead of scalar arrays | throughput becomes insensitive to unrolling decisions |
| H5 | The narrow types fail for a different reason | check whether int8/int16 show the same cliff — they widen to int32, so the register pressure differs | if the cliff is absent for int8/int16, register pressure is confirmed as the cause rather than the loop structure |

H1 and H2 are the cheap ones and should be run first: both are parameter changes,
not rewrites, and between them they separate "tile too big" from "unrolling
destroyed vectorization".

## Why this matters beyond PPE

MTL5's own GEMM derives its register tile as

```cpp
area = nvec * fma_latency * fma_units;   // 4 * 4 * 2 = 32
nr   = round_up(isqrt_ceil(area), nvec); // 8
mr   = ceil_div(area, nr);               // 4
```

giving `mr × nr = 4 × 8` — **exactly** the same shape that fails here, and
exactly the minimum accumulator count needed to cover FMA latency, with no
headroom. That library sustains ~72–78% of peak single-core
([mtl5#351](https://github.com/stillwater-sc/mtl5/issues/351)).

PPE is the cheap place to test whether tile shape is the limiter, because H1
here is a one-parameter sweep with no library API to preserve.
