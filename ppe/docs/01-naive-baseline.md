# 01 — The naive baseline

## The kernel

```cpp
for i: for j: { acc = 0; for p: acc += A[i*k+p] * B[p*n+j]; C[i*n+j] = acc; }
```

The textbook definition, transcribed directly. It is the right baseline for two
reasons: it is what someone writes first, and it is *correct*, so it doubles as
the reference every other kernel is verified against.

## Why it is slow

The inner loop walks `B` **down a column**, with stride `n`. Every access is a
different cache line, and at `N ≥ 256` a different page for a large part of the
sweep. Three consequences, in increasing order of cost:

1. **No spatial locality on B.** One useful element per 64-byte line fetched, so
   the effective bandwidth is 1/8 of the real bandwidth for fp64.
2. **The prefetcher cannot help.** Strided access with a stride that changes per
   `j` looks like noise to a stride predictor.
3. **TLB pressure.** For fp64 at N=1024 a column touches 1024 lines spread over
   8 MB — far past L2 and the L1 TLB reach.

`A` is walked contiguously and `C` is written once per output, so `B` is the
whole story.

## What it establishes

Everything after this is measured as a multiple of `v0`, which is why it is worth
having a baseline that is *honest* rather than deliberately pessimised: no
`volatile`, no aliasing tricks, compiled at `-O3 -march=native` exactly like the
others. The compiler is free to do whatever it can — it just cannot fix the
access pattern.

## Hypothesis carried into the next step

If `B`'s traversal is the dominant cost, then **reordering the loops so the inner
loop walks `B` along a row should recover most of the gap**, without touching
blocking, packing, or vectorization. That is `v1`, and it is testable as a single
change.
