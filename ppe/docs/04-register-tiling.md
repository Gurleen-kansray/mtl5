# 04 — Register tiling

## The change

`v1`–`v3` write to `C` on every `k` step. For a `k`-deep accumulation that is
`k` read-modify-writes per output element, all of them to memory.

`v4` accumulates an `MR × NR` microtile in **local variables** across the whole
`kc` depth and writes `C` once:

```cpp
TC acc[MR][NR] = {};
for p in [0,kc): for ii: for jj: acc[ii][jj] += Ap[...] * Bp[...];
// then one write-back of MR*NR values
```

This is the step that converts a memory-bound inner loop into a compute-bound
one. It is also the step where *how many* accumulators you keep starts to matter.

## Why the accumulator count is the whole game

A multiply-accumulate has latency. To keep the FMA units busy you need enough
**independent** accumulation chains in flight to cover it:

```
chains needed  ≈  FMA latency × FMA issue width
```

On this target that is 4 × 2 = **8 independent chains**. Fewer, and the pipeline
stalls waiting on the previous update to the same accumulator; the kernel becomes
latency-bound and no amount of blocking helps.

More is not automatically better either: every accumulator is a register, and
when they stop fitting they spill. The window between "enough to hide latency"
and "too many to hold" is what the microtile shape has to land in — and it is
narrow.

## What to watch

`v4` uses **runtime** `mr`/`nr` bounds, which means the compiler cannot fully
unroll the tile and will instead vectorize the inner `jj` loop. That is an
accident of how the kernel is written, and it turns out to matter enormously —
see [05](05-vectorization-cliff.md), where making the bounds compile-time makes
the kernel *slower*.

The interesting comparison is therefore not "did `v4` beat `v3`" but "what did
`v4` leave on the table, and does forcing the compiler's hand recover it or
destroy it".
