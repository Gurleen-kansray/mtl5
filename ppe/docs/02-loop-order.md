# 02 — Loop order and locality

## The change

One idea: swap the `j` and `p` loops so the innermost loop walks `B` and `C`
along contiguous rows.

```cpp
for i: for p: { a = A[i*k+p]; for j: C[i*n+j] += a * B[p*n+j]; }
```

The arithmetic is identical. The order of accumulation into each `C[i][j]` is
identical. Only the traversal changes — and with it, everything about how the
memory system sees the kernel.

## Why it is the largest single step in the progression

The inner loop is now: read a contiguous row of `B`, scale by a loop-invariant
scalar, accumulate into a contiguous row of `C`. That is

- **unit stride on both streams**, so every cache line fetched is fully used;
- **prefetcher-friendly**, being two linear streams;
- **vectorizable**, which the compiler does automatically — this is where SIMD
  first enters the progression, without a single intrinsic.

The cost is that `C` is now read-modify-written `k` times instead of once. That
trade is overwhelmingly favourable, and undoing it is exactly what register
tiling (`v4`) is for.

## What to watch in the results

Two things are worth reading off the curves rather than assumed:

- **Where the win appears.** If it were purely about cache lines it would be
  size-independent; if it is also about vectorization it should hold at small N.
- **Where it starts to fade.** `v1` streams the whole of `B` for every row of
  `A`. Once `B` no longer fits in cache, this must degrade with N — and the size
  at which it does is a direct read of the effective cache the kernel gets.

That fade is the hypothesis `v2` exists to test.
