# 03 — Blocking and packing

Two steps, both classic, both smaller here than the folklore suggests. Worth
recording *why*.

## v2 — cache blocking

Tile the iteration space into `MC × NC × KC` blocks so a block's working set is
reused while it is resident, instead of streaming all of `B` for every row of `A`.

```
for jc: for pc: for ic:  <ikj over the tile>
```

Defaults here are `MC=64, NC=256, KC=128`, chosen to put the `B` tile
(`KC × NC`) comfortably inside L2 for the 4-byte types.

**Expected**: no effect while the problem already fits in cache; a growing win
as N passes the cache the kernel was getting.

## v3 — packing

Copy each tile into a dense, sequentially traversed buffer before computing on
it. Blocking fixed *which* data is touched; packing fixes *how it is laid out*.

The usual justification is cache lines, but the real one is TLB and prefetch:
after packing, the inner loop walks one linear buffer, so the hardware sees a
single stream with a constant stride from a single page sequence.

**Expected**: a win that grows with N, offset by a fixed copy cost that is pure
overhead at small N. Packing should therefore *lose* at small N and win at large
N, and the crossover is a real measurement rather than a rule of thumb.

## The honest caveat

`v3` packs into the **accumulator** type, not the operand type. For `int8 → int32`
that means the packed buffer is 4× the size of the source. This is a deliberate
simplification — it keeps the microkernel single-typed — but it inflates the
packing cost for exactly the narrow types that packing is supposed to help.

That is a confound, and it is a candidate explanation for any narrow-type result
that looks worse than expected. **Experiment**: pack in the operand type and widen
inside the microkernel, then compare. If the narrow-type packing cost drops
sharply, the confound is confirmed and the current numbers understate what
`int8`/`int16` can do.
