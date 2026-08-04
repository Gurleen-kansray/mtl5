# The smoothers: stationary iterations as reusable sweeps

How MTL5 implements Jacobi, Gauss-Seidel, and SOR, and *why* they are packaged as
small, reusable *sweep* functors rather than solve-to-convergence routines. A
smoother applies one matrix-splitting sweep of `A·x = b`, cheaply damping the
high-frequency components of the error — its role inside
[multigrid](multigrid-architecture.md) — while also serving, in its symmetric
form, as a Krylov [preconditioner](iterative-solvers-architecture.md#decision-3--the-preconditioner-is-a-concept-solve--adjoint_solve).

The subsystem is three methods and their sweep-direction variants in
[`itl/smoother/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/smoother)
(namespace `mtl::itl::smoother`). They share one row kernel and differ only in how
the update is ordered.

---

## The common form: one splitting, three orderings

Every smoother computes the same per-row update — solve row `i` for `x_i`, holding
the other unknowns fixed:

```text
x_i  ←  D_ii⁻¹ · ( b_i  −  Σ_{j≠i} A_ij · x_j )        # D = diag(A)
```

The three methods differ only in *which* `x_j` the row sum reads and how the result
is blended in:

| Method | Reads, for the row sum | Update |
|---|---|---|
| **Jacobi** | the **old** `x` everywhere (simultaneous) | write to a fresh `x_new`, copy back |
| **Gauss-Seidel** | the **latest** `x` — `x_j` for `j<i` already updated this sweep | in place |
| **SOR** | latest `x` (as GS) | `x_i ← ω·(GS update) + (1−ω)·x_i` |

That one shared kernel with three orderings is the whole family; the code is
organized to reflect it.

---

## Decision 1 — a smoother is a functor: construct from `A`, apply as one sweep

Each smoother is a small class parameterized on the matrix type, built once from
`A` and then applied repeatedly:

```cpp
gauss_seidel<compressed2D<double>> S(A);   // ctor precomputes inv(diag A)
S(x, b);                                    // ONE forward sweep, mutates x
S(x, b);                                    // apply again — a few sweeps is the usage
```

The constructor precomputes the inverse diagonal (`dia_inv_`); `operator()(x, b)`
runs a single sweep and returns the updated `x`. Crucially, a smoother does **not**
iterate to a tolerance — one call is one sweep. This is deliberate: a smoother is a
*primitive*, applied a fixed small number of times (two or three) by whatever drives
it. Its job is not to solve the system but to make the error smooth, and "how many
sweeps / when to stop" belongs to the caller (multigrid, or a stationary-iteration
loop), not the smoother. The name says it: *smoother*, not *solver*.

---

## Decision 2 — Jacobi (simultaneous) vs Gauss-Seidel (in-place) vs SOR (relaxed)

The three differ in exactly the way the mathematics does:

- **Jacobi** computes every `x_new(i)` from the *old* `x`, then copies `x_new` back.
  It needs an O(n) temporary, and the sweep is order-independent (every update sees
  the same input) — which is what makes Jacobi trivially parallel, at the cost of
  slower convergence.
- **Gauss-Seidel** writes in place, so when it reaches row `i` the entries `x_j`,
  `j<i`, already hold this sweep's updated values. No temporary, faster
  convergence, but the sweep carries a **sequential dependency** (row `i` depends on
  the rows before it). Sweep *direction* therefore matters, and the code exposes
  `forward()` (ascending rows) and `backward()` (descending).
- **SOR** is Gauss-Seidel with a relaxation factor `ω`: `x_i = ω·gs_update +
  (1−ω)·x_i`. `ω = 1` recovers GS; `ω > 1` over-relaxes to accelerate convergence;
  `ω < 1` under-relaxes for stability. It reuses the GS row sum and adds the blend.

Jacobi allocates its `x_new`; GS and SOR are in place. That memory-vs-parallelism
distinction is intrinsic to the methods, and the implementations wear it plainly.

---

## Decision 3 — sweep-direction and symmetric variants are first-class types

Beyond the base forward sweep, the layer ships named variants:

```text
gauss_seidel            forward() / backward()
backward_gauss_seidel   one descending sweep
symmetric_gauss_seidel  forward THEN backward, per application
sor / backward_sor / symmetric_sor   the same, relaxed
```

The **symmetric** variant (a forward sweep immediately followed by a backward one)
is provided as a first-class primitive for a specific reason, stated in the source:
for symmetric `A`, symmetric Gauss-Seidel is itself a **symmetric, SPD-preserving**
operation, which is exactly the property a
[Conjugate Gradient preconditioner](iterative-solvers-architecture.md#decision-3--the-preconditioner-is-a-concept-solve--adjoint_solve)
must have. So SGS/SSOR are not left for the caller to compose from two half-sweeps;
they are named types, because that composed operation *is* a distinct, useful
object — the kernel behind the `pc::ssor` preconditioner. A sweep direction is a
correctness-relevant choice here, not a detail, so it is lifted into the type.

---

## Decision 4 — dense and sparse specializations (O(n²) vs O(nnz))

Each smoother has two implementations, chosen by the matrix type:

- The **generic** template computes the row sum with `A(i, j)` over all `j` —
  O(n²) per sweep, correct for any matrix but touching every (mostly-zero) entry.
- A **`compressed2D` specialization** walks the raw CRS arrays
  (`starts`/`indices`/`data`), summing only stored nonzeros — **O(nnz)** per sweep —
  and (for Gauss-Seidel) caches each row's diagonal position so the split into
  diagonal and off-diagonal is a lookup, not a search.

This is the same "specialize the sparse fast path over the generic floor" pattern
the whole library uses. It matters especially here: smoothers earn their keep on
the large sparse systems from PDE discretizations, where the O(n²) dense sweep would
be unusable and the O(nnz) CRS sweep is the point.

---

## Decision 5 — the per-row `Accumulator`, instrumented for mixed-precision study

Every smoother carries an optional `Accumulator` template parameter governing the
precision in which the off-diagonal row sum `σ = Σ_{j≠i} A_ij x_j` is accumulated,
routed through `accumulator_traits` — the same mechanism as the
[Krylov solvers](iterative-solvers-architecture.md#decision-7--mixed-precision-threads-through-from-the-operation-layer)
and `void` (naive `value_type` accumulation) by default.

But the smoothers are a *deliberate research instrument* for mixed-precision
iterative methods, and two design choices reflect that:

- **The accumulator scope is per row** — cleared at the start of each row. Combined
  with Gauss-Seidel's in-place sweep (row `i` reads the freshly-updated, possibly
  already-rounded `x_j`, `j<i`), this exposes exactly the **intra-sweep error
  propagation** that low-precision-iterative studies target. The accumulator governs
  the rounding of one row sum; it does not change which entries are read.
- **In SOR, the `ω` blend is kept *outside* the accumulator** — only the row sum is
  accumulated; `dia_inv·(b−σ)` and the `ω`-blend are ordinary scalar arithmetic.
  This lets a study vary `ω` against precision *independently* of the sum's rounding.

The exact/quire accumulators themselves live downstream — "MTL5 stays
Universal-free." The smoothers just provide the precisely-scoped seam.

---

## What the design deliberately does not do

- **One sweep per call, no convergence loop.** A smoother is a primitive; running it
  to a tolerance is the caller's job. That keeps it composable (multigrid calls it
  `nu` times; a preconditioner wraps one application).
- **No colored / red-black parallel Gauss-Seidel.** The in-place GS sweep is
  inherently sequential, and the shipped version keeps it so — a legible reference,
  not a parallelized reordering. (Jacobi is the parallel-friendly member, by
  construction.)
- **Smoothers expose `operator()`, not `solve()`.** They are not themselves
  `Preconditioner`-concept types; the `pc::ssor` wrapper adapts the symmetric sweep
  to the `solve`/`adjoint_solve` face when a Krylov method needs one.

---

## Where they are used

A smoother is a building block for two higher-level constructs:

- **Multigrid** applies a smoother `nu_pre`/`nu_post` times at each grid level to
  damp high-frequency error before/after the coarse-grid correction — its primary
  reason for existing. See the [multigrid architecture](multigrid-architecture.md).
- **Krylov preconditioning** uses the symmetric variants (via `pc::ssor`) as an
  SPD-preserving `M` for CG and friends.

Each is a small, single-purpose sweep; the power is in how the layers above compose
them.

---

## File map

| File | Role |
|---|---|
| [`itl/smoother/jacobi.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/smoother/jacobi.hpp) | Jacobi (simultaneous update; dense + CRS specializations) |
| [`itl/smoother/gauss_seidel.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/smoother/gauss_seidel.hpp) | Gauss-Seidel + `backward_` / `symmetric_` variants |
| [`itl/smoother/sor.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/smoother/sor.hpp) | SOR (relaxed GS) + `backward_` / `symmetric_` variants |
| [`itl/pc/ssor.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/pc/ssor.hpp) | adapts the symmetric sweep to the `Preconditioner` concept |

For the driver that applies these sweeps across a grid hierarchy, see the
[multigrid architecture](multigrid-architecture.md); for the CRS matrix they sweep
over, the [compressed2D](../architecture/containers/compressed2d-architecture.md)
doc; for the Krylov methods that precondition with them, the
[iterative solvers](iterative-solvers-architecture.md) doc.
