# Multigrid: the V-cycle over pluggable smoothers and grids

How MTL5's multigrid driver works, and *why* it is built as a recursion over four
pluggable roles — a grid hierarchy, inter-grid transfer operators, a
[smoother](smoothers-architecture.md), and a coarse solver. Multigrid is the method
that turns the slow tail of a stationary iteration into an O(n) solve, and the
design mirrors the composability of the [Krylov solvers](iterative-solvers-architecture.md):
the driver owns only the cycle; every numerical ingredient is supplied.

The subsystem is the `multigrid` driver plus the grid-transfer operators in
[`itl/mg/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/mg)
(namespace `mtl::itl::mg`).

---

## The idea: split the error by frequency across grids

A [smoother](smoothers-architecture.md) damps the *high-frequency* (oscillatory)
components of the error in a few sweeps, but barely touches the *low-frequency*
(smooth) components — which is why a stationary iteration stalls. Multigrid's
insight is that **smooth error on a fine grid looks oscillatory on a coarser grid**,
where a smoother *can* damp it cheaply. So it recursively pushes the residual down to
coarser grids, smooths at each, and interpolates the correction back up:

```text
smooth (kill high freq)  →  restrict residual to coarser grid
   →  [solve the coarse problem the same way, recursively]
   →  prolongate the correction back  →  smooth again
```

Each grid handles the frequency band it damps efficiently, and the coarsest grid —
small enough to solve exactly — mops up what remains. The result is a solver whose
cost is O(n) for the model problems it targets, instead of the O(n^1.5) or worse of a
lone smoother or unpreconditioned Krylov method on a PDE.

---

## Decision 1 — the V-cycle (and W-cycle) recursion

The core is `vcycle(x, b, level)`, a direct transcription of the idea:

```cpp
void vcycle(vector_type& x, const vector_type& b, int level) {
    if (level == coarsest) { coarse_solve_(x, b); return; }       // base case: solve exactly

    for (int s = 0; s < nu_pre_;  ++s) smoothers_[level](x, b);    // pre-smooth
    r        = b - A[level] * x;                                   // residual
    r_coarse = restrict(R[level], r);                             // fine -> coarse
    e_coarse = 0;
    vcycle(e_coarse, r_coarse, level + 1);                        // recurse (solve A_c e_c = r_c)
    x += prolongate(P[level], e_coarse);                         // coarse -> fine, correct
    for (int s = 0; s < nu_post_; ++s) smoothers_[level](x, b);   // post-smooth
}
```

Pre-smooth, restrict the residual, recurse on the coarse-grid *error equation*
(starting from zero), prolongate the correction, post-smooth. The recursion bottoms
out at the coarsest level with a direct solve. The **W-cycle** is the same body with
the recursion invoked **twice** per level — more coarse-grid work per visit, more
robust for harder problems. V and W are one recursion shape with a one-line
difference, so they are the two methods and nothing else.

---

## Decision 2 — composition over four pluggable roles

The `multigrid` object is assembled from independent ingredients, exactly the
composition pattern the [Krylov solvers](iterative-solvers-architecture.md#decision-6--the-cartesian-product-for-free)
use:

```cpp
multigrid(levels,        // A[0] (finest) … A[L-1] (coarsest) — the grid hierarchy
          restrictors,   // R[0] … R[L-2] — fine→coarse transfer matrices
          prolongators,  // P[0] … P[L-2] — coarse→fine transfer matrices
          smoother_factory,   // makes a smoother for a given level's matrix
          coarse_solver,      // solves the coarsest level exactly
          nu_pre, nu_post);   // smoothing counts
```

The driver knows the *cycle*; it does not know *which* smoother or coarse solver you
use. Any [`itl::smoother`](smoothers-architecture.md) (Jacobi, Gauss-Seidel, SOR)
plugs in through the factory, and any callable that solves the coarse system plugs in
as `coarse_solver`. That is what ties this subsystem to the smoother one: multigrid
is the composition layer, the smoothers are the components. Swap the smoother, the
cycle count, or the hierarchy depth without touching the driver.

---

## Decision 3 — inter-grid transfer is a sparse matvec

Restriction and prolongation are not special machinery — they are
[`compressed2D`](../architecture/containers/compressed2d-architecture.md) matrices,
and applying them is an ordinary sparse matrix-vector product:

```text
r_coarse = R · r_fine        R : n_coarse × n_fine   (fine → coarse)
e_fine   = P · e_coarse       P : n_fine × n_coarse   (coarse → fine)
```

For the shipped 1-D geometric case, `make_restriction_1d` builds `R` with the
standard **full-weighting** stencil `[¼, ½, ¼]`, and `make_prolongation_1d` builds
`P` by **linear interpolation** — related by the classic `P = 2·Rᵀ`. Both are
assembled with the [`compressed2D` inserter](../architecture/containers/compressed2d-architecture.md) and applied
with a hand CRS matvec. Casting grid transfer as "just" SpMV with purpose-built
sparse operators means the entire sparse-matrix and matvec toolchain is reused; the
multigrid driver adds no new linear-algebra primitive, only the cycle that sequences
them.

---

## Decision 4 — type erasure for the per-level smoother collection

One implementation choice stands out against MTL5's usual compile-time-generic
style. The per-level smoothers are stored as a homogeneous vector of
`std::function`:

```cpp
std::vector<std::function<void(vector_type&, const vector_type&)>> smoothers_;
std::function<void(vector_type&, const vector_type&)>              coarse_solve_;
```

The constructor builds one smoother per level via the factory and **type-erases** it
into a `std::function`. This is a runtime-dispatch cost the rest of the library
avoids — and it is the right call *here*, because the hierarchy holds a
**collection** of smoothers, one per level, that must live in a single container and
be indexed by level at run time. A `std::vector` needs a single element type;
`std::function` provides it while still letting each level's smoother be any callable
the factory produced. The erasure is confined to the driver's storage; the smoothers
themselves stay statically typed.

---

## Decision 5 — multigrid is both a solver and a Krylov preconditioner

The driver exposes two faces:

- **A standalone iteration.** `operator()(x, b)` applies one V-cycle; repeated
  application is a convergent solver in its own right.
- **A preconditioner.** `solve(x, b)` / `adjoint_solve(x, b)` apply one V-cycle as
  `M⁻¹`, satisfying the [`Preconditioner` concept](iterative-solvers-architecture.md#decision-3--the-preconditioner-is-a-concept-solve--adjoint_solve).
  This is the powerful combination — **multigrid-preconditioned CG** — where one
  V-cycle per Krylov step gives grid-independent convergence.

That second face closes a loop across the whole `itl/` subsystem: multigrid is *built
from* smoothers (which, symmetric, are themselves Krylov preconditioners), and
multigrid *is itself* a Krylov preconditioner. The same small parts compose at every
level. (`adjoint_solve` equals `solve`, valid when the cycle is symmetric — symmetric
smoothers with matching pre/post counts; and `solve` `const_cast`s to run the
mutating cycle through the const preconditioner interface — a small, honest wart of
that adaptation.)

---

## What the design deliberately does not do

- **Geometric, 1-D transfer operators only.** `make_restriction_1d` /
  `make_prolongation_1d` build the 1-D full-weighting / linear-interpolation
  operators. There is no **algebraic multigrid** (AMG), where `R`/`P` are derived
  from `A`'s graph, and no built-in 2-D/3-D geometric transfers — though the driver
  itself is dimension-agnostic given the matrices, so a caller can supply their own.
- **The coarsest solve is exact and external.** The base case delegates to the
  supplied `coarse_solver`; the driver does not prescribe how the small coarse system
  is solved.
- **Reference-grade, not a tuned MG library.** Like the Krylov eigensolvers, this is
  a legible reference implementation — the cycle and the composition are the point,
  not a production AMG with aggressive coarsening and smoother tuning.

---

## Where it sits in the stack

```text
containers    (compressed2D level matrices + R/P transfer operators; dense_vector state)
   │
smoothers     (itl/smoother: the per-level high-frequency error damper)
   │
▶ MULTIGRID    (itl/mg: the V/W-cycle that sequences smooth → restrict → recurse →
                prolongate → smooth, over a supplied hierarchy)
   │
(optionally) Krylov solver   (itl/krylov: preconditioned by one V-cycle)
```

Multigrid adds no arithmetic primitive of its own. It is pure orchestration: it
sequences smoother sweeps, sparse matvecs (the transfers), and a coarse solve into a
recursion whose whole is far more than its parts — an O(n) solver, or a
grid-independent preconditioner, assembled from pieces documented elsewhere.

---

## File map

| File | Role |
|---|---|
| [`itl/mg/multigrid.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/mg/multigrid.hpp) | the V-cycle / W-cycle driver; solver `operator()` and `Preconditioner` `solve` |
| [`itl/mg/restriction.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/mg/restriction.hpp) | `make_restriction_1d` (full-weighting) + `restrict` (SpMV) |
| [`itl/mg/prolongation.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/mg/prolongation.hpp) | `make_prolongation_1d` (linear interp, `P = 2·Rᵀ`) + `prolongate` |
| [`itl/smoother/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/smoother) | the smoothers the driver applies per level |

For the smoothers it sequences, see the
[smoothers architecture](smoothers-architecture.md); for the Krylov methods it can
precondition, the [iterative solvers](iterative-solvers-architecture.md) doc; for the
CRS matrices and inserter it builds transfers with, the
[compressed2D](../architecture/containers/compressed2d-architecture.md) doc.
