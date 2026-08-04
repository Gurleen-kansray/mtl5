# The eigensolvers: dense QR and matrix-free Krylov, composed

How MTL5 solves the eigenvalue problem `A·v = λ·v`, and *why* it ships two
families rather than one. A **dense** path factors a moderate matrix outright —
Householder tridiagonalization followed by shifted QR iteration, dispatched to
LAPACK when available. A **matrix-free Krylov** path handles the large-and-sparse
regime, projecting the operator onto a small subspace and then handing that small
matrix to the dense path. The second family is built *on* the first; understanding
that composition is understanding the layer.

This is a [Library subsystem](operation-dispatch-architecture.md) sitting at the
top of the stack — the dense path is a set of `mtl::` operations that dispatch like
any other, and the iterative path is a close cousin of the
[Krylov linear solvers](iterative-solvers-architecture.md), sharing their
matrix-free operator abstraction.

The dense solvers live in
[`operation/eigenvalue_symmetric.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/eigenvalue_symmetric.hpp)
and
[`operation/eigenvalue.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/eigenvalue.hpp);
the iterative ones in
[`itl/eigen/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/eigen)
(`power_iteration`, `lanczos`, `arnoldi`).

---

## Two regimes for one problem

The eigenvalue problem has two operating points, and the library provides a family
for each — the same dense-vs-iterative split the *linear* solvers make (direct
factorization vs Krylov), now for eigenvalues:

| | Dense (`mtl::eigen_symmetric`, `eigenvalue`) | Iterative (`itl::lanczos`, `arnoldi`, `power_iteration`) |
|---|---|---|
| Operand | a stored dense matrix, moderate `n` | large / sparse / **matrix-free** operator |
| Computes | **all** eigenpairs | a **few** extremal eigenpairs |
| Cost | O(n³) | O(nnz · m), `m` = subspace ≪ n |
| Method | tridiagonalize + shifted QR (or LAPACK) | Krylov projection, then the dense solver on the projection |
| Needs | `A`'s entries | only `A·x` |

You pick by which regime you are in; the two are used through different signatures
because they answer different questions ("give me the whole spectrum of this
matrix" vs "give me the five largest eigenvalues of this operator").

---

## Decision 1 — the dense path: tridiagonalize, then shifted QR, dispatched

For a symmetric matrix the algorithm is the classical two-phase one:

1. **Householder tridiagonalization** — reduce `A` to a symmetric tridiagonal `T`
   by a sequence of orthogonal reflectors (a similarity transform, so the spectrum
   is preserved).
2. **Implicit QR with Wilkinson shifts** — iterate `T` to diagonal form; the
   diagonal converges to the eigenvalues, and the accumulated reflectors + Givens
   rotations form the orthogonal eigenvector matrix `Q`, so `A = Q·diag(λ)·Qᵀ`.

Wrapped in the same build-is-the-backend dispatch as the rest of the
[operation layer](operation-dispatch-architecture.md#decision-4--the-build-is-the-backend):

```cpp
template <Matrix M>
auto eigenvalue_symmetric(const M& A, ...) {
#ifdef MTL5_HAS_LAPACK
    if constexpr (/* float/double dense */) { /* LAPACK syev */ return ...; }
#endif
    return eigenvalue_symmetric_generic(A, ...);   // native tridiag + Wilkinson QR
}
```

When LAPACK is linked and the type qualifies, the call routes to `syev`; otherwise
it runs MTL5's own tridiagonalize-and-QR. Same ordered preference, same
never-optional native floor as `mult` and the factorizations.

---

## Decision 2 — the native path is separately callable, for testability

Note that the generic algorithm is not buried inside the dispatcher — it is a
public function of its own, `eigenvalue_symmetric_generic`, that
`eigenvalue_symmetric` falls through to:

> This is the C++ reference path. `eigenvalue_symmetric` dispatches to LAPACK when
> available and otherwise calls this; benchmarks and tests can call this directly
> to exercise the generic algorithm regardless of `MTL5_HAS_LAPACK`.

This is a deliberate testability decision. Because the dispatcher would *hide* the
native path whenever LAPACK is present, a test that only called
`eigenvalue_symmetric` would never exercise the reference implementation on a
LAPACK build — the native QR iteration could rot undetected. Exposing
`_generic` as a first-class entry point means the reference algorithm is always
directly reachable and tested, independent of the build configuration. The floor
is not just present; it is *addressable*.

---

## Decision 3 — symmetric and general are distinct algorithms

Symmetry is not a minor optimization here; it changes the mathematics, so the two
cases are separate functions:

- **Symmetric** (`eigenvalue_symmetric` / `eigen_symmetric`) — the spectrum is
  **real** and the eigenvectors **orthogonal**. The tridiagonal form and the real
  QR sweep exploit that, and `eigen_symmetric` accumulates real Householder +
  Givens transforms into an orthogonal `Q`. A `static_assert` rejects complex
  element types, because `H·A·H` is a similarity transform only for real
  Householder reflectors.
- **General** (`eigenvalue`) — the spectrum may be **complex**; the reduction is to
  **Hessenberg** form (not tridiagonal) followed by the Francis QR iteration,
  returning complex eigenvalues.

Keeping them apart lets each use the right normal form and the right arithmetic,
rather than forcing the general (complex, Hessenberg) machinery onto the common
symmetric case that can stay real and cheaper.

---

## Decision 4 — the iterative path is matrix-free

The Krylov eigensolvers require nothing of `A` but the ability to apply it. Their
one interaction with the operator is a matvec, materialized by a shared helper:

```cpp
template <typename LinearOp, typename T>
vec::dense_vector<T> ev_matvec(const LinearOp& A, const vec::dense_vector<T>& x) {
    auto w = A * x;   // the ONLY thing power/lanczos/arnoldi ask of A
    /* copy into a dense_vector and return */
}
```

`power_iteration`, `lanczos`, and `arnoldi` are templated on an unconstrained
`LinearOp` and only ever call `A · x`. So — exactly as with the
[Krylov linear solvers](iterative-solvers-architecture.md#decision-2--the-operator-is-duck-typed-it-only-needs-multa-p-q) —
a [`dense2D`](../architecture/containers/dense2d-architecture.md), a
[`compressed2D`](../architecture/containers/compressed2d-architecture.md), or a
user-supplied matrix-free operator all work identically, and the solver never
learns which it was given. This is what makes the iterative family usable on the
large sparse problems where forming or factoring `A` is out of the question: they
compute a few eigenpairs from matvecs alone, in O(1)–O(m) vectors of storage.

---

## Decision 5 — Krylov projection: reduce big → small, then defer

The iterative solvers do not compute eigenvalues themselves. Their algorithm is
**dimension reduction**; the eigen-decomposition is delegated to the dense path.
Lanczos is the clearest case:

```text
1. Build an orthonormal Krylov basis  V = [q0 q1 … q_{m-1}]  (n × m)  by m matvecs,
   generating a symmetric tridiagonal projection  T = Vᵀ A V  (the α, β coefficients).
2. Solve the SMALL dense problem:  eigen_symmetric(T)  →  (θ_i, s_i)      # m × m, m ≪ n
3. Lift back:  Ritz value λ_i = θ_i,   Ritz vector  v_i = V · s_i.
```

That is why
[`lanczos.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/lanczos.hpp)
`#include`s `operation/eigenvalue_symmetric.hpp` — step 2 is a call straight into
Decision 1. Arnoldi is the same shape with a **Hessenberg** projection solved by the
general dense path. The composition is the architecture: **the iterative layer turns
an intractable n×n (matrix-free) operator into a tractable m×m dense matrix, and the
dense layer solves that.** Each family does what it is good at — Krylov at reducing
dimension through matvecs, dense QR at diagonalizing a small matrix — and they meet
at the projected matrix.

---

## Decision 6 — Ritz pairs, spectrum selection, and residual convergence

The iterative results are **Ritz pairs** — approximate eigenpairs drawn from the
subspace — carried in a small result type:

```cpp
template <typename Value>              // real for Lanczos (symmetric), complex for Arnoldi
struct ritz_pairs {
    vec::dense_vector<Value> values;   // k wanted Ritz values
    mat::dense2D<Value>      vectors;  // n × k, column i is the Ritz vector
    int  subspace;                     // Krylov dimension actually built
    bool converged;
};
```

Two policy inputs shape the answer. `eigen_which` selects **which end of the
spectrum** to return — `largest_magnitude`, `smallest_magnitude`,
`largest_algebraic`, `smallest_algebraic` — because a Krylov method computes
*extremal* eigenvalues well and you must say which extreme. Convergence is judged
by the **Ritz residual** `‖A·v − λ·v‖`, not the linear-solver residual `‖b − A·x‖`:
`power_iteration` stops when `‖A v − λ v‖ ≤ tol·(|λ| + tol)`, with `λ` the Rayleigh
quotient `vᵀA v` of the unit iterate. The eigenproblem has its own notion of "close
enough," and the convergence test encodes it. (The eigensolvers take plain
`max_iter`/`tol` parameters rather than the pluggable
[iteration object](iterative-solvers-architecture.md#decision-4--the-iteration-object-owns-the-stopping-policy)
of the linear solvers — a lighter interface for a family with a single, standard
stopping rule.)

---

## Decision 7 — reference-implementation robustness, stated honestly

Lanczos in exact arithmetic loses basis orthogonality catastrophically; production
codes fight this with selective or partial reorthogonalization and implicit
restarts (ARPACK's IRAM). MTL5's Lanczos instead does **full reorthogonalization** —
two Gram-Schmidt passes against *every* built column each step — and the header
names the trade:

> Full reorthogonalization keeps the basis orthogonal for a robust reference
> implementation.

That is O(m²n) work and O(mn) storage for the basis, more than a restarted method
would use, and it is a deliberate choice: these are **correct, legible reference
solvers**, prioritizing a provably-orthogonal basis over the memory/flop economy of
a restarted scheme. The layer is honest about being reference-grade rather than a
tuned large-scale eigensolver — the same "correct floor first" posture the dense
generic path takes.

---

## What the design deliberately does not do

- **No implicit restarting / deflation** in the Krylov solvers (no IRAM). Subspace
  size is chosen up front; robustness comes from full reorthogonalization, not
  restarts. Fine for a modest number of eigenpairs; not a replacement for ARPACK on
  the largest problems.
- **No unified "eigensolver" interface** across the two families. Dense
  (`eigenvalue_symmetric(A)`) and iterative (`lanczos(A, v0, k, which)`) have
  different signatures because they answer different questions; conflating them
  would obscure the whole-spectrum vs few-extremal distinction.
- **The dense path computes the whole spectrum.** Wanting only a few eigenpairs of
  a *dense* matrix still costs O(n³); the "few eigenpairs" regime is the iterative
  family's job, on operators where O(n³) is impossible anyway.

---

## Where it sits in the stack

```text
operations   (mult / dot / norms — the matvec and inner products both families use)
   │
dense eigensolvers  (operation/eigenvalue*.hpp: tridiag/Hessenberg + shifted QR,
   │                 LAPACK-dispatched — solves a matrix outright)
   │
▶ iterative eigensolvers  (itl/eigen/: power / lanczos / arnoldi)
        matrix-free Krylov projection  →  small dense matrix  →  dense eigensolver  →  Ritz pairs
```

The iterative family is a *client* of the dense family: it reduces a large operator
to a small matrix and calls down. The dense family is a client of the operation
layer for its kernels and its LAPACK dispatch. Neither invents new arithmetic
primitives — they orchestrate matvecs, orthogonalization, and small dense
diagonalizations into the two answers the eigenproblem admits.

---

## File map

| File | Role |
|---|---|
| [`operation/eigenvalue_symmetric.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/eigenvalue_symmetric.hpp) | dense symmetric: `eigenvalue_symmetric` (+ `_generic`), `eigen_symmetric`; LAPACK `syev` dispatch |
| [`operation/eigenvalue.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/eigenvalue.hpp) | dense general: Hessenberg + Francis QR, complex spectrum |
| [`itl/eigen/eigen_common.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/eigen_common.hpp) | `ritz_pairs`, `eigen_which`, `ev_matvec` shared by the iterative solvers |
| [`itl/eigen/power_iteration.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/power_iteration.hpp) | dominant eigenpair (Rayleigh quotient + Ritz residual) |
| [`itl/eigen/lanczos.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/lanczos.hpp) | symmetric: tridiagonal projection → `eigen_symmetric` |
| [`itl/eigen/arnoldi.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/arnoldi.hpp) | general: Hessenberg projection → general dense solver |
| [`itl/eigen/eigensolvers.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/eigen/eigensolvers.hpp) | the iterative-eigensolver umbrella |

For the matrix-free operator abstraction the iterative family shares, see the
[iterative solvers](iterative-solvers-architecture.md) doc; for the LAPACK dispatch
the dense family uses, the [operation dispatch layer](operation-dispatch-architecture.md);
for the typical operands, the
[dense2D](../architecture/containers/dense2d-architecture.md) and
[compressed2D](../architecture/containers/compressed2d-architecture.md) docs.
