# compressed2D: architecture of the CSR sparse matrix

How `mtl::mat::compressed2D` is put together, and *why* it is shaped so
differently from its dense sibling. Where
[`dense2D`](dense2d-architecture.md) is a story about **compile-time
configuration** of one owned memory block, `compressed2D` is a story about
**representation and lifecycle**: a fixed three-array CSR layout that is cheap to
read and traverse but expensive to mutate, so mutation is deliberately pushed out
of the container and into dedicated builders.

The organizing idea is a **read/build split**. `compressed2D` is the *frozen*,
read-optimized form. You do not insert into it element by element; you assemble
it — through an inserter or from a coordinate list — and then read and multiply.
Everything below follows from that one commitment.

The implementation is ~150 lines in
[`include/mtl/mat/compressed2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/compressed2D.hpp).

---

## The three arrays

CSR (Compressed Row Storage, a.k.a. CRS) represents a sparse matrix as three
parallel arrays:

```text
compressed2D<Value, Parameters = parameters<>>
   nrows_, ncols_                         logical shape
   starts_  : vector<size_type>  len nrows_+1   row r occupies [starts_[r], starts_[r+1])
   indices_ : vector<size_type>  len nnz        column index of each stored entry
   data_    : vector<Value>      len nnz        the stored values

   Example  [ a . b ]      starts_  = [0, 2, 3, 5]
            [ . . c ]      indices_ = [0, 2, 2, 1, 2]
            [ . d e ]      data_    = [a, b, c, d, e]
```

Three invariants make the reads fast and must be preserved by whoever builds the
matrix:

1. `starts_` is non-decreasing and `starts_[nrows_] == nnz`.
2. Within a row, `indices_` is **sorted ascending** — this is what lets element
   access binary-search.
3. Each `(row, column)` appears at most once — duplicates are resolved at build
   time, not stored.

Because the container cannot cheaply *maintain* these under per-element
insertion, it does not offer per-element insertion at all.

---

## Decision 1 — separate the immutable read from the mutable build

`compressed2D`'s public surface is deliberately lopsided:

- **Read** is rich: `operator()(r,c) const`, `num_rows/num_cols/nnz`, raw array
  access, and (via the operation layer) SpMV.
- **Write** is coarse: `change_dim` and `make_empty` **clear** the matrix;
  there is *no* `operator()` that returns a mutable reference, and no
  `insert(r,c,v)` method.

**Why no element write.** Inserting one entry into the middle of a sorted CSR row
means shifting every subsequent entry in `indices_` and `data_` — O(nnz) per
insert, O(nnz²) to assemble a matrix. That is the classic CSR trap. Rather than
offer a correct-but-catastrophically-slow mutator, `compressed2D` offers none,
and routes construction through builders that assemble the whole matrix in a
scatter-friendly structure and write the CSR arrays **once**. The container you
keep and pass to solvers is always the frozen, invariant-holding form.

The raw arrays *are* exposed — `ref_major()` (→ `starts_`), `ref_minor()` (→
`indices_`), `ref_data()` (→ `data_`) — precisely so the builders (and BLAS-like
interchange, and direct-solver bindings such as KLU/SuperLU) can fill or read
them wholesale. The `major`/`minor` naming keeps those accessors format-neutral
in spirit: for CSR, major = row, minor = column.

---

## Decision 2 — the inserter: slots + overflow + RAII finalize

The primary builder is
[`compressed2D_inserter`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/inserter.hpp).
It is an RAII object that accumulates entries in a mutation-friendly layout and
compresses them into the matrix in its destructor:

```cpp
compressed2D<double> A(n, n);
{
    mtl::mat::inserter<compressed2D<double>> ins(A, slots_per_row);
    ins[i][j] << v;        // proxy chain -> do_insert(i, j, v)
    ...
}                          // ~inserter() -> finalize(): merge, sort, write CSR
```

Its working storage is a two-tier structure:

- **Flat slots.** `slots_` and a parallel `col_idx_` are flat arrays of
  `nrows × slot_size`, with a per-row `count_`. The first `slot_size` distinct
  columns of each row land here — contiguous, allocation-free, cache-friendly.
  This is tuned for the shape that dominates FEM/stencil assembly: a bounded,
  small number of nonzeros per row.
- **Overflow map.** A row that exceeds `slot_size` distinct columns spills into a
  `std::map<pair<row,col>, Value>`. Pathological rows are handled correctly
  without forcing every row to pay for a map or requiring the caller to size
  rows up front.

`finalize()` walks the rows, merges each row's slots with its overflow entries,
**sorts them by column**, and appends to `starts_/indices_/data_` — establishing
the three invariants above in a single pass. Doing this in the destructor means
the matrix is guaranteed compressed exactly when the inserter's scope ends;
there is no "did you remember to finalize" state. The inserter is non-copyable
and non-movable, so its lifetime unambiguously owns the build.

---

## Decision 3 — updater functors decide how duplicates combine

`do_insert` does not assume what a second write to the same `(r,c)` means; that
is a policy, supplied as the `Updater` template parameter:

```cpp
template <typename T> struct update_store { void operator()(T& a, const T& b) const { a = b; } };
template <typename T> struct update_plus  { void operator()(T& a, const T& b) const { a += b; } };
```

- `update_store` (default) — last write wins; the natural semantics for
  specifying a matrix.
- `update_plus` — **accumulate**; the natural semantics for finite-element
  assembly, where the same global `(i,j)` receives contributions from many
  element matrices.

Choosing the behavior at the type level means the hot `do_insert` path contains
no branch on "insert vs accumulate" — the functor is inlined. This is the same
"policy as a type" instinct that drives `dense2D`'s parameter bundle, applied to
the assembly semantics instead of the storage layout.

---

## Decision 4 — row-major only, enforced at compile time (#355)

`compressed2D` accepts a `Parameters` bundle for uniformity with `dense2D`, but
it is **CSR unconditionally**: `starts_` is indexed by row everywhere —
`operator()`, the array constructor's assert, the inserter, and every SpMV
kernel. Nothing ever read `Parameters::orientation`. So a `col_major`
instantiation compiled and ran, and was byte-for-byte a CSR matrix.

That is worse than merely unsupported. A caller who built a `col_major`
`compressed2D` from genuine **CSC** arrays got the **transpose**, silently:
`mult` returned Aᵀx, and `A(0,2)` returned the value that belonged at `A(2,0)` —
with no diagnostic anywhere. The container reported success while meaning
something else.

The fix is a single `static_assert`:

```cpp
static_assert(std::is_same_v<typename Parameters::orientation, tag::row_major>,
    "compressed2D is CSR (row-major) storage only; a col_major instantiation "
    "would be byte-identical to CSR and would silently read CSC input as its "
    "transpose (#355). ...");
```

**Why rejecting at compile time is free.** No `col_major compressed2D` could ever
have been *correct*, so none can exist in working code to break. The assert turns
a silent wrong-answer bug into a compile error that names the remedy: convert CSC
to CSR at the boundary, and use `mtl::trans` for a transposed sparse product.
This is the doc's clearest lesson — an unused policy parameter is not neutral; if
it *looks* configurable but is silently ignored, it manufactures wrong answers.

---

## Decision 5 — `std::vector` backing, not the dense memory block

`dense2D` owns one `contiguous_memory_block` (custom, over-aligned, stack-or-heap
by policy). `compressed2D` holds three plain `std::vector`s. The choice is
deliberate, and the opposite one for good reasons:

- **The arrays grow and are rebuilt.** Assembly and `change_dim` resize the
  arrays; `std::vector`'s amortized growth and `resize/assign/clear` are exactly
  the right primitives, whereas the dense block is sized once and reallocated
  wholesale.
- **No SIMD-alignment obligation.** The dense block is 64-byte over-aligned so
  `data()` feeds aligned GEMM loads. Sparse index traversal is gather/scatter and
  pointer-chasing; there is no aligned-vector-load contract to honor on
  `indices_`, so the machinery that guarantees it would be pure overhead.
- **Three heterogeneous arrays.** CSR is two `size_type` arrays plus one `Value`
  array of different lengths (`nnz` vs `nnz` vs `nrows+1`). That simply is not one
  block; forcing it into the single-block model would buy nothing.

So the two matrix types share concepts (a `Value`, a `Parameters` bundle, the
category/ashape traits) but not storage machinery, because their access patterns
are genuinely different.

---

## Element access: binary search and the implicit zero

Read access is `const`-only and logarithmic within a row:

```cpp
value_type operator()(size_type r, size_type c) const {
    auto begin = indices_.begin() + starts_[r];
    auto end   = indices_.begin() + starts_[r + 1];
    auto it = std::lower_bound(begin, end, c);         // requires sorted row (invariant 2)
    if (it != end && *it == c) return data_[it - indices_.begin()];
    return math::zero<value_type>();                   // structural zero
}
```

Two design notes:

- It returns a `value_type` **by value**, not a reference — a structurally absent
  entry has no storage to refer to, so it yields `math::zero<value_type>()`. Using
  the algebraic-identity `math::zero<T>()` rather than a literal `0` keeps this
  correct for custom number types (posits, intervals, matrices-of-matrices), in
  line with the library's mixed-precision purpose.
- `operator()` is for convenience and testing, not for kernels. Hot paths
  (SpMV, triangular solves) traverse `starts_/indices_/data_` directly, one row's
  contiguous run at a time — which is the whole performance point of CSR.

---

## Two on-ramps to CSR, plus a decorator

The read/build split means there is more than one way to *become* a
`compressed2D`:

- **`compressed2D_inserter`** — direct assembly with `ins[i][j] << v`, slots +
  overflow, finalize on scope exit. Best when you generate entries in roughly
  row order.
- **`coordinate2D` (COO)** — a
  [triplet list](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/coordinate2D.hpp)
  of `(row, col, value)` with unordered `insert`, then `sort()` and `compress()`,
  which sorts by `(row, col)`, accumulates duplicates, and emits a
  `compressed2D`. Best when entries arrive in arbitrary order or from I/O (e.g.
  Matrix Market).
- **`shifted_inserter`** — a
  [decorator](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/shifted_inserter.hpp)
  that wraps any inserter and offsets row/column indices, so a local element
  block can be inserted at a global offset without the caller doing the
  arithmetic — the FEM-assembly convenience layer over the same mechanism.

All three converge on the same frozen three-array form.

---

## Trait plumbing: category = sparse

Like `dense2D`, `compressed2D` connects to the dispatch machinery with two
external specializations — but the category differs, and that difference is the
point:

```cpp
template <...> struct traits::category<compressed2D<...>> { using type = tag::sparse; };
template <...> struct ashape::ashape<compressed2D<...>>   { using type = mat<Value>; };
```

`category = tag::sparse` (vs `dense2D`'s `tag::dense`) is what lets a single free
function like `mult` select the SpMV kernel by `if constexpr`/tag dispatch at
compile time, with no virtual call and no runtime type check. `ashape = mat<Value>`
places it in the same algebraic-shape algebra as the dense matrix, so a sparse
`A` and a dense `x` compose into a dense `A*x` correctly. The container carries
data and identity; the *operations* live in `mtl::operation` and choose their
implementation from the category tag.

---

## How it differs from dense2D

The two primary matrix types answer different questions, and almost every design
choice diverges accordingly:

| Aspect | `dense2D` | `compressed2D` |
|---|---|---|
| Representation | one contiguous element array | three arrays (`starts`/`indices`/`data`) |
| Storage machinery | custom `contiguous_memory_block` (aligned) | three `std::vector`s |
| What `Parameters` configures | orientation, index, dims, storage, size | size only (orientation locked to row_major) |
| Element write | direct `operator()(r,c)` reference | **none** — assemble via inserter / COO |
| Element read | O(1) offset | O(log nnz_row) binary search |
| Mutation model | in-place, resizable | batched build, then frozen |
| Absent element | n/a (all present) | structural zero, `math::zero<T>()` |
| Alignment | 64-byte for SIMD | none required |
| Category tag | `tag::dense` | `tag::sparse` |

They are siblings in the type system — same `Value`/`Parameters` shape, same
trait hooks — but not in their internals, because dense element access and sparse
row traversal are different problems.

---

## File map

| File | Role |
|---|---|
| [`mat/compressed2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/compressed2D.hpp) | the CSR container, `operator()`, raw-array access, `#355` row-major assert |
| [`mat/inserter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/inserter.hpp) | `compressed2D_inserter` (slots + overflow + finalize), `update_store`/`update_plus` |
| [`mat/shifted_inserter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/shifted_inserter.hpp) | index-offset decorator for block/FEM assembly |
| [`mat/coordinate2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/coordinate2D.hpp) | COO triplet format with `sort()` / `compress()` to CSR |
| [`mat/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/parameter.hpp) | the shared `parameters<>` bundle |
| [`math/identity.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/math/identity.hpp) | `math::zero<T>()` for the structural-zero return |
| [`operation/mult.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/mult.hpp) | SpMV and friends, selected by the `sparse` category tag |

For the dense counterpart and the policy-configuration story it tells, see the
[dense2D architecture](dense2d-architecture.md).
