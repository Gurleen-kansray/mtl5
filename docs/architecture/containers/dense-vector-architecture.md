# dense_vector: architecture of the primary dense vector

How `mtl::vec::dense_vector` is put together, and *why*. In its machinery it is
the one-dimensional sibling of [`dense2D`](dense2d-architecture.md) — same owned
memory block, same policy-bundle pattern, same expression-template assignment,
same fixed/dynamic-size trick. This document does not re-derive that shared
foundation; it summarizes it and points to the dense2D doc, then spends its depth
on what a *vector* needs that a matrix does not: an orientation that means **row
vs column**, an algebraic shape that distinguishes the two, and a unit-stride
contract.

The implementation is ~300 lines in
[`include/mtl/vec/dense_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/dense_vector.hpp).

---

## The class in one picture

```text
dense_vector<Value, Parameters = parameters<>>
   │
   │  vec::parameters bundles FOUR compile-time policies (one fewer than a matrix):
   │     orientation      col_major (column vector, default) | row_major (row vector)
   │     dimensions_type  fixed::dimension<N> | non_fixed::dimension
   │     storage          on_heap | on_stack
   │     size_type        std::size_t
   │     (no index_type — vectors are 0-based, always)
   │
   ├── memory_type mem_   contiguous_memory_block<Value, storage, static_size>
   └── dim_type    dim_   [[no_unique_address]] — 0 bytes when fixed
```

Two data members, one of them free for fixed-size vectors. Element access is
`v(i)` *and* `v[i]`; storage is always contiguous, stride 1.

---

## The shared foundation (see the dense2D doc)

Everything below is identical in mechanism to `dense2D` and is treated in depth
there; only the one-line rationale is repeated here:

| Shared decision | Mechanism | Detail in |
|---|---|---|
| Composition over inheritance | holds a `contiguous_memory_block` by value; copy/destroy `= default` | [dense2D §Decision 2](dense2d-architecture.md#decision-2--composition-over-inheritance) |
| One memory type, stack or heap | `std::conditional_t` + `if constexpr`, 64-byte over-aligned, own/external/view | [dense2D §Decision 3](dense2d-architecture.md#decision-3--one-memory-type-stack-or-heap-chosen-by-if-constexpr) |
| Fixed vs dynamic size at zero cost | `[[no_unique_address]] dim_`, `if constexpr (is_fixed)` | [dense2D §Decision 4](dense2d-architecture.md#decision-4--fixed-vs-runtime-dimensions-at-zero-cost-when-fixed) |
| `if constexpr` bounds check | gated on `bounds_checking`, no branch under `NDEBUG` | [dense2D §Element access](dense2d-architecture.md#element-access-and-the-compile-time-bounds-check) |
| Expression assignment, fused, mixed-precision | concept-constrained `operator=`, `parallel_ewise`, `static_cast<Value>` | [dense2D §Expression templates](dense2d-architecture.md#expression-templates-assignment-and-mixed-precision) |
| Move leaves a valid-empty source | member move + `set_size(0)` on the dynamic dim | [dense2D §Move](dense2d-architecture.md#move-semantics-and-the-valid-empty-state) |

The one-dimensional `vec::dimension` (`fixed::dimension<N>` exposing a `constexpr
value`, or `non_fixed::dimension` holding one `size_t`) is the vector analog of
the matrix's 2-D dimensions; `static_size = dim_type::value` feeds the memory
block exactly as in `dense2D`. If you have read the dense2D document, you already
understand the plumbing of `dense_vector`. The rest of this page is the delta.

---

## Decision 1 — orientation means row-vs-column vector, not memory layout

This is the one genuinely different idea. For a matrix, `tag::row_major` /
`tag::col_major` selects the **storage order** — which axis is contiguous. A
vector is one-dimensional; its storage is always a contiguous stride-1 run, and
there is no layout choice to make. So `dense_vector` reuses the *same two tags*
for a different meaning:

- `col_major` (the **default**) — a **column vector**, shape *n × 1*.
- `row_major` — a **row vector**, shape *1 × n*.

The bytes are identical either way; orientation here is an **algebraic**
attribute, not a physical one. The default is `col_major` because a bare vector
in linear algebra is a column.

This is a deliberate overload of the orientation tags, and worth flagging as
such: the same `tag::col_major` names "column-contiguous storage" on a matrix and
"column vector" on a vector. The reuse keeps one small tag vocabulary across the
type system, and both meanings are about the same underlying question — which way
does this object face — but they are not the same property, and the vector case
touches no memory decision at all.

---

## Decision 2 — the algebraic shape is `rvec` / `cvec`, not `mat`

Where `dense2D` always classifies as `ashape::mat<Value>`, a `dense_vector`
classifies by orientation:

```cpp
template <...> struct ashape<dense_vector<Value, Parameters>> {
    using type = std::conditional_t<
        std::is_same_v<typename Parameters::orientation, tag::col_major>,
        cvec<Value>,    // column vector
        rvec<Value>>;   // row vector
};
```

MTL5's algebraic-shape taxonomy (`include/mtl/traits/ashape.hpp`) is a small
classification used as a **secondary dispatch axis** alongside the category tag:

```text
universe
├── scal                 (any plain scalar — the default)
└── nonscal
    ├── rvec<Value>       row vector
    ├── cvec<Value>       column vector
    └── mat<Value>        matrix
```

**Why the row/column split is load-bearing.** The distinction is exactly what an
operation needs to tell products apart by shape: a *row × column* is an inner
product yielding a **scalar** (`dot`), a *column × row* is an outer product
yielding a **matrix**, and matrix-vector products must know whether the vector is
the *n × 1* operand on the right or the *1 × n* operand on the left. Collapsing
both into a single "vector" shape would make those cases ambiguous. By promoting
orientation into the `rvec`/`cvec` classification, the type of a vector carries
enough information for shape-aware dispatch to pick the right result — the whole
reason the taxonomy separates the two in the first place. (The category tag stays
`tag::dense`, same as `dense2D`; ashape is the *orthogonal* axis that adds the
row/column/matrix distinction on top of dense/sparse.)

---

## Decision 3 — a vector presents a matrix shape

Because a vector is algebraically an *n × 1* or *1 × n* matrix, `dense_vector`
answers `num_rows()` / `num_cols()` accordingly, decided at compile time from the
orientation:

```cpp
size_type num_rows() const { if constexpr (col_major) return size(); else return 1; }
size_type num_cols() const { if constexpr (col_major) return 1;      else return size(); }
```

This lets a vector flow through generic code written against a matrix-shaped
interface — a `gemv` that treats its operand as *n × 1*, a printer, a shape check
— without a separate adapter. `size()` remains the natural 1-D length; the
`num_rows`/`num_cols` pair is the 2-D face the vector shows to matrix-generic
algorithms. Both are free functions too (`vec::num_rows`, `vec::num_cols`), so
argument-dependent lookup finds them in generic contexts.

---

## Decision 4 — always unit stride; strided access is a different type

```cpp
static constexpr size_type stride() { return 1; }
```

`dense_vector` is, by definition, contiguous and unit-stride. It exposes
`data()`/`begin()`/`end()` over that tight run, so it is a direct BLAS-1 operand
and a drop-in for `std::` range algorithms. Non-unit stride — a column of a
matrix, an aliased sub-range, every k-th element — is deliberately **not** a mode
of this type; it is the job of a separate
[`strided_vector_ref`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/strided_vector_ref.hpp).
This mirrors `dense2D`'s choice to keep `ldim` equal to the fast-axis extent and
push padded/strided views to other types: the common container stays maximally
simple and contiguous, and the irregular cases live where they cannot complicate
the hot path. `stride()` is `constexpr 1` so any kernel templated on a
vector-like type can specialize the unit-stride case with no runtime test.

---

## Decision 5 — dual `()` and `[]` access

A `dense_vector` offers both spellings, and `[]` simply forwards to `()`:

```cpp
reference operator()(size_type i);              // MTL functional access, matches A(r,c)
reference operator[](size_type i) { return operator()(i); }   // idiomatic vector subscript
```

`operator()` is the form shared with matrices and expressions — the same call
syntax `A(r,c)` uses — which keeps generic code that indexes "an element of an
operand" uniform across vectors and matrices. `operator[]` is the spelling vector
code naturally reaches for. Routing `[]` through `()` means the bounds check and
any future access policy live in exactly one place; the two are never
inconsistent.

---

## A trimmed parameter bundle

The vector `parameters<>` has **four** axes to the matrix's five: it drops
`index_type`. There is no `c_index`/`f_index` choice because MTL5 vectors are
0-based, full stop — a 1-based vector was neither needed nor worth a policy. It
keeps the axes that do matter for a vector: orientation (row/column), dimensions
(fixed/dynamic), storage (stack/heap), and size type. The same invariant guard as
the matrix bundle is present — stack storage requires a fixed size:

```cpp
static_assert(!std::is_same_v<Storage, tag::on_stack> || is_fixed,
    "Stack storage requires fixed-size dimensions");
```

So `dense_vector<double>` is a heap, dynamic-size, **column** vector — the
sensible default — and every other configuration is a named deviation from it.

---

## How it relates to dense2D

Same foundation, vector-specific face:

| Aspect | [`dense2D`](dense2d-architecture.md) | `dense_vector` |
|---|---|---|
| Dimensionality | 2-D (rows × cols) | 1-D (length) |
| `Parameters` axes | 5 (incl. `index_type`) | 4 (no `index_type`) |
| What orientation means | **memory layout** (row/col-major storage) | **shape** (row vs column vector) |
| ashape | always `mat<Value>` | `cvec<Value>` or `rvec<Value>` by orientation |
| Access | `A(r,c)` | `v(i)` and `v[i]` |
| Stride / layout | tight, `ldim` = fast axis | tight, `stride() == 1` |
| Irregular variant | strided/padded views (separate) | `strided_vector_ref` (separate) |
| Memory block, expr templates, fixed-size trick | — | **identical** |
| Category tag | `tag::dense` | `tag::dense` |

They are intentionally the same object in miniature: learn one and the other's
internals are already familiar, leaving only the row/column-shape semantics to
absorb.

---

## File map

| File | Role |
|---|---|
| [`vec/dense_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/dense_vector.hpp) | the vector class, free functions, `rvec`/`cvec` ashape specialization |
| [`vec/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/parameter.hpp) | the 4-axis vector `parameters<>` bundle |
| [`vec/dimension.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/dimension.hpp) | `fixed::dimension<N>` / `non_fixed::dimension` |
| [`vec/strided_vector_ref.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/strided_vector_ref.hpp) | the non-unit-stride vector view |
| [`traits/ashape.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/traits/ashape.hpp) | the `scal`/`rvec`/`cvec`/`mat` shape taxonomy |
| [`detail/contiguous_memory_block.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/contiguous_memory_block.hpp) | the storage shared with `dense2D` |
| [`detail/ewise.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/ewise.hpp) | `parallel_ewise` used by assignment |

For the two-dimensional counterpart and the full treatment of the shared
machinery, see the [dense2D architecture](dense2d-architecture.md) doc.
