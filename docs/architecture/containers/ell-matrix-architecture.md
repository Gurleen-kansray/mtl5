# ell_matrix: architecture of the ELLPACK sparse matrix

How `mtl::mat::ell_matrix` is put together, and *why* it trades memory for
regularity. Where [`compressed2D`](compressed2d-architecture.md) stores exactly
the nonzeros of each row in a variable-length run, ELLPACK (ELL) **pads every
row to a fixed width** so the whole matrix becomes a regular, dense-shaped
`nrows × width` grid of index/value pairs. That regularity is the entire point:
a branch-free, fixed-trip-count inner loop is what makes sparse matrix-vector
products vectorize cleanly on SIMD units and coalesce on GPUs.

ELL is not a build format and not a general-purpose store. It is a **re-encoding
of a finished CSR matrix** for one particular access pattern. Its design is best
read as a deliberate set of trade-offs around that single idea.

The implementation is ~130 lines in
[`include/mtl/mat/ell_matrix.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/ell_matrix.hpp).

---

## The representation: two padded `nrows × width` arrays

```text
ell_matrix<Value, Parameters = parameters<>>
   nrows_, ncols_, width_                      shape + the fixed row width
   indices_ : vector<size_type>  len nrows*width   column index of each slot
   data_    : vector<Value>      len nrows*width   value of each slot
   invalid = size_type(-1)                     sentinel for an unused slot

   Matrix          CSR rows              ELL, width = max row nnz = 2
   [ a . b ]       0: {0:a, 2:b}         row r, slot k  ->  index [r*width + k]
   [ . . c ]       1: {2:c}
   [ . d e ]       2: {1:d, 2:e}         indices_ = [ 0, 2 | 2, -1 | 1, 2 ]
                                         data_    = [ a, b | c,  0 | d, e ]
                                                     row0    row1    row2
```

Every row occupies exactly `width_` consecutive slots, whether or not it has that
many nonzeros. Row 1 has a single entry, so its second slot is *padding*: index
`invalid` (`-1`), value `math::zero<Value>()`. Element `k` of row `r` is always
at `indices_[r*width_ + k]` — no row-pointer indirection, no per-row length
lookup. The address is pure arithmetic.

---

## Decision 1 — pad every row to a fixed width

The defining choice is that all rows have the *same* number of slots. This costs
memory — the padding slots store nothing useful — and buys a **regular access
pattern**:

- **No indirection.** CSR reads `starts_[r]` to find where a row begins;
  ELL computes `r*width_` directly. The row-pointer array is gone.
- **A fixed trip count.** A per-row loop runs exactly `width_` iterations for
  every row. A kernel can unroll or vectorize it without a data-dependent bound —
  the property CSR lacks, because CSR row lengths vary and stall a vector unit
  that wants a uniform stride.
- **A dense-shaped footprint.** `indices_`/`data_` are just `nrows × width_`
  arrays; they index and stream like a dense matrix, which is why ELL is the
  classic format for GPU SpMV (one thread per row, uniform work) and SIMD kernels.

This implementation stores rows **contiguously** (`r*width_ + k`), so a single
row's slots are adjacent — the natural fit for a per-row unrolled/vectorized
inner loop on a CPU. (Some ELL variants transpose to slot-major, `k*nrows + r`,
for GPU memory coalescing; MTL5's layout is the row-contiguous form.)

---

## Decision 2 — `width_` is the max row length, and that is the whole trade

`width_` is set to the **longest** row in the source matrix:

```cpp
width_ = 0;
for (size_type i = 0; i < nrows_; ++i)
    width_ = std::max(width_, starts[i+1] - starts[i]);   // max nnz over all rows
```

This single line is where ELL's suitability is decided. Total storage is
`nrows × width_`, so the wasted space is `nrows × width_ − nnz` — the sum over
rows of `(width_ − nnz_row)`. That is small when rows are **uniform** (structured
grids, stencils, finite-difference operators: every interior row has the same
degree) and catastrophic when they are not: a single dense row forces `width_`
wide, and every other row then pads out to that width.

The design states this honestly in its own doc comment — ELL is *"good for GPU
and matrices with uniform row lengths."* It is deliberately **not** a
general-purpose sparse store; a matrix with a few dense rows belongs in CSR, or
in a hybrid ELL+overflow format (which MTL5 does not currently provide). Choosing
ELL is choosing to assert your rows are near-uniform.

---

## Decision 3 — derived from CSR, read-only, with no build path of its own

`ell_matrix` has no inserter, no `insert`, and no mutable `operator()`. Its only
populating constructor takes a `compressed2D`:

```cpp
explicit ell_matrix(const compressed2D<Value, Parameters>& crs);   // the one fill path
```

(The other constructor merely allocates an empty, all-padding grid of a given
width.) This is a deliberate placement in the pipeline: you **assemble** a matrix
in COO or an inserter, **freeze** it to CSR, and only then **re-encode** it to
ELL for a vectorized compute phase.

**Why no independent build path.** ELL's whole value is the fixed width, and you
cannot know the right width until every row's final length is known — which is
exactly the point at which you already have a CSR matrix. Building ELL
incrementally would mean either guessing `width_` up front (and reallocating the
entire padded grid when a row exceeds it) or scanning for the max anyway. Deriving
from a completed CSR does the width computation once and scatters each row's
entries left-packed into its slots. ELL is downstream of CSR by construction:

```text
COO / inserter  ->  compress  ->  CSR  ->  ell_matrix(crs)  ->  vectorized SpMV
```

---

## Decision 4 — left-packed slots and early-exit access

Entries are packed to the **left** of each row; padding is always on the right.
`operator()` exploits that to stop early:

```cpp
value_type operator()(size_type r, size_type c) const {
    for (size_type k = 0; k < width_; ++k) {
        size_type idx = indices_[r*width_ + k];
        if (idx == invalid) break;              // packed left => no real entry past here
        if (idx == c) return data_[r*width_ + k];
    }
    return math::zero<Value>();                 // absent -> structural zero
}
```

Because a row's real entries are a contiguous left prefix, the first `invalid`
slot marks the end of that row's data, and the scan breaks — so a lookup costs
O(nnz_row), not O(width_), in the common case. The scan is linear rather than a
CSR-style binary search because `width_` is small and bounded by design; a branchy
binary search would not pay off over a handful of contiguous slots, and the linear
form is what a compiler can unroll. As with every MTL5 matrix, an absent element
returns `math::zero<value_type>()`, so custom number types get the correct
additive identity.

Entries within a row keep the column order they had in CSR (ascending), since the
constructor copies them in `starts[i]..starts[i+1]` order — a property a kernel
may rely on even though `operator()` does not require it.

---

## Decision 5 — row-major only, enforced early (#355, lower severity)

Like its CSR source, `ell_matrix` accepts a `Parameters` bundle but is row-padded
**unconditionally**: every access is `indices_[r*width_ + k]`, and nothing ever
read `Parameters::orientation`. A `col_major` instantiation was therefore accepted
silently while being byte-for-byte the row-major layout — the same *inert-tag*
defect as `compressed2D`. A compile-time `static_assert` rejects it:

```cpp
static_assert(std::is_same_v<typename Parameters::orientation, tag::row_major>,
    "ell_matrix is row-padded ELLPACK storage only; a col_major instantiation "
    "would be byte-identical to the row-major layout and silently misdescribe it (#355). ...");
```

The severity here is genuinely **lower** than in CSR, and the source comment is
careful to say why rather than assert it by analogy: ELL's *only* fill path is the
`compressed2D` constructor, which for `col_major` parameters is *already* a compile
error (CSR's own `static_assert` fires). So no `col_major` ELL can be populated
today, and none can be returning wrong answers. What a `col_major` ELL could still
do is be *constructed empty while claiming a layout it does not implement* — and
the instant someone adds an inserter or a setter, that latent hole reopens
silently. The assert closes it **now, at the point of misuse**, so a future
contributor gets a clear diagnostic on `ell_matrix` rather than a confusing one
about `compressed2D`. It is a fix applied before the bug can bite, not after.

---

## Storage: `std::vector`, regular in shape but not over-aligned

Both arrays are plain `std::vector` (`assign`-ed to `nrows*width_` on
construction), like the other sparse containers and unlike
[`dense2D`](dense2d-architecture.md)'s custom, 64-byte-aligned
`contiguous_memory_block`. Worth being precise about what ELL does and does not
guarantee:

- It guarantees **structural regularity** — the fixed stride `width_` that a
  vectorized kernel needs to compute addresses without branches.
- It does **not** guarantee **alignment** — `indices_.data()` and `data_.data()`
  carry only `std::vector`'s default alignment. Aligned loads, if a kernel wants
  them, are a separate concern (an aligned allocator or padding the leading
  dimension), not something the container currently provides.

Regularity and alignment are independent properties; ELL supplies the first,
which is the one that distinguishes it from CSR. The second remains available to
layer on if a tuned ELL kernel needs it.

---

## Trait plumbing: category = sparse

The same two external specializations as the other sparse types:

```cpp
template <...> struct traits::category<ell_matrix<...>> { using type = tag::sparse; };
template <...> struct ashape::ashape<ell_matrix<...>>   { using type = mat<Value>; };
```

`category = tag::sparse` and `ashape = mat<Value>` give ELL the same type-system
identity as CSR and COO. Note that MTL5 does not yet ship a *tuned* ELL SpMV
kernel — the format is present as storage (and is consumed by the
[`spy`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/io/spy.hpp)
visualizers), with `ref_indices()`/`ref_data()` exposing the raw padded arrays
for a kernel to stream. The regular layout is the enabling *precondition* for such
a kernel; the trait plumbing is what will let it be selected by category when it
lands.

---

## Where ELL sits among the formats

ELL is a fourth core format, but it belongs to a different phase than the
COO→CSR build chain — it is a compute-oriented **re-encoding** of CSR, chosen when
row lengths are uniform enough to make the padding cheap:

| | [`dense2D`](dense2d-architecture.md) | [`coordinate2D`](coordinate2d-architecture.md) (COO) | [`compressed2D`](compressed2d-architecture.md) (CSR) | `ell_matrix` (ELL) |
|---|---|---|---|---|
| Optimized for | dense compute | building | general SpMV / solvers | **regular / vectorized SpMV** |
| Storage | one aligned block | triplet vector | three arrays | two padded `nrows×width` arrays |
| Per row | full row | n/a | exact nnz, variable | fixed `width_`, padded |
| Build path | in place | `insert` | inserter / COO | **none — from CSR only** |
| Read cost | O(1) | O(nnz) scan | O(log nnz_row) | O(nnz_row), fixed-trip loop |
| Best when | dense | any-order fill | general sparse | rows near-uniform length |
| Weakness | — | not queryable | irregular vector access | memory blows up on skewed rows |

The intended flow when ELL is appropriate: **COO/inserter → compress → CSR →
`ell_matrix(crs)` → vectorized SpMV**. ELL owns only that last re-encoding step,
and only earns its keep when the rows are uniform.

---

## File map

| File | Role |
|---|---|
| [`mat/ell_matrix.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/ell_matrix.hpp) | the ELL container: padded arrays, CSR constructor, early-exit `operator()`, `#355` assert |
| [`mat/compressed2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/compressed2D.hpp) | the CSR source ELL is built from |
| [`mat/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/parameter.hpp) | the shared `parameters<>` bundle |
| [`math/identity.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/math/identity.hpp) | `math::zero<T>()` for padding and absent-element returns |
| [`io/spy.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/io/spy.hpp) | current consumer of ELL (sparsity visualization) |

For the source format and the rest of the family, see the
[compressed2D](compressed2d-architecture.md),
[coordinate2D](coordinate2d-architecture.md), and
[dense2D](dense2d-architecture.md) architecture docs.
