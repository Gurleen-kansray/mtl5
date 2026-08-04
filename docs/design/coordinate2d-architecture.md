# coordinate2D: architecture of the COO sparse matrix

How `mtl::mat::coordinate2D` is put together, and *why* it is almost the exact
inverse of [`compressed2D`](compressed2d-architecture.md). If CSR is the frozen,
read-optimized form, COO (Coordinate format) is the **write-optimized** form: an
append-only bag of `(row, column, value)` triplets that accepts entries in any
order, as fast as a `push_back`, and defers *all* structure — sorting,
deduplication, row pointers — to a single `compress()` step that produces a
`compressed2D`.

The organizing idea is **staging**. `coordinate2D` is not a matrix you compute
with; it is the buffer you accumulate *into* while building a matrix, before
handing a compressed form to the solvers. Every design choice follows from being
the cheap-to-fill, expensive-to-query end of the sparse pipeline.

The implementation is ~150 lines in
[`include/mtl/mat/coordinate2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/coordinate2D.hpp).

---

## The representation: one array of triplets

Where CSR uses three parallel arrays (structure-of-arrays, tuned for row
traversal), COO uses a single array of triplets (array-of-structs, tuned for
appending):

```text
coordinate2D<Value, Parameters = parameters<>>
   nrows_, ncols_                                   logical shape
   entries_ : vector<tuple<size_type,size_type,Value>>   the (row, col, value) list
   sorted_  : bool                                  is entries_ ordered by (row,col)?

   Example  [ a . b ]     entries_ = [ (0,0,a), (0,2,b), (1,2,c), (2,1,d), (2,2,e) ]
            [ . . c ]                 ...in ANY order, with duplicates allowed:
            [ . d e ]                 [ (2,2,e1), (0,0,a), (2,2,e2), ... ]  is fine
```

Keeping `(r, c, v)` together in one tuple is what makes both core operations
trivial: an insert is one `emplace_back`, and ordering is a single `std::sort`
that carries the value along with its coordinates automatically. There are no
invariants to maintain on insert — not sortedness, not uniqueness, not row
grouping. That absence *is* the design.

---

## Decision 1 — insertion is an unordered, duplicate-tolerant O(1) append

```cpp
void insert(size_type r, size_type c, Value v) {
    assert(r < nrows_ && c < ncols_);
    entries_.emplace_back(r, c, v);   // append; no search, no ordering
    sorted_ = false;                  // the list may no longer be ordered
}
```

`insert` does no lookup and no merge. A second write to an existing `(r,c)` does
**not** overwrite or combine — it appends a second triplet, and the two are
reconciled later. This is exactly right for the workloads COO exists to serve:

- **Finite-element / finite-volume assembly**, where the same global `(i,j)`
  receives contributions from many elements, generated in element order, not
  matrix order.
- **Matrix I/O**, where a Matrix Market file lists entries in whatever order the
  producer chose.
- **Unstructured fill**, where you simply don't know the final column pattern of
  a row until you've seen every contribution.

Making `insert` O(1) and order-free means assembly costs one `push_back` per
contribution and nothing else. `reserve(nnz)` lets a caller pre-size `entries_`
to avoid reallocation when the count is known.

---

## Decision 2 — defer all structure to `compress()`

Nothing about `coordinate2D` tries to be incrementally organized. The `sorted_`
flag merely *records* whether the list happens to be ordered (it starts `true`
for the empty matrix and flips to `false` on the first `insert`); `sort()` will
order the triplets in place by `(row, col)` on demand. But the real work —
turning an unordered, duplicate-laden triplet bag into a valid CSR matrix — is
concentrated in one method, `compress()`.

**Why defer.** You cannot correctly deduplicate or lay out a row until you have
seen *all* of its contributions. Any per-insert organization would either be
redundant (re-sorting on every append) or premature (committing to a row layout
that the next insert invalidates). Concentrating the work in `compress()` means
it is done once, over the complete data, at the moment the caller declares the
matrix finished — the same "assemble, then freeze" philosophy the
[inserter](compressed2d-architecture.md#decision-2--the-inserter-slots--overflow--raii-finalize)
follows, expressed here as an explicit conversion rather than an RAII finalize.

---

## Decision 3 — `compress()`: sort, accumulate, prefix-sum, and it's `const`

`compress()` is the bridge to the read-optimized world, and it is non-destructive:

```cpp
compressed2D<Value, Parameters> compress() const {
    // 1. sort a COPY by (row, col) -- leaves *this untouched
    std::vector<triplet_type> sorted_entries(entries_);
    std::sort(sorted_entries.begin(), sorted_entries.end(), by_row_then_col);

    // 2. run-length merge duplicates, counting entries per row
    std::vector<size_type> starts(nrows_ + 1, 0), indices;
    std::vector<Value> data;
    for (size_type k = 0; k < sorted_entries.size(); ) {
        auto [r, c, v] = sorted_entries[k]; Value acc = v; ++k;
        while (k < N && same (r,c) as sorted_entries[k]) { acc += get<2>(sorted_entries[k]); ++k; }
        indices.push_back(c); data.push_back(acc); starts[r + 1]++;
    }

    // 3. prefix-sum the per-row counts into CSR row pointers
    for (size_type i = 0; i < nrows_; ++i) starts[i + 1] += starts[i];

    return compressed2D<Value, Parameters>(nrows_, ncols_, data.size(),
                                           starts.data(), indices.data(), data.data());
}
```

Three properties are deliberate:

- **It works on a copy and is marked `const`.** Compressing does not consume the
  COO buffer. A caller can `compress()`, keep accumulating more entries, and
  `compress()` again — useful when a matrix is built and solved in stages. The
  cost is one copy of the triplet array; the benefit is a reusable, side-effect-
  free staging object.
- **It sorts unconditionally.** `compress()` does not consult `sorted_` to skip
  the sort. That keeps it correct regardless of the object's state at the cost of
  a redundant sort when the list is already ordered — a simplicity-over-cleverness
  trade for a step that already dominates by being O(nnz log nnz).
- **It builds row pointers by counting then prefix-summing** — the standard COO→CSR
  construction: tally how many distinct entries land in each row (`starts[r+1]++`),
  then turn counts into offsets with a cumulative sum. The result satisfies every
  CSR invariant (sorted rows, unique entries, `starts[nrows] == nnz`) by
  construction, so the `compressed2D` it returns is immediately valid.

---

## Decision 4 — duplicates accumulate (assembly semantics, baked in)

The run-length merge does `acc += ...`: repeated `(r,c)` triplets are **summed**.
COO's `compress()` therefore always has additive-assembly semantics, matching the
finite-element use case where duplicates *are* separate physical contributions to
the same entry.

This is a narrower contract than the inserter's, which chooses between
`update_store` (overwrite) and `update_plus` (accumulate) via a policy type. COO
does not offer that choice: it accumulates, full stop. The division of labor is
intentional — the inserter is the general, policy-parameterized front door; COO
is the lightweight accumulation buffer whose one semantic is "sum contributions."
If last-write-wins is what you need, that is the inserter's `update_store`, not
COO.

---

## Decision 5 — `operator()` is an honest O(nnz) linear scan

Random element access exists, but it is deliberately naive:

```cpp
value_type operator()(size_type r, size_type c) const {
    Value sum = math::zero<Value>();
    for (const auto& [er, ec, ev] : entries_)      // scan EVERY triplet
        if (er == r && ec == c) sum += ev;          // sum matches (pre-compress)
    return sum;
}
```

It scans the whole triplet list, summing every match — O(nnz) per lookup, and it
reflects the accumulate semantics even before `compress()` runs. The comment in
the source is candid: this is *"mainly for testing."* COO is not a queryable
matrix, and the interface says so by not pretending random access is cheap. Code
that needs to *read* the matrix compresses to CSR first, where access is
O(log nnz_row); code that needs to *build* it stays in COO. Like the sibling
containers, an absent entry yields `math::zero<value_type>()`, so custom number
types (posits, intervals, ...) get the correct additive identity.

---

## COO vs the inserter: two on-ramps to CSR

MTL5 offers two ways to assemble a `compressed2D`, and `coordinate2D` is one of
them. They are complementary, not redundant:

| | `coordinate2D` (COO) | `compressed2D_inserter` |
|---|---|---|
| Buffer | one triplet vector | flat per-row slots + `std::map` overflow |
| Best input order | any order | roughly row-major |
| Duplicate policy | always sum | `update_store` or `update_plus` (choose) |
| Reusable after build | yes — `compress()` is `const` | no — RAII, finalizes once at scope exit |
| Extra tuning | none | `slot_size` per row |
| Natural source | I/O, unstructured fill | direct programmatic assembly |

Choose COO when entries arrive out of order or from a file and you want the
simplest possible staging buffer; choose the inserter when you generate entries
in row order and want slot tuning or a choice of duplicate policy. Both converge
on the same frozen CSR form.

One inherited constraint: because `compress()` instantiates
`compressed2D<Value, Parameters>`, a `col_major` `coordinate2D` compiles for
*accumulation* but fails to `compress()` — it trips CSR's row-major
`static_assert` (#355). In practice `coordinate2D`, like CSR, is row-major.

---

## Trait plumbing: category = sparse

The same two external specializations as its siblings, with the sparse category:

```cpp
template <...> struct traits::category<coordinate2D<...>> { using type = tag::sparse; };
template <...> struct ashape::ashape<coordinate2D<...>>   { using type = mat<Value>; };
```

`category = tag::sparse` and `ashape = mat<Value>` mean COO carries the same
type-system identity as CSR. In practice, though, you rarely dispatch operations
on a `coordinate2D` — it is a staging type, and the intended flow is to
`compress()` first and let the operation layer work on the CSR result. The traits
are there for uniformity and so generic code that inspects category/ashape treats
all three matrix types consistently.

---

## The three matrix formats at a glance

`coordinate2D` is the third corner of MTL5's core matrix triangle. Each optimizes
a different phase:

| | [`dense2D`](dense2d-architecture.md) | `coordinate2D` (COO) | [`compressed2D`](compressed2d-architecture.md) (CSR) |
|---|---|---|---|
| Optimized for | dense compute | **building** | reading / SpMV |
| Storage | one aligned block | one triplet vector | three arrays |
| Insert cost | O(1) in place | **O(1) append, any order** | not supported |
| Read cost | O(1) | O(nnz) scan | O(log nnz_row) |
| Duplicates | n/a | allowed, summed on compress | not stored |
| Ordering held | n/a | none until `sort`/`compress` | rows sorted (invariant) |
| Role | numerical workhorse | assembly staging | solver / kernel input |

The intended lifecycle for a sparse matrix is **COO (or inserter) → compress →
CSR → solve**: accumulate cheaply and in any order, freeze once, then compute.
`coordinate2D` owns the first step.

---

## File map

| File | Role |
|---|---|
| [`mat/coordinate2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/coordinate2D.hpp) | the COO container: `insert`, `sort`, `compress`, scan-based `operator()` |
| [`mat/compressed2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/compressed2D.hpp) | the CSR form `compress()` produces |
| [`mat/inserter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/inserter.hpp) | the alternative CSR builder (slots + overflow, updater policy) |
| [`mat/parameter.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/parameter.hpp) | the shared `parameters<>` bundle |
| [`math/identity.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/math/identity.hpp) | `math::zero<T>()` for the absent-element return |

For the read-optimized target of `compress()` and the dense workhorse, see the
[compressed2D](compressed2d-architecture.md) and
[dense2D](dense2d-architecture.md) architecture docs.
