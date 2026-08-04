# unit_vector: architecture of the basis-vector factory

How `mtl::vec::unit_vector` is put together, and *why* it is the shortest file in
the container family — because it is not a container at all. Every other type
documented here is a class with storage, traits, and an access interface.
`unit_vector` is a **factory function**: it builds and returns a fully-materialized
[`dense_vector`](dense-vector-architecture.md) with a `1` at position `k` and `0`
everywhere else. The design decision worth documenting is the one it *didn't* make
— it is not a lazy synthetic basis-vector *type*, and this page is about why.

The implementation is ~20 lines in
[`include/mtl/vec/unit_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/unit_vector.hpp).

---

## The whole thing

```cpp
template <typename Value = double>
dense_vector<Value> unit_vector(std::size_t n, std::size_t k) {
    assert(k < n && "unit_vector: k must be less than n");
    dense_vector<Value> v(n, math::zero<Value>());   // all zeros
    v(k) = math::one<Value>();                        // ...except position k
    return v;                                         // a normal, owning dense_vector
}
```

That is the entire type. There is no class, no `Parameters`, no traits, no
iterator. `unit_vector(n, k)` is the basis vector *eₖ* ∈ ℝⁿ, returned as an
ordinary vector the caller owns.

---

## Decision 1 — a factory that materializes, not a lazy synthetic type

The obvious alternative — and the one some libraries (and MTL4-era thinking) reach
for — is a dedicated `unit_vector<Value>` *class* that stores only `n` and `k` and
computes elements on demand: `operator()(i)` returns `math::one` if `i == k` else
`math::zero`, in O(1) space. MTL5 deliberately does **not** do that. It allocates
the full `n` elements and writes them.

**Why materialize.** A lazy basis-vector type would save `n-1` stored zeros, but at
a real cost in surface area: it would need its own `category`/`ashape` traits, an
`operator()`, very likely an iterator, participation in the expression-template
system, and correct behavior everywhere an operation consumes a vector — an entire
integration surface, maintained forever, to avoid writing some zeros. And the
saving rarely matters: *eₖ* is used in small numbers and almost always as an
operand to a much larger operation — extracting a column with `A * eₖ`, seeding a
canonical basis in a solver, probing an operator. The O(n) materialization is
dominated by the O(nnz) or O(n²) work around it. MTL5 judges that trade decisively:
a negligible allocation is cheaper — in code, not just cycles — than a whole type.
This is a *negative-space* decision, the choice not to build something, and it is
as much a part of the architecture as the classes are.

---

## Decision 2 — build on `dense_vector`, inherit its integration for free

Because the factory returns a plain `dense_vector`, it needs **zero** new
integration. The result already:

- classifies as `category = tag::dense` and `ashape = cvec`/`rvec`,
- flows through every operation, expression, and algorithm that accepts a vector,
- is mutable, copyable, assignable, and printable,

…because it *is* a `dense_vector`, with all of that treatment
[already documented](dense-vector-architecture.md). A lazy `unit_vector` type would
have had to re-earn every one of those; the factory gets them by construction. This
is the entire payoff of the decision above: the cheapest way to make a value work
everywhere in the library is to make it one of the library's existing types.

---

## Decision 3 — algebraic identities, not literal `0` and `1`

The fill uses `math::zero<Value>()` and the set uses `math::one<Value>()`, never a
literal `0` or `1`:

```cpp
dense_vector<Value> v(n, math::zero<Value>());
v(k) = math::one<Value>();
```

For `double` these are `0.0` and `1.0`, but the point is the general case. MTL5
targets custom number types — posits, LNS, interval and block types — for which the
additive and multiplicative identities are not always the literal tokens `0` and
`1`, and may be nontrivial objects. Routing through `math::zero<T>()` /
`math::one<T>()` makes *eₖ* correct for any such `Value`, consistent with how the
rest of the library expresses identities. A basis vector is defined *by* those two
identities, so it is exactly the place to use them rather than magic numbers.

---

## Decision 4 — value semantics: a first-class, owning result

`unit_vector` returns a `dense_vector` **by value** (moved / RVO'd out). The caller
receives a normal owning vector: it can be mutated, stored, resized, passed to any
sink. There is no lifetime tie to anything, no aliasing, no read-only constraint.

This is worth contrasting with the family's *other* "not a new container" type,
[`strided_vector_ref`](strided-vector-ref-architecture.md), because they solve
opposite problems in opposite ways:

- `strided_vector_ref` **avoids a copy** by *borrowing* — it is a non-owning handle
  to storage that already exists.
- `unit_vector` **avoids a type** by *producing* — it manufactures standard owning
  storage that did not exist yet.

One is a view over data; the other is a constructor of data. Neither is a new
container, and the two mark the two clean ways to *not* add one: reference what
exists, or build what already has a home.

---

## A named default: `Value = double`

`template <typename Value = double>` means `unit_vector(n, k)` yields a
`dense_vector<double>` with no type argument — the overwhelmingly common case —
while `unit_vector<float>(n, k)` or a custom type stays one token away. The same
"sensible default, explicit deviation" instinct that runs through the
`Parameters` bundles, applied to a single function's element type.

---

## The one case the trade would lose

Honesty about the tradeoff: if an algorithm needed *many* unit vectors in a tight
loop — sweeping *eₖ* for every `k` to form a full basis, reallocating each time —
the per-call O(n) allocation could add up where a lazy O(1) *eₖ* would not. MTL5
accepts that edge because it is not the common pattern (such code typically fuses
the loop and never forms the vectors explicitly), and because the remedy is local:
write the loop against raw indices, not a stream of materialized basis vectors. The
factory optimizes the frequent, incidental use of *eₖ* and declines to complicate
the library for the rare, hot one.

---

## File map

| File | Role |
|---|---|
| [`vec/unit_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/unit_vector.hpp) | the factory function |
| [`vec/dense_vector.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/dense_vector.hpp) | the owning vector it returns |
| [`math/identity.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/math/identity.hpp) | `math::zero<T>()` / `math::one<T>()` that define *eₖ* |

For the type it materializes, see the
[dense_vector architecture](dense-vector-architecture.md) doc; for the other,
opposite way the library avoids adding a container, the
[strided_vector_ref architecture](strided-vector-ref-architecture.md) doc.
