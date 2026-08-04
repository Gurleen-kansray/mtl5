# strided_vector_ref: architecture of the non-owning strided view

How `mtl::vec::strided_vector_ref` is put together, and *why*. Every container
documented so far — [`dense2D`](dense2d-architecture.md),
[`dense_vector`](dense-vector-architecture.md), the sparse types — **owns** its
storage. `strided_vector_ref` owns nothing. It is a **view**: a pointer, a length,
and a stride, presenting somebody else's memory as a vector. It is the type
[`dense_vector`](dense-vector-architecture.md#decision-4--always-unit-stride-strided-access-is-a-different-type)
explicitly points to when it declares that non-unit stride is *not* its job — the
irregular case, given its own small, purpose-built home.

The implementation is ~150 lines in
[`include/mtl/vec/strided_vector_ref.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/strided_vector_ref.hpp),
and the file's own header notes it is *"simplified from MTL4's 275-line CRTP
version: no ownership, no clone, no CRTP."*

---

## The type in one picture

```text
strided_vector_ref<Value>            // ONE template parameter — no Parameters bundle
   data_   : Value*                  borrowed pointer (NOT owned)
   size_   : size_type               logical length
   stride_ : size_type               step between elements

   element i  ==  data_[i * stride_]

   Viewing column 2 of a 3x4 row-major matrix (ldim = 4):
     matrix memory:  [ .  .  x  . | .  .  y  . | .  .  z  . ]
                            └────────┘─────────┘
     strided_vector_ref(data + 2, length = 3, stride = 4)  ->  [ x, y, z ]
```

Three words of state, no allocation, no `Parameters`. Copying a
`strided_vector_ref` copies the handle — pointer, length, stride — not the data.

---

## Decision 1 — a view, not a container: it owns nothing

The defining property, and the one that flips every assumption from the owning
containers. `strided_vector_ref` holds a raw `Value*` it did not allocate and will
not free. There is no `contiguous_memory_block`, no `std::vector`, no destructor
logic — the three members are trivially copyable, and the type has no user-defined
special members at all.

- **Copy is aliasing, not duplication.** Copying yields a second view of the *same*
  memory. Two `strided_vector_ref`s can name overlapping data; writing through one
  is visible through the other. This is the intended semantics — it is a reference.
- **The lifetime contract is the caller's.** Because it borrows, the viewed storage
  **must outlive the view**. A `strided_vector_ref` into a `dense2D` that is then
  destroyed dangles, exactly as `std::span` or `std::string_view` would. The type
  buys zero-copy access at the cost of that discipline, and does not try to soften
  it with reference counting or ownership modes — MTL4's version carried clone and
  ownership machinery; MTL5 drops all of it, on the principle that a view should be
  a fat pointer and nothing more.

Everything else in the design follows from "it is a borrowed handle."

---

## Decision 2 — the stride is the reason it exists

Element `i` lives at `data_[i * stride_]`, and that multiply is the whole point.
With `stride_ == 1` this would be an ordinary contiguous span; with `stride_ > 1`
it presents **non-contiguous** memory as a logically contiguous vector. The
motivating case, called out in the file header, is extracting a **column of a
row-major `dense2D`** without copying:

```text
row-major dense2D: row r, col c  lives at  data[r * ldim + c]   (ldim = num_cols)
column c is the set {c, ldim+c, 2*ldim+c, ...}  ->  stride = ldim
   strided_vector_ref(A.data() + c, A.num_rows(), A.get_ldim())
```

A column's elements are `ldim` apart in memory; a stride of `ldim` makes them a
vector. This is the exact complement of `dense_vector`'s decision to be *always*
unit-stride: `dense_vector` keeps the common owning container tight and pushes the
irregular case out, and `strided_vector_ref` is where it lands. Together they form
a pair — **owning + unit-stride** vs **borrowing + any-stride** — that covers both
needs without either type paying for the other's complexity.

---

## Decision 3 — a full random-access strided iterator

For strided data to work with `std::` algorithms and range-based `for`, the stride
has to disappear behind ordinary iterator arithmetic. `strided_vector_ref` supplies
a complete `std::random_access_iterator_tag` iterator whose every move scales by
the stride:

```cpp
strided_iterator& operator++()               { ptr_ += stride_; return *this; }
reference operator[](difference_type n) const { return ptr_[n * stride_]; }
friend difference_type operator-(const strided_iterator& a, const strided_iterator& b) {
    return (a.ptr_ - b.ptr_) / a.stride_;     // distance in ELEMENTS, not bytes
}
```

The full suite is there — increment/decrement, `+= / -=`, `+ / -`, all six
comparisons, and `end()` computed as `data_ + size_ * stride_`. To a caller it is
just a random-access iterator; the `* stride_` is hidden inside every operation, so
a strided column feeds `std::accumulate`, a sort, or a ranged loop identically to a
dense vector. This iterator is most of the file's substance — making irregular data
look regular is the work.

---

## Decision 4 — const-correctness through the templated pointer

The iterator is parameterized on its pointer type, and the reference type is
*derived* from it:

```cpp
template <typename Ptr> class strided_iterator {
    using reference = decltype(*std::declval<Ptr>());   // Value& or const Value&, to match Ptr
    ...
};
using iterator       = strided_iterator<pointer>;        // Value*       -> Value&
using const_iterator = strided_iterator<const_pointer>;  // const Value* -> const Value&
```

One iterator template yields both the mutable and const iterators, and `const`-ness
propagates automatically: a const view hands out `const_iterator`s whose
dereference is `const Value&`. Writing the iterator once, parameterized on the
pointer, avoids a duplicated const/non-const pair and guarantees the two never
drift — the same instinct that makes `dense2D` provide const and non-const
`operator()` from one body.

---

## Decision 5 — composable sub-views

A view of a view is still a view. `sub_vector` slices a `strided_vector_ref` into a
sub-range without copying, by advancing the pointer and preserving the stride:

```cpp
sub_vector(v, start, finish)
   -> strided_vector_ref<Value>(v.data() + start * v.stride(), finish - start, v.stride());
```

Two overloads exist, and the const one is the interesting half: given a
`const strided_vector_ref<Value>&` it returns a `strided_vector_ref<const Value>` —
a read-only view, with `const` pushed into the element type so the sub-view cannot
be used to mutate what the source promised not to. Slicing composes: a stride-`ldim`
column view can be narrowed to rows `[start, finish)` and stays a zero-copy handle
throughout.

---

## Decision 6 — no `Parameters` bundle; hardwired dense and column-vector

Unlike every owning vector, `strided_vector_ref<Value>` takes a **single** template
parameter. There is no `Parameters` bundle — and that is correct, because the bundle
configures *storage*, and a view has no storage to configure: no orientation-as-
layout, no stack/heap choice, no fixed size. Its traits are therefore fixed rather
than computed:

```cpp
template <...> struct traits::category<strided_vector_ref<Value>> { using type = tag::dense; };
template <...> struct ashape<strided_vector_ref<Value>>           { using type = cvec<Value>; };
```

- **`category = tag::dense`** — strided data is still *dense* in the sparsity sense:
  every logical position has a value; the stride is a layout detail, not a sparsity
  one. So it dispatches down the dense path.
- **`ashape = cvec<Value>`, unconditionally** — it is always a *column* vector,
  because its reason for existing is to alias a matrix column. There is no
  orientation parameter to make it an `rvec`; the shape is baked in with the
  purpose. A simplification that matches what the type is for, rather than a policy
  it doesn't need.

---

## What it deliberately is *not*

- **Not an expression-assignment target.** There is no `operator=` from an
  expression and no `parallel_ewise` sweep. You read and write elements *through*
  the view into the aliased storage (`operator()` returns a mutable reference), or
  iterate it — but you do not materialize a fused expression into it the way you do
  into a `dense_vector`. It is an access handle, not an assignable container.
- **Not on the library bounds-checking policy.** Access uses a bare `assert(i <
  size_)`, not the `bounds_checking` constant and `std::out_of_range` throw that the
  owning types use. It follows the standard-library view convention (like `span`)
  of debug-assert rather than throwing — a small, deliberate divergence for a
  lightweight handle.
- **Not a row vector, and not owning.** The two things a view of a matrix column
  does not need — an orientation choice and a storage policy — are simply absent.

---

## Where it sits among the containers

`strided_vector_ref` is the first **non-owning** type in the family. The clean way
to place it is owning-vs-viewing against its unit-stride counterpart:

| | [`dense_vector`](dense-vector-architecture.md) | `strided_vector_ref` |
|---|---|---|
| Owns storage | yes (`contiguous_memory_block`) | **no** (borrowed `Value*`) |
| Stride | always 1 | any (`stride_`) |
| Copies | deep-copy the data | alias the same data |
| `Parameters` bundle | 4 axes | **none** (single `Value` param) |
| ashape | `cvec`/`rvec` by orientation | `cvec`, fixed |
| Expression-assignable | yes (fused) | no (access handle) |
| Bounds policy | `bounds_checking` + throw | `assert` |
| Lifetime | self-contained | tied to the viewed storage |
| Primary purpose | own a dense 1-D array | view a matrix column / any strided run |

They are complementary halves of "a dense vector": one owns and is contiguous, the
other borrows and is strided. Neither carries the other's cost.

---

## File map

| File | Role |
|---|---|
| [`vec/strided_vector_ref.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/strided_vector_ref.hpp) | the view: pointer+length+stride, strided random-access iterator, `sub_vector`, fixed dense/`cvec` traits |
| [`mat/dense2D.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/mat/dense2D.hpp) | the primary source viewed (a row-major column has stride `ldim`) |
| [`traits/ashape.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/traits/ashape.hpp) | the `cvec` shape it is hardwired to |

For the owning, unit-stride counterpart and the decision that sends strided access
here, see the [dense_vector architecture](dense-vector-architecture.md) doc; for
the matrix whose columns it views, the [dense2D architecture](dense2d-architecture.md) doc.
