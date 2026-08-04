# The expression-template layer: lazy arithmetic without temporaries

How MTL5 evaluates `x = a + b*2 - c` **without** building three intermediate
vectors, and *why* the machinery is shaped the way it is. This is the layer the
container docs keep pointing at: the reason
[`dense_vector`](dense-vector-architecture.md) and
[`dense2D`](dense2d-architecture.md) have a templated `operator=` that takes "an
expression." This page is what that expression *is*.

The organizing goal is **fusion**: an arithmetic combination of vectors (or
matrices) should compile to a single pass that computes each output element's full
formula once, allocating nothing in between — and it should do so while composing
freely, staying mixed-precision-correct, and never dangling a reference to a
temporary. Each of those requirements drives one decision below.

The layer lives in [`vec/expr/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/vec/expr),
[`mat/expr/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/mat/expr),
[`vec/operators.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/operators.hpp),
and the two small helpers
[`traits/is_expression.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/traits/is_expression.hpp)
and
[`detail/expr_storage.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/expr_storage.hpp).

---

## The problem it solves

The naive `operator+` returning a vector evaluates eagerly:

```text
x = a + b*2 - c;
   t1 = b*2       // allocate n, loop n
   t2 = a + t1    // allocate n, loop n
   t3 = t2 - c    // allocate n, loop n
   x  = t3        // copy n
```

Four allocations, four passes over memory, for an operation that is fundamentally
one pass. On the memory-bound L1/L2 kernels (`axpy`-shaped work) that overhead
*is* the cost. Expression templates replace it with:

```text
for each i:  x(i) = a(i) + b(i)*2 - c(i)     // one pass, zero temporaries
```

The trick is to make `a + b*2 - c` build a *description* of the computation — a
typed tree — that is only evaluated, element by element, when it is assigned into
a real container.

---

## Decision 1 — expressions are a *concept*, not a CRTP base class

This is the headline, and the clearest break from MTL4. In MTL4 every expression
derived from a CRTP base (`vec_expr<Derived>` / `mat_expr<Derived>`); the base
hierarchy is how operands and results were recognized as "expressions." MTL5
deletes the hierarchy. The `vec_expr.hpp` umbrella says it outright:

> In MTL4 this was a CRTP base class for all vector expressions. In MTL5, C++20
> concepts (`Vector`, `DenseVector`) replace the CRTP hierarchy. Expression types
> simply satisfy the `Vector` concept by providing `value_type`, `size_type`,
> `size()`, `operator()(i)`.

An "expression" is therefore anything that **models the `Vector` concept**:

```cpp
concept Vector = Collection<T> && requires(const T& v, std::size_t i) {
    { v(i) } -> std::convertible_to<typename T::value_type>;
};
```

A concrete `dense_vector` and a lazy `vec_vec_op_expr` are interchangeable operands
because both satisfy that contract — not because they share a base. Structural
(duck) typing via concepts replaces nominal typing via inheritance. The payoff is
that expression nodes are plain classes with no base, no virtual anything, and no
CRTP `Derived` parameter threading through them; the concept is the contract, and
the compiler checks it structurally at each use.

---

## Decision 2 — an expression is a lazy tree of small nodes

Each operator produces a node that stores its operands and computes element `i` on
demand. The binary element-wise node is the archetype:

```cpp
template <typename E1, typename E2, template <typename,typename> class SF>
class vec_vec_op_expr {
    E1 e1_;  E2 e2_;                          // the two operands (see Decision 6 for E1/E2)
public:
    using value_type = typename SF<e1_vt, e2_vt>::result_type;
    size_type size() const { return e1_.size(); }
    value_type operator()(size_type i) const {   // evaluate THIS element, recursively
        return SF<e1_vt, e2_vt>::apply(e1_(i), e2_(i));
    }
};
```

Leaves of the tree are concrete containers; internal nodes are op-exprs. `operator()(i)`
recurses down the tree evaluating only position `i`; nothing is stored. `size()`
(and `num_rows`/`num_cols`) just delegate to an operand. Because a node itself
models `Vector`, nodes nest: the `E1` of one node can be another node. `a + b*2`
is a `vec_vec_op_expr<dense_vector&, vec_scal_op_expr<...>, plus>` — a two-level
tree, entirely in the type.

---

## Decision 3 — the operation is a functor *policy*, which also sets the result type

The node is parameterized on the operation as a **template-template parameter**
`SF`, not hard-coded. One node template serves every element-wise binary op; the
functor supplies both the computation and the result type:

```cpp
template <typename T1, typename T2 = T1>
struct plus {
    using result_type = std::common_type_t<T1, T2>;              // the result element type
    static constexpr result_type apply(const T1& a, const T2& b) { return a + b; }
};
```

`vec_vec_op_expr`'s `value_type` is `SF<T1,T2>::result_type`, and `operator()`
calls `SF::apply`. Adding `+`/`-`/`*`/`/` support is a matter of a functor, not a
new expression class. Separating *what op* (the functor) from *what shape of
expression* (the node) keeps the node count small and the operation set open.

---

## Decision 4 — mixed precision falls out of the result type

Because the functor computes `result_type` from the two operand element types,
combining precisions is correct **by construction**. `plus<float,double>::result_type`
is `std::common_type_t<float,double>` = `double`, so

```cpp
dense_vector<float>  a;
dense_vector<double> b;
auto e = a + b;      // e.value_type == double
```

is a `double`-valued expression with no narrowing at the node. Materializing it
into a `dense_vector<T>` then does one explicit `static_cast<T>` at the boundary
(see Decision 7). This is the library's reason for existing — mixed-precision
algorithm design with custom number types — expressed at the level where two
operand types meet. A custom number type participates the moment its `common_type`
(or a functor specialization) is defined.

---

## Decision 5 — operators build the tree and return it

The overloads in `operators.hpp` are free functions, concept-constrained, that
construct and return the node — computing nothing:

```cpp
template <typename V1, typename V2>
    requires (Vector<std::remove_cvref_t<V1>> && Vector<std::remove_cvref_t<V2>>)
auto operator+(V1&& a, V2&& b) {
    return expr::vec_vec_op_expr<expr_store_t<V1>, expr_store_t<V2>, functor::scalar::plus>(
        std::forward<V1>(a), std::forward<V2>(b));
}
```

Two things make chaining work. The `requires Vector<...>` clause means these
operators are found only for vector-like operands (no ambiguity with unrelated
`+`s), and — crucially — the *returned node also models `Vector`*, so it is itself
a legal operand to the next operator. `auto` return preserves the exact node type.
`a + b*2 - c` thus type-checks left to right, each sub-result feeding the next, and
the whole thing collapses to one nested type with no evaluation.

---

## Decision 6 — the lifetime rule: lvalues by const-ref, rvalues by value

This is the layer's most important *correctness* mechanism, and the classic
expression-template hazard. In `a + b*2`, the sub-expression `b*2` is a **temporary**
that dies at the end of the full statement. If the enclosing `+` node stored it by
reference, the reference would dangle before the expression is ever assigned. But
named operands like `a` *do* outlive the statement, and copying them would defeat
the whole point.

MTL5 resolves this with perfect forwarding plus a storage trait. The operators take
forwarding references (`V&&`), and `detail::expr_store_t` decides how the node holds
each operand:

```cpp
template <typename T>
using expr_store_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,          // was the operand an lvalue?
    const std::remove_reference_t<T>&,      //   yes -> store a const reference (no copy)
    std::remove_cvref_t<T>>;                //   no (rvalue) -> store BY VALUE (move it in)
```

So the node's `E1`/`E2` template parameters *are the storage types*: a named vector
is held as `const dense_vector&` (aliased, zero copy), while a temporary
sub-expression is **moved into** the parent node and held by value, keeping it alive
exactly as long as the tree. A node can mix both — reference one operand, own the
other. This is what makes `x = a + b*2 - c` safe: every temporary in the tree is
owned by its parent, every named leaf is borrowed, and nothing dangles. (Scalars
are always stored by value — they are cheap and often temporaries themselves.)

---

## Decision 7 — materialization is the fusion point

The tree is inert until it meets a concrete container. The container's templated
assignment — documented on the
[dense_vector](dense-vector-architecture.md#the-shared-foundation-see-the-dense2d-doc)
and [dense2D](dense2d-architecture.md#expression-templates-assignment-and-mixed-precision)
side — is where evaluation and fusion happen:

```cpp
template <typename Expr>
    requires (Vector<Expr> && traits::is_expression_v<Expr>
              && std::convertible_to<typename Expr::value_type, Value>)
dense_vector& operator=(const Expr& expr) {
    change_dim(expr.size());
    detail::parallel_ewise(size(), 1, [&](std::size_t i) {
        (*this)(i) = static_cast<Value>(expr(i));   // evaluate the WHOLE tree at i, once
    });
    return *this;
}
```

One loop over `i`; each iteration walks the entire expression tree for that element
and writes the result straight into storage. No temporary vectors, one pass, one
`static_cast` at the precision boundary. This is where the laziness pays off, and
why the nodes never needed to store results: the *consumer* drives evaluation, so
the producer can stay a description. The sweep is element-independent, so it is
bit-identical to the serial loop at `MTL5_NUM_THREADS=1` and trivially parallel
otherwise.

---

## Decision 8 — every node carries its trait identity

Each expression node specializes three traits, exactly as a concrete container
does:

```cpp
template <...> struct is_expression<vec_vec_op_expr<...>> : std::true_type {};
template <...> struct category<vec_vec_op_expr<...>>      { using type = tag::dense; };
template <...> struct ashape<vec_vec_op_expr<...>>        { using type = cvec<value_type>; };
```

- **`is_expression = true`** is the bit that distinguishes a lazy tree from a
  concrete container. Both satisfy `Vector`, so the concept alone cannot tell them
  apart; the assignment operator's `requires` clause pairs `Vector<Expr>` with
  `is_expression_v<Expr>` so that assigning *from an expression* takes the fusing
  path, while assigning from another concrete vector takes the ordinary copy. It is
  the concrete-vs-lazy discriminator.
- **`category` / `ashape`** give the node the same dense/`cvec` identity a real
  vector has, so an expression is a first-class typed operand in dispatch and the
  shape algebra — a `mat * (a + b)` composes correctly because `(a + b)` classifies
  as a column vector.

---

## The two families — and what is deliberately *not* element-wise

The vector layer above has a mirror image for matrices in
[`mat/expr/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/mat/expr):
`mat_mat_op_expr` (element-wise), `mat_scal_op_expr`, `mat_negate_expr`, with dense
and sparse expression umbrellas (`dmat_expr` / `smat_expr`). The same node/functor/
forwarding design carries over unchanged.

Two neighbors are intentionally *not* built on the element-wise fusion model,
because their arithmetic is different:

- **Matrix multiply** (`mat_mat_times_expr`). `C = A*B` is not element-wise —
  each `C(i,j)` is a dot product, O(n³) total — so lazy per-element evaluation
  would be catastrophic (recomputing rows/columns). It is a distinct expression
  that exists to be recognized and dispatched to a blocked GEMM kernel, not
  fused into a scalar loop. Element-wise fusion is a memory-bound-kernel win; the
  compute-bound kernel wants blocking instead (see the
  [BLAS kernel architecture](blas-kernel-architecture.md)).
- **Structural views** ([`mat/view/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/mat/view):
  `transposed_view`, `upper_view`/`lower_view`, `banded_view`, `hermitian_view`,
  `map_view`). These are lazy too, and also model `Matrix`, but they remap
  *indices* or apply an element function rather than combine operands — a transpose
  swaps `(i,j)`, a triangular view masks below the diagonal. They are adaptors, a
  cousin of expressions rather than arithmetic nodes, and share the "compute on
  access, satisfy the concept" spirit.

---

## What the design deliberately does not do

- **No common-subexpression elimination.** If a subtree appears twice, it is
  evaluated twice per element — expression templates do not cache. This is left to
  the compiler, which inlines the whole tree and usually removes the redundancy;
  the layer does not attempt its own CSE.
- **Fusion targets element-wise work only.** The win is on L1/L2-shaped,
  memory-bound arithmetic. It does nothing for matmul (handled separately) and is
  not a general loop optimizer.
- **Element-wise, so alias-safe by independence.** Because materialization writes
  output element `i` reading only input position `i`, `x = x + y` is correct even
  though the expression aliases the target — each output depends solely on the same
  input index. (An operation that read *other* positions would not have this
  property; the element-wise nodes deliberately do not.)

---

## Relationship to MTL4 and the C++20 rewrite

The pattern is MTL4's; the mechanism is modern C++:

| Concern | MTL4 | MTL5 |
|---|---|---|
| "Is this an expression?" | derive from a CRTP base | model the `Vector`/`Matrix` concept + `is_expression` trait |
| Recognize operands | base-class type | concept constraint on the operator |
| Operand lifetime | ownership/ref bookkeeping in the base | perfect forwarding + `expr_store_t` |
| Operation selection | tagged functors | template-template functor policy |
| Result element type | trait computation | functor `result_type` (`common_type`) |

The result is a layer with no inheritance: expression nodes are small, independent
classes unified only by the concept they model, made safe by a two-line storage
trait, and fused at the point of assignment.

---

## File map

| File | Role |
|---|---|
| [`vec/operators.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/operators.hpp) | the `+ - * /` overloads that build vector expression trees |
| [`vec/expr/vec_vec_op_expr.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/expr/vec_vec_op_expr.hpp) | binary element-wise node (the archetype) |
| [`vec/expr/vec_scal_op_expr.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/expr/vec_scal_op_expr.hpp) | scalar-vector and vector-scalar nodes |
| [`vec/expr/vec_negate_expr.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/vec/expr/vec_negate_expr.hpp) | unary negation node |
| [`mat/expr/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/mat/expr) | the matrix mirror (`mat_mat_op_expr`, `mat_scal_op_expr`, `mat_mat_times_expr`, …) |
| [`mat/view/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/mat/view) | lazy structural views (transpose, triangular, banded, hermitian, map) |
| [`detail/expr_storage.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/expr_storage.hpp) | `expr_store_t`: lvalue→`const&`, rvalue→by value |
| [`traits/is_expression.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/traits/is_expression.hpp) | the lazy-vs-concrete discriminator trait |
| [`functor/scalar/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/functor/scalar) | `plus`/`minus`/`times`/`divide` functors (`apply` + `result_type`) |
| [`detail/ewise.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/detail/ewise.hpp) | `parallel_ewise`, the materialization sweep |

For where expressions are *consumed*, see the
[dense_vector](dense-vector-architecture.md) and
[dense2D](dense2d-architecture.md) architecture docs; for the GEMM path that
matrix-multiply expressions dispatch to, the
[BLAS kernel architecture](blas-kernel-architecture.md).
