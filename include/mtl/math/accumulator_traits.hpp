#pragma once
// MTL5 -- accumulator policy for inner-product accumulation (issue #158)
//
// A single, cross-cutting customization point for how an inner product (a dot
// product, the columns of an LU, a GEMM element) is accumulated. It lets the
// THREE precisions of a mixed-precision tensor op be chosen independently:
//
//   * Value  -- the element/storage precision of the operands (bandwidth in)
//   * Acc    -- the accumulator precision the products are summed in (registers)
//   * Result -- the precision the rounded result is delivered/stored in (out)
//
// The trait abstracts a sum-of-products reduction behind clear/assign/
// add_product/value, so a kernel writes the same loop regardless of HOW the
// terms are combined. Three reduction configurations are expressible:
//
//   1. Plain accumulate (acc += product) -- the DEFAULT primary template.
//      The product `m*v` is formed in the accumulator precision `Acc`, rounded,
//      then added: two rounding events per term. Zero overhead and byte-
//      identical to hand-written arithmetic when `Acc == Value`; a wider `Acc`
//      than `Value` (e.g. fp32 accumulate over bf16 elements) gains accuracy
//      out of the box. Use any arithmetic type as `Acc` (e.g. `float`,`double`).
//
//   2. Fused multiply-add (acc = fma(m, v, acc)) -- `fma_accumulator<T>`.
//      The product is never rounded: `m*v + acc` is formed as if in infinite
//      precision and rounded ONCE to `T` per term. One rounding event per term
//      instead of two. The fused step is `using std::fma; fma(...)`, so `T` may
//      be a built-in float or any custom arithmetic type with an ADL-found `fma`.
//
//   3. Super-accumulator (exact sum of products, single final round-out).
//      A caller supplies a custom `Acc` (a compensated/Kahan accumulator, or a
//      Universal `quire` exact dot product) by specializing
//      `accumulator_traits<Acc, Value>`. MTL5 stays free of any external number
//      library, so the quire super-accumulator itself lives in the peer repo
//      that pairs MTL5 with Universal; this header only defines the contract it
//      plugs into. `value<Result>` then rounds the exact accumulator out once.
//
// Used by the sparse factorizations and the dense BLAS-level operations.

#include <cmath>        // std::fma
#include <type_traits>

namespace mtl::math {

/// Accumulator policy for accumulating products of `Value`s into an `Acc`.
///
/// Configuration 1 (plain `acc += product`): the primary template. The product
/// is formed in `Acc`, so `Acc` wider than `Value` accumulates more accurately.
template <typename Acc, typename Value>
struct accumulator_traits {
    /// Reset the accumulator to zero.
    static void clear(Acc& a) { a = Acc{}; }

    /// Set the accumulator to a single value.
    static void assign(Acc& a, const Value& v) { a = v; }

    /// Round the accumulator out to `Result` (default: the element type `Value`),
    /// once, at the point the accumulated entry is consumed -- giving
    /// single-rounding ("exact dot product") semantics when `Acc` is exact, and
    /// fusing the accumulate->output conversion when `Result` differs from `Acc`.
    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a); }

    /// a += m * v : the canonical accumulate-a-product primitive (a dot product /
    /// quire is a sum of products). Callers pass a negated multiplier for a
    /// subtraction (e.g. the LU elimination update). The product is formed in the
    /// accumulator precision `Acc`, so a wider `Acc` than `Value` (e.g. fp32
    /// accumulate over bf16 elements) gains accuracy out of the box; this is a
    /// no-op cast and byte-identical when `Acc == Value`.
    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += static_cast<Acc>(m) * static_cast<Acc>(v);
    }
};

/// Configuration 2 -- fused multiply-add accumulator.
///
/// A distinguished accumulator type that selects the FMA reduction: pass it as
/// the `Acc` template argument to any accumulator-aware operation to fuse each
/// `m*v` into the running sum, so the intermediate product is never rounded (one
/// rounding per term instead of two). `T` is the accumulation precision held in
/// registers; it need not equal the element precision `Value`.
///
/// `T` is intentionally unconstrained so this stays usable by custom arithmetic
/// types: the fused step is `using std::fma; fma(...)`, which selects `std::fma`
/// for the built-in floating types and, via ADL, a type-specific fused multiply-
/// add for a custom number type (e.g. a posit `fma`). A type that has no fused
/// operation simply does not satisfy the call and is not a valid `T`.
template <typename T = double>
struct fma_accumulator {
    T sum{};
};

/// Accumulator policy for the FMA reduction (configuration 2).
///
/// `add_product` computes `sum = m*v + sum` with a single rounding to `T`, so the
/// product `m*v` incurs no separate rounding event -- one rounding per term
/// instead of two. When `T` is wider than `Value` the operand widening is exact.
/// Eliminating the product rounding generally improves accuracy over the plain
/// path at the same `T`; over a long reduction the two accumulation-error streams
/// can occasionally align differently, so this is a per-term guarantee, not a
/// strict ordering on every possible sum.
template <typename T, typename Value>
struct accumulator_traits<fma_accumulator<T>, Value> {
    using Acc = fma_accumulator<T>;

    static void clear(Acc& a) { a.sum = T{}; }

    static void assign(Acc& a, const Value& v) { a.sum = static_cast<T>(v); }

    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a.sum); }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        using std::fma;   // ADL: std::fma for built-ins, a custom fma for user types
        a.sum = fma(static_cast<T>(m), static_cast<T>(v), a.sum);
    }
};

/// The arithmetic type an accumulator should be rounded out to before a scalar
/// post-processing step such as `sqrt`.
///
/// `value<Result>` rounds the accumulator out to a DELIVERY type. Passing the
/// accumulator itself is a no-op for configuration 1 and so appears to work,
/// but for `fma_accumulator<T>` it yields an `fma_accumulator<T>` and for a
/// super-accumulator a quire -- neither of which has a `sqrt` (#324).
///
/// Rounding straight to the magnitude type compiles everywhere, but throws away
/// the precision the accumulator was chosen for: the sum would be narrowed to
/// the element magnitude BEFORE the square root. So prefer the accumulator's own
/// arithmetic precision where one is nameable:
///
///   * configuration 1 (arithmetic `Acc`)   -> `Acc` itself
///   * configuration 2 (`fma_accumulator<T>`) -> `T`
///   * configuration 3 (custom/quire)       -> `Mag`, the documented delivery
///     type. A super-accumulator is exact, so this is the single final
///     round-out its contract already promises, and external specializations
///     need not provide anything new.
template <typename Acc, typename Mag>
struct accumulator_round_type {
    using type = std::conditional_t<std::is_arithmetic_v<Acc>, Acc, Mag>;
};

template <typename T, typename Mag>
struct accumulator_round_type<fma_accumulator<T>, Mag> {
    using type = T;
};

template <typename Acc, typename Mag>
using accumulator_round_type_t = typename accumulator_round_type<Acc, Mag>::type;

} // namespace mtl::math
