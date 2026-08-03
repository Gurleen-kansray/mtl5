#pragma once
// PPE -- a stated per-type peak model, so "efficiency" means something.
//
// Peak here is the SINGLE-CORE rate at which the machine could retire the
// 2*m*n*k operations a GEMM performs, given what the ISA offers for that type.
// It is a model. It is written down in full so a reader can disagree with a
// specific number rather than with the word "peak", and every measurement is
// checked against it -- a result above 100% means the MODEL is wrong, not that
// the kernel is superhuman, and the driver says so loudly.
//
// Two earlier attempts and why they were discarded, because the failures are
// instructive:
//
//   1. A first model put int64 at 2 ops/cycle (scalar imul). Measurements hit
//      119% of it. AVX2 has no 64-bit SIMD multiply, but the compiler emulates
//      one from 32-bit pieces, so the real ceiling is higher than "scalar".
//      Corrected to 8 ops/cycle (4 lanes x mul+add).
//
//   2. A probe that MEASURED achievable rate looked more honest but was worse:
//      with literal operands the compiler folded the loop away and it reported
//      3.8 million GOP/s; once the operands were made opaque it reported 17.7
//      GOP/s for fp64, BELOW the 37 GOP/s the GEMM kernels actually achieve,
//      because the accumulator array did not stay in vector registers. A probe
//      that is beaten by the thing it is meant to bound is not a ceiling either.
//      Writing a portable probe that reliably reaches vector peak is its own
//      project; the ISA model is the honest tool at this scope.
//
// Target: Alder Lake P-core, AVX2 + FMA + AVX-VNNI, no AVX-512.
//
//   fp64   2 FMA units x 4 lanes x 2 ops per FMA            = 16 ops/cycle
//   fp32   2 FMA units x 8 lanes x 2                        = 32
//   fp16   no native arithmetic without AVX512-FP16: operands convert to fp32
//          and back, so fp32 is the CEILING and the conversions are extra work
//          on top -- expect a large shortfall, that is the point       = 32
//   int8   no integer FMA. Accumulating in int32 means widening; the realistic
//          vector path is the int16 multiply, so modelled as int16     = 32
//   int16  vpmullw, 16 lanes, paired with an add                       = 32
//   int32  vpmulld, 8 lanes (2 uops, roughly 1/cycle), plus an add     = 16
//   int64  no 64-bit SIMD multiply before AVX-512DQ; emulated from 32-bit
//          pieces across 4 lanes                                       = 8
//
// The sustained single-core clock is a property of the machine, supplied by the
// caller rather than guessed here.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ppe {

/// Operations per cycle the ISA can retire for this operand type, under the
/// model documented above.
template <typename TA>
constexpr double ops_per_cycle() {
    if constexpr (std::is_same_v<TA, double>)             return 16.0;
    else if constexpr (std::is_same_v<TA, float>)         return 32.0;
    else if constexpr (std::is_same_v<TA, std::int64_t>)  return 8.0;
    else if constexpr (std::is_same_v<TA, std::int32_t>)  return 16.0;
    else if constexpr (std::is_same_v<TA, std::int16_t>)  return 32.0;
    else if constexpr (std::is_same_v<TA, std::int8_t>)   return 32.0;
    else                                                  return 32.0;  // _Float16 -> fp32 path
}

/// Modelled single-core peak in GOP/s at a given sustained clock (GHz).
template <typename TA>
constexpr double peak_gops(double ghz) {
    return ops_per_cycle<TA>() * ghz;
}

}  // namespace ppe
