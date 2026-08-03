#pragma once
// PPE -- a progression of GEMM implementations, from a naive triple loop to a
// packed, register-tiled, vectorizable microkernel.
//
// Every kernel computes the same thing:  C = A * B,  A is m x k row-major,
// B is k x n row-major, C is m x n row-major, C is zeroed by the caller.
//
// Each kernel is templated on <TA, TC>:
//   TA -- the operand (storage) type
//   TC -- the accumulator/output type
// so the integer cases can accumulate in a wider type, which is what real
// integer GEMM does (quantized inference is int8 x int8 -> int32). The
// widening is part of what is being measured, not an artefact.
//
// The steps are deliberately cumulative: each adds ONE idea to the previous, so
// the measured delta is attributable. See ppe/docs for the write-up of each.

#include <cstddef>
#include <vector>
#include <algorithm>

namespace ppe {

// ---------------------------------------------------------------------------
// v0 -- naive ijk. The textbook triple loop.
//
// The inner loop walks B down a column with stride n, so every access is a
// separate cache line and the hardware prefetcher cannot help. This is the
// baseline everything else is measured against.
// ---------------------------------------------------------------------------
template <typename TA, typename TC>
void gemm_v0_naive(std::size_t m, std::size_t n, std::size_t k,
                   const TA* A, const TA* B, TC* C) {
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            TC acc = TC(0);
            for (std::size_t p = 0; p < k; ++p)
                acc += static_cast<TC>(A[i * k + p]) * static_cast<TC>(B[p * n + j]);
            C[i * n + j] = acc;
        }
}

// ---------------------------------------------------------------------------
// v1 -- loop reorder to ikj.
//
// One idea: make the innermost loop walk B and C along contiguous rows. The
// scalar a is loop-invariant in j. Same arithmetic, same order of operations
// per output element; only the traversal changes.
// ---------------------------------------------------------------------------
template <typename TA, typename TC>
void gemm_v1_ikj(std::size_t m, std::size_t n, std::size_t k,
                 const TA* A, const TA* B, TC* C) {
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p) {
            const TC a = static_cast<TC>(A[i * k + p]);
            const TA* brow = B + p * n;
            TC* crow = C + i * n;
            for (std::size_t j = 0; j < n; ++j)
                crow[j] += a * static_cast<TC>(brow[j]);
        }
}

// ---------------------------------------------------------------------------
// v2 -- cache blocking on top of ikj.
//
// One idea: tile the iteration space so the working set of each tile fits in
// cache and is reused, instead of streaming the whole of B for every row of A.
// ---------------------------------------------------------------------------
template <typename TA, typename TC>
void gemm_v2_blocked(std::size_t m, std::size_t n, std::size_t k,
                     const TA* A, const TA* B, TC* C,
                     std::size_t MC = 64, std::size_t NC = 256, std::size_t KC = 128) {
    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                for (std::size_t i = 0; i < mc; ++i)
                    for (std::size_t p = 0; p < kc; ++p) {
                        const TC a = static_cast<TC>(A[(ic + i) * k + (pc + p)]);
                        const TA* brow = B + (pc + p) * n + jc;
                        TC* crow = C + (ic + i) * n + jc;
                        for (std::size_t j = 0; j < nc; ++j)
                            crow[j] += a * static_cast<TC>(brow[j]);
                    }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v3 -- pack A and B into contiguous panels.
//
// One idea: copy each tile into a dense, sequentially-traversed buffer before
// computing on it. Blocking fixed WHICH data is touched; packing fixes HOW it
// is laid out -- the inner loop then walks memory linearly, which is about TLB
// pressure and prefetch friendliness as much as cache lines.
// ---------------------------------------------------------------------------
template <typename TA, typename TC>
void gemm_v3_packed(std::size_t m, std::size_t n, std::size_t k,
                    const TA* A, const TA* B, TC* C,
                    std::size_t MC = 64, std::size_t NC = 256, std::size_t KC = 128) {
    std::vector<TC> Ap(MC * KC), Bp(KC * NC);
    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            for (std::size_t p = 0; p < kc; ++p)                    // pack B tile
                for (std::size_t j = 0; j < nc; ++j)
                    Bp[p * nc + j] = static_cast<TC>(B[(pc + p) * n + jc + j]);
            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                for (std::size_t i = 0; i < mc; ++i)                // pack A tile
                    for (std::size_t p = 0; p < kc; ++p)
                        Ap[i * kc + p] = static_cast<TC>(A[(ic + i) * k + pc + p]);
                for (std::size_t i = 0; i < mc; ++i) {
                    TC* crow = C + (ic + i) * n + jc;
                    for (std::size_t p = 0; p < kc; ++p) {
                        const TC a = Ap[i * kc + p];
                        const TC* brow = Bp.data() + p * nc;
                        for (std::size_t j = 0; j < nc; ++j)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v4 -- register tiling: an MR x NR block of C held in locals.
//
// One idea: stop writing C to memory on every k step. An MR x NR microtile is
// accumulated in registers across the whole kc depth and written once. This is
// what turns a memory-bound inner loop into a compute-bound one, and it is why
// the number of independent accumulators matters -- they are what hides FMA
// latency.
// ---------------------------------------------------------------------------
template <typename TA, typename TC, std::size_t MR = 4, std::size_t NR = 8>
void gemm_v4_regtile(std::size_t m, std::size_t n, std::size_t k,
                     const TA* A, const TA* B, TC* C,
                     std::size_t MC = 64, std::size_t NC = 256, std::size_t KC = 128) {
    std::vector<TC> Ap(MC * KC), Bp(KC * NC);
    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            for (std::size_t p = 0; p < kc; ++p)
                for (std::size_t j = 0; j < nc; ++j)
                    Bp[p * nc + j] = static_cast<TC>(B[(pc + p) * n + jc + j]);
            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                for (std::size_t i = 0; i < mc; ++i)
                    for (std::size_t p = 0; p < kc; ++p)
                        Ap[i * kc + p] = static_cast<TC>(A[(ic + i) * k + pc + p]);

                for (std::size_t i = 0; i < mc; i += MR) {
                    const std::size_t mr = std::min(MR, mc - i);
                    for (std::size_t j = 0; j < nc; j += NR) {
                        const std::size_t nr = std::min(NR, nc - j);
                        TC acc[MR][NR] = {};
                        for (std::size_t p = 0; p < kc; ++p)
                            for (std::size_t ii = 0; ii < mr; ++ii) {
                                const TC a = Ap[(i + ii) * kc + p];
                                const TC* brow = Bp.data() + p * nc + j;
                                for (std::size_t jj = 0; jj < nr; ++jj)
                                    acc[ii][jj] += a * brow[jj];
                            }
                        for (std::size_t ii = 0; ii < mr; ++ii)
                            for (std::size_t jj = 0; jj < nr; ++jj)
                                C[(ic + i + ii) * n + jc + j + jj] += acc[ii][jj];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v5 -- full-tile microkernel with compile-time extents.
//
// One idea: make the hot microtile a separate function with COMPILE-TIME MR/NR
// and no edge tests, so the compiler can fully unroll it, keep the accumulators
// in registers, and vectorize the NR dimension. Edges fall back to v4's guarded
// path. Nothing about the arithmetic changes -- only what the compiler can see.
// ---------------------------------------------------------------------------
namespace detail {
template <typename TC, std::size_t MR, std::size_t NR>
inline void micro(std::size_t kc, const TC* __restrict Ap, std::size_t lda,
                  const TC* __restrict Bp, std::size_t ldb,
                  TC* __restrict C, std::size_t ldc) {
    TC acc[MR][NR] = {};
    for (std::size_t p = 0; p < kc; ++p) {
        const TC* brow = Bp + p * ldb;
        for (std::size_t ii = 0; ii < MR; ++ii) {
            const TC a = Ap[ii * lda + p];
            for (std::size_t jj = 0; jj < NR; ++jj)
                acc[ii][jj] += a * brow[jj];
        }
    }
    for (std::size_t ii = 0; ii < MR; ++ii)
        for (std::size_t jj = 0; jj < NR; ++jj)
            C[ii * ldc + jj] += acc[ii][jj];
}
}  // namespace detail

template <typename TA, typename TC, std::size_t MR = 4, std::size_t NR = 8>
void gemm_v5_micro(std::size_t m, std::size_t n, std::size_t k,
                   const TA* A, const TA* B, TC* C,
                   std::size_t MC = 64, std::size_t NC = 256, std::size_t KC = 128) {
    std::vector<TC> Ap(MC * KC), Bp(KC * NC);
    for (std::size_t jc = 0; jc < n; jc += NC) {
        const std::size_t nc = std::min(NC, n - jc);
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc);
            for (std::size_t p = 0; p < kc; ++p)
                for (std::size_t j = 0; j < nc; ++j)
                    Bp[p * nc + j] = static_cast<TC>(B[(pc + p) * n + jc + j]);
            for (std::size_t ic = 0; ic < m; ic += MC) {
                const std::size_t mc = std::min(MC, m - ic);
                for (std::size_t i = 0; i < mc; ++i)
                    for (std::size_t p = 0; p < kc; ++p)
                        Ap[i * kc + p] = static_cast<TC>(A[(ic + i) * k + pc + p]);

                for (std::size_t i = 0; i < mc; i += MR) {
                    const std::size_t mr = std::min(MR, mc - i);
                    for (std::size_t j = 0; j < nc; j += NR) {
                        const std::size_t nr = std::min(NR, nc - j);
                        if (mr == MR && nr == NR) {
                            detail::micro<TC, MR, NR>(kc, Ap.data() + i * kc, kc,
                                                      Bp.data() + j, nc,
                                                      C + (ic + i) * n + jc + j, n);
                        } else {                                    // guarded edge
                            TC acc[MR][NR] = {};
                            for (std::size_t p = 0; p < kc; ++p)
                                for (std::size_t ii = 0; ii < mr; ++ii) {
                                    const TC a = Ap[(i + ii) * kc + p];
                                    const TC* brow = Bp.data() + p * nc + j;
                                    for (std::size_t jj = 0; jj < nr; ++jj)
                                        acc[ii][jj] += a * brow[jj];
                                }
                            for (std::size_t ii = 0; ii < mr; ++ii)
                                for (std::size_t jj = 0; jj < nr; ++jj)
                                    C[(ic + i + ii) * n + jc + j + jj] += acc[ii][jj];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ppe
