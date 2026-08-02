#pragma once
// MTL5 -- Householder reflections for QR factorization
// Computes v, beta such that (I - beta*v*v^T)*x = ||x||*e_1
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl {

/// Compute Householder vector v and scalar beta for a column vector x.
/// The reflection (I - beta*v*v^T) zeroes out x(1:end), leaving x(0) = -sign(x0)*||x||.
/// v(0) is always 1 (implicit). Returns {v, beta}.
template <typename T>
std::pair<vec::dense_vector<T>, T> householder(const vec::dense_vector<T>& x) {
    using std::sqrt;
    using std::abs;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = x.size();

    vec::dense_vector<T> v(n);
    if (n == 0) return {v, math::zero<T>()};   // no element at index 0 to write

    // Scale by the largest magnitude before squaring. Forming sum(x(i)^2)
    // directly underflows to zero once the entries are small -- and the old
    // `sigma == 0` test only caught EXACT zero, so a merely tiny sigma fell
    // through to `beta = 2*v0*v0 / (sigma + v0*v0)` with both terms flushed to
    // zero, i.e. 0/0 = NaN, plus `v(i) /= v0` overflowing to infinity.
    //
    // The SVD's alternating-QR iteration drives exactly that: as it converges
    // the trailing columns shrink toward zero, so the reflectors underflow and
    // poison an already-correct answer with NaN (#337). Scaling keeps the
    // squares O(1), so the only way sigma vanishes now is that x(1:) really is
    // negligible against x(0) -- which is the genuine "already along e_1" case.
    T scale = math::zero<T>();
    for (size_type i = 0; i < n; ++i) {
        const T axi = abs(x(i));
        if (axi > scale) scale = axi;
    }

    for (size_type i = 0; i < n; ++i)
        v(i) = x(i);
    v(0) = math::one<T>();

    if (scale == math::zero<T>())
        return {v, math::zero<T>()};   // x is the zero vector

    // sigma = sum((x(i)/scale)^2) for i >= 1, computed in the scaled variables
    T sigma = math::zero<T>();
    for (size_type i = 1; i < n; ++i) {
        const T xs = x(i) / scale;
        sigma += xs * xs;
    }

    if (sigma == math::zero<T>())
        return {v, math::zero<T>()};   // x is already along e_1

    const T x0 = x(0) / scale;
    const T norm_x = sqrt(x0 * x0 + sigma);
    T v0;
    if (x0 <= math::zero<T>())
        v0 = x0 - norm_x;
    else
        v0 = -sigma / (x0 + norm_x);

    const T beta = T(2) * v0 * v0 / (sigma + v0 * v0);

    // beta == 0 means the reflection is the identity, so v carries no
    // information -- and computing it would divide by a v0 small enough to
    // overflow. Return a clean unit v: qr_factor STORES v below the diagonal,
    // so letting huge or infinite entries through would poison the factor and
    // every later use of it.
    if (beta == math::zero<T>())
        return {v, math::zero<T>()};

    // Normalize so that v(0) = 1. Both numerator and denominator are in the
    // scaled variables, so the ratio is the same as before the scaling.
    for (size_type i = 1; i < n; ++i)
        v(i) = (x(i) / scale) / v0;
    v(0) = math::one<T>();

    return {v, beta};
}

/// Apply Householder reflection (I - beta*v*v^T) to columns col..ncols-1
/// of matrix A, rows row..nrows-1. Modifies A in-place.
template <Matrix M, typename T>
void apply_householder_left(M& A, const vec::dense_vector<T>& v, T beta,
                            typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type n = A.num_cols();
    const size_type vlen = v.size();

    // Each column j is updated independently (reads the shared reflector v and
    // only column j of A, writes only column j), so partitioning the columns
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target column: no work (matches the serial loop, and
    // avoids an unsigned underflow in n - col before deriving the work range).
    if (col >= n) return;
    // beta == 0 is the identity reflection. Returning early is not just an
    // optimisation: `beta * v(i) * w` would evaluate 0 * inf = NaN if any
    // reflector entry had overflowed.
    if (beta == math::zero<T>()) return;
    const size_type ncols = n - col;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(ncols), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type j = col + static_cast<size_type>(t);
                // w = v^T * A(:,j)
                T w = math::zero<T>();
                for (size_type i = 0; i < vlen; ++i)
                    w += v(i) * A(row + i, j);
                // A(:,j) -= beta * v * w
                for (size_type i = 0; i < vlen; ++i)
                    A(row + i, j) -= beta * v(i) * w;
            }
        });
}

/// Apply Householder reflection on the right: A * (I - beta*v*v^T)
/// Modifies columns col..col+vlen-1 of rows row..nrows-1.
template <Matrix M, typename T>
void apply_householder_right(M& A, const vec::dense_vector<T>& v, T beta,
                             typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type vlen = v.size();

    // Each row i is updated independently (reads the shared reflector v and only
    // its own vlen entries of A, writes only those), so partitioning the rows
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target row: no work (matches the serial loop, and
    // avoids an unsigned underflow in m - row before deriving the work range).
    if (row >= m) return;
    // Identity reflection -- see the note in apply_householder_left.
    if (beta == math::zero<T>()) return;
    const size_type nrows = m - row;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(nrows), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type i = row + static_cast<size_type>(t);
                // w = A(i,:) * v
                T w = math::zero<T>();
                for (size_type j = 0; j < vlen; ++j)
                    w += A(i, col + j) * v(j);
                // A(i,:) -= beta * w * v^T
                for (size_type j = 0; j < vlen; ++j)
                    A(i, col + j) -= beta * w * v(j);
            }
        });
}

} // namespace mtl
