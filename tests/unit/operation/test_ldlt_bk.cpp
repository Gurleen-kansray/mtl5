#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>

using namespace mtl;

// Helper: compute ||Ax - b|| / (||A||*||x||)
static double backward_error(const mat::dense2D<double>& A,
                             const vec::dense_vector<double>& x,
                             const vec::dense_vector<double>& b) {
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double d = Ax(i) - b(i);
        res += d * d;
    }
    res = std::sqrt(res);
    double Anorm = frobenius_norm(A);
    double xnorm = two_norm(x);
    return (Anorm * xnorm > 0.0) ? res / (Anorm * xnorm) : res;
}

TEST_CASE("Bunch-Kaufman on SPD matrix", "[operation][ldlt_bk]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);
    REQUIRE(pivots.ipiv.size() == 3);

    // Solve
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on symmetric indefinite matrix", "[operation][ldlt_bk]") {
    // Indefinite: eigenvalues +/- sqrt(5)
    // A = {{1, 2}, {2, -1}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 2;  A(1,1) = -1;

    mat::dense2D<double> Aorig(2, 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {5.0, 3.0};
    vec::dense_vector<double> x(2);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on matrix requiring 2x2 pivot", "[operation][ldlt_bk]") {
    // Matrix where diagonal is zero but off-diagonals are large
    // Forces 2x2 pivot selection
    // A = {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}} - singular, but
    // Use: A = {{0, 3, 1}, {3, 0, 2}, {1, 2, 5}}
    // A(0,0) = 0 forces a pivot swap or 2x2
    mat::dense2D<double> A(3, 3);
    A(0,0) = 0; A(0,1) = 3; A(0,2) = 1;
    A(1,0) = 3; A(1,1) = 0; A(1,2) = 2;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 5;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    // Check that at least one 2x2 pivot was used (negative ipiv)
    bool has_2x2 = false;
    for (int p : pivots.ipiv)
        if (p < 0) has_2x2 = true;
    REQUIRE(has_2x2);

    vec::dense_vector<double> b = {4.0, 5.0, 12.0};
    vec::dense_vector<double> x(3);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 1x1 matrix", "[operation][ldlt_bk]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 7.0;

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {21.0};
    vec::dense_vector<double> x(1);
    ldlt_bk_solve(A, pivots, x, b);
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Bunch-Kaufman on 4x4 indefinite matrix", "[operation][ldlt_bk]") {
    // Symmetric indefinite 4x4 - the type that arises in UKF covariance updates
    mat::dense2D<double> A(4, 4);
    A(0,0) =  2; A(0,1) =  1; A(0,2) = -1; A(0,3) =  0;
    A(1,0) =  1; A(1,1) = -3; A(1,2) =  2; A(1,3) =  1;
    A(2,0) = -1; A(2,1) =  2; A(2,2) =  4; A(2,3) = -2;
    A(3,0) =  0; A(3,1) =  1; A(3,2) = -2; A(3,3) =  1;

    mat::dense2D<double> Aorig(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {1.0, -2.0, 3.0, -1.0};
    vec::dense_vector<double> x(4);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 5x5 SPD matrix", "[operation][ldlt_bk]") {
    // Deterministic well-conditioned SPD: Lehmer matrix (small)
    constexpr std::size_t n = 5;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = double(std::min(i, j) + 1) / double(std::max(i, j) + 1);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 8x8 diagonally dominant SPD", "[operation][ldlt_bk]") {
    // Deterministic: tridiagonal SPD with strong diagonal dominance
    constexpr std::size_t n = 8;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) { A(i, i-1) = -1.0; A(i-1, i) = -1.0; }
    }

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on large indefinite system", "[operation][ldlt_bk]") {
    // Construct a symmetric indefinite matrix: diag with alternating signs
    // plus off-diagonal coupling
    constexpr std::size_t n = 20;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = (i % 2 == 0) ? 10.0 : -5.0;
        if (i + 1 < n) {
            A(i, i + 1) = 2.0;
            A(i + 1, i) = 2.0;
        }
    }

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-10);
}

TEST_CASE("Bunch-Kaufman detects singular matrix", "[operation][ldlt_bk]") {
    // Singular: A = {{0, 0}, {0, 0}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 0;

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info != 0);
}

TEST_CASE("Bunch-Kaufman on empty matrix", "[operation][ldlt_bk]") {
    mat::dense2D<double> A(0, 0);
    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);
}

// ---------------------------------------------------------------------------
// Regression: #335 -- wrong solutions whenever a pivot interchange occurs.
//
// The factorization permuted the L columns already written by earlier steps,
// putting the stored factor in the "single global P" convention while
// ldlt_bk_solve replays the interchanges one step at a time. The two agree only
// when no interchange happens, so the pre-existing n = 2 and n = 3 cases above
// all passed while anything larger silently returned a wrong answer with
// info == 0 -- backward errors around 1e-1 rather than 1e-16.
//
// The reported 4x4 is exercised verbatim, plus a randomized sweep over sizes
// large enough to force interchanges and 2x2 blocks.
// ---------------------------------------------------------------------------

TEST_CASE("Bunch-Kaufman with pivot interchange (#335)", "[operation][ldlt_bk][regression]") {
    const std::size_t n = 4;
    const double a[4][4] = {
        { 1.22060812, -1.33949552,  0.42837340, -0.12346316},
        {-1.33949552,  1.41437718, -0.12405069,  2.00815707},
        { 0.42837340, -0.12405069,  0.22988654,  0.60489374},
        {-0.12346316,  2.00815707,  0.60489374,  1.62715984}};

    mat::dense2D<double> A(n, n), Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { A(i,j) = a[i][j]; Aorig(i,j) = a[i][j]; }

    vec::dense_vector<double> b(n), x(n);
    b[0] = 1.59456055; b[1] = 0.23043417; b[2] = -0.06491033; b[3] = -0.96898025;

    bk_pivot_info pivots;
    REQUIRE(ldlt_bk_factor(A, pivots) == 0);
    ldlt_bk_solve(A, pivots, x, b);

    // This matrix pivots: ipiv is {1,4,4,4}, i.e. two 1x1 interchanges and no
    // 2x2 block. The test is only meaningful if an interchange actually occurs.
    bool interchanged = false;
    for (std::size_t k = 0; k < n; ++k)
        if (pivots.ipiv[k] > 0 && static_cast<std::size_t>(pivots.ipiv[k] - 1) != k)
            interchanged = true;
    REQUIRE(interchanged);

    REQUIRE(backward_error(Aorig, x, b) < 1e-12);
}

TEST_CASE("Bunch-Kaufman random symmetric indefinite sweep (#335)",
          "[operation][ldlt_bk][regression]") {
    // Deterministic LCG so the sweep is reproducible without <random> policy.
    std::uint64_t seed = 20260802u;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>((seed >> 11) % 2000001) / 1000000.0 - 1.0;  // [-1,1]
    };

    std::size_t saw_2x2 = 0, saw_swap = 0, checked = 0;

    for (std::size_t n = 4; n <= 12; ++n) {
        for (int trial = 0; trial < 20; ++trial) {
            mat::dense2D<double> A(n, n), Aorig(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j) {
                    double v = next();
                    A(i,j) = v; A(j,i) = v; Aorig(i,j) = v; Aorig(j,i) = v;
                }

            vec::dense_vector<double> b(n), x(n);
            for (std::size_t i = 0; i < n; ++i) b[i] = next();

            bk_pivot_info pivots;
            if (ldlt_bk_factor(A, pivots) != 0) continue;   // singular draw
            ldlt_bk_solve(A, pivots, x, b);
            ++checked;

            for (std::size_t k = 0; k < n; ++k) {
                if (pivots.ipiv[k] < 0) { ++saw_2x2; break; }
                if (static_cast<std::size_t>(pivots.ipiv[k] - 1) != k) { ++saw_swap; break; }
            }

            INFO("n = " << n << ", trial = " << trial);
            REQUIRE(backward_error(Aorig, x, b) < 1e-10);
        }
    }

    REQUIRE(checked > 0);
    // Guard the guard: if the sweep stopped producing interchanges or 2x2
    // blocks it would pass while covering only the regime that never broke.
    REQUIRE(saw_2x2 > 0);
    REQUIRE(saw_swap > 0);
}
