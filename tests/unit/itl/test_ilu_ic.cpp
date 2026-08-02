#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

static mat::compressed2D<double> make_tridiagonal(std::size_t n, double diag, double off) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << diag;
            if (i > 0)     ins[i][i-1] << off;
            if (i < n - 1) ins[i][i+1] << off;
        }
    }
    return A;
}

TEST_CASE("ILU(0) preconditioned BiCGSTAB on tridiagonal", "[itl][pc][ilu_0]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ilu_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("ILU(0) converges faster than identity PC", "[itl][pc][ilu_0]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // With identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_identity = iter1.iterations();

    // With ILU(0) PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ilu_0<double> ilu_pc(A);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ilu_pc, iter2);
    int iters_ilu = iter2.iterations();

    REQUIRE(iters_ilu <= iters_identity);
}

TEST_CASE("IC(0) preconditioned CG on SPD tridiagonal", "[itl][pc][ic_0]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ic_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// ---------------------------------------------------------------------------
// Regression: #323 -- ilu_0::solve counted the diagonal in its back-substitution
// off-diagonal sum, so it computed
//     x(i) = (y(i) - sum_{j>=i} U(i,j)*x(j)) / U(i,i)
// instead of
//     x(i) = (y(i) - sum_{j>i}  U(i,j)*x(j)) / U(i,i)
// -- subtracting U(i,i)*x(i) and then dividing by that same U(i,i). Wrong for
// every input.
//
// The existing cases above only check that a preconditioned Krylov solve
// converges, and it still did: an approximate inverse that is merely wrong can
// let BiCGSTAB limp to the answer. These cases instead pin the operator itself
// on inputs where ILU(0) is EXACT, so the expected result is unambiguous.
// ---------------------------------------------------------------------------

TEST_CASE("ILU(0) of a diagonal matrix is an exact solve (#323)", "[itl][pc][ilu_0][regression]") {
    const std::size_t n = 3;
    std::size_t starts[]  = {0, 1, 2, 3};
    std::size_t indices[] = {0, 1, 2};
    double      data[]    = {4.0, 2.0, 8.0};
    mtl::mat::compressed2D<double> A(n, n, 3, starts, indices, data);

    mtl::vec::dense_vector<double> b(n), x(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = 1.0;

    mtl::itl::pc::ilu_0<double> P(A);
    P.solve(x, b);

    // ILU(0) of a diagonal matrix is exact, so solve() must be b / diag.
    REQUIRE_THAT(x[0], Catch::Matchers::WithinAbs(0.25,  1e-14));
    REQUIRE_THAT(x[1], Catch::Matchers::WithinAbs(0.5,   1e-14));
    REQUIRE_THAT(x[2], Catch::Matchers::WithinAbs(0.125, 1e-14));

    // ic_0 was the control that always got this right; they must agree here.
    mtl::vec::dense_vector<double> y(n);
    mtl::itl::pc::ic_0<double> Q(A);
    Q.solve(y, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x[i], Catch::Matchers::WithinAbs(y[i], 1e-14));
}

TEST_CASE("ILU(0) is an exact solve on no-fill patterns (#323)", "[itl][pc][ilu_0][regression]") {
    // ILU(0) drops nothing when elimination creates no fill, so for these
    // patterns L*U == A and solve() must return A^-1 * b exactly. Checked by
    // multiplying the result back through A.
    for (std::size_t n : {2u, 5u, 17u, 40u}) {
        for (int pattern = 0; pattern < 3; ++pattern) {   // tri, lower-bi, upper-bi
            mtl::mat::compressed2D<double> A(n, n);
            {
                mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
                for (std::size_t i = 0; i < n; ++i) {
                    ins[i][i] << 5.0 + static_cast<double>(i % 3);
                    const double off = 1.0 + static_cast<double>((i * 7) % 5) * 0.1;
                    if (pattern != 2 && i > 0)     ins[i][i-1] << -off;
                    if (pattern != 1 && i + 1 < n) ins[i][i+1] <<  off * 0.5;
                }
            }

            mtl::vec::dense_vector<double> b(n), x(n);
            for (std::size_t i = 0; i < n; ++i) b[i] = 1.0 + static_cast<double>(i % 4);

            mtl::itl::pc::ilu_0<double> P(A);
            P.solve(x, b);

            const auto Ax = A * x;
            INFO("n = " << n << ", pattern = " << pattern);
            for (std::size_t i = 0; i < n; ++i)
                REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
        }
    }
}

TEST_CASE("ILU(0) as an exact preconditioner converges immediately (#323)",
          "[itl][pc][ilu_0][regression]") {
    // With L*U == A the preconditioner IS the inverse, so a Krylov method must
    // finish in a single iteration. Before the fix this took 11 -- convergence
    // alone was never evidence the operator was right.
    const std::size_t n = 200;
    mtl::mat::compressed2D<double> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0 + static_cast<double>(i % 3) * 0.25;
            if (i > 0)     ins[i][i-1] << -1.0;
            if (i + 1 < n) ins[i][i+1] << -0.5;
        }
    }
    mtl::vec::dense_vector<double> b(n, 1.0), x(n, 0.0);
    mtl::itl::pc::ilu_0<double> P(A);
    mtl::itl::basic_iteration<double> iter(b, 500, 1e-12, 0.0);
    mtl::itl::bicgstab(A, x, b, P, iter);
    REQUIRE(iter.iterations() <= 2);
}
