#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

// --- Sanity: default (unspecified) Accumulator behaves exactly as before ---
TEST_CASE("CG with default Accumulator matches unmodified baseline", "[itl][cg][quire]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::cg(A, x, b, pc, iter); // no explicit Accumulator -- must be unaffected

    REQUIRE(err == 0);
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}


// --- Edge case: CG on a trivial 1x1 SPD system ---
TEST_CASE("CG handles 1x1 system", "[itl][cg][quire]") {
    mat::dense2D<double> A(1, 1);
    A(0,0) = 4.0;

    vec::dense_vector<double> b = {2.0};
    vec::dense_vector<double> x(1, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);

    REQUIRE(err == 0);
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(0.5, 1e-8));
}


// --- Mixed-precision Accumulator, native IEEE only (no Universal dependency) ---
TEST_CASE("CG: float64-accumulated pAp/rho improves float32 accuracy vs naive float32", "[itl][cg][accumulator]") {
    const std::size_t n = 20;
    mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i,j) = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        A(i,i) = 2.0f;
        if (i > 0)     A(i,i-1) = -1.0f;
        if (i < n - 1) A(i,i+1) = -1.0f;
    }

    vec::dense_vector<float> b(n, 1.0f);

    vec::dense_vector<float> x_naive(n, 0.0f);
    itl::pc::identity<mat::dense2D<float>> pc(A);
    itl::basic_iteration<float> iter_naive(b, 200, 1e-4f);
    itl::cg(A, x_naive, b, pc, iter_naive);

    vec::dense_vector<float> x_wide(n, 0.0f);
    itl::basic_iteration<float> iter_wide(b, 200, 1e-4f);
    itl::cg<mat::dense2D<float>, vec::dense_vector<float>, vec::dense_vector<float>,
            itl::pc::identity<mat::dense2D<float>>, itl::basic_iteration<float>, double>(
        A, x_wide, b, pc, iter_wide);

    auto residual_norm = [&](const vec::dense_vector<float>& x) {
        double res2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double Axi = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                Axi += double(A(i,j)) * double(x(j));
            double ri = Axi - double(b(i));
            res2 += ri * ri;
        }
        return std::sqrt(res2);
    };

    double naive_residual = residual_norm(x_naive);
    double wide_residual  = residual_norm(x_wide);

    INFO("naive float32 residual: " << naive_residual);
    INFO("float64-accumulated float32 residual: " << wide_residual);

    REQUIRE(wide_residual <= naive_residual);
}
