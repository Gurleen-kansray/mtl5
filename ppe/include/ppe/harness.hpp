#pragma once
// PPE measurement harness: timing, verification, and the throughput/efficiency
// model.
//
// Throughput is reported in GOP/s (giga-operations per second) rather than
// GFLOP/s, because half the type matrix is integer. A GEMM performs 2*m*n*k
// operations (one multiply and one add per inner step) under the usual
// convention, for integer and floating-point alike, so the two halves are
// directly comparable on the same axis.
//
// Efficiency is throughput as a fraction of a STATED per-type peak model. The
// model is an estimate, and is documented in ppe/docs/00-overview.md together
// with what it does and does not account for -- a number like "72% of peak" is
// only meaningful alongside the peak it is measured against.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace ppe {

struct measurement {
    std::string kernel;
    std::string type;
    std::size_t n = 0;
    double      gops = 0.0;      // giga-operations per second
    double      peak_gops = 0.0; // modelled peak for this type
    double      efficiency = 0.0;// gops / peak_gops
    double      seconds = 0.0;   // median wall clock of one call
    std::size_t reps = 0;
    bool        correct = false; // matched the v0 reference
};

/// Operations in one m x n x k GEMM: one multiply and one add per inner step.
inline double gemm_ops(std::size_t m, std::size_t n, std::size_t k) {
    return 2.0 * double(m) * double(n) * double(k);
}

/// Fill with small values so integer accumulation cannot overflow the
/// accumulator at the sizes tested, and so floating-point results stay exactly
/// representable -- which lets verification use exact equality for integers and
/// a tight relative tolerance for floats.
template <typename T>
void fill(std::vector<T>& v, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    for (auto& x : v) {
        const int r = int(g() % 9) - 4;          // [-4, 4]
        x = static_cast<T>(r);
    }
}

/// Median of the per-call wall clock over `reps` calls, after one warm-up.
template <typename F>
double time_median(F&& f, std::size_t reps) {
    f();                                          // warm up: caches, page faults
    std::vector<double> t;
    t.reserve(reps);
    for (std::size_t r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

/// Repetitions that keep each measurement around a tenth of a second without
/// spending minutes on the naive kernels at large N.
inline std::size_t reps_for(double seconds_estimate) {
    if (seconds_estimate <= 0.0) return 5;
    const std::size_t r = static_cast<std::size_t>(0.1 / seconds_estimate);
    return std::min<std::size_t>(std::max<std::size_t>(r, 1), 15);
}

/// Exact for integers; relative tolerance for floating point.
template <typename TC>
bool matches(const std::vector<TC>& got, const std::vector<TC>& ref) {
    if (got.size() != ref.size()) return false;
    if constexpr (std::is_integral_v<TC>) {
        for (std::size_t i = 0; i < got.size(); ++i)
            if (got[i] != ref[i]) return false;
    } else {
        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double g = double(got[i]), r = double(ref[i]);
            worst = std::max(worst, std::fabs(g - r));
            scale = std::max(scale, std::fabs(r));
        }
        return worst <= 1e-6 * (scale > 0.0 ? scale : 1.0);
    }
    return true;
}

}  // namespace ppe
