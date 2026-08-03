// PPE driver: run the GEMM progression across the integer and floating-point
// type families and emit one CSV row per (kernel, type, N).
//
//   ppe_gemm [--sizes 64,128,...] [--ghz 5.0] [--csv out.csv]
//
// Every kernel is verified against the v0 naive reference for the same type and
// size before its timing is reported, so a fast wrong answer cannot be recorded
// as a result.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <type_traits>

#include <ppe/kernels.hpp>
#include <ppe/harness.hpp>
#include <ppe/peak.hpp>

namespace {

std::vector<std::size_t> parse_sizes(const char* s) {
    std::vector<std::size_t> v;
    const char* p = s;
    while (*p) {
        v.push_back(static_cast<std::size_t>(std::strtoull(p, nullptr, 10)));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

std::vector<ppe::measurement> g_results;

/// Run every kernel in the progression for one operand/accumulator type pair.
template <typename TA, typename TC>
void run_type(const char* type_name, const std::vector<std::size_t>& sizes, double ghz) {
    const double peak = ppe::peak_gops<TA>(ghz);
    for (std::size_t N : sizes) {
        const std::size_t m = N, n = N, k = N;
        std::vector<TA> A(m * k), B(k * n);
        ppe::fill(A, 0x9E3779B97F4A7C15ull ^ N);
        ppe::fill(B, 0xBF58476D1CE4E5B9ull ^ N);

        std::vector<TC> ref(m * n, TC(0));
        ppe::gemm_v0_naive<TA, TC>(m, n, k, A.data(), B.data(), ref.data());

        const double ops  = ppe::gemm_ops(m, n, k);

        struct entry { const char* name; void (*fn)(std::size_t, std::size_t, std::size_t,
                                                   const TA*, const TA*, TC*); };
        const entry kernels[] = {
            {"v0_naive",   [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v0_naive<TA,TC>(m_,n_,k_,a,b,c); }},
            {"v1_ikj",     [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v1_ikj<TA,TC>(m_,n_,k_,a,b,c); }},
            {"v2_blocked", [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v2_blocked<TA,TC>(m_,n_,k_,a,b,c); }},
            {"v3_packed",  [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v3_packed<TA,TC>(m_,n_,k_,a,b,c); }},
            {"v4_regtile", [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v4_regtile<TA,TC>(m_,n_,k_,a,b,c); }},
            {"v5_micro",   [](std::size_t m_, std::size_t n_, std::size_t k_,
                              const TA* a, const TA* b, TC* c) { ppe::gemm_v5_micro<TA,TC>(m_,n_,k_,a,b,c); }},
        };

        for (const auto& kern : kernels) {
            std::vector<TC> C(m * n, TC(0));
            // One untimed call to size the repetition count and to verify.
            const double t_probe = ppe::time_median([&] {
                std::fill(C.begin(), C.end(), TC(0));
                kern.fn(m, n, k, A.data(), B.data(), C.data());
            }, 1);
            const bool ok = ppe::matches(C, ref);

            const std::size_t reps = ppe::reps_for(t_probe);
            const double t = ppe::time_median([&] {
                std::fill(C.begin(), C.end(), TC(0));
                kern.fn(m, n, k, A.data(), B.data(), C.data());
            }, reps);

            ppe::measurement r;
            r.kernel = kern.name;
            r.type = type_name;
            r.n = N;
            r.seconds = t;
            r.reps = reps;
            r.gops = (t > 0.0) ? ops / t / 1e9 : 0.0;
            r.peak_gops = peak;
            r.efficiency = (peak > 0.0) ? r.gops / peak : 0.0;
            r.correct = ok;
            g_results.push_back(r);

            std::printf("  %-11s %-8s N=%-5zu %8.2f GOP/s  %5.1f%% of peak  %s\n",
                        r.kernel.c_str(), type_name, N, r.gops, 100.0 * r.efficiency,
                        ok ? "ok" : "*** MISMATCH ***");
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes{64, 128, 256, 512, 1024};
    double ghz = 5.0;                      // sustained single-core clock, GHz
    std::string csv;

    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(2); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--sizes")) sizes = parse_sizes(next("--sizes"));
        else if (!std::strcmp(argv[i], "--ghz"))   ghz   = std::atof(next("--ghz"));
        else if (!std::strcmp(argv[i], "--csv"))   csv   = next("--csv");
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }

    std::printf("=== integer family (operand -> accumulator) ===\n");
    run_type<std::int8_t,  std::int32_t>("int8",  sizes, ghz);
    run_type<std::int16_t, std::int32_t>("int16", sizes, ghz);
    run_type<std::int32_t, std::int32_t>("int32", sizes, ghz);
    run_type<std::int64_t, std::int64_t>("int64", sizes, ghz);

    std::printf("=== floating-point family ===\n");
#ifdef __FLT16_MAX__
    run_type<_Float16, float>("fp16", sizes, ghz);
#endif
    run_type<float,  float> ("fp32", sizes, ghz);
    run_type<double, double>("fp64", sizes, ghz);

    if (!csv.empty()) {
        std::ofstream out(csv);
        out << "kernel,type,n,gops,peak_gops,efficiency,seconds,reps,correct\n";
        for (const auto& r : g_results)
            out << r.kernel << ',' << r.type << ',' << r.n << ',' << r.gops << ','
                << r.peak_gops << ',' << r.efficiency << ',' << r.seconds << ','
                << r.reps << ',' << (r.correct ? 1 : 0) << '\n';
        std::printf("wrote %s (%zu rows)\n", csv.c_str(), g_results.size());
    }

    std::size_t bad = 0, over = 0;
    for (const auto& r : g_results) {
        if (!r.correct) ++bad;
        // A result above the modelled peak means the MODEL is wrong. Say so
        // rather than publishing an efficiency over 100%.
        if (r.efficiency > 1.0) {
            ++over;
            std::fprintf(stderr,
                "PEAK MODEL TOO LOW: %s %s N=%zu measured %.2f GOP/s vs modelled peak %.2f\n",
                r.kernel.c_str(), r.type.c_str(), r.n, r.gops, r.peak_gops);
        }
    }
    if (bad)  { std::fprintf(stderr, "%zu MISMATCHED results\n", bad); return 1; }
    if (over) { std::fprintf(stderr, "%zu result(s) exceed the peak model\n", over); return 1; }
    return 0;
}
