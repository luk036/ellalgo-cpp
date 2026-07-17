/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 *
 *  Benchmark comparing Ell vs EllStable runtime performance
 *  with correctness verification across multiple problem sizes.
 */
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <valarray>

#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <ellalgo/ell_config.hpp>
#include <ellalgo/ell_stable.hpp>
#include <ellalgo/oracles/lowpass_oracle.hpp>
#include <ellalgo/oracles/profit_oracle.hpp>

#include "benchmark/benchmark.h"

using Vec = std::valarray<double>;

// ============================================================
//  Shared test data
// ============================================================

static const auto unit_price = 20.0;
static const auto scale      = 40.0;
static const auto limit      = 30.5;
static const auto elasticities = Vec{0.1, 0.4};
static const auto price_out    = Vec{10.0, 35.0};

// ============================================================
//  Helper: approximate valarray comparison
// ============================================================

static bool approx_equal(const Vec& a, const Vec& b, double tol = 1e-8) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) continue;
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// ============================================================
//  Deterministic random-cut oracle (iteration-dependent)
//  Same cuts regardless of floating-point path.
// ============================================================

class DetRandOracle {
    using Cut = std::pair<Vec, double>;
    size_t _n;
    unsigned _iter = 0;
public:
    using ArrayType = Vec;
    DetRandOracle(size_t n) : _n{n} {}

    auto assess_optim(const Vec& /*x*/, double& /*gamma*/) -> std::tuple<Cut, bool> {
        auto seed = _iter++;
        std::mt19937 rng{seed};
        Vec g(_n);
        for (size_t i = 0; i < _n; ++i)
            g[i] = std::normal_distribution<double>{0.0, 1.0}(rng);
        return {{g, 0.01}, false};
    }
};

// ============================================================
//  Correctness Verification
// ============================================================

struct VerifResult {
    std::string name;
    bool pass;
    Vec x_ell;
    Vec x_stable;
    size_t iters_ell{};
    size_t iters_stable{};
};

struct LowpassCase {
    size_t N;
    const char* label;
    bool parallel_cut;
};

struct ScaleCase {
    size_t N;
    const char* label;
    size_t max_iters;
};

static VerifResult verify_profit_normal() {
    Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
    double gamma = 0.0;
    auto [x_ell, iters_ell] = cutting_plane_optim(omega, ellip, gamma);

    EllStable<Vec> ellip2{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega2{unit_price, scale, limit, elasticities, price_out};
    double gamma2 = 0.0;
    auto [x_stable, iters_stable] = cutting_plane_optim(omega2, ellip2, gamma2);

    bool pass = approx_equal(x_ell, x_stable, 1e-6);
    return {"Profit normal", pass, std::move(x_ell), std::move(x_stable), iters_ell, iters_stable};
}

static VerifResult verify_profit_rb() {
    Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracleRb omega{unit_price, scale, limit, elasticities, price_out,
                         Vec{0.003, 0.007}, 1.0};
    double gamma = 0.0;
    auto [x_ell, iters_ell] = cutting_plane_optim(omega, ellip, gamma);

    EllStable<Vec> ellip2{100.0, Vec{0.0, 0.0}};
    ProfitOracleRb omega2{unit_price, scale, limit, elasticities, price_out,
                          Vec{0.003, 0.007}, 1.0};
    double gamma2 = 0.0;
    auto [x_stable, iters_stable] = cutting_plane_optim(omega2, ellip2, gamma2);

    bool pass = approx_equal(x_ell, x_stable, 1e-6);
    return {"Profit robust", pass, std::move(x_ell), std::move(x_stable), iters_ell, iters_stable};
}

static VerifResult verify_lowpass(const LowpassCase& tc) {
    auto [omega, spsq] = create_lowpass_case(tc.N);
    Ell<Vec> ellip{40.0, Vec(0.0, tc.N)};
    ellip.set_use_parallel_cut(tc.parallel_cut);
    Options opts; opts.max_iters = 50000;
    auto [x_ell, iters_ell] = cutting_plane_optim(omega, ellip, spsq, opts);

    auto [omega2, spsq2] = create_lowpass_case(tc.N);
    EllStable<Vec> ellip2{40.0, Vec(0.0, tc.N)};
    ellip2.set_use_parallel_cut(tc.parallel_cut);
    Options opts2; opts2.max_iters = 50000;
    auto [x_stable, iters_stable] = cutting_plane_optim(omega2, ellip2, spsq2, opts2);

    bool pass = approx_equal(x_ell, x_stable, 1e-6);
    return {tc.label, pass, std::move(x_ell), std::move(x_stable), iters_ell, iters_stable};
}

static VerifResult verify_scaling(const ScaleCase& tc) {
    DetRandOracle ora{tc.N};
    Ell<Vec> ellip{40.0, Vec(0.0, tc.N)};
    Options opts; opts.max_iters = tc.max_iters; opts.tolerance = 0.0;
    double gamma = 0.0;
    auto [x_ell, iters_ell] = cutting_plane_optim(ora, ellip, gamma, opts);

    DetRandOracle ora2{tc.N};
    EllStable<Vec> ellip2{40.0, Vec(0.0, tc.N)};
    Options opts2; opts2.max_iters = tc.max_iters; opts2.tolerance = 0.0;
    double gamma2 = 0.0;
    auto [x_stable, iters_stable] = cutting_plane_optim(ora2, ellip2, gamma2, opts2);

    bool pass = approx_equal(x_ell, x_stable, 1e-6);
    return {tc.label, pass, std::move(x_ell), std::move(x_stable), iters_ell, iters_stable};
}

static void run_verification() {
    std::cout << "\n============================================\n";
    std::cout   << "  Ell vs EllStable - Correctness Verification\n";
    std::cout   << "============================================\n\n";

    LowpassCase lpcases[] = {
        {32, "LP-32 par", true},
        {32, "LP-32 ser", false},
        {48, "LP-48 par", true},
        {64, "LP-64 par", true},
    };
    ScaleCase scases[] = {
        {16, "Rand-16", 4000},
        {32, "Rand-32", 4000},
        {64, "Rand-64", 2000},
    };

    static constexpr int N_VERIF = 9;
    VerifResult results[N_VERIF] = {
        verify_profit_normal(),
        verify_profit_rb(),
        verify_lowpass(lpcases[0]),
        verify_lowpass(lpcases[1]),
        verify_lowpass(lpcases[2]),
        verify_lowpass(lpcases[3]),
        verify_scaling(scases[0]),
        verify_scaling(scases[1]),
        verify_scaling(scases[2]),
    };

    size_t passed = 0;
    for (int i = 0; i < N_VERIF; ++i) {
        const auto& r = results[i];
        std::cout << std::left << std::setw(18) << r.name << "  ";
        if (r.pass) { std::cout << "  PASS"; ++passed; }
        else         { std::cout << "  FAIL"; }
        std::cout << "  | iters: Ell=" << std::setw(5) << r.iters_ell
                  << "  Stable=" << std::setw(5) << r.iters_stable
                  << "  | xc[0]=" << (r.x_ell.size() ? r.x_ell[0] : -1.0)
                  << "  dim=" << (r.x_ell.size() ? r.x_ell.size() : 0);
        if (!r.pass && r.x_ell.size() > 0 && r.x_stable.size() > 0) {
            double maxd = 0.0;
            for (size_t j = 0; j < r.x_ell.size(); ++j)
                maxd = std::max(maxd, std::abs(r.x_ell[j] - r.x_stable[j]));
            std::cout << "  | maxdiff=" << maxd;
        }
        std::cout << '\n';
    }

    std::cout << "\n  Result: " << passed << "/" << N_VERIF << " passed.\n";
    std::cout << "============================================\n\n";
    if (passed != size_t(N_VERIF)) {
        std::cerr << "ERROR: Verification failed - aborting benchmarks.\n";
        std::exit(1);
    }
}

// ============================================================
//  Benchmark helpers
// ============================================================

enum class Alg { Ell, EllStable };

template <Alg alg>
static void run_lowpass_bench(benchmark::State& state, size_t N, bool parallel_cut) {
    while (state.KeepRunning()) {
        auto [omega, spsq] = create_lowpass_case(N);
        Options opts; opts.max_iters = 50000;
        if constexpr (alg == Alg::Ell) {
            Ell<Vec> ellip{40.0, Vec(0.0, N)};
            ellip.set_use_parallel_cut(parallel_cut);
            auto result = cutting_plane_optim(omega, ellip, spsq, opts);
            benchmark::DoNotOptimize(result);
        } else {
            EllStable<Vec> ellip{40.0, Vec(0.0, N)};
            ellip.set_use_parallel_cut(parallel_cut);
            auto result = cutting_plane_optim(omega, ellip, spsq, opts);
            benchmark::DoNotOptimize(result);
        }
    }
}

template <Alg alg>
static void run_scale_bench(benchmark::State& state, size_t N, size_t max_iters) {
    while (state.KeepRunning()) {
        DetRandOracle ora{N};
        Options opts; opts.max_iters = max_iters; opts.tolerance = 0.0;
        double gamma = 0.0;
        if constexpr (alg == Alg::Ell) {
            Ell<Vec> ellip{40.0, Vec(0.0, N)};
            auto result = cutting_plane_optim(ora, ellip, gamma, opts);
            benchmark::DoNotOptimize(result);
        } else {
            EllStable<Vec> ellip{40.0, Vec(0.0, N)};
            auto result = cutting_plane_optim(ora, ellip, gamma, opts);
            benchmark::DoNotOptimize(result);
        }
    }
}

// ============================================================
//  Lowpass benchmarks
// ============================================================

static void BM_Ell_lp32_par(benchmark::State& s) { run_lowpass_bench<Alg::Ell>(s, 32, true); }
BENCHMARK(BM_Ell_lp32_par);
static void BM_Stable_lp32_par(benchmark::State& s) { run_lowpass_bench<Alg::EllStable>(s, 32, true); }
BENCHMARK(BM_Stable_lp32_par);

static void BM_Ell_lp32_ser(benchmark::State& s) { run_lowpass_bench<Alg::Ell>(s, 32, false); }
BENCHMARK(BM_Ell_lp32_ser);
static void BM_Stable_lp32_ser(benchmark::State& s) { run_lowpass_bench<Alg::EllStable>(s, 32, false); }
BENCHMARK(BM_Stable_lp32_ser);

static void BM_Ell_lp48_par(benchmark::State& s) { run_lowpass_bench<Alg::Ell>(s, 48, true); }
BENCHMARK(BM_Ell_lp48_par);
static void BM_Stable_lp48_par(benchmark::State& s) { run_lowpass_bench<Alg::EllStable>(s, 48, true); }
BENCHMARK(BM_Stable_lp48_par);

static void BM_Ell_lp64_par(benchmark::State& s) { run_lowpass_bench<Alg::Ell>(s, 64, true); }
BENCHMARK(BM_Ell_lp64_par);
static void BM_Stable_lp64_par(benchmark::State& s) { run_lowpass_bench<Alg::EllStable>(s, 64, true); }
BENCHMARK(BM_Stable_lp64_par);

// ============================================================
//  Profit benchmarks
// ============================================================

static void BM_Ell_norm(benchmark::State& s) {
    while (s.KeepRunning()) {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        benchmark::DoNotOptimize(cutting_plane_optim(omega, ellip, gamma));
    }
}
BENCHMARK(BM_Ell_norm);

static void BM_Stable_norm(benchmark::State& s) {
    while (s.KeepRunning()) {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        benchmark::DoNotOptimize(cutting_plane_optim(omega, ellip, gamma));
    }
}
BENCHMARK(BM_Stable_norm);

static void BM_Ell_rb(benchmark::State& s) {
    while (s.KeepRunning()) {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price, scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        benchmark::DoNotOptimize(cutting_plane_optim(omega, ellip, gamma));
    }
}
BENCHMARK(BM_Ell_rb);

static void BM_Stable_rb(benchmark::State& s) {
    while (s.KeepRunning()) {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price, scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        benchmark::DoNotOptimize(cutting_plane_optim(omega, ellip, gamma));
    }
}
BENCHMARK(BM_Stable_rb);

// ============================================================
//  Scaling benchmarks (deterministic random cuts)
//  Run for fixed max_iters to measure per-iteration speed.
// ============================================================

static void BM_Ell_s16(benchmark::State& s) { run_scale_bench<Alg::Ell>(s, 16, 4000); }
BENCHMARK(BM_Ell_s16);
static void BM_Stable_s16(benchmark::State& s) { run_scale_bench<Alg::EllStable>(s, 16, 4000); }
BENCHMARK(BM_Stable_s16);

static void BM_Ell_s32(benchmark::State& s) { run_scale_bench<Alg::Ell>(s, 32, 4000); }
BENCHMARK(BM_Ell_s32);
static void BM_Stable_s32(benchmark::State& s) { run_scale_bench<Alg::EllStable>(s, 32, 4000); }
BENCHMARK(BM_Stable_s32);

static void BM_Ell_s64(benchmark::State& s) { run_scale_bench<Alg::Ell>(s, 64, 2000); }
BENCHMARK(BM_Ell_s64);
static void BM_Stable_s64(benchmark::State& s) { run_scale_bench<Alg::EllStable>(s, 64, 2000); }
BENCHMARK(BM_Stable_s64);

static void BM_Ell_s128(benchmark::State& s) { run_scale_bench<Alg::Ell>(s, 128, 1000); }
BENCHMARK(BM_Ell_s128);
static void BM_Stable_s128(benchmark::State& s) { run_scale_bench<Alg::EllStable>(s, 128, 1000); }
BENCHMARK(BM_Stable_s128);

static void BM_Ell_s256(benchmark::State& s) { run_scale_bench<Alg::Ell>(s, 256, 500); }
BENCHMARK(BM_Ell_s256);
static void BM_Stable_s256(benchmark::State& s) { run_scale_bench<Alg::EllStable>(s, 256, 500); }
BENCHMARK(BM_Stable_s256);

// ============================================================
//  Custom main: verify first, then benchmark
// ============================================================

int main(int argc, char** argv) {
    run_verification();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
