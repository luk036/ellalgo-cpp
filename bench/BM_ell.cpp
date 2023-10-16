/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include <cmath>                              // for log
#include <ellalgo/cutting_plane.hpp>          // for cutting_plane_optim, cutti...
#include <ellalgo/ell.hpp>                    // for Ell
#include <ellalgo/ell_config.hpp>             // for CInfo
#include <ellalgo/ell_stable.hpp>             // for EllStable
#include <ellalgo/oracles/profit_oracle.hpp>  // for ProfitOracle, profit_r...
#include <tuple>                              // for get
#include <type_traits>                        // for move, remove_reference<...

#include "benchmark/benchmark.h"  // for BENCHMARK, State, BENCHMARK_...

using Vec = std::valarray<double>;

static const auto unit_price = 20.0;
static const auto scale = 40.0;
static const auto limit = 30.5;
static const auto elasticities = Vec{0.1, 0.4};
static const auto price_out = Vec{10.0, 35.0};

static void ELL_normal(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;

        auto result = cutting_plane_optim(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_normal);

static void ELL_stable(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;

        auto result = cutting_plane_optim(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_stable);

static void ELL_normal_rb(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price,        scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;

        auto result = cutting_plane_optim(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_normal_rb);

static void ELL_stable_rb(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price,        scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;

        auto result = cutting_plane_optim(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_stable_rb);

static void ELL_normal_q(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;

        auto result = cutting_plane_optim_q(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_normal_q);

static void ELL_stable_q(benchmark::State &state) {
    // Code inside this loop is measured repeatedly
    while (state.KeepRunning()) {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;

        auto result = cutting_plane_optim_q(omega, ellip, gamma);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(ELL_stable_q);

BENCHMARK_MAIN();
