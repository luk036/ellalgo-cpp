/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include "benchmark/benchmark.h" // for BENCHMARK, State, BENCHMARK_...

#include <ellalgo/cutting_plane.hpp> // for cutting_plane_optim, cutti...
#include <ellalgo/ell.hpp>           // for Ell
#include <ellalgo/ell_config.hpp>    // for CInfo
#include <ellalgo/ell_stable.hpp>    // for EllStable
#include <ellalgo/oracles/profit_oracle.hpp> // for ProfitOracle, profit_r...

#include <cmath>       // for log
#include <tuple>       // for get
#include <type_traits> // for move, remove_reference<...

static void bm_ell_normal(benchmark::State &state) {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;

  const auto p = 20.0;
  const auto A = 40.0;
  const auto k = 30.5;
  const auto a = Vec{0.1, 0.4};
  const auto v = Vec{10.0, 35.0};

  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega{p, A, k, a, v};

    auto result = cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    // CHECK_EQ(num_iters, 36);
    benchmark::DoNotOptimize(result);
  }
}
// Register the function as a benchmark
BENCHMARK(bm_ell_normal);

static void bm_ell_stable(benchmark::State &state) {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;

  const auto p = 20.0;
  const auto A = 40.0;
  const auto k = 30.5;
  const auto a = Vec{0.1, 0.4};
  const auto v = Vec{10.0, 35.0};

  // Code inside this loop is measured repeatedly
  for (auto _ : state) {
    EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega{p, A, k, a, v};

    auto result = cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    // CHECK_EQ(num_iters, 41);
    benchmark::DoNotOptimize(result);
  }
}
// Register the function as a benchmark
BENCHMARK(bm_ell_stable);

BENCHMARK_MAIN();
