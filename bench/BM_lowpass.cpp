#include <ellalgo/cutting_plane.hpp>           // for cutting_plane_optim
#include <ellalgo/ell.hpp>                     // for Ell
#include <ellalgo/oracles/lowpass_oracle.hpp>  // for LowpassOracle, filter_...
#include <tuple>                               // for make_tuple, tuple
#include <type_traits>                         // for move, add_const<>::type
#include <valarray>

#include "benchmark/benchmark.h"  // for BENCHMARK, State, BENCHMARK_...

using Vec = std::valarray<double>;
using Mat = std::valarray<Vec>;
using ParallelCut = std::pair<Vec, Vec>;

// ********************************************************************
// optimization
// ********************************************************************

auto run_lowpass(bool use_parallel_cut) {
    constexpr int N = 32;

    auto r0 = Vec(0.0, N);  // initial x0
    auto ellip = Ell<Vec>(40.0, r0);
    auto result = create_lowpass_case(N);
    auto omega = result.first;
    auto gamma = result.second;
    auto options = Options();

    options.max_iters = 50000;
    ellip.set_use_parallel_cut(use_parallel_cut);
    const auto result2 = cutting_plane_optim(omega, ellip, gamma, options);
    const auto r = std::get<0>(result2);
    const auto num_iters = std::get<1>(result2);

    // std::cout << "lowpass r: " << r << '\n';
    // auto Ustop = 20 * std::log10(std::sqrt(Spsq_new));
    // std::cout << "Min attenuation in the stopband is " << Ustop << " dB.\n";
    // CHECK(r[0] >= 0.0);

    return std::make_tuple(r.size() != 0U, num_iters);
}

// TEST_CASE("Lowpass Filter (w/ parallel cut)") {
static void lowpass_w_parallel_cut(benchmark::State& state) {
    while (state.KeepRunning()) {
        auto result = run_lowpass(true);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(lowpass_w_parallel_cut);

// TEST_CASE("Lowpass Filter (w/o parallel cut)") {
static void lowpass_wo_parallel_cut(benchmark::State& state) {
    while (state.KeepRunning()) {
        auto result = run_lowpass(false);
        benchmark::DoNotOptimize(result);
    }
}
// Register the function as a benchmark
BENCHMARK(lowpass_wo_parallel_cut);

BENCHMARK_MAIN();

/*
------------------------------------------------------------------
Benchmark                        Time             CPU   Iterations
------------------------------------------------------------------
lowpass_w_parallel_cut     4327390 ns      4301226 ns          163
lowpass_wo_parallel_cut   53440994 ns     53162937 ns           13
*/
