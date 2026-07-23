/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <ellalgo/cutting_plane.hpp>           // for cutting_plane_optim
#include <ellalgo/ell.hpp>                     // for Ell
#include <ellalgo/oracles/lowpass_oracle.hpp>  // for LowpassOracle, filter_...
#include <tuple>                               // for make_tuple, tuple
#include <type_traits>                         // for move, add_const<>::type
#include <valarray>

using Vec = std::valarray<double>;
using Mat = std::valarray<Vec>;
using ParallelCut = std::pair<Vec, Vec>;

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

    return std::make_tuple(r.size() != 0U, num_iters);
}

int main() {
    ankerl::nanobench::Bench bench;
    bench.title("Lowpass filter benchmarks").unit("op").warmup(1).epochs(3).minEpochIterations(1);

    bench.run("lowpass_w_parallel_cut", [&] {
        auto result = run_lowpass(true);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("lowpass_wo_parallel_cut", [&] {
        auto result = run_lowpass(false);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
