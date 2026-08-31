/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <chrono>                                // for steady_clock, duration
#include <cmath>                                // for log10, sqrt
#include <cstddef>                              // for size_t
#include <cstdio>                               // for printf
#include <string>                               // for to_string
#include <tuple>                                // for get
#include <utility>                              // for move
#include <valarray>                             // for valarray

#include <ellalgo/cutting_plane.hpp>            // for cutting_plane_optim
#include <ellalgo/ell.hpp>                      // for Ell
#include <ellalgo/ell_config.hpp>               // for Options
#include <ellalgo/oracles/lowpass_oracle.hpp>   // for LowpassOracle, create_lowpass_case

using Vec = std::valarray<double>;

/**
 * @brief Solve the FIR lowpass design problem for a given filter length.
 *
 * @param[in] n number of FIR coefficients (autocorrelation length).
 * @return (wall time in seconds, iterations, final stopband power).
 */
static auto solve_lowpass(std::size_t n) -> std::tuple<double, std::size_t, double> {
    auto r0 = Vec(0.0, n);  // initial x0
    auto ellip = Ell<Vec>(40.0, r0);
    ellip.set_use_parallel_cut(true);
    auto [omega, gamma] = create_lowpass_case(n);
    auto options = Options();
    options.max_iters = 50000;
    options.tolerance = 1e-14;
    const auto start = std::chrono::steady_clock::now();
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto stop = std::chrono::steady_clock::now();
    const auto secs = std::chrono::duration<double>(stop - start).count();
    const auto num_iters = std::get<1>(result);
    return {secs, num_iters, gamma};
}

int main() {
    static constexpr std::size_t SIZES[] = {24, 32, 48, 64, 80};  // filter lengths
    std::printf("%4s %14s %8s %10s %10s\n", "n", "ellipsoid(s)", "iters", "atten(dB)", "db_rel");
    std::printf("%s\n", std::string(52, '-').c_str());
    // Reference attenuation from the Python/CVXPY run for relative-error reporting.
    static constexpr double DB_REF[] = {-20.38, -33.84, -53.62, -74.89, -84.64};
    for (std::size_t i = 0; i != std::size(SIZES); ++i) {
        const auto n = SIZES[i];
        const auto [secs, iters, gamma] = solve_lowpass(n);
        const auto db = 20.0 * std::log10(std::sqrt(gamma));
        const auto db_rel = (DB_REF[i] != 0.0) ? db / DB_REF[i] : 0.0;
        std::printf("%4zu %14.6f %8zu %10.2f %10.4f\n", n, secs, iters, db, db_rel);
    }

    ankerl::nanobench::Bench bench;
    bench.title("FIR lowpass design: ellipsoid method (n = 24..80)").unit("op").warmup(1).epochs(3);
    for (const auto n : SIZES) {
        bench.run("n=" + std::to_string(n), [&] {
            auto r0 = Vec(0.0, n);
            auto ellip = Ell<Vec>(40.0, r0);
            ellip.set_use_parallel_cut(true);
            auto [omega, gamma] = create_lowpass_case(n);
            auto options = Options();
            options.max_iters = 50000;
            options.tolerance = 1e-14;
            const auto result = cutting_plane_optim(omega, ellip, gamma, options);
            ankerl::nanobench::doNotOptimizeAway(result);
        });
    }
}
