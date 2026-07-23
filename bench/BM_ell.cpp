/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cmath>                              // for log
#include <ellalgo/cutting_plane.hpp>          // for cutting_plane_optim, cutti...
#include <ellalgo/ell.hpp>                    // for Ell
#include <ellalgo/ell_config.hpp>             // for CInfo
#include <ellalgo/ell_stable.hpp>             // for EllStable
#include <ellalgo/oracles/profit_oracle.hpp>  // for ProfitOracle, profit_r...
#include <tuple>                              // for get
#include <type_traits>                        // for move, remove_reference<...

using Vec = std::valarray<double>;

static const auto unit_price = 20.0;
static const auto scale = 40.0;
static const auto limit = 30.5;
static const auto elasticities = Vec{0.1, 0.4};
static const auto price_out = Vec{10.0, 35.0};

int main() {
    ankerl::nanobench::Bench bench;
    bench.title("Ell vs EllStable (profit oracle)").unit("op").warmup(100).epochs(50);

    bench.run("ELL_normal", [&] {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("ELL_stable", [&] {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("ELL_normal_rb", [&] {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price,        scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("ELL_stable_rb", [&] {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price,        scale, limit, elasticities, price_out,
                             Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("ELL_normal_q", [&] {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        auto result = cutting_plane_optim_q(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("ELL_stable_q", [&] {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, scale, limit, elasticities, price_out};
        double gamma = 0.0;
        auto result = cutting_plane_optim_q(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
