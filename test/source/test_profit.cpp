/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, CHECK

#include <cmath>                              // for log
#include <ellalgo/cutting_plane.hpp>          // for cutting_plane_optim, cutti...
#include <ellalgo/ell.hpp>                    // for Ell
#include <ellalgo/ell_config.hpp>             // for CInfo
#include <ellalgo/ell_stable.hpp>             // for EllStable
#include <ellalgo/oracles/profit_oracle.hpp>  // for ProfitOracle, profit_r...
#include <tuple>                              // for get

TEST_CASE("Profit Test") {
    using Vec = std::valarray<double>;

    const auto unit_price = 20.0;
    const auto A = 40.0;
    const auto limit = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10.0, 35.0};

    /* The code is performing a test case for the "Profit Test". */
    [&]() {
        Ell<Vec> ellip{Vec{100.0, 100.0}, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, A, limit, a, v};
        double gamma = 0.0;

        const auto __result = cutting_plane_optim(omega, ellip, gamma);
        const auto &y = std::get<0>(__result);
        const auto &num_iters = std::get<1>(__result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 83);
    }();

    [&]() {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price, A, limit, a, v, Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        const auto result = cutting_plane_optim(omega, ellip, gamma);
        const auto &y = std::get<0>(result);
        const auto &num_iters = std::get<1>(result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 90);
    }();

    [&]() {
        Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, A, limit, a, v};
        double gamma = 0.0;
        const auto result = cutting_plane_optim_q(omega, ellip, gamma);
        const auto &y = std::get<0>(result);
        const auto &num_iters = std::get<1>(result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 29);
    }();
}

TEST_CASE("Profit Test (Stable)") {
    using Vec = std::valarray<double>;

    const auto unit_price = 20.0;
    const auto A = 40.0;
    const auto limit = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10.0, 35.0};

    [&]() {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracle omega{unit_price, A, limit, a, v};
        double gamma = 0.0;

        const auto result = cutting_plane_optim(omega, ellip, gamma);
        const auto &y = std::get<0>(result);
        const auto &num_iters = std::get<1>(result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 83);
    }();

    [&]() {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleRb omega{unit_price, A, limit, a, v, Vec{0.003, 0.007}, 1.0};
        double gamma = 0.0;
        const auto result = cutting_plane_optim(omega, ellip, gamma);
        const auto &y = std::get<0>(result);
        const auto &num_iters = std::get<1>(result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 90);
    }();

    [&] {
        EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
        ProfitOracleQ omega{unit_price, A, limit, a, v};
        double gamma = 0.0;
        const auto result = cutting_plane_optim_q(omega, ellip, gamma);
        const auto &y = std::get<0>(result);
        const auto &num_iters = std::get<1>(result);
        REQUIRE_EQ(y.size(), 2U);
        CHECK(y[0] <= std::log(limit));
        CHECK_EQ(num_iters, 29);
    }();
}
