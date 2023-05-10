/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include <doctest/doctest.h> // for ResultBuilder, CHECK

#include <ellalgo/cutting_plane.hpp> // for cutting_plane_optim, cutti...
#include <ellalgo/ell.hpp>           // for Ell
#include <ellalgo/ell_config.hpp>    // for CInfo
#include <ellalgo/ell_stable.hpp>    // for EllStable
#include <ellalgo/oracles/profit_oracle.hpp> // for ProfitOracle, profit_r...

#include <cmath>       // for log
#include <tuple>       // for get
#include <type_traits> // for move, remove_reference<...

TEST_CASE("Profit Test") {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;

  const auto p = 20.0;
  const auto A = 40.0;
  const auto k = 30.5;
  const auto a = Vec{0.1, 0.4};
  const auto v = Vec{10.0, 35.0};

  {
    Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega{p, A, k, a, v};

    const auto result =
        cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    REQUIRE_EQ(y.size(), 2U);
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 36);
  }

  {
    Ell<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracleRb omega{p, A, k, a, v, Vec{0.003, 0.007}, 1.0};
    const auto result =
        cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    // CHECK(y != Vec{});
    REQUIRE_EQ(y.size(), 2U);
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 41);
  }

  {
    Ell<Vec> ellip{100.0, Vec{2.0, 0.0}};
    ProfitOracleQ omega{p, A, k, a, v};
    const auto result =
        cutting_plane_q(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    REQUIRE_EQ(y.size(), 2U);
    // CHECK(y != Vec{});
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 27);
  }
}

TEST_CASE("Profit Test (Stable)") {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;

  const auto p = 20.0;
  const auto A = 40.0;
  const auto k = 30.5;
  const auto a = Vec{0.1, 0.4};
  const auto v = Vec{10.0, 35.0};

  {
    EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracle omega{p, A, k, a, v};

    const auto result =
        cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    // CHECK(y != Vec{});
    REQUIRE_EQ(y.size(), 2U);
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 36);
  }

  {
    EllStable<Vec> ellip{100.0, Vec{0.0, 0.0}};
    ProfitOracleRb omega{p, A, k, a, v, Vec{0.003, 0.007}, 1.0};
    const auto result =
        cutting_plane_optim(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    // CHECK(y != Vec{});
    REQUIRE_EQ(y.size(), 2U);
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 41);
  }

  {
    EllStable<Vec> ellip{100.0, Vec{2.0, 0.0}};
    ProfitOracleQ omega{p, A, k, a, v};
    const auto result =
        cutting_plane_q(std::move(omega), std::move(ellip), 0.0);
    const auto &y = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);
    // CHECK(y != Vec{});
    REQUIRE_EQ(y.size(), 2U);
    CHECK(y[0] <= std::log(k));
    CHECK_EQ(num_iters, 27);
  }
}
