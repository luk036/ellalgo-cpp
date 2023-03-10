// -*- coding: utf-8 -*-
#include <doctest/doctest.h> // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>   // for cutting_plane_dc
#include <ellalgo/ell_stable.hpp>      // for ell_stable
#include <tuple>                       // for get, tuple
#include <xtensor/xaccessible.hpp>     // for xconst_accessible
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp> // for xarray

#include "ellalgo/cut_config.hpp" // for CInfo, CUTStatus, CUTStatus::...

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::pair<Arr, double>;

/**
 * @brief
 *
 * @param[in] z
 * @param[in,out] t
 * @return std::tuple<Cut, double>
 */
auto my_oracle(const Arr &z, double &t) -> std::tuple<Cut, bool> {
  auto x = z[0];
  auto y = z[1];

  // constraint 1: x + y <= 3
  auto fj = x + y - 3.0;
  if (fj > 0.0) {
    return {{Arr{1.0, 1.0}, fj}, false};
  }

  // constraint 2: x - y >= 1
  fj = -x + y + 1.0;
  if (fj > 0.0) {
    return {{Arr{-1.0, 1.0}, fj}, false};
  }

  // objective: maximize x + y
  auto f0 = x + y;
  fj = t - f0;
  if (fj < 0.0) {
    t = f0;
    return {{Arr{-1.0, -1.0}, 0.0}, true};
  }
  return {{Arr{-1.0, -1.0}, fj}, false};
}

TEST_CASE("Example 1, test feasible") {
  ell_stable E{10.0, Arr{0.0, 0.0}};
  const auto P = my_oracle;
  auto t = -1.e100; // std::numeric_limits<double>::min()
  const auto result = cutting_plane_dc(P, E, t);
  const auto &x = std::get<0>(result);
  const auto &ell_info = std::get<1>(result);
  CHECK(x[0] >= 0.0);
  CHECK(ell_info.feasible);
}

TEST_CASE("Example 1, test infeasible 1") {
  ell_stable E{10.0, Arr{100.0, 100.0}}; // wrong initial guess
                                         // or ellipsoid is too small
  const auto P = my_oracle;
  auto t = -1.e100; // std::numeric_limits<double>::min()
  const auto result = cutting_plane_dc(P, E, t);
  const auto &ell_info = std::get<1>(result);
  CHECK(!ell_info.feasible);
  CHECK_EQ(ell_info.status, CUTStatus::nosoln); // no sol'n
}

TEST_CASE("Example 1, test infeasible 2") {
  ell_stable E{10.0, Arr{0.0, 0.0}};
  const auto P = my_oracle;
  const auto result = cutting_plane_dc(P, E, 100.0); // wrong initial guess
  const auto &ell_info = std::get<1>(result);
  CHECK(!ell_info.feasible);
}
