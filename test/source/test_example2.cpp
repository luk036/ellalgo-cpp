/* -*- coding: utf-8 -*- */
#include <doctest/doctest.h>

#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <ellalgo/ell_stable.hpp>
// #include <optional>

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::tuple<Arr, double>;

/**
 * @brief
 *
 * @param[in] z
 * @return std::optional<Cut>
 */
auto my_oracle2(const Arr& z) -> Cut* {
    static auto cut1 = Cut{Arr{1., 1.}, 0.};
    static auto cut2 = Cut{Arr{-1., 1.}, 0.};

    auto x = z[0];
    auto y = z[1];

    // constraint 1: x + y <= 3
    auto fj = x + y - 3.;
    if (fj > 0.) {
        std::get<1>(cut1) = fj;
        return &cut1;
    }

    // constraint 2: x - y >= 1
    fj = -x + y + 1.;
    if (fj > 0.) {
        std::get<1>(cut2) = fj;
        return &cut2;
    }

    return nullptr;
}

TEST_CASE("Example 2") {
    ell_stable E{10., Arr{0., 0.}};

    const auto P = my_oracle2;
    auto ell_info = cutting_plane_feas(P, E);
    CHECK(ell_info.feasible);
}
