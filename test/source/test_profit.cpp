/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include <doctest/doctest.h>

#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <ellalgo/ell_stable.hpp>
#include <ellalgo/oracles/profit_oracle.hpp>
#include <xtensor/xarray.hpp>

// using namespace fun;

TEST_CASE("Profit Test") {
    using Vec = xt::xarray<double, xt::layout_type::row_major>;

    const auto p = 20.;
    const auto A = 40.;
    const auto k = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10., 35.};

    {
        ell E{100., Vec{0., 0.}};
        profit_oracle P{p, A, k, a, v};

        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 37);
    }

    {
        ell E{100., Vec{0., 0.}};
        profit_rb_oracle P{p, A, k, a, v, Vec{0.003, 0.007}, 1.};
        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 42);
    }

    {
        ell E{100., Vec{2., 0.}};
        profit_q_oracle P{p, A, k, a, v};
        const auto result = cutting_plane_q(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 28);
    }
}

TEST_CASE("Profit Test (Stable)") {
    using Vec = xt::xarray<double, xt::layout_type::row_major>;

    const auto p = 20.;
    const auto A = 40.;
    const auto k = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10., 35.};

    {
        ell_stable E{100., Vec{0., 0.}};
        profit_oracle P{p, A, k, a, v};

        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 42);
    }

    {
        ell_stable E{100., Vec{0., 0.}};
        profit_rb_oracle P{p, A, k, a, v, Vec{0.003, 0.007}, 1.};
        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 38);
    }

    {
        ell_stable E{100., Vec{2., 0.}};
        profit_q_oracle P{p, A, k, a, v};
        const auto result = cutting_plane_q(std::move(P), std::move(E), 0.);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 30);
    }
}
