/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include <doctest/doctest.h>  // for ResultBuilder, CHECK

#include <cmath>                              // for log
#include <ellalgo/cutting_plane.hpp>          // for cutting_plane_dc, cutti...
#include <ellalgo/ell.hpp>                    // for ell
#include <ellalgo/ell_stable.hpp>             // for ell_stable
#include <ellalgo/oracles/profit_oracle.hpp>  // for profit_oracle, profit_r...
#include <tuple>                              // for get
#include <type_traits>                        // for move, remove_reference<...
#include <xtensor/xaccessible.hpp>            // for xconst_accessible
#include <xtensor/xarray.hpp>                 // for xarray_container
#include <xtensor/xlayout.hpp>                // for layout_type, layout_typ...
#include <xtensor/xtensor_forward.hpp>        // for xarray

#include "ellalgo/cut_config.hpp"  // for CInfo

// using namespace fun;

TEST_CASE("Profit Test") {
    using Vec = xt::xarray<double, xt::layout_type::row_major>;

    const auto p = 20.0;
    const auto A = 40.0;
    const auto k = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10.0, 35.0};

    {
        ell E{100.0, Vec{0.0, 0.0}};
        profit_oracle P{p, A, k, a, v};

        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 37);
    }

    {
        ell E{100.0, Vec{0.0, 0.0}};
        profit_rb_oracle P{p, A, k, a, v, Vec{0.003, 0.007}, 1.0};
        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 42);
    }

    {
        ell E{100.0, Vec{2.0, 0.0}};
        profit_q_oracle P{p, A, k, a, v};
        const auto result = cutting_plane_q(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 28);
    }
}

TEST_CASE("Profit Test (Stable)") {
    using Vec = xt::xarray<double, xt::layout_type::row_major>;

    const auto p = 20.0;
    const auto A = 40.0;
    const auto k = 30.5;
    const auto a = Vec{0.1, 0.4};
    const auto v = Vec{10.0, 35.0};

    {
        ell_stable E{100.0, Vec{0.0, 0.0}};
        profit_oracle P{p, A, k, a, v};

        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 42);
    }

    {
        ell_stable E{100.0, Vec{0.0, 0.0}};
        profit_rb_oracle P{p, A, k, a, v, Vec{0.003, 0.007}, 1.0};
        const auto result = cutting_plane_dc(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 38);
    }

    {
        ell_stable E{100.0, Vec{2.0, 0.0}};
        profit_q_oracle P{p, A, k, a, v};
        const auto result = cutting_plane_q(std::move(P), std::move(E), 0.0);
        const auto& y = std::get<0>(result);
        const auto& ell_info = std::get<1>(result);
        CHECK(y[0] <= std::log(k));
        CHECK(ell_info.num_iters == 30);
    }
}
