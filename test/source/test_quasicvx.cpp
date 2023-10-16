// -*- coding: utf-8 -*-
#include <doctest/doctest.h>  // for ResultBuilder, Approx, CHECK_EQ

#include <cmath>                      // for exp
#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_deep_cut
#include <ellalgo/ell.hpp>            // for Ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <ellalgo/ell_stable.hpp>     // for EllStable
#include <tuple>                      // for get, tuple

// using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Vec = std::valarray<double>;

struct MyQuasicCvxOracle {
    using ArrayType = Vec;
    using CutChoices = double;  // single cut
    using Cut = std::pair<Vec, double>;

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] gamma
     * @return std::tuple<Cut, double>
     */
    auto assess_optim(const Vec &z, double &gamma) -> std::tuple<Cut, bool> {
        double sqrtx = z[0];
        double ly = z[1];

        // constraint 1: exp(x) <= y, or sqrtx**2 <= ly
        double fj = sqrtx * sqrtx - ly;
        if (fj > 0.0) {
            return {{Vec{2 * sqrtx, -1.0}, fj}, false};
        }

        // constraint 2: x - y >= 1
        double tmp2 = std::exp(ly);
        double tmp3 = gamma* tmp2;
        fj = -sqrtx + tmp3;
        if (fj < 0.0)  // feasible
        {
            gamma = sqrtx / tmp2;
            return {{Vec{-1.0, sqrtx}, 0}, true};
        }

        return {{Vec{-1.0, tmp3}, fj}, false};
    }
};

TEST_CASE("xtensor") {
    auto x = Vec{};
    CHECK_EQ(x.size(), 0U);

    x = Vec{1.0, 2.0};
    CHECK_NE(x.size(), 0U);
}

TEST_CASE("Quasiconvex 1, test feasible") {
    Ell<Vec> ellip{10.0, Vec{0.0, 0.0}};

    auto omega = MyQuasicCvxOracle{};
    auto gamma = 0.0;
    const auto options = Options{2000, 1e-8};
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const Vec &x = std::get<0>(result);
    REQUIRE_EQ(x.size(), 2U);
    // CHECK_EQ(-gamma, doctest::Approx(-0.4288673397));
    // CHECK_EQ(x[0] * x[0], doctest::Approx(0.499876));
    // CHECK_EQ(std::exp(x[1]), doctest::Approx(1.64852));
    // const auto &x = std::get<0>(result);
    // const CInfo &num_iters = std::get<1>(result);
    // CHECK(ell_info.feasible);
    CHECK_EQ(-gamma, doctest::Approx(-0.4288673397));
    CHECK_EQ(x[0] * x[0], doctest::Approx(0.5029823096));
    CHECK_EQ(std::exp(x[1]), doctest::Approx(1.6536872635));
}

TEST_CASE("Quasiconvex 1, test feasible (stable)") {
    EllStable<Vec> ellip{10.0, Vec{0.0, 0.0}};
    auto omega = MyQuasicCvxOracle{};
    auto gamma = 0.0;
    const auto options = Options{2000, 1e-8};
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto x = std::get<0>(result);
    REQUIRE_EQ(x.size(), 2U);
    // const auto &num_iters = std::get<1>(result);
    // CHECK(ell_info.feasible);

    // const auto x = *x_opt;
    // CHECK_EQ(-gamma, doctest::Approx(-0.4288673397));
    // CHECK_EQ(x[0] * x[0], doctest::Approx(0.5029823096));
    // CHECK_EQ(std::exp(x[1]), doctest::Approx(1.6536872635));
}
