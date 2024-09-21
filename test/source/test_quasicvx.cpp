// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, Approx, CHECK_EQ

#include <cmath>                      // for exp
#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_bias_cut
#include <ellalgo/ell.hpp>            // for Ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <ellalgo/ell_stable.hpp>     // for EllStable
#include <tuple>                      // for get, tuple

// using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Vec = std::valarray<double>;

struct MyQuasicCvxOracle {
    using ArrayType = Vec;
    using CutChoice = double;  // single cut
    using Cut = std::pair<Vec, double>;

    int idx = -1;
    double tmp2 = 0.0;
    // double tmp3 = 0.0;

    /**
     * The function assess_optim takes a vector xc and a double gamma, assesses optimization
     * constraints, and returns a tuple containing a Cut object and a boolean value.
     *
     * @param[in] xc The `xc` parameter is a `Vec` object that contains two elements. The first
     * element is stored in `xc[0]` and represents the square root of a value (`sqrtx`), while the
     * second element is stored in `xc[1]` and represents another value (`ly
     * @param[in,out] gamma Gamma is a parameter passed by reference to the `assess_optim` function.
     * It is a double type variable that is modified within the function and used in calculations
     * related to constraints.
     *
     * @return The function `assess_optim` returns a `std::tuple` containing a `Cut` object and a
     * boolean value.
     */
    auto assess_optim(const Vec &xc, double &gamma) -> std::tuple<Cut, bool> {
        double sqrtx = xc[0];
        double logy = xc[1];
        double fj;
        double y = std::exp(logy);
        double tmp3;

        for (int i = 0; i != 2; i++) {
            this->idx++;
            if (this->idx == 2) {
                this->idx = 0;  // round robin
            }

            switch (this->idx) {
                case 0:  // constraint 1: exp(x) <= y, or sqrtx**2 <= logy
                    if ((fj = sqrtx * sqrtx - logy) > 0.0) {
                        return {{Vec{2 * sqrtx, -1.0}, fj}, false};
                    }
                    break;
                case 1:  // constraint 2
                    // y = std::exp(logy);
                    tmp3 = gamma * y;
                    if ((fj = -sqrtx + tmp3) > 0.0) {
                        return {{Vec{-1.0, tmp3}, fj}, false};
                    }
                    break;
                default:
                    exit(0);
            }
        }
        gamma = sqrtx / y;
        return {{Vec{-1.0, sqrtx}, 0}, true};
    }
};

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
    const auto num_iters = std::get<1>(result);
    CHECK_EQ(num_iters, 35);
    CHECK_EQ(gamma, doctest::Approx(0.4288673397));
    CHECK_EQ(x[0] * x[0], doctest::Approx(0.496544));
    CHECK_EQ(std::exp(x[1]), doctest::Approx(1.64306));
}

TEST_CASE("Quasiconvex 1, test feasible (stable)") {
    EllStable<Vec> ellip{10.0, Vec{0.0, 0.0}};
    auto omega = MyQuasicCvxOracle{};
    auto gamma = 0.0;
    const auto options = Options{2000, 1e-8};
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto x = std::get<0>(result);
    REQUIRE_EQ(x.size(), 2U);
    const auto num_iters = std::get<1>(result);
    CHECK_EQ(num_iters, 35);
    // const auto &num_iters = std::get<1>(result);
    // CHECK(ell_info.feasible);

    // const auto x = *x_opt;
    // CHECK_EQ(-gamma, doctest::Approx(-0.4288673397));
    // CHECK_EQ(x[0] * x[0], doctest::Approx(0.5029823096));
    // CHECK_EQ(std::exp(x[1]), doctest::Approx(1.6536872635));
}
