#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/ell_calc_core.hpp>  // for EllCalcCore
#include <tuple>                      // for get, tuple

TEST_CASE("EllCalcCore, test central cut") {
    auto ell_calc_core = EllCalcCore(4);
    double rho;
    double sigma;
    double delta;
    std::tie(rho, sigma, delta) = ell_calc_core.calc_central_cut(2.0);
    CHECK_EQ(rho, doctest::Approx(0.4));
    CHECK_EQ(sigma, doctest::Approx(0.4));
    CHECK_EQ(delta, doctest::Approx(16.0 / 15.0));
}

TEST_CASE("EllCalcCore, test bias cut") {
    auto ell_calc_core = EllCalcCore(4);
    double rho;
    double sigma;
    double delta;
    std::tie(rho, sigma, delta) = ell_calc_core.calc_bias_cut(1.0, 2.0);
    CHECK_EQ(rho, doctest::Approx(1.2));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));
}

TEST_CASE("EllCalcCore, test parallel central cut") {
    auto ell_calc_core = EllCalcCore(4);
    double rho;
    double sigma;
    double delta;
    std::tie(rho, sigma, delta) = ell_calc_core.calc_parallel_central_cut(1.0, 4.0);
    CHECK_EQ(rho, doctest::Approx(0.4));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(1.2));
}

TEST_CASE("EllCalcCore, test parallel cut") {
    auto ell_calc_core = EllCalcCore(4);
    double rho;
    double sigma;
    double delta;
    std::tie(rho, sigma, delta) = ell_calc_core.calc_parallel_cut(1.0, 2.0, 4.0);
    CHECK_EQ(rho, doctest::Approx(1.2));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));
}
