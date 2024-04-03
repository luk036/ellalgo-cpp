#include "ellalgo/ell_config.hpp"
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/ell_calc.hpp>  // for EllCalc
#include <tuple>                 // for get, tuple

TEST_CASE("EllCalc, test central cut") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_central_cut(0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.02));
    CHECK_EQ(sigma, doctest::Approx(0.4));
    CHECK_EQ(delta, doctest::Approx(16.0 / 15.0));
}

TEST_CASE("EllCalc, test deep cut") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_bias_cut(0.11, 0.01);
    CHECK_EQ(status, CutStatus::NoSoln);
    std::tie(status, result) = ell_calc.calc_bias_cut(0.01, 0.01);
    CHECK_EQ(status, CutStatus::Success);

    std::tie(status, result) = ell_calc.calc_bias_cut(0.05, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.06));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));
}

TEST_CASE("EllCalc, test parallel central cut") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_parallel_central_cut(0.05, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.02));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(1.2));
}

TEST_CASE("EllCalc, test parallel deep cut") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.07, 0.05, 0.01);
    CHECK_EQ(status, CutStatus::NoSoln);

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.02));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(1.2));

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.06));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.0232));
    CHECK_EQ(sigma, doctest::Approx(0.928));
    CHECK_EQ(delta, doctest::Approx(1.232));
}

// TEST_CASE("EllCalc, test parallel deep cut (no effect)") {
//     auto ell_calc = EllCalc(4);
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_calc.calc_parallel_bias_cut_q(-0.04, 0.0625, 0.01);
//     CHECK_EQ(status, CutStatus::NoEffect);
// }

TEST_CASE("EllCalc, test deep cut q") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_bias_cut_q(0.11, 0.01);
    CHECK_EQ(status, CutStatus::NoSoln);
    std::tie(status, result) = ell_calc.calc_bias_cut_q(0.01, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(status, result) = ell_calc.calc_bias_cut_q(-0.05, 0.01);
    CHECK_EQ(status, CutStatus::NoEffect);

    std::tie(status, result) = ell_calc.calc_bias_cut_q(0.05, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.06));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));
}

TEST_CASE("EllCalc, test parallel deep cut q") {
    auto ell_calc = EllCalc(4);
    double rho;
    double sigma;
    double delta;
    CutStatus status;
    std::tuple<double, double, double> result;
    std::tie(status, result) = ell_calc.calc_parallel_bias_cut_q(0.07, 0.03, 0.01);
    CHECK_EQ(status, CutStatus::NoSoln);
    std::tie(status, result) = ell_calc.calc_parallel_bias_cut_q(-0.04, 0.0625, 0.01);
    CHECK_EQ(status, CutStatus::NoEffect);

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.02));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(1.2));

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.06));
    CHECK_EQ(sigma, doctest::Approx(0.8));
    CHECK_EQ(delta, doctest::Approx(0.8));

    std::tie(status, result) = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    CHECK_EQ(status, CutStatus::Success);
    std::tie(rho, sigma, delta) = result;
    CHECK_EQ(rho, doctest::Approx(0.0232));
    CHECK_EQ(sigma, doctest::Approx(0.928));
    CHECK_EQ(delta, doctest::Approx(1.232));
}

// TEST_CASE("EllCalc, test bias cut") {
//     auto ell_calc = EllCalc(4);
//     double rho;
//     double sigma;
//     double delta;
//     std::tie(rho, sigma, delta) = ell_calc.calc_bias_cut(0.05, 0.1);
//     CHECK_EQ(rho, doctest::Approx(0.06));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(0.8));
// }

// TEST_CASE("EllCalc, test parallel central cut") {
//     auto ell_calc = EllCalc(4);
//     double rho;
//     double sigma;
//     double delta;
//     std::tie(rho, sigma, delta) = ell_calc.calc_parallel_central_cut(1.0, 4.0);
//     CHECK_EQ(rho, doctest::Approx(0.4));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(1.2));
// }

// TEST_CASE("EllCalc, test parallel cut") {
//     auto ell_calc = EllCalc(4);
//     double rho;
//     double sigma;
//     double delta;
//     std::tie(rho, sigma, delta) = ell_calc.calc_parallel_cut(0.01, 0.04, 0.01);
//     CHECK_EQ(rho, doctest::Approx(0.0232));
//     CHECK_EQ(sigma, doctest::Approx(0.928));
//     CHECK_EQ(delta, doctest::Approx(1.232));
// }

// TEST_CASE("EllCalc, test parallel cut (no effect)") {
//     auto ell_calc = EllCalc(4);
//     double rho;
//     double sigma;
//     double delta;
//     std::tie(rho, sigma, delta) = ell_calc.calc_parallel_cut(-0.04, 0.0625, 0.01);
//     CHECK_EQ(rho, doctest::Approx(0.0));
//     CHECK_EQ(sigma, doctest::Approx(0.0));
//     CHECK_EQ(delta, doctest::Approx(1.0));
// }
