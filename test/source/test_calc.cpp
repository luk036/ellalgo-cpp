#include "ellalgo/ell_config.hpp"
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/ell_calc.hpp>

TEST_CASE("EllCalc, test central cut") {
    auto ell_calc = EllCalc(4);
    // CutStatus status;
    // std::tuple<double, double, double> result;
    auto r = ell_calc.calc_central_cut(0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.02));
    CHECK_EQ(r.sigma, doctest::Approx(0.4));
    CHECK_EQ(r.delta, doctest::Approx(16.0 / 15.0));
}

TEST_CASE("EllCalc, test deep cut") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_bias_cut(0.11, 0.01);
    CHECK_EQ(r.status, CutStatus::NoSoln);
    r = ell_calc.calc_bias_cut(0.01, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);

    r = ell_calc.calc_bias_cut(0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.06));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(0.8));
}

TEST_CASE("EllCalc, test parallel central cut") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_central_cut(0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.02));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(1.2));
}

TEST_CASE("EllCalc, test parallel deep cut") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_bias_cut(0.07, 0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::NoSoln);

    r = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.02));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(1.2));

    r = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.06));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(0.8));

    r = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.0232));
    CHECK_EQ(r.sigma, doctest::Approx(0.928));
    CHECK_EQ(r.delta, doctest::Approx(1.232));
}

TEST_CASE("EllCalc, test parallel deep cut (no effect)") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_bias_cut_q(-0.04, 0.0625, 0.01);
    CHECK_EQ(r.status, CutStatus::NoEffect);
}

TEST_CASE("EllCalc, test deep cut q") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_bias_cut_q(0.11, 0.01);
    CHECK_EQ(r.status, CutStatus::NoSoln);
    r = ell_calc.calc_bias_cut_q(0.01, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    r = ell_calc.calc_bias_cut_q(-0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::NoEffect);

    r = ell_calc.calc_bias_cut_q(0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.06));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(0.8));
}

TEST_CASE("EllCalc, test parallel deep cut q") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_bias_cut_q(0.07, 0.03, 0.01);
    CHECK_EQ(r.status, CutStatus::NoSoln);
    r = ell_calc.calc_parallel_bias_cut_q(-0.04, 0.0625, 0.01);
    CHECK_EQ(r.status, CutStatus::NoEffect);

    r = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.02));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(1.2));

    r = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.06));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(0.8));

    r = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.0232));
    CHECK_EQ(r.sigma, doctest::Approx(0.928));
    CHECK_EQ(r.delta, doctest::Approx(1.232));
}

TEST_CASE("EllCalc, test bias cut") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_bias_cut(0.05, 0.1);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.103246));
    CHECK_EQ(r.sigma, doctest::Approx(0.563832));
    CHECK_EQ(r.delta, doctest::Approx(1.04));
}

TEST_CASE("EllCalc, test parallel central cut v1") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_central_cut(1.0, 4.0);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.4));
    CHECK_EQ(r.sigma, doctest::Approx(0.8));
    CHECK_EQ(r.delta, doctest::Approx(1.2));
}

TEST_CASE("EllCalc, test parallel cut v1") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    CHECK_EQ(r.status, CutStatus::Success);
    CHECK_EQ(r.rho, doctest::Approx(0.0232));
    CHECK_EQ(r.sigma, doctest::Approx(0.928));
    CHECK_EQ(r.delta, doctest::Approx(1.232));
}

TEST_CASE("EllCalc, test parallel cut (no effect) v2") {
    auto ell_calc = EllCalc(4);
    auto r = ell_calc.calc_parallel_bias_cut_q(-0.04, 0.0625, 0.01);
    CHECK_EQ(r.status, CutStatus::NoEffect);
    CHECK_EQ(r.rho, doctest::Approx(0.0));
    CHECK_EQ(r.sigma, doctest::Approx(0.0));
    CHECK_EQ(r.delta, doctest::Approx(1.0));
}
