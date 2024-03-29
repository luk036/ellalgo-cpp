#include "ellalgo/ell_config.hpp"
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/ell_core.hpp>  // for EllCore
#include <tuple>                 // for get, tuple

using Vec = std::valarray<double>;

TEST_CASE("EllCore, test central cut") {
    auto ell_core = EllCore(0.01, 4);
    CHECK(!ell_core.no_defer_trick);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_central_cut(grad, 0.0);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(ell_core.tsq(), 0.01);
    CHECK_EQ(grad[0], 0.01);
}

TEST_CASE("EllCore, test deep cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_deep_cut(grad, 0.05);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(ell_core.tsq(), 0.01);
    CHECK_EQ(grad[0], doctest::Approx(0.03));
}

// TEST_CASE("EllCore, test deep cut") {
//     auto ell_core = EllCore(4);
//     double rho;
//     double sigma;
//     double delta;
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_core.calc_deep_cut(0.11, 0.01);
//     CHECK_EQ(status, CutStatus::NoSoln);
//     std::tie(status, result) = ell_core.calc_deep_cut(0.01, 0.01);
//     CHECK_EQ(status, CutStatus::Success);

//     std::tie(status, result) = ell_core.calc_deep_cut(0.05, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.06));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(0.8));
// }

// TEST_CASE("EllCore, test parallel central cut") {
//     auto ell_core = EllCore(4);
//     double rho;
//     double sigma;
//     double delta;
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_core.calc_parallel_central_cut(0.05, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.02));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(1.2));
// }

// TEST_CASE("EllCore, test parallel deep cut") {
//     auto ell_core = EllCore(4);
//     double rho;
//     double sigma;
//     double delta;
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.07, 0.05, 0.01);
//     CHECK_EQ(status, CutStatus::NoSoln);

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.0, 0.05, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.02));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(1.2));

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.05, 0.11, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.06));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(0.8));

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.01, 0.04, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.0232));
//     CHECK_EQ(sigma, doctest::Approx(0.928));
//     CHECK_EQ(delta, doctest::Approx(1.232));
// }

// // TEST_CASE("EllCore, test parallel deep cut (no effect)") {
// //     auto ell_core = EllCore(4);
// //     CutStatus status;
// //     std::tuple<double, double, double> result;
// //     std::tie(status, result) = ell_core.calc_parallel_deep_cut_q(-0.04, 0.0625, 0.01);
// //     CHECK_EQ(status, CutStatus::NoEffect);
// // }

// TEST_CASE("EllCore, test deep cut q") {
//     auto ell_core = EllCore(4);
//     double rho;
//     double sigma;
//     double delta;
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_core.calc_deep_cut_q(0.11, 0.01);
//     CHECK_EQ(status, CutStatus::NoSoln);
//     std::tie(status, result) = ell_core.calc_deep_cut_q(0.01, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(status, result) = ell_core.calc_deep_cut_q(-0.05, 0.01);
//     CHECK_EQ(status, CutStatus::NoEffect);

//     std::tie(status, result) = ell_core.calc_deep_cut_q(0.05, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.06));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(0.8));
// }

// TEST_CASE("EllCore, test parallel deep cut q") {
//     auto ell_core = EllCore(4);
//     double rho;
//     double sigma;
//     double delta;
//     CutStatus status;
//     std::tuple<double, double, double> result;
//     std::tie(status, result) = ell_core.calc_parallel_deep_cut_q(0.07, 0.03, 0.01);
//     CHECK_EQ(status, CutStatus::NoSoln);
//     std::tie(status, result) = ell_core.calc_parallel_deep_cut_q(-0.04, 0.0625, 0.01);
//     CHECK_EQ(status, CutStatus::NoEffect);

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.0, 0.05, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.02));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(1.2));

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.05, 0.11, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.06));
//     CHECK_EQ(sigma, doctest::Approx(0.8));
//     CHECK_EQ(delta, doctest::Approx(0.8));

//     std::tie(status, result) = ell_core.calc_parallel_deep_cut(0.01, 0.04, 0.01);
//     CHECK_EQ(status, CutStatus::Success);
//     std::tie(rho, sigma, delta) = result;
//     CHECK_EQ(rho, doctest::Approx(0.0232));
//     CHECK_EQ(sigma, doctest::Approx(0.928));
//     CHECK_EQ(delta, doctest::Approx(1.232));
// }

// // TEST_CASE("EllCore, test bias cut") {
// //     auto ell_core = EllCore(4);
// //     double rho;
// //     double sigma;
// //     double delta;
// //     std::tie(rho, sigma, delta) = ell_core.calc_bias_cut(0.05, 0.1);
// //     CHECK_EQ(rho, doctest::Approx(0.06));
// //     CHECK_EQ(sigma, doctest::Approx(0.8));
// //     CHECK_EQ(delta, doctest::Approx(0.8));
// // }

// // TEST_CASE("EllCore, test parallel central cut") {
// //     auto ell_core = EllCore(4);
// //     double rho;
// //     double sigma;
// //     double delta;
// //     std::tie(rho, sigma, delta) = ell_core.calc_parallel_central_cut(1.0, 4.0);
// //     CHECK_EQ(rho, doctest::Approx(0.4));
// //     CHECK_EQ(sigma, doctest::Approx(0.8));
// //     CHECK_EQ(delta, doctest::Approx(1.2));
// // }

// // TEST_CASE("EllCore, test parallel cut") {
// //     auto ell_core = EllCore(4);
// //     double rho;
// //     double sigma;
// //     double delta;
// //     std::tie(rho, sigma, delta) = ell_core.calc_parallel_cut(0.01, 0.04, 0.01);
// //     CHECK_EQ(rho, doctest::Approx(0.0232));
// //     CHECK_EQ(sigma, doctest::Approx(0.928));
// //     CHECK_EQ(delta, doctest::Approx(1.232));
// // }

// // TEST_CASE("EllCore, test parallel cut (no effect)") {
// //     auto ell_core = EllCore(4);
// //     double rho;
// //     double sigma;
// //     double delta;
// //     std::tie(rho, sigma, delta) = ell_core.calc_parallel_cut(-0.04, 0.0625, 0.01);
// //     CHECK_EQ(rho, doctest::Approx(0.0));
// //     CHECK_EQ(sigma, doctest::Approx(0.0));
// //     CHECK_EQ(delta, doctest::Approx(1.0));
// // }
