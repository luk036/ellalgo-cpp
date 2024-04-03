#include "ellalgo/ell_config.hpp"
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/ell_core.hpp>  // for EllCore

using Vec = std::valarray<double>;

TEST_CASE("EllCore, test central cut") {
    auto ell_core = EllCore(0.01, 4);
    CHECK(!ell_core.no_defer_trick);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_central_cut(grad, 0.0);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], 0.01);
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore, test deep cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_bias_cut(grad, 0.05);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.03));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore, test parallel central cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_central_cut(grad, Vec{0.0, 0.05});
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.01));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore, test parallel cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_bias_cut(grad, Vec{0.01, 0.04});
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.0116));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore, test parallel cut (no effect)") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_bias_cut(grad, Vec{-0.04, 0.0625});
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.0));
    CHECK_EQ(ell_core.tsq(), 0.01);
}


TEST_CASE("EllCore, test parallel cut (no effect)") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_q(grad, Vec{-0.04, 0.0625});
    CHECK_EQ(status, CutStatus::NoEffect);
    CHECK_EQ(grad[0], doctest::Approx(0.5));
    CHECK_EQ(ell_core.tsq(), 0.01);
}


// Stable version
TEST_CASE("EllCore (stable), test central cut") {
    auto ell_core = EllCore(0.01, 4);
    CHECK(!ell_core.no_defer_trick);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_stable_central_cut(grad, 0.0);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], 0.01);
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore (stable), test deep cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_stable_bias_cut(grad, 0.05);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.03));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore (stable), test parallel central cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_stable_central_cut(grad, Vec{0.0, 0.05});
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.01));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore (stable), test parallel cut") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_stable_bias_cut(grad, Vec{0.01, 0.04});
    CHECK_EQ(status, CutStatus::Success);
    CHECK_EQ(grad[0], doctest::Approx(0.0116));
    CHECK_EQ(ell_core.tsq(), 0.01);
}

TEST_CASE("EllCore (stable), test parallel cut (no effect)") {
    auto ell_core = EllCore(0.01, 4);
    auto grad = Vec(0.5, 4);
    auto status = ell_core.update_stable_q(grad, Vec{-0.04, 0.0625});
    CHECK_EQ(status, CutStatus::NoEffect);
    CHECK_EQ(grad[0], doctest::Approx(0.5));
    CHECK_EQ(ell_core.tsq(), 0.01);
}
