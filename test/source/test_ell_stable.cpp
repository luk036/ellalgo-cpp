// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/ell_stable.hpp>
#include <vector>

TEST_CASE("EllStable, test constructor") {
    auto x = std::vector<double>{0.0, 0.0};
    auto ellip = EllStable<std::vector<double>>(10.0, x);
    CHECK_EQ(ellip.xc()[0], 0.0);
    CHECK_EQ(ellip.xc()[1], 0.0);
    CHECK_EQ(ellip.tsq(), 0.0);
}

TEST_CASE("EllStable, test update bias cut") {
    auto x = std::vector<double>{0.0, 0.0};
    auto ellip = EllStable<std::vector<double>>(10.0, x);
    auto grad = std::vector<double>{1.0, 1.0};
    auto beta = 0.0;
    auto cut = std::make_pair(grad, beta);
    auto status = ellip.update_bias_cut(cut);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_NE(ellip.xc()[0], 0.0);
    CHECK_NE(ellip.xc()[1], 0.0);
}

TEST_CASE("EllStable, test update central cut") {
    auto x = std::vector<double>{0.0, 0.0};
    auto ellip = EllStable<std::vector<double>>(10.0, x);
    auto grad = std::vector<double>{1.0, 1.0};
    auto beta = 0.0;
    auto cut = std::make_pair(grad, beta);
    auto status = ellip.update_central_cut(cut);
    CHECK_EQ(status, CutStatus::Success);
    CHECK_NE(ellip.xc()[0], 0.0);
    CHECK_NE(ellip.xc()[1], 0.0);
}
