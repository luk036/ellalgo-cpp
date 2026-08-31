// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/ell_matrix.hpp>           // for Matrix
#include <ellalgo/lmi_problem.hpp>          // for LMIProblem, make_lmi_problem
#include <ellalgo/oracles/lmi_factory.hpp>  // for make_lmi_oracle
#include <valarray>
#include <vector>

namespace {
    auto make_sample_problem() -> std::pair<std::vector<Matrix>, Matrix> {
        using Vec = std::valarray<double>;

        auto m0F = Matrix(2);
        m0F.row(0) = Vec{-7.0, -11.0};
        m0F.row(1) = Vec{-11.0, 3.0};

        auto m1F = Matrix(2);
        m1F.row(0) = Vec{7.0, -18.0};
        m1F.row(1) = Vec{-18.0, 8.0};

        auto m2F = Matrix(2);
        m2F.row(0) = Vec{-2.0, -8.0};
        m2F.row(1) = Vec{-8.0, 1.0};

        auto B = Matrix(2);
        B.row(0) = Vec{33.0, -9.0};
        B.row(1) = Vec{-9.0, 26.0};

        return {std::vector<Matrix>{m0F, m1F, m2F}, std::move(B)};
    }
}  // namespace

TEST_CASE("make_lmi_oracle factory produces a working oracle") {
    using Vec = std::valarray<double>;
    auto [F, B] = make_sample_problem();

    auto omega = make_lmi_oracle<Vec, Matrix>(2, F, std::move(B));
    auto cut = omega.assess_feas(Vec{0.0, 0.0, 0.0});
    CHECK(cut == nullptr);  // origin is feasible (B is positive definite)
}

TEST_CASE("LMIProblem facade solves an LMI feasibility problem") {
    using Vec = std::valarray<double>;
    auto [F, B] = make_sample_problem();

    auto problem = make_lmi_problem<Vec, Matrix>(2, std::move(F), std::move(B));
    const auto result = problem.solve_feas(Vec{10.0, 10.0, 10.0}, Vec{0.0, 0.0, 0.0});
    const auto& x = std::get<0>(result);
    const auto& num_iters = std::get<1>(result);

    CHECK_NE(x.size(), 0U);
    CHECK(num_iters < 2000);
}

TEST_CASE("LMIProblem facade, alpha-scaled initial space") {
    using Vec = std::valarray<double>;
    auto [F, B] = make_sample_problem();

    auto problem = LMIProblem<Vec, Matrix>(2, std::move(F), std::move(B));
    const auto result = problem.solve_feas(10.0, Vec{0.0, 0.0, 0.0});
    const auto& x = std::get<0>(result);
    const auto& num_iters = std::get<1>(result);

    CHECK_NE(x.size(), 0U);
    CHECK(num_iters < 2000);
}
