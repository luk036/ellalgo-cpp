// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>

#include <ellalgo/ell_matrix.hpp>           // for Matrix
#include <ellalgo/oracles/lmi0_oracle.hpp>  // for Lmi0Oracle
#include <valarray>
#include <vector>

namespace {

    using Vec = std::valarray<double>;
    using M_t = std::vector<Matrix>;

    /**
     * @brief Build F0 = I, F1 = diag(1, -1).
     *
     * A(x) = x0*F0 + x1*F1 = diag(x0 + x1, x0 - x1).
     * A(x) is SPD  <=>  x0 > |x1|.
     */
    auto make_diag_problem() -> M_t {
        auto f0 = Matrix(2);
        f0.row(0) = Vec{1.0, 0.0};
        f0.row(1) = Vec{0.0, 1.0};

        auto f1 = Matrix(2);
        f1.row(0) = Vec{1.0, 0.0};
        f1.row(1) = Vec{0.0, -1.0};

        return M_t{f0, f1};
    }

}  // namespace

TEST_CASE("Lmi0Oracle, feasible when A(x) is SPD") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    // x0 > |x1|  =>  diag(x0+x1, x0-x1) is positive definite
    CHECK(omega.assess_feas(Vec{2.0, 0.0}) == nullptr);
    CHECK(omega.assess_feas(Vec{3.0, -2.0}) == nullptr);
    CHECK(omega.assess_feas(Vec{0.5, 0.25}) == nullptr);
}

TEST_CASE("Lmi0Oracle, cut when A(x) is not SPD") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    // x0 <= |x1|  =>  not positive definite  =>  a cut is returned
    auto* cut1 = omega.assess_feas(Vec{2.0, 3.0});  // diag(5, -1)
    REQUIRE(cut1 != nullptr);
    CHECK_EQ(cut1->first.size(), 2U);
    CHECK(cut1->second > 0.0);  // deep cut (witness magnitude)

    // semidefinite boundary: diag(2, 0) is not definite
    auto* cut2 = omega.assess_feas(Vec{1.0, 1.0});
    REQUIRE(cut2 != nullptr);

    auto* cut3 = omega.assess_feas(Vec{-1.0, 0.0});  // diag(-1, -1)
    REQUIRE(cut3 != nullptr);
}

TEST_CASE("Lmi0Oracle, exact cut values") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    // x = (2, 3): A = diag(5, -1), witness w = (0, 1)
    //   g[0] = -w'F0 w = -1,  g[1] = -w'F1 w = +1,  beta = witness() = 1
    auto* cut = omega.assess_feas(Vec{2.0, 3.0});
    REQUIRE(cut != nullptr);
    CHECK_EQ(cut->first[0], -1.0);
    CHECK_EQ(cut->first[1], 1.0);
    CHECK_EQ(cut->second, 1.0);

    // x = (-1, 0): A = diag(-1, -1), witness w = (1, 0)
    //   g[0] = -1,  g[1] = -1,  beta = 1
    cut = omega.assess_feas(Vec{-1.0, 0.0});
    REQUIRE(cut != nullptr);
    CHECK_EQ(cut->first[0], -1.0);
    CHECK_EQ(cut->first[1], -1.0);
    CHECK_EQ(cut->second, 1.0);
}

TEST_CASE("Lmi0Oracle, cut separates the feasible region") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    // A valid cutting plane (g, beta) at infeasible x_i must satisfy
    //   g . (x_f - x_i) + beta <= 0   for every feasible x_f.
    const auto feasible = std::vector<Vec>{{2.0, 0.0}, {3.0, -2.0}, {0.5, 0.25}, {4.0, 1.0}};
    const auto infeasible = std::vector<Vec>{{2.0, 3.0}, {1.0, 1.0}, {-1.0, 0.0}, {-3.0, 1.0}};

    for (const auto& xi : infeasible) {
        auto* cut = omega.assess_feas(xi);
        REQUIRE(cut != nullptr);
        for (const auto& xf : feasible) {
            auto lhs = 0.0;
            for (auto k = 0U; k != xf.size(); ++k) {
                lhs += cut->first[k] * (xf[k] - xi[k]);
            }
            lhs += cut->second;
            CHECK(lhs <= 1e-12);
        }
    }
}

TEST_CASE("Lmi0Oracle, operator() delegates to assess_feas") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    const auto x = Vec{2.0, 3.0};  // infeasible
    auto* via_call = omega(x);
    auto* via_method = omega.assess_feas(x);
    REQUIRE(via_call != nullptr);
    CHECK(via_call == via_method);
    CHECK_EQ(via_call->first[0], via_method->first[0]);
    CHECK_EQ(via_call->first[1], via_method->first[1]);
    CHECK_EQ(via_call->second, via_method->second);
}

TEST_CASE("Lmi0Oracle, public _mq reflects feasibility state") {
    auto F = make_diag_problem();
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    CHECK(omega.assess_feas(Vec{2.0, 0.0}) == nullptr);  // SPD
    CHECK(omega._mq.is_spd());

    CHECK(omega.assess_feas(Vec{2.0, 3.0}) != nullptr);  // not SPD
    CHECK_FALSE(omega._mq.is_spd());
}

TEST_CASE("Lmi0Oracle, non-diagonal matrices") {
    // F0 = [[2, 1], [1, 2]],  F1 = I.
    // A(x) = x0*F0 + x1*I, eigenvalues {3x0 + x1, x0 + x1} => SPD
    //   <=>  x0 + x1 > 0 and 3x0 + x1 > 0.
    auto f0 = Matrix(2);
    f0.row(0) = Vec{2.0, 1.0};
    f0.row(1) = Vec{1.0, 2.0};

    auto f1 = Matrix(2);
    f1.row(0) = Vec{1.0, 0.0};
    f1.row(1) = Vec{0.0, 1.0};

    const auto F = M_t{f0, f1};
    Lmi0Oracle<Vec, Matrix> omega{2, F};

    // Feasible: 3*1 + 0 = 3 > 0 and 1 + 0 = 1 > 0
    CHECK(omega.assess_feas(Vec{1.0, 0.0}) == nullptr);
    // Feasible: 3*1 + 0.5 = 3.5 > 0, 1 + 0.5 = 1.5 > 0
    CHECK(omega.assess_feas(Vec{1.0, 0.5}) == nullptr);
    // Infeasible: 3*1 + (-2) = 1 > 0 but 1 + (-2) = -1 < 0
    CHECK(omega.assess_feas(Vec{1.0, -2.0}) != nullptr);
    // Infeasible: both eigenvalues negative
    CHECK(omega.assess_feas(Vec{0.0, -0.5}) != nullptr);
}
