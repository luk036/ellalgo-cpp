#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for Ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <tuple>                      // for get, tuple

using Vec = std::valarray<double>;

struct MyOracle {
    using ArrayType = Vec;
    using CutChoice = double;  // single cut
    using Cut = std::pair<Vec, double>;

    mutable int idx = -1;  // for round robin
    const int num_constraints = 3;

    /**
     * The function assess_optim assesses the optimality of a given point in a mathematical
     * optimization problem with constraints and an objective function.
     *
     * @param[in] xc The parameter `xc` is a vector containing two elements, representing the
     * coordinates of a point in a 2D space. The first element corresponds to the x-coordinate, and
     * the second element corresponds to the y-coordinate.
     * @param[in,out] gamma The `gamma` parameter in the `assess_optim` function is a reference to a
     * `double` type variable. It is passed as an input-output parameter, meaning its value can be
     * modified within the function and the updated value will be reflected outside the function
     * scope as well.
     *
     * @return The function `assess_optim` returns a tuple containing a `Cut` object and a boolean
     * value. The `Cut` object represents a cut in a mathematical optimization context, and the
     * boolean value indicates whether the optimization assessment was successful.
     */
    auto assess_optim(const Vec &xc, double &gamma) const -> std::tuple<Cut, bool> {
        const auto x = xc[0];
        const auto y = xc[1];
        const auto f0 = x + y;

        for (int i = 0; i != this->num_constraints; i++) {
            this->idx++;
            if (this->idx == this->num_constraints) {
                this->idx = 0;  // round robin
            }
            double fj;
            switch (this->idx) {
                case 0:  // constraint 1: x + y <= 3
                    if (f0 > 3.0) {
                        return {{Vec{1.0, 1.0}, f0 - 3.0}, false};
                    }
                    break;
                case 1:  // constraint 2: x - y >= 1
                    if ((fj = -x + y + 1.0) > 0.0) {
                        return {{Vec{-1.0, 1.0}, fj}, false};
                    }
                    break;
                case 2:  // objective: maximize x + y
                    if ((fj = gamma - f0) > 0.0) {
                        return {{Vec{-1.0, -1.0}, fj}, false};
                    };
                    break;
                default:
                    exit(0);
            }
        }
        gamma = f0;
        return {{Vec{-1.0, -1.0}, 0.0}, true};
    }
};

TEST_CASE("Example 1, test feasible") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle{};
    auto gamma = -1.0e100;  // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-10};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    CHECK(x[0] >= 0.0);
}

TEST_CASE("Example 1, test infeasible1") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{100.0, 100.0});  // wrong initial guess
                                                              // or ellipsoid is too small
    auto oracle = MyOracle{};
    auto gamma = -1.0e100;  // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto x = std::get<0>(result);
    CHECK_EQ(x.size(), 0U);
}

TEST_CASE("Example 1, test infeasible2") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle{};
    auto gamma = 100.0;  // wrong initial best-so-far value
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto x = std::get<0>(result);
    CHECK_EQ(x.size(), 0U);
}

TEST_CASE("Example 1, test feasible2") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{1.0, 1.0});
    auto oracle = MyOracle{};
    auto gamma = -1.0e100;  // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-10};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    CHECK(x[0] >= 0.0);
}

struct MyOracle2 {
    using ArrayType = Vec;
    using CutChoice = double;  // single cut
    using Cut = std::pair<Vec, double>;

    mutable int idx = -1;  // for round robin
    const int num_constraints = 3;

    auto assess_optim(const Vec &xc, double &gamma) const -> std::tuple<Cut, bool> {
        const auto x = xc[0];
        const auto y = xc[1];
        const auto f0 = x + y;

        for (int i = 0; i != this->num_constraints; i++) {
            this->idx++;
            if (this->idx == this->num_constraints) {
                this->idx = 0;  // round robin
            }
            double fj;
            switch (this->idx) {
                case 0:  // constraint 1: x + y <= 3
                    if (f0 > 3.0) {
                        return {{Vec{1.0, 1.0}, f0 - 3.0}, false};
                    }
                    break;
                case 1:  // constraint 2: x - y >= 1
                    if ((fj = -x + y + 1.0) > 0.0) {
                        return {{Vec{-1.0, 1.0}, fj}, false};
                    }
                    break;
                case 2:  // objective: maximize x - y
                    if ((fj = gamma - (x - y)) > 0.0) {
                        return {{Vec{-1.0, 1.0}, fj}, false};
                    };
                    break;
                default:
                    exit(0);
            }
        }
        gamma = x - y;
        return {{Vec{-1.0, 1.0}, 0.0}, true};
    }
};

TEST_CASE("Example 1, test objective") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle2{};
    auto gamma = -1.0e100;  // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-10};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    CHECK(x[0] >= 0.0);
}
