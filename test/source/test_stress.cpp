#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for Ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <tuple>                      // for get, tuple

using Vec = std::valarray<double>;

struct MyStressOracle {
    using ArrayType = Vec;
    using CutChoice = double;  // single cut
    using Cut = std::pair<Vec, double>;

    mutable int idx = -1;  // for round robin
    const int num_constraints = 5;

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
                case 2: // constraint 3: x >= 0
                    if (-x > 0.0) {
                        return {{Vec{-1.0, 0.0}, -x}, false};
                    }
                    break;
                case 3: // constraint 4: y >= 0
                    if (-y > 0.0) {
                        return {{Vec{0.0, -1.0}, -y}, false};
                    }
                    break;
                case 4:  // objective: maximize x + y
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

TEST_CASE("Stress test") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyStressOracle{};
    auto gamma = -1.0e100;
    const auto options = Options{10000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    CHECK(x[0] >= 0.0);
    CHECK(x[1] >= 0.0);
    CHECK(x[0] + x[1] <= 3.000000001);
    CHECK(x[0] - x[1] >= 0.999999999);
}

struct MyQuadraticOracle {
    using ArrayType = Vec;
    using CutChoice = double;  // single cut
    using Cut = std::pair<Vec, double>;

    auto assess_optim(const Vec &x, double &gamma) const -> std::tuple<Cut, bool> {
        const auto x1 = x[0];
        const auto x2 = x[1];

        // Constraint 1: x1 + x2 <= 2
        double fj = x1 + x2 - 2.0;
        if (fj > 0.0) {
            return {{Vec{1.0, 1.0}, fj}, false};
        }

        // Constraint 2: x1 - x2 >= -1
        fj = -x1 + x2 - 1.0;
        if (fj > 0.0) {
            return {{Vec{-1.0, 1.0}, fj}, false};
        }

        // Objective: minimize -(x1^2 + x2^2)
        double f0 = -(x1*x1 + x2*x2);
        fj = f0 - gamma;
        if (fj > 0.0) {
            return {{Vec{-2.0*x1, -2.0*x2}, fj}, false};
        }

        gamma = f0;
        return {{Vec{-2.0*x1, -2.0*x2}, 0.0}, true};
    }
};

TEST_CASE("Stress test Quadratic") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyQuadraticOracle{};
    auto gamma = 1.0e100; // For minimization, gamma should be initialized to a large value
    const auto options = Options{20000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    CHECK(x[0] + x[1] <= 2.000000001);
    CHECK(x[0] - x[1] >= -1.000000001);
    CHECK(x[0] * x[0] + x[1] * x[1] <= 2.500000001); // Now checking for <= -2.5 (original maximization)
}