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
