#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for Ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <tuple>                      // for get, tuple

using Vec = std::valarray<double>;

struct MyOracle {
    using ArrayType = Vec;
    using CutChoices = double;  // single cut
    using Cut = std::pair<Vec, double>;

    int idx = 0;

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] gamma
     * @return std::pair<Cut, double>
     */
    auto assess_optim(const Vec &z, double &gamma) -> std::tuple<Cut, bool> {
        const auto x = z[0];
        const auto y = z[1];
        const auto f0 = x + y;

        for (int i = 0; i < 3; i++) {
            this->idx++;
            if (this->idx == 3) {
                this->idx = 0;  // round robin
            }
            double fj = 0.0;
            switch (this->idx) {
                case 0:  // constraint 1: x + y <= 3
                    fj = f0 - 3.0;
                    break;
                case 1:  // constraint 2: x - y >= 1
                    fj = -x + y + 1.0;
                    break;
                case 2:  // objective: maximize x + y
                    fj = gamma - f0;
                    break;
                default:
                    exit(0);
            }
            if (fj > 0.0) {
                switch (this->idx) {
                    case 0:
                        return {{Vec{1.0, 1.0}, fj}, false};
                    case 1:
                        return {{Vec{-1.0, 1.0}, fj}, false};
                    case 2:
                        return {{Vec{-1.0, -1.0}, fj}, false};
                    default:
                        exit(0);
                }
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
    REQUIRE_EQ(x.size(), 0U);
}

TEST_CASE("Example 1, test infeasible22") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle{};
    auto gamma = 100.0;
    // wrong initial guess
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, gamma, options);
    const auto x = std::get<0>(result);
    REQUIRE_EQ(x.size(), 0U);
}
