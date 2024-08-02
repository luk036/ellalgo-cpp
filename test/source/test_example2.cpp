#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <utility>                    // for pair

using Vec = std::valarray<double>;

struct MyOracle {
    using ArrayType = Vec;
    using CutChoices = double;
    using Cut = std::pair<Vec, double>;

    int idx = -1;

    /**
     * The function assess_feas assesses the feasibility of a given vector xc with respect to two
     * constraints and returns a pointer to a Cut object representing the violated constraint.
     *
     * @param[in] xc The parameter `xc` is of type `Vec`, which seems to be a vector or array-like
     * structure containing two elements representing coordinates in a 2D space. In this function
     * `assess_feas`, the elements of `xc` are accessed as `x` and `y` respectively.
     *
     * @return A pointer to a `Cut` object is being returned. The `Cut` object being returned is
     * either `cut1` or `cut2`, which are static variables initialized within the function.
     */
    auto assess_feas(const Vec &xc) -> Cut * {
        static auto cut1 = Cut{Vec{1.0, 1.0}, 0.0};
        static auto cut2 = Cut{Vec{-1.0, 1.0}, 0.0};

        const auto x = xc[0];
        const auto y = xc[1];

        for (int i = 0; i != 2; ++i) {
            this->idx++;
            if (this->idx == 2) {
                this->idx = 0;  // round robin
            }
            double fj;
            switch (this->idx) {
                case 0:  // constraint 1: x + y <= 3
                    fj = x + y - 3.0;
                    break;
                case 1:  // constraint 2: x - y >= 1
                    fj = -x + y + 1.0;
                    break;
                default:
                    exit(0);
            }
            if (fj > 0.0) {
                switch (this->idx) {
                    case 0:
                        cut1.second = fj;
                        return &cut1;
                    case 1:
                        cut2.second = fj;
                        return &cut2;
                    default:
                        exit(0);
                }
            }
        }
        return nullptr;
    }
};

TEST_CASE("Example 2, test feasible") {
    auto ellip = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto omega = MyOracle{};
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_feas(omega, ellip, options);
    const auto x_feas = std::get<0>(result);
    CHECK_NE(x_feas.size(), 0U);
}

TEST_CASE("Example 2, test infeasible") {
    auto ellip = Ell<Vec>(Vec{10.0, 10.0}, Vec{100.0, 100.0});
    auto omega = MyOracle{};
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_feas(omega, ellip, options);
    const auto x_feas = std::get<0>(result);
    CHECK_EQ(x_feas.size(), 0U);
}
