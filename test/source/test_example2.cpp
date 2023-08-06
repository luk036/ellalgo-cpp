#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <utility>                    // for pair

// using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Vec = std::valarray<double>;

struct MyOracle {
    using ArrayType = Vec;
    using CutChoices = double;
    using Cut = std::pair<Vec, double>;

    /**
     * @brief
     *
     * @param[in] z
     * @return std::optional<Cut>
     */
    auto assess_feas(const Vec &z) -> Cut * {
        static auto cut1 = Cut{Vec{1.0, 1.0}, 0.0};
        static auto cut2 = Cut{Vec{-1.0, 1.0}, 0.0};

        const auto x = z[0];
        const auto y = z[1];

        // constraint 1: x + y <= 3
        const auto fj = x + y - 3.0;
        if (fj > 0.0) {
            cut1.second = fj;
            return &cut1;
        }
        // constraint 2: x - y >= 1
        const auto fj2 = -x + y + 1.0;
        if (fj2 > 0.0) {
            cut2.second = fj2;
            return &cut2;
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
    CHECK(x_feas.size() != 0U);
}
