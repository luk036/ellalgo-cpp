#include <doctest/doctest.h> // for ResultBuilder, TestCase, CHECK_EQ

#include <ellalgo/cutting_plane.hpp> // for cutting_plane_optim
#include <ellalgo/ell.hpp>           // for Ell
#include <ellalgo/ell_config.hpp>    // for CInfo, CutStatus, CutStatus::...

#include <tuple> // for get, tuple

// using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Vec = std::valarray<double>;

struct MyOracle {
    using ArrayType = Vec;
    using CutChoices = double; // single cut
    using Cut = std::pair<Vec, double>;

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] t
     * @return std::pair<Cut, double>
     */
    auto assess_optim(const Vec &z, double &t) -> std::tuple<Cut, bool> {
        const auto x = z[0];
        const auto y = z[1];

        // constraint 1: x + y <= 3
        const auto fj = x + y - 3.0;
        if (fj > 0.0) {
            return {{Vec{1.0, 1.0}, fj}, false};
        }
        // constraint 2: x - y >= 1
        const auto fj2 = -x + y + 1.0;
        if (fj2 > 0.0) {
            return {{Vec{-1.0, 1.0}, fj2}, false};
        }
        // objective: maximize x + y
        const auto f0 = x + y;
        const auto fj3 = t - f0;
        if (fj3 < 0.0) {
            t = f0;
            return {{Vec{-1.0, -1.0}, 0.0}, true};
        }
        return {{Vec{-1.0, -1.0}, fj3}, false};
    }
};

TEST_CASE("Example 1, test feasible") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle{};
    auto t = -1.0e100; // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-10};
    const auto result = cutting_plane_optim(oracle, ell, t, options);
    // const auto x = std::get<0>(result); // make clang compiler happy
    // CHECK(x[0] >= 0.0);
    const auto &x = std::get<0>(result);
    REQUIRE_NE(x.size(), 0U);
    // const auto &num_iters = std::get<1>(result);
    // REQUIRE(ell_info.feasible);
    CHECK(x[0] >= 0.0);
}

TEST_CASE("Example 1, test infeasible1") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0},
                        Vec{100.0, 100.0}); // wrong initial guess
                                            // or ellipsoid is too small
    auto oracle = MyOracle{};
    auto t = -1.0e100; // std::numeric_limits<double>::min()
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, t, options);
    const auto x = std::get<0>(result);
    // const auto s1 = std::get<2>(result);
    REQUIRE_EQ(x.size(), 0U);
    // CHECK_EQ(s1, CutStatus::NoSoln); // no sol'n
    // const auto &num_iters = std::get<1>(result);
    // CHECK(!ell_info.feasible);
    // CHECK_EQ(ell_info.status, CutStatus::NoSoln); // no sol'n
}

TEST_CASE("Example 1, test infeasible22") {
    auto ell = Ell<Vec>(Vec{10.0, 10.0}, Vec{0.0, 0.0});
    auto oracle = MyOracle{};
    auto t = 100.0;
    // wrong initial guess
    const auto options = Options{2000, 1e-12};
    const auto result = cutting_plane_optim(oracle, ell, t, options);
    const auto x = std::get<0>(result);
    // const auto s1 = std::get<2>(result);
    REQUIRE_EQ(x.size(), 0U);
    // CHECK_EQ(s1, CutStatus::NoSoln); // no sol'n
    // const auto &num_iters = std::get<1>(result);
    // CHECK(!ell_info.feasible);
}
