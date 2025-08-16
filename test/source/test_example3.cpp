#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>  // for cutting_plane_optim
#include <ellalgo/ell.hpp>            // for ell
#include <ellalgo/ell_config.hpp>     // for CInfo, CutStatus, CutStatus::...
#include <utility>                    // for pair

using Vec = std::valarray<double>;

struct MyOracle3 {
    using ArrayType = Vec;
    using CutChoice = double;
    using Cut = std::pair<Vec, double>;

    int idx = -1;
    double target = -1e100;

    void update(double gamma) { this->target = gamma; }

    /**
     * The assess_feas function assesses the feasibility of a given point based on predefined
     * constraints and returns a pointer to the corresponding cut if violated.
     *
     * @param[in] xc The parameter `xc` is of type `Vec`, which seems to be a vector containing two
     * elements representing coordinates in a 2D space. In the function `assess_feas`, the elements
     * of `xc` are accessed as `x` and `y` using `xc[0]
     *
     * @return A pointer to a `Cut` object is being returned from the `assess_feas` function. The
     * function returns a pointer to one of the static `Cut` objects (`cut1`, `cut2`, `cut3`,
     * `cut4`) based on the constraints being assessed.
     */
    auto assess_feas(const Vec &xc) -> Cut * {
        static auto cut1 = Cut{Vec{-1.0, 0.0}, 0.0};
        static auto cut2 = Cut{Vec{0.0, -1.0}, 0.0};
        static auto cut3 = Cut{Vec{1.0, 1.0}, 0.0};
        static auto cut4 = Cut{Vec{2.0, -3.0}, 0.0};

        const auto x = xc[0];
        const auto y = xc[1];

        for (int i = 0; i != 4; ++i) {
            this->idx++;
            if (this->idx == 4) {
                this->idx = 0;  // round robin
            }
            double fj;
            switch (this->idx) {
                case 0:  // constraint 1: x >= -1
                    fj = -x - 1.0;
                    break;
                case 1:  // constraint 2: y >= -2
                    fj = -y + 2.0;
                    break;
                case 2:  // constraint 3: x + y <= 1
                    fj = x + y - 1.0;
                    break;
                case 3:  // constraint 3: x + y <= 1
                    fj = 2.0 * x + 3.0 * y - this->target;
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
                    case 2:
                        cut3.second = fj;
                        return &cut3;
                    case 3:
                        cut4.second = fj;
                        return &cut4;
                    default:
                        exit(0);
                }
            }
        }
        return nullptr;
    }
};

TEST_CASE("Example 2, test feasible") {
    auto ellip = Ell<Vec>(Vec{100.0, 100.0}, Vec{0.0, 0.0});
    auto omega = MyOracle3{};
    const auto options = Options{2000, 1e-8};
    BSearchAdaptor<MyOracle3, Ell<Vec> > adaptor(omega, ellip, options);
    auto intvl = std::pair<double, double>{-100.0, 100.0};
    const auto result = bsearch(adaptor, intvl, options);
    const auto num_iters = std::get<1>(result);
    CHECK_EQ(num_iters, 34);
}

TEST_CASE("Example 3, test feasible2") {
    auto ellip = Ell<Vec>(Vec{100.0, 100.0}, Vec{0.0, 0.0});
    auto omega = MyOracle3{};
    const auto options = Options{2000, 1e-8};
    BSearchAdaptor<MyOracle3, Ell<Vec> > adaptor(omega, ellip, options);
    auto intvl = std::pair<double, double>{-200.0, 200.0};
    const auto result = bsearch(adaptor, intvl, options);
    const auto num_iters = std::get<1>(result);
    CHECK_EQ(num_iters, 35);
}
