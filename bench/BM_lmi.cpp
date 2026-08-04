/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <ellalgo/cutting_plane.hpp>           // for cutting_plane_optim
#include <ellalgo/ell.hpp>                     // for Ell
#include <ellalgo/ell_matrix.hpp>              // for Matrix
#include <ellalgo/oracles/lmi_old_oracle.hpp>  // for LmiOldOracle
#include <ellalgo/oracles/lmi_oracle.hpp>      // for LmiOracle
#include <tuple>                               // for tuple
#include <type_traits>                         // for move
#include <vector>                              // for vector

namespace {
    template <typename O, typename V>
    concept LmiCallable = requires(O& o, const V& x) {
        { o(x) } -> std::convertible_to<bool>;
    };
}  // namespace

template <typename Oracle>
    requires LmiCallable<Oracle, std::valarray<double>>
class MyOracle {
    using Vec = std::valarray<double>;
    using Cut = std::pair<Vec, double>;

    Oracle lmi1;
    Oracle lmi2;
    const Vec c;

  public:
    MyOracle(size_t m1, const std::vector<Matrix>& F1, const Matrix& B1, size_t m2,
             const std::vector<Matrix>& F2, const Matrix& B2, Vec c)
        : lmi1{m1, F1, B1}, lmi2{m2, F2, B2}, c{std::move(c)} {}

    std::tuple<Cut, bool> assess_optim(const Vec& x, double& gamma) {
        const auto f0 = (this->c * x).sum();
        const auto f1 = f0 - gamma;
        if (f1 > 0.0) {
            return {{this->c, f1}, false};
        }
        if (const auto cut1 = this->lmi1(x)) {
            return {*cut1, false};
        }
        if (const auto cut2 = this->lmi2(x)) {
            return {*cut2, false};
        }
        gamma = f0;
        return {{this->c, 0.0}, true};
    }

    std::tuple<Cut, bool> operator()(const Vec& x, double& gamma) {
        return this->assess_optim(x, gamma);
    }
};

int main() {
    using Vec = std::valarray<double>;
    using M_t = std::vector<Matrix>;

    // Common problem data
    auto c = Vec{1.0, -1.0, 1.0};

    auto m0F1 = Matrix(2);
    m0F1.row(0) = Vec{-7.0, -11.0};
    m0F1.row(1) = Vec{-11.0, 3.0};

    auto m1F1 = Matrix(2);
    m1F1.row(0) = Vec{7.0, -18.0};
    m1F1.row(1) = Vec{-18.0, 8.0};

    auto m2F1 = Matrix(2);
    m2F1.row(0) = Vec{-2.0, -8.0};
    m2F1.row(1) = Vec{-8.0, 1.0};

    auto F1 = M_t{m0F1, m1F1, m2F1};

    auto B1 = Matrix(2);
    B1.row(0) = Vec{33.0, -9.0};
    B1.row(1) = Vec{-9.0, 26.0};

    auto m0F2 = Matrix(3);
    m0F2.row(0) = Vec{-21.0, -11.0, 0.0};
    m0F2.row(1) = Vec{-11.0, 10.0, 8.0};
    m0F2.row(2) = Vec{0.0, 8.0, 5.0};

    auto m1F2 = Matrix(3);
    m1F2.row(0) = Vec{0.0, 10.0, 16.0};
    m1F2.row(1) = Vec{10.0, -10.0, -10.0};
    m1F2.row(2) = Vec{16.0, -10.0, 3.0};

    auto m2F2 = Matrix(3);
    m2F2.row(0) = Vec{-5.0, 2.0, -17.0};
    m2F2.row(1) = Vec{2.0, -6.0, 8.0};
    m2F2.row(2) = Vec{-17.0, 8.0, 6.0};

    auto F2 = M_t{m0F2, m1F2, m2F2};

    auto B2 = Matrix(3);
    B2.row(0) = Vec{14.0, 9.0, 40.0};
    B2.row(1) = Vec{9.0, 91.0, 10.0};
    B2.row(2) = Vec{40.0, 10.0, 15.0};

    ankerl::nanobench::Bench bench;
    bench.title("LMI benchmarks").unit("op").warmup(100).epochs(50);

    bench.run("LMI_Lazy", [&] {
        auto omega = MyOracle<LmiOracle<Vec, Matrix>>(2, F1, B1, 3, F2, B2, Vec{1.0, -1.0, 1.0});
        auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});
        auto gamma = 1e100;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("LMI_old", [&] {
        auto omega = MyOracle<LmiOldOracle<Vec, Matrix>>(2, F1, B1, 3, F2, B2, Vec{1.0, -1.0, 1.0});
        auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});
        auto gamma = 1e100;
        auto result = cutting_plane_optim(omega, ellip, gamma);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
