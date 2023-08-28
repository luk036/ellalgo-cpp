#include <ellalgo/cutting_plane.hpp>           // for cutting_plane_optim
#include <ellalgo/ell.hpp>                     // for Ell
#include <ellalgo/ell_matrix.hpp>              // for Matrix
#include <ellalgo/oracles/lmi_old_oracle.hpp>  // for LmiOldOracle
#include <ellalgo/oracles/lmi_oracle.hpp>      // for LmiOracle
#include <tuple>                               // for tuple
#include <type_traits>                         // for move
#include <vector>                              // for vector

#include "benchmark/benchmark.h"  // for BENCHMARK, State, BENCHMARK_...

/**
 * @brief
 *
 * @tparam Oracle
 */
template <typename Oracle> class MyOracle {
    using Vec = std::valarray<double>;
    using Cut = std::pair<Vec, double>;

  private:
    Oracle lmi1;
    Oracle lmi2;
    const Vec c;

  public:
    /**
     * @brief Construct a new my oracle object
     *
     * @param[in] F1
     * @param[in] B1
     * @param[in] F2
     * @param[in] B2
     * @param[in] c
     */
    MyOracle(size_t m1, const std::vector<Matrix> &F1, const Matrix &B1, size_t m2,
             const std::vector<Matrix> &F2, const Matrix &B2, Vec c)
        : lmi1{m1, F1, B1}, lmi2{m2, F2, B2}, c{std::move(c)} {}

    /**
     * @brief
     *
     * @param[in] x
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    std::tuple<Cut, bool> assess_optim(const Vec &x, double &t) {
        const auto f0 = (this->c * x).sum();
        const auto f1 = f0 - t;
        if (f1 > 0.0) {
            return {{this->c, f1}, false};
        }
        if (const auto cut1 = this->lmi1(x)) {
            return {*cut1, false};
        }
        if (const auto cut2 = this->lmi2(x)) {
            return {*cut2, false};
        }
        t = f0;
        return {{this->c, 0.0}, true};
    }

    /**
     * @brief
     *
     * @param[in] x
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    std::tuple<Cut, bool> operator()(const Vec &x, double &t) { return this->assess_optim(x, t); }
};

/**
 * @brief
 *
 * @param[in,out] state
 */
static void LMI_Lazy(benchmark::State &state) {
    using Vec = std::valarray<double>;
    using M_t = std::vector<Matrix>;

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

    while (state.KeepRunning()) {
        auto omega = MyOracle<LmiOracle<Vec, Matrix>>(2, F1, B1, 3, F2, B2, Vec{1.0, -1.0, 1.0});
        auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});
        auto t = 1e100;  // std::numeric_limits<double>::max()
        auto result = cutting_plane_optim(omega, ellip, t);
        benchmark::DoNotOptimize(result);
    }
}

// Register the function as a benchmark
BENCHMARK(LMI_Lazy);

//~~~~~~~~~~~~~~~~

/**
 * @brief Define another benchmark
 *
 * @param[in,out] state
 */
static void LMI_old(benchmark::State &state) {
    using Vec = std::valarray<double>;
    using M_t = std::vector<Matrix>;

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

    while (state.KeepRunning()) {
        auto omega = MyOracle<LmiOldOracle<Vec, Matrix>>(2, F1, B1, 3, F2, B2, Vec{1.0, -1.0, 1.0});
        auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});
        auto target = 1e100;  // std::numeric_limits<double>::max()
        auto result = cutting_plane_optim(omega, ellip, target);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(LMI_old);

// /**
//  * @brief
//  *
//  * @param[in,out] state
//  */
// static void LMI_No_Trick(benchmark::State &state) {
//   using Vec = std::valarray<double>;
//   using M_t = std::vector<Matrix>;
//
//   auto c = Vec{1.0, -1.0, 1.0};
//
//   auto m0F1 = Matrix(2);
//   m0F1.row(0) = Vec{-7.0, -11.0};
//   m0F1.row(1) = Vec{-11.0, 3.0};
//
//   auto m1F1 = Matrix(2);
//   m1F1.row(0) = Vec{7.0, -18.0};
//   m1F1.row(1) = Vec{-18.0, 8.0};
//
//   auto m2F1 = Matrix(2);
//   m2F1.row(0) = Vec{-2.0, -8.0};
//   m2F1.row(1) = Vec{-8.0, 1.0};
//
//   auto F1 = M_t{m0F1, m1F1, m2F1};
//
//   auto B1 = Matrix(2);
//   B1.row(0) = Vec{33.0, -9.0};
//   B1.row(1) = Vec{-9.0, 26.0};
//
//   auto m0F2 = Matrix(3);
//   m0F2.row(0) = Vec{-21.0, -11.0, 0.0};
//   m0F2.row(1) = Vec{-11.0, 10.0, 8.0};
//   m0F2.row(2) = Vec{0.0, 8.0, 5.0};
//
//   auto m1F2 = Matrix(3);
//   m1F2.row(0) = Vec{0.0, 10.0, 16.0};
//   m1F2.row(1) = Vec{10.0, -10.0, -10.0};
//   m1F2.row(2) = Vec{16.0, -10.0, 3.0};
//
//   auto m2F2 = Matrix(3);
//   m2F2.row(0) = Vec{-5.0, 2.0, -17.0};
//   m2F2.row(1) = Vec{2.0, -6.0, 8.0};
//   m2F2.row(2) = Vec{-17.0, 8.0, 6.0};
//
//   auto F2 = M_t{m0F2, m1F2, m2F2};
//
//   auto B2 = Matrix(3);
//   B2.row(0) = Vec{14.0, 9.0, 40.0};
//   B2.row(1) = Vec{9.0, 91.0, 10.0};
//   B2.row(2) = Vec{40.0, 10.0, 15.0};
//
//   while (state.KeepRunning()) {
//     auto omega = MyOracle<LmiOracle<Matrix>>(2, F1, B1, 3, F2, B2, Vec{1.0,
//     -1.0, 1.0}); auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});
//     ellip.no_defer_trick = true;
//     auto t = 1e100; // std::numeric_limits<double>::max()
//     [[maybe_unused]] const auto rslt = cutting_plane_optim(omega, ellip, t);
//   }
// }
//
// // Register the function as a benchmark
// BENCHMARK(LMI_No_Trick);

BENCHMARK_MAIN();

/*
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
LMI_Lazy         131235 ns       131245 ns         4447
LMI_old          196694 ns       196708 ns         3548
LMI_No_Trick     129743 ns       129750 ns         5357
*/
