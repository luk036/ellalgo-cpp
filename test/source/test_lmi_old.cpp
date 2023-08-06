/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#include <doctest/doctest.h>  // for ResultBuilder, CHECK, TestCase

#include <ellalgo/cutting_plane.hpp>           // for cutting_plane_optim
#include <ellalgo/ell.hpp>                     // for Ell
#include <ellalgo/ell_matrix.hpp>              // for EllStable
#include <ellalgo/ell_stable.hpp>              // for EllStable
#include <ellalgo/oracles/lmi_old_oracle.hpp>  // for LmiOldOracle
// #include <gsl/span> // for span
// #include <spdlog/sinks/stdout_sinks.h>
// #include <spdlog/spdlog.h>
// #include <optional>    // for optional
#include <tuple>        // for tuple
#include <type_traits>  // for move, add_const<>::type
#include <valarray>
#include <vector>  // for vector
// #include <xtensor-blas/xlinalg.hpp>

/**
 * @brief MyOracle
 *
 */
class MyOldOracle {
    // using Arr = xt::xarray<double, xt::layout_type::row_major>;
    using Vec = std::valarray<double>;
    using M_t = std::vector<Matrix>;
    using Cut = std::pair<Vec, double>;

  private:
    LmiOldOracle<Vec, Matrix> lmi1;
    LmiOldOracle<Vec, Matrix> lmi2;
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
    MyOldOracle(size_t m1, const std::vector<Matrix> &F1, const Matrix &B1, size_t m2,
                const std::vector<Matrix> &F2, const Matrix &B2, Vec c)
        : lmi1{m1, F1, B1}, lmi2{m2, F2, B2}, c{std::move(c)} {}

    /**
     * @brief
     *
     * @param[in] x
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    auto assess_optim(const Vec &x, double &t) -> std::tuple<Cut, bool> {
        const auto cut1 = this->lmi1(x);
        if (cut1) {
            return {*cut1, false};
        }
        const auto cut2 = this->lmi2(x);
        if (cut2) {
            return {*cut2, false};
        }
        const auto f0 = (this->c * x).sum();
        const auto f1 = f0 - t;
        if (f1 > 0.0) {
            return {{this->c, f1}, false};
        }
        t = f0;
        return {{this->c, 0.0}, true};
    }

    /**
     * @brief
     *
     * @param[in] x
     * @param[in,out] t the best-so-far optimal value
     * @return std::tuple<Cut, double>
     */
    auto operator()(const Vec &x, double &t) -> std::tuple<Cut, bool> {
        return this->assess_optim(x, t);
    }
};

TEST_CASE("LMI test (stable)") {
    // using Arr = xt::xarray<double, xt::layout_type::row_major>;
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

    auto omega = MyOldOracle(2, F1, B1, 3, F2, B2, std::move(c));
    auto ellip = Ell<Vec>(10.0, Vec{0.0, 0.0, 0.0});

    auto t = 1e100;  // std::numeric_limits<double>::max()
    // const auto [x, num_iters] = cutting_plane_optim(omega, ellip, t);
    const auto result = cutting_plane_optim(omega, ellip, t);
    auto x = std::get<0>(result);
    auto num_iters = std::get<1>(result);
    // fmt::print("{:f} {} {} \n", t, num_iters, ell_info.feasible);
    // std::cout << "LMI xbest: " << xb << "\n";
    // std::cout << "LMI result: " << fb << ", " << niter << ", " << feasible <<
    // ", " << status
    //           << "\n";

    // create color multi threaded logger
    // auto console = spdlog::stdout_logger_mt("console");
    // auto err_logger = spdlog::stderr_logger_mt("stderr");
    // spdlog::get("console")->info("loggers can be retrieved from a global "
    //                              "registry using the
    //                              spdlog::get(logger_name)");

    CHECK(x.size() != 0U);
    CHECK(num_iters == 112);
}

TEST_CASE("LMI test ") {
    // using Arr = xt::xarray<double, xt::layout_type::row_major>;
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

    auto omega = MyOldOracle(2, F1, B1, 3, F2, B2, std::move(c));
    auto ellip = EllStable<Vec>(10.0, Vec{0.0, 0.0, 0.0});

    auto t = 1e100;  // std::numeric_limits<double>::max()
    // const auto [x, num_iters] = cutting_plane_optim(omega, ellip, t);
    const auto result = cutting_plane_optim(omega, ellip, t);
    const auto &x = std::get<0>(result);
    const auto &num_iters = std::get<1>(result);

    CHECK(x.size() != 0U);
    CHECK(num_iters == 112);
}
