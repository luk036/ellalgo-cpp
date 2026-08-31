#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <cmath>    // for sqrt
#include <cstddef>  // for size_t
#include <random>   // for mt19937, normal_distribution
#include <tuple>    // for get
#include <utility>  // for pair
#include <valarray> // for valarray
#include <vector>   // for vector

#include <ellalgo/cutting_plane.hpp>             // for cutting_plane_optim
#include <ellalgo/ell.hpp>                       // for Ell
#include <ellalgo/ell_config.hpp>                // for Options
#include <ellalgo/oracles/chebyshev_oracle.hpp>  // for ChebyshevOracle

using Vec = std::valarray<double>;

/**
 * @brief Generate the same random bounded polyhedron as the benchmark.
 *
 * @param[in] n ambient dimension of the polyhedron.
 * @param[in] m number of random halfspaces.
 * @param[in] seed random seed for reproducibility.
 * @return (A, b): one row of A per constraint.
 */
static auto generate_polyhedron(std::size_t n, std::size_t m, unsigned seed)
    -> std::pair<std::vector<Vec>, Vec> {
    std::mt19937 rng{seed};
    std::normal_distribution<double> normal{0.0, 1.0};
    std::uniform_real_distribution<double> uniform{0.3, 0.8};

    std::vector<Vec> A;
    A.reserve(2 * n + m);
    Vec b(2 * n + m);
    for (std::size_t i = 0; i != n; ++i) {
        auto epos = Vec(n);
        epos[i] = 1.0;
        A.push_back(epos);
        b[i] = 1.0;
        auto eneg = Vec(n);
        eneg[i] = -1.0;
        A.push_back(eneg);
        b[n + i] = 1.0;
    }
    for (std::size_t i = 0; i != m; ++i) {
        auto a = Vec(n);
        for (std::size_t j = 0; j != n; ++j) {
            a[j] = normal(rng);
        }
        a /= std::sqrt((a * a).sum());
        A.push_back(a);
        b[2 * n + i] = uniform(rng);
    }
    return {std::move(A), std::move(b)};
}

TEST_CASE("Chebyshev center, n = 10, m = 50") {
    const auto n = std::size_t{10};
    const auto m = std::size_t{50};
    const auto [A, b] = generate_polyhedron(n, m, 0);

    ChebyshevOracle omega{A, b};
    const auto kappa = std::sqrt(static_cast<double>(n + 1)) + 1.0;
    Ell<Vec> ellip{kappa, Vec(0.0, n + 1)};
    auto options = Options{500 * (n + 1) * (n + 1), 1e-10};
    double gamma = -1.0e100;

    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto& x = std::get<0>(result);
    const auto num_iters = std::get<1>(result);

    REQUIRE_NE(x.size(), 0U);  // a solution was found
    CHECK_GT(num_iters, 0U);

    // The found ball must be feasible: max_i (aᵢᵀx + ‖aᵢ‖ r − bᵢ) ≤ 0 (with tolerance)
    const auto r = x[n];
    double max_viol = 0.0;
    for (std::size_t i = 0; i != A.size(); ++i) {
        double slack = -b[i] + std::sqrt((A[i] * A[i]).sum()) * r;
        for (std::size_t j = 0; j != n; ++j) {
            slack += A[i][j] * x[j];
        }
        max_viol = std::max(max_viol, slack);
    }
    CHECK_LE(max_viol, 1e-6);
    CHECK_GT(r, 0.0);
    CHECK_LT(r, 1.0);
}

TEST_CASE("Chebyshev center, several sizes") {
    for (std::size_t n = 2; n <= 10; ++n) {
        const auto m = 5 * n;
        const auto [A, b] = generate_polyhedron(n, m, static_cast<unsigned>(n));  // different seed per size

        ChebyshevOracle omega{A, b};
        const auto kappa = std::sqrt(static_cast<double>(n + 1)) + 1.0;
        Ell<Vec> ellip{kappa, Vec(0.0, n + 1)};
        auto options = Options{500 * (n + 1) * (n + 1), 1e-10};
        double gamma = -1.0e100;

        const auto result = cutting_plane_optim(omega, ellip, gamma, options);
        const auto& x = std::get<0>(result);
        REQUIRE_NE(x.size(), 0U);

        const auto r = x[n];
        CHECK_GT(r, 0.0);
        CHECK_LT(r, 1.0);
    }
}
