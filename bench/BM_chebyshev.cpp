/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cmath>      // for log, sqrt
#include <cstddef>    // for size_t
#include <random>     // for default_random_engine, normal_distribution
#include <tuple>      // for get
#include <utility>    // for move
#include <valarray>   // for valarray
#include <vector>     // for vector

#include <ellalgo/cutting_plane.hpp>            // for cutting_plane_optim
#include <ellalgo/ell.hpp>                      // for Ell
#include <ellalgo/ell_config.hpp>               // for Options
#include <ellalgo/oracles/chebyshev_oracle.hpp>  // for ChebyshevOracle

using Vec = std::valarray<double>;

/**
 * @brief Generate a random bounded polyhedron P = {x : A x ≤ b}.
 *
 * The polyhedron is the intersection of the box [-1, 1]^n (which makes it
 * bounded) with `m` random halfspaces whose offsets are positive, so the
 * origin is strictly feasible and the Chebyshev radius is > 0.
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
    // Box constraints: -1 ≤ x_i ≤ 1
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
    // Random halfspaces: unit normals aᵢ, offsets in (0.3, 0.8)
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

/**
 * @brief Solve the Chebyshev center problem for a given size.
 *
 * @param[in] n center dimension (design variables = n + 1).
 * @param[in] m number of random halfspaces.
 * @return (wall time in seconds, iterations).
 */
static auto solve_ellipsoid(std::size_t n, std::size_t m) -> std::pair<double, std::size_t> {
    const auto [A, b] = generate_polyhedron(n, m, 0);
    ChebyshevOracle omega{A, b};
    // Feasible region lies in [-1, 1]^n × [0, 1] ⊂ ball of radius √(n+1).
    const auto kappa = std::sqrt(static_cast<double>(n + 1)) + 1.0;
    Ell<Vec> ellip{kappa, Vec(0.0, n + 1)};
    auto options = Options{500 * (n + 1) * (n + 1), 1e-10};
    double gamma = -1.0e100;  // like float("-inf") in the Python demo
    const auto start = std::chrono::steady_clock::now();
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto stop = std::chrono::steady_clock::now();
    const auto secs = std::chrono::duration<double>(stop - start).count();
    const auto num_iters = std::get<1>(result);
    return {secs, num_iters};
}

int main() {
    static constexpr std::size_t SIZES[] = {5, 10, 15, 20, 30};  // n values, m = 5n
    std::printf("%4s %5s %14s %8s\n", "n", "m", "ellipsoid(s)", "iters");
    std::printf("%s\n", std::string(38, '-').c_str());
    for (const auto n : SIZES) {
        const auto m = 5 * n;
        const auto [secs, iters] = solve_ellipsoid(n, m);
        std::printf("%4zu %5zu %14.6f %8zu\n", n, m, secs, iters);
    }

    ankerl::nanobench::Bench bench;
    bench.title("Chebyshev center: ellipsoid method (n = 5..30)").unit("op").warmup(100).epochs(50);
    for (const auto n : SIZES) {
        const auto m = 5 * n;
        const auto [A, b] = generate_polyhedron(n, m, 0);
        bench.run("n=" + std::to_string(n), [&] {
            ChebyshevOracle omega{A, b};
            const auto kappa = std::sqrt(static_cast<double>(n + 1)) + 1.0;
            Ell<Vec> ellip{kappa, Vec(0.0, n + 1)};
            auto options = Options{500 * (n + 1) * (n + 1), 1e-10};
            double gamma = -1.0e100;
            const auto result = cutting_plane_optim(omega, ellip, gamma, options);
            ankerl::nanobench::doNotOptimizeAway(result);
        });
    }
}
