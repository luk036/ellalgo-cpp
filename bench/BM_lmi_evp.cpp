/*
 *  Distributed under the MIT License (See accompanying file /LICENSE )
 */
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <chrono>                          // for steady_clock, duration
#include <cmath>                           // for sqrt
#include <cstddef>                         // for size_t
#include <cstdio>                          // for printf
#include <ellalgo/cutting_plane.hpp>       // for cutting_plane_optim
#include <ellalgo/ell.hpp>                 // for Ell
#include <ellalgo/ell_config.hpp>          // for Options
#include <ellalgo/ell_matrix.hpp>          // for Matrix
#include <ellalgo/oracles/lmi_oracle.hpp>  // for LmiOracle
#include <random>                          // for mt19937, normal_distribution
#include <string>                          // for to_string
#include <tuple>                           // for get
#include <utility>                         // for move
#include <valarray>                        // for valarray
#include <vector>                          // for vector

using Vec = std::valarray<double>;
using M_t = std::vector<Matrix>;

/**
 * @brief Generate a random symmetric n×n matrix.
 *
 * @param[in] n matrix dimension.
 * @param[in,out] rng random number generator.
 * @return Matrix the symmetric matrix (M + Mᵀ)/2 with standard-normal entries.
 */
static auto make_symmetric(std::size_t n, std::mt19937& rng) -> Matrix {
    std::normal_distribution<double> normal{0.0, 1.0};
    auto M = Matrix(n);
    for (std::size_t i = 0; i != n; ++i) {
        for (std::size_t j = 0; j != n; ++j) {
            M(i, j) = normal(rng);
        }
    }
    for (std::size_t i = 0; i != n; ++i) {
        for (std::size_t j = i + 1; j != n; ++j) {
            const auto v = (M(i, j) + M(j, i)) / 2.0;
            M(i, j) = v;
            M(j, i) = v;
        }
    }
    return M;
}

/**
 * @brief Generate a random affine matrix pencil A0 + Σ xᵢAᵢ.
 *
 * @param[in] n LMI matrix dimension.
 * @param[in] m number of design variables.
 * @param[in,out] rng random number generator.
 * @return A0 and the list of Aᵢ matrices.
 */
static auto generate_pencil(std::size_t n, std::size_t m, std::mt19937& rng)
    -> std::pair<Matrix, M_t> {
    auto A0 = make_symmetric(n, rng);
    auto A_list = M_t{};
    A_list.reserve(m);
    for (std::size_t i = 0; i != m; ++i) {
        A_list.push_back(make_symmetric(n, rng));
    }
    return {std::move(A0), std::move(A_list)};
}

/**
 * @brief Frobenius norm of a square matrix (an upper bound on the 2-norm).
 *
 * @param[in] M the matrix.
 * @param[in] d matrix dimension.
 * @return double the Frobenius norm ‖M‖_F.
 */
static auto fro_norm(const Matrix& M, std::size_t d) -> double {
    double s = 0.0;
    for (std::size_t i = 0; i != d; ++i) {
        for (std::size_t j = 0; j != d; ++j) {
            s += M(i, j) * M(i, j);
        }
    }
    return std::sqrt(s);
}

/**
 * @brief Optimization oracle for the min-eigenvalue EVP.
 *
 * The search point is xc = (x, t) ∈ R^{m+1}.  Minimize t subject to
 * t·I − A(x) ⪰ 0 (an LMI) and box constraints −1 ≤ xᵢ ≤ 1.
 * The LMI is expressed as B − Σ Fₖ·xcₖ ⪰ 0 with B = −A0,
 * Fᵢ = Aᵢ (i = 1..m) and F_{m+1} = −I, handled by LmiOracle.
 */
class EigenOracle {
    using Cut = std::pair<Vec, double>;

    M_t _F;     // A₁..A_m, −I  (must precede _lmi which references it)
    Matrix _B;  // −A0
    LmiOracle<Vec, Matrix> _lmi;
    std::size_t _m;  // number of x variables (t is the last component)
    Vec _c;          // objective gradient: e_{m+1} (selects t)

  public:
    /**
     * @brief Construct the EVP oracle.
     *
     * @param[in] n LMI matrix dimension.
     * @param[in] m number of x variables.
     * @param[in] A0 constant matrix of the pencil.
     * @param[in] A_list variable matrices A₁..A_m.
     */
    EigenOracle(std::size_t n, std::size_t m, const Matrix& A0, const M_t& A_list)
        : _F{A_list}, _B{A0 * -1.0}, _lmi{n, this->_F, this->_B}, _m{m}, _c(Vec(0.0, m + 1)) {
        auto I = Matrix(n);
        I.identity();
        this->_F.push_back(I * -1.0);  // F_{m+1} = −I
        this->_c[m] = 1.0;             // objective: minimize t
    }

    /**
     * @brief Assess feasibility and optimality at (x, t).
     *
     * @param[in] xc candidate point (x, t).
     * @param[in,out] gamma best-so-far objective value.
     * @return (cut, shrunk): the cutting plane and whether gamma improved.
     */
    auto assess_optim(const Vec& xc, double& gamma) -> std::tuple<Cut, bool> {
        for (std::size_t i = 0; i != this->_m; ++i) {  // box: −1 ≤ xᵢ ≤ 1
            if (const auto fj = xc[i] - 1.0; fj > 0.0) {
                auto g = Vec(0.0, xc.size());
                g[i] = 1.0;
                return {{std::move(g), fj}, false};
            }
            if (const auto fj = -1.0 - xc[i]; fj > 0.0) {
                auto g = Vec(0.0, xc.size());
                g[i] = -1.0;
                return {{std::move(g), fj}, false};
            }
        }
        if (const auto* const cut = this->_lmi(xc)) {  // t·I − A(x) ⪰ 0
            return {*cut, false};
        }
        const auto f0 = (this->_c * xc).sum();  // objective: minimize t
        if (const auto fj = f0 - gamma; fj > 0.0) {
            return {{this->_c, fj}, false};  // deep objective cut
        }
        gamma = f0;
        return {{this->_c, 0.0}, true};  // improved -> central cut
    }
};

/**
 * @brief Solve the EVP for a given number of x variables.
 *
 * @param[in] n LMI matrix dimension (fixed at 5).
 * @param[in] m number of x variables (design variables = m + 1).
 * @return (wall time in seconds, iterations, final objective t).
 */
static auto solve_evp(std::size_t n, std::size_t m) -> std::tuple<double, std::size_t, double> {
    std::mt19937 rng{0};
    const auto [A0, A_list] = generate_pencil(n, m, rng);
    EigenOracle omega{n, m, A0, A_list};

    // Feasible region: x ∈ [−1,1]^m, |t| ≤ ‖A0‖_F + √m·max‖Aᵢ‖_F.
    auto bound_t = fro_norm(A0, n);
    for (const auto& A : A_list) {
        bound_t = std::max(bound_t, fro_norm(A, n));
    }
    bound_t *= std::sqrt(static_cast<double>(m));
    const auto kappa = std::sqrt(static_cast<double>(m) + bound_t * bound_t) + 1.0;
    Ell<Vec> ellip{kappa, Vec(0.0, m + 1)};
    auto options = Options{20000, 1e-10};

    double gamma = 1e100;
    const auto start = std::chrono::steady_clock::now();
    const auto result = cutting_plane_optim(omega, ellip, gamma, options);
    const auto stop = std::chrono::steady_clock::now();
    const auto secs = std::chrono::duration<double>(stop - start).count();
    const auto num_iters = std::get<1>(result);
    return {secs, num_iters, gamma};
}

int main() {
    constexpr std::size_t N = 5;                                  // LMI matrix dimension
    static constexpr std::size_t M_SIZES[] = {5, 8, 12, 16, 20};  // x-variable counts
    std::printf("%4s %5s %14s %8s %12s\n", "m", "vars", "ellipsoid(s)", "iters", "t*");
    std::printf("%s\n", std::string(50, '-').c_str());
    for (const auto m : M_SIZES) {
        const auto [secs, iters, tstar] = solve_evp(N, m);
        std::printf("%4zu %5zu %14.6f %8zu %12.6f\n", m, m + 1, secs, iters, tstar);
    }

    ankerl::nanobench::Bench bench;
    bench.title("Min-eigenvalue LMI: ellipsoid method (m = 5..20)")
        .unit("op")
        .warmup(100)
        .epochs(50);
    for (const auto m : M_SIZES) {
        bench.run("m=" + std::to_string(m), [&] {
            std::mt19937 rng{0};
            const auto [A0, A_list] = generate_pencil(N, m, rng);
            EigenOracle omega{N, m, A0, A_list};
            auto bound_t = fro_norm(A0, N);
            for (const auto& A : A_list) {
                bound_t = std::max(bound_t, fro_norm(A, N));
            }
            bound_t *= std::sqrt(static_cast<double>(m));
            const auto kappa = std::sqrt(static_cast<double>(m) + bound_t * bound_t) + 1.0;
            Ell<Vec> ellip{kappa, Vec(0.0, m + 1)};
            auto options = Options{20000, 1e-10};
            double gamma = 1e100;
            const auto result = cutting_plane_optim(omega, ellip, gamma, options);
            ankerl::nanobench::doNotOptimizeAway(result);
        });
    }
}
