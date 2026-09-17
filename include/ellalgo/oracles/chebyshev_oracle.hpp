/**
 * @file chebyshev_oracle.hpp
 * @brief Oracle for the Chebyshev center problem (largest inscribed ball)
 */

// -*- coding: utf-8 -*-
#pragma once

#include <cmath>  // for sqrt
#include <cstddef>
#include <tuple>  // for tuple
#include <utility>
#include <valarray>
#include <vector>

#include "../round_robin.hpp"

/**
 * @brief Oracle for the Chebyshev center of a polyhedron.
 *
 * Given a polyhedron P = {u ∈ R^n : A u ≤ b}, find the largest Euclidean
 * ball B(x, r) = {u : ‖u − x‖₂ ≤ r} contained in P:
 *
 *     max     r
 *     s.t.    aᵢᵀx + ‖aᵢ‖₂ r ≤ bᵢ,   i = 1, …, m,
 *
 * with design variables (x, r) ∈ R^{n+1}.
 *
 * Every constraint is affine in (x, r) with a *constant* gradient
 * (aᵢ, ‖aᵢ‖₂), so each violated halfspace yields a perfect cutting plane;
 * the objective is linear, so the optimality cut is trivial. This makes
 * the problem an ideal showcase for the cutting-plane method.
 *
 * @note Strategy pattern: constraints are scanned cyclically via a
 *       `RoundRobin` counter (mirroring ProfitOracle).
 */
class ChebyshevOracle {
  public:
    using Vec = std::valarray<double>;
    using ArrayType = Vec;
    using Cut = std::pair<Vec, double>;

    /**
     * @brief Construct from halfspace data.
     *
     * @param[in] A m×n matrix; row i is the normal aᵢ.
     * @param[in] b offsets bᵢ (length m).
     */
    ChebyshevOracle(const std::vector<Vec>& A, const Vec& b)
        : _A{A}, _b{b}, _norms(Vec(A.size())), _rr{A.size()} {
        for (std::size_t i = 0; i != A.size(); ++i) {
            this->_norms[i] = std::sqrt((this->_A[i] * this->_A[i]).sum());
        }
    }

    ChebyshevOracle(const ChebyshevOracle&) = delete;
    auto operator=(const ChebyshevOracle&) -> ChebyshevOracle& = delete;
    ChebyshevOracle(ChebyshevOracle&&) = delete;
    auto operator=(ChebyshevOracle&&) -> ChebyshevOracle& = delete;
    ~ChebyshevOracle() = default;

    /**
     * @brief Assess feasibility and optimality at a candidate point.
     *
     * @param[in] xc candidate point (x, r); length n + 1.
     * @param[in,out] gamma best-so-far objective value (radius).
     * @return (cut, shrunk): the cutting plane, and whether gamma improved.
     */
    auto assess_optim(const Vec& xc, double& gamma) -> std::tuple<Cut, bool> {
        const auto n = xc.size() - 1;  // dimension of the center x
        const auto r = xc[n];          // radius

        // Feasibility: aᵢᵀx + ‖aᵢ‖₂ r ≤ bᵢ (round robin)
        for (std::size_t i = 0; i != this->_norms.size(); ++i) {
            const auto k = this->_rr.next();
            double fj = -this->_b[k] + this->_norms[k] * r;
            for (std::size_t j = 0; j != n; ++j) {
                fj += this->_A[k][j] * xc[j];
            }
            if (fj > 0.0) {
                auto g = Vec(xc.size());
                for (std::size_t j = 0; j != n; ++j) {
                    g[j] = this->_A[k][j];
                }
                g[n] = this->_norms[k];
                return {{std::move(g), fj}, false};
            }
        }

        // Optimality: maximize r
        const auto f0 = r;
        auto g = Vec(xc.size());
        g[n] = -1.0;  // gradient of -(r - gamma)
        if (gamma - f0 > 0.0) {
            return {{std::move(g), gamma - f0}, false};  // deep cut toward gamma
        }
        gamma = f0;  // improved
        return {{std::move(g), 0.0}, true};
    }

  private:
    std::vector<Vec> _A;
    Vec _b;
    Vec _norms;
    RoundRobin _rr;
};
