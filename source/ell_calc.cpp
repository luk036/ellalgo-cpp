/**
 * @file ell_calc.cpp
 * @brief Implementation of ellipsoid calculation utilities
 *
 * This file implements the ellipsoid calculation methods for different
 * types of cutting planes. It provides the interface between the
 * core calculations and the ellipsoid algorithm.
 */

#include <cassert>
#include <cmath>                      // for sqrt
#include <ellalgo/ell_assert.hpp>     // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>       // for EllCalc
#include <ellalgo/ell_calc_core.hpp>  // for EllCalcCore
#include <ellalgo/ell_config.hpp>     // for CutStatus, CutStatus::Success

/**
 * @brief Parallel bias cut computation
 *
 * Two parallel constraints: beta0 ≤ g'(x - xc) ≤ beta1.
 * Falls back to calc_bias_cut when parallel cut is ineffective.
 *
 * @param[in] beta0 Lower bound of the parallel cut
 * @param[in] beta1 Upper bound of the parallel cut
 * @param[in] tsq   Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_parallel_bias_cut(const double beta0, const double beta1, const double tsq) const
    -> CutResult {
    if (beta1 < beta0) {
        return {.status = CutStatus::NoSoln, .rho = 0.0, .sigma = 0.0, .delta = 0.0};  // no sol'n
    }
    if ((beta1 > 0 && tsq <= beta1 * beta1) || !this->use_parallel_cut) {
        return this->calc_bias_cut(beta0, tsq);
    }
    auto&& core = this->_helper.calc_parallel_cut(beta0, beta1, tsq);
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
}

/**
 * @brief Parallel central cut computation
 *
 * One central cut through the center plus one parallel constraint:
 * 0 ≤ g'(x - xc) ≤ beta1.
 *
 * @param[in] beta1 Upper bound of the parallel cut
 * @param[in] tsq   Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_parallel_central_cut(const double beta1, const double tsq) const -> CutResult {
    if (beta1 < 0.0) {
        return {.status = CutStatus::NoSoln, .rho = 0.0, .sigma = 0.0, .delta = 0.0};  // no sol'n
    }
    if (tsq <= beta1 * beta1 || !this->use_parallel_cut) {
        return this->calc_central_cut(tsq);
    }
    auto&& core = this->_helper.calc_parallel_central_cut(beta1, tsq);
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
    // this->_mu ???
}

/**
 * @brief Deep (bias) cut
 *
 * Single constraint: g'(x - xc) + beta ≤ 0.
 * Checks feasibility then delegates to EllCalcCore for computation.
 *
 * @param[in] beta Bias term (≥ 0)
 * @param[in] tsq  Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_bias_cut(const double beta, const double tsq) const -> CutResult {
    assert(beta >= 0.0);
    if (tsq < beta * beta) {
        return {.status = CutStatus::NoSoln, .rho = 0.0, .sigma = 0.0, .delta = 0.0};  // no sol'n
    }
    auto&& core = this->_helper.calc_bias_cut(beta, std::sqrt(tsq));
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
}

/**
 * @brief Central cut
 *
 * Single constraint passing through center: g'(x - xc) ≤ 0.
 *
 * @param[in] tsq Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_central_cut(const double tsq) const -> CutResult {
    // auto sigma = this->_c2;
    // auto rho = std::sqrt(tsq) / this->_nPlus1;
    // auto delta = this->_c1;
    auto&& core = this->_helper.calc_central_cut(std::sqrt(tsq));
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
}

/**
 * @brief Parallel bias cut (Q-version for discrete optimization)
 *
 * Two parallel constraints with alternative fallback (NoEffect).
 * Used by cutting_plane_optim_q for discrete convex problems.
 *
 * @param[in] beta0 Lower bound of the parallel cut
 * @param[in] beta1 Upper bound of the parallel cut
 * @param[in] tsq   Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_parallel_bias_cut_q(const double beta0, const double beta1,
                                       const double tsq) const -> CutResult {
    if (beta1 < beta0) {
        return {.status = CutStatus::NoSoln, .rho = 0.0, .sigma = 0.0, .delta = 0.0};  // no sol'n
    }

    if ((beta1 > 0.0 && tsq <= beta1 * beta1) || !this->use_parallel_cut) {
        return this->calc_bias_cut_q(beta0, tsq);
    }

    const auto b0b1 = beta0 * beta1;
    const auto eta = tsq + this->_n_f * b0b1;
    if (ELL_UNLIKELY(eta <= 0.0)) {
        return {
            .status = CutStatus::NoEffect, .rho = 0.0, .sigma = 0.0, .delta = 1.0};  // no effect
    }
    auto&& core = this->_helper.calc_parallel_cut_fast(beta0, beta1, tsq, b0b1, eta);
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
}

/**
 * @brief Deep cut (Q-version for discrete optimization)
 *
 * Single constraint with NoEffect fallback (instead of NoSoln).
 * Used by cutting_plane_optim_q for discrete convex problems.
 *
 * @param[in] beta Bias term
 * @param[in] tsq  Squared ellipsoid radius τ²
 * @return CutResult with status, rho, sigma, delta
 */
auto EllCalc::calc_bias_cut_q(const double beta, const double tsq) const -> CutResult {
    const auto tau = std::sqrt(tsq);
    if (tau < beta) {
        return {.status = CutStatus::NoSoln, .rho = 0.0, .sigma = 0.0, .delta = 0.0};  // no sol'n
    }
    const auto eta = tau + this->_n_f * beta;
    if (ELL_UNLIKELY(eta <= 0.0)) {
        return {
            .status = CutStatus::NoEffect, .rho = 0.0, .sigma = 0.0, .delta = 1.0};  // no effect
    }
    auto&& core = this->_helper.calc_bias_cut_fast(beta, tau, eta);
    return {.status = CutStatus::Success,
            .rho = std::get<0>(core),
            .sigma = std::get<1>(core),
            .delta = std::get<2>(core)};
}
