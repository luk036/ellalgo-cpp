#pragma once

#include <cassert>

#include "ell_calc_core.hpp"
#include "ell_config.hpp"

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalc= {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllCalc {
  public:
    bool use_parallel_cut = true;

  protected:
    double _n_f;
    EllCalcCore _helper;

  public:
    /**
     * @brief Construct a new EllCalcobject
     *
     * @param[in] ndim
     */
    explicit EllCalc(const size_t ndim) : _n_f{double(ndim)}, _helper{ndim} {
        assert(ndim >= 2U);  // do not accept one-dimensional
    }

    /**
     * @brief Construct a new EllCalcobject
     *
     * @param[in] E (move)
     */
    EllCalc(EllCalc&& E) = default;

    /**
     * @brief Copy assignment operator
     */
    EllCalc& operator=(const EllCalc& other) = default;

    /**
     * @brief Move assignment operator
     */
    EllCalc& operator=(EllCalc&& other) = default;

    /**
     * @brief Destroy the EllCalcobject
     *
     */
    ~EllCalc() = default;

    /**
     * @brief Construct a new EllCalcobject
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param[in] E
     */
    EllCalc(const EllCalc& E) = default;

    /**
     * @brief Parallel deep cut
     *
     * Two parallel constraints: beta0 ≤ g'(x - xc) ≤ beta1.
     * Falls back to calc_bias_cut when parallel cut is ineffective.
     *
     * @param[in] beta0  Lower bound
     * @param[in] beta1  Upper bound
     * @param[in] tsq    Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_parallel_bias_cut(double beta0, double beta1, double tsq) const -> CutResult;

    /**
     * @brief Parallel central cut
     *
     * One central + one parallel: 0 ≤ g'(x - xc) ≤ beta1.
     * Falls back to calc_central_cut when parallel cut is ineffective.
     *
     * @param[in] beta1  Upper bound
     * @param[in] tsq    Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_parallel_central_cut(double beta1, double tsq) const -> CutResult;

    /**
     * @brief Deep (bias) cut
     *
     * Single constraint: g'(x - xc) + beta ≤ 0.
     *
     * @param[in] beta  Bias term (≥ 0)
     * @param[in] tsq   Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_bias_cut(double beta, double tsq) const -> CutResult;

    /**
     * @brief Central cut
     *
     * Single constraint passing through center: g'(x - xc) ≤ 0.
     *
     * @param[in] tsq  Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_central_cut(double tsq) const -> CutResult;

    /**
     * @brief Parallel deep cut (Q-version for discrete optimization)
     *
     * Two parallel constraints with alternative fallback (NoEffect).
     * Used by cutting_plane_optim_q for discrete convex problems.
     *
     * @param[in] beta0  Lower bound
     * @param[in] beta1  Upper bound
     * @param[in] tsq    Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_parallel_bias_cut_q(double beta0, double beta1, double tsq) const -> CutResult;

    /**
     * @brief Deep cut (Q-version for discrete optimization)
     *
     * Single constraint with NoEffect fallback (instead of NoSoln).
     * Used by cutting_plane_optim_q for discrete convex problems.
     *
     * @param[in] beta  Bias term
     * @param[in] tsq   Squared ellipsoid radius (τ²)
     * @return CutResult with rho, sigma, delta
     */
    auto calc_bias_cut_q(double beta, double tsq) const -> CutResult;
};  // } EllCalc
