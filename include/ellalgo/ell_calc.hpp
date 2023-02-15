// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <tuple>

// forward declaration
enum class CUTStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        ell_calc = {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class ell_calc {
  public:
    bool use_parallel_cut = true;
    double _tsq{};
    double _rho{};
    double _sigma{};
    double _delta{};

  private:
    // const int _n;

    const double _nFloat;
    const double _nPlus1;
    const double _nMinus1;
    const double _halfN;
    const double _halfNplus1;
    const double _halfNminus1;
    const double _nSq;
    const double _c1;
    const double _c2;
    const double _c3;

    /**
     * @brief Construct a new ell_calc object
     *
     * @param[in] E
     */
    // auto operator=(const ell_calc& E) -> ell_calc& = delete;

  public:
    /**
     * @brief Construct a new ell_calc object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param Q
     * @param x
     */
    ell_calc(double nFloat) noexcept
        : _nFloat{nFloat},
          _nPlus1{_nFloat + 1.0},
          _nMinus1{_nFloat - 1.0},
          _halfN{_nFloat / 2.0},
          _halfNplus1{_nPlus1 / 2.0},
          _halfNminus1{_nMinus1 / 2.0},
          _nSq{_nFloat * _nFloat},
          _c1{_nSq / (_nSq - 1.0)},
          _c2{2.0 / _nPlus1},
          _c3{_nFloat / _nPlus1} {}

  public:
    /**
     * @brief Construct a new ell_calc object
     *
     * @param[in] E (move)
     */
    ell_calc(ell_calc&& E) = default;

    /**
     * @brief Destroy the ell_calc object
     *
     */
    ~ell_calc() {}

    /**
     * @brief Construct a new ell_calc object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    ell_calc(const ell_calc& E) = default;

    /**
     * @brief Calculate new ellipsoid under Parallel Cut
     *
     *        g' (x - xc) + beta0 \le 0
     *        g' (x - xc) + beta1 \ge 0
     *
     * @param[in] b0
     * @param[in] b1
     * @return int
     */
    auto _calc_ll_core(const double& b0, const double& b1) -> CUTStatus;

    /**
     * @brief Calculate new ellipsoid under Parallel Cut, one of them is central
     *
     *        g' (x - xc) \le 0
     *        g' (x - xc) + beta1 \ge 0
     *
     * @param[in] b1
     * @param[in] b1sq
     */
    auto _calc_ll_cc(const double& b1, const double& b1sq) -> void;

    /**
     * @brief Calculate new ellipsoid under Deep Cut
     *
     *        g' (x - xc) + beta \le 0
     *
     * @param[in] beta
     */
    auto _calc_dc(const double& beta) noexcept -> CUTStatus;

    /**
     * @brief Calculate new ellipsoid under Central Cut
     *
     *        g' (x - xc) \le 0
     *
     * @param[in] tau
     */
    auto _calc_cc(const double& tau) noexcept -> void;

};  // } ell_calc
