// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <ellalgo/utility.hpp>
#include <tuple>
#include <xtensor/xarray.hpp>

#include "ell_calc.hpp"

// forward declaration
enum class CUTStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        ell = {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class ell {
  public:
    using Arr = xt::xarray<double, xt::layout_type::row_major>;
    // using params_t = std::tuple<double, double, double>;
    // using return_t = std::tuple<int, params_t>;

    bool no_defer_trick = false;

  private:
    const int _n;
    ell_calc _helper;
    double _kappa;
    Arr _Q;
    Arr _xc;

    /**
     * @brief Construct a new ell object
     *
     * @param[in] E
     */
    auto operator=(const ell& E) -> ell& = delete;

    /**
     * @brief Construct a new ell object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param Q
     * @param x
     */
    template <typename V, typename U> ell(V&& kappa, Arr&& Q, U&& x) noexcept
        : _n{int(x.size())},
          _helper{double(_n)},
          _kappa{std::forward<V>(kappa)},
          _Q{std::move(Q)},
          _xc{std::forward<U>(x)} {}

  public:
    /**
     * @brief Construct a new ell object
     *
     * @param[in] val
     * @param[in] x
     */
    ell(const Arr& val, Arr x) noexcept : ell{1.0, xt::diag(val), std::move(x)} {}

    /**
     * @brief Construct a new ell object
     *
     * @param[in] alpha
     * @param[in] x
     */
    ell(const double& alpha, Arr x) noexcept : ell{alpha, xt::eye(x.size()), std::move(x)} {}

    /**
     * @brief Construct a new ell object
     *
     * @param[in] E (move)
     */
    ell(ell&& E) = default;

    /**
     * @brief Destroy the ell object
     *
     */
    ~ell() {}

    /**
     * @brief Construct a new ell object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    explicit ell(const ell& E) = default;

    /**
     * @brief explicitly copy
     *
     * @return ell
     */
    auto copy() const -> ell { return ell(*this); }

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    auto xc() const -> Arr { return _xc; }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    void set_xc(const Arr& xc) { _xc = xc; }

    void set_use_parallel_cut(bool value) { this->_helper.use_parallel_cut = value; }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T> auto update(const std::tuple<Arr, T>& cut)
        -> std::tuple<CUTStatus, double>;

  protected:
    auto _update_cut(const double& beta) -> CUTStatus { return this->_helper._calc_dc(beta); }

    auto _update_cut(const Arr& beta) -> CUTStatus {  // parallel cut
        if (beta.shape()[0] < 2) {
            return this->_helper._calc_dc(beta[0]);
        }
        return this->_helper._calc_ll_core(beta[0], beta[1]);
    }
};  // } ell
