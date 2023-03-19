// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <ellalgo/utility.hpp>
#include <tuple>
#include <xtensor/xarray.hpp>

#include "ell_calc.hpp"

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        ell_ss {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class Ell {
public:
  using Arr = xt::xarray<double, xt::layout_type::row_major>;
  // using params_t = std::tuple<double, double, double>;
  // using return_t = std::tuple<int, params_t>;

  bool no_defer_trick = false;

private:
  const int _n;
  EllCalc _helper;
  double _kappa;
  Arr _Q;
  Arr _xc;

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] E
   */
  auto operator=(const Ell &E) -> Ell & = delete;

  /**
   * @brief Construct a new Ell object
   *
   * @tparam V
   * @tparam U
   * @param kappa
   * @param Q
   * @param x
   */
  template <typename V, typename U>
  Ell(V &&kappa, Arr &&Q, U &&x) noexcept
      : _n{int(x.size())}, _helper{double(_n)}, _kappa{std::forward<V>(kappa)},
        _Q{std::move(Q)}, _xc{std::forward<U>(x)} {}

public:
  /**
   * @brief Construct a new Ell object
   *
   * @param[in] val
   * @param[in] x
   */
  Ell(const Arr &val, Arr x) noexcept : Ell{1.0, xt::diag(val), std::move(x)} {}

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] alpha
   * @param[in] x
   */
  Ell(const double &alpha, Arr x) noexcept
      : Ell{alpha, xt::eye(x.size()), std::move(x)} {}

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] E (move)
   */
  Ell(Ell &&E) = default;

  /**
   * @brief Destroy the Ell object
   *
   */
  ~Ell() {}

  /**
   * @brief Construct a new Ell object
   *
   * To avoid accidentally copying, only explicit copy is allowed
   *
   * @param E
   */
  explicit Ell(const Ell &E) = default;

  /**
   * @brief explicitly copy
   *
   * @return Ell
   */
  auto copy() const -> Ell { return Ell(*this); }

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
  void set_xc(const Arr &xc) { _xc = xc; }

  void set_use_parallel_cut(bool value) {
    this->_helper.use_parallel_cut = value;
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update(const std::pair<Arr, T> &cut) -> std::tuple<CutStatus, double>;

protected:
  auto _update_cut(const double &beta) -> CutStatus {
    return this->_helper._calc_dc(beta);
  }

  auto _update_cut(const Arr &beta) -> CutStatus { // parallel cut
    if (beta.shape()[0] < 2) {
      return this->_helper._calc_dc(beta[0]);
    }
    return this->_helper._calc_ll_core(beta[0], beta[1]);
  }
}; // } Ell
