// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <ellalgo/utility.hpp>
#include <tuple>
#include <xtensor/xarray.hpp>

#include "ell_calc.hpp"
#include "ell_matrix.hpp"

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        EllStable = {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllStable {
public:
  using Arr = xt::xarray<double, xt::layout_type::row_major>;
  // using params_t = std::tuple<double, double, double>;
  // using return_t = std::tuple<int, params_t>;

  bool no_defer_trick = false;

protected:
  const int _n;
  EllCalc _helper;
  double _kappa;
  Matrix _Q;
  Arr _xc;

  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] E
   */
  auto operator=(const EllStable &E) -> EllStable & = delete;

  /**
   * @brief Construct a new EllStable object
   *
   * @tparam V
   * @tparam U
   * @param kappa
   * @param Q
   * @param x
   */
  template <typename V, typename U>
  EllStable(V &&kappa, Matrix &&Q, U &&x) noexcept
      : _n{int(x.size())}, _helper{double(_n)}, _kappa{std::forward<V>(kappa)},
        _Q{std::move(Q)}, _xc{std::forward<U>(x)} {}

public:
  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] val
   * @param[in] x
   */
  EllStable(const Arr &val, Arr x)
      : EllStable{1.0, Matrix(int(x.size())), std::move(x)} {
    for (auto i = 0; i != this->_n; ++i) {
      this->_Q(i, i) = val[i];
    }
  }

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] alpha
   * @param[in] x
   */
  EllStable(const double &alpha, Arr x)
      : EllStable{alpha, Matrix(int(x.size())), std::move(x)} {
    this->_Q.identity();
  }

  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] E (move)
   */
  EllStable(EllStable &&E) = default;

  /**
   * @brief Destroy the EllStable object
   *
   */
  ~EllStable() {}

  /**
   * @brief Construct a new EllStable object
   *
   * To avoid accidentally copying, only explicit copy is allowed
   *
   * @param E
   */
  explicit EllStable(const EllStable &E) = default;

  /**
   * @brief explicitly copy
   *
   * @return EllStable
   */
  auto copy() const -> EllStable { return EllStable(*this); }

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
}; // } EllStable
