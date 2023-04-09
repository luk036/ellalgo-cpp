// -*- coding: utf-8 -*-
#pragma once

#include "ell_calc.hpp"
#include "ell_matrix.hpp"
#include <cmath>
#include <tuple>
#include <valarray>

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        ell_ss {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllCore {
public:
  using Vec = std::valarray<double>;

  // using params_t = std::tuple<double, double, double>;
  // using return_t = std::tuple<int, params_t>;

  bool no_defer_trick = false;

private:
  const size_t _n;
  EllCalc _helper;
  double _kappa;
  Matrix _Q;

  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] E
   */
  auto operator=(const EllCore &E) -> EllCore & = delete;

  /**
   * @brief Construct a new EllCore object
   *
   * @tparam V
   * @tparam U
   * @param kappa
   * @param Q
   * @param x
   */
  template <typename V>
  EllCore(V &&kappa, Matrix &&Q, size_t dim) noexcept
      : _n{dim}, _helper{double(_n)}, _kappa{std::forward<V>(kappa)},
        _Q{std::move(Q)} {}

public:
  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] val
   * @param[in] x
   */
  EllCore(const Vec &val, size_t dim) : EllCore{1.0, Matrix(dim), dim} {
    this->_Q.diagonal() = val;
  }

  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] alpha
   * @param[in] x
   */
  EllCore(const double &alpha, size_t dim) : EllCore{alpha, Matrix(dim), dim} {
    this->_Q.identity();
  }

  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] E (move)
   */
  EllCore(EllCore &&E) = default;

  /**
   * @brief Destroy the EllCore object
   *
   */
  ~EllCore() {}

  /**
   * @brief Construct a new EllCore object
   *
   * To avoid accidentally copying, only explicit copy is allowed
   *
   * @param E
   */
  explicit EllCore(const EllCore &E) = default;

  /**
   * @brief explicitly copy
   *
   * @return EllCore
   */
  auto copy() const -> EllCore { return EllCore(*this); }

  /**
   * @brief copy the whole array anyway
   *
   * @return Vec
   */

  /**
   * @brief
   *
   * @return Arr
   */
  auto tsq() const -> double { return this->_helper._tsq; }

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
  template <typename T> auto update(Vec &grad, const T &beta) -> CutStatus;

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_stable(Vec &grad, const T &beta) -> CutStatus;

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T> auto update_cc(Vec &grad, const T &beta) -> CutStatus;

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_stable_cc(Vec &grad, const T &beta) -> CutStatus;

private:
  auto _update_cut(const double &beta) -> CutStatus {
    return this->_helper._calc_dc(beta);
  }

  auto _update_cut(const std::valarray<double> &beta)
      -> CutStatus { // parallel cut
    if (beta.size() < 2) {
      return this->_helper._calc_dc(beta[0]);
    }
    return this->_helper._calc_ll_core(beta[0], beta[1]);
  }
}; // } EllCore
