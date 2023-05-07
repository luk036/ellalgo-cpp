// -*- coding: utf-8 -*-
#pragma once

#include "ell_calc.hpp"
#include "ell_config.hpp"
#include "ell_matrix.hpp"
#include <cmath>
#include <tuple>
#include <valarray>

/**
 * @brief Ellipsoid Search Space Core
 *
 * Without the knowledge of the type of xc
 *
 *  \mathcal{E} {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 */
template <typename Calc = EllCalc> class EllCore {
public:
  using Vec = std::valarray<double>;
  bool no_defer_trick = false;

private:
  size_t _n;
  double _kappa;
  Matrix _mq;
  Calc _helper;

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
   * @param mq
   * @param x
   */
  EllCore(double &&kappa, Matrix &&mq, size_t ndim)
      : _n{ndim}, _kappa{std::move(kappa)}, _mq{std::move(mq)}, _helper{_n} {}

public:
  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] val
   * @param[in] x
   */
  EllCore(const Vec &val, size_t ndim) : EllCore{1.0, Matrix(ndim), ndim} {
    this->_mq.diagonal() = val;
  }

  /**
   * @brief Construct a new EllCore object
   *
   * @param[in] alpha
   * @param[in] x
   */
  EllCore(double alpha, size_t ndim)
      : EllCore{std::move(alpha), Matrix(ndim), ndim} {
    this->_mq.identity();
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
  template <typename T> auto update(Vec &grad, const T &beta) -> CutStatus {
    return this->_update_single_or_ll(
        grad, beta, [&](const T &beta) { return this->_update_cut(beta); });
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T> auto update_cc(Vec &grad, const T &beta) -> CutStatus {
    return this->_update_single_or_ll(
        grad, beta, [&](const T &beta) { return this->_update_cut_cc(beta); });
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_stable(Vec &grad, const T &beta) -> CutStatus {
    return this->_update_stable_single_or_ll(
        grad, beta, [&](const T &beta) { return this->_update_cut(beta); });
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_stable_cc(Vec &grad, const T &beta) -> CutStatus {
    return this->_update_stable_single_or_ll(
        grad, beta, [&](const T &beta) { return this->_update_cut_cc(beta); });
  }

private:
  template <typename T, typename Fn>
  auto _update_single_or_ll(Vec &grad, const T &beta, Fn &&f_core)
      -> CutStatus {
    std::valarray<double> Qg(0.0, this->_n);
    auto omega = 0.0;
    for (auto i = 0U; i != this->_n; ++i) {
      for (auto j = 0U; j != this->_n; ++j) {
        Qg[i] += this->_mq(i, j) * grad[j];
      }
      omega += Qg[i] * grad[i];
    }

    this->_helper._tsq = this->_kappa * omega;

    auto status = f_core(beta);

    if (status != CutStatus::Success) {
      return status;
    }

    // n*(n+1)/2 + n
    // this->_mq -= (this->_sigma / omega) * xt::linalg::outer(Qg, Qg);
    const auto r = this->_helper._sigma / omega;
    for (auto i = 0U; i != this->_n; ++i) {
      const auto rQg = r * Qg[i];
      for (auto j = 0U; j != i; ++j) {
        this->_mq(i, j) -= rQg * Qg[j];
        this->_mq(j, i) = this->_mq(i, j);
      }
      this->_mq(i, i) -= rQg * Qg[i];
    }

    this->_kappa *= this->_helper._delta;

    if (this->no_defer_trick) {
      this->_mq *= this->_kappa;
      this->_kappa = 1.0;
    }

    grad = Qg * (this->_helper._rho / omega);
    return status; // g++-7 is ok
  }

  template <typename T, typename Fn>
  auto _update_stable_single_or_ll(Vec &g, const T &beta, Fn &&f_core)
      -> CutStatus {
    // calculate inv(L)*grad: (n-1)*n/2 multiplications
    auto invLg{g}; // initially
    for (auto i = 1U; i != this->_n; ++i) {
      for (auto j = 0U; j != i; ++j) {
        this->_mq(i, j) = this->_mq(j, i) * invLg[j];
        // keep for rank-one update
        invLg[i] -= this->_mq(i, j);
      }
    }

    // calculate inv(D)*inv(L)*grad: n
    auto invDinvLg{invLg}; // initially
    for (auto i = 0U; i != this->_n; ++i) {
      invDinvLg[i] *= this->_mq(i, i);
    }

    // calculate omega: n
    auto gQg{invDinvLg}; // initially
    auto omega = 0.0;    // initially
    for (auto i = 0U; i != this->_n; ++i) {
      gQg[i] *= invLg[i];
      omega += gQg[i];
    }

    this->_helper._tsq = this->_kappa * omega;

    auto status = f_core(beta);

    if (status != CutStatus::Success) {
      return status;
    }

    // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
    auto Qg{invDinvLg};                        // initially
    for (auto i = this->_n - 1; i != 0; --i) { // backward subsituition
      for (auto j = i; j != this->_n; ++j) {
        Qg[i - 1] -= this->_mq(i, j) * Qg[j]; // ???
      }
    }

    // rank-one update: 3*n + (n-1)*n/2
    // const auto r = this->_sigma / omega;
    const auto mu = this->_helper._sigma / (1.0 - this->_helper._sigma);
    auto oldt = omega / mu; // initially
    const auto m = this->_n - 1;
    for (auto j = 0U; j != m; ++j) {
      const auto t = oldt + gQg[j];
      const auto beta2 = invDinvLg[j] / t;
      this->_mq(j, j) *= oldt / t; // update invD
      for (auto l = j + 1; l != this->_n; ++l) {
        this->_mq(j, l) += beta2 * this->_mq(l, j);
      }
      oldt = t;
    }

    const auto t = oldt + gQg[m];
    this->_mq(m, m) *= oldt / t; // update invD
    this->_kappa *= this->_helper._delta;

    // calculate xc: n
    g = Qg * (this->_helper._rho / omega);
    return status;
  }

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

  auto _update_cut_cc(const double &) -> CutStatus {
    return this->_helper._calc_cc();
  }

  auto _update_cut_cc(const std::valarray<double> &beta)
      -> CutStatus { // parallel cut
    if (beta.size() < 2) {
      return this->_helper._calc_cc();
    }
    return this->_helper._calc_ll_cc(beta[1]);
  }
}; // } EllCore
