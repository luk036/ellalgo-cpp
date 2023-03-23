// -*- coding: utf-8 -*-
#pragma once

#include <cmath>
#include <tuple>
#include <valarray>

#include "ell_config.hpp"
#include "ell_core.hpp"
#include "ell_matrix.hpp"

// forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *        ell_ss {x | (x - xc)' Q^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
template <typename Arr> class EllStable {
public:
  // bool no_defer_trick = false;
  using Vec = std::valarray<double>;
  using ArrayType = Arr;

private:
  const size_t _n;
  Arr _xc;
  EllCore _mgr;

  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] E
   */
  auto operator=(const EllStable &E) -> EllStable & = delete;

public:
  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] val
   * @param[in] x
   */
  EllStable(const Vec &val, Arr x)
      : _n{x.size()}, _xc{std::move(x)}, _mgr(val, _n) {}

  /**
   * @brief Construct a new EllStable object
   *
   * @param[in] alpha
   * @param[in] x
   */
  EllStable(const double &alpha, Arr x)
      : _n{x.size()}, _xc{std::move(x)}, _mgr(alpha, _n) {}

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
    this->_mgr.set_use_parallel_cut(value);
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update(const std::pair<Arr, T> &cut) -> std::tuple<CutStatus, double> {
    const auto &grad = cut.first;
    const auto &beta = cut.second;
    std::valarray<double> g(this->_n);
    for (auto i = 0U; i != this->_n; ++i) {
      g[i] = grad[i];
    }

    auto result = this->_mgr.update_stable(g, beta);
    if (std::get<0>(result) == CutStatus::Success) {
      for (auto i = 0U; i != this->_n; ++i) {
        this->_xc[i] -= g[i];
      }
    }

    return result;
  }
}; // } EllStable
