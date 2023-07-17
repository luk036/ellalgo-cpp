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
 * The `Ell` class represents an ellipsoid search space:
 *
 *        ell = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * It is used to define and manipulate ellipsoids in a multidimensional space.
 * The ellipsoid is defined by a center point (`_xc`) and a core function
 * (`_mgr`). The core function is responsible for updating the ellipsoid based
 * on cutting planes. The `Ell` class provides methods to update the ellipsoid
 * using different types of cutting planes and to retrieve information about the
 * ellipsoid, such as the center point and the squared radius.
 *
 * This version keeps $Q$ symmetric but no promise of positive definite
 */
template <typename Arr> class Ell {
public:
  // bool no_defer_trick = false;
  using Vec = std::valarray<double>;
  using ArrayType = Arr;

private:
  const size_t _n;
  Arr _xc;
  EllCore _mgr;

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] E
   */
  auto operator=(const Ell &E) -> Ell & = delete;

public:
  /**
   * @brief Construct a new Ell object
   *
   * @param[in] val
   * @param[in] x
   */
  Ell(const Vec &val, Arr x) : _n{x.size()}, _xc{std::move(x)}, _mgr(val, _n) {}

  /**
   * @brief Construct a new Ell object
   *
   * @param[in] alpha
   * @param[in] x
   */
  Ell(const double &alpha, Arr x)
      : _n{x.size()}, _xc{std::move(x)}, _mgr(alpha, _n) {}

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
  auto xc() const -> Arr { return this->_xc; }

  /**
   * @brief Set the xc object
   *
   * @param[in] xc
   */
  void set_xc(const Arr &xc) { this->_xc = xc; }

  /**
   * @brief
   *
   * @return double
   */
  auto tsq() const -> double { return this->_mgr.tsq(); }

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
  auto update_dc(const std::pair<Arr, T> &cut) -> CutStatus {
    return this->_update_core(cut, [&](Vec &grad, const T &beta) {
      return this->_mgr.update_dc(grad, beta);
    });
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_cc(const std::pair<Arr, T> &cut) -> CutStatus {
    return this->_update_core(cut, [&](Vec &grad, const T &beta) {
      return this->_mgr.update_cc(grad, beta);
    });
  }

  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T>
  auto update_q(const std::pair<Arr, T> &cut) -> CutStatus {
    return this->_update_core(cut, [&](Vec &grad, const T &beta) {
      return this->_mgr.update_q(grad, beta);
    });
  }

private:
  /**
   * @brief Update ellipsoid core function using the cut(s)
   *
   * @tparam T
   * @param[in] cut cutting-plane
   * @return std::tuple<int, double>
   */
  template <typename T, typename Fn>
  auto _update_core(const std::pair<Arr, T> &cut, Fn &&cut_strategy)
      -> CutStatus {
    const auto &grad = cut.first;
    const auto &beta = cut.second;
    std::valarray<double> g(this->_n);
    for (auto i = 0U; i != this->_n; ++i) {
      g[i] = grad[i];
    }

    auto result = cut_strategy(g, beta);

    if (result == CutStatus::Success) {
      for (auto i = 0U; i != this->_n; ++i) {
        this->_xc[i] -= g[i];
      }
    }

    return result;
  }
}; // } Ell
