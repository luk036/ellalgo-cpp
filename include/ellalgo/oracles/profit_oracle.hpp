// -*- coding: utf-8 -*-
#pragma once

#include <cmath> // for log
#include <tuple> // for tuple
#include <valarray>
// #include <xtensor/xaccessible.hpp>     // for xconst_accessible, xaccessible
// #include <xtensor/xarray.hpp>          // for xarray_container
// #include <xtensor/xlayout.hpp>         // for layout_type,
// layout_type::row... #include <xtensor/xoperation.hpp>      // for
// xfunction_type_t, operator+ #include <xtensor/xtensor_forward.hpp> // for
// xarray

/**
 * @brief Oracle for a profit maximization problem.
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *        max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *        s.t.    x1 \le k
 *
 *    where:
 *
 *        p(A x1^alpha x2^beta): Cobb-Douglas production function
 *        p: the market price per unit
 *        A: the scale of production
 *        alpha, beta: the output elasticities
 *        x: input quantity
 *        v: output price
 *        k: a given constant that restricts the quantity of x1
 */
class ProfitOracle {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;
  using Cut = std::pair<Vec, double>;

private:
  const double _log_pA;
  const double _log_k;
  const Vec _v;

public:
  Vec _a;

  /**
   * @brief Construct a new profit oracle object
   *
   * @param[in] p the market price per unit
   * @param[in] A the scale of production
   * @param[in] k a given constant that restricts the quantity of x1
   * @param[in] a the output elasticities
   * @param[in] v output price
   */
  ProfitOracle(double p, double A, double k, const Vec &a, const Vec &v)
      : _log_pA{std::log(p * A)}, _log_k{std::log(k)}, _v{v}, _a{a} {}

  /**
   * @brief Construct a new profit oracle object (only explicitly)
   *
   */
  ProfitOracle(const ProfitOracle &) = delete;

  /**
   * @brief
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @return std::tuple<Cut, double> Cut and the updated best-so-far value
   */
  auto assess_optim(const Vec &y, double &t) const -> std::tuple<Cut, bool>;

  /**
   * @brief
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @return std::tuple<Cut, double> Cut and the updated best-so-far value
   */
  auto operator()(const Vec &y, double &t) const -> std::tuple<Cut, bool> {
    return this->assess_optim(y, t);
  }
};

/**
 * @brief Oracle for a profit maximization problem (robust version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *        max     p'(A x1^alpha' x2^beta') - v1'*x1 - v2'*x2
 *        s.t.    x1 \le k'
 *
 *    where:
 *        alpha' = alpha \pm e1
 *        beta' = beta \pm e2
 *        p' = p \pm e3
 *        k' = k \pm e4
 *        v' = v \pm e5
 *
 * @see ProfitOracle
 */
class ProfitOracleRb {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;
  using Cut = std::pair<Vec, double>;

private:
  const Vec _uie;
  Vec _a;
  ProfitOracle _P;

public:
  /**
   * @brief Construct a new profit rb oracle object
   *
   * @param[in] p the market price per unit
   * @param[in] A the scale of production
   * @param[in] k a given constant that restricts the quantity of x1
   * @param[in] a the output elasticities
   * @param[in] v output price
   * @param[in] e paramters for uncertainty
   * @param[in] e3 paramters for uncertainty
   */
  ProfitOracleRb(double p, double A, double k, const Vec &a, const Vec &v,
                   const Vec &e, double e3)
      : _uie{e}, _a{a}, _P(p - e3, A, k - e3, a, v + e3) {}

  /**
   * @brief Make object callable for cutting_plane_optim()
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @return Cut and the updated best-so-far value
   *
   * @see cutting_plane_optim
   */
  auto assess_optim(const Vec &y, double &t) -> std::tuple<Cut, bool> {
    auto a_rb = this->_a;
    a_rb[0] += y[0] > 0.0 ? -this->_uie[0] : this->_uie[0];
    a_rb[1] += y[1] > 0.0 ? -this->_uie[1] : this->_uie[1];
    this->_P._a = a_rb;
    return this->_P(y, t);
  }

  /**
   * @brief Make object callable for cutting_plane_optim()
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @return Cut and the updated best-so-far value
   *
   * @see cutting_plane_optim
   */
  auto operator()(const Vec &y, double &t) -> std::tuple<Cut, bool> {
    return this->assess_optim(y, t);
  }
};

/**
 * @brief Oracle for profit maximization problem (discrete version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]
 *
 *        max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *        s.t.    x1 \le k
 *
 *    where:
 *
 *        p(A x1^alpha x2^beta): Cobb-Douglas production function
 *        p: the market price per unit
 *        A: the scale of production
 *        alpha, beta: the output elasticities
 *        x: input quantity (must be integer value)
 *        v: output price
 *        k: a given constant that restricts the quantity of x1
 *
 * @see ProfitOracle
 */
class ProfitOracleQ {
  // using Arr = xt::xarray<double, xt::layout_type::row_major>;
  using Vec = std::valarray<double>;
  using Cut = std::pair<Vec, double>;

private:
  ProfitOracle _P;
  Vec _yd;

public:
  /**
   * @brief Construct a new profit q oracle object
   *
   * @param[in] p the market price per unit
   * @param[in] A the scale of production
   * @param[in] k a given constant that restricts the quantity of x1
   * @param[in] a the output elasticities
   * @param[in] v output price
   */
  ProfitOracleQ(double p, double A, double k, const Vec &a, const Vec &v)
      : _P{p, A, k, a, v} {}

  /**
   * @brief Make object callable for cutting_plane_q()
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @param[in] retry whether it is a retry
   * @return Cut and the updated best-so-far value
   *
   * @see cutting_plane_q
   */
  auto assess_q(const Vec &y, double &t, bool retry)
      -> std::tuple<Cut, bool, Vec, bool>;

  /**
   * @brief Make object callable for cutting_plane_q()
   *
   * @param[in] y input quantity (in log scale)
   * @param[in,out] t the best-so-far optimal value
   * @param[in] retry whether it is a retry
   * @return Cut and the updated best-so-far value
   *
   * @see cutting_plane_q
   */
  auto operator()(const Vec &y, double &t, bool retry)
      -> std::tuple<Cut, bool, Vec, bool> {
    return this->assess_q(y, t, retry);
  }
};
