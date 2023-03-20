// -*- coding: utf-8 -*-
#pragma once

#include <cassert>
#include <cmath>
#include <tuple>

#include "ell_config.hpp"
#include "half_nonnegative.hpp"

/**
 * @brief Find a point in a convex set (defined through a cutting-plane oracle).
 *
 *     A function f(x) is *convex* if there always exist a g(x)
 *     such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f.
 *     Note that dom f does not need to be a convex set in our definition.
 *     The affine function g' (x - xc) + beta is called a cutting-plane,
 *     or a ``cut'' for short.
 *     This algorithm solves the following feasibility problem:
 *
 *             find x
 *             s.t. f(x) <= 0,
 *
 *     A *separation oracle* asserts that an evalution point x0 is feasible,
 *     or provide a cut that separates the feasible region and x0.
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space>
auto cutting_plane_feas(Oracle &&omega, Space &&ss,
                        const Options &options = Options()) -> CInfo {
  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto cut = omega.assess_feas(ss.xc()); // query the oracle at ss.xc()
    if (!cut) {                                  // feasible sol'n obtained
      return {true, niter, CutStatus::Success};
    }
    const auto result = ss.update(*cut); // update ss

    const auto &cutstatus = std::get<0>(result);
    const auto &tsq = std::get<1>(result);
    if (cutstatus != CutStatus::Success) {
      return {false, niter, cutstatus};
    }
    if (tsq < options.tol) { // no more
      return {false, niter, CutStatus::SmallEnough};
    }
  }
  return {false, options.max_iter, CutStatus::NoSoln};
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam opt_type
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space, typename opt_type>
auto cutting_plane_optim(Oracle &&omega, Space &&ss, opt_type &&t,
                         const Options &options = Options()) {
  const auto t_orig = t;
  decltype(ss.xc()) x_best{};
  auto status = CutStatus::Success;

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto result1 = omega.assess_optim(ss.xc(), t);
    const auto &cut = std::get<0>(result1);
    const auto &shrunk = std::get<1>(result1);
    if (shrunk) { // best t obtained
      x_best = ss.xc();
    }
    const auto result2 = ss.update(cut);

    const auto &cutstatus = std::get<0>(result2);
    const auto &tsq = std::get<1>(result2);
    if (cutstatus != CutStatus::Success) // ???
    {
      return std::make_tuple(std::move(x_best),
                             CInfo{t != t_orig, niter, cutstatus});
    }
    if (tsq < options.tol) { // no more
      return std::make_tuple(std::move(x_best),
                             CInfo{t != t_orig, niter, CutStatus::SmallEnough});
    }
  }
  return std::make_tuple(std::move(x_best),
                         CInfo{t != t_orig, options.max_iter, status});
} // END

/**
    Cutting-plane method for solving convex discrete optimization problem
    input
             oracle        perform assessment on x0
             ss(xc)         Search space containing x*
             t             best-so-far optimal sol'n
             max_iter        maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter          number of iterations performed
**/

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space, typename opt_type>
auto cutting_plane_q(Oracle &&omega, Space &&ss, opt_type &&t,
                     const Options &options = Options()) {
  const auto t_orig = t;
  decltype(ss.xc()) x_best{};
  auto status = CutStatus::NoSoln; // note!!!
  auto retry = (status == CutStatus::NoEffect);

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    // auto retry = (status == CutStatus::NoEffect);
    const auto result1 = omega.assess_q(ss.xc(), t, retry);
    const auto &cut = std::get<0>(result1);
    const auto &shrunk = std::get<1>(result1);
    const auto &x0 = std::get<2>(result1);
    const auto &more_alt = std::get<3>(result1);
    if (shrunk) { // best t obtained
      // t = t1;
      x_best = x0; // x0
    }
    const auto result2 = ss.update(cut);
    const auto &cutstatus = std::get<0>(result2);
    const auto &tsq = std::get<1>(result2);

    if (cutstatus == CutStatus::NoEffect) {
      if (!more_alt) { // more alt?
        break;         // no more alternative cut
      }
      status = cutstatus;
      retry = true;
    }
    if (cutstatus == CutStatus::NoSoln) {
      return std::make_tuple(std::move(x_best),
                             CInfo{t != t_orig, niter, cutstatus});
    }
    if (tsq < options.tol) { // no more
      return std::make_tuple(std::move(x_best),
                             CInfo{t != t_orig, niter, CutStatus::SmallEnough});
    }
  }
  return std::make_tuple(std::move(x_best),
                         CInfo{t != t_orig, options.max_iter, status});
} // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega    perform assessment on x0
 * @param[in,out] I        interval containing x*
 * @param[in]     options  maximum iteration and error tolerance etc.
 * @return CInfo
 */
template <typename Oracle, typename Space>
auto bsearch(Oracle &&omega, Space &&I, const Options &options = Options())
    -> CInfo {
  // assume monotone
  // auto& [lower, upper] = I;
  auto &lower = I.first;
  auto &upper = I.second;
  assert(lower <= upper);
  const auto u_orig = upper;
  auto status = CutStatus::Success;

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    auto tau = algo::half_nonnegative(upper - lower);
    if (tau < options.tol) { // no more
      return {upper != u_orig, niter, CutStatus::SmallEnough};
    }

    auto t = lower; // l may be `int` or `Fraction`
    t += tau;
    if (omega.assess_bs(t)) { // feasible sol'n obtained
      upper = t;
    } else {
      lower = t;
    }
  }
  return {upper != u_orig, options.max_iter, status};
}

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 */
template <typename Oracle, typename Space> //
class bsearch_adaptor {
private:
  Oracle &_omega;
  Space &_ss;
  const Options _options;

public:
  /**
   * @brief Construct a new bsearch adaptor object
   *
   * @param[in,out] omega perform assessment on x0
   * @param[in,out] ss search Space containing x*
   */
  bsearch_adaptor(Oracle &omega, Space &ss)
      : bsearch_adaptor{omega, ss, Options()} {}

  /**
   * @brief Construct a new bsearch adaptor object
   *
   * @param[in,out] omega perform assessment on x0
   * @param[in,out] ss search Space containing x*
   * @param[in] options maximum iteration and error tolerance etc.
   */
  bsearch_adaptor(Oracle &omega, Space &ss, const Options &options)
      : _omega{omega}, _ss{ss}, _options{options} {}

  /**
   * @brief get best x
   *
   * @return auto
   */
  auto x_best() const { return this->_ss.xc(); }

  /**
   * @brief
   *
   * @tparam BestSoFar Could be integer or floating point
   * @param[in,out] t the best-so-far optimal value
   * @return bool
   */
  template <typename BestSoFar> auto assess_bs(const BestSoFar &t) -> bool {
    Space ss = this->_ss.copy();
    this->_omega.update(t);
    const auto ell_info = cutting_plane_feas(this->_omega, ss, this->_options);
    if (ell_info.feasible) {
      this->_ss.set_xc(ss.xc());
    }
    return ell_info.feasible;
  }
};
