// -*- coding: utf-8 -*-
#pragma once

#include <cassert>
#include <cmath>
#include <tuple>

#include "cut_config.hpp"
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
 * @param[in,out] S     search Space containing x*
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space>
auto cutting_plane_feas(Oracle&& omega, Space&& S, const Options& options = Options()) -> CInfo {
    auto feasible = false;
    auto status = CUTStatus::nosoln;

    auto niter = 0U;
    while (++niter != options.max_it) {
        const auto cut = omega(S.xc());  // query the oracle at S.xc()
        if (!cut) {                      // feasible sol'n obtained
            feasible = true;
            break;
        }
        const auto result = S.update(*cut);  // update S

        const auto& cutstatus = std::get<0>(result);
        const auto& tsq = std::get<1>(result);
        if (cutstatus != CUTStatus::success) {
            status = cutstatus;
            break;
        }
        if (tsq < options.tol) {  // no more
            status = CUTStatus::smallenough;
            break;
        }
    }
    return {feasible, niter, status};
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam opt_type
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] S     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space, typename opt_type>
auto cutting_plane_dc(Oracle&& omega, Space&& S, opt_type&& t, const Options& options = Options()) {
    const auto t_orig = t;
    decltype(S.xc()) x_best;
    auto status = CUTStatus::success;

    auto niter = 0U;
    while (++niter != options.max_it) {
        const auto result1 = omega(S.xc(), t);
        const auto& cut = std::get<0>(result1);
        const auto& shrunk = std::get<1>(result1);
        if (shrunk) {  // best t obtained
            x_best = S.xc();
        }
        const auto result2 = S.update(cut);

        const auto& cutstatus = std::get<0>(result2);
        const auto& tsq = std::get<1>(result2);
        if (cutstatus != CUTStatus::success)  // ???
        {
            status = cutstatus;
            break;
        }
        if (tsq < options.tol) {  // no more
            status = CUTStatus::smallenough;
            break;
        }
    }
    return std::make_tuple(std::move(x_best), CInfo{t != t_orig, niter, status});
}  // END

/**
    Cutting-plane method for solving convex discrete optimization problem
    input
             oracle        perform assessment on x0
             S(xc)         Search space containing x*
             t             best-so-far optimal sol'n
             max_it        maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter          number of iterations performed
**/
// #include <boost/numeric/ublas/symmetric.hpp>
// namespace bnu = boost::numeric::ublas;
// #include <xtensor-blas/xlinalg.hpp>
// #include <xtensor/xarray.hpp>

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] S     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space, typename opt_type>
auto cutting_plane_q(Oracle&& omega, Space&& S, opt_type&& t, const Options& options = Options()) {
    const auto t_orig = t;
    decltype(S.xc()) x_best;
    auto status = CUTStatus::nosoln;  // note!!!
    auto retry = (status == CUTStatus::noeffect);

    auto niter = 0U;
    while (++niter != options.max_it) {
        // auto retry = (status == CUTStatus::noeffect);
        const auto result1 = omega(S.xc(), t, retry);
        const auto& cut = std::get<0>(result1);
        const auto& shrunk = std::get<1>(result1);
        const auto& x0 = std::get<2>(result1);
        const auto& more_alt = std::get<3>(result1);
        if (shrunk) {  // best t obtained
            // t = t1;
            x_best = x0;  // x0
        }
        const auto result2 = S.update(cut);
        const auto& cutstatus = std::get<0>(result2);
        const auto& tsq = std::get<1>(result2);

        if (cutstatus == CUTStatus::noeffect) {
            if (!more_alt) {  // more alt?
                break;        // no more alternative cut
            }
            status = cutstatus;
            retry = true;
        }
        if (cutstatus == CUTStatus::nosoln) {
            status = cutstatus;
            break;
        }
        if (tsq < options.tol) {
            status = CUTStatus::smallenough;
            break;
        }
    }
    return std::make_tuple(std::move(x_best), CInfo{t != t_orig, niter, status});
}  // END

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
auto bsearch(Oracle&& omega, Space&& I, const Options& options = Options()) -> CInfo {
    // assume monotone
    // auto& [lower, upper] = I;
    auto& lower = I.first;
    auto& upper = I.second;
    assert(lower <= upper);
    const auto u_orig = upper;
    auto niter = 0U;
    auto status = CUTStatus::success;

    for (; niter != options.max_it; ++niter) {
        auto tau = algo::half_nonnegative(upper - lower);
        if (tau < options.tol) {
            status = CUTStatus::smallenough;
            break;
        }

        auto t = lower;  // l may be `int` or `Fraction`
        t += tau;
        if (omega(t)) {  // feasible sol'n obtained
            upper = t;
        } else {
            lower = t;
        }
    }
    return {upper != u_orig, niter + 1, status};
}

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 */
template <typename Oracle, typename Space>  //
class bsearch_adaptor {
  private:
    Oracle& _P;
    Space& _S;
    const Options _options;

  public:
    /**
     * @brief Construct a new bsearch adaptor object
     *
     * @param[in,out] P perform assessment on x0
     * @param[in,out] S search Space containing x*
     */
    bsearch_adaptor(Oracle& P, Space& S) : bsearch_adaptor{P, S, Options()} {}

    /**
     * @brief Construct a new bsearch adaptor object
     *
     * @param[in,out] P perform assessment on x0
     * @param[in,out] S search Space containing x*
     * @param[in] options maximum iteration and error tolerance etc.
     */
    bsearch_adaptor(Oracle& P, Space& S, const Options& options)
        : _P{P}, _S{S}, _options{options} {}

    /**
     * @brief get best x
     *
     * @return auto
     */
    auto x_best() const { return this->_S.xc(); }

    /**
     * @brief
     *
     * @param[in,out] t the best-so-far optimal value
     * @return bool
     */
    template <typename opt_type> auto operator()(const opt_type& t) -> bool {
        Space S = this->_S.copy();
        this->_P.update(t);
        const auto ell_info = cutting_plane_feas(this->_P, S, this->_options);
        if (ell_info.feasible) {
            this->_S.set_xc(S.xc());
        }
        return ell_info.feasible;
    }
};
