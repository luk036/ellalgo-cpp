#pragma once

#include <cassert>
#include <cmath>
#include <tuple>
#include <type_traits>

#include "ell_config.hpp"
#include "half_nonnegative.hpp"

template <typename SearchSpace> using CuttingPlaneArrayType =
    typename std::remove_reference<SearchSpace>::type::ArrayType;

template <typename T> inline auto invalid_value() ->
    typename std::enable_if<std::is_floating_point<T>::value, T>::type {
    return std::nan("1");
}

template <typename T> inline auto invalid_value() ->
    typename std::enable_if<!std::is_floating_point<T>::value, T>::type {
    return T{};
}

/**
 * @brief Find a point in a convex set (defined through a cutting-plane oracle).
 *
 * The `cutting_plane_feas` function implements the cutting-plane method for
 * solving a convex feasibility problem:
 *
 *   find x
 *   s.t. f(x) <= 0,
 *
 * It takes a cutting-plane oracle `omega`, a search space `space`, and an
 * options object as input. A function f(x) is *convex* if there always exist a
 * g(x) such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f. Note
 * that dom f does not need to be a convex set in our definition. The affine
 * function g' (x - xc) + beta is called a cutting-plane, or a ``cut'' for
 * short.
 *
 * A *separation oracle* asserts that an evalution point xc is feasible,
 * or provide a cut that separates the feasible region and xc.
 *
 * @tparam OracleFeas
 * @tparam SearchSpace
 * @param[in,out] omega   perform assessment on x0
 * @param[in,out] space   search Space containing x*
 * @param[in]     options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename OracleFeas, typename SearchSpace>
inline auto cutting_plane_feas(OracleFeas &omega, SearchSpace &space,
                               const Options &options = Options())
    -> std::tuple<CuttingPlaneArrayType<SearchSpace>, size_t> {
    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        const auto cut = omega.assess_feas(space.xc());
        if (!cut) {  // feasible sol'n obtained
            return {space.xc(), niter};
        }
        const auto status = space.update_bias_cut(*cut);  // update space
        if (status != CutStatus::Success || space.tsq() < options.tolerance) {
            auto res = invalid_value<CuttingPlaneArrayType<SearchSpace>>();
            return {std::move(res), niter};
        }
    }
    auto res = invalid_value<CuttingPlaneArrayType<SearchSpace>>();
    return {std::move(res), options.max_iters};
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * The `cutting_plane_optim` function implements the cutting-plane method for
 * solving a convex optimization problem:
 *
 *   min  gamma
 *   s.t. f(x, gamma) <= 0, x \in R
 *
 * It takes a cutting-plane oracle `omega`, a search space `space`, and an
 * options object as input. A function f(x) is *convex* if there always exist a
 * g(x) such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f. Note
 * that dom f does not need to be a convex set in our definition. The affine
 * function g' (x - xc) + beta is called a cutting-plane, or a ``cut'' for
 * short.
 *
 * @tparam OracleOptim
 * @tparam SearchSpace
 * @tparam Num
 * @param[in,out] omega   perform assessment on x0
 * @param[in,out] space   search Space containing x*
 * @param[in,out] gamma   best-so-far optimal sol'n
 * @param[in]     options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename OracleOptim, typename SearchSpace, typename Num>
inline auto cutting_plane_optim(OracleOptim &omega, SearchSpace &space, Num &gamma,
                                const Options &options = Options())
    -> std::tuple<CuttingPlaneArrayType<SearchSpace>, size_t> {
    auto x_best = invalid_value<CuttingPlaneArrayType<SearchSpace>>();
    for (auto niter = 0U; niter < options.max_iters; ++niter) {
        const auto __result1 = omega.assess_optim(space.xc(), gamma);
        const auto &cut = std::get<0>(__result1);
        const auto &shrunk = std::get<1>(__result1);
        const auto status = [&]() {
            if (shrunk) {  // best gamma obtained
                x_best = space.xc();
                return space.update_central_cut(cut);  // should update_central_cut
            } else {
                return space.update_bias_cut(cut);
            }
        }();
        if (status != CutStatus::Success || space.tsq() < options.tolerance) {  // no more
            return {std::move(x_best), niter};
        }
    }
    return {std::move(x_best), options.max_iters};
}  // END

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * The `cutting_plane_optim_q` function implements the cutting-plane method for
 * solving a discrete convex optimization problem:
 *
 *   min  gamma
 *   s.t. f(x, gamma) <= 0, x \in D
 *
 * It takes a cutting-plane oracle `omega`, a search space `space`, and an
 * options object as input. A function f(x) is *convex* if there always exist a
 * g(x) such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f. Note
 * that dom f does not need to be a convex set in our definition. The affine
 * function g' (x - xc) + beta is called a cutting-plane, or a ``cut'' for
 * short.
 *
 * @tparam OracleOptimQ
 * @tparam SearchSpaceQ
 * @tparam Num
 * @param[in,out] omega   perform assessment on x0
 * @param[in,out] space_q   search Space containing x*
 * @param[in,out] gamma   best-so-far optimal sol'n
 * @param[in]     options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename OracleOptimQ, typename SearchSpaceQ, typename Num>
inline auto cutting_plane_optim_q(OracleOptimQ &omega, SearchSpaceQ &space_q, Num &gamma,
                                  const Options &options = Options())
    -> std::tuple<CuttingPlaneArrayType<SearchSpaceQ>, size_t> {
    auto x_best = invalid_value<CuttingPlaneArrayType<SearchSpaceQ>>();
    auto retry = false;

    for (auto niter = 0U; niter < options.max_iters; ++niter) {
        const auto result1 = omega.assess_optim_q(space_q.xc(), gamma, retry);
        const auto &cut = std::get<0>(result1);
        const auto &shrunk = std::get<1>(result1);
        if (shrunk) {  // best gamma obtained
            auto x_q = std::get<2>(result1);
            x_best = std::move(x_q);
            retry = false;
        }
        auto status = space_q.update_q(cut);
        if (status == CutStatus::Success) {
            retry = false;
        } else if (status == CutStatus::NoSoln) {
            return {std::move(x_best), niter};
        } else if (status == CutStatus::NoEffect) {
            const auto &more_alt = std::get<3>(result1);
            if (!more_alt) {  // more alt?
                break;        // no more alternative cut
            }
            retry = true;
        }
        if (space_q.tsq() < options.tolerance) {  // no more
            return {std::move(x_best), niter};
        }
    }
    return {std::move(x_best), options.max_iters};
}  // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 */
template <typename Oracle, typename Space>  //
class BSearchAdaptor {
    using ArrayType = typename Space::ArrayType;

    Oracle &_omega;
    Space &_space;
    const Options _options;

  public:
    /**
     * @brief Construct a new bsearch adaptor object
     *
     * @param[in,out] omega perform assessment on x0
     * @param[in,out] space search Space containing x*
     */
    BSearchAdaptor(Oracle &omega, Space &space) : BSearchAdaptor{omega, space, Options()} {}

    /**
     * @brief Construct a new bsearch adaptor object
     *
     * @param[in,out] omega   perform assessment on x0
     * @param[in,out] space   search space containing x*
     * @param[in]     options maximum iteration and error tolerance etc.
     */
    BSearchAdaptor(Oracle &omega, Space &space, const Options &options)
        : _omega{omega}, _space{space}, _options{options} {}

    /**
     * @brief get best x
     *
     * @return auto
     */
    auto x_best() const -> ArrayType { return this->_space.xc(); }

    /**
     * @brief
     *
     * @tparam Num
     * @param[in,out] gamma the best-so-far optimal value
     * @return bool
     */
    template <typename Num> auto assess_bs(Num &gamma) -> bool {
        Space space = this->_space.copy();  // copy
        this->_omega.update(gamma);
        const auto result = cutting_plane_feas(this->_omega, space, this->_options);
        auto x_feas = std::get<0>(result);
        if (x_feas.size() != 0U) {
            this->_space.set_xc(x_feas);
            return true;
        }
        return false;
    }
};

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega   perform assessment on x0
 * @param[in,out] intvl   interval containing x*
 * @param[in]     options maximum iteration and error tolerance etc.
 * @return CInfo
 */
template <typename Oracle, typename T>
inline auto bsearch(Oracle &omega, const std::pair<T, T> &intvl, const Options &options = Options())
    -> std::tuple<T, size_t> {
    // assume monotone
    // auto& [lower, upper] = intvl;
    auto lower = intvl.first;
    auto upper = intvl.second;
    assert(lower <= upper);

    for (auto niter = 0U; niter < options.max_iters; ++niter) {
        auto tau = algo::half_nonnegative(upper - lower);
        if (tau < options.tolerance) {  // no more
            return {upper, niter};
        }
        auto gamma = lower;  // l may be `int` or `Fraction`
        gamma += tau;
        if (omega.assess_bs(gamma)) {  // feasible sol'n obtained
            upper = gamma;
        } else {
            lower = gamma;
        }
    }
    return {upper, options.max_iters};
}
