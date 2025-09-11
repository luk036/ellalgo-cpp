#include <cassert>
#include <cmath>                      // for sqrt
#include <ellalgo/ell_assert.hpp>     // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>       // for EllCalc
#include <ellalgo/ell_calc_core.hpp>  // for EllCalcCore
#include <ellalgo/ell_config.hpp>     // for CutStatus, CutStatus::Success
#include <tuple>                      // for tuple

/**
 * @brief Parallel- or deep-cut
 *
 * The function `calc_parallel_bias_cut` calculates and returns the values of rho, sigma,
 * and delta based on the given input parameters.
 *
 * @param[in] beta0 The parameter `beta0` represents a double value.
 * @param[in] beta1 The parameter `beta1` represents a double value.
 * @param[in] tsq tsq is a constant value of type double. It represents the
 * square of the variable tau.
 *
 * @return a tuple containing the following values:
 * 1. CutStatus: An enum value indicating the status of the calculation.
 * 2. rho: A double value representing the calculated rho.
 * 3. sigma: A double value representing the calculated sigma.
 * 4. delta: A double value representing the calculated delta.
 */
auto EllCalc::calc_parallel_bias_cut(const double &beta0, const double &beta1,
                                     const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    if (beta1 < beta0) {
        return {CutStatus::NoSoln, {0.0, 0.0, 0.0}};  // no sol'n
    }
    if ((beta1 > 0 && tsq <= beta1 * beta1) || !this->use_parallel_cut) {
        return this->calc_bias_cut(beta0, tsq);
    }
    auto &&result = this->_helper.calc_parallel_cut(beta0, beta1, tsq);
    return {CutStatus::Success, result};
}

/**
 * @brief Parallel central cut
 *
 * The function `calc_parallel_central_cut` calculates and returns the values of rho, sigma,
 * and delta based on the given input parameters.
 *
 * @param[in] beta1 The parameter `beta1` represents a double value.
 * @param[in] tsq tsq is a constant value of type double. It represents the
 * square of the variable tau.
 *
 * @return a tuple containing the following values:
 * 1. CutStatus: An enum value indicating the status of the calculation.
 * 2. rho: A double value representing the calculated rho.
 * 3. sigma: A double value representing the calculated sigma.
 * 4. delta: A double value representing the calculated delta.
 */
auto EllCalc::calc_parallel_central_cut(const double &beta1, const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    if (beta1 < 0.0) {
        return {CutStatus::NoSoln, {0.0, 0.0, 0.0}};  // no sol'n
    }
    if (tsq <= beta1 * beta1 || !this->use_parallel_cut) {
        return this->calc_central_cut(tsq);
    }
    auto &&result = this->_helper.calc_parallel_central_cut(beta1, tsq);
    return {CutStatus::Success, result};
    // this->_mu ???
}

/**
 * @brief Deep-Cut
 *
 * The function `calc_bias_cut` calculates the values of `tau`, `eta` under the
 * deep-cut:
 *
 *        g' (x - xc) + beta \le 0,
 *
 * and calls another function `_calc_bias_cut_core` to calculate the values of
 * `CutStatus`, `double`, `double`, and `double` based on the input values of
 * `beta` and `tsq`.
 *
 * @param[in] beta The parameter "beta" represents a value that needs to be
 * greater than or equal to 0.0. It is used in the calculation of "eta" and is
 * compared with "tsq" in the if statement.
 * @param[in] tsq tsq is a variable of type double, which represents the square
 * of the value tau.
 *
 * @return The function `calc_bias_cut` returns a tuple containing four values:
 * `CutStatus`, `double`, `double`, `double`.
 */
auto EllCalc::calc_bias_cut(const double &beta, const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    assert(beta >= 0.0);
    if (tsq < beta * beta) {
        return {CutStatus::NoSoln, {0.0, 0.0, 0.0}};  // no sol'n
    }
    auto &&result = this->_helper.calc_bias_cut(beta, std::sqrt(tsq));
    return {CutStatus::Success, result};
}

/**
 * @brief Central Cut
 *
 * The function `_calc_bias_cut_core` calculates and returns the values of rho, sigma,
 * and delta based on the given beta, tau, and eta values under the
 * central-cut:
 *
 *        g' (x - xc) \le 0
 *
 * @param[in] tsq tsq is a constant value of type double. It represents the
 * square of the variable tau.
 *
 * @return A tuple containing the following values:
 * 1. CutStatus::Success
 * 2. rho
 * 3. sigma
 * 4. delta
 */
auto EllCalc::calc_central_cut(const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    // auto sigma = this->_c2;
    // auto rho = std::sqrt(tsq) / this->_nPlus1;
    // auto delta = this->_c1;
    auto &&result = this->_helper.calc_central_cut(std::sqrt(tsq));
    return {CutStatus::Success, result};
}

/**
 * The function `calc_parallel_bias_cut_q` calculates the parallel deep cut for a given range of
 * beta values and a given tsq value.
 *
 * @param[in] beta0 The parameter `beta0` represents the value of beta at the starting point of the
 * calculation. It is a constant reference to a double.
 * @param[in] beta1 The parameter `beta1` represents a value that is being compared to `beta0` in
 * the `if` statement on line 2. It is also used in calculations on lines 6 and 11. Without more
 * context, it is difficult to determine the exact meaning of `beta1`.
 * @param[in] tsq tsq is a variable of type double, which represents the square of the parameter
 * gamma.
 *
 * @return The function `calc_parallel_bias_cut_q` returns a tuple of type `std::tuple<CutStatus,
 * std::tuple<double, double, double>>`.
 */
auto EllCalc::calc_parallel_bias_cut_q(const double &beta0, const double &beta1,
                                       const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    if (beta1 < beta0) {
        return {CutStatus::NoSoln, {0.0, 0.0, 0.0}};  // no sol'n
    }

    if ((beta1 > 0.0 && tsq <= beta1 * beta1) || !this->use_parallel_cut) {
        return this->calc_bias_cut_q(beta0, tsq);
    }

    const auto b0b1 = beta0 * beta1;
    const auto eta = tsq + this->_n_f * b0b1;
    if (ELL_UNLIKELY(eta <= 0.0)) {
        return {CutStatus::NoEffect, {0.0, 0.0, 1.0}};  // no effect
    }
    auto &&result = this->_helper.calc_parallel_cut_fast(beta0, beta1, tsq, b0b1, eta);
    return {CutStatus::Success, result};
}

/**
 * The function calculates the deep cut for a given beta and tsq value.
 *
 * @param[in] beta The parameter `beta` represents a value used in the calculation. It is passed by
 * reference as a constant double.
 * @param[in] tsq The parameter `tsq` represents the square of the value `tau`, which is a variable
 * used in the calculation.
 *
 * @return The function `calc_bias_cut_q` returns a tuple containing four values: `CutStatus`,
 * `double`, `double`, `double`.
 */
auto EllCalc::calc_bias_cut_q(const double &beta, const double &tsq) const
    -> std::tuple<CutStatus, std::tuple<double, double, double>> {
    const auto tau = std::sqrt(tsq);
    if (tau < beta) {
        return {CutStatus::NoSoln, {0.0, 0.0, 0.0}};  // no sol'n
    }
    const auto eta = tau + this->_n_f * beta;
    if (ELL_UNLIKELY(eta <= 0.0)) {
        return {CutStatus::NoEffect, {0.0, 0.0, 1.0}};  // no effect
    }
    auto &&result = this->_helper.calc_bias_cut_fast(beta, tau, eta);
    return {CutStatus::Success, result};
}
