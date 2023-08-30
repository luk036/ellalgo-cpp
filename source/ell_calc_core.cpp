#include <cassert>
#include <cmath>                      // for sqrt
#include <ellalgo/ell_calc_core.hpp>  // for EllCalcCore
#include <tuple>                      // for tuple

/**
 * The function calculates and returns three values (rho, sigma, and delta) based on the input
 * parameters (beta0, beta1, and tsq).
 *
 * @param[in] beta0 The parameter `beta0` represents the value of beta for the first variable.
 * @param[in] beta1 The parameter `beta1` represents a value used in the calculation.
 * @param[in] tsq tsq is a constant value of type double. It represents the square of the parameter
 * t.
 *
 * @return The function `calc_parallel_cut` returns a tuple containing three values: `rho`, `sigma`,
 * and `delta`.
 */
auto EllCalcCore::calc_parallel_cut_fast(const double& beta0, const double& beta1,
                                         const double& tsq, const double& b0b1,
                                         const double& gamma) const
    -> std::tuple<double, double, double> {
    auto bsum = beta0 + beta1;
    auto bsumsq = bsum * bsum;
    auto h = tsq + b0b1 + this->_half_n * bsumsq;
    auto temp2 = h + std::sqrt(h * h - gamma * this->_n_plus_1 * bsumsq);
    auto inv_mu_plus_2 = gamma / temp2;
    auto inv_mu = gamma / (temp2 - 2.0 * gamma);
    auto&& rho = bsum * inv_mu_plus_2;
    auto&& sigma = 2.0 * inv_mu_plus_2;
    auto&& delta = 1.0 + (-2.0 * b0b1 + bsumsq * inv_mu_plus_2) * inv_mu / tsq;
    return {rho, sigma, delta};
}

/**
 * @brief Parallel Central Cut
 *
 * The function `calc_parallel_central_cut` calculates and returns the values of rho, sigma, and
 * delta based on the given input parameters under the parallel central cuts:
 *
 *        g' (x - xc) <= 0,
 *        g' (x - xc) + beta1 >= 0.
 *
 * @param[in] beta1 The parameter `beta1` represents a double value.
 * @param[in] tsq tsq is a constant value representing the square of the variable tau.
 *
 * @return The function `calc_parallel_central_cut` returns a tuple containing three values: rho,
 * sigma, and delta.
 */
auto EllCalcCore::calc_parallel_central_cut(const double& beta1, const double& tsq) const
    -> std::tuple<double, double, double> {
    auto b1sq = beta1 * beta1;
    auto a1sq = b1sq / tsq;
    auto temp = this->_half_n * a1sq;
    auto mu_plus_1 = temp + std::sqrt(1.0 - a1sq + temp * temp);
    auto mu_plus_2 = mu_plus_1 + 1.0;
    auto temp2 = this->_n_f * mu_plus_1;
    auto&& rho = beta1 / mu_plus_2;
    auto&& sigma = 2.0 / mu_plus_2;
    auto&& delta = temp2 / (temp2 - 1.0);
    return {rho, sigma, delta};
}

/**
 * @brief Calculate new ellipsoid under Non-central Cut
 *
 * The function `calc_bias_cut` calculates and returns the values of rho, sigma, and delta based on
 * the given beta and tau values under the bias-cut:
 *
 *        g' (x - xc) + beta \le 0
 *
 * @param[in] beta The parameter "beta" represents a value used in the calculation. It is a double
 * value.
 * @param[in] tau The parameter "tau" represents a value used in the calculation. It is not
 * specified in the code snippet provided. You would need to provide a value for "tau" in order to
 * use the `calc_bias_cut` function.
 *
 * @return The function `calc_bias_cut` returns a tuple containing the following values:
 */
auto EllCalcCore::calc_bias_cut_fast(const double& beta, const double& tau,
                                     const double& gamma) const
    -> std::tuple<double, double, double> {
    auto alpha = beta / tau;
    auto&& sigma = this->_cst2 * gamma / (tau + beta);
    auto&& rho = gamma / this->_n_plus_1;
    auto&& delta = this->_cst1 * (1.0 - alpha * alpha);
    return {rho, sigma, delta};
}

/**
 * @brief Central Cut
 *
 * The function `_calc_deep_cut_core` calculates and returns the values of rho, sigma, and delta
 * based on the given beta, tau, and gamma values under the central-cut:
 *
 *        g' (x - xc) \le 0.
 *
 * @param[in] tau tau is a constant value of type double. It represents the square of the variable
 * tau.
 *
 * @return A tuple containing the values of rho, sigma, and delta.
 */
auto EllCalcCore::calc_central_cut(const double& tau) const -> std::tuple<double, double, double> {
    auto&& sigma = this->_cst2;
    auto&& rho = tau / this->_n_plus_1;
    auto&& delta = this->_cst1;
    return {rho, sigma, delta};
}
