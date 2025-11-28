#include <cmath>                      // for sqrt
#include <ellalgo/ell_calc_core.hpp>  // for EllCalcCore
#include <tuple>                      // for tuple

/**
 * The function calculates and returns three values (rho, sigma, and delta) based on the input
 * parameters (beta0, beta1, and tsq).
 *
 *                 _.-'''''''-._
 *               ,'     |       `.
 *              /  |    |         \
 *             .   |    |          .
 *             |   |    |          |
 *             |   |    |.         |
 *             |   |    |          |
 *             :\  |    |         /:
 *             | `._    |      _.' |
 *             |   |'-.......-'    |
 *             |   |    |          |
 *            "-τ" "-β" "-β"      +τ
 *                   1    0
 *
 * @param[in] beta0 The parameter `beta0` represents the value of beta for the first variable.
 * @param[in] beta1 The parameter `beta1` represents a value used in the calculation.
 * @param[in] tsq tsq is a constant value of type double. It represents the square of the parameter
 * gamma.
 *
 * @return The function `calc_parallel_cut` returns a tuple containing three values: `rho`, `sigma`,
 * and `delta`.
 */
auto EllCalcCore::calc_parallel_cut_fast(const double beta0, const double beta1, const double tsq,
                                         const double b0b1, const double eta) const noexcept
    -> std::tuple<double, double, double> {
    const double bavg = 0.5 * (beta0 + beta1);
    const double bavgsq = bavg * bavg;
    const double h = 0.5 * (tsq + b0b1) + _n_f * bavgsq;
    const double sqrt_term = std::sqrt(h * h - _n_plus_1 * eta * bavgsq);
    const double k = h + sqrt_term;
    const double inv_mu_plus_1 = eta / k;
    const double inv_mu = eta / (k - eta);
    return {
        bavg * inv_mu_plus_1,                                   // rho
        inv_mu_plus_1,                                          // sigma
        (tsq + inv_mu * (bavgsq * inv_mu_plus_1 - b0b1)) / tsq  // delta
    };
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
 *         _.-'''''''-._
 *       ,'             `.
 *      /                 \
 *     .                   .
 *     |                   |
 *     |         .         |
 *     |                   |
 *      \                 /
 *       `._           _.'
 *          '-.......-'
 *
 *          |    |
 *     +----+----+---------+
 *    "-τ" "-β"  0        "+τ"
 *           1
 *
 * @param[in] beta1 The parameter `beta1` represents a double value.
 * @param[in] tsq tsq is a constant value representing the square of the variable tau.
 *
 * @return The function `calc_parallel_central_cut` returns a tuple containing three values: rho,
 * sigma, and delta.
 */
auto EllCalcCore::calc_parallel_central_cut(const double beta1, const double tsq) const noexcept
    -> std::tuple<double, double, double> {
    const double b1sq = beta1 * beta1;
    const double a1sq = b1sq / tsq;
    const double k = _half_n * a1sq;
    const double r = k + std::sqrt(1.0 - a1sq + k * k);
    const double r_plus_1 = r + 1.0;
    return {
        beta1 / r_plus_1,  // rho
        2.0 / r_plus_1,    // sigma
        r / (r - _inv_n)   // delta
    };
}

/**
 * @brief Calculate new ellipsoid under Non-central Cut
 *
 * The function `calc_bias_cut` calculates and returns the values of rho, sigma, and delta based on
 * the given beta and tau values under the bias-cut:
 *
 *        g' (x - xc) + β \le 0
 *
 *           _.-'''''''-._
 *         ,'             `.
 *        /                 \
 *       .                   .
 *       |                   |
 *       |         .         |
 *       |                   |
 *        \                 /
 *         `._           _.'
 *            '-.......-'
 *
 *       |      |
 *       +------+------------+
 *      "-τ"   "-β"          "+τ"
 *
 *
 * @param[in] beta The parameter "beta" represents a value used in the calculation. It is a double
 * value.
 * @param[in] tau The parameter "tau" represents a value used in the calculation. It is not
 * specified in the code snippet provided. You would need to provide a value for "tau" in order to
 * use the `calc_bias_cut` function.
 *
 * @return The function `calc_bias_cut` returns a tuple containing the following values:
 */
auto EllCalcCore::calc_bias_cut_fast(const double beta, const double tau,
                                     const double eta) const noexcept
    -> std::tuple<double, double, double> {
    return {
        eta / _n_plus_1,                             // rho
        _cst2 * eta / (tau + beta),                  // sigma
        _cst1 * (1.0 - (beta * beta) / (tau * tau))  // delta
    };
}

/**
 * @brief Central Cut
 *
 * The function `_calc_bias_cut_core` calculates and returns the values of rho, sigma, and delta
 * based on the given beta, tau, and eta values under the central-cut:
 *
 *        g' (x - xc) \le 0.
 *
 *           _.-'''''''-._
 *         ,'             `.
 *        /                 \
 *       .                   .
 *       |                   |
 *       |         .         |
 *       |                   |
 *        \                 /
 *         `._           _.'
 *            '-.......-'
 *
 *       |         |
 *       +---------+---------+
 *      "-τ"       "0"       "+τ"
 *
 *
 * @param[in] tau tau is a constant value of type double. It represents the square of the variable
 * tau.
 *
 * @return A tuple containing the values of rho, sigma, and delta.
 */
auto EllCalcCore::calc_central_cut(const double tau) const noexcept
    -> std::tuple<double, double, double> {
    return {
        tau / _n_plus_1,  // rho
        _cst2,            // sigma
        _cst1             // delta
    };
}
