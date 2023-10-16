#include <cassert>
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
 *             β  + β                            
 *         _    0    1                           
 *         β = ───────                           
 *                2                              
 *                                               
 *             1   ⎛ 2          ⎞       _2       
 *         h = ─ ⋅ ⎜τ  + β  ⋅ β ⎟ + n ⋅ β        
 *             2   ⎝      0    1⎠                
 *                    _____________________      
 *                   ╱ 2                 _2      
 *         k = h + ╲╱ h  - (n + 1) ⋅ η ⋅ β       
 *                                               
 *               1     η                         
 *         σ = ───── = ─                         
 *             μ + 1   k                         
 *                                               
 *         1     η                               
 *         ─ = ─────                             
 *         μ   k - η                             
 *
 *             _                                         
 *         ϱ = β ⋅ σ                            
 *                                               
 *              2    2   1   ⎛_2              ⎞  
 *         δ ⋅ τ  = τ  + ─ ⋅ ⎜β  ⋅ σ - β  ⋅ β ⎟  
 *                       μ   ⎝          0    1⎠   
 *
 * @param[in] beta0 The parameter `beta0` represents the value of beta for the first variable.
 * @param[in] beta1 The parameter `beta1` represents a value used in the calculation.
 * @param[in] tsq tsq is a constant value of type double. It represents the square of the parameter
 * gamma.
 *
 * @return The function `calc_parallel_cut` returns a tuple containing three values: `rho`, `sigma`,
 * and `delta`.
 */
auto EllCalcCore::calc_parallel_cut_fast(const double& beta0, const double& beta1,
                                         const double& tsq, const double& b0b1,
                                         const double& eta) const
    -> std::tuple<double, double, double> {
    auto bavg = 0.5 * (beta0 + beta1);
    auto bavgsq = bavg * bavg;
    auto h = 0.5 * (tsq + b0b1) + this->_n_f * bavgsq;
    auto k = h + std::sqrt(h * h - this->_n_plus_1 * eta * bavgsq);
    auto inv_mu_plus_1 = eta / k;
    auto inv_mu = eta / (k - eta);
    auto&& rho = bavg * inv_mu_plus_1;
    auto&& sigma = inv_mu_plus_1;
    auto&& delta = (tsq + inv_mu * (bavgsq * inv_mu_plus_1 - b0b1))  / tsq;
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
 *                 _.-'''''''-._
 *               ,'      |      `.
 *              /  |     |        \
 *             .   |     |         .
 *             |   |               |
 *             |   |     .         |
 *             |   |               |
 *             :\  |     |        /:
 *             | `._     |     _.' |
 *             |   |'-.......-'    |
 *             |   |     |         |
 *            "-τ" "-β"  0        +τ
 *                   1
 *
 *          2    2    2
 *         α  = β  / τ
 *
 *             n    2
 *         k = ─ ⋅ α
 *             2
 *                    ___________
 *                   ╱ 2        2
 *         r = k + ╲╱ k  + 1 - α
 *
 *               β
 *         ϱ = ─────
 *             r + 1
 *
 *               2
 *         σ = ─────
 *             r + 1
 *
 *                 r
 *         δ = ─────────
 *             r - 1 / n
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
    auto k = this->_half_n * a1sq;
    auto r = k + std::sqrt(1.0 - a1sq + k * k);
    auto r_plus_1 = r + 1.0;
    auto&& rho = beta1 / r_plus_1;
    auto&& sigma = 2.0 / r_plus_1;
    auto&& delta = r / (r - this->_inv_n);
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
 *              _.-'''''''-._
 *            ,'   |         `.
 *           /     |           \
 *          .      |            .
 *          |      |            |
 *          |      |  .         |
 *          |      |            |
 *          :\     |           /:
 *          | `._  |        _.' |
 *          |    '-.......-'    |
 *          |      |            |
 *         "-τ"     "-β"       +τ
 *       
 *          η = τ + n ⋅ β
 *       
 *                η
 *          ϱ = ─────
 *              n + 1
 *       
 *              2 ⋅ ϱ
 *          σ = ─────
 *              τ + β
 *       
 *                 2       2    2
 *                n       τ  - β
 *          δ = ────── ⋅  ───────
 *               2           2
 *              n  - 1      τ
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
                                     const double& eta) const
    -> std::tuple<double, double, double> {
    auto alpha = beta / tau;
    auto&& sigma = this->_cst2 * eta / (tau + beta);
    auto&& rho = eta / this->_n_plus_1;
    auto&& delta = this->_cst1 * (1.0 - alpha * alpha);
    return {rho, sigma, delta};
}

/**
 * @brief Central Cut
 *
 * The function `_calc_deep_cut_core` calculates and returns the values of rho, sigma, and delta
 * based on the given beta, tau, and eta values under the central-cut:
 *
 *        g' (x - xc) \le 0.
 *
 *            _.-'''''''-._
 *          ,'      |      `.
 *         /        |        \
 *        .         |         .
 *        |                   |
 *        |         .         |
 *        |                   |
 *        :\        |        /:
 *        | `._     |     _.' |
 *        |    '-.......-'    |
 *        |         |         |
 *       "-τ"       0        +τ
 *    
 *              2
 *        σ = ─────
 *            n + 1
 *    
 *              τ
 *        ϱ = ─────
 *            n + 1
 *    
 *               2
 *              n
 *        δ = ──────
 *             2
 *            n  - 1
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
