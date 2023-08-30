#pragma once

#include <cmath>
#include <tuple>

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalcCore = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllCalcCore {
  private:
    const double _n_f;
    const double _n_plus_1;
    const double _half_n;
    const double _n_sq;
    const double _cst1;
    const double _cst2;

  public:
    /**
     * @brief Construct a new EllCalcCore object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    explicit EllCalcCore(size_t ndim)
        : _n_f{double(ndim)},
          _n_plus_1{_n_f + 1.0},
          _half_n{_n_f / 2.0},
          _n_sq{_n_f * _n_f},
          _cst1{_n_sq / (_n_sq - 1.0)},
          _cst2{2.0 / _n_plus_1} {}

    /**
     * @brief Construct a new EllCalcCore object
     *
     * @param[in] E (move)
     */
    EllCalcCore(EllCalcCore &&E) = default;

    /**
     * @brief Destroy the EllCalcCore object
     *
     */
    ~EllCalcCore() = default;

    /**
     * @brief Construct a new EllCalcCore object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    EllCalcCore(const EllCalcCore &E) = default;

    /**
     * @brief Parallel Cut
     *
     * The function calculates and returns three values (rho, sigma, and delta) based on the input
     * parameters (beta0, beta1, and tsq) under the parallel-cuts:
     *
     *        g' (x - xc) + beta0 <= 0,
     *        g' (x - xc) + beta1 >= 0.
     *
     * @param[in] beta0 The parameter `beta0` represents the value of beta for the first variable.
     * @param[in] beta1 The parameter `beta1` represents a value used in the calculation.
     * @param[in] tsq tsq is a constant value of type double. It represents the square of the
     * parameter tau.
     *
     * @return The function `calc_parallel_cut` returns a tuple containing three values: `rho`,
     * `sigma`, and `delta`.
     */
    auto calc_parallel_cut(const double &beta0, const double &beta1, const double &tsq) const
        -> std::tuple<double, double, double> {
        auto b0b1 = beta0 * beta1;
        auto gamma = tsq + this->_n_f * b0b1;
        return this->calc_parallel_cut_fast(beta0, beta1, tsq, b0b1, gamma);
    }

    auto calc_parallel_cut_fast(const double &beta0, const double &beta1, const double &tsq,
                                const double &b0b1, const double &gamma) const
        -> std::tuple<double, double, double>;

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
     * @return The function `calc_parallel_central_cut` returns a tuple containing three values:
     * rho, sigma, and delta.
     */
    auto calc_parallel_central_cut(const double &beta1, const double &tsq) const
        -> std::tuple<double, double, double>;

    /**
     * @brief Calculate new ellipsoid under Non-central Cut
     *
     * The function `calc_bias_cut` calculates and returns the values of rho, sigma, and delta based
     * on the given beta and tau values under the bias-cut:
     *
     *        g' (x - xc) + beta \le 0
     *
     * @param[in] beta The parameter "beta" represents a value used in the calculation. It is a
     * double value.
     * @param[in] tau The parameter "tau" represents a value used in the calculation. It is not
     * specified in the code snippet provided. You would need to provide a value for "tau" in order
     * to use the `calc_bias_cut` function.
     *
     * @return The function `calc_bias_cut` returns a tuple containing the following values:
     */
    auto calc_bias_cut(const double &beta, const double &tau) const
        -> std::tuple<double, double, double> {
        return this->calc_bias_cut_fast(beta, tau, tau + this->_n_f * beta);
    }

    auto calc_bias_cut_fast(const double &beta, const double &tau, const double &gamma) const
        -> std::tuple<double, double, double>;

    /**
     * @brief Central Cut
     *
     * The function `_calc_deep_cut_core` calculates and returns the values of rho, sigma, and delta
     * based on the given beta, tau, and gamma values under the central-cut:
     *
     *        g' (x - xc) \le 0.
     *
     * @param[in] tau tau is a constant value of type double. It represents the square of the
     * variable tau.
     *
     * @return A tuple containing the values of rho, sigma, and delta.
     */
    auto calc_central_cut(const double &tau) const -> std::tuple<double, double, double>;
};  // } EllCalcCore
