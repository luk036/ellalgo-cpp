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
    /**
     * Member variables for EllCalcCore class.
     */
    const double _n_f;
    const double _n_plus_1;
    const double _half_n;
    const double _inv_n;
    const double _n_sq;
    const double _cst1;
    const double _cst2;

  public:
    /**
     * Constructor for EllCalcCore class.
     * Initializes member variables based on input ndim.
     *
     * Example:
     * EllCalcCore E(2);
     *
     * @param[in] ndim Number of dimensions for EllCalcCore object.
     */
    explicit EllCalcCore(size_t ndim) noexcept
        : _n_f{double(ndim)},
          _n_plus_1{_n_f + 1.0},
          _half_n{_n_f / 2.0},
          _inv_n{1.0 / _n_f},
          _n_sq{_n_f * _n_f},
          _cst1{_n_sq / (_n_sq - 1.0)},
          _cst2{2.0 / _n_plus_1} {}

    /**
     * Move constructor for EllCalcCore.
     *
     * This is a move constructor that allows EllCalcCore objects to be efficiently moved instead of
     * copied. It takes an rvalue reference to another EllCalcCore object and steals its resources.
     *
     * @param[in] E An rvalue reference to the EllCalcCore object being moved.
     */
    EllCalcCore(EllCalcCore &&E) noexcept = default;

    /**
     * Copy constructor for EllCalcCore.
     *
     * Allows copying an existing EllCalcCore object into a new EllCalcCore object.
     * The new object will be an exact copy of the original.
     *
     * @param[in] E The EllCalcCore object to copy.
     */
    EllCalcCore(const EllCalcCore &E) noexcept = default;

    /**
     * Calculates the new ellipsoid parameters rho, sigma, and delta after
     * applying parallel cuts defined by beta0 and beta1.
     *
     * This is a public member function that computes the ellipsoid parameters
     * after parallel cuts of the form:
     *
     *   g'(x - xc) + beta0 <= 0
     *   g'(x - xc) + beta1 >= 0
     *
     * It takes in the parallel cut parameters beta0, beta1, and the squared
     * value of tau. It computes intermediate values b0b1 and eta, then calls
     * calc_parallel_cut_fast() to calculate rho, sigma, delta which are
     * returned as a tuple.
     *
     * Example:
     * auto [rho, sigma, delta] = E.calc_parallel_cut(beta0, beta1, tsq);
     *
     * @param[in] beta0 The parameter `beta0` represents the value of beta for the first variable.
     * @param[in] beta1 The parameter `beta1` represents a value used in the calculation.
     * @param[in] tsq tsq is a constant value of type double. It represents the square of the
     * parameter tau.
     *
     * @return The function `calc_parallel_cut` returns a tuple containing three values: `rho`,
     * `sigma`, and `delta`.
     */
    auto calc_parallel_cut(double beta0, double beta1,
                           double tsq) const noexcept -> std::tuple<double, double, double> {
        const auto b0b1 = beta0 * beta1;
        const auto eta = tsq + this->_n_f * b0b1;
        return this->calc_parallel_cut_fast(beta0, beta1, tsq, b0b1, eta);
    }

    /**
     * Calculates ellipsoid parameters after parallel cuts.
     *
     * This public member function computes the ellipsoid parameters rho, sigma,
     * and delta after applying parallel cuts defined by beta0 and beta1. The
     * parameters beta0, beta1, tsq, b0b1, and eta are used in the calculation.
     *
     * @param[in] beta0 First parallel cut parameter.
     * @param[in] beta1 Second parallel cut parameter.
     * @param[in] tsq Square of tau parameter.
     * @param[in] b0b1 Product of beta0 and beta1.
     * @param[in] eta Computed intermediate value.
     * @return Tuple containing computed rho, sigma and delta.
     */
    auto calc_parallel_cut_fast(double beta0, double beta1, double tsq,
                                double b0b1, double eta) const noexcept -> std::tuple<double, double, double>;

    /**
     * Calculates ellipsoid parameters after parallel central cuts.
     *
     * This is a public member function that computes the ellipsoid parameters
     * after parallel central cuts of the form:
     *
     *   g'(x - xc) <= 0
     *   g'(x - xc) + beta1 >= 0
     *
     * This function computes the ellipsoid parameters rho, sigma,
     * and delta after applying parallel central cuts defined by
     * beta1 and tau. The parameters beta1 and tsq are used in
     * the calculation.
     *
     * @param[in] beta1 Parallel cut parameter.
     * @param[in] tsq Square of tau parameter.
     * @return Tuple containing computed rho, sigma and delta.
     */
    auto calc_parallel_central_cut(double beta1,
                                   double tsq) const noexcept -> std::tuple<double, double, double>;

    /**
     * Calculates new ellipsoid parameters after bias cut.
     *
     * This function computes the new ellipsoid parameters rho, sigma,
     * and delta after applying a bias cut of the form:
     *
     *   g'(x - xc) + beta <= 0
     *
     * It takes the bias parameter beta and tau as inputs.
     * Tau is the square of the original tau parameter.
     * It returns the computed rho, sigma, delta tuple.
     *
     * @param[in] beta The parameter "beta" represents a value used in the calculation.
     * @param[in] tau The parameter "tau" represents a value used in the calculation. It is not
     * specified in the code snippet provided. You would need to provide a value for "tau" in order
     * to use the `calc_bias_cut` function.
     *
     * @return The function `calc_bias_cut` returns a tuple containing the following values:
     */
    auto calc_bias_cut(double beta,
                       double tau) const noexcept -> std::tuple<double, double, double> {
        return this->calc_bias_cut_fast(beta, tau, tau + this->_n_f * beta);
    }

    /**
     * @brief Fast Bias Cut
     *
     * This is a public member function that calculates the new ellipsoid parameters
     * rho, sigma, and delta after applying a bias cut of the form:
     *
     *   g'(x - xc) + beta <= 0
     *
     * It takes the bias parameter beta, tau, and eta as inputs and returns a
     * tuple containing the computed rho, sigma, delta values.
     */
    auto calc_bias_cut_fast(double beta, double tau,
                            double eta) const noexcept -> std::tuple<double, double, double>;

    /**
     * Calculates new ellipsoid parameters after applying a central cut.
     *
     * This function takes the tau parameter as input and computes
     * the new ellipsoid parameters rho, sigma, and delta after
     * applying a central cut defined by:
     *
     *   g'(x - xc) <= 0
     *
     * @param[in] tau The tau parameter.
     * @return Tuple containing computed rho, sigma and delta.
     */
    auto calc_central_cut(double tau) const noexcept -> std::tuple<double, double, double>;
};  // } EllCalcCore
