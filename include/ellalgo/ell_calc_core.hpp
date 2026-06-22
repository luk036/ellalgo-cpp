/**
 * @file ell_calc_core.hpp
 * @brief Core ellipsoid calculation routines
 */

#pragma once

#include <cstddef>  // For size_t
#include <tuple>

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalcCore = {x | (x - xc)' mq^-1 (x - xc) ≤ κ}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllCalcCore {
  private:
    /**
     * Member variables for EllCalcCore class.
     */
    double _n_f;
    double _n_plus_1;
    double _half_n;
    double _inv_n;
    double _n_sq;
    double _cst1;
    double _cst2;

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
    explicit EllCalcCore(const size_t ndim) noexcept
        : _n_f{double(ndim)},
          _n_plus_1{_n_f + 1.0},
          _half_n{_n_f / 2.0},
          _inv_n{1.0 / _n_f},
          _n_sq{_n_f * _n_f},
          _cst1{_n_sq / (_n_sq - 1.0)},
          _cst2{2.0 / _n_plus_1} {}

    // Rule of five: explicitly default destructor and assignment operators
    ~EllCalcCore() noexcept = default;
    EllCalcCore& operator=(const EllCalcCore&) noexcept = default;
    EllCalcCore& operator=(EllCalcCore&&) noexcept = default;

    /**
     * Move constructor for EllCalcCore.
     *
     * This is a move constructor that allows EllCalcCore objects to be efficiently moved instead of
     * copied. It takes an rvalue reference to another EllCalcCore object and steals its resources.
     *
     * @param[in] E An rvalue reference to the EllCalcCore object being moved.
     */
    EllCalcCore(EllCalcCore&& E) noexcept = default;

    /**
     * Copy constructor for EllCalcCore.
     *
     * Allows copying an existing EllCalcCore object into a new EllCalcCore object.
     * The new object will be an exact copy of the original.
     *
     * @param[in] E The EllCalcCore object to copy.
     */
    EllCalcCore(const EllCalcCore& E) noexcept = default;

    /**
     * @brief Compute new ellipsoid parameters for parallel bias cut
     *
     * Two parallel constraints:
     * @f[
     *     \beta_0 \le g^T (x - x_c) \le \beta_1
     * @f]
     * Computes intermediate values \f$b_0 b_1\f$ and \f$\eta = \tau^2 + n b_0 b_1\f$,
     * then delegates to calc_parallel_cut_fast().
     *
     * @param[in] beta0  Lower bound of the parallel cut
     * @param[in] beta1  Upper bound of the parallel cut
     * @param[in] tsq    Squared ellipsoid radius τ²
     * @return Tuple (rho, sigma, delta) — step size, scaling factor, contraction
     */
    auto calc_parallel_cut(const double beta0, const double beta1, const double tsq) const noexcept
        -> std::tuple<double, double, double> {
        const auto b0b1 = beta0 * beta1;
        const auto eta = tsq + this->_n_f * b0b1;
        return this->calc_parallel_cut_fast(beta0, beta1, tsq, b0b1, eta);
    }

    auto calc_parallel_cut_fast_old(const double beta0, const double beta1, const double tsq,
                                    const double b0b1, const double eta) const noexcept
        -> std::tuple<double, double, double>;

    /**
     * @brief Fast parallel cut computation with pre-computed values
     *
     * Uses pre-computed b0b1 and eta to compute rho, sigma, delta
     * for parallel bias cuts without re-computing intermediates.
     *
     * @param[in] beta0 First parallel cut parameter
     * @param[in] beta1 Second parallel cut parameter
     * @param[in] tsq   Squared tau (τ²)
     * @param[in] b0b1  Product beta0 × beta1
     * @param[in] eta   Intermediate value (tsq + n × b0b1)
     * @return Tuple (rho, sigma, delta)
     */
    auto calc_parallel_cut_fast(const double beta0, const double beta1, const double tsq,
                                const double b0b1, const double eta) const noexcept
        -> std::tuple<double, double, double>;

    /**
     * @brief Compute new ellipsoid parameters for parallel central cut
     *
     * One central cut through the center plus one parallel constraint:
     * @f[
     *     g^T (x - x_c) \le 0 \quad\text{and}\quad
     *     g^T (x - x_c) + \beta_1 \ge 0
     * @f]
     *
     * @param[in] beta1 Upper bound of the parallel constraint
     * @param[in] tsq   Squared ellipsoid radius τ²
     * @return Tuple (rho, sigma, delta)
     */
    auto calc_parallel_central_cut(const double beta1, const double tsq) const noexcept
        -> std::tuple<double, double, double>;

    /**
     * @brief Compute new ellipsoid parameters for bias (deep) cut
     *
     * Single constraint with bias:
     * @f[
     *     g^T (x - x_c) + \beta \le 0, \qquad \beta \ge 0
     * @f]
     * With \f$\eta = \tau + n\beta\f$, the update is:
     * @f[
     *     \rho = \frac{2\beta}{n+1}, \quad
     *     \sigma = \frac{n^2}{n^2-1}\Bigl(1 - \frac{\eta^2}{n\tau^2}\Bigr), \quad
     *     \delta = \frac{n^2}{n^2-1}\Bigl(1 - \frac{\eta^2}{\tau^2}\Bigr)
     * @f]
     * Delegates to calc_bias_cut_fast().
     *
     * @param[in] beta Bias parameter (≥ 0)
     * @param[in] tau  Square root of τ² (i.e. the ellipsoid radius)
     * @return Tuple (rho, sigma, delta)
     */
    auto calc_bias_cut(const double beta, const double tau) const noexcept
        -> std::tuple<double, double, double> {
        return this->calc_bias_cut_fast(beta, tau, tau + this->_n_f * beta);
    }

    /**
     * @brief Fast bias cut computation with pre-computed eta
     *
     * Uses pre-computed \f$\eta = \tau + n\beta\f$ to directly compute:
     * @f[
     *     \rho = \frac{2\beta}{n+1}, \qquad
     *     \sigma = \frac{n^2}{n^2-1}\Bigl(1 - \frac{\eta^2}{n\tau^2}\Bigr), \qquad
     *     \delta = \frac{n^2}{n^2-1}\Bigl(1 - \frac{\eta^2}{\tau^2}\Bigr)
     * @f]
     *
     * @param[in] beta Bias parameter
     * @param[in] tau  Ellipsoid radius
     * @param[in] eta  Intermediate value (tau + n × beta)
     * @return Tuple (rho, sigma, delta)
     */
    auto calc_bias_cut_fast(const double beta, const double tau, const double eta) const noexcept
        -> std::tuple<double, double, double>;

    /**
     * @brief Compute new ellipsoid parameters for central cut
     *
     * A central cut through the ellipsoid center:
     * @f[
     *     g^T (x - x_c) \le 0
     * @f]
     * The update formulas are:
     * @f[
     *     \rho = \frac{1}{n+1}, \qquad
     *     \sigma = \frac{n^2}{n^2-1}, \qquad
     *     \delta = \frac{n^2}{n^2-1}
     * @f]
     * where \f$n\f$ is the dimension and \f$\tau\f$ is the ellipsoid radius.
     * The cut passes through the ellipsoid center, making \f$\beta = 0\f$.
     *
     * @param[in] tau Ellipsoid radius (τ)
     * @return Tuple (rho, sigma, delta)
     */
    auto calc_central_cut(const double tau) const noexcept -> std::tuple<double, double, double>;
};  // } EllCalcCore
