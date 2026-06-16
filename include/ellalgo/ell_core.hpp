/**
 * @file ell_core.hpp
 * @brief Ellipsoid search space core with matrix update
 */

// -*- coding: utf-8 -*-
#pragma once

#include <tuple>
#include <utility>
#include <valarray>

#include "ell_calc.hpp"
#include "ell_config.hpp"
#include "ell_matrix.hpp"

/**
 * @brief Ellipsoid Search Space Core
 *
 * Without the knowledge of the type of xc
 *
 *  ℰ {x | (x - xc)' mq^-1 (x - xc) ≤ κ}
 *
 * <pre>
 *    2D Ellipsoid Core Visualization
 *        y
 *        ^
 *        │
 *      ┌─┼─┐  ← ellipsoid boundary
 *    ┌─┘ │ └─┐
 *  ──┼───●───┼──→ x  ← center (xc)
 *    └─┐ │ ┌─┘
 *      └─┼─┘
 *        │
 *        ● (x)
 *      point inside/outside
 *      ellipsoid
 * </pre>
 */
class EllCore {
    using Vec = std::valarray<double>;

    size_t _n;
    double _kappa;
    Matrix _mq;
    EllCalc _helper;
    double _tsq{};
    Vec _scratch;

  public:
    bool no_defer_trick = false;

  private:
    /// @brief Deleted copy assignment operator (non-copyable).
    auto operator=(const EllCore& E) -> EllCore& = delete;

    /**
     * @brief Construct EllCore with initial kappa, matrix, and dimension
     *
     * @param[in] kappa Initial scaling factor κ for the ellipsoid
     * @param[in] mq Initial shape matrix Q (moved in)
     * @param[in] ndim Number of dimensions
     */
    EllCore(double kappa, Matrix&& mq, size_t ndim)
        : _n{ndim}, _kappa{kappa}, _mq{std::move(mq)}, _helper{_n}, _scratch(0.0, ndim) {}

  public:
    /**
     * The function constructs a new EllCore object with a given value and dimension.
     *
     * @param[in] val The parameter `val` is a reference to a `Vec` object, which represents a
     * vector of values. It is used to initialize the diagonal elements of the `_mq` matrix in the
     * `EllCore` object.
     * @param[in] ndim The parameter `ndim` represents the number of dimensions for the `EllCore`
     * object.
     */
    EllCore(const Vec& val, size_t ndim) : EllCore{1.0, Matrix(ndim), ndim} {
        this->_mq.diagonal() = val;
    }

    /**
     * @brief Construct a new EllCore object
     *
     * The function constructs a new EllCore object with a given alpha and ndim, and initializes the
     * matrix _mq to an identity matrix.
     *
     * @param[in] alpha The alpha parameter is a double value that represents the scaling factor for
     * the EllCore object. It is used to adjust the size of the ellipsoid.
     * @param[in] ndim The parameter `ndim` represents the number of dimensions for the EllCore
     * object. It specifies the size of the matrix used in the construction of the object.
     */
    EllCore(const double alpha, const size_t ndim) : EllCore{alpha, Matrix(ndim), ndim} {
        this->_mq.identity();
    }

    /**
     * @brief Construct a new EllCore object
     *
     * The function is a constructor for an EllCore object that takes an rvalue reference as a
     * parameter.
     *
     * @param[in] E The parameter "E" is an rvalue reference to an object of type "EllCore".
     *
     */
    EllCore(EllCore&& E) = default;

    /**
     * @brief Move assignment operator
     */
    EllCore& operator=(EllCore&&) = default;

    /**
     * @brief Destroy the EllCore object
     *
     */
    ~EllCore() = default;

    /**
     * @brief Construct a new EllCore object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param[in] E The parameter "E" is a reference to an object of type "EllCore".
     */
    explicit EllCore(const EllCore& E) = default;

    /**
     * @brief explicitly copy
     *
     * @return EllCore
     */
    auto copy() const -> EllCore { return EllCore(*this); }

    /**
     * @brief Get the squared ellipsoid radius τ²
     *
     * @return double The squared radius
     */
    constexpr auto tsq() const -> double { return this->_tsq; }

    /**
     * The function sets the value of the use_parallel_cut property in the _mgr object.
     *
     * @param[in] value The value parameter is a boolean value that determines whether or not to use
     * parallel cut.
     */
    void set_use_parallel_cut(bool value) { this->_helper.use_parallel_cut = value; }

    /**
     * @brief Update ellipsoid core function using the deep cut(s)
     *
     * The `update_bias_cut` function is a member function of the `EllCore` class. It is used to
     * update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_bias_cut(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_bias_cut(beta_l, tsq_l);
        });
    }

    /**
     * @brief Update ellipsoid core function using the central cut(s)
     *
     * The `update_central_cut` function is a member function of the `EllCore` class. It is used to
     * update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_central_cut(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_central_cut(beta_l, tsq_l);
        });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `update_q` function is a member function of the `EllCore` class. It is used to update the
     * ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_q(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_q(beta_l, tsq_l);
        });
    }

    /**
     * @brief Update ellipsoid core function using the deep cut(s)
     *
     * The `update_stable_bias_cut` function is a member function of the `EllCore` class. It is used
     * to update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_stable_bias_cut(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_bias_cut(beta_l, tsq_l);
        });
    }

    /**
     * @brief Update ellipsoid core function using the central cut(s)
     *
     * The `update_stable_central_cut` function is a member function of the `EllCore` class. It is
     * used to update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_stable_central_cut(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_central_cut(beta_l, tsq_l);
        });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `update_stable_q` function is a member function of the `EllCore` class. It is used to
     * update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @param[in] grad gradient
     * @param[in] beta
     * @return CutStatus
     */
    template <typename T> auto update_stable_q(Vec& grad, const T& beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T& beta_l, const double tsq_l) {
            return this->_update_cut_q(beta_l, tsq_l);
        });
    }

  private:
    /**
     * @brief Update ellipsoid core using the cut(s)
     *
     * The cut_strategy callable must take (beta, tsq) and return a CutResult.
     *
     * @tparam T
     * @tparam Fn
     * @param[in] grad gradient
     * @param[in] beta
     * @param[in] cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_core(Vec& grad, const T& beta, Fn&& cut_strategy) -> CutStatus {
        std::valarray<double> grad_t(0.0, this->_n);
        for (size_t i = 0; i != this->_n; ++i) {
            for (size_t j = 0; j != this->_n; ++j) {
                grad_t[i] += this->_mq(i, j) * grad[j];
            }
        }

        const auto omega = (grad_t * grad).sum();
        this->_tsq = this->_kappa * omega;

        if (omega == 0.0) {
            grad = grad_t;
            return CutStatus::Success;
        }

        auto result = std::forward<Fn>(cut_strategy)(beta, this->_tsq);
        if (result.status != CutStatus::Success) {
            return result.status;
        }

        // n (n+1) / 2 + n
        const auto r = result.sigma / omega;
        for (size_t i = 0; i != this->_n; ++i) {
            const auto rQg = r * grad_t[i];
            for (size_t j = 0; j != i; ++j) {
                this->_mq(i, j) -= rQg * grad_t[j];
                this->_mq(j, i) = this->_mq(i, j);
            }
            this->_mq(i, i) -= rQg * grad_t[i];
        }

        this->_kappa *= result.delta;

        if (this->no_defer_trick) {
            this->_mq *= this->_kappa;
            this->_kappa = 1.0;
        }

        grad = grad_t * (result.rho / omega);
        return result.status;
    }

    /**
     * @brief Update ellipsoid core using the cut(s) — numerically stable LDL^T form
     *
     * Uses LDL^T factorization instead of the direct Q-update in _update_core.
     * The cut_strategy callable must take (beta, tsq) and return a CutResult.
     *
     * @tparam T
     * @tparam Fn
     * @param[in] g gradient
     * @param[in] beta
     * @param[in] cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_stable_core(Vec& g, const T& beta, Fn&& cut_strategy) -> CutStatus {
        auto& invLg = this->_scratch;
        invLg = g;
        for (size_t j = 0; j != this->_n - 1; ++j) {
            for (size_t i = j + 1; i != this->_n; ++i) {
                this->_mq(j, i) = this->_mq(i, j) * invLg[j];
                invLg[i] -= this->_mq(j, i);
            }
        }

        auto invDinvLg{invLg};
        for (size_t i = 0; i != this->_n; ++i) {
            invDinvLg[i] *= this->_mq(i, i);
        }

        auto omega = 0.0;
        for (size_t i = 0; i != this->_n; ++i) {
            omega += invDinvLg[i] * invLg[i];
        }

        this->_tsq = this->_kappa * omega;

        auto result = std::forward<Fn>(cut_strategy)(beta, this->_tsq);
        if (result.status != CutStatus::Success) {
            return result.status;
        }

        // Calculate the (L')^-1 * D^-1 * L^-1 * grad : (n-1)n / 2
        auto grad_t{invDinvLg};                     // initially
        for (auto i = this->_n - 1; i != 0; --i) {  // backward subsituition
            for (auto j = i; j != this->_n; ++j) {
                grad_t[i - 1] -= this->_mq(j, i - 1) * grad_t[j];  // ???
            }
        }

        // rank-one update: 3*n + (n-1)*n/2
        const auto mu = result.sigma / (1.0 - result.sigma);
        auto oldt = omega / mu;  // initially
        // const auto m = this->_n - 1;
        auto v{g};
        for (size_t j = 0; j != this->_n; ++j) {
            const auto p = v[j];
            const auto temp = invDinvLg[j];
            const auto newt = oldt + p * temp;
            const auto beta2 = temp / newt;
            this->_mq(j, j) *= oldt / newt;
            for (auto k = j + 1; k != this->_n; ++k) {
                v[k] -= this->_mq(j, k);
                this->_mq(k, j) += beta2 * v[k];
            }
            oldt = newt;
        }
        // const auto gamma = oldt + gg_t[m];
        // this->_mq(m, m) *= oldt / gamma; // update invD
        //
        this->_kappa *= result.delta;
        g = grad_t * (result.rho / omega);
        return result.status;
    }

    /**
     * @brief Delegate bias cut to EllCalc (single beta)
     *
     * @param[in] beta  Bias term
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_bias_cut(double beta, double tsq) const -> CutResult {
        return this->_helper.calc_bias_cut(beta, tsq);
    }

    /**
     * @brief Delegate bias cut to EllCalc (valarray beta = parallel cut)
     *
     * Dispatches to calc_parallel_bias_cut when beta.size() ≥ 2.
     *
     * @param[in] beta  Vector containing [beta0, beta1] for parallel cut
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_bias_cut(const std::valarray<double>& beta, double tsq) const -> CutResult {
        if (beta.size() < 2) {
            return this->_helper.calc_bias_cut(beta[0], tsq);
        }
        return this->_helper.calc_parallel_bias_cut(beta[0], beta[1], tsq);
    }

    /**
     * @brief Delegate central cut to EllCalc (single beta, unused)
     *
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_central_cut(double /*unused*/, double tsq) const -> CutResult {
        return this->_helper.calc_central_cut(tsq);
    }

    /**
     * @brief Delegate central cut to EllCalc (valarray beta = parallel cut)
     *
     * Dispatches to calc_parallel_central_cut when beta.size() ≥ 2.
     *
     * @param[in] beta  Vector containing [beta0, beta1] for parallel cut
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_central_cut(const std::valarray<double>& beta, double tsq) const -> CutResult {
        if (beta.size() < 2) {
            return this->_helper.calc_central_cut(tsq);
        }
        return this->_helper.calc_parallel_central_cut(beta[1], tsq);
    }

    /**
     * @brief Delegate bias-cut-Q to EllCalc (single beta)
     *
     * @param[in] beta  Bias term
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_q(double beta, double tsq) const -> CutResult {
        return this->_helper.calc_bias_cut_q(beta, tsq);
    }

    /**
     * @brief Delegate bias-cut-Q to EllCalc (valarray beta = parallel cut)
     *
     * Dispatches to calc_parallel_bias_cut_q when beta.size() ≥ 2.
     *
     * @param[in] beta  Vector containing [beta0, beta1] for parallel cut
     * @param[in] tsq   Squared radius
     * @return CutResult
     */
    auto _update_cut_q(const std::valarray<double>& beta, double tsq) const -> CutResult {
        if (beta.size() < 2) {
            return this->_helper.calc_bias_cut_q(beta[0], tsq);
        }
        return this->_helper.calc_parallel_bias_cut_q(beta[0], beta[1], tsq);
    }

};  // } EllCore
