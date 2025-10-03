// -*- coding: utf-8 -*-
#pragma once

#include <tuple>
#include <valarray>

#include "ell_calc.hpp"
#include "ell_config.hpp"
#include "ell_matrix.hpp"

/**
 * @brief Ellipsoid Search Space Core
 *
 * Without the knowledge of the type of xc
 *
 *  \mathcal{E} {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 */
class EllCore {
    using Vec = std::valarray<double>;

    size_t _n;
    double _kappa;
    Matrix _mq;
    EllCalc _helper;
    double _tsq{};

  public:
    bool no_defer_trick = false;

  private:
    /**
     * @brief Construct a new EllCore object
     *
     * The `operator=` function is being deleted in this code. This means that the assignment
     * operator is not allowed for objects of the `EllCore` class. By deleting this function, the
     * code prevents objects of the `EllCore` class from being assigned to each other.
     *
     * @param[in] E The parameter "E" is a reference to an object of type "EllCore".
     */
    auto operator=(const EllCore &E) -> EllCore & = delete;

    /**
     * @brief Construct a new EllCore object
     *
     * The function is a constructor for the EllCore class that takes in a kappa value, a Matrix
     * object, and a size_t value as parameters.
     *
     * @param[in] kappa The kappa parameter is a constant value of type double. It is used in the
     * construction of the EllCore object.
     * @param[in] mq The parameter `mq` is a matrix of type `Matrix` that is being moved into the
     * `_mq` member variable of the `EllCore` object. The type `Matrix` is not specified in the code
     * snippet, so it would need to be defined elsewhere in the code.
     * @param[in] ndim The parameter `ndim` represents the number of dimensions for the EllCore
     * object.
     */
    EllCore(const double &kappa, Matrix &&mq, size_t ndim)
        : _n{ndim}, _kappa{kappa}, _mq{std::move(mq)}, _helper{_n} {}

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
    EllCore(const Vec &val, size_t ndim) : EllCore{1.0, Matrix(ndim), ndim} {
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
    EllCore(double alpha, size_t ndim) : EllCore{alpha, Matrix(ndim), ndim} {
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
    EllCore(EllCore &&E) = default;

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
    explicit EllCore(const EllCore &E) = default;

    /**
     * @brief explicitly copy
     *
     * @return EllCore
     */
    auto copy() const -> EllCore { return EllCore(*this); }

    /**
     * @brief
     *
     * @return double
     */
    auto tsq() const -> double { return this->_tsq; }

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
    template <typename T> auto update_bias_cut(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
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
    template <typename T> auto update_central_cut(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
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
    template <typename T> auto update_q(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
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
    template <typename T> auto update_stable_bias_cut(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
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
    template <typename T> auto update_stable_central_cut(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
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
    template <typename T> auto update_stable_q(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(grad, beta, [this](const T &beta_l, const double &tsq_l) {
            return this->_update_cut_q(beta_l, tsq_l);
        });
    }

  private:
    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `_update_core` function is a private member function of the `EllCore` class. It is used
     * to update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @tparam Fn
     * @param[in] grad gradient
     * @param[in] beta
     * @param[in] cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_core(Vec &grad, const T &beta, Fn &&cut_strategy) -> CutStatus {
        std::valarray<double> grad_t(0.0, this->_n);
        for (auto i = 0U; i != this->_n; ++i) {
            for (auto j = 0U; j != this->_n; ++j) {
                grad_t[i] += this->_mq(i, j) * grad[j];
            }
        }

        const auto omega = (grad_t * grad).sum();
        this->_tsq = this->_kappa * omega;

        if (omega == 0.0) {
            grad = grad_t;
            return CutStatus::Success;
        }

        auto __result = cut_strategy(beta, this->_tsq);
        auto status = std::get<0>(__result);
        if (status != CutStatus::Success) {
            return status;
        }

        double rho;
        double sigma;
        double delta;
        std::tie(rho, sigma, delta) = std::get<1>(__result);

        // n (n+1) / 2 + n
        const auto r = sigma / omega;
        for (auto i = 0U; i != this->_n; ++i) {
            const auto rQg = r * grad_t[i];
            for (auto j = 0U; j != i; ++j) {
                this->_mq(i, j) -= rQg * grad_t[j];
                this->_mq(j, i) = this->_mq(i, j);
            }
            this->_mq(i, i) -= rQg * grad_t[i];
        }

        this->_kappa *= delta;

        if (this->no_defer_trick) {
            this->_mq *= this->_kappa;
            this->_kappa = 1.0;
        }

        grad = grad_t * (rho / omega);
        return status;  // g++-7 is ok
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * The `_update_stable_core` function is a private member function of the `EllCore` class. It is
     * used to update the ellipsoid core function using a cutting plane.
     *
     * @tparam T
     * @tparam Fn
     * @param[in] grad gradient
     * @param[in] beta
     * @param[in] cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_stable_core(Vec &g, const T &beta, Fn &&cut_strategy) -> CutStatus {
        // Calculate L^-1 * grad: (n-1)*n/2 multiplications
        auto invLg{g};  // initially
        for (auto j = 0U; j != this->_n - 1; ++j) {
            for (auto i = j + 1; i != this->_n; ++i) {
                this->_mq(j, i) = this->_mq(i, j) * invLg[j];
                // keep for rank-one update
                invLg[i] -= this->_mq(j, i);
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        auto invDinvLg{invLg};  // initially
        for (auto i = 0U; i != this->_n; ++i) {
            invDinvLg[i] *= this->_mq(i, i);
        }

        // calculate omega: n
        auto gg_t{invDinvLg};  // initially
        auto omega = 0.0;      // initially
        for (auto i = 0U; i != this->_n; ++i) {
            gg_t[i] *= invLg[i];
            omega += gg_t[i];
        }

        this->_tsq = this->_kappa * omega;

        auto __result = cut_strategy(beta, this->_tsq);
        auto status = std::get<0>(__result);
        if (status != CutStatus::Success) {
            return status;
        }

        double rho;
        double sigma;
        double delta;
        std::tie(rho, sigma, delta) = std::get<1>(__result);

        // Calculate the (L')^-1 * D^-1 * L^-1 * grad : (n-1)n / 2
        auto grad_t{invDinvLg};                     // initially
        for (auto i = this->_n - 1; i != 0; --i) {  // backward subsituition
            for (auto j = i; j != this->_n; ++j) {
                grad_t[i - 1] -= this->_mq(j, i - 1) * grad_t[j];  // ???
            }
        }

        // rank-one update: 3*n + (n-1)*n/2
        // const auto r = this->_sigma / omega;
        const auto mu = sigma / (1.0 - sigma);
        auto oldt = omega / mu;  // initially
        // const auto m = this->_n - 1;
        auto v{g};
        for (auto j = 0U; j != this->_n; ++j) {
            const auto p = v[j];
            const auto temp = invDinvLg[j];
            const auto newt = oldt + p * temp;
            const auto beta2 = temp / newt;
            this->_mq(j, j) *= oldt / newt;  // update invD
            for (auto k = j + 1; k != this->_n; ++k) {
                v[k] -= this->_mq(j, k);
                this->_mq(k, j) += beta2 * v[k];
            }
            oldt = newt;
        }
        // const auto gamma = oldt + gg_t[m];
        // this->_mq(m, m) *= oldt / gamma; // update invD
        //
        this->_kappa *= delta;
        g = grad_t * (rho / omega);
        return status;
    }

    /**
     * The function `_update_cut_bias_cut` calculates a deep cut using the `calc_bias_cut` function
     * from the `_helper` object.
     *
     * @param[in] beta The beta parameter is a constant value of type double.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return The function `_update_cut_bias_cut` returns a tuple containing a `CutStatus` enum
     * value and another tuple containing three `double` values.
     */
    auto _update_cut_bias_cut(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {
        return this->_helper.calc_bias_cut(beta, tsq);
    }

    /**
     * The function `_update_cut_bias_cut` calculates the deep cut value based on the beta values
     * and tsq parameter.
     *
     * @param[in] beta A valarray of double values representing the beta values.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return a tuple containing a `CutStatus` enum value and another tuple containing three
     * `double` values.
     */
    auto _update_cut_bias_cut(const std::valarray<double> &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {  // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_bias_cut(beta[0], tsq);
        }
        return this->_helper.calc_parallel_bias_cut(beta[0], beta[1], tsq);
    }

    /**
     * The function `_update_cut_central_cut` calculates a central cut using the `calc_central_cut`
     * function from the `_helper` object.
     *
     * @param[in] beta The beta parameter is a constant value of type double.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return The function `_update_cut_bias_cut` returns a tuple containing a `CutStatus` enum
     * value and another tuple containing three `double` values.
     */
    auto _update_cut_central_cut(const double &, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {
        return this->_helper.calc_central_cut(tsq);
    }

    /**
     * The function `_update_cut_central_cut` calculates the central cut value based on the beta
     * values and tsq parameter.
     *
     * @param[in] beta A valarray of double values representing the beta values.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return a tuple containing a `CutStatus` enum value and another tuple containing three
     * `double` values.
     */
    auto _update_cut_central_cut(const std::valarray<double> &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {  // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_central_cut(tsq);
        }
        return this->_helper.calc_parallel_central_cut(beta[1], tsq);
    }

    /**
     * The function `_update_cut_q` calculates a deep cut q using the `calc_bias_cut_q` function
     * from the `_helper` object.
     *
     * @param[in] beta The beta parameter is a constant value of type double.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return The function `_update_cut_bias_cut` returns a tuple containing a `CutStatus` enum
     * value and another tuple containing three `double` values.
     */
    auto _update_cut_q(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {
        return this->_helper.calc_bias_cut_q(beta, tsq);
    }

    /**
     * The function `_update_cut_q` calculates the deep cut q value based on the beta values and tsq
     * parameter.
     *
     * @param[in] beta A valarray of double values representing the beta values.
     * @param[in] tsq tsq is a constant value of type double.
     *
     * @return a tuple containing a `CutStatus` enum value and another tuple containing three
     * `double` values.
     */
    auto _update_cut_q(const std::valarray<double> &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>> {  // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_bias_cut_q(beta[0], tsq);
        }
        return this->_helper.calc_parallel_bias_cut_q(beta[0], beta[1], tsq);
    }

};  // } EllCore
