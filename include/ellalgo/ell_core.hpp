// -*- coding: utf-8 -*-
#pragma once

#include "ell_calc.hpp"
#include "ell_config.hpp"
#include "ell_matrix.hpp"
#include <cmath>
#include <tuple>
#include <valarray>

/**
 * @brief Ellipsoid Search Space Core
 *
 * Without the knowledge of the type of xc
 *
 *  \mathcal{E} {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 */
class EllCore {
  public:
    using Vec = std::valarray<double>;
    bool no_defer_trick = false;

  private:
    size_t _n;
    double _kappa;
    Matrix _mq;
    EllCalc _helper;
    double _tsq{};

    /**
     * @brief Construct a new EllCore object
     *
     * @param[in] E
     */
    auto operator=(const EllCore &E) -> EllCore & = delete;

    /**
     * @brief Construct a new EllCore object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    EllCore(double &&kappa, Matrix &&mq, size_t ndim)
        : _n{ndim}, _kappa{std::move(kappa)}, _mq{std::move(mq)}, _helper{_n} {}

  public:
    /**
     * @brief Construct a new EllCore object
     *
     * @param[in] val
     * @param[in] x
     */
    EllCore(const Vec &val, size_t ndim) : EllCore{1.0, Matrix(ndim), ndim} {
        this->_mq.diagonal() = val;
    }

    /**
     * @brief Construct a new EllCore object
     *
     * @param[in] alpha
     * @param[in] x
     */
    EllCore(double alpha, size_t ndim)
        : EllCore{std::move(alpha), Matrix(ndim), ndim} {
        this->_mq.identity();
    }

    /**
     * @brief Construct a new EllCore object
     *
     * @param[in] E (move)
     */
    EllCore(EllCore &&E) = default;

    /**
     * @brief Destroy the EllCore object
     *
     */
    ~EllCore() {}

    /**
     * @brief Construct a new EllCore object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    explicit EllCore(const EllCore &E) = default;

    /**
     * @brief explicitly copy
     *
     * @return EllCore
     */
    auto copy() const -> EllCore { return EllCore(*this); }

    /**
     * @brief copy the whole array anyway
     *
     * @return Vec
     */

    /**
     * @brief
     *
     * @return Arr
     */
    auto tsq() const -> double { return this->_tsq; }

    void set_use_parallel_cut(bool value) {
        this->_helper.use_parallel_cut = value;
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T>
    auto update_dc(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta,
                                  [&](const T &beta, const double &tsq) {
                                      return this->_update_cut_dc(beta, tsq);
                                  });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T>
    auto update_cc(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta,
                                  [&](const T &beta, const double &tsq) {
                                      return this->_update_cut_cc(beta, tsq);
                                  });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T> auto update_q(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_core(grad, beta,
                                  [&](const T &beta, const double &tsq) {
                                      return this->_update_cut_q(beta, tsq);
                                  });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return std::tuple<int, double>
     */
    template <typename T>
    auto update_stable_dc(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(
            grad, beta, [&](const T &beta, const double &tsq) {
                return this->_update_cut_dc(beta, tsq);
            });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param grad
     * @param beta
     * @return CutStatus
     */
    template <typename T>
    auto update_stable_cc(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(
            grad, beta, [&](const T &beta, const double &tsq) {
                return this->_update_cut_cc(beta, tsq);
            });
    }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param grad
     * @param beta
     * @return CutStatus
     */
    template <typename T>
    auto update_stable_q(Vec &grad, const T &beta) -> CutStatus {
        return this->_update_stable_core(
            grad, beta, [&](const T &beta, const double &tsq) {
                return this->_update_cut_q(beta, tsq);
            });
    }

  private:
    /**
     * @brief
     *
     * @tparam T
     * @tparam Fn
     * @param grad
     * @param beta
     * @param cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_core(Vec &grad, const T &beta, Fn &&cut_strategy)
        -> CutStatus {
        std::valarray<double> grad_t(0.0, this->_n);
        auto omega = 0.0;
        for (auto i = 0U; i != this->_n; ++i) {
            for (auto j = 0U; j != this->_n; ++j) {
                grad_t[i] += this->_mq(i, j) * grad[j];
            }
            omega += grad_t[i] * grad[i];
        }

        this->_tsq = this->_kappa * omega;

        auto __result = cut_strategy(beta, this->_tsq);
        auto status = std::get<0>(__result);
        if (status != CutStatus::Success) {
            return status;
        }

        auto rho = std::get<1>(__result);
        auto sigma = std::get<2>(__result);
        auto delta = std::get<3>(__result);
        // n*(n+1)/2 + n
        // this->_mq -= (this->_sigma / omega) * xt::linalg::outer(grad_t,
        // grad_t);
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
        return status; // g++-7 is ok
    }

    /**
     * @brief
     *
     * @tparam T
     * @tparam Fn
     * @param g
     * @param beta
     * @param cut_strategy
     * @return CutStatus
     */
    template <typename T, typename Fn>
    auto _update_stable_core(Vec &g, const T &beta, Fn &&cut_strategy)
        -> CutStatus {
        // calculate inv(L)*grad: (n-1)*n/2 multiplications
        auto invLg{g}; // initially
        for (auto j = 0U; j != this->_n - 1; ++j) {
            for (auto i = j + 1; i != this->_n; ++i) {
                this->_mq(j, i) = this->_mq(i, j) * invLg[j];
                // keep for rank-one update
                invLg[i] -= this->_mq(j, i);
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        auto invDinvLg{invLg}; // initially
        for (auto i = 0U; i != this->_n; ++i) {
            invDinvLg[i] *= this->_mq(i, i);
        }

        // calculate omega: n
        auto gg_t{invDinvLg}; // initially
        auto omega = 0.0;     // initially
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

        auto rho = std::get<1>(__result);
        auto sigma = std::get<2>(__result);
        auto delta = std::get<3>(__result);

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
        auto grad_t{invDinvLg};                    // initially
        for (auto i = this->_n - 1; i != 0; --i) { // backward subsituition
            for (auto j = i; j != this->_n; ++j) {
                grad_t[i - 1] -= this->_mq(j, i - 1) * grad_t[j]; // ???
            }
        }

        // rank-one update: 3*n + (n-1)*n/2
        // const auto r = this->_sigma / omega;
        const auto mu = sigma / (1.0 - sigma);
        auto oldt = omega / mu; // initially
        // const auto m = this->_n - 1;
        auto v{g};
        for (auto j = 0U; j != this->_n; ++j) {
            const auto p = v[j];
            const auto temp = invDinvLg[j];
            const auto newt = oldt + p * temp;
            const auto beta2 = temp / newt;
            this->_mq(j, j) *= oldt / newt; // update invD
            for (auto k = j + 1; k != this->_n; ++k) {
                v[k] -= this->_mq(j, k);
                this->_mq(k, j) += beta2 * v[k];
            }
            oldt = newt;
        }
        // const auto t = oldt + gg_t[m];
        // this->_mq(m, m) *= oldt / t; // update invD
        //
        this->_kappa *= delta;
        g = grad_t * (rho / omega);
        return status;
    }

    /**
     * @brief
     *
     * @param beta
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_dc(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> {
        return this->_helper.calc_dc(beta, tsq);
    }

    /**
     * @brief
     *
     * @param beta
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_dc(const std::valarray<double> &beta,
                        const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> { // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_dc(beta[0], tsq);
        }
        return this->_helper.calc_ll_dc(beta[0], beta[1], tsq);
    }

    /**
     * @brief
     *
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_cc(const double &, const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> {
        return this->_helper.calc_cc(tsq);
    }

    /**
     * @brief
     *
     * @param beta
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_cc(const std::valarray<double> &beta,
                        const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> { // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_cc(tsq);
        }
        return this->_helper.calc_ll_cc(beta[1], tsq);
    }

    /**
     * @brief
     *
     * @param beta
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_q(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> {
        return this->_helper.calc_dc_q(beta, tsq);
    }

    /**
     * @brief
     *
     * @param beta
     * @param tsq
     * @return std::tuple<CutStatus, double, double, double>
     */
    auto _update_cut_q(const std::valarray<double> &beta,
                       const double &tsq) const
        -> std::tuple<CutStatus, double, double, double> { // parallel cut
        if (beta.size() < 2) {
            return this->_helper.calc_dc_q(beta[0], tsq);
        }
        return this->_helper.calc_ll_dc_q(beta[0], beta[1], tsq);
    }

}; // } EllCore
