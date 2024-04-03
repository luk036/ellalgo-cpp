#pragma once

#include <cassert>
#include <tuple>

#include "ell_calc_core.hpp"

// Forward declaration
enum class CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalc= {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $Q$ symmetric but no promise of positive definite
 */
class EllCalc {
  public:
    bool use_parallel_cut = true;

  protected:
    const double _n_f;
    const EllCalcCore _helper;

  public:
    /**
     * @brief Construct a new EllCalcobject
     *
     * @tparam V
     * @tparam U
     * @param[in] kappa
     * @param[in] mq
     * @param[in] x
     */
    explicit EllCalc(size_t ndim) : _n_f{double(ndim)}, _helper{ndim} {
        assert(ndim >= 2U);  // do not accept one-dimensional
    }

    /**
     * @brief Construct a new EllCalcobject
     *
     * @param[in] E (move)
     */
    EllCalc(EllCalc &&E) = default;

    /**
     * @brief Destroy the EllCalcobject
     *
     */
    ~EllCalc() = default;

    /**
     * @brief Construct a new EllCalcobject
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param[in] E
     */
    EllCalc(const EllCalc &E) = default;

    /**
     * @brief Calculate a new ellipsoid under a parallel cut
     *
     *        g' (x - xc) + beta0 \le 0
     *        g' (x - xc) + beta1 \ge 0
     *
     * @param[in] beta0
     * @param[in] beta1
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_parallel_bias_cut(const double &beta0, const double &beta1, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    /**
     * @brief Calculate new ellipsoid under Parallel Cut, one of them is central
     *
     *        g' (x - xc) \le 0
     *        g' (x - xc) + beta1 \ge 0
     *
     * @param[in] beta1
     * @param[in] b1sq
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_parallel_central_cut(const double &beta1, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    /**
     * @brief Calculate new ellipsoid under Deep Cut
     *
     *        g' (x - xc) + beta \le 0
     *
     * @param[in] beta
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_bias_cut(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    /**
     * @brief Calculate new ellipsoid under Central Cut
     *
     *        g' (x - xc) \le 0
     *
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_central_cut(const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    /**
     * @brief Calculate new ellipsoid under Parallel Cut
     *
     *        g' (x - xc) + beta0 \le 0
     *        g' (x - xc) + beta1 \ge 0
     *
     * @param[in] beta0
     * @param[in] beta1
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_parallel_bias_cut_q(const double &beta0, const double &beta1, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    /**
     * @brief Calculate new ellipsoid under Deep Cut
     *
     *        g' (x - xc) + beta \le 0
     *
     * @param[in] beta
     * @param[in] tsq
     * @return std::tuple<CutStatus, std::tuple<double, double, double>>
     */
    auto calc_bias_cut_q(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;
};  // } EllCalc
