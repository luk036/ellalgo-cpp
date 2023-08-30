#pragma once

#include <cmath>
#include <tuple>

#include "ell_calc_core.hpp"

// forward declaration
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

    // const double _nPlus1;
    // const double _halfN;
    // const double _nSq;
    // const double _c1;
    // const double _c2;
    // const double _c3;

  public:
    /**
     * @brief Construct a new EllCalcobject
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    explicit EllCalc(size_t ndim) : _n_f{double(ndim)}, _helper{ndim} {}
    // _nPlus1{_n_f + 1.0},
    // _halfN{_n_f / 2.0},
    // _nSq{_n_f * _n_f},
    // _c1{_nSq / (_nSq - 1.0)},
    // _c2{2.0 / _nPlus1},
    // _c3{_n_f / _nPlus1} {}

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
     * @param E
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
    auto calc_parallel_deep_cut(const double &beta0, const double &beta1, const double &tsq) const
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
    auto calc_deep_cut(const double &beta, const double &tsq) const
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
    auto calc_parallel_deep_cut_q(const double &beta0, const double &beta1, const double &tsq) const
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
    auto calc_deep_cut_q(const double &beta, const double &tsq) const
        -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    // private:
    //   /**
    //    * @brief
    //    *
    //    * @param beta0
    //    * @param beta1
    //    * @param b1sq
    //    * @param b0b1
    //    * @param tsq
    //    * @return std::tuple<CutStatus, std::tuple<double, double, double>>
    //    */
    //   auto _calc_parallel_core(const double &beta0, const double &beta1, const double &b1sq,
    //                            const double &b0b1, const double &tsq) const
    //       -> std::tuple<CutStatus, std::tuple<double, double, double>>;

    //   /**
    //    * @brief
    //    *
    //    * @param beta
    //    * @param tau
    //    * @param gamma
    //    * @return std::tuple<CutStatus, std::tuple<double, double, double>>
    //    */
    //   auto _calc_deep_cut_core(const double &beta, const double &tau, const double &gamma) const
    //       -> std::tuple<CutStatus, std::tuple<double, double, double>>;
};  // } EllCalc
