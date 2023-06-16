#pragma once

#include <cmath>
#include <tuple>

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
  const double _nFloat;
  const double _nPlus1;
  const double _halfN;
  const double _nSq;
  const double _c1;
  const double _c2;
  const double _c3;

  /**
   * @brief Construct a new EllCalcobject
   *
   * @param[in] E
   */
  // auto operator=(const EllCalc& E) const -> EllCalc& = delete;

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
  EllCalc(size_t ndim)
      : _nFloat{double(ndim)}, _nPlus1{_nFloat + 1.0}, _halfN{_nFloat / 2.0},
        _nSq{_nFloat * _nFloat}, _c1{_nSq / (_nSq - 1.0)}, _c2{2.0 / _nPlus1},
        _c3{_nFloat / _nPlus1} {}

public:
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
  ~EllCalc() {}

  /**
   * @brief Construct a new EllCalcobject
   *
   * To avoid accidentally copying, only explicit copy is allowed
   *
   * @param E
   */
  EllCalc(const EllCalc &E) = default;

  /**
   * @brief Calculate new ellipsoid under Parallel Cut
   *
   *        g' (x - xc) + beta0 \le 0
   *        g' (x - xc) + beta1 \ge 0
   *
   * @param[in] b0
   * @param[in] b1
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_ll_dc(const double &b0, const double &b1, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief Calculate new ellipsoid under Parallel Cut, one of them is central
   *
   *        g' (x - xc) \le 0
   *        g' (x - xc) + beta1 \ge 0
   *
   * @param[in] b1
   * @param[in] b1sq
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_ll_cc(const double &b1, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief Calculate new ellipsoid under Deep Cut
   *
   *        g' (x - xc) + beta \le 0
   *
   * @param[in] beta
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_dc(const double &beta, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief Calculate new ellipsoid under Central Cut
   *
   *        g' (x - xc) \le 0
   *
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_cc(const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief Calculate new ellipsoid under Parallel Cut
   *
   *        g' (x - xc) + beta0 \le 0
   *        g' (x - xc) + beta1 \ge 0
   *
   * @param[in] b0
   * @param[in] b1
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_ll_dc_q(const double &b0, const double &b1, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief Calculate new ellipsoid under Deep Cut
   *
   *        g' (x - xc) + beta \le 0
   *
   * @param[in] beta
   * @param[in] tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto calc_dc_q(const double &beta, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

private:
  /**
   * @brief
   *
   * @param b0
   * @param b1
   * @param b1sq
   * @param b0b1
   * @param tsq
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto _calc_ll_core(const double &b0, const double &b1, const double &b1sq,
                     const double &b0b1, const double &tsq) const
      -> std::tuple<CutStatus, double, double, double>;

  /**
   * @brief
   *
   * @param beta
   * @param tau
   * @param gamma
   * @return std::tuple<CutStatus, double, double, double>
   */
  auto _calc_dc_core(const double &beta, const double &tau,
                     const double &gamma) const
      -> std::tuple<CutStatus, double, double, double>;
}; // } EllCalc
