#include <cassert>
#include <cmath>                  // for sqrt
#include <ellalgo/ell_assert.hpp> // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>   // for EllCalc, EllCalcQ
#include <ellalgo/ell_config.hpp> // for CutStatus, CutStatus::Success
#include <tuple>                  // for tuple

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return int
 */
auto EllCalc::_calc_ll_core(const double &b0, const double &b1,
                            const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  if (b1 < b0) {
    return {CutStatus::NoSoln, 0.0, 0.0, 0.0}; // no sol'n
  }

  const auto b1sq = b1 * b1;
  if (tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_dc(b0, tsq);
  }

  const auto b1sqn = b1sq / tsq;
  const auto t1n = 1.0 - b1sqn;
  const auto b0b1n = b0 * (b1 / tsq);
  const auto t0n = 1.0 - b0 * (b0 / tsq);
  // const auto t1 = tsq - b1sq;
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->_halfN * bsumn * (b1 - b0);
  const auto xi = std::sqrt(t0n * t1n + tempn * tempn);
  auto sigma = this->_c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / this->_nPlus1;
  auto rho = sigma * bav;
  auto delta = this->_c1 * ((t0n + t1n) / 2.0 + xi / this->_nFloat);
  return {CutStatus::Success, rho, sigma, delta};
}

/**
 * @brief
 *
 * @param[in] b1
 * @param[in] b1sq
 * @return void
 */
auto EllCalc::_calc_ll_cc(const double &b1, const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  if (b1 < 0.0) {
    return {CutStatus::NoSoln, 0.0, 0.0, 0.0}; // no sol'n
  }
  const auto b1sq = b1 * b1;
  if (tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_cc(tsq);
  }
  const auto b1sqn = b1sq / tsq;
  const auto temp = this->_halfN * b1sqn;
  const auto xi = std::sqrt(1.0 - b1sqn + temp * std::move(temp));
  auto delta = this->_c1 * (1.0 - b1sqn / 2.0 + xi / this->_nFloat);
  auto sigma = this->_c3 + this->_c2 * (1.0 - std::move(xi)) / b1sqn;
  auto rho = sigma * b1 / 2;
  return {CutStatus::Success, rho, sigma, delta};
  // this->_mu ???
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return int
 */
auto EllCalc::_calc_dc(const double &beta, const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  assert(beta >= 0.0);
  auto bsq = beta * beta;
  if (tsq < bsq) {
    return {CutStatus::NoSoln, 0.0, 0.0, 0.0}; // no sol'n
  }
  auto tau = std::sqrt(tsq);
  auto gamma = tau + this->_nFloat * beta;
  auto rho = std::move(gamma) / this->_nPlus1;
  auto sigma = 2.0 * rho / (std::move(tau) + beta);
  auto delta = this->_c1 * (1.0 - std::move(bsq) / tsq);
  return {CutStatus::Success, rho, sigma, delta};
}

/**
 * @brief Central Cut
 *
 * @return int
 */
auto EllCalc::_calc_cc(const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  auto sigma = this->_c2;
  auto rho = std::sqrt(tsq) / this->_nPlus1;
  auto delta = this->_c1;
  return {CutStatus::Success, rho, sigma, delta};
}

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return int
 */
auto EllCalcQ::_calc_ll_core(const double &b0, const double &b1,
                             const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  if (b1 < b0) {
    return {CutStatus::NoSoln, 0.0, 0.0, 0.0}; // no sol'n
  }

  const auto b1sq = b1 * b1;
  if (tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_dc(b0, tsq);
  }

  const auto b0b1 = b0 * b1;
  if (ELL_UNLIKELY(this->_nFloat * b0b1 < -tsq)) {
    return {CutStatus::NoEffect, 0.0, 0.0, 0.0}; // no effect
  }

  const auto b0b1n = b0b1 / tsq;
  const auto b1sqn = b1sq / tsq;
  const auto t1n = 1.0 - b1sqn;
  // const auto t0 = tsq - b0 * b0;
  const auto t0n = 1.0 - b0 * (b0 / tsq);
  // const auto t1 = tsq - b1sq;
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->_halfN * bsumn * (b1 - b0);
  const auto xi = std::sqrt(t0n * t1n + tempn * tempn);
  auto sigma = this->_c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / this->_nPlus1;
  auto rho = sigma * bav;
  auto delta = this->_c1 * ((t0n + t1n) / 2.0 + xi / this->_nFloat);
  return {CutStatus::Success, rho, sigma, delta};
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return int
 */
auto EllCalcQ::_calc_dc(const double &beta, const double &tsq) const
    -> std::tuple<CutStatus, double, double, double> {
  const auto tau = std::sqrt(tsq);
  if (tau < beta) {
    return {CutStatus::NoSoln, 0.0, 0.0, 0.0}; // no sol'n
  }
  const auto gamma = tau + this->_nFloat * beta;
  if (ELL_UNLIKELY(gamma <= 0.0)) {
    return {CutStatus::NoEffect, 0.0, 0.0, 0.0}; // no effect
  }
  auto rho = std::move(gamma) / this->_nPlus1;
  auto sigma = 2.0 * rho / (std::move(tau) + beta);
  auto delta = this->_c1 * (1.0 - beta * (beta / tsq));
  return {CutStatus::Success, rho, sigma, delta};
}
