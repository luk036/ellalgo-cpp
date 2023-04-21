#include <cmath>                  // for sqrt
#include <ellalgo/ell_assert.hpp> // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>   // for EllCalc
#include <ellalgo/ell_config.hpp> // for CutStatus, CutStatus::Success
#include <tuple>                  // for tuple

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return int
 */
auto EllCalc::_calc_ll_core(const double &b0, const double &b1) -> CutStatus {
  // const auto b1sq = b1 * b1;
  const auto b1sqn = b1 * (b1 / this->_tsq);
  if (b0 == 0.0) // central cut
  {
    this->_calc_ll_cc(b1, b1sqn);
    return CutStatus::Success;
  }

  const auto t1n = 1.0 - b1sqn;
  if (t1n < 0.0 || !this->use_parallel_cut) {
    return this->_calc_dc(b0);
  }

  const auto bdiff = b1 - b0;
  if (bdiff < 0.0) {
    return CutStatus::NoSoln; // no sol'n
  }

  const auto b0b1n = b0 * (b1 / this->_tsq);
  if (ELL_UNLIKELY(this->_nFloat * b0b1n < -1.0)) {
    return CutStatus::NoEffect; // no effect
  }

  // const auto t0 = this->_tsq - b0 * b0;
  const auto t0n = 1.0 - b0 * (b0 / this->_tsq);
  // const auto t1 = this->_tsq - b1sq;
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / this->_tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->_halfN * bsumn * bdiff;
  const auto xi = std::sqrt(t0n * t1n + tempn * tempn);
  this->_sigma = this->_c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / this->_nPlus1;
  this->_rho = this->_sigma * bav;
  this->_delta = this->_c1 * ((t0n + t1n) / 2.0 + xi / this->_nFloat);
  return CutStatus::Success;
}

/**
 * @brief
 *
 * @param[in] b1
 * @param[in] b1sq
 * @return void
 */
void EllCalc::_calc_ll_cc(const double &b1, const double &b1sqn) {
  const auto temp = this->_halfN * b1sqn;
  const auto xi = std::sqrt(1.0 - b1sqn + temp * temp);
  this->_sigma = this->_c3 + this->_c2 * (1.0 - xi) / b1sqn;
  this->_rho = this->_sigma * b1 / 2;
  this->_delta = this->_c1 * (1.0 - b1sqn / 2.0 + xi / this->_nFloat);
  // this->_mu ???
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return int
 */
auto EllCalc::_calc_dc(const double &beta) noexcept -> CutStatus {
  const auto tau = std::sqrt(this->_tsq);
  if (beta == 0.0) {
    this->_calc_cc(tau);
    return CutStatus::Success;
  }

  const auto bdiff = tau - beta;
  if (bdiff < 0.0) {
    return CutStatus::NoSoln; // no sol'n
  }

  const auto gamma = tau + this->_nFloat * beta;
  if (ELL_UNLIKELY(gamma < 0)) {
    return CutStatus::NoEffect; // no effect
  }

  // this->_mu = (bdiff / gamma) * this->_halfNminus1;
  this->_rho = gamma / this->_nPlus1;
  this->_sigma = 2.0 * this->_rho / (tau + beta);
  this->_delta = this->_c1 * (1.0 - beta * (beta / this->_tsq));
  return CutStatus::Success;
}

/**
 * @brief Central Cut
 *
 * @param[in] tau
 * @return int
 */
void EllCalc::_calc_cc(const double &tau) noexcept {
  // this->_mu = this->_halfNminus1;
  this->_sigma = this->_c2;
  this->_rho = tau / this->_nPlus1;
  this->_delta = this->_c1;
}
