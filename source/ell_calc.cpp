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
auto EllCalc::_calc_ll_core(const double &b0, const double &b1) -> CutStatus {
  if (b1 < b0) {
    return CutStatus::NoSoln; // no sol'n
  }

  const auto b1sq = b1 * b1;
  if (this->_tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_dc(b0);
  }

  const auto b1sqn = b1sq / this->_tsq;
  const auto t1n = 1.0 - b1sqn;
  const auto b0b1n = b0 * (b1 / this->_tsq);
  const auto t0n = 1.0 - b0 * (b0 / this->_tsq);
  // const auto t1 = this->_tsq - b1sq;
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / this->_tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->_halfN * bsumn * (b1 - b0);
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
auto EllCalc::_calc_ll_cc(const double &b1) -> CutStatus {
  if (b1 < 0.0) {
    return CutStatus::NoSoln; // no sol'n
  }
  const auto b1sq = b1 * b1;
  if (this->_tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_cc();
  }
  const auto b1sqn = b1sq / this->_tsq;
  const auto temp = this->_halfN * b1sqn;
  const auto xi = std::sqrt(1.0 - b1sqn + temp * std::move(temp));
  this->_delta = this->_c1 * (1.0 - b1sqn / 2.0 + xi / this->_nFloat);
  this->_sigma = this->_c3 + this->_c2 * (1.0 - std::move(xi)) / b1sqn;
  this->_rho = this->_sigma * b1 / 2;
  return CutStatus::Success;
  // this->_mu ???
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return int
 */
auto EllCalc::_calc_dc(const double &beta) -> CutStatus {
  assert(beta >= 0.0);
  auto bsq = beta * beta;
  if (this->_tsq < bsq) {
    return CutStatus::NoSoln; // no sol'n
  }
  auto tau = std::sqrt(this->_tsq);
  auto gamma = tau + this->_nFloat * beta;
  this->_rho = std::move(gamma) / this->_nPlus1;
  this->_sigma = 2.0 * this->_rho / (std::move(tau) + beta);
  this->_delta = this->_c1 * (1.0 - std::move(bsq) / this->_tsq);
  return CutStatus::Success;
}

/**
 * @brief Central Cut
 *
 * @return int
 */
auto EllCalc::_calc_cc() -> CutStatus {
  this->_sigma = this->_c2;
  this->_rho = std::sqrt(this->_tsq) / this->_nPlus1;
  this->_delta = this->_c1;
  return CutStatus::Success;
}

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return int
 */
auto EllCalcQ::_calc_ll_core(const double &b0, const double &b1) -> CutStatus {
  if (b1 < b0) {
    return CutStatus::NoSoln; // no sol'n
  }

  const auto b1sq = b1 * b1;
  if (this->_tsq < b1sq || !this->use_parallel_cut) {
    return this->_calc_dc(b0);
  }

  const auto b0b1 = b0 * b1;
  if (ELL_UNLIKELY(this->_nFloat * b0b1 < -this->_tsq)) {
    return CutStatus::NoEffect; // no effect
  }

  const auto b0b1n = b0b1 / this->_tsq;
  const auto b1sqn = b1sq / this->_tsq;
  const auto t1n = 1.0 - b1sqn;
  // const auto t0 = this->_tsq - b0 * b0;
  const auto t0n = 1.0 - b0 * (b0 / this->_tsq);
  // const auto t1 = this->_tsq - b1sq;
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / this->_tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->_halfN * bsumn * (b1 - b0);
  const auto xi = std::sqrt(t0n * t1n + tempn * tempn);
  this->_sigma = this->_c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / this->_nPlus1;
  this->_rho = this->_sigma * bav;
  this->_delta = this->_c1 * ((t0n + t1n) / 2.0 + xi / this->_nFloat);
  return CutStatus::Success;
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return int
 */
auto EllCalcQ::_calc_dc(const double &beta) -> CutStatus {
  const auto tau = std::sqrt(this->_tsq);
  if (tau < beta) {
    return CutStatus::NoSoln; // no sol'n
  }
  const auto gamma = tau + this->_nFloat * beta;
  if (ELL_UNLIKELY(gamma <= 0.0)) {
    return CutStatus::NoEffect; // no effect
  }
  this->_rho = std::move(gamma) / this->_nPlus1;
  this->_sigma = 2.0 * this->_rho / (std::move(tau) + beta);
  this->_delta = this->_c1 * (1.0 - beta * (beta / this->_tsq));
  return CutStatus::Success;
}
