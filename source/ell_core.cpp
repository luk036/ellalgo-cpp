#include <cmath>                  // for sqrt
#include <ellalgo/ell_assert.hpp> // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>   // for EllCalc, EllCalc::Arr
#include <ellalgo/ell_config.hpp> // for CutStatus, CutStatus::Success
#include <ellalgo/ell_core.hpp>   // for EllCore, EllCore::Arr
#include <tuple>                  // for tuple

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        grad' * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in, out] grad in: gradient; out: xc
 * @return std::tuple<int, double>
 */
using Vec = std::valarray<double>;

template <typename T>
auto EllCore::update(Vec &grad, const T &beta)
    -> std::tuple<CutStatus, double> {
  // const auto& [grad, beta] = cut;
  // auto Qg = zeros({this->_n}); // initial x0
  std::valarray<double> Qg(0.0, this->_n);
  auto omega = 0.0;
  for (auto i = 0U; i != this->_n; ++i) {
    for (auto j = 0U; j != this->_n; ++j) {
      Qg[i] += this->_Q(i, j) * grad[j];
    }
    omega += Qg[i] * grad[i];
  }

  this->_helper._tsq = this->_kappa * omega;
  auto status = this->_update_cut(beta);
  if (status != CutStatus::Success) {
    return {status, this->_helper._tsq};
  }

  // n*(n+1)/2 + n
  // this->_Q -= (this->_sigma / omega) * xt::linalg::outer(Qg, Qg);
  const auto r = this->_helper._sigma / omega;
  for (auto i = 0U; i != this->_n; ++i) {
    const auto rQg = r * Qg[i];
    for (auto j = 0U; j != i; ++j) {
      this->_Q(i, j) -= rQg * Qg[j];
      this->_Q(j, i) = this->_Q(i, j);
    }
    this->_Q(i, i) -= rQg * Qg[i];
  }

  this->_kappa *= this->_helper._delta;

  if (this->no_defer_trick) {
    this->_Q *= this->_kappa;
    this->_kappa = 1.0;
  }

  grad = Qg * (this->_helper._rho / omega);
  return {status, this->_helper._tsq}; // g++-7 is ok
}

template <typename T>
auto EllCore::update_stable(Vec &g, const T &beta)
    -> std::tuple<CutStatus, double> {
  // calculate inv(L)*grad: (n-1)*n/2 multiplications
  auto invLg{g}; // initially
  for (auto i = 1U; i != this->_n; ++i) {
    for (auto j = 0U; j != i; ++j) {
      this->_Q(i, j) = this->_Q(j, i) * invLg[j];
      // keep for rank-one update
      invLg[i] -= this->_Q(i, j);
    }
  }

  // calculate inv(D)*inv(L)*grad: n
  auto invDinvLg{invLg}; // initially
  for (auto i = 0U; i != this->_n; ++i) {
    invDinvLg[i] *= this->_Q(i, i);
  }

  // calculate omega: n
  auto gQg{invDinvLg}; // initially
  auto omega = 0.0;    // initially
  for (auto i = 0U; i != this->_n; ++i) {
    gQg[i] *= invLg[i];
    omega += gQg[i];
  }

  this->_helper._tsq = this->_kappa * omega;
  auto status = this->_update_cut(beta);
  if (status != CutStatus::Success) {
    return {status, this->_helper._tsq};
  }

  // calculate Q*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
  auto Qg{invDinvLg};                        // initially
  for (auto i = this->_n - 1; i != 0; --i) { // backward subsituition
    for (auto j = i; j != this->_n; ++j) {
      Qg[i - 1] -= this->_Q(i, j) * Qg[j]; // ???
    }
  }

  // rank-one update: 3*n + (n-1)*n/2
  // const auto r = this->_sigma / omega;
  const auto mu = this->_helper._sigma / (1.0 - this->_helper._sigma);
  auto oldt = omega / mu; // initially
  const auto m = this->_n - 1;
  for (auto j = 0U; j != m; ++j) {
    const auto t = oldt + gQg[j];
    const auto beta2 = invDinvLg[j] / t;
    this->_Q(j, j) *= oldt / t; // update invD
    for (auto l = j + 1; l != this->_n; ++l) {
      this->_Q(j, l) += beta2 * this->_Q(l, j);
    }
    oldt = t;
  }

  const auto t = oldt + gQg[m];
  this->_Q(m, m) *= oldt / t; // update invD

  this->_kappa *= this->_helper._delta;

  // if (this->no_defer_trick)
  // {
  //     this->_Q *= this->_kappa;
  //     this->_kappa = 1.0;
  // }

  // calculate xc: n
  g = Qg * (this->_helper._rho / omega);
  return {status, this->_helper._tsq};
}

// Instantiation
template std::tuple<CutStatus, double> //
EllCore::update(Vec &grad, const double &beta);

template std::tuple<CutStatus, double> //
EllCore::update(Vec &grad, const Vec &beta);

template std::tuple<CutStatus, double> //
EllCore::update_stable(Vec &grad, const double &beta);

template std::tuple<CutStatus, double> //
EllCore::update_stable(Vec &grad, const Vec &beta);
