#include <cmath>                       // for sqrt
#include <ellalgo/ell_assert.hpp>      // for ELL_UNLIKELY
#include <ellalgo/ell_config.hpp>      // for CutStatus, CutStatus::Success
#include <ellalgo/ell_stable.hpp>      // for EllStable, EllStable::Arr
#include <tuple>                       // for tuple
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xcontainer.hpp>      // for xcontainer
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>      // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>       // for xsemantic_base
#include <xtensor/xtensor_forward.hpp> // for xarray

#include "ellalgo/utility.hpp" // for zeros
// #include <xtensor-blas/xlinalg.hpp>

using Arr = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        grad' * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in] cut
 * @return std::tuple<int, double>
 */
template <typename T>
auto EllStable::update(const std::pair<Arr, T> &cut)
    -> std::tuple<CutStatus, double> {
  // const auto& [grad, beta] = cut;
  const auto &grad = cut.first;
  const auto &beta = cut.second;
  // calculate inv(L)*grad: (n-1)*n/2 multiplications
  Arr invLg{grad}; // initially
  for (auto i = 1; i != this->_n; ++i) {
    for (auto j = 0; j != i; ++j) {
      this->_Q(i, j) = this->_Q(j, i) * invLg(j);
      // keep for rank-one update
      invLg(i) -= this->_Q(i, j);
    }
  }

  // calculate inv(D)*inv(L)*grad: n
  Arr invDinvLg{invLg}; // initially
  for (auto i = 0; i != this->_n; ++i) {
    invDinvLg(i) *= this->_Q(i, i);
  }

  // calculate omega: n
  Arr gQg{invDinvLg}; // initially
  auto omega = 0.0;   // initially
  for (auto i = 0; i != this->_n; ++i) {
    gQg(i) *= invLg(i);
    omega += gQg(i);
  }

  this->_helper._tsq = this->_kappa * omega;
  auto status = this->_update_cut(beta);
  if (status != CutStatus::Success) {
    return {status, this->_helper._tsq};
  }

  // calculate Q*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
  Arr Qg{invDinvLg};                         // initially
  for (auto i = this->_n - 1; i != 0; --i) { // backward subsituition
    for (auto j = i; j != this->_n; ++j) {
      Qg(i - 1) -= this->_Q(i, j) * Qg(j); // ???
    }
  }

  // rank-one update: 3*n + (n-1)*n/2
  // const auto r = this->_sigma / omega;
  const auto mu = this->_helper._sigma / (1.0 - this->_helper._sigma);
  auto oldt = omega / mu; // initially
  const auto m = this->_n - 1;
  for (auto j = 0; j != m; ++j) {
    const auto t = oldt + gQg(j);
    const auto beta2 = invDinvLg(j) / t;
    this->_Q(j, j) *= oldt / t; // update invD
    for (auto l = j + 1; l != this->_n; ++l) {
      this->_Q(j, l) += beta2 * this->_Q(l, j);
    }
    oldt = t;
  }

  const auto t = oldt + gQg(m);
  this->_Q(m, m) *= oldt / t; // update invD

  this->_kappa *= this->_helper._delta;

  // if (this->no_defer_trick)
  // {
  //     this->_Q *= this->_kappa;
  //     this->_kappa = 1.0;
  // }

  // calculate xc: n
  this->_xc -= (this->_helper._rho / omega) * Qg;
  return {status, this->_helper._tsq}; // g++-7 is ok
}

// Instantiation
template std::tuple<CutStatus, double>
EllStable::update(const std::pair<Arr, double> &cut);

template std::tuple<CutStatus, double>
EllStable::update(const std::pair<Arr, Arr> &cut);

template std::tuple<CutStatus, double>
EllStable::update(const std::pair<Arr, std::valarray<double>> &cut);
