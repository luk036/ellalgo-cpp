#include <cmath>                       // for sqrt
#include <ellalgo/ell.hpp>             // for Ell, Ell::Arr
#include <ellalgo/ell_assert.hpp>      // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>        // for Ell, Ell::Arr
#include <ellalgo/ell_config.hpp>      // for CutStatus, CutStatus::Success
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
auto Ell::update(const std::pair<Arr, T> &cut)
    -> std::tuple<CutStatus, double> {
  // const auto& [grad, beta] = cut;
  const auto &grad = cut.first;
  const auto &beta = cut.second;
  // n^2
  // const auto Qg = Arr{xt::linalg::dot(this->_Q, grad)};  // n^2
  // const auto omega = xt::linalg::dot(grad, Qg)();        // n

  auto Qg = zeros({this->_n}); // initial x0
  auto omega = 0.0;
  for (auto i = 0; i != this->_n; ++i) {
    for (auto j = 0; j != this->_n; ++j) {
      Qg(i) += this->_Q(i, j) * grad(j);
    }
    omega += Qg(i) * grad(i);
  }

  this->_helper._tsq = this->_kappa * omega;
  auto status = this->_update_cut(beta);
  if (status != CutStatus::Success) {
    return {status, this->_helper._tsq};
  }

  // n*(n+1)/2 + n
  // this->_Q -= (this->_sigma / omega) * xt::linalg::outer(Qg, Qg);
  const auto r = this->_helper._sigma / omega;
  for (auto i = 0; i != this->_n; ++i) {
    const auto rQg = r * Qg(i);
    for (auto j = 0; j != i; ++j) {
      this->_Q(i, j) -= rQg * Qg(j);
      this->_Q(j, i) = this->_Q(i, j);
    }
    this->_Q(i, i) -= rQg * Qg(i);
  }

  this->_kappa *= this->_helper._delta;

  if (this->no_defer_trick) {
    this->_Q *= this->_kappa;
    this->_kappa = 1.0;
  }

  this->_xc -= (this->_helper._rho / omega) * Qg; // n
  return {status, this->_helper._tsq};            // g++-7 is ok
}

// Instantiation
template std::tuple<CutStatus, double>
Ell::update(const std::pair<Arr, double> &cut);
template std::tuple<CutStatus, double>
Ell::update(const std::pair<Arr, Arr> &cut);
