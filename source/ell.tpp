#include <cmath>                  // for sqrt
#include <ellalgo/ell.hpp>        // for Ell, Ell::Arr
#include <ellalgo/ell_assert.hpp> // for ELL_UNLIKELY
#include <ellalgo/ell_calc.hpp>   // for Ell, Ell::Arr
#include <ellalgo/ell_config.hpp> // for CutStatus, CutStatus::Success
#include <tuple>                  // for tuple

// #include "ellalgo/utility.hpp" // for zeros
// #include <xtensor-blas/xlinalg.hpp>

// using Arr = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        grad' * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in] cut
 * @return std::tuple<int, double>
 */
template <typename Arr>
template <typename T>
auto Ell<Arr>::update(const std::pair<Arr, T> &cut)
    -> std::tuple<CutStatus, double> {
  // const auto& [grad, beta] = cut;
  const auto &grad = cut.first;
  const auto &beta = cut.second;
  // n^2
  // const auto grad_t = Arr{xt::linalg::dot(this->_mq, grad)};  // n^2
  // const auto omega = xt::linalg::dot(grad, grad_t)();        // n

  std::valarray<double> g(this->_n);
  for (auto i = 0; i != this->_n; ++i) {
    g[i] = grad[i];
  }

  // auto grad_t = zeros({this->_n}); // initial x0
  std::valarray<double> grad_t(0.0, this->_n);
  auto omega = 0.0;
  for (auto i = 0; i != this->_n; ++i) {
    for (auto j = 0; j != this->_n; ++j) {
      grad_t[i] += this->_mq(i, j) * g[j];
    }
    omega += grad_t[i] * g[i];
  }

  this->_helper._tsq = this->_kappa * omega;
  auto status = this->_update_cut(beta);
  if (status != CutStatus::Success) {
    return {status, this->_helper._tsq};
  }

  // n*(n+1)/2 + n
  // this->_mq -= (this->_sigma / omega) * xt::linalg::outer(grad_t, grad_t);
  const auto r = this->_helper._sigma / omega;
  for (auto i = 0; i != this->_n; ++i) {
    const auto rQg = r * grad_t[i];
    for (auto j = 0; j != i; ++j) {
      this->_mq(i, j) -= rQg * grad_t[j];
      this->_mq(j, i) = this->_mq(i, j);
    }
    this->_mq(i, i) -= rQg * grad_t[i];
  }

  this->_kappa *= this->_helper._delta;

  if (this->no_defer_trick) {
    this->_mq *= this->_kappa;
    this->_kappa = 1.0;
  }

  g = grad_t * (this->_helper._rho / omega);

  for (auto i = 0; i != this->_n; ++i) {
    this->_xc[i] -= g[i];
  }
  return {status, this->_helper._tsq}; // g++-7 is ok
}

#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xcontainer.hpp>      // for xcontainer
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>      // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>       // for xsemantic_base
#include <xtensor/xtensor_forward.hpp> // for xarray
                                       //
// Instantiation
using XArr = xt::xarray<double, xt::layout_type::row_major>;

template std::tuple<CutStatus, double>
Ell<XArr>::update(const std::pair<XArr, double> &cut);

template std::tuple<CutStatus, double>
Ell<XArr>::update(const std::pair<XArr, XArr> &cut);

template std::tuple<CutStatus, double>
Ell<XArr>::update(const std::pair<XArr, std::valarray<double>> &cut);
