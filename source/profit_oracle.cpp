#include <ellalgo/oracles/profit_oracle.hpp>
#include <type_traits> // for move

// using Arr = std::xarray<double, std::layout_type::row_major>;
using Vec = std::valarray<double>;
using Cut = std::pair<Vec, double>;

/**
 * @brief
 *
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return std::tuple<Cut, double>
 */
auto ProfitOracle::assess_optim(const Vec &y, double &t) const
    -> std::tuple<Cut, bool> {
  // y0 <= log k
  const auto f1 = y[0] - this->_log_k;
  if (f1 > 0.0) {
    return {{Vec{1.0, 0.0}, f1}, false};
  }

  const auto log_Cobb = this->_log_pA + this->_a[0] * y[0] + this->_a[1] * y[1];
  const Vec x = std::exp(y);
  const auto vx = this->_v[0] * x[0] + this->_v[1] * x[1];
  auto te = t + vx;

  auto fj = std::log(te) - log_Cobb;
  if (fj < 0.0) {
    te = std::exp(log_Cobb);
    t = te - vx;
    Vec g = (this->_v * x) / te - this->_a;
    return {{std::move(g), 0.0}, true};
  }
  Vec g = (this->_v * x) / te - this->_a;
  return {{std::move(g), fj}, false};
}

/**
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return std::tuple<Cut, double, Vec, int>
 */
auto ProfitOracleQ::assess_q(const Vec &y, double &t, bool retry)
    -> std::tuple<Cut, bool, Vec, bool> {
  if (!retry) {
    Vec x = std::exp(y);
    x = x.apply([](double n) -> double { return std::round(n); });
    if (x[0] == 0.0) {
      x[0] = 1.0; // nearest integer than 0
    }
    if (x[1] == 0.0) {
      x[1] = 1.0;
    }
    this->_yd = std::log(x);
  }
  auto result1 = this->_P(this->_yd, t);
  auto &cut = std::get<0>(result1);
  auto &shrunk = std::get<1>(result1);
  auto &g = std::get<0>(cut);
  auto &h = std::get<1>(cut);
  // h += std::linalg::dot(g, this->_yd - y)();
  auto d = this->_yd - y;
  h += g[0] * d[0] + g[1] * d[1];
  return {std::move(cut), shrunk, this->_yd, !retry};
}
