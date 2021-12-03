#include <ellalgo/oracles/profit_oracle.hpp>
// #include <xtensor-blas/xlinalg.hpp>

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::tuple<Arr, double>;

/**
 * @brief
 *
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return std::tuple<Cut, double>
 */
auto profit_oracle::operator()(const Arr& y, double& t) const -> std::tuple<Cut, bool> {
    // y0 <= log k
    const auto f1 = y[0] - this->_log_k;
    if (f1 > 0.) {
        return {{Arr{1., 0.}, f1}, false};
    }

    const auto log_Cobb = this->_log_pA + this->_a(0) * y(0) + this->_a(1) * y(1);
    const Arr x = xt::exp(y);
    const auto vx = this->_v(0) * x(0) + this->_v(1) * x(1);
    auto te = t + vx;

    auto fj = std::log(te) - log_Cobb;
    if (fj < 0.) {
        te = std::exp(log_Cobb);
        t = te - vx;
        Arr g = (this->_v * x) / te - this->_a;
        return {{std::move(g), 0.}, true};
    }
    Arr g = (this->_v * x) / te - this->_a;
    return {{std::move(g), fj}, false};
}

/**
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return std::tuple<Cut, double, Arr, int>
 */
auto profit_q_oracle::operator()(const Arr& y, double& t, bool retry)
    -> std::tuple<Cut, bool, Arr, bool> {
    if (!retry) {
        Arr x = xt::round(xt::exp(y));
        if (x[0] == 0.) {
            x[0] = 1.;  // nearest integer than 0
        }
        if (x[1] == 0.) {
            x[1] = 1.;
        }
        this->_yd = xt::log(x);
    }
    auto result1 = this->_P(this->_yd, t);
    auto& cut = std::get<0>(result1);
    auto& shrunk = std::get<1>(result1);
    auto& g = std::get<0>(cut);
    auto& h = std::get<1>(cut);
    // h += xt::linalg::dot(g, this->_yd - y)();
    auto d = this->_yd - y;
    h += g(0) * d(0) + g(1) * d(1);
    return {std::move(cut), shrunk, this->_yd, !retry};
}
