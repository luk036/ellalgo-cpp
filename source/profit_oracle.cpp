#include <ellalgo/oracles/profit_oracle.hpp>
#include <type_traits>  // for move

using Vec = std::valarray<double>;
using Cut = std::pair<Vec, double>;

/**
 * @brief
 *
 * @param[in] y
 * @param[in,out] target the best-so-far optimal value
 * @return std::tuple<Cut, double>
 */
auto ProfitOracle::assess_optim(const Vec &y, double &target) const -> std::tuple<Cut, bool> {
    // y0 <= log k
    const auto f1 = y[0] - this->_log_k;
    if (f1 > 0.0) {
        return {{Vec{1.0, 0.0}, f1}, false};
    }

    const auto log_Cobb
        = this->_log_pA + this->_elasticities[0] * y[0] + this->_elasticities[1] * y[1];
    const Vec x = std::exp(y);
    const auto vx = this->_price_out[0] * x[0] + this->_price_out[1] * x[1];
    auto te = target + vx;

    auto fj = std::log(te) - log_Cobb;
    if (fj < 0.0) {
        te = std::exp(log_Cobb);
        target = te - vx;
        Vec g = (this->_price_out * x) / te - this->_elasticities;
        return {{std::move(g), 0.0}, true};
    }
    Vec g = (this->_price_out * x) / te - this->_elasticities;
    return {{std::move(g), fj}, false};
}

/**
 * @param[in] y
 * @param[in,out] target the best-so-far optimal value
 * @return std::tuple<Cut, double, Vec, int>
 */
auto ProfitOracleQ::assess_optim_q(const Vec &y, double &target, bool retry)
    -> std::tuple<Cut, bool, Vec, bool> {
    if (!retry) {
        Vec x = std::exp(y);
        x = x.apply([](double n) -> double { return std::round(n); });
        if (x[0] == 0.0) {
            x[0] = 1.0;  // nearest integer than 0
        }
        if (x[1] == 0.0) {
            x[1] = 1.0;
        }
        this->_yd = std::log(x);
    }
    auto result1 = this->_P.assess_optim(this->_yd, target);
    auto &cut = std::get<0>(result1);
    auto &shrunk = std::get<1>(result1);
    auto &g = std::get<0>(cut);
    auto &h = std::get<1>(cut);
    auto d = this->_yd - y;
    h += g[0] * d[0] + g[1] * d[1];
    return {std::move(cut), shrunk, this->_yd, false};
}
