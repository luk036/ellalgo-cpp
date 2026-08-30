/**
 * @file profit_oracle.cpp
 * @brief Implementation of profit maximization oracle
 *
 * This file implements oracles for profit maximization problems using
 * the ellipsoid algorithm. It includes both continuous and discrete
 * optimization variants.
 */

#include <cmath>  // for log, exp, round
#include <ellalgo/oracles/profit_oracle.hpp>

using Vec = std::valarray<double>;
using Cut = std::pair<Vec, double>;

/**
 * @brief Capacity constraint: y0 <= log k
 *
 * Returns a cut when the constraint is violated, nullptr otherwise.
 *
 * @param[in] y input quantity (in log scale)
 * @param[in] x input quantity (natural scale, unused)
 * @param[in] gamma the best-so-far optimal value (unused)
 * @return Cut* pointer to the cut, or nullptr if satisfied
 */
auto ProfitOracle::_constraint_capacity(const Vec& y, const Vec&, const double) -> Cut* {
    static auto cut = Cut{Vec{1.0, 0.0}, 0.0};
    const auto fj = y[0] - this->_log_k;
    if (fj <= 0.0) {
        return nullptr;
    }
    cut.second = fj;
    return &cut;
}

/**
 * @brief Cobb-Douglas profit constraint
 *
 * Computes the production value and cost; returns a cut (the profit
 * gradient) when the profit falls below the target gamma, nullptr otherwise.
 * Also caches `_log_Cobb` and `_vx` for use by assess_optim.
 *
 * @param[in] y input quantity (in log scale)
 * @param[in] x input quantity (natural scale)
 * @param[in] gamma the best-so-far optimal value
 * @return Cut* pointer to the cut, or nullptr if satisfied
 */
auto ProfitOracle::_constraint_profit(const Vec& y, const Vec& x, const double gamma) -> Cut* {
    static auto cut = Cut{Vec{-1.0, 1.0}, 0.0};
    this->_log_Cobb = this->_log_pA + this->_elasticities[0] * y[0] + this->_elasticities[1] * y[1];
    this->_vx = this->_price_out[0] * x[0] + this->_price_out[1] * x[1];
    const auto te = gamma + this->_vx;
    const auto fj = std::log(te) - this->_log_Cobb;
    if (fj <= 0.0) {
        return nullptr;
    }
    cut.first = (this->_price_out * x) / te - this->_elasticities;
    cut.second = fj;
    return &cut;
}

/**
 * The function assess_feas assesses the feasibility of a given solution based on certain conditions
 * and returns a tuple containing a cut and a boolean value.
 *
 * @param[in] y The parameter `y` is a vector of values. It is used to calculate various values in
 * the function. The specific meaning of each element in the vector depends on the context and the
 * specific implementation of the `ProfitOracle` class.
 * @param[in,out] gamma The `gamma` parameter is a reference to a `double` variable. It is used to
 * store the best-so-far value for the feasibility process. The function `assess_feas` assesses
 * the feasibility of a given solution and updates the `gamma` value if necessary.
 *
 * @return The function `assess_feas` returns a tuple containing two elements. The first element is
 * of type `Cut`, which is a struct or class that contains a vector `g` and a double `fj`. The
 * second element is of type `bool`.
 */
auto ProfitOracle::assess_feas(const Vec& y, const double& gamma) -> Cut* {
    using ConstraintFn = auto (ProfitOracle::*)(const Vec&, const Vec&, const double)->Cut*;
    static constexpr ConstraintFn constraints[2]
        = {&ProfitOracle::_constraint_capacity, &ProfitOracle::_constraint_profit};

    const Vec x = std::exp(y);
    for (int i = 0; i < 2; i++) {
        const auto k = this->_rr.next();
        auto* cut = (this->*constraints[k])(y, x, gamma);
        if (cut != nullptr) {
            return cut;
        }
    }

    return nullptr;
}

/**
 * The function assess_optim assesses the optimality of a given solution based on certain conditions
 * and returns a tuple containing a cut and a boolean value.
 *
 * @param[in] y The parameter `y` is a vector of values. It is used to calculate various values in
 * the function. The specific meaning of each element in the vector depends on the context and the
 * specific implementation of the `ProfitOracle` class.
 * @param[in,out] gamma The `gamma` parameter is a reference to a `double` variable. It is used to
 * store the best-so-far value for the optimization process. The function `assess_optim` assesses
 * the optimality of a given solution and updates the `gamma` value if necessary.
 *
 * @return The function `assess_optim` returns a tuple containing two elements. The first element is
 * of type `Cut`, which is a struct or class that contains a vector `g` and a double `fj`. The
 * second element is of type `bool`.
 */
auto ProfitOracle::assess_optim(const Vec& y, double& gamma) -> std::tuple<Cut, bool> {
    auto* cut = this->assess_feas(y, gamma);
    if (cut != nullptr) {
        return {*cut, false};
    }

    const Vec x = std::exp(y);
    auto te = std::exp(this->_log_Cobb);
    gamma = te - this->_vx;
    Vec grad = (this->_price_out * x) / te - this->_elasticities;
    return {{std::move(grad), 0.0}, true};
}

/**
 * @brief Assess optimality for discrete profit maximization
 *
 * Uses round-to-nearest-integer for discrete variables and adjusts
 * the beta term to account for the discretization gap.
 *
 * @param[in] y     Input vector (log-scaled quantities)
 * @param[in,out] gamma Best-so-far optimal value
 * @param[in] retry Whether to re-use cached discrete point
 * @return Tuple (cut, shrunk, discrete_y, more_alt)
 */
auto ProfitOracleQ::assess_optim_q(const Vec& y, double& gamma, bool retry)
    -> std::tuple<Cut, bool, Vec, bool> {
    if (!retry) {
        auto* cut = this->P.assess_feas(y, gamma);
        if (cut != nullptr) {
            return {*cut, false, y, true};
        }

        Vec x = std::exp(y);
        x = x.apply([](double n) { return std::round(n); });
        if (x[0] == 0.0) {
            x[0] = 1.0;  // nearest integer than 0
        }
        if (x[1] == 0.0) {
            x[1] = 1.0;
        }
        this->_yd = std::log(x);
    }
    auto result1 = this->P.assess_optim(this->_yd, gamma);
    auto& cut = std::get<0>(result1);
    auto& shrunk = std::get<1>(result1);
    auto& grad = std::get<0>(cut);
    auto& beta = std::get<1>(cut);
    auto diff = this->_yd - y;
    beta += grad[0] * diff[0] + grad[1] * diff[1];
    return {std::move(cut), shrunk, std::move(this->_yd), !retry};
}
